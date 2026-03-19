from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fire
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidMLP
from spd.settings import SPD_OUT_DIR
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import get_scheduled_value, set_seed

from koko_experiments.feature_recovery.configs import FeatureRecoveryTrainConfig
from koko_experiments.feature_recovery.feature_datasets import DictionaryFeatureDataset
from koko_experiments.feature_recovery.metrics import compute_dictionary_recovery_metrics
from koko_experiments.feature_recovery.results import save_feature_recovery_target_bundle
from koko_experiments.feature_recovery.synthetic import ActivationGenerator, FeatureDictionary


def _default_out_dir(run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SPD_OUT_DIR / "feature_recovery" / "targets" / f"{run_name}_{stamp}"


def _init_embedding(model: ResidMLP, mode: str, feature_vectors: Tensor, freeze: bool) -> None:
    if mode == "learned":
        return
    if mode == "identity":
        assert model.config.n_features == model.config.d_embed
        model.W_E.data.copy_(torch.eye(model.config.d_embed))
        model.W_U.data.copy_(torch.eye(model.config.d_embed))
    elif mode == "random":
        model.W_E.data.copy_(torch.randn_like(model.W_E))
        model.W_E.data /= model.W_E.data.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        model.W_U.data.copy_(torch.linalg.pinv(model.W_E.data))
    elif mode == "feature_dictionary":
        assert feature_vectors.shape == model.W_E.shape
        model.W_E.data.copy_(feature_vectors)
        model.W_U.data.copy_(torch.linalg.pinv(model.W_E.data))
    else:
        raise ValueError(f"Unknown embedding mode: {mode}")

    if freeze:
        model.W_E.requires_grad = False
        model.W_U.requires_grad = False


def _loss_fn(
    model: ResidMLP,
    out: Float[Tensor, "batch n_features"] | Float[Tensor, "batch d_embed"],
    labels: Float[Tensor, "batch n_features"],
    loss_type: str,
) -> Tensor:
    if loss_type == "readoff":
        return torch.mean((out - labels) ** 2)
    resid_out: Float[Tensor, "batch d_embed"] = out
    resid_labels = labels @ model.W_E
    return torch.mean((resid_out - resid_labels) ** 2)


def main(config_path: str) -> None:
    config = FeatureRecoveryTrainConfig.from_file(config_path)
    set_seed(config.seed)
    device = get_device()

    feature_dict = FeatureDictionary.from_config(config.dataset.dictionary)
    activation_generator = ActivationGenerator(config.dataset.generator, feature_dict.num_features)
    label_coeffs = (
        torch.tensor(config.label_coeffs, dtype=torch.float32)
        if config.label_coeffs is not None
        else torch.ones(feature_dict.num_features, dtype=torch.float32)
    )
    dataset = DictionaryFeatureDataset(
        feature_dict=feature_dict,
        activation_generator=activation_generator,
        label_type=config.label_type,
        label_coeffs=label_coeffs,
        act_fn_name=config.resid_mlp_model_config.act_fn_name,
        device=device,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    model = ResidMLP(config.resid_mlp_model_config).to(device)
    _init_embedding(model, config.fixed_embedding, feature_dict.feature_vectors, config.freeze_embedding)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.lr_schedule.start_val,
        weight_decay=0.01,
    )

    for step, (latents, labels) in zip(range(config.steps), dataloader, strict=False):
        current_lr = get_scheduled_value(step, config.steps, config.lr_schedule)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        out = model(latents, return_residual=config.loss_type == "resid")
        loss = _loss_fn(model, out, labels, config.loss_type)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config.print_freq == 0 or step + 1 == config.steps:
            print(f"step={step:05d} loss={loss.item():.6f} lr={current_lr:.6f}")

    eval_losses: list[float] = []
    for _ in range(config.n_eval_batches):
        latents, labels = dataset.generate_batch(config.batch_size)
        with torch.no_grad():
            out = model(latents, return_residual=config.loss_type == "resid")
            eval_losses.append(float(_loss_fn(model, out, labels, config.loss_type).item()))

    summary = {
        "mean_eval_loss": sum(eval_losses) / len(eval_losses),
        "embedding_recovery": compute_dictionary_recovery_metrics(
            learned_vectors=model.W_E.detach().cpu(),
            true_vectors=feature_dict.feature_vectors,
        ),
    }
    out_dir = config.out_dir or _default_out_dir(config.run_name)
    save_feature_recovery_target_bundle(out_dir, config, model, feature_dict, label_coeffs, summary)
    print(out_dir)


if __name__ == "__main__":
    fire.Fire(main)

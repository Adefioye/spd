from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fire
import torch

from spd.settings import SPD_OUT_DIR
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed

from koko_experiments.feature_recovery.autoencoder import SparseAutoencoder
from koko_experiments.feature_recovery.configs import SAEBaselineConfig
from koko_experiments.feature_recovery.feature_datasets import HiddenActivationDataset
from koko_experiments.feature_recovery.metrics import evaluate_sae
from koko_experiments.feature_recovery.results import save_sae_bundle
from koko_experiments.feature_recovery.synthetic import ActivationGenerator, FeatureDictionary


def _default_out_dir(run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SPD_OUT_DIR / "feature_recovery" / "sae" / f"{run_name}_{stamp}"


def main(config_path: str) -> None:
    config = SAEBaselineConfig.from_file(config_path)
    set_seed(config.seed)
    device = get_device()

    feature_dict = FeatureDictionary.from_config(config.dataset.dictionary)
    activation_generator = ActivationGenerator(config.dataset.generator, feature_dict.num_features)
    dataset = HiddenActivationDataset(feature_dict, activation_generator, device=device)
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    model = SparseAutoencoder(d_in=feature_dict.hidden_dim, d_sae=config.d_sae).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for step, (hidden, _) in zip(range(config.steps), dataloader, strict=False):
        recon, codes = model(hidden)
        recon_loss = torch.mean((recon - hidden) ** 2)
        l1_loss = codes.abs().mean()
        loss = recon_loss + config.l1_coefficient * l1_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step + 1 == config.steps:
            print(
                f"step={step:05d} loss={loss.item():.6f} recon={recon_loss.item():.6f} l1={l1_loss.item():.6f}"
            )

    eval_hidden, eval_latents = dataset.generate_batch(config.eval_num_samples)
    metrics = evaluate_sae(model, eval_hidden, eval_latents, feature_dict.feature_vectors)
    out_dir = config.out_dir or _default_out_dir(config.run_name)
    save_sae_bundle(out_dir, config, model.state_dict(), feature_dict, metrics)
    print(out_dir)


if __name__ == "__main__":
    fire.Fire(main)

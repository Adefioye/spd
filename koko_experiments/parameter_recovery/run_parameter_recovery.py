import argparse
from dataclasses import asdict
import json
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import Tensor

from spd.configs import Config
from spd.experiments.resid_mlp.models import ResidMLP
from spd.experiments.tms.models import TMSModel
from spd.models.component_model import ComponentModel
from spd.run_spd import run_experiment
from spd.settings import SPD_OUT_DIR
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import get_scheduled_value, set_seed
from spd.utils.module_utils import expand_module_patterns
from spd.utils.run_utils import generate_run_id, save_file

from koko_experiments.parameter_recovery.configs import (
    ParameterRecoveryExperimentConfig,
    TargetTrainingConfig,
)
from koko_experiments.parameter_recovery.feature_datasets import (
    ObservedActivationDataset,
)
from koko_experiments.parameter_recovery.metrics import (
    load_spd_loss_summary,
    summarize_spd_evaluation,
)
from koko_experiments.parameter_recovery.results import (
    load_unified_target_bundle,
    save_unified_target_bundle,
)
from koko_experiments.parameter_recovery.synthetic import ActivationGenerator, FeatureDictionary


def _timestamped_dir(base: Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{run_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPD-only parameter-recovery runner for target training, decomposition, and analysis."
    )
    parser.add_argument("config", type=Path, help="Path to a YAML/JSON experiment config")
    parser.add_argument(
        "--stages",
        default="target,spd,analyze",
        help="Comma-separated subset of target,spd,analyze",
    )
    return parser.parse_args()


def _build_synthetic_family(
    config: ParameterRecoveryExperimentConfig,
    device: str,
) -> tuple[FeatureDictionary, ActivationGenerator]:
    feature_dict = FeatureDictionary.from_config(config.synthetic.dictionary, device=device)
    activation_generator = ActivationGenerator(
        config=config.synthetic.generator,
        num_features=config.synthetic.dictionary.num_features,
        device=device,
    )
    return feature_dict, activation_generator


def _make_label_coeffs(
    observed_dim: int,
    target_config: TargetTrainingConfig,
    device: str,
    seed: int,
) -> Tensor:
    if target_config.label_coeffs is not None:
        coeffs = torch.tensor(target_config.label_coeffs, dtype=torch.float32, device=device)
        assert coeffs.shape == (observed_dim,)
        return coeffs
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(observed_dim, generator=generator, device=device) + 1.0


def _build_target_dataset(
    config: ParameterRecoveryExperimentConfig,
    feature_dict: FeatureDictionary,
    activation_generator: ActivationGenerator,
    device: str,
) -> tuple[ObservedActivationDataset, Tensor]:
    observed_dim = config.synthetic.dictionary.hidden_dim
    label_coeffs = _make_label_coeffs(
        observed_dim=observed_dim,
        target_config=config.target,
        device=device,
        seed=config.seed,
    )
    act_fn_name = "relu"
    if config.target.resid_mlp_model_config is not None:
        act_fn_name = config.target.resid_mlp_model_config.act_fn_name
    dataset = ObservedActivationDataset(
        feature_dict=feature_dict,
        activation_generator=activation_generator,
        label_type=config.target.label_type,
        label_coeffs=label_coeffs,
        act_fn_name=act_fn_name,
        device=device,
    )
    return dataset, label_coeffs


def _build_target_model(
    config: ParameterRecoveryExperimentConfig,
    device: str,
) -> TMSModel | ResidMLP:
    observed_dim = config.synthetic.dictionary.hidden_dim
    if config.target.model_type == "tms":
        assert config.target.tms_model_config is not None
        assert config.target.tms_model_config.n_features == observed_dim, (
            "TMS input dimension must match observed activation dimension"
        )
        return TMSModel(config.target.tms_model_config).to(device)

    assert config.target.resid_mlp_model_config is not None
    assert config.target.resid_mlp_model_config.n_features == observed_dim, (
        "ResidMLP input dimension must match observed activation dimension"
    )
    model = ResidMLP(config.target.resid_mlp_model_config).to(device)
    if config.target.fixed_embedding == "identity":
        model.W_E.data.copy_(torch.eye(model.config.d_embed, device=device))
        model.W_U.data.copy_(torch.eye(model.config.d_embed, device=device))
        model.W_E.requires_grad = False
        model.W_U.requires_grad = False
    elif config.target.fixed_embedding == "random":
        model.W_E.data.copy_(torch.randn_like(model.W_E))
        model.W_E.data /= model.W_E.data.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        model.W_U.data.copy_(torch.linalg.pinv(model.W_E.data))
        model.W_E.requires_grad = False
        model.W_U.requires_grad = False
    return model


def _train_tms_target(
    model: TMSModel,
    dataset: ObservedActivationDataset,
    config: ParameterRecoveryExperimentConfig,
    device: str,
) -> dict[str, float]:
    dataloader = DatasetGeneratedDataLoader(
        dataset,
        batch_size=config.target.batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.target.lr_schedule.start_val)
    eval_losses: list[float] = []

    for step, (observed, labels) in zip(range(config.target.steps), dataloader, strict=False):
        current_lr = get_scheduled_value(step, config.target.steps, config.target.lr_schedule)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        observed = observed.to(device)
        labels = labels.to(device)
        out = model(observed)
        loss = torch.mean((out - labels.abs()) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config.target.print_freq == 0 or step + 1 == config.target.steps:
            print(f"step={step:05d} loss={loss.item():.6f} lr={current_lr:.6f}")

    for _ in range(10):
        observed, labels = dataset.generate_batch(config.target.batch_size)
        with torch.no_grad():
            eval_losses.append(
                float(torch.mean((model(observed.to(device)) - labels.to(device).abs()) ** 2).item())
            )
    return {"mean_eval_loss": sum(eval_losses) / len(eval_losses)}


def _resid_loss(
    model: ResidMLP,
    outputs: Tensor,
    labels: Tensor,
    loss_type: str,
) -> Tensor:
    if loss_type == "readoff":
        return torch.mean((outputs - labels) ** 2)
    residual_labels = labels @ model.W_E
    return torch.mean((outputs - residual_labels) ** 2)


def _train_resid_target(
    model: ResidMLP,
    dataset: ObservedActivationDataset,
    config: ParameterRecoveryExperimentConfig,
    device: str,
) -> dict[str, float]:
    dataloader = DatasetGeneratedDataLoader(
        dataset,
        batch_size=config.target.batch_size,
        shuffle=False,
    )
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.target.lr_schedule.start_val,
        weight_decay=0.01,
    )
    eval_losses: list[float] = []

    for step, (observed, labels) in zip(range(config.target.steps), dataloader, strict=False):
        current_lr = get_scheduled_value(step, config.target.steps, config.target.lr_schedule)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        observed = observed.to(device)
        labels = labels.to(device)
        outputs = model(observed, return_residual=config.target.loss_type == "resid")
        loss = _resid_loss(model, outputs, labels, config.target.loss_type)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config.target.print_freq == 0 or step + 1 == config.target.steps:
            print(f"step={step:05d} loss={loss.item():.6f} lr={current_lr:.6f}")

    for _ in range(10):
        observed, labels = dataset.generate_batch(config.target.batch_size)
        with torch.no_grad():
            outputs = model(observed.to(device), return_residual=config.target.loss_type == "resid")
            eval_losses.append(
                float(_resid_loss(model, outputs, labels.to(device), config.target.loss_type).item())
            )
    return {"mean_eval_loss": sum(eval_losses) / len(eval_losses)}


def _save_target_bundle(
    config: ParameterRecoveryExperimentConfig,
    model: TMSModel | ResidMLP,
    feature_dict: FeatureDictionary,
    label_coeffs: Tensor,
    summary: dict[str, float],
) -> Path:
    base_dir = config.out_dir or SPD_OUT_DIR / "parameter_recovery" / "targets"
    out_dir = _timestamped_dir(base_dir, config.run_name)
    save_unified_target_bundle(
        out_dir=out_dir,
        config=config,
        model=model,
        feature_dict=feature_dict,
        label_coeffs=label_coeffs,
        summary=summary,
    )
    return out_dir


def _prepare_spd_config(path: Path) -> Config:
    return Config.from_file(path)


def _run_spd_stage(
    config: ParameterRecoveryExperimentConfig,
    model: TMSModel | ResidMLP,
    dataset: ObservedActivationDataset,
    target_bundle_dir: Path,
    device: str,
) -> Path:
    spd_config = _prepare_spd_config(config.spd.spd_config_path)
    train_loader = DatasetGeneratedDataLoader(
        dataset,
        batch_size=spd_config.batch_size,
        shuffle=False,
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset,
        batch_size=spd_config.eval_batch_size,
        shuffle=False,
    )
    run_id = generate_run_id("spd")
    run_experiment(
        target_model=model,
        config=spd_config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_tag=f"parameter_recovery_{config.target.model_type}",
        run_id=run_id,
        launch_id=None,
        evals_id=None,
        sweep_params=None,
        target_model_train_config=config,
    )
    run_dir = SPD_OUT_DIR / "spd" / run_id
    feature_vectors = torch.load(
        target_bundle_dir / "feature_dictionary.pt",
        map_location="cpu",
        weights_only=True,
    )["feature_vectors"]
    save_file({"feature_vectors": feature_vectors}, run_dir / "feature_dictionary.pt")
    save_file(
        {"target_bundle_dir": str(target_bundle_dir)},
        run_dir / "parameter_recovery_metadata.json",
        indent=2,
    )
    return run_dir


def _instantiate_target_model(
    config: ParameterRecoveryExperimentConfig,
    model_type: str,
    state_dict: dict[str, Tensor],
) -> TMSModel | ResidMLP:
    if model_type == "tms":
        assert config.target.tms_model_config is not None
        model = TMSModel(config.target.tms_model_config)
    else:
        assert config.target.resid_mlp_model_config is not None
        model = ResidMLP(config.target.resid_mlp_model_config)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model


def _analyze_spd_stage(
    config: ParameterRecoveryExperimentConfig,
    spd_run_dir: Path,
    target_bundle_dir: Path,
) -> Path:
    config_from_bundle, model_type, state_dict, feature_vectors, _ = load_unified_target_bundle(
        target_bundle_dir
    )
    target_model = _instantiate_target_model(config_from_bundle, model_type, state_dict)
    final_config = Config.from_file(spd_run_dir / "final_config.yaml")
    module_path_info = expand_module_patterns(target_model, final_config.all_module_info)
    component_model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_config=final_config.ci_config,
        sigmoid_type=final_config.sigmoid_type,
        pretrained_model_output_attr=final_config.pretrained_model_output_attr,
    )
    checkpoint_path = sorted(spd_run_dir.glob("model*.pth"))[-1]
    component_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    component_model.load_state_dict(component_state)
    component_model.eval()

    spd_summary, layer_metrics = summarize_spd_evaluation(
        component_model=component_model,
        n_probe_features=feature_vectors.shape[0],
        input_magnitude=config.analysis.input_magnitude,
        sampling=config.analysis.sampling,
    )
    output = {
        "spd_run_dir": str(spd_run_dir),
        "target_run_dir": str(target_bundle_dir),
        "summary": asdict(spd_summary),
        "losses": load_spd_loss_summary(spd_run_dir),
        "layers": [asdict(layer) for layer in layer_metrics],
    }
    out_path = spd_run_dir / "parameter_recovery_analysis.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    return out_path


def main() -> None:
    args = _parse_args()
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    assert stages <= {"target", "spd", "analyze"}, "Stages must be drawn from target,spd,analyze"
    if "analyze" in stages:
        assert "spd" in stages, "Analyze stage requires running SPD in the same invocation"

    config = ParameterRecoveryExperimentConfig.from_file(args.config)

    set_seed(config.seed)
    device = get_device()
    feature_dict, activation_generator = _build_synthetic_family(config, device=device)

    dataset, label_coeffs = _build_target_dataset(config, feature_dict, activation_generator, device)
    model = _build_target_model(config, device)
    target_bundle_dir: Path | None = None
    spd_run_dir: Path | None = None

    if {"target", "spd", "analyze"} & stages:
        if config.target.model_type == "tms":
            summary = _train_tms_target(model, dataset, config, device)
        else:
            assert isinstance(model, ResidMLP)
            summary = _train_resid_target(model, dataset, config, device)
        target_bundle_dir = _save_target_bundle(config, model, feature_dict, label_coeffs, summary)
        print(f"target_dir={target_bundle_dir}")

    if "spd" in stages:
        assert target_bundle_dir is not None
        spd_run_dir = _run_spd_stage(config, model, dataset, target_bundle_dir, device)
        print(f"spd_dir={spd_run_dir}")

    if "analyze" in stages:
        assert target_bundle_dir is not None
        assert spd_run_dir is not None
        analysis_path = _analyze_spd_stage(config, spd_run_dir, target_bundle_dir)
        print(f"analysis_path={analysis_path}")


if __name__ == "__main__":
    main()

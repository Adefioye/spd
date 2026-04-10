import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from spd.configs import Config, TMSTaskConfig
from spd.experiments.tms.models import TMSModel, TMSTargetRunInfo
from spd.models.component_model import ComponentModel, handle_deprecated_state_dict_keys_
from spd.plotting import get_single_feature_causal_importances
from spd.utils.data_utils import SparseFeatureDataset
from spd.utils.module_utils import expand_module_patterns


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "spd_out" / "parameter_recovery" / "results"
OUTPUTS_DIR = Path(
    os.environ.get(
        "SHRINKAGE_ANALYSIS_OUTPUT_DIR",
        str(REPO_ROOT / "koko_notebooks" / "shrinkage_analysis" / "outputs"),
    )
)
PLOTS_DIR = OUTPUTS_DIR / "plots"
EXP07_RUN_RE = re.compile(
    r"^exp_07_tms_5_2_(?P<depth>\d)layer_(?P<architecture>tied|untied)(?:_(?P<suffix>.*))?$"
)


@dataclass(frozen=True)
class Exp07RunMetadata:
    run_name: str
    depth: int
    architecture: str
    replicate: int
    suffix: str


def parse_exp07_run_name(run_name: str) -> Exp07RunMetadata:
    match = EXP07_RUN_RE.match(run_name)
    assert match is not None, f"Unrecognized exp_07 run name: {run_name}"
    raw_suffix = match.group("suffix") or ""
    suffix_tokens = [token for token in raw_suffix.split("_") if token]
    replicate = 1
    if suffix_tokens and re.fullmatch(r"rep\d+", suffix_tokens[-1]) is not None:
        replicate = int(suffix_tokens[-1][3:])
        suffix_tokens = suffix_tokens[:-1]
    suffix = "_".join(suffix_tokens)
    return Exp07RunMetadata(
        run_name=run_name,
        depth=int(match.group("depth")),
        architecture=str(match.group("architecture")),
        replicate=replicate,
        suffix=suffix,
    )


def available_checkpoint_steps(spd_run_dir: Path) -> list[int]:
    checkpoint_paths = sorted(spd_run_dir.glob("model_*.pth"))
    assert checkpoint_paths, f"No checkpoints found in {spd_run_dir}"
    return sorted(int(path.stem.split("_")[-1]) for path in checkpoint_paths)


def checkpoint_path_for_step(spd_run_dir: Path, step: int) -> Path:
    checkpoint_path = spd_run_dir / f"model_{step}.pth"
    assert checkpoint_path.exists(), f"Missing checkpoint: {checkpoint_path}"
    return checkpoint_path


def load_analysis_payload(path: Path) -> dict[str, object]:
    assert path.exists(), f"Missing analysis payload: {path}"
    return json.loads(path.read_text())


def discover_exp07_analysis_jsons(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    assert results_dir.exists(), f"Missing results directory: {results_dir}"
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("exp_07_tms_5_2*_parameter_recovery_analysis.json")):
        payload = load_analysis_payload(path)
        run_name = str(payload["run_name"])
        metadata = parse_exp07_run_name(run_name)
        spd_run_dir = Path(str(payload["spd_run_dir"]))
        target_run_path = Path(str(payload["target_run_dir"]))
        rows.append(
            {
                "analysis_json_path": str(path),
                "run_name": run_name,
                "depth": metadata.depth,
                "architecture": metadata.architecture,
                "replicate": metadata.replicate,
                "suffix": metadata.suffix,
                "spd_run_dir": str(spd_run_dir),
                "target_run_path": str(target_run_path),
                "checkpoint_steps": available_checkpoint_steps(spd_run_dir),
            }
        )
    assert rows, f"No exp_07 analysis JSONs found under {results_dir}"
    return pd.DataFrame(rows).sort_values(
        ["depth", "architecture", "run_name", "replicate"]
    ).reset_index(drop=True)


def latest_result_per_run(manifest_df: pd.DataFrame) -> pd.DataFrame:
    latest_rows = []
    for run_name, group in manifest_df.groupby("run_name", sort=False):
        latest_rows.append(group.iloc[-1])
    return pd.DataFrame(latest_rows).reset_index(drop=True)


def select_consistent_replicate(
    manifest_df: pd.DataFrame,
    replicate: int,
) -> pd.DataFrame:
    filtered_df = manifest_df[manifest_df["replicate"] == replicate].copy()
    assert not filtered_df.empty, f"No rows found for replicate {replicate}"
    condition_counts = (
        filtered_df.groupby(["depth", "architecture"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
    )
    assert (condition_counts["n_rows"] == 1).all(), (
        "Expected exactly one run per depth/architecture for the selected replicate; "
        f"got counts:\n{condition_counts.to_string(index=False)}"
    )
    expected_conditions = {
        (depth, architecture) for depth in [2, 3, 4, 5, 6] for architecture in ["tied", "untied"]
    }
    observed_conditions = set(zip(filtered_df["depth"], filtered_df["architecture"], strict=True))
    missing_conditions = expected_conditions - observed_conditions
    assert not missing_conditions, (
        f"Missing depth/architecture pairs for replicate {replicate}: {sorted(missing_conditions)}"
    )
    return filtered_df.sort_values(["depth", "architecture"]).reset_index(drop=True)


def load_metrics_jsonl(spd_run_dir: Path) -> pd.DataFrame:
    metrics_path = spd_run_dir / "metrics.jsonl"
    assert metrics_path.exists(), f"Missing metrics file: {metrics_path}"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    assert rows, f"Empty metrics file: {metrics_path}"
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def load_target_model_from_spd_run_dir(spd_run_dir: Path) -> TMSModel:
    tms_checkpoint = spd_run_dir / "tms.pth"
    if tms_checkpoint.exists():
        run_info = TMSTargetRunInfo.from_path(tms_checkpoint)
        target_model = TMSModel.from_run_info(run_info)
    else:
        final_config = Config.from_file(spd_run_dir / "final_config.yaml")
        assert final_config.pretrained_model_path is not None, (
            f"No saved target checkpoint in {spd_run_dir} and pretrained_model_path is unset"
        )
        run_info = TMSTargetRunInfo.from_path(final_config.pretrained_model_path)
        target_model = TMSModel.from_run_info(run_info)
    target_model.eval()
    target_model.requires_grad_(False)
    return target_model


def load_component_model_for_checkpoint(
    spd_run_dir: Path,
    step: int,
    device: str = "cpu",
) -> tuple[ComponentModel, TMSModel, Config]:
    final_config = Config.from_file(spd_run_dir / "final_config.yaml")
    target_model = load_target_model_from_spd_run_dir(spd_run_dir).to(device)
    module_path_info = expand_module_patterns(target_model, final_config.all_module_info)
    component_model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_config=final_config.ci_config,
        sigmoid_type=final_config.sigmoid_type,
        pretrained_model_output_attr=final_config.pretrained_model_output_attr,
    )
    state_dict = torch.load(checkpoint_path_for_step(spd_run_dir, step), map_location=device, weights_only=True)
    handle_deprecated_state_dict_keys_(state_dict)
    component_model.load_state_dict(state_dict)
    component_model.to(device)
    component_model.eval()
    return component_model, target_model, final_config


def singleton_probe_batch(n_features: int, input_magnitude: float, device: str) -> Tensor:
    return torch.eye(n_features, device=device) * input_magnitude


def exhaustive_binary_probe_batch(n_features: int, device: str) -> Tensor:
    values = torch.arange(2**n_features, device=device)
    bits = ((values[:, None] >> torch.arange(n_features, device=device)) & 1).float()
    return bits


def sampled_probe_batch(
    task_config: TMSTaskConfig,
    batch_size: int,
    device: str,
    seed: int,
) -> Tensor:
    assert task_config.task_name == "tms"
    torch.manual_seed(seed)
    dataset = SparseFeatureDataset(
        n_features=5,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=None,
    )
    batch, _labels = dataset.generate_batch(batch_size)
    return batch


def collect_pre_weight_acts(component_model: ComponentModel, batch: Tensor) -> dict[str, Tensor]:
    return component_model(batch, cache_type="input").cache


def collect_ci_outputs(
    component_model: ComponentModel,
    batch: Tensor,
    sampling: str,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    pre_weight_acts = collect_pre_weight_acts(component_model, batch)
    ci_outputs = component_model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts, sampling=sampling
    )
    return ci_outputs.pre_sigmoid, ci_outputs.lower_leaky, ci_outputs.upper_leaky


def expected_component_mask(ci_lower: Tensor) -> Tensor:
    return (1.0 + ci_lower) / 2.0


def batched_expected_masked_weight(components: torch.nn.Module, ci_lower: Tensor) -> Tensor:
    expected_mask = expected_component_mask(ci_lower)
    return torch.einsum("bc,ic,co->boi", expected_mask, components.V.detach(), components.U.detach())


def batched_expected_train_mask_weight(
    components: torch.nn.Module,
    ci_lower: Tensor,
    delta_weight: Tensor,
) -> Tensor:
    masked_weight = batched_expected_masked_weight(components, ci_lower)
    return masked_weight + 0.5 * delta_weight.detach().unsqueeze(0)


def component_strengths(components: torch.nn.Module) -> Tensor:
    u_norm = components.U.detach().norm(dim=-1)
    v_norm = components.V.detach().norm(dim=0)
    return u_norm * v_norm


def layer_weight_metric_row(layer_name: str, target_weight: Tensor, raw_weight: Tensor, delta_weight: Tensor) -> dict[str, object]:
    target_singular_values = torch.linalg.svdvals(target_weight.detach())
    raw_singular_values = torch.linalg.svdvals(raw_weight.detach())
    return {
        "layer_name": layer_name,
        "target_fro_norm": float(target_weight.norm().item()),
        "raw_fro_norm": float(raw_weight.norm().item()),
        "delta_fro_norm": float(delta_weight.norm().item()),
        "raw_fro_ratio": float((raw_weight.norm() / target_weight.norm().clamp_min(1e-12)).item()),
        "delta_fro_ratio": float((delta_weight.norm() / target_weight.norm().clamp_min(1e-12)).item()),
        "raw_spectral_ratio": float(
            (
                torch.linalg.matrix_norm(raw_weight, ord=2)
                / torch.linalg.matrix_norm(target_weight, ord=2).clamp_min(1e-12)
            ).item()
        ),
        "faithfulness_mse": float(torch.mean((raw_weight - target_weight) ** 2).item()),
        "target_singular_values": target_singular_values.tolist(),
        "raw_singular_values": raw_singular_values.tolist(),
    }


def ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def save_dataframe(df: pd.DataFrame, relative_path: str) -> Path:
    out_dir = ensure_outputs_dir()
    output_path = out_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_json(obj: object, relative_path: str) -> Path:
    out_dir = ensure_outputs_dir()
    output_path = out_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(obj, indent=2))
    return output_path


def canonical_singleton_ci_summary(
    component_model: ComponentModel,
    input_magnitude: float,
    sampling: str,
) -> dict[str, dict[str, float]]:
    ci_output = get_single_feature_causal_importances(
        model=component_model,
        batch_shape=(5, 5),
        input_magnitude=input_magnitude,
        sampling=sampling,
    )
    summary: dict[str, dict[str, float]] = {}
    for layer_name, values in ci_output.lower_leaky.items():
        summary[layer_name] = {
            "mean_lower_ci": float(values.mean().item()),
            "mean_top1_lower_ci": float(values.max(dim=-1).values.mean().item()),
        }
    return summary

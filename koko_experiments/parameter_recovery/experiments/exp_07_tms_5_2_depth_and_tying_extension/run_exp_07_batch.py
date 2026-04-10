# ruff: noqa: E402

import argparse
import json
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koko_experiments.parameter_recovery.tms_pretrained_analysis import analyze_tms_spd_run
from spd.base_config import BaseConfig
from spd.configs import Config, TMSTaskConfig
from spd.experiments.tms.configs import TMSTrainConfig
from spd.experiments.tms.models import TMSModel, TMSTargetRunInfo
from spd.experiments.tms.train_tms import run_train
from spd.run_spd import run_experiment
from spd.settings import SPD_OUT_DIR
from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import generate_run_id


class Exp07RunSpec(BaseConfig):
    run_name: str
    target_config: TMSTrainConfig
    spd_config: Config


class Exp07BatchConfig(BaseConfig):
    wandb_project: str
    runs: list[Exp07RunSpec]


def _analysis_output_path(base: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}_{stamp}_parameter_recovery_analysis.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp_07 TMS depth/tied-vs-untied target training, SPD, and analysis."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "exp_07_batch.yaml",
        help="Path to the exp_07 batch YAML.",
    )
    parser.add_argument(
        "--run-names",
        type=str,
        default=None,
        help="Optional comma-separated subset of run names to execute.",
    )
    parser.add_argument(
        "--reuse-existing-targets",
        action="store_true",
        help="Reuse previously saved exp_07 target checkpoints instead of retraining target models.",
    )
    parser.add_argument(
        "--spd-steps",
        type=int,
        default=None,
        help="Override the SPD optimization step count for all selected runs.",
    )
    parser.add_argument(
        "--run-name-suffix",
        type=str,
        default="",
        help="Optional suffix appended to each run name for W&B naming and result bundles.",
    )
    parser.add_argument(
        "--spd-replicates",
        type=int,
        default=1,
        help="Number of SPD runs to execute per selected target checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Execution device: "auto", "cuda", or "cpu".',
    )
    return parser.parse_args()


def _require_local_spd_out_dir() -> None:
    expected = (REPO_ROOT / "spd_out").resolve()
    actual = SPD_OUT_DIR.resolve()
    assert actual == expected, (
        f"SPD_OUT_DIR must be {expected} for exp_07, got {actual}. "
        'Run with: export SPD_OUT_DIR="$PWD/spd_out"'
    )


def _materialize_target_config(
    target_config: TMSTrainConfig, wandb_project: str, device: str
) -> TMSTrainConfig:
    return target_config.model_copy(
        update={
            "wandb_project": target_config.wandb_project or wandb_project,
            "tms_model_config": target_config.tms_model_config.model_copy(
                update={"device": device}
            ),
        }
    )


def _materialize_spd_config(spd_config: Config, wandb_project: str, run_name: str) -> Config:
    return spd_config.model_copy(
        update={
            "wandb_project": spd_config.wandb_project or wandb_project,
            "wandb_run_name": spd_config.wandb_run_name or run_name,
        }
    )


def _run_name_with_suffix(run_name: str, suffix: str) -> str:
    return run_name if suffix == "" else f"{run_name}_{suffix}"


def _replicate_run_name(run_name: str, replicate_idx: int, n_replicates: int) -> str:
    if n_replicates == 1:
        return run_name
    return f"{run_name}_rep{replicate_idx}"


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    assert device_arg in {"cuda", "cpu"}, f"Unsupported device: {device_arg}"
    if device_arg == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but no GPU is available"
    return device_arg


def _find_existing_target_checkpoint(run_name: str) -> Path:
    result_root = SPD_OUT_DIR / "parameter_recovery" / "results"
    candidate_paths = sorted(result_root.glob(f"{run_name}_*_parameter_recovery_analysis.json"))
    assert candidate_paths, f"No saved analysis JSONs found for {run_name} under {result_root}"

    latest_path = candidate_paths[-1]
    analysis_payload = json.loads(latest_path.read_text())
    target_checkpoint_path = Path(str(analysis_payload["target_run_dir"]))
    assert target_checkpoint_path.exists(), (
        f"Saved target checkpoint for {run_name} does not exist at {target_checkpoint_path}"
    )
    return target_checkpoint_path


def _run_spd_for_target(target_checkpoint_path: Path, spd_config: Config, device: str) -> Path:
    target_run_info = TMSTargetRunInfo.from_path(target_checkpoint_path)
    target_model = TMSModel.from_run_info(target_run_info).to(device)
    target_model.eval()

    task_config = spd_config.task_config
    assert isinstance(task_config, TMSTaskConfig)

    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=target_run_info.config.synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=spd_config.batch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=spd_config.eval_batch_size, shuffle=False
    )
    tied_weights = [("linear1", "linear2")] if target_model.config.tied_weights else None

    run_id = generate_run_id("spd")
    run_experiment(
        target_model=target_model,
        config=spd_config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_tag="parameter_recovery_exp_07_tms",
        run_id=run_id,
        launch_id=None,
        evals_id=None,
        sweep_params=None,
        target_model_train_config=target_run_info.config,
        tied_weights=tied_weights,
    )
    spd_run_dir = SPD_OUT_DIR / "spd" / run_id
    assert spd_run_dir.exists(), f"Missing SPD run dir at {spd_run_dir}"
    return spd_run_dir


def _save_run_outputs(
    run_name: str,
    spd_run_dir: Path,
) -> Path:
    analysis_payload = analyze_tms_spd_run(spd_run_dir=spd_run_dir, run_name=run_name)
    spd_out_path = spd_run_dir / "parameter_recovery_analysis.json"
    spd_out_path.write_text(json.dumps(analysis_payload, indent=2, default=str))

    analysis_path = _analysis_output_path(SPD_OUT_DIR / "parameter_recovery" / "results", run_name)
    analysis_path.write_text(json.dumps(analysis_payload, indent=2, default=str))
    important_outputs_dir = REPO_ROOT / "important_outputs"
    important_outputs_dir.mkdir(parents=True, exist_ok=True)
    important_outputs_path = important_outputs_dir / analysis_path.name
    shutil.copy2(analysis_path, important_outputs_path)
    print(f"analysis_path={analysis_path}")
    print(f"important_outputs_path={important_outputs_path}")
    return analysis_path


def _selected_run_names(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    selected = {item.strip() for item in raw.split(",") if item.strip()}
    assert selected, "At least one non-empty run name is required when using --run-names"
    return selected


def _write_index(index_path: Path, rows: list[dict[str, str]]) -> None:
    index_path.write_text(json.dumps(rows, indent=2))


def _extract_linear1_metrics(analysis_payload: dict[str, object]) -> dict[str, float]:
    layers = analysis_payload["layers"]
    assert isinstance(layers, list), "Expected layers to be a list"
    for layer in layers:
        assert isinstance(layer, dict), "Expected each layer record to be a dict"
        if layer["layer_name"] == "linear1":
            return {
                "mmcs": float(layer["mmcs"]),
                "ml2r": float(layer["ml2r"]),
            }
    raise AssertionError("linear1 metrics not found in analysis payload")


def _extract_other_layers_metrics(analysis_payload: dict[str, object]) -> dict[str, float]:
    layers = analysis_payload["layers"]
    assert isinstance(layers, list), "Expected layers to be a list"

    other_layers = []
    for layer in layers:
        assert isinstance(layer, dict), "Expected each layer record to be a dict"
        if layer["layer_name"] != "linear1":
            other_layers.append(layer)

    assert other_layers, "Expected at least one non-linear1 layer in analysis payload"
    return {
        "mmcs": statistics.mean(float(layer["mmcs"]) for layer in other_layers),
        "ml2r": statistics.mean(float(layer["ml2r"]) for layer in other_layers),
    }


def _summarize_replicates(
    replicate_reports: dict[str, list[dict[str, object]]],
    run_name_suffix: str,
    metric_group_name: str,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = f"exp_07_tms_5_2_replicate_{metric_group_name}_summary"
    if run_name_suffix:
        batch_name = f"{batch_name}_{run_name_suffix}"

    summary_payload: dict[str, object] = {
        "experiment_name": batch_name,
        "generated_at": datetime.now().isoformat(),
        "n_base_runs": len(replicate_reports),
        "base_runs": {},
    }
    mmcs_key = f"{metric_group_name}_mmcs"
    ml2r_key = f"{metric_group_name}_ml2r"
    for base_run_name in sorted(replicate_reports):
        reports = replicate_reports[base_run_name]
        mmcs_values = [float(report[mmcs_key]) for report in reports]
        ml2r_values = [float(report[ml2r_key]) for report in reports]
        summary_payload["base_runs"][base_run_name] = {
            "n_replicates": len(reports),
            metric_group_name: {
                "MMCS": {
                    "mean": statistics.mean(mmcs_values),
                    "std": statistics.stdev(mmcs_values) if len(mmcs_values) > 1 else 0.0,
                    "values": mmcs_values,
                },
                "ML2R": {
                    "mean": statistics.mean(ml2r_values),
                    "std": statistics.stdev(ml2r_values) if len(ml2r_values) > 1 else 0.0,
                    "values": ml2r_values,
                },
            },
            "replicates": reports,
        }

    result_root = SPD_OUT_DIR / "parameter_recovery" / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / f"{batch_name}_{stamp}.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    important_outputs_dir = REPO_ROOT / "important_outputs"
    important_outputs_dir.mkdir(parents=True, exist_ok=True)
    important_outputs_path = important_outputs_dir / summary_path.name
    shutil.copy2(summary_path, important_outputs_path)
    print(f"replicate_summary_path={summary_path}")
    print(f"replicate_summary_important_outputs_path={important_outputs_path}")
    return summary_path


def main() -> None:
    args = _parse_args()
    _require_local_spd_out_dir()
    assert args.spd_replicates >= 1, "--spd-replicates must be at least 1"
    device = _resolve_device(args.device)

    config_path = args.config_path.expanduser().resolve()
    assert config_path.exists(), f"Missing batch config at {config_path}"
    batch_config = Exp07BatchConfig.from_file(config_path)
    selected_run_names = _selected_run_names(args.run_names)

    materialized_root = Path(__file__).resolve().parent / "materialized_configs"
    materialized_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []
    replicate_reports: dict[str, list[dict[str, object]]] = {}

    print(f"SPD_OUT_DIR={SPD_OUT_DIR}")
    print(f"wandb_project={batch_config.wandb_project}")
    print(f"device={device}")
    print(f"spd_replicates={args.spd_replicates}")

    for run_spec in batch_config.runs:
        if selected_run_names is not None and run_spec.run_name not in selected_run_names:
            continue

        effective_base_run_name = _run_name_with_suffix(run_spec.run_name, args.run_name_suffix)
        print(f"=== Starting {effective_base_run_name} ===")
        target_config = _materialize_target_config(
            run_spec.target_config, batch_config.wandb_project, device
        )

        materialized_dir = materialized_root / effective_base_run_name
        materialized_dir.mkdir(parents=True, exist_ok=True)
        target_config_path = materialized_dir / "target_train_config.yaml"
        target_config.to_file(target_config_path)

        if args.reuse_existing_targets:
            target_checkpoint_path = _find_existing_target_checkpoint(run_spec.run_name)
            target_run_dir = target_checkpoint_path.parent
            print(f"reused_target_checkpoint_path={target_checkpoint_path}")
        else:
            set_seed(target_config.seed)
            target_run_dir = run_train(target_config, device=device)
            target_checkpoint_path = target_run_dir / "tms.pth"
            assert target_checkpoint_path.exists(), (
                f"Missing target checkpoint at {target_checkpoint_path}"
            )
            print(f"target_run_dir={target_run_dir}")

        replicate_reports[effective_base_run_name] = []
        for replicate_idx in range(1, args.spd_replicates + 1):
            effective_run_name = _replicate_run_name(
                effective_base_run_name, replicate_idx, args.spd_replicates
            )
            print(
                f"--- SPD replicate {replicate_idx}/{args.spd_replicates}: {effective_run_name} ---"
            )

            spd_config = _materialize_spd_config(
                run_spec.spd_config, batch_config.wandb_project, effective_run_name
            )
            if args.spd_steps is not None:
                spd_config = spd_config.model_copy(update={"steps": args.spd_steps})

            materialized_spd_config = spd_config.model_copy(
                update={"pretrained_model_path": str(target_checkpoint_path)}
            )
            spd_config_path = materialized_dir / f"spd_config_rep{replicate_idx}.yaml"
            materialized_spd_config.to_file(spd_config_path)

            spd_run_dir = _run_spd_for_target(
                target_checkpoint_path, materialized_spd_config, device
            )
            print(f"spd_run_dir={spd_run_dir}")

            analysis_path = _save_run_outputs(
                run_name=effective_run_name,
                spd_run_dir=spd_run_dir,
            )
            analysis_payload = json.loads(analysis_path.read_text())
            linear1_metrics = _extract_linear1_metrics(analysis_payload)
            other_layers_metrics = _extract_other_layers_metrics(analysis_payload)
            replicate_record: dict[str, object] = {
                "replicate_index": replicate_idx,
                "run_name": effective_run_name,
                "source_run_name": run_spec.run_name,
                "target_run_dir": str(target_run_dir),
                "spd_run_dir": str(spd_run_dir),
                "analysis_path": str(analysis_path),
                "linear1_mmcs": linear1_metrics["mmcs"],
                "linear1_ml2r": linear1_metrics["ml2r"],
                "other_layers_mmcs": other_layers_metrics["mmcs"],
                "other_layers_ml2r": other_layers_metrics["ml2r"],
            }
            summary_rows.append({k: str(v) for k, v in replicate_record.items()})
            replicate_reports[effective_base_run_name].append(replicate_record)
            _write_index(materialized_root / "latest_run_index.json", summary_rows)

        print(f"=== Finished {effective_base_run_name} ===")

    _summarize_replicates(replicate_reports, args.run_name_suffix, metric_group_name="linear1")
    _summarize_replicates(replicate_reports, args.run_name_suffix, metric_group_name="other_layers")


if __name__ == "__main__":
    main()

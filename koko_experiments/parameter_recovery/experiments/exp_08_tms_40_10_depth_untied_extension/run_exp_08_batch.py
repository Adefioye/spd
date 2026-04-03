# ruff: noqa: E402

import argparse
import json
import shutil
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


class Exp08RunSpec(BaseConfig):
    run_name: str
    target_config: TMSTrainConfig
    spd_config: Config


class Exp08BatchConfig(BaseConfig):
    wandb_project: str
    runs: list[Exp08RunSpec]


def _timestamped_dir(base: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp_08 TMS-40-10 untied depth target training, SPD, and analysis."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "exp_08_batch.yaml",
        help="Path to the exp_08 batch YAML.",
    )
    parser.add_argument(
        "--run-names",
        type=str,
        default=None,
        help="Optional comma-separated subset of run names to execute.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Execution device, e.g. "cpu", "cuda", or "cuda:0".',
    )
    parser.add_argument(
        "--reuse-existing-targets",
        action="store_true",
        help="Reuse previously saved exp_08 target checkpoints instead of retraining target models.",
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
    return parser.parse_args()


def _require_local_spd_out_dir() -> None:
    expected = (REPO_ROOT / "spd_out").resolve()
    actual = SPD_OUT_DIR.resolve()
    assert actual == expected, (
        f"SPD_OUT_DIR must be {expected} for exp_08, got {actual}. "
        'Run with: export SPD_OUT_DIR="$PWD/spd_out"'
    )


def _validate_device(device: str) -> None:
    assert device == "cpu" or device.startswith("cuda"), (
        f'Unsupported device "{device}". Expected "cpu" or "cuda[:index]".'
    )
    if device != "cpu":
        assert torch.cuda.is_available(), f"Requested device {device}, but CUDA is not available"


def _materialize_target_config(
    target_config: TMSTrainConfig, wandb_project: str, device: str
) -> TMSTrainConfig:
    return target_config.model_copy(
        update={
            "wandb_project": target_config.wandb_project or wandb_project,
            "tms_model_config": target_config.tms_model_config.model_copy(update={"device": device}),
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


def _find_existing_target_checkpoint(run_name: str) -> Path:
    result_root = SPD_OUT_DIR / "parameter_recovery" / "results"
    candidate_paths = sorted(
        result_root.glob(f"{run_name}_*/{run_name}_parameter_recovery_analysis.json")
    )
    assert candidate_paths, f"No saved analysis bundles found for {run_name} under {result_root}"

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
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=spd_config.batch_size, shuffle=False)
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
        experiment_tag="parameter_recovery_exp_08_tms",
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
    source_batch_config: Path,
    target_config_path: Path,
    spd_config_path: Path,
    spd_run_dir: Path,
) -> Path:
    analysis_payload = analyze_tms_spd_run(spd_run_dir=spd_run_dir, run_name=run_name)
    spd_out_path = spd_run_dir / "parameter_recovery_analysis.json"
    spd_out_path.write_text(json.dumps(analysis_payload, indent=2, default=str))

    result_root = _timestamped_dir(SPD_OUT_DIR / "parameter_recovery" / "results", run_name)
    shutil.copy2(source_batch_config, result_root / source_batch_config.name)
    shutil.copy2(target_config_path, result_root / target_config_path.name)
    shutil.copy2(spd_config_path, result_root / spd_config_path.name)
    analysis_path = result_root / f"{run_name}_parameter_recovery_analysis.json"
    analysis_path.write_text(json.dumps(analysis_payload, indent=2, default=str))
    print(f"analysis_path={analysis_path}")
    return analysis_path


def _selected_run_names(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    selected = {item.strip() for item in raw.split(",") if item.strip()}
    assert selected, "At least one non-empty run name is required when using --run-names"
    return selected


def _write_index(index_path: Path, rows: list[dict[str, str]]) -> None:
    index_path.write_text(json.dumps(rows, indent=2))


def main() -> None:
    args = _parse_args()
    _require_local_spd_out_dir()
    _validate_device(args.device)

    config_path = args.config_path.expanduser().resolve()
    assert config_path.exists(), f"Missing batch config at {config_path}"
    batch_config = Exp08BatchConfig.from_file(config_path)
    selected_run_names = _selected_run_names(args.run_names)

    materialized_root = Path(__file__).resolve().parent / "materialized_configs"
    materialized_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []

    print(f"SPD_OUT_DIR={SPD_OUT_DIR}")
    print(f"wandb_project={batch_config.wandb_project}")
    print(f"device={args.device}")

    for run_spec in batch_config.runs:
        if selected_run_names is not None and run_spec.run_name not in selected_run_names:
            continue

        effective_run_name = _run_name_with_suffix(run_spec.run_name, args.run_name_suffix)
        print(f"=== Starting {effective_run_name} ===")
        target_config = _materialize_target_config(
            run_spec.target_config, batch_config.wandb_project, args.device
        )
        spd_config = _materialize_spd_config(
            run_spec.spd_config, batch_config.wandb_project, effective_run_name
        )
        if args.spd_steps is not None:
            spd_config = spd_config.model_copy(update={"steps": args.spd_steps})

        materialized_dir = materialized_root / effective_run_name
        materialized_dir.mkdir(parents=True, exist_ok=True)
        target_config_path = materialized_dir / "target_train_config.yaml"
        target_config.to_file(target_config_path)

        if args.reuse_existing_targets:
            target_checkpoint_path = _find_existing_target_checkpoint(run_spec.run_name)
            target_run_dir = target_checkpoint_path.parent
            print(f"reused_target_checkpoint_path={target_checkpoint_path}")
        else:
            set_seed(target_config.seed)
            target_run_dir = run_train(target_config, device=args.device)
            target_checkpoint_path = target_run_dir / "tms.pth"
            assert target_checkpoint_path.exists(), f"Missing target checkpoint at {target_checkpoint_path}"
            print(f"target_run_dir={target_run_dir}")

        materialized_spd_config = spd_config.model_copy(
            update={"pretrained_model_path": str(target_checkpoint_path)}
        )
        spd_config_path = materialized_dir / "spd_config.yaml"
        materialized_spd_config.to_file(spd_config_path)

        spd_run_dir = _run_spd_for_target(target_checkpoint_path, materialized_spd_config, args.device)
        print(f"spd_run_dir={spd_run_dir}")

        analysis_path = _save_run_outputs(
            run_name=effective_run_name,
            source_batch_config=config_path,
            target_config_path=target_config_path,
            spd_config_path=spd_config_path,
            spd_run_dir=spd_run_dir,
        )
        summary_rows.append(
            {
                "run_name": effective_run_name,
                "source_run_name": run_spec.run_name,
                "target_run_dir": str(target_run_dir),
                "spd_run_dir": str(spd_run_dir),
                "analysis_path": str(analysis_path),
            }
        )
        _write_index(materialized_root / "latest_run_index.json", summary_rows)
        print(f"=== Finished {effective_run_name} ===")


if __name__ == "__main__":
    main()

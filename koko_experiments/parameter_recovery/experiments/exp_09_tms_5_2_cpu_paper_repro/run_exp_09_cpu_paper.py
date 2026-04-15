# ruff: noqa: E402

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spd.configs import Config, TMSTaskConfig
from spd.experiments.tms.models import TMSModel, TMSTargetRunInfo
from spd.run_spd import run_experiment
from spd.settings import SPD_OUT_DIR
from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import generate_run_id, parse_config

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "koko_experiments" / "configs" / "tms_5-2_cpu_paper.yaml"
DEFAULT_WANDB_PROJECT = "spd_tms_5_2_exp09"
DEFAULT_WANDB_RUN_NAME = "exp_09_tms_5_2_cpu_paper"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp_09: TMS 5-2 paper-style CPU SPD decomposition with explicit W&B naming."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the base SPD YAML config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Execution device, expected to be "cpu" for this experiment.',
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="W&B project name to use for the SPD run.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=DEFAULT_WANDB_RUN_NAME,
        help="W&B display name to use for the SPD run.",
    )
    parser.add_argument(
        "--spd-steps",
        type=int,
        default=None,
        help="Optional override for the SPD optimization step count.",
    )
    return parser.parse_args()


def _require_local_spd_out_dir() -> None:
    expected = (REPO_ROOT / "spd_out").resolve()
    actual = SPD_OUT_DIR.resolve()
    assert actual == expected, (
        f"SPD_OUT_DIR must be {expected} for exp_09, got {actual}. "
        'Run with: export SPD_OUT_DIR="$PWD/spd_out"'
    )


def _materialize_spd_config(
    base_config: Config,
    wandb_project: str,
    wandb_run_name: str,
    spd_steps: int | None,
) -> Config:
    updates: dict[str, object] = {
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    }
    if spd_steps is not None:
        updates["steps"] = spd_steps
    return base_config.model_copy(update=updates)


def _write_run_record(
    run_id: str,
    spd_run_dir: Path,
    materialized_config_path: Path,
    config: Config,
) -> None:
    record_dir = EXPERIMENT_DIR / "run_records"
    record_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "run_dir": str(spd_run_dir),
        "wandb_project": config.wandb_project,
        "wandb_run_name": config.wandb_run_name,
        "materialized_config_path": str(materialized_config_path),
        "steps": config.steps,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    record_path = record_dir / f"{run_id}.json"
    record_path.write_text(json.dumps(payload, indent=2))
    (record_dir / "last_run.json").write_text(json.dumps(payload, indent=2))
    print(f"run_record_path={record_path}")


def main() -> None:
    args = _parse_args()
    _require_local_spd_out_dir()
    assert args.device == "cpu", f'exp_09 is configured for CPU execution, got "{args.device}"'

    config_path = args.config_path.expanduser().resolve()
    assert config_path.exists(), f"Missing config path at {config_path}"
    base_config = parse_config(config_path=config_path, config_json=None)
    effective_config = _materialize_spd_config(
        base_config=base_config,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        spd_steps=args.spd_steps,
    )

    materialized_dir = EXPERIMENT_DIR / "materialized_configs"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    materialized_config_path = materialized_dir / f"{effective_config.wandb_run_name}.yaml"
    effective_config.to_file(materialized_config_path)
    print(f"materialized_config_path={materialized_config_path}")

    set_seed(effective_config.seed)

    task_config = effective_config.task_config
    assert isinstance(task_config, TMSTaskConfig)
    assert effective_config.pretrained_model_path is not None, "pretrained_model_path must be set"

    target_run_info = TMSTargetRunInfo.from_path(effective_config.pretrained_model_path)
    target_model = TMSModel.from_run_info(target_run_info).to(args.device)
    target_model.eval()

    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=args.device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=target_run_info.config.synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=effective_config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=effective_config.eval_batch_size, shuffle=False
    )

    tied_weights = [("linear1", "linear2")] if target_model.config.tied_weights else None
    run_id = generate_run_id("spd")
    print(f"run_id={run_id}")
    print(f"wandb_project={effective_config.wandb_project}")
    print(f"wandb_run_name={effective_config.wandb_run_name}")
    print(f"configured_steps={effective_config.steps}")

    run_experiment(
        target_model=target_model,
        config=effective_config,
        device=args.device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_tag="parameter_recovery_exp_09_tms",
        run_id=run_id,
        launch_id=None,
        evals_id=None,
        sweep_params=None,
        target_model_train_config=target_run_info.config,
        tied_weights=tied_weights,
    )

    spd_run_dir = SPD_OUT_DIR / "spd" / run_id
    assert spd_run_dir.exists(), f"Missing SPD run dir at {spd_run_dir}"
    print(f"spd_run_dir={spd_run_dir}")
    _write_run_record(
        run_id=run_id,
        spd_run_dir=spd_run_dir,
        materialized_config_path=materialized_config_path,
        config=effective_config,
    )


if __name__ == "__main__":
    main()

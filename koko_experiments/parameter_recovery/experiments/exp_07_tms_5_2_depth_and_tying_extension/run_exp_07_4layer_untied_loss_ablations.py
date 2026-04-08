# ruff: noqa: E402

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koko_experiments.parameter_recovery.experiments.exp_07_tms_5_2_depth_and_tying_extension.run_exp_07_batch import (  # noqa: E501
    _find_existing_target_checkpoint,
    _materialize_spd_config,
    _materialize_target_config,
    _require_local_spd_out_dir,
    _run_spd_for_target,
    _save_run_outputs,
    _selected_run_names,
    _write_index,
)
from spd.base_config import BaseConfig
from spd.configs import Config
from spd.experiments.tms.configs import TMSTrainConfig


class Exp07LossAblationRunSpec(BaseConfig):
    run_name: str
    source_target_run_name: str
    target_config: TMSTrainConfig
    spd_config: Config


class Exp07LossAblationBatchConfig(BaseConfig):
    wandb_project: str
    runs: list[Exp07LossAblationRunSpec]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp_07 SPD experiments against reused target checkpoints."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "exp_07_4layer_untied_loss_ablations.yaml",
        help="Path to the reuse-target batch YAML.",
    )
    parser.add_argument(
        "--run-names",
        type=str,
        default=None,
        help="Optional comma-separated subset of run names to execute.",
    )
    parser.add_argument(
        "--spd-steps",
        type=int,
        default=None,
        help="Override the SPD optimization step count for all selected runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require_local_spd_out_dir()

    config_path = args.config_path.expanduser().resolve()
    assert config_path.exists(), f"Missing batch config at {config_path}"
    batch_config = Exp07LossAblationBatchConfig.from_file(config_path)
    selected_run_names = _selected_run_names(args.run_names)

    materialized_root = Path(__file__).resolve().parent / "materialized_configs" / config_path.stem
    materialized_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []

    print(f"wandb_project={batch_config.wandb_project}")

    for run_spec in batch_config.runs:
        if selected_run_names is not None and run_spec.run_name not in selected_run_names:
            continue

        print(f"=== Starting {run_spec.run_name} ===")
        target_config = _materialize_target_config(run_spec.target_config, batch_config.wandb_project)
        spd_config = _materialize_spd_config(
            run_spec.spd_config, batch_config.wandb_project, run_spec.run_name
        )
        if args.spd_steps is not None:
            spd_config = spd_config.model_copy(update={"steps": args.spd_steps})

        materialized_dir = materialized_root / run_spec.run_name
        materialized_dir.mkdir(parents=True, exist_ok=True)
        target_config_path = materialized_dir / "target_train_config.yaml"
        target_config.to_file(target_config_path)

        target_checkpoint_path = _find_existing_target_checkpoint(run_spec.source_target_run_name)
        target_run_dir = target_checkpoint_path.parent
        print(f"reused_target_checkpoint_path={target_checkpoint_path}")

        materialized_spd_config = spd_config.model_copy(
            update={"pretrained_model_path": str(target_checkpoint_path)}
        )
        spd_config_path = materialized_dir / "spd_config.yaml"
        materialized_spd_config.to_file(spd_config_path)

        spd_run_dir = _run_spd_for_target(target_checkpoint_path, materialized_spd_config)
        print(f"spd_run_dir={spd_run_dir}")

        analysis_path = _save_run_outputs(
            run_name=run_spec.run_name,
            source_batch_config=config_path,
            target_config_path=target_config_path,
            spd_config_path=spd_config_path,
            spd_run_dir=spd_run_dir,
        )
        summary_rows.append(
            {
                "run_name": run_spec.run_name,
                "source_target_run_name": run_spec.source_target_run_name,
                "target_run_dir": str(target_run_dir),
                "spd_run_dir": str(spd_run_dir),
                "analysis_path": str(analysis_path),
            }
        )
        _write_index(materialized_root / "latest_run_index.json", summary_rows)
        print(f"=== Finished {run_spec.run_name} ===")


if __name__ == "__main__":
    main()

# ruff: noqa: E402

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koko_experiments.parameter_recovery.summarize_tms_hyperparameter_grid_search import (
    summarize_grid_search_results,
)
from koko_experiments.parameter_recovery.tms_pretrained_analysis import analyze_tms_spd_run
from spd.configs import Config
from spd.experiments.tms.tms_decomposition import main as run_tms_decomposition
from spd.settings import SPD_OUT_DIR
from spd.utils.run_utils import apply_nested_updates, generate_run_id

GRID_SEARCH_WANDB_PROJECT = "tms-5-2-hyperparams-grid-search"


@dataclass(frozen=True)
class SweepCombination:
    n_mask_samples: int
    stochastic_recon_coeff: float
    stochastic_recon_layerwise_coeff: float
    importance_minimality_coeff: float
    importance_minimality_pnorm: float


def _timestamped_dir(base: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full TMS SPD hyperparameter grid search and summarize MMCS/ML2R."
    )
    parser.add_argument("experiment_dir", type=Path, help="Experiment directory containing sweep_plan.json")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Resume into an existing result directory instead of creating a new one.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force a new timestamped result directory instead of resuming the latest partial run.",
    )
    return parser.parse_args()


def _load_plan(experiment_dir: Path) -> tuple[str, Path, dict[str, list[float | int]]]:
    plan_path = experiment_dir / "sweep_plan.json"
    assert plan_path.exists(), f"Missing sweep plan at {plan_path}"
    plan = json.loads(plan_path.read_text())
    experiment_name = str(plan["experiment_name"])
    base_config = experiment_dir / str(plan["base_config"])
    assert base_config.exists(), f"Missing base config at {base_config}"

    hyperparameter_grid = plan["hyperparameter_grid"]
    required_keys = {
        "n_mask_samples",
        "stochastic_recon_coeff",
        "stochastic_recon_layerwise_coeff",
        "importance_minimality_coeff",
        "importance_minimality_pnorm",
    }
    assert set(hyperparameter_grid) == required_keys, (
        f"Expected grid keys {sorted(required_keys)}, got {sorted(hyperparameter_grid)}"
    )
    for key, values in hyperparameter_grid.items():
        assert isinstance(values, list), f"Expected list for {key}, got {type(values)}"
        assert values, f"Expected at least one value for {key}"
    return experiment_name, base_config, hyperparameter_grid


def _iter_combinations(grid: dict[str, list[float | int]]) -> list[SweepCombination]:
    return [
        SweepCombination(
            n_mask_samples=int(n_mask_samples),
            stochastic_recon_coeff=float(stochastic_recon_coeff),
            stochastic_recon_layerwise_coeff=float(stochastic_recon_layerwise_coeff),
            importance_minimality_coeff=float(importance_minimality_coeff),
            importance_minimality_pnorm=float(importance_minimality_pnorm),
        )
        for (
            n_mask_samples,
            stochastic_recon_coeff,
            stochastic_recon_layerwise_coeff,
            importance_minimality_coeff,
            importance_minimality_pnorm,
        ) in itertools.product(
            grid["n_mask_samples"],
            grid["stochastic_recon_coeff"],
            grid["stochastic_recon_layerwise_coeff"],
            grid["importance_minimality_coeff"],
            grid["importance_minimality_pnorm"],
        )
    ]


def _find_loss_coeff(config_data: dict[str, object], loss_name: str) -> float:
    loss_configs = config_data["loss_metric_configs"]
    assert isinstance(loss_configs, list), "loss_metric_configs must be a list"
    for loss_config in loss_configs:
        assert isinstance(loss_config, dict), "loss metric config entries must be dictionaries"
        if loss_config["classname"] == loss_name:
            return float(loss_config["coeff"])
    raise ValueError(f"Missing loss config for {loss_name}")


def _make_run_label(combo: SweepCombination) -> str:
    return (
        f"nmasks{combo.n_mask_samples:02d}"
        f"_sr{combo.stochastic_recon_coeff:.2e}"
        f"_srl{combo.stochastic_recon_layerwise_coeff:.2e}"
        f"_imp{combo.importance_minimality_coeff:.2e}"
        f"_p{combo.importance_minimality_pnorm:.2e}"
    )


def _make_wandb_project_name() -> str:
    return GRID_SEARCH_WANDB_PROJECT


def _make_wandb_run_name(base_config: Config, combo: SweepCombination) -> str:
    return f"exp05_{_make_run_label(combo)}"


def _build_run_config(base_config: Config, experiment_name: str, combo: SweepCombination) -> Config:
    updated_data = apply_nested_updates(
        base_config.model_dump(mode="json"),
        {
            "n_mask_samples": combo.n_mask_samples,
            "loss_metric_configs.StochasticReconLoss.coeff": combo.stochastic_recon_coeff,
            "loss_metric_configs.StochasticReconLayerwiseLoss.coeff": combo.stochastic_recon_layerwise_coeff,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": combo.importance_minimality_coeff,
            "loss_metric_configs.ImportanceMinimalityLoss.pnorm": combo.importance_minimality_pnorm,
            "wandb_project": _make_wandb_project_name(),
            "wandb_run_name": _make_wandb_run_name(base_config, combo),
        },
    )
    return Config.model_validate(updated_data)


def _resolve_result_root(
    experiment_name: str,
    requested_result_root: Path | None,
    fresh: bool,
) -> Path:
    results_base = SPD_OUT_DIR / "parameter_recovery" / "results"
    if requested_result_root is not None:
        result_root = requested_result_root.expanduser().resolve()
        assert result_root.exists(), f"Requested result root does not exist: {result_root}"
        return result_root
    if not fresh:
        existing_roots = sorted(results_base.glob(f"{experiment_name}_*"))
        if existing_roots:
            latest_root = existing_roots[-1]
            if not (latest_root / "top_5_hyperparameter_combinations.json").exists():
                print(f"Resuming existing partial result root: {latest_root}")
                return latest_root
    return _timestamped_dir(results_base, experiment_name)


def _prepare_result_root(result_root: Path, base_config_path: Path, experiment_dir: Path) -> None:
    (result_root / "configs").mkdir(parents=True, exist_ok=True)
    (result_root / "per_run_metrics").mkdir(parents=True, exist_ok=True)
    copied_base_config = result_root / base_config_path.name
    copied_base_config.write_text(base_config_path.read_text())
    copied_plan_path = result_root / "sweep_plan.json"
    copied_plan_path.write_text((experiment_dir / "sweep_plan.json").read_text())


def _build_existing_spd_run_index() -> dict[str, Path]:
    run_index: dict[str, Path] = {}
    final_config_paths = sorted(
        (SPD_OUT_DIR / "spd").glob("s-*/final_config.yaml"),
        key=lambda path: path.parent.stat().st_mtime,
        reverse=True,
    )
    for final_config_path in final_config_paths:
        config_data = yaml.safe_load(final_config_path.read_text())
        wandb_run_name = config_data.get("wandb_run_name")
        if wandb_run_name is None or wandb_run_name in run_index:
            continue
        run_index[str(wandb_run_name)] = final_config_path.parent
    return run_index


def _run_checkpoint_path(spd_run_dir: Path, steps: int) -> Path:
    return spd_run_dir / f"model_{steps}.pth"


def _run_finished(spd_run_dir: Path, steps: int) -> bool:
    return _run_checkpoint_path(spd_run_dir, steps).exists()


def _write_compact_result(
    experiment_name: str,
    run_index: int,
    run_label: str,
    run_config: Config,
    run_config_path: Path,
    spd_run_dir: Path,
    per_run_path: Path,
) -> None:
    analysis_path = spd_run_dir / "parameter_recovery_analysis.json"
    if analysis_path.exists():
        analysis_payload = json.loads(analysis_path.read_text())
    else:
        analysis_payload = analyze_tms_spd_run(spd_run_dir=spd_run_dir, run_name=run_label)
        analysis_path.write_text(json.dumps(analysis_payload, indent=2))

    compact_result = {
        "experiment_name": experiment_name,
        "run_label": run_label,
        "run_index": run_index,
        "run_id": spd_run_dir.name,
        "wandb_project": run_config.wandb_project,
        "wandb_run_name": run_config.wandb_run_name,
        "spd_run_dir": str(spd_run_dir),
        "config_path": str(run_config_path),
        "analysis_path": str(analysis_path),
        "hyperparameters": {
            "n_mask_samples": run_config.n_mask_samples,
            "stochastic_recon_coeff": _find_loss_coeff(
                run_config.model_dump(mode="json"), "StochasticReconLoss"
            ),
            "stochastic_recon_layerwise_coeff": _find_loss_coeff(
                run_config.model_dump(mode="json"), "StochasticReconLayerwiseLoss"
            ),
            "importance_minimality_coeff": _find_loss_coeff(
                run_config.model_dump(mode="json"), "ImportanceMinimalityLoss"
            ),
            "importance_minimality_pnorm": next(
                loss_config.pnorm
                for loss_config in run_config.loss_metric_configs
                if loss_config.classname == "ImportanceMinimalityLoss"
            ),
        },
        "metrics": _extract_compact_metrics(analysis_payload),
    }
    per_run_path.write_text(json.dumps(compact_result, indent=2, sort_keys=True))


def _completed_run_labels(per_run_dir: Path) -> set[str]:
    return {
        path.name.removesuffix("_sweep_metrics.json")
        for path in per_run_dir.glob("*_sweep_metrics.json")
    }


def _combo_status_message(
    run_index: int,
    n_combinations: int,
    experiment_name: str,
    run_label: str,
    action: str,
) -> None:
    print(f"[{run_index}/{n_combinations}] experiment={experiment_name} run_label={run_label} action={action}")


def _run_or_resume_combination(
    experiment_name: str,
    run_index: int,
    n_combinations: int,
    run_label: str,
    run_config: Config,
    run_config_path: Path,
    per_run_path: Path,
    existing_run_index: dict[str, Path],
) -> None:
    if per_run_path.exists():
        _combo_status_message(run_index, n_combinations, experiment_name, run_label, "skip_completed")
        return

    existing_spd_run_dir = existing_run_index.get(str(run_config.wandb_run_name))
    if existing_spd_run_dir is not None and _run_finished(existing_spd_run_dir, run_config.steps):
        _combo_status_message(
            run_index,
            n_combinations,
            experiment_name,
            run_label,
            f"recover_finished_run run_id={existing_spd_run_dir.name}",
        )
        _write_compact_result(
            experiment_name=experiment_name,
            run_index=run_index,
            run_label=run_label,
            run_config=run_config,
            run_config_path=run_config_path,
            spd_run_dir=existing_spd_run_dir,
            per_run_path=per_run_path,
        )
        return

    if existing_spd_run_dir is not None:
        _combo_status_message(
            run_index,
            n_combinations,
            experiment_name,
            run_label,
            f"rerun_interrupted_run previous_run_id={existing_spd_run_dir.name}",
        )
    else:
        _combo_status_message(run_index, n_combinations, experiment_name, run_label, "launch_new_run")

    run_id = generate_run_id("spd")
    print(f"run_id={run_id}")
    run_tms_decomposition(config_path=run_config_path, run_id=run_id)

    spd_run_dir = SPD_OUT_DIR / "spd" / run_id
    _write_compact_result(
        experiment_name=experiment_name,
        run_index=run_index,
        run_label=run_label,
        run_config=run_config,
        run_config_path=run_config_path,
        spd_run_dir=spd_run_dir,
        per_run_path=per_run_path,
    )


def _extract_compact_metrics(analysis_payload: dict[str, object]) -> dict[str, float]:
    summary = analysis_payload["summary"]
    assert isinstance(summary, dict), f"Expected summary dict, got {type(summary)}"
    recovery_metrics = summary["recovery_metrics"]
    assert isinstance(recovery_metrics, dict), (
        f"Expected recovery metrics dict, got {type(recovery_metrics)}"
    )
    return {
        "mmcs": float(recovery_metrics["mmcs"]),
        "ml2r": float(recovery_metrics["ml2r"]),
    }


def main() -> None:
    args = _parse_args()
    assert not (args.result_root is not None and args.fresh), (
        "Use either --result-root to resume a specific directory or --fresh to start a new one, not both."
    )
    experiment_dir = args.experiment_dir.expanduser().resolve()
    experiment_name, base_config_path, hyperparameter_grid = _load_plan(experiment_dir)
    base_config = Config.from_file(base_config_path)

    result_root = _resolve_result_root(experiment_name, args.result_root, args.fresh)
    _prepare_result_root(result_root, base_config_path, experiment_dir)
    config_dir = result_root / "configs"
    per_run_dir = result_root / "per_run_metrics"
    existing_run_index = _build_existing_spd_run_index()

    combinations = _iter_combinations(hyperparameter_grid)
    completed_labels = _completed_run_labels(per_run_dir)
    print(
        f"result_root={result_root} completed_runs={len(completed_labels)} total_runs={len(combinations)}"
    )
    for run_index, combo in enumerate(combinations, start=1):
        run_label = _make_run_label(combo)
        run_config = _build_run_config(base_config, experiment_name, combo)
        run_config_path = config_dir / f"{run_label}.yaml"
        run_config.to_file(run_config_path)
        per_run_path = per_run_dir / f"{run_label}_sweep_metrics.json"
        _run_or_resume_combination(
            experiment_name=experiment_name,
            run_index=run_index,
            n_combinations=len(combinations),
            run_label=run_label,
            run_config=run_config,
            run_config_path=run_config_path,
            per_run_path=per_run_path,
            existing_run_index=existing_run_index,
        )

    summary_path, top_path = summarize_grid_search_results(result_root=result_root, top_k=5)
    print(f"result_root={result_root}")
    print(f"summary_path={summary_path}")
    print(f"top_path={top_path}")


if __name__ == "__main__":
    main()

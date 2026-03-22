# ruff: noqa: E402

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


def _make_wandb_run_name(base_config: Config, combo: SweepCombination) -> str:
    config_data = base_config.model_dump(mode="json")
    faithfulness_coeff = _find_loss_coeff(config_data, "FaithfulnessLoss")
    first_module = base_config.module_info[0]
    feature_probability_pct = int(base_config.task_config.feature_probability * 100)
    return (
        f"exp05_nmasks{combo.n_mask_samples:02d}"
        f"_faith{faithfulness_coeff:.2e}"
        f"_stochrecon{combo.stochastic_recon_coeff:.2e}"
        f"_stochreconlayer{combo.stochastic_recon_layerwise_coeff:.2e}"
        f"_p{combo.importance_minimality_pnorm:.2e}"
        f"_impmin{combo.importance_minimality_coeff:.2e}"
        f"_C{first_module.C}"
        f"_sd{base_config.seed}"
        f"_lr{base_config.lr_schedule.start_val:.2e}"
        f"_bs{base_config.batch_size}"
        f"_ft{feature_probability_pct}"
        f"_steps{base_config.steps}"
    )


def _build_run_config(base_config: Config, combo: SweepCombination) -> Config:
    updated_data = apply_nested_updates(
        base_config.model_dump(mode="json"),
        {
            "n_mask_samples": combo.n_mask_samples,
            "loss_metric_configs.StochasticReconLoss.coeff": combo.stochastic_recon_coeff,
            "loss_metric_configs.StochasticReconLayerwiseLoss.coeff": combo.stochastic_recon_layerwise_coeff,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": combo.importance_minimality_coeff,
            "loss_metric_configs.ImportanceMinimalityLoss.pnorm": combo.importance_minimality_pnorm,
            "wandb_run_name": _make_wandb_run_name(base_config, combo),
        },
    )
    return Config.model_validate(updated_data)


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
    experiment_dir = args.experiment_dir.expanduser().resolve()
    experiment_name, base_config_path, hyperparameter_grid = _load_plan(experiment_dir)
    base_config = Config.from_file(base_config_path)

    result_root = _timestamped_dir(SPD_OUT_DIR / "parameter_recovery" / "results", experiment_name)
    config_dir = result_root / "configs"
    per_run_dir = result_root / "per_run_metrics"
    config_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir.mkdir(parents=True, exist_ok=True)

    copied_base_config = result_root / base_config_path.name
    copied_base_config.write_text(base_config_path.read_text())
    copied_plan_path = result_root / "sweep_plan.json"
    copied_plan_path.write_text((experiment_dir / "sweep_plan.json").read_text())

    combinations = _iter_combinations(hyperparameter_grid)
    for run_index, combo in enumerate(combinations, start=1):
        run_label = _make_run_label(combo)
        run_config = _build_run_config(base_config, combo)
        run_config_path = config_dir / f"{run_label}.yaml"
        run_config.to_file(run_config_path)

        run_id = generate_run_id("spd")
        print(
            f"[{run_index}/{len(combinations)}] experiment={experiment_name} run_label={run_label} run_id={run_id}"
        )
        run_tms_decomposition(config_path=run_config_path, run_id=run_id)

        spd_run_dir = SPD_OUT_DIR / "spd" / run_id
        analysis_payload = analyze_tms_spd_run(spd_run_dir=spd_run_dir, run_name=run_label)
        analysis_path = spd_run_dir / "parameter_recovery_analysis.json"
        analysis_path.write_text(json.dumps(analysis_payload, indent=2))

        compact_result = {
            "experiment_name": experiment_name,
            "run_label": run_label,
            "run_index": run_index,
            "run_id": run_id,
            "wandb_run_name": run_config.wandb_run_name,
            "spd_run_dir": str(spd_run_dir),
            "config_path": str(run_config_path),
            "analysis_path": str(analysis_path),
            "hyperparameters": {
                "n_mask_samples": combo.n_mask_samples,
                "stochastic_recon_coeff": combo.stochastic_recon_coeff,
                "stochastic_recon_layerwise_coeff": combo.stochastic_recon_layerwise_coeff,
                "importance_minimality_coeff": combo.importance_minimality_coeff,
                "importance_minimality_pnorm": combo.importance_minimality_pnorm,
            },
            "metrics": _extract_compact_metrics(analysis_payload),
        }
        per_run_path = per_run_dir / f"{run_label}_sweep_metrics.json"
        per_run_path.write_text(json.dumps(compact_result, indent=2, sort_keys=True))

    summary_path, top_path = summarize_grid_search_results(result_root=result_root, top_k=5)
    print(f"result_root={result_root}")
    print(f"summary_path={summary_path}")
    print(f"top_path={top_path}")


if __name__ == "__main__":
    main()

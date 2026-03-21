import argparse
import json
import statistics
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spd.configs import Config
from spd.experiments.tms.tms_decomposition import main as run_tms_decomposition
from spd.settings import SPD_OUT_DIR
from spd.utils.general_utils import replace_pydantic_model
from spd.utils.run_utils import generate_run_id

from koko_experiments.scripts.tms_alignment_report import build_alignment_report


@dataclass(frozen=True)
class PlannedRun:
    run_label: str
    seed: int


def _timestamped_dir(base: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated paper-style TMS SPD experiments.")
    parser.add_argument("experiment_dir", type=Path, help="Experiment directory containing sweep_plan.json")
    return parser.parse_args()


def _load_plan(experiment_dir: Path) -> tuple[str, Path, list[PlannedRun]]:
    plan_path = experiment_dir / "sweep_plan.json"
    assert plan_path.exists(), f"Missing sweep plan at {plan_path}"
    plan = json.loads(plan_path.read_text())
    experiment_name = plan["experiment_name"]
    base_config_name = plan["base_config"]
    planned_runs = [
        PlannedRun(run_label=run["run_label"], seed=int(run["seed"])) for run in plan["runs"]
    ]
    assert len(planned_runs) == 5, "This script expects exactly 5 planned runs"
    return experiment_name, experiment_dir / base_config_name, planned_runs


def _summarize_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    assert reports, "No reports to summarize"
    layer_names = sorted(reports[0]["layers"].keys())
    summary: dict[str, object] = {
        "experiment_name": reports[0]["experiment_name"],
        "n_runs": len(reports),
        "layers": {},
        "runs": [
            {
                "run_label": report["run_label"],
                "seed": report["seed"],
                "run_id": report["run_id"],
                "spd_run_dir": report["run_dir"],
                "analysis_path": report["analysis_path"],
            }
            for report in reports
        ],
    }
    for layer_name in layer_names:
        mmcs_values = [report["layers"][layer_name]["MMCS"] for report in reports]
        ml2r_values = [report["layers"][layer_name]["ML2R"] for report in reports]
        summary["layers"][layer_name] = {
            "MMCS": {
                "mean": statistics.mean(mmcs_values),
                "std": statistics.stdev(mmcs_values) if len(mmcs_values) > 1 else 0.0,
            },
            "ML2R": {
                "mean": statistics.mean(ml2r_values),
                "std": statistics.stdev(ml2r_values) if len(ml2r_values) > 1 else 0.0,
            },
        }
    return summary


def main() -> None:
    args = _parse_args()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    experiment_name, base_config_path, planned_runs = _load_plan(experiment_dir)
    assert base_config_path.exists(), f"Missing base config at {base_config_path}"

    result_root = _timestamped_dir(SPD_OUT_DIR / "parameter_recovery" / "results", experiment_name)
    config_dir = result_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    copied_base_config = result_root / base_config_path.name
    copied_base_config.write_text(base_config_path.read_text())

    reports: list[dict[str, object]] = []
    for run_index, planned_run in enumerate(planned_runs, start=1):
        config = replace_pydantic_model(Config.from_file(base_config_path), {"seed": planned_run.seed})
        run_config_path = config_dir / f"{planned_run.run_label}.yaml"
        config.to_file(run_config_path)

        run_id = generate_run_id("spd")
        print(
            f"[{run_index}/5] experiment={experiment_name} run_label={planned_run.run_label} seed={planned_run.seed} run_id={run_id}"
        )
        run_tms_decomposition(config_path=run_config_path, run_id=run_id)

        spd_run_dir = SPD_OUT_DIR / "spd" / run_id
        report = build_alignment_report(run_dir=spd_run_dir)
        report = deepcopy(report)
        report["experiment_name"] = experiment_name
        report["run_label"] = planned_run.run_label
        report["seed"] = planned_run.seed
        report["run_index"] = run_index
        report["run_id"] = run_id
        analysis_path = result_root / f"{planned_run.run_label}_parameter_recovery_analysis.json"
        analysis_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        report["analysis_path"] = str(analysis_path)
        reports.append(report)

    summary = _summarize_reports(reports)
    summary["result_root"] = str(result_root)
    summary["base_config_path"] = str(copied_base_config)
    summary_path = result_root / "mmcs_ml2r_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"result_root={result_root}")
    print(f"summary_path={summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

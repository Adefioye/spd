import argparse
import json
import statistics
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MMCS/ML2R across repeated TMS paper runs.")
    parser.add_argument("result_dirs", nargs="+", type=Path, help="Result directories created by run_tms_paper_replicates.py")
    return parser.parse_args()


def _summarize_result_dir(result_dir: Path) -> dict[str, object]:
    report_paths = sorted(result_dir.glob("*_parameter_recovery_analysis.json"))
    assert report_paths, f"No per-run analysis files found in {result_dir}"
    reports = [json.loads(path.read_text()) for path in report_paths]
    layer_names = sorted(reports[0]["layers"].keys())
    summary: dict[str, object] = {
        "experiment_name": reports[0]["experiment_name"],
        "result_dir": str(result_dir),
        "n_runs": len(reports),
        "layers": {},
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
    summaries = [_summarize_result_dir(path.expanduser().resolve()) for path in args.result_dirs]
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a TMS SPD hyperparameter grid search and save the top-ranked runs."
    )
    parser.add_argument("result_root", type=Path, help="Result directory produced by the sweep runner")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top hyperparameter combinations to save",
    )
    return parser.parse_args()


def _load_run_records(result_root: Path) -> list[dict[str, object]]:
    per_run_dir = result_root / "per_run_metrics"
    assert per_run_dir.exists(), f"Missing per-run metrics directory at {per_run_dir}"
    records: list[dict[str, object]] = []
    for path in sorted(per_run_dir.glob("*_sweep_metrics.json")):
        records.append(json.loads(path.read_text()))
    assert records, f"No per-run metric files found under {per_run_dir}"
    return records


def _rank_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for record in records:
        metrics = record["metrics"]
        assert isinstance(metrics, dict), f"Expected metrics dict, got {type(metrics)}"
        mmcs = float(metrics["mmcs"])
        ml2r = float(metrics["ml2r"])
        combined_score = 0.5 * (mmcs + ml2r)
        ranked_record = dict(record)
        ranked_record["combined_score"] = combined_score
        ranked.append(ranked_record)
    return sorted(
        ranked,
        key=lambda record: (
            float(record["combined_score"]),
            float(record["metrics"]["mmcs"]),
            float(record["metrics"]["ml2r"]),
            str(record["run_label"]),
        ),
        reverse=True,
    )


def summarize_grid_search_results(result_root: Path, top_k: int) -> tuple[Path, Path]:
    result_root = result_root.expanduser().resolve()
    records = _load_run_records(result_root)
    ranked = _rank_records(records)
    experiment_name = str(ranked[0]["experiment_name"])

    summary_payload = {
        "experiment_name": experiment_name,
        "result_root": str(result_root),
        "n_runs": len(ranked),
        "ranking_metric": "combined_score = 0.5 * (mmcs + ml2r)",
        "runs": ranked,
    }
    summary_path = result_root / "mmcs_ml2r_sweep_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

    top_payload = {
        "experiment_name": experiment_name,
        "result_root": str(result_root),
        "ranking_metric": "combined_score = 0.5 * (mmcs + ml2r)",
        "top_k": top_k,
        "runs": ranked[:top_k],
    }
    top_path = result_root / f"top_{top_k}_hyperparameter_combinations.json"
    top_path.write_text(json.dumps(top_payload, indent=2, sort_keys=True))
    return summary_path, top_path


def main() -> None:
    args = _parse_args()
    summary_path, top_path = summarize_grid_search_results(
        result_root=args.result_root,
        top_k=args.top_k,
    )
    print(f"summary_path={summary_path}")
    print(f"top_path={top_path}")


if __name__ == "__main__":
    main()

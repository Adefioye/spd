# ruff: noqa: E402

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koko_experiments.parameter_recovery.tms_pretrained_analysis import (
    analyze_tms_spd_run,
)
from spd.experiments.tms.tms_decomposition import main as run_tms_decomposition
from spd.settings import SPD_OUT_DIR
from spd.utils.run_utils import generate_run_id


def _analysis_output_path(base: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}_{stamp}_parameter_recovery_analysis.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single pretrained TMS SPD config and save a rich parameter-recovery analysis bundle."
    )
    parser.add_argument("config_path", type=Path, help="Path to an SPD YAML/JSON config")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Result bundle name. Defaults to the parent directory name of config_path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config_path.expanduser().resolve()
    assert config_path.exists(), f"Missing config at {config_path}"
    experiment_name = args.experiment_name or config_path.parent.name

    run_id = generate_run_id("spd")
    print(f"run_id={run_id}")
    run_tms_decomposition(config_path=config_path, run_id=run_id)

    spd_run_dir = SPD_OUT_DIR / "spd" / run_id
    assert spd_run_dir.exists(), f"Missing SPD run dir at {spd_run_dir}"
    analysis_payload = analyze_tms_spd_run(spd_run_dir=spd_run_dir, run_name=experiment_name)

    spd_out_path = spd_run_dir / "parameter_recovery_analysis.json"
    spd_out_path.write_text(json.dumps(analysis_payload, indent=2, default=str))

    analysis_path = _analysis_output_path(
        SPD_OUT_DIR / "parameter_recovery" / "results", experiment_name
    )
    analysis_path.write_text(json.dumps(analysis_payload, indent=2, default=str))

    print(f"spd_out_path={spd_out_path}")
    print(f"analysis_path={analysis_path}")


if __name__ == "__main__":
    main()

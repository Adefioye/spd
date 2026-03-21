from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fire
import json


def main(root_dir: str, output_path: str | None = None) -> None:
    root = Path(root_dir)
    rows = []
    for path in sorted(root.rglob("parameter_recovery_analysis.json")):
        rows.append(json.loads(path.read_text()))
    output = Path(output_path) if output_path is not None else root / "parameter_recovery_analysis.jsonl"
    output.write_text("\n".join(json.dumps(row) for row in rows))
    print(output)


if __name__ == "__main__":
    fire.Fire(main)

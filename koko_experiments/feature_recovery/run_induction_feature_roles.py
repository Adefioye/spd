from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fire
import json
import yaml

from spd.configs import Config
from spd.experiments.ih.configs import InductionHeadsTrainConfig
from spd.experiments.ih.ih_decomposition import main as run_ih_spd
from spd.experiments.ih.train_ih import run_train as train_ih_target
from spd.settings import SPD_OUT_DIR
from spd.utils.distributed_utils import get_device


def _latest_train_dir(before: set[Path]) -> Path:
    after = {p for p in (SPD_OUT_DIR / "train").glob("t-*")}
    new_dirs = sorted(after - before)
    assert new_dirs, "No new induction training directory found"
    return new_dirs[-1]


def main(train_config_path: str | None = None, spd_config_path: str | None = None) -> None:
    train_dir: Path | None = None
    if train_config_path is not None:
        train_raw = yaml.safe_load(Path(train_config_path).read_text())
        train_cfg = InductionHeadsTrainConfig(**train_raw)
        before = {p for p in (SPD_OUT_DIR / "train").glob("t-*")}
        train_ih_target(train_cfg, get_device())
        train_dir = _latest_train_dir(before)
        print(train_dir)

    if spd_config_path is None:
        return
    spd_cfg = Config.from_file(spd_config_path)
    spd_dict = spd_cfg.model_dump(mode="json")
    if train_dir is not None:
        spd_dict["pretrained_model_path"] = str(train_dir / "ih.pth")
    run_ih_spd(config_json=json.dumps(spd_dict))


if __name__ == "__main__":
    fire.Fire(main)

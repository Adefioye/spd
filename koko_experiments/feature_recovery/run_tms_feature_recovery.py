from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fire
import json
import yaml

from spd.configs import Config
from spd.experiments.tms.configs import TMSModelConfig, TMSTrainConfig
from spd.experiments.tms.tms_decomposition import main as run_tms_spd
from spd.experiments.tms.train_tms import run_train as train_tms_target
from spd.settings import SPD_OUT_DIR
from spd.utils.distributed_utils import get_device


def _latest_train_dir(before: set[Path]) -> Path:
    after = {p for p in (SPD_OUT_DIR / "train").glob("t-*")}
    new_dirs = sorted(after - before)
    assert new_dirs, "No new TMS training directory found"
    return new_dirs[-1]


def main(train_config_path: str, spd_config_path: str | None = None) -> None:
    train_raw = yaml.safe_load(Path(train_config_path).read_text())
    tms_cfg = TMSTrainConfig(
        wandb_project=None,
        tms_model_config=TMSModelConfig(
            n_features=train_raw["n_features"],
            n_hidden=train_raw["n_hidden"],
            n_hidden_layers=train_raw["n_hidden_layers"],
            tied_weights=train_raw["tied_weights"],
            init_bias_to_zero=train_raw["init_bias_to_zero"],
            device=get_device(),
        ),
        feature_probability=train_raw["feature_probability"],
        batch_size=train_raw["batch_size"],
        steps=train_raw["steps"],
        seed=train_raw["seed"],
        lr_schedule=train_raw["lr_schedule"],
        data_generation_type=train_raw["data_generation_type"],
    )
    before = {p for p in (SPD_OUT_DIR / "train").glob("t-*")}
    train_tms_target(tms_cfg, get_device())
    train_dir = _latest_train_dir(before)
    print(train_dir)
    if spd_config_path is None:
        return
    spd_cfg = Config.from_file(spd_config_path)
    spd_dict = spd_cfg.model_dump(mode="json")
    spd_dict["pretrained_model_path"] = str(train_dir / "tms.pth")
    run_tms_spd(config_json=json.dumps(spd_dict))


if __name__ == "__main__":
    fire.Fire(main)

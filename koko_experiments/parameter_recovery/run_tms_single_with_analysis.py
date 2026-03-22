import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from spd.configs import Config
from spd.experiments.tms.configs import TMSModelConfig
from spd.experiments.tms.models import TMSModel
from spd.experiments.tms.tms_decomposition import main as run_tms_decomposition
from spd.models.component_model import ComponentModel
from spd.settings import SPD_OUT_DIR
from spd.utils.module_utils import expand_module_patterns
from spd.utils.run_utils import generate_run_id

from koko_experiments.parameter_recovery.metrics import (
    load_spd_loss_summary,
    summarize_spd_evaluation,
)


def _timestamped_dir(base: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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


def _load_target_model_from_spd_run(spd_run_dir: Path) -> TMSModel:
    target_state = torch.load(spd_run_dir / "tms.pth", map_location="cpu", weights_only=True)
    linear1_weight = target_state["linear1.weight"]
    linear2_weight = target_state["linear2.weight"]
    linear2_bias = target_state["linear2.bias"]
    target_config = TMSModelConfig(
        n_features=linear1_weight.shape[1],
        n_hidden=linear1_weight.shape[0],
        n_hidden_layers=0,
        tied_weights=torch.allclose(linear2_weight, linear1_weight.T, atol=1e-5, rtol=1e-4),
        init_bias_to_zero=torch.allclose(
            linear2_bias, torch.zeros_like(linear2_bias), atol=1e-7, rtol=0
        ),
        device="cpu",
    )
    target_model = TMSModel(target_config)
    target_model.load_state_dict(target_state)
    target_model.eval()
    target_model.requires_grad_(False)
    return target_model


def _analyze_tms_spd_run(spd_run_dir: Path, run_name: str) -> dict[str, object]:
    final_config = Config.from_file(spd_run_dir / "final_config.yaml")
    target_model = _load_target_model_from_spd_run(spd_run_dir)
    module_path_info = expand_module_patterns(target_model, final_config.all_module_info)
    component_model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_config=final_config.ci_config,
        sigmoid_type=final_config.sigmoid_type,
        pretrained_model_output_attr=final_config.pretrained_model_output_attr,
    )
    checkpoint_path = sorted(spd_run_dir.glob("model*.pth"))[-1]
    component_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    component_model.load_state_dict(component_state)
    component_model.eval()

    spd_summary, layer_metrics = summarize_spd_evaluation(
        component_model=component_model,
        n_probe_features=target_model.config.n_features,
        input_magnitude=1.0,
        sampling="lower_leaky",
    )
    return {
        "run_name": run_name,
        "target_model_type": "tms",
        "spd_run_dir": str(spd_run_dir),
        "target_run_dir": str(final_config.pretrained_model_path),
        "summary": asdict(spd_summary),
        "losses": load_spd_loss_summary(spd_run_dir),
        "layers": [asdict(layer) for layer in layer_metrics],
    }


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
    analysis_payload = _analyze_tms_spd_run(spd_run_dir=spd_run_dir, run_name=experiment_name)

    spd_out_path = spd_run_dir / "parameter_recovery_analysis.json"
    spd_out_path.write_text(json.dumps(analysis_payload, indent=2, default=str))

    result_root = _timestamped_dir(SPD_OUT_DIR / "parameter_recovery" / "results", experiment_name)
    shutil.copy2(config_path, result_root / config_path.name)
    analysis_path = result_root / f"{experiment_name}_parameter_recovery_analysis.json"
    analysis_path.write_text(json.dumps(analysis_payload, indent=2, default=str))

    print(f"spd_out_path={spd_out_path}")
    print(f"result_root={result_root}")
    print(f"analysis_path={analysis_path}")


if __name__ == "__main__":
    main()

from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fire
import torch

from spd.configs import Config
from spd.experiments.resid_mlp.models import ResidMLP
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import expand_module_patterns

from koko_experiments.feature_recovery.metrics import analyze_component_model_directions
from koko_experiments.feature_recovery.results import load_feature_recovery_target_bundle


def _load_component_model(spd_run_dir: Path, target_run_dir: Path) -> ComponentModel:
    train_config, state_dict, feature_vectors, _ = load_feature_recovery_target_bundle(target_run_dir)
    final_config = Config.from_file(spd_run_dir / "final_config.yaml")
    target_model = ResidMLP(train_config.resid_mlp_model_config)
    target_model.load_state_dict(state_dict)
    target_model.eval()
    target_model.requires_grad_(False)
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
    return component_model, feature_vectors


def main(spd_run_dir: str, target_run_dir: str) -> None:
    spd_dir = Path(spd_run_dir)
    target_dir = Path(target_run_dir)
    component_model, feature_vectors = _load_component_model(spd_dir, target_dir)
    layer_metrics = analyze_component_model_directions(component_model, feature_vectors)
    output = {
        "spd_run_dir": str(spd_dir),
        "target_run_dir": str(target_dir),
        "layers": [asdict(metric) for metric in layer_metrics],
    }
    out_path = spd_dir / "feature_recovery_analysis.json"
    out_path.write_text(__import__("json").dumps(output, indent=2))
    print(out_path)


if __name__ == "__main__":
    fire.Fire(main)

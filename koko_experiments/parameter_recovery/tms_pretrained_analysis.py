from dataclasses import asdict
from pathlib import Path

import torch

from koko_experiments.parameter_recovery.metrics import (
    load_spd_loss_summary,
    summarize_spd_evaluation,
)
from spd.configs import Config
from spd.experiments.tms.models import TMSModel, TMSTargetRunInfo
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import expand_module_patterns


def load_tms_target_model_from_spd_run(spd_run_dir: Path) -> TMSModel:
    run_info = TMSTargetRunInfo.from_path(spd_run_dir / "tms.pth")
    target_model = TMSModel.from_run_info(run_info)
    target_model.eval()
    target_model.requires_grad_(False)
    return target_model


def analyze_tms_spd_run(spd_run_dir: Path, run_name: str) -> dict[str, object]:
    final_config = Config.from_file(spd_run_dir / "final_config.yaml")
    target_model = load_tms_target_model_from_spd_run(spd_run_dir)
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

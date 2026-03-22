from dataclasses import asdict
from pathlib import Path

import torch

from koko_experiments.parameter_recovery.metrics import (
    load_spd_loss_summary,
    summarize_spd_evaluation,
)
from spd.configs import Config
from spd.experiments.tms.configs import TMSModelConfig
from spd.experiments.tms.models import TMSModel
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import expand_module_patterns


def load_tms_target_model_from_spd_run(spd_run_dir: Path) -> TMSModel:
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

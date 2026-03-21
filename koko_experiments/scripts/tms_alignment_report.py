#!/usr/bin/env python3
"""Compute paper and extra alignment metrics for a local TMS SPD run.

Usage:
    uv run python koko_experiments/scripts/tms_alignment_report.py --run-dir "$SPD_OUT_DIR"/spd/s-xxxx
"""

import argparse
import json
import re
from pathlib import Path

import torch
from torch import Tensor

from spd.configs import Config
from spd.experiments.tms.configs import TMSModelConfig
from spd.experiments.tms.models import TMSModel
from spd.models.component_model import ComponentModel
from spd.settings import SPD_OUT_DIR
from spd.utils.module_utils import expand_module_patterns


def _step_from_checkpoint_name(path: Path) -> int:
    match = re.search(r"_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def _latest_component_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("model*.pth"), key=_step_from_checkpoint_name)
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints found in {run_dir}")
    return checkpoints[-1]


def _infer_tms_config_from_state_dict(state_dict: dict[str, Tensor]) -> TMSModelConfig:
    linear1_w = state_dict["linear1.weight"]
    linear2_w = state_dict["linear2.weight"]
    linear2_b = state_dict["linear2.bias"]

    n_hidden, n_features = linear1_w.shape

    hidden_layer_idxs = {
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("hidden_layers.") and k.endswith(".weight")
    }
    n_hidden_layers = max(hidden_layer_idxs) + 1 if hidden_layer_idxs else 0

    tied_weights = torch.allclose(linear2_w, linear1_w.T, atol=1e-5, rtol=1e-4)
    init_bias_to_zero = torch.allclose(linear2_b, torch.zeros_like(linear2_b), atol=1e-7, rtol=0)

    return TMSModelConfig(
        n_features=n_features,
        n_hidden=n_hidden,
        n_hidden_layers=n_hidden_layers,
        tied_weights=tied_weights,
        init_bias_to_zero=init_bias_to_zero,
        device="cpu",
    )


def _build_models(run_dir: Path, checkpoint_path: Path) -> tuple[TMSModel, ComponentModel]:
    final_config_path = run_dir / "final_config.yaml"
    target_checkpoint_path = run_dir / "tms.pth"

    if not final_config_path.exists():
        raise FileNotFoundError(f"Missing {final_config_path}")
    if not target_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {target_checkpoint_path}. This script expects a TMS SPD run directory."
        )

    run_config = Config.from_file(final_config_path)
    if run_config.task_config.task_name != "tms":
        raise ValueError(
            f"Expected a TMS run, got task '{run_config.task_config.task_name}' from {final_config_path}"
        )

    target_state = torch.load(target_checkpoint_path, map_location="cpu", weights_only=True)
    target_cfg = _infer_tms_config_from_state_dict(target_state)
    target_model = TMSModel(config=target_cfg)
    target_model.load_state_dict(target_state)
    target_model.eval()
    target_model.requires_grad_(False)

    module_path_info = expand_module_patterns(target_model, run_config.all_module_info)
    component_model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_config=run_config.ci_config,
        sigmoid_type=run_config.sigmoid_type,
        pretrained_model_output_attr=run_config.pretrained_model_output_attr,
    )

    component_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    component_model.load_state_dict(component_state)
    component_model.eval()

    return target_model, component_model


def _calc_layer_metrics(
    target_weight: Tensor, component_v: Tensor, component_u: Tensor, eps: float = 1e-12
) -> dict[str, float]:
    # target_weight shape: (d_out, d_in)
    # component_v shape: (d_in, C)
    # component_u shape: (C, d_out)
    # component_column_vectors shape: (C, d_in, d_out)
    component_column_vectors = torch.einsum("ic,cd->cid", component_v, component_u)
    target_columns = target_weight.T  # (d_in, d_out)

    component_norm = component_column_vectors.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_norm = target_columns.norm(dim=-1, keepdim=True).clamp_min(eps)

    component_unit = component_column_vectors / component_norm
    target_unit = target_columns / target_norm

    cosine_sim = torch.einsum("cid,id->ci", component_unit, target_unit)
    max_cos, max_idx = cosine_sim.max(dim=0)  # per target column

    matched_component_columns = component_column_vectors[max_idx, torch.arange(target_columns.shape[0])]
    l2_ratio = matched_component_columns.norm(dim=-1) / target_columns.norm(dim=-1).clamp_min(eps)

    matched_component_norms = matched_component_columns.norm(dim=-1)
    target_column_norms = target_columns.norm(dim=-1)

    component_strength = component_u.norm(dim=1) * component_v.norm(dim=0)
    threshold = 0.01 * component_strength.max().item() if component_strength.numel() > 0 else 0.0
    effective_components = int((component_strength > threshold).sum().item())

    reconstructed_weight = torch.einsum("ic,cd->di", component_v, component_u)
    faithfulness_mse = torch.mean((reconstructed_weight - target_weight) ** 2).item()

    return {
        "MMCS": max_cos.mean().item(),
        "ML2R": l2_ratio.mean().item(),
        "faithfulness_mse": faithfulness_mse,
        "matched_component_norm_mean": matched_component_norms.mean().item(),
        "target_column_norm_mean": target_column_norms.mean().item(),
        "effective_components@1pct_max_strength": float(effective_components),
    }


def _default_run_dir() -> Path:
    run_dirs = sorted((SPD_OUT_DIR / "spd").glob("s-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No SPD runs found under {(SPD_OUT_DIR / 'spd')}")
    return run_dirs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a local SPD run directory (defaults to latest under $SPD_OUT_DIR/spd/)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional specific model checkpoint path (defaults to latest model*.pth in run dir)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else _default_run_dir().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else _latest_component_checkpoint(run_dir)

    target_model, component_model = _build_models(run_dir, checkpoint)

    layer_names = ["linear1", "linear2"]
    output: dict[str, object] = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "target_model": {
            "n_features": target_model.config.n_features,
            "n_hidden": target_model.config.n_hidden,
            "n_hidden_layers": target_model.config.n_hidden_layers,
        },
        "layers": {},
    }

    for layer_name in layer_names:
        if layer_name not in component_model.components:
            continue
        component_layer = component_model.components[layer_name]
        target_layer = target_model.get_submodule(layer_name)
        target_weight = target_layer.weight.detach().cpu()
        component_v = component_layer.V.detach().cpu()
        component_u = component_layer.U.detach().cpu()
        layer_metrics = _calc_layer_metrics(
            target_weight=target_weight,
            component_v=component_v,
            component_u=component_u,
        )
        output["layers"][layer_name] = layer_metrics

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

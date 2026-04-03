import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from spd.plotting import get_single_feature_causal_importances


@dataclass
class RecoveryMetrics:
    mmcs: float
    ml2r: float


@dataclass
class LayerParameterMetrics:
    layer_name: str
    mmcs: float
    ml2r: float
    faithfulness_mse: float


@dataclass
class SPDEvaluation:
    recovery_metrics: RecoveryMetrics
    faithfulness_mse: float
    ci_summary: dict[str, dict[str, float]]


def _calc_layer_alignment_stats(
    target_weight: Tensor,
    component_v: Tensor,
    component_u: Tensor,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, float]:
    component_column_vectors = torch.einsum("ic,co->cio", component_v, component_u)
    # Transpose to (d_in, d_out) 
    target_columns = target_weight.T

    component_norm = component_column_vectors.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_norm = target_columns.norm(dim=-1, keepdim=True).clamp_min(eps)

    component_unit = component_column_vectors / component_norm
    target_unit = target_columns / target_norm

    cosine_sim = torch.einsum("cio,io->ci", component_unit, target_unit)
    max_cos, max_idx = cosine_sim.max(dim=0)

    matched_component_columns = component_column_vectors[
        max_idx, torch.arange(target_columns.shape[0], device=target_columns.device)
    ]
    l2_ratio = matched_component_columns.norm(dim=-1) / target_columns.norm(dim=-1).clamp_min(eps)

    reconstructed_weight = torch.einsum("ic,co->oi", component_v, component_u)
    faithfulness_mse = float(torch.mean((reconstructed_weight - target_weight) ** 2).item())
    return max_cos, l2_ratio, faithfulness_mse


def analyze_component_model_layers(component_model: torch.nn.Module) -> list[LayerParameterMetrics]:
    metrics: list[LayerParameterMetrics] = []
    for layer_name, components in component_model.components.items():
        if not (hasattr(components, "V") and hasattr(components, "U")):
            continue
        target_weight = component_model.target_weight(layer_name).detach()
        max_cos, l2_ratio, faithfulness_mse = _calc_layer_alignment_stats(
            target_weight=target_weight,
            component_v=components.V.detach(),
            component_u=components.U.detach(),
        )
        metrics.append(
            LayerParameterMetrics(
                layer_name=layer_name,
                mmcs=float(max_cos.mean().item()),
                ml2r=float(l2_ratio.mean().item()),
                faithfulness_mse=faithfulness_mse,
            )
        )
    return metrics


def singleton_feature_ci_summary(
    component_model: torch.nn.Module,
    n_features: int,
    input_magnitude: float,
    sampling: str,
) -> dict[str, dict[str, float]]:
    ci = get_single_feature_causal_importances(
        model=component_model,
        batch_shape=(n_features, n_features),
        input_magnitude=input_magnitude,
        sampling=sampling,
    )
    summary: dict[str, dict[str, float]] = {}
    for layer_name, values in ci.lower_leaky.items():
        best_mass = values.max(dim=-1).values
        probs = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(probs.clamp_min(1e-8).log() * probs).sum(dim=-1)
        summary[layer_name] = {
            "mean_top1_ci": float(best_mass.mean().item()),
            "mean_ci_entropy": float(entropy.mean().item()),
        }
    return summary


def summarize_spd_evaluation(
    component_model: torch.nn.Module,
    n_probe_features: int,
    input_magnitude: float,
    sampling: str,
) -> tuple[SPDEvaluation, list[LayerParameterMetrics]]:
    layer_metrics = analyze_component_model_layers(component_model)
    assert layer_metrics, "No SPD component layers with U/V factors were found"

    all_max_cos: list[Tensor] = []
    all_l2_ratio: list[Tensor] = []
    squared_error_sum = 0.0
    n_params = 0
    for layer_name, components in component_model.components.items():
        if not (hasattr(components, "V") and hasattr(components, "U")):
            continue
        target_weight = component_model.target_weight(layer_name).detach()
        max_cos, l2_ratio, _ = _calc_layer_alignment_stats(
            target_weight=target_weight,
            component_v=components.V.detach(),
            component_u=components.U.detach(),
        )
        reconstructed_weight = components.weight.detach()
        squared_error_sum += float(torch.sum((reconstructed_weight - target_weight) ** 2).item())
        n_params += target_weight.numel()
        all_max_cos.append(max_cos)
        all_l2_ratio.append(l2_ratio)

    total_max_cos = torch.cat(all_max_cos)
    total_l2_ratio = torch.cat(all_l2_ratio)
    ci_summary = singleton_feature_ci_summary(
        component_model=component_model,
        n_features=n_probe_features,
        input_magnitude=input_magnitude,
        sampling=sampling,
    )
    summary = SPDEvaluation(
        recovery_metrics=RecoveryMetrics(
            mmcs=float(total_max_cos.mean().item()),
            ml2r=float(total_l2_ratio.mean().item()),
        ),
        faithfulness_mse=squared_error_sum / n_params,
        ci_summary=ci_summary,
    )
    return summary, layer_metrics


def load_spd_loss_summary(spd_run_dir: Path) -> dict[str, float]:
    metrics_path = spd_run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}

    relevant_keys = {
        "train/loss/total",
        "train/loss/FaithfulnessLoss",
        "loss/FaithfulnessLoss",
        "loss/ImportanceMinimalityLoss",
        "loss/StochasticReconLoss",
        "loss/StochasticReconLayerwiseLoss",
        "loss/StochasticHiddenActsReconLoss",
    }
    latest: dict[str, float] = {}
    for line in metrics_path.read_text().splitlines():
        row = json.loads(line)
        for key in relevant_keys:
            if key in row:
                latest[key] = float(row[key])
    return latest

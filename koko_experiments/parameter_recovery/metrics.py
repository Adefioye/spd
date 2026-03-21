from dataclasses import asdict, dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from spd.plotting import get_single_feature_causal_importances


@dataclass
class RecoveryMetrics:
    mmcs: float
    ml2r: float
    coverage_at_095: float
    coverage_at_099: float


@dataclass
class SPDEvaluation:
    best_layer: str
    best_direction_role: str
    recovery_metrics: RecoveryMetrics
    ci_summary: dict[str, dict[str, float]]


@dataclass
class LayerDirectionMetrics:
    layer_name: str
    direction_role: str
    mmcs: float
    ml2r: float
    coverage_at_095: float
    coverage_at_099: float
    faithfulness_mse: float


def _normalize_rows(x: Tensor) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def compute_recovery_metrics(
    learned_vectors: Float[Tensor, "n_learned d"],
    true_vectors: Float[Tensor, "n_true d"],
) -> RecoveryMetrics:
    learned = _normalize_rows(learned_vectors)
    true = _normalize_rows(true_vectors)
    cosine = learned @ true.T

    best_cos, best_idx = cosine.abs().max(dim=0)
    learned_norms = learned_vectors.norm(dim=-1)
    true_norms = true_vectors.norm(dim=-1).clamp_min(1e-8)
    matched_norms = learned_norms[best_idx]
    l2_ratio = matched_norms / true_norms

    return RecoveryMetrics(
        mmcs=float(best_cos.mean().item()),
        ml2r=float(l2_ratio.mean().item()),
        coverage_at_095=float((best_cos >= 0.95).float().mean().item()),
        coverage_at_099=float((best_cos >= 0.99).float().mean().item()),
    )


def analyze_component_model_directions(
    component_model: torch.nn.Module,
    true_dictionary: Float[Tensor, "num_features hidden_dim"],
) -> list[LayerDirectionMetrics]:
    metrics: list[LayerDirectionMetrics] = []
    for layer_name, components in component_model.components.items():
        target_weight = component_model.target_weight(layer_name).detach()
        learned_weight = components.weight.detach()
        faithfulness_mse = float(torch.mean((learned_weight - target_weight) ** 2).item())

        if hasattr(components, "V") and components.V.shape[0] == true_dictionary.shape[1]:
            in_metrics = compute_recovery_metrics(
                learned_vectors=components.V.detach().T,
                true_vectors=true_dictionary,
            )
            metrics.append(
                LayerDirectionMetrics(
                    layer_name=layer_name,
                    direction_role="input",
                    faithfulness_mse=faithfulness_mse,
                    **asdict(in_metrics),
                )
            )
        if hasattr(components, "U") and components.U.shape[1] == true_dictionary.shape[1]:
            out_metrics = compute_recovery_metrics(
                learned_vectors=components.U.detach(),
                true_vectors=true_dictionary,
            )
            metrics.append(
                LayerDirectionMetrics(
                    layer_name=layer_name,
                    direction_role="output",
                    faithfulness_mse=faithfulness_mse,
                    **asdict(out_metrics),
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
    true_dictionary: Float[Tensor, "num_features hidden_dim"],
    input_magnitude: float,
    sampling: str,
) -> SPDEvaluation:
    layer_metrics = analyze_component_model_directions(component_model, true_dictionary)
    assert layer_metrics, "No SPD component directions matched the observed-space dimensionality"
    best_layer_metric = max(layer_metrics, key=lambda metric: metric.mmcs)
    ci_summary = singleton_feature_ci_summary(
        component_model=component_model,
        n_features=true_dictionary.shape[0],
        input_magnitude=input_magnitude,
        sampling=sampling,
    )
    return SPDEvaluation(
        best_layer=best_layer_metric.layer_name,
        best_direction_role=best_layer_metric.direction_role,
        recovery_metrics=RecoveryMetrics(
            mmcs=best_layer_metric.mmcs,
            ml2r=best_layer_metric.ml2r,
            coverage_at_095=best_layer_metric.coverage_at_095,
            coverage_at_099=best_layer_metric.coverage_at_099,
        ),
        ci_summary=ci_summary,
    )

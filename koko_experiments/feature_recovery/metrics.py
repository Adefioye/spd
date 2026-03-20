import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from sae_lens import SAE
from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    SyntheticDataEvalResult,
    eval_sae_on_synthetic_data,
)
from torch import Tensor

from spd.plotting import get_single_feature_causal_importances
from spd.utils.linear_sum_assignment import linear_sum_assignment
from spd.utils.run_utils import save_file


@dataclass
class DictionaryRecoveryMetrics:
    mean_matched_cosine: float
    coverage_at_095: float
    coverage_at_099: float
    fragmentation: float
    merging: float


@dataclass
class SAEEvaluation:
    explained_variance: float
    true_l0: float
    learned_l0: float
    dead_latents: int
    shrinkage: float
    mcc: float
    uniqueness: float
    dictionary_recovery: DictionaryRecoveryMetrics


@dataclass
class SPDEvaluation:
    best_layer: str
    best_direction_role: str
    dictionary_recovery: DictionaryRecoveryMetrics
    ci_summary: dict[str, dict[str, float]]


@dataclass
class LayerDirectionMetrics:
    layer_name: str
    direction_role: str
    mean_matched_cosine: float
    coverage_at_095: float
    coverage_at_099: float
    fragmentation: float
    merging: float


def _normalize_rows(x: Tensor) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def match_dictionary_vectors(
    learned_vectors: Float[Tensor, "n_learned d"],
    true_vectors: Float[Tensor, "n_true d"],
) -> tuple[Tensor, Tensor, Tensor]:
    learned = _normalize_rows(learned_vectors)
    true = _normalize_rows(true_vectors)
    cosine = learned @ true.T
    cost_matrix = -cosine.abs().detach().cpu().numpy()
    learned_idx, true_idx = linear_sum_assignment(cost_matrix)
    return cosine, torch.tensor(learned_idx), torch.tensor(true_idx)


def compute_dictionary_recovery_metrics(
    learned_vectors: Float[Tensor, "n_learned d"],
    true_vectors: Float[Tensor, "n_true d"],
) -> DictionaryRecoveryMetrics:
    cosine, learned_idx, true_idx = match_dictionary_vectors(learned_vectors, true_vectors)
    matched = cosine[learned_idx, true_idx].abs()
    best_per_true = cosine.abs().max(dim=0).values
    best_per_learned = cosine.abs().max(dim=1).values
    fragmentation = float((1.0 - best_per_true.mean()).item())
    merging = float((1.0 - best_per_learned.mean()).item())
    return DictionaryRecoveryMetrics(
        mean_matched_cosine=float(matched.mean().item()),
        coverage_at_095=float((best_per_true >= 0.95).float().mean().item()),
        coverage_at_099=float((best_per_true >= 0.99).float().mean().item()),
        fragmentation=fragmentation,
        merging=merging,
    )


def evaluate_sae_baseline(
    sae: SAE,
    feature_dict: FeatureDictionary,
    activations_generator: ActivationGenerator,
    eval_num_samples: int,
    batch_size: int,
) -> SAEEvaluation:
    eval_result: SyntheticDataEvalResult = eval_sae_on_synthetic_data(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_generator,
        num_samples=eval_num_samples,
        batch_size=batch_size,
    )
    d_recovery = compute_dictionary_recovery_metrics(
        learned_vectors=sae.W_dec.detach(),
        true_vectors=feature_dict.feature_vectors.detach(),
    )
    return SAEEvaluation(
        explained_variance=eval_result.explained_variance,
        true_l0=eval_result.true_l0,
        learned_l0=eval_result.sae_l0,
        dead_latents=eval_result.dead_latents,
        shrinkage=eval_result.shrinkage,
        mcc=eval_result.mcc,
        uniqueness=eval_result.uniqueness,
        dictionary_recovery=d_recovery,
    )


def analyze_component_model_directions(
    component_model: torch.nn.Module,
    true_dictionary: Float[Tensor, "num_features hidden_dim"],
) -> list[LayerDirectionMetrics]:
    metrics: list[LayerDirectionMetrics] = []
    for layer_name, components in component_model.components.items():
        if hasattr(components, "V") and components.V.shape[0] == true_dictionary.shape[1]:
            in_metrics = compute_dictionary_recovery_metrics(
                learned_vectors=components.V.detach().T,
                true_vectors=true_dictionary,
            )
            metrics.append(
                LayerDirectionMetrics(
                    layer_name=layer_name,
                    direction_role="input",
                    **asdict(in_metrics),
                )
            )
        if hasattr(components, "U") and components.U.shape[1] == true_dictionary.shape[1]:
            out_metrics = compute_dictionary_recovery_metrics(
                learned_vectors=components.U.detach(),
                true_vectors=true_dictionary,
            )
            metrics.append(
                LayerDirectionMetrics(
                    layer_name=layer_name,
                    direction_role="output",
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
        summary[layer_name] = {
            "mean_top1_ci": float(best_mass.mean().item()),
            "mean_ci_entropy": float(
                (-(values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8).log()
                 * (values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8))).sum(dim=-1).mean().item()
            ),
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
    best_layer_metric = max(layer_metrics, key=lambda metric: metric.mean_matched_cosine)
    ci_summary = singleton_feature_ci_summary(
        component_model=component_model,
        n_features=true_dictionary.shape[0],
        input_magnitude=input_magnitude,
        sampling=sampling,
    )
    return SPDEvaluation(
        best_layer=best_layer_metric.layer_name,
        best_direction_role=best_layer_metric.direction_role,
        dictionary_recovery=DictionaryRecoveryMetrics(
            mean_matched_cosine=best_layer_metric.mean_matched_cosine,
            coverage_at_095=best_layer_metric.coverage_at_095,
            coverage_at_099=best_layer_metric.coverage_at_099,
            fragmentation=best_layer_metric.fragmentation,
            merging=best_layer_metric.merging,
        ),
        ci_summary=ci_summary,
    )


def save_metrics_json(data: dict, path: Path) -> None:
    save_file(data, path, indent=2)

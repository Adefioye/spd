import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from jaxtyping import Float
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
    reconstruction_mse: float
    true_l0: float
    learned_l0: float
    dead_latents: int
    shrinkage: float
    dictionary_recovery: DictionaryRecoveryMetrics


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


def evaluate_sae(
    sae: torch.nn.Module,
    hidden: Float[Tensor, "batch hidden_dim"],
    true_latents: Float[Tensor, "batch num_features"],
    true_dictionary: Float[Tensor, "num_features hidden_dim"],
) -> SAEEvaluation:
    with torch.no_grad():
        recon, codes = sae(hidden)
    d_recovery = compute_dictionary_recovery_metrics(
        learned_vectors=sae.decoder.weight.detach().T,
        true_vectors=true_dictionary,
    )
    shrinkage = float((recon.norm(dim=-1) / hidden.norm(dim=-1).clamp_min(1e-8)).mean().item())
    return SAEEvaluation(
        reconstruction_mse=float(torch.mean((recon - hidden) ** 2).item()),
        true_l0=float((true_latents > 0).float().sum(dim=-1).mean().item()),
        learned_l0=float((codes > 0).float().sum(dim=-1).mean().item()),
        dead_latents=int(((codes > 0).float().sum(dim=0) == 0).sum().item()),
        shrinkage=shrinkage,
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


def save_metrics_json(data: dict, path: Path) -> None:
    save_file(data, path, indent=2)

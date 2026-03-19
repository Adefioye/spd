from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from .configs import ActivationGeneratorConfig, FeatureDictionaryConfig


@dataclass
class FeatureDictionary:
    feature_vectors: Tensor

    @property
    def num_features(self) -> int:
        return self.feature_vectors.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.feature_vectors.shape[1]

    @classmethod
    def from_config(cls, config: FeatureDictionaryConfig) -> "FeatureDictionary":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        if config.initializer == "orthogonal" and config.num_features <= config.hidden_dim:
            base = torch.randn(config.hidden_dim, config.hidden_dim, generator=generator)
            q, _ = torch.linalg.qr(base)
            feature_vectors = q[:, : config.num_features].T.contiguous()
        else:
            feature_vectors = torch.randn(
                config.num_features,
                config.hidden_dim,
                generator=generator,
            )
        if config.normalize_vectors:
            feature_vectors = feature_vectors / feature_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return cls(feature_vectors=feature_vectors)

    def encode(self, latents: Tensor) -> Tensor:
        return latents @ self.feature_vectors


def zipfian_firing_probabilities(
    num_features: int,
    exponent: float,
    max_prob: float,
    min_prob: float,
) -> Tensor:
    ranks = torch.arange(1, num_features + 1, dtype=torch.float32)
    weights = 1.0 / ranks.pow(exponent)
    weights = weights / weights.max()
    return min_prob + (max_prob - min_prob) * weights


def generate_random_correlation_matrix(
    num_features: int,
    strength: float,
    rank: int,
    seed: int,
) -> Tensor:
    if strength == 0.0:
        return torch.eye(num_features)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    factors = torch.randn(num_features, rank, generator=generator)
    cov = factors @ factors.T
    cov = cov / cov.diag().sqrt().unsqueeze(0)
    cov = cov / cov.diag().sqrt().unsqueeze(1)
    eye = torch.eye(num_features)
    mixed = (1 - strength) * eye + strength * cov
    mixed = (mixed + mixed.T) / 2
    mixed = mixed + 1e-5 * eye
    return mixed


class ActivationGenerator:
    def __init__(self, config: ActivationGeneratorConfig, num_features: int):
        self.config = config
        self.num_features = num_features
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(config.seed)
        self.parent_to_children = defaultdict(list)
        for parent, child in config.hierarchy_edges:
            self.parent_to_children[parent].append(child)

        self.firing_probabilities = self._build_probabilities()
        self.mean_firing_magnitudes = self._as_feature_tensor(config.mean_firing_magnitudes)
        self.std_firing_magnitudes = self._as_feature_tensor(config.std_firing_magnitudes)
        self.correlation_matrix = generate_random_correlation_matrix(
            num_features=num_features,
            strength=config.correlation_strength,
            rank=config.correlation_rank,
            seed=config.seed,
        )
        self.normal = torch.distributions.Normal(0, 1)

    def _build_probabilities(self) -> Tensor:
        cfg = self.config
        if cfg.firing_probabilities is not None:
            probs = torch.tensor(cfg.firing_probabilities, dtype=torch.float32)
            assert probs.shape == (self.num_features,)
            return probs
        if cfg.zipf_exponent is not None:
            assert cfg.max_prob is not None and cfg.min_prob is not None
            return zipfian_firing_probabilities(
                num_features=self.num_features,
                exponent=cfg.zipf_exponent,
                max_prob=cfg.max_prob,
                min_prob=cfg.min_prob,
            )
        return torch.full((self.num_features,), cfg.default_probability, dtype=torch.float32)

    def _as_feature_tensor(self, value: float | list[float]) -> Tensor:
        if isinstance(value, list):
            tensor = torch.tensor(value, dtype=torch.float32)
            assert tensor.shape == (self.num_features,)
            return tensor
        return torch.full((self.num_features,), float(value), dtype=torch.float32)

    def sample(self, batch_size: int) -> Tensor:
        mask = self._sample_mask(batch_size)
        magnitudes = torch.randn(batch_size, self.num_features, generator=self.generator)
        magnitudes = magnitudes * self.std_firing_magnitudes + self.mean_firing_magnitudes
        magnitudes = magnitudes.clamp_min(0.0)
        activations = magnitudes * mask
        activations = self._apply_hierarchy(activations)
        activations = self._apply_mutual_exclusion(activations)
        return activations

    def _sample_mask(self, batch_size: int) -> Tensor:
        if self.config.correlation_strength == 0.0:
            uniforms = torch.rand(batch_size, self.num_features, generator=self.generator)
        else:
            mvn = torch.distributions.MultivariateNormal(
                torch.zeros(self.num_features),
                covariance_matrix=self.correlation_matrix,
            )
            gaussian = mvn.sample((batch_size,))
            uniforms = self.normal.cdf(gaussian)
        return (uniforms < self.firing_probabilities).float()

    def _apply_hierarchy(self, activations: Tensor) -> Tensor:
        if not self.parent_to_children:
            return activations
        out = activations.clone()
        frontier = list(self.parent_to_children.keys())
        while frontier:
            parent = frontier.pop()
            children = self.parent_to_children[parent]
            child_mask = out[:, parent] > 0
            for child in children:
                out[~child_mask, child] = 0.0
                if child in self.parent_to_children:
                    frontier.append(child)
        return out

    def _apply_mutual_exclusion(self, activations: Tensor) -> Tensor:
        if not self.config.mutually_exclusive_groups:
            return activations
        out = activations.clone()
        for group in self.config.mutually_exclusive_groups:
            group_vals = out[:, group]
            active_counts = (group_vals > 0).sum(dim=-1)
            if not (active_counts > 1).any():
                continue
            winners = group_vals.argmax(dim=-1)
            out[:, group] = 0.0
            for batch_idx, winner_idx in enumerate(winners.tolist()):
                winner_feature = group[winner_idx]
                out[batch_idx, winner_feature] = group_vals[batch_idx, winner_idx]
        return out

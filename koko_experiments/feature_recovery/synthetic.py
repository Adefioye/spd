from collections import defaultdict

import torch
from sae_lens.synthetic import (
    ActivationGenerator as SAEActivationGenerator,
    FeatureDictionary as SAEFeatureDictionary,
    HierarchyNode,
    generate_random_correlation_matrix,
    hierarchy_modifier,
    zipfian_firing_probabilities,
)
from torch import Tensor

from .configs import ActivationGeneratorConfig, FeatureDictionaryConfig


def _identity_initializer() -> None:
    return None


def _build_feature_vectors(
    config: FeatureDictionaryConfig,
    device: str,
) -> Tensor:
    if config.initializer == "identity":
        assert config.num_features == config.hidden_dim, (
            "Identity dictionaries require num_features == hidden_dim"
        )
        return torch.eye(config.hidden_dim, device=device)

    feature_dict = SAEFeatureDictionary(
        num_features=config.num_features,
        hidden_dim=config.hidden_dim,
        bias=False,
        initializer=None if config.initializer == "random" else None,
        device=device,
        seed=config.seed,
    )
    if config.initializer == "orthogonal":
        orthogonal = SAEFeatureDictionary(
            num_features=config.num_features,
            hidden_dim=config.hidden_dim,
            bias=False,
            device=device,
            seed=config.seed,
        )
        return orthogonal.feature_vectors.detach().clone()
    return feature_dict.feature_vectors.detach().clone()


def _match_mutually_exclusive_groups(
    parent_to_children: dict[int, list[int]],
    mutually_exclusive_groups: list[list[int]],
) -> dict[int, bool]:
    mutually_exclusive_lookup: dict[int, bool] = defaultdict(bool)
    remaining_groups = [set(group) for group in mutually_exclusive_groups]
    for parent, children in parent_to_children.items():
        child_set = set(children)
        for group in list(remaining_groups):
            if group == child_set:
                mutually_exclusive_lookup[parent] = True
                remaining_groups.remove(group)
    assert not remaining_groups, (
        "Each mutually exclusive group must exactly match one parent's direct children"
    )
    return mutually_exclusive_lookup


def _build_hierarchy_roots(
    hierarchy_edges: list[tuple[int, int]],
    mutually_exclusive_groups: list[list[int]],
) -> list[HierarchyNode] | None:
    if not hierarchy_edges and not mutually_exclusive_groups:
        return None

    parent_to_children: dict[int, list[int]] = defaultdict(list)
    child_to_parent: dict[int, int] = {}

    for parent, child in hierarchy_edges:
        assert child not in child_to_parent, f"Feature {child} cannot have multiple parents"
        child_to_parent[child] = parent
        parent_to_children[parent].append(child)

    if mutually_exclusive_groups and not hierarchy_edges:
        return [
            HierarchyNode(
                feature_index=None,
                children=[HierarchyNode(feature_index=child) for child in group],
                mutually_exclusive_children=True,
            )
            for group in mutually_exclusive_groups
        ]

    exclusivity = _match_mutually_exclusive_groups(
        parent_to_children=parent_to_children,
        mutually_exclusive_groups=mutually_exclusive_groups,
    )

    def build_node(feature_index: int) -> HierarchyNode:
        children = [build_node(child) for child in parent_to_children.get(feature_index, [])]
        return HierarchyNode(
            feature_index=feature_index,
            children=children,
            mutually_exclusive_children=exclusivity.get(feature_index, False),
        )

    roots = [parent for parent in parent_to_children if parent not in child_to_parent]
    return [build_node(root) for root in roots]


class FeatureDictionary(SAEFeatureDictionary):
    @classmethod
    def from_config(
        cls,
        config: FeatureDictionaryConfig,
        device: str = "cpu",
    ) -> "FeatureDictionary":
        feature_vectors = _build_feature_vectors(config, device)
        feature_dict = cls(
            num_features=config.num_features,
            hidden_dim=config.hidden_dim,
            bias=False,
            initializer=None,
            device=device,
            seed=config.seed,
        )
        feature_dict.feature_vectors.data.copy_(feature_vectors)
        feature_dict.bias.data.zero_()
        return feature_dict

    def encode(self, latents: Tensor) -> Tensor:
        return self.forward(latents)


class ActivationGenerator(SAEActivationGenerator):
    def __init__(
        self,
        config: ActivationGeneratorConfig,
        num_features: int,
        device: str = "cpu",
    ):
        if config.firing_probabilities is not None:
            firing_probabilities: Tensor | float = torch.tensor(
                config.firing_probabilities,
                dtype=torch.float32,
                device=device,
            )
            assert firing_probabilities.shape == (num_features,)
        elif config.zipf_exponent is not None:
            assert config.max_prob is not None and config.min_prob is not None
            firing_probabilities = zipfian_firing_probabilities(
                num_features=num_features,
                exponent=config.zipf_exponent,
                max_prob=config.max_prob,
                min_prob=config.min_prob,
            ).to(device)
        else:
            firing_probabilities = config.default_probability

        correlation_matrix = None
        if config.use_random_correlation_matrix:
            correlation_matrix = generate_random_correlation_matrix(
                num_features=num_features,
                positive_ratio=config.correlation_positive_ratio,
                uncorrelated_ratio=config.correlation_uncorrelated_ratio,
                min_correlation_strength=config.min_correlation_strength,
                max_correlation_strength=config.max_correlation_strength,
                seed=config.seed,
                device=device,
            )

        hierarchy = _build_hierarchy_roots(
            hierarchy_edges=config.hierarchy_edges,
            mutually_exclusive_groups=config.mutually_exclusive_groups,
        )
        modify_activations = hierarchy_modifier(hierarchy) if hierarchy is not None else None

        super().__init__(
            num_features=num_features,
            firing_probabilities=firing_probabilities,
            std_firing_magnitudes=config.std_firing_magnitudes,
            mean_firing_magnitudes=config.mean_firing_magnitudes,
            modify_activations=modify_activations,
            correlation_matrix=correlation_matrix,
            device=device,
        )


def sample_observed_activations(
    feature_dict: FeatureDictionary,
    activation_generator: ActivationGenerator,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    latents = activation_generator.sample(batch_size)
    observed = feature_dict(latents)
    return observed, latents

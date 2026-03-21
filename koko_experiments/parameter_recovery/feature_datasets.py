from typing import Literal, override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset

from .synthetic import ActivationGenerator, FeatureDictionary, sample_observed_activations


class HiddenActivationDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch hidden_dim"],
            Float[Tensor, "batch num_features"],
        ]
    ]
):
    def __init__(
        self,
        feature_dict: FeatureDictionary,
        activation_generator: ActivationGenerator,
        device: str = "cpu",
    ):
        self.feature_dict = feature_dict
        self.activation_generator = activation_generator
        self.device = device

    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch hidden_dim"], Float[Tensor, "batch num_features"]]:
        hidden, latents = sample_observed_activations(
            feature_dict=self.feature_dict,
            activation_generator=self.activation_generator,
            batch_size=batch_size,
        )
        return hidden.to(self.device), latents.to(self.device)


def make_observed_labels(
    observed: Float[Tensor, "batch observed_dim"],
    label_type: Literal["identity", "act_plus_resid", "abs"],
    label_coeffs: Tensor,
    act_fn_name: Literal["relu", "gelu"],
) -> Float[Tensor, "batch observed_dim"]:
    weighted = observed * label_coeffs
    if label_type == "identity":
        return observed
    if label_type == "abs":
        return weighted.abs()
    act_fn = F.relu if act_fn_name == "relu" else F.gelu
    return act_fn(weighted) + observed


class ObservedActivationDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch observed_dim"],
            Float[Tensor, "batch observed_dim"],
        ]
    ]
):
    def __init__(
        self,
        feature_dict: FeatureDictionary,
        activation_generator: ActivationGenerator,
        label_type: Literal["identity", "act_plus_resid", "abs"],
        label_coeffs: Tensor,
        act_fn_name: Literal["relu", "gelu"],
        device: str = "cpu",
    ):
        self.feature_dict = feature_dict
        self.activation_generator = activation_generator
        self.label_type = label_type
        self.label_coeffs = label_coeffs
        self.act_fn_name = act_fn_name
        self.device = device

    def __len__(self) -> int:
        return 2**31

    def sample_ground_truth(
        self,
        batch_size: int,
    ) -> tuple[
        Float[Tensor, "batch observed_dim"],
        Float[Tensor, "batch num_features"],
    ]:
        observed, latents = sample_observed_activations(
            feature_dict=self.feature_dict,
            activation_generator=self.activation_generator,
            batch_size=batch_size,
        )
        return observed.to(self.device), latents.to(self.device)

    def generate_batch(
        self,
        batch_size: int,
    ) -> tuple[
        Float[Tensor, "batch observed_dim"],
        Float[Tensor, "batch observed_dim"],
    ]:
        observed, _ = self.sample_ground_truth(batch_size)
        labels = make_observed_labels(
            observed=observed,
            label_type=self.label_type,
            label_coeffs=self.label_coeffs.to(observed.device),
            act_fn_name=self.act_fn_name,
        )
        return observed, labels.to(self.device)


def materialize_observed_dataset(
    dataset: ObservedActivationDataset,
    num_samples: int,
    chunk_size: int,
) -> TensorDataset:
    observed_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []
    remaining = num_samples
    while remaining > 0:
        current_chunk = min(chunk_size, remaining)
        observed, labels = dataset.generate_batch(current_chunk)
        observed_chunks.append(observed.cpu())
        label_chunks.append(labels.cpu())
        remaining -= current_chunk
    return TensorDataset(torch.cat(observed_chunks, dim=0), torch.cat(label_chunks, dim=0))


class DictionaryFeatureDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch num_features"],
            Float[Tensor, "batch num_features"],
        ]
    ]
):
    def __init__(
        self,
        feature_dict: FeatureDictionary,
        activation_generator: ActivationGenerator,
        label_type: Literal["act_plus_resid", "abs", "identity"],
        label_coeffs: Tensor,
        act_fn_name: Literal["relu", "gelu"],
        device: str = "cpu",
    ):
        self.feature_dict = feature_dict
        self.activation_generator = activation_generator
        self.label_type = label_type
        self.label_coeffs = label_coeffs
        self.act_fn_name = act_fn_name
        self.device = device

    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch num_features"], Float[Tensor, "batch num_features"]]:
        latents = self.activation_generator.sample(batch_size)
        return latents.to(self.device), self.make_labels(latents).to(self.device)

    def make_labels(self, latents: Float[Tensor, "batch num_features"]) -> Float[Tensor, "batch num_features"]:
        weighted = latents * self.label_coeffs
        if self.label_type == "identity":
            return latents
        if self.label_type == "abs":
            return weighted.abs()
        act_fn = F.relu if self.act_fn_name == "relu" else F.gelu
        return act_fn(weighted) + latents

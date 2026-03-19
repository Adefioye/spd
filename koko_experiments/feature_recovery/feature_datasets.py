from typing import Literal, override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset

from .synthetic import ActivationGenerator, FeatureDictionary


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
        latents = self.activation_generator.sample(batch_size)
        hidden = self.feature_dict.encode(latents)
        return hidden.to(self.device), latents.to(self.device)


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

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from spd.base_config import BaseConfig
from spd.experiments.resid_mlp.configs import ResidMLPModelConfig
from spd.experiments.tms.configs import TMSModelConfig
from spd.configs import ScheduleConfig


class FeatureDictionaryConfig(BaseConfig):
    num_features: PositiveInt
    hidden_dim: PositiveInt
    seed: int = 0
    initializer: Literal["identity", "orthogonal", "random"] = "orthogonal"

    @model_validator(mode="after")
    def validate_dictionary(self) -> Self:
        if self.initializer == "identity":
            assert self.num_features == self.hidden_dim, (
                "Identity dictionaries require num_features == hidden_dim"
            )
        return self


class ActivationGeneratorConfig(BaseConfig):
    default_probability: PositiveFloat = 0.1
    firing_probabilities: list[float] | None = None
    mean_firing_magnitudes: float | list[float] = 1.0
    std_firing_magnitudes: float | list[float] = 0.05
    zipf_exponent: float | None = None
    max_prob: float | None = None
    min_prob: float | None = None
    hierarchy_edges: list[tuple[int, int]] = Field(default_factory=list)
    mutually_exclusive_groups: list[list[int]] = Field(default_factory=list)
    use_random_correlation_matrix: bool = False
    correlation_positive_ratio: float = 0.5
    correlation_uncorrelated_ratio: float = 0.3
    min_correlation_strength: float = 0.1
    max_correlation_strength: float = 0.8
    seed: int = 0

    @model_validator(mode="after")
    def validate_probabilities(self) -> "ActivationGeneratorConfig":
        if self.firing_probabilities is not None:
            assert self.zipf_exponent is None, (
                "Use either firing_probabilities or zipf_exponent, not both"
            )
        if self.zipf_exponent is not None:
            assert self.max_prob is not None and self.min_prob is not None, (
                "Zipfian probabilities require max_prob and min_prob"
            )
        if self.use_random_correlation_matrix:
            assert 0.0 <= self.correlation_positive_ratio <= 1.0
            assert 0.0 <= self.correlation_uncorrelated_ratio <= 1.0
            assert 0.0 <= self.min_correlation_strength <= self.max_correlation_strength <= 1.0
        return self


class SyntheticFamilyConfig(BaseConfig):
    stage: Literal["axis_aligned", "hidden_activations"]
    dictionary: FeatureDictionaryConfig
    generator: ActivationGeneratorConfig

    @model_validator(mode="after")
    def validate_stage(self) -> Self:
        if self.stage == "axis_aligned":
            assert self.dictionary.initializer == "identity", (
                "Axis-aligned stage requires an identity feature dictionary"
            )
        if self.stage == "hidden_activations":
            assert self.dictionary.initializer != "identity", (
                "Hidden-activation stage requires a non-identity dictionary"
            )
        return self


class TargetTrainingConfig(BaseConfig):
    model_type: Literal["tms", "resid_mlp"]
    tms_model_config: TMSModelConfig | None = None
    resid_mlp_model_config: ResidMLPModelConfig | None = None
    batch_size: PositiveInt = 1024
    steps: PositiveInt = 5_000
    print_freq: PositiveInt = 100
    lr_schedule: ScheduleConfig
    label_type: Literal["identity", "act_plus_resid", "abs"] = "identity"
    loss_type: Literal["readoff", "resid"] = "readoff"
    fixed_embedding: Literal["learned", "identity", "random"] = "learned"
    label_coeffs: list[float] | None = None

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.model_type == "tms":
            assert self.tms_model_config is not None, "tms_model_config must be provided"
            assert self.resid_mlp_model_config is None, (
                "resid_mlp_model_config must be omitted for model_type='tms'"
            )
            assert self.label_type == "identity", "TMS targets only support identity labels"
            assert self.loss_type == "readoff", "TMS targets only support readoff loss"
        else:
            assert self.resid_mlp_model_config is not None, (
                "resid_mlp_model_config must be provided"
            )
            assert self.tms_model_config is None, "tms_model_config must be omitted"
            if self.fixed_embedding == "identity":
                assert (
                    self.resid_mlp_model_config.n_features
                    == self.resid_mlp_model_config.d_embed
                ), "Identity embeddings require n_features == d_embed"
            if self.loss_type == "resid":
                assert self.label_type != "identity", (
                    "Residual-space loss requires non-identity observed-space labels"
                )
        return self


class SPDExecutionConfig(BaseConfig):
    spd_config_path: Path


class AnalysisConfig(BaseConfig):
    enabled: bool = True
    input_magnitude: float = 1.0
    sampling: Literal["lower_leaky", "exactly_zero"] = "lower_leaky"


class FeatureRecoveryExperimentConfig(BaseConfig):
    run_name: str
    out_dir: Path | None = None
    seed: int = 0
    synthetic: SyntheticFamilyConfig
    target: TargetTrainingConfig
    spd: SPDExecutionConfig
    analysis: AnalysisConfig = AnalysisConfig()

from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from spd.base_config import BaseConfig
from spd.configs import Config, ScheduleConfig
from spd.experiments.resid_mlp.configs import ResidMLPModelConfig


class FeatureDictionaryConfig(BaseConfig):
    num_features: PositiveInt
    hidden_dim: PositiveInt
    seed: int = 0
    initializer: Literal["orthogonal", "random"] = "orthogonal"
    normalize_vectors: bool = True


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
    correlation_strength: float = 0.0
    correlation_rank: PositiveInt = 4
    seed: int = 0

    @model_validator(mode="after")
    def validate_probabilities(self) -> "ActivationGeneratorConfig":
        if self.firing_probabilities is not None:
            assert self.zipf_exponent is None, (
                "Use either firing_probabilities or zipf_exponent, not both"
            )
        return self


class DictionaryDatasetConfig(BaseConfig):
    dictionary: FeatureDictionaryConfig
    generator: ActivationGeneratorConfig


class SAEBaselineConfig(BaseConfig):
    dataset: DictionaryDatasetConfig
    d_sae: PositiveInt
    batch_size: PositiveInt
    steps: PositiveInt
    lr: PositiveFloat
    l1_coefficient: PositiveFloat
    eval_num_samples: PositiveInt = 10000
    seed: int = 0
    run_name: str = "sae_baseline"
    out_dir: Path | None = None


class FeatureRecoveryTrainConfig(BaseConfig):
    dataset: DictionaryDatasetConfig
    resid_mlp_model_config: ResidMLPModelConfig
    label_type: Literal["act_plus_resid", "abs", "identity"] = "act_plus_resid"
    loss_type: Literal["readoff", "resid"] = "readoff"
    label_coeffs: list[float] | None = None
    fixed_embedding: Literal["learned", "identity", "random", "feature_dictionary"] = "feature_dictionary"
    freeze_embedding: bool = True
    batch_size: PositiveInt = 2048
    steps: PositiveInt = 10000
    print_freq: PositiveInt = 100
    lr_schedule: ScheduleConfig
    n_eval_batches: PositiveInt = 20
    seed: int = 0
    run_name: str = "resid_mlp_feature_recovery"
    out_dir: Path | None = None


class FeatureRecoverySPDConfig(BaseConfig):
    spd_config_path: Path
    target_run_dir: Path
    n_eval_steps: PositiveInt = 100
    seed: int = 0
    run_name: str = "feature_recovery_spd"

    def load_spd_config(self) -> Config:
        return Config.from_file(self.spd_config_path)

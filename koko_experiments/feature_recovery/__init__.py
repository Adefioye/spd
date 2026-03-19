"""Feature-recovery experiment helpers."""

from .configs import (
    ActivationGeneratorConfig,
    DictionaryDatasetConfig,
    FeatureDictionaryConfig,
    FeatureRecoverySPDConfig,
    FeatureRecoveryTrainConfig,
    SAEBaselineConfig,
)
from .synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    generate_random_correlation_matrix,
    zipfian_firing_probabilities,
)

__all__ = [
    "ActivationGenerator",
    "ActivationGeneratorConfig",
    "DictionaryDatasetConfig",
    "FeatureDictionary",
    "FeatureDictionaryConfig",
    "FeatureRecoverySPDConfig",
    "FeatureRecoveryTrainConfig",
    "SAEBaselineConfig",
    "generate_random_correlation_matrix",
    "zipfian_firing_probabilities",
]

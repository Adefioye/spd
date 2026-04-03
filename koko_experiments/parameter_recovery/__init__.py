"""Feature-recovery experiment helpers."""

from .configs import (
    ActivationGeneratorConfig,
    AnalysisConfig,
    FeatureDictionaryConfig,
    ParameterRecoveryExperimentConfig,
    SPDExecutionConfig,
    SyntheticFamilyConfig,
    TargetTrainingConfig,
)
from .synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    generate_random_correlation_matrix,
    sample_observed_activations,
    zipfian_firing_probabilities,
)

__all__ = [
    "ActivationGenerator",
    "ActivationGeneratorConfig",
    "AnalysisConfig",
    "FeatureDictionary",
    "FeatureDictionaryConfig",
    "ParameterRecoveryExperimentConfig",
    "SPDExecutionConfig",
    "SyntheticFamilyConfig",
    "TargetTrainingConfig",
    "generate_random_correlation_matrix",
    "sample_observed_activations",
    "zipfian_firing_probabilities",
]

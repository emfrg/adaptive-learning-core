"""Item Response Theory (IRT) module for question quality analysis.

Implements IRT models for:
1. Estimating question difficulty and discrimination
2. Estimating learner ability
3. Evaluating question quality
4. Identifying problematic questions

IRT Models:
- 1PL (Rasch): P(X=1|θ,b) = 1 / (1 + e^(-(θ-b)))
- 2PL: P(X=1|θ,a,b) = 1 / (1 + e^(-a(θ-b)))
- 3PL: P(X=1|θ,a,b,c) = c + (1-c) / (1 + e^(-a(θ-b)))

Where:
- θ = learner ability
- b = item difficulty
- a = item discrimination
- c = guessing parameter
"""

from adaptive_learning_core.irt.models import (
    IRTModel,
    Rasch1PL,
    Model2PL,
    Model3PL,
    ItemParameters,
    ResponsePattern,
)
from adaptive_learning_core.irt.estimator import (
    IRTEstimator,
    EstimationConfig,
    EstimationResult,
    ItemEstimator,
    AbilityEstimator,
)
from adaptive_learning_core.irt.discrimination import (
    DiscriminationAnalyzer,
    DiscriminationResult,
    QualityCategory,
)

__all__ = [
    # Models
    "IRTModel",
    "Rasch1PL",
    "Model2PL",
    "Model3PL",
    "ItemParameters",
    "ResponsePattern",
    # Estimation
    "IRTEstimator",
    "EstimationConfig",
    "EstimationResult",
    "ItemEstimator",
    "AbilityEstimator",
    # Discrimination
    "DiscriminationAnalyzer",
    "DiscriminationResult",
    "QualityCategory",
]

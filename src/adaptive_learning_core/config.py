"""Configuration management for the adaptive learning system.

This module provides dataclasses for configuring all algorithm parameters
with sensible defaults based on the research paper.

Parameters:
    α (alpha): Mastery Progression Rate - how quickly mastery scores change
    β (beta): Mastery Decay Parameter - rate of knowledge decay over time
    γ (gamma): Prior probability for unseen cards in selection
    δ (delta): Boosting factor for gamma when quiz score exceeds threshold
    θ (theta): Quiz score threshold for boosting gamma
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MasteryConfig:
    """Configuration for mastery score updates (Equation 3).

    The mastery update rule: M(c) = M(c) + α × C_{c,q} × δ

    Attributes:
        alpha_slow: Progression rate for slow pace preference (0.05)
        alpha_medium: Progression rate for medium pace preference (0.10)
        alpha_fast: Progression rate for fast pace preference (0.15)
        min_score: Minimum mastery score floor (0.0)
        max_score: Maximum mastery score ceiling (1.0)
        learned_threshold: Mastery score threshold to consider a card "learned" (0.8)
    """
    alpha_slow: float = 0.05
    alpha_medium: float = 0.10
    alpha_fast: float = 0.15
    min_score: float = 0.0
    max_score: float = 1.0
    learned_threshold: float = 0.8

    def get_alpha(self, pace: Literal["slow", "medium", "fast"]) -> float:
        """Get the alpha parameter for a given pace preference."""
        pace_map = {
            "slow": self.alpha_slow,
            "medium": self.alpha_medium,
            "fast": self.alpha_fast,
        }
        return pace_map.get(pace, self.alpha_medium)


@dataclass(frozen=True)
class DecayConfig:
    """Configuration for knowledge decay (Equations 7-8).

    The decay function: M(c) = M(c) × e^{-β × λ × τ}

    Where λ (decay multiplier) is:
        - 0 if τ < threshold_hours
        - same_day_multiplier if τ ≥ threshold_hours and same date
        - different_day_multiplier if τ ≥ threshold_hours and different date

    Attributes:
        beta: Base decay parameter (β)
        threshold_hours: Hours before decay begins (4.0)
        same_day_multiplier: λ when same calendar date (1.0)
        different_day_multiplier: λ when different calendar date (2.0)
    """
    beta: float = 0.1
    threshold_hours: float = 4.0
    same_day_multiplier: float = 1.0
    different_day_multiplier: float = 2.0


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for card and question selection (Equations 4-5).

    Card selection probability: P(c) = (1 - M(c) + γ) / Z
    Gamma boosting: γ_new = γ × δ when quiz_score > θ

    Attributes:
        gamma: Prior probability constant for unseen cards (γ)
        delta: Boosting factor for gamma (δ)
        score_threshold: Quiz score threshold θ for boosting (0.8)
        default_cards_per_cycle: Default number of cards per learning cycle
        default_questions_per_quiz: Default number of questions per quiz
    """
    gamma: float = 0.1
    delta: float = 1.5
    score_threshold: float = 0.8
    default_cards_per_cycle: int = 5
    default_questions_per_quiz: int = 5


@dataclass(frozen=True)
class QuestionLinkConfig:
    """Configuration for question-to-card linking (Equations 1-2).

    Attributes:
        top_k: Number of top cards to link to each question
        min_similarity: Minimum cosine similarity threshold for linking
    """
    top_k: int = 3
    min_similarity: float = 0.3


@dataclass(frozen=True)
class BloomConfig:
    """Configuration for Bloom's Taxonomy classification.

    Attributes:
        levels_for_novice: Bloom levels appropriate for Novice (0-1)
        levels_for_beginner: Bloom levels appropriate for Beginner (1-2)
        levels_for_intermediate: Bloom levels appropriate for Intermediate (2-3)
        levels_for_advanced: Bloom levels appropriate for Advanced (3-4)
        levels_for_expert: Bloom levels appropriate for Expert (4-6)
    """
    # Mapping from learner stereotype level (0-4) to max Bloom level
    stereotype_to_max_bloom: dict[int, int] = field(default_factory=lambda: {
        0: 1,  # Novice -> Remember only
        1: 2,  # Beginner -> Remember, Understand
        2: 3,  # Intermediate -> up to Apply
        3: 4,  # Advanced -> up to Analyze
        4: 6,  # Expert -> all levels including Evaluate, Create
    })


@dataclass(frozen=True)
class IRTConfig:
    """Configuration for Item Response Theory models.

    Attributes:
        default_difficulty: Default item difficulty (b) for new questions
        default_discrimination: Default discrimination (a) for new questions
        default_guessing: Default guessing parameter (c) for 3PL model
        min_responses_for_estimation: Minimum responses needed to estimate parameters
        ability_prior_mean: Prior mean for learner ability (θ)
        ability_prior_std: Prior std for learner ability
    """
    default_difficulty: float = 0.0
    default_discrimination: float = 1.0
    default_guessing: float = 0.25  # For 4-option MCQ
    min_responses_for_estimation: int = 30
    ability_prior_mean: float = 0.0
    ability_prior_std: float = 1.0


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for question quality evaluation.

    Attributes:
        min_quality_score: Minimum score for question acceptance
        discrimination_threshold_low: Below this, flag as poor question
        discrimination_threshold_high: Above this, consider excellent
    """
    min_quality_score: float = 0.6
    discrimination_threshold_low: float = 0.5
    discrimination_threshold_high: float = 1.0


@dataclass(frozen=True)
class AdaptiveLearningConfig:
    """Master configuration for the entire adaptive learning system.

    Provides centralized configuration with sensible defaults that can be
    overridden as needed.

    Example:
        >>> config = AdaptiveLearningConfig()
        >>> config.mastery.get_alpha("fast")
        0.15
        >>> custom_config = AdaptiveLearningConfig(
        ...     mastery=MasteryConfig(alpha_fast=0.20)
        ... )
    """
    mastery: MasteryConfig = field(default_factory=MasteryConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    question_link: QuestionLinkConfig = field(default_factory=QuestionLinkConfig)
    bloom: BloomConfig = field(default_factory=BloomConfig)
    irt: IRTConfig = field(default_factory=IRTConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# Default configuration instance
DEFAULT_CONFIG = AdaptiveLearningConfig()

"""Mastery score update algorithm (Equation 3 from paper).

The Mastery Update Rule governs how a learner's understanding of each
card evolves based on quiz performance.

Equation 3:
    M(c) = M(c) + α × C_{c,q} × δ

Where:
    - M(c) is the mastery score for card c
    - α is the Mastery Progression Rate Parameter (from pace preference)
    - C_{c,q} is the contribution weight of card c to question q
    - δ = +1 if correct, -1 if incorrect
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from adaptive_learning_core.models.card import CardState
from adaptive_learning_core.models.question import Question


@dataclass(frozen=True)
class MasteryUpdateConfig:
    """Configuration for mastery score updates.

    Attributes:
        alpha_slow: Progression rate for slow pace preference
        alpha_medium: Progression rate for medium pace preference
        alpha_fast: Progression rate for fast pace preference
        min_score: Minimum mastery score (floor)
        max_score: Maximum mastery score (ceiling)
    """

    alpha_slow: float = 0.05
    alpha_medium: float = 0.10
    alpha_fast: float = 0.15
    min_score: float = 0.0
    max_score: float = 1.0

    def get_alpha(self, pace: Literal["slow", "medium", "fast"]) -> float:
        """Get progression rate parameter based on pace preference.

        Args:
            pace: One of "slow", "medium", "fast"

        Returns:
            The α parameter value
        """
        pace_map = {
            "slow": self.alpha_slow,
            "medium": self.alpha_medium,
            "fast": self.alpha_fast,
        }
        return pace_map.get(pace, self.alpha_medium)


@dataclass
class MasteryUpdateResult:
    """Result of a mastery update operation.

    Attributes:
        card_id: The card that was updated
        old_score: Previous mastery score
        new_score: Updated mastery score
        delta: The direction (+1 or -1)
        contribution_weight: The C_{c,q} weight used
        was_correct: Whether the answer was correct
        alpha_used: The α value that was applied
    """

    card_id: str
    old_score: float
    new_score: float
    delta: float
    contribution_weight: float
    was_correct: bool
    alpha_used: float

    @property
    def score_change(self) -> float:
        """Absolute change in score."""
        return abs(self.new_score - self.old_score)

    @property
    def effective_change(self) -> float:
        """Signed change in score (positive = improved)."""
        return self.new_score - self.old_score


class MasteryUpdater:
    """Updates mastery scores based on quiz performance.

    Implements Equation 3 from Section 3.2.4:

        M(c) = M(c) + α × C_{c,q} × δ

    The update is applied to all cards linked to a question when
    the learner answers that question.

    Example:
        >>> updater = MasteryUpdater()
        >>> result = updater.update_single(
        ...     card_state, question, is_correct=True, pace="medium"
        ... )
        >>> print(f"Score changed from {result.old_score} to {result.new_score}")
    """

    def __init__(self, config: Optional[MasteryUpdateConfig] = None):
        """Initialize the updater with optional custom config.

        Args:
            config: Custom configuration, or None for defaults
        """
        self.config = config or MasteryUpdateConfig()

    def update_single(
        self,
        card_state: CardState,
        question: Question,
        is_correct: bool,
        pace: Literal["slow", "medium", "fast"] = "medium",
        update_timestamp: Optional[datetime] = None,
    ) -> MasteryUpdateResult:
        """Update mastery score for a single card based on one question.

        Applies Equation 3: M(c) = M(c) + α × C_{c,q} × δ

        Args:
            card_state: Current state of the card (will be modified in place)
            question: The question that was answered
            is_correct: Whether the answer was correct
            pace: Learner's pace preference
            update_timestamp: Optional timestamp for the update

        Returns:
            MasteryUpdateResult with old/new scores and update details
        """
        alpha = self.config.get_alpha(pace)

        # Get contribution weight for this card-question pair
        # Default to equal weight if not specified
        contribution = question.get_contribution_weight(card_state.card_id)

        # δ = +1 for correct, -1 for incorrect
        delta = 1.0 if is_correct else -1.0

        # Apply update rule: M(c) = M(c) + α × C_{c,q} × δ
        old_score = card_state.mastery_score
        raw_new_score = old_score + (alpha * contribution * delta)

        # Clamp to valid range [0, 1]
        new_score = max(
            self.config.min_score,
            min(self.config.max_score, raw_new_score),
        )

        # Update the card state in place
        card_state.mastery_score = new_score
        card_state.times_seen += 1
        if is_correct:
            card_state.times_correct += 1
        else:
            card_state.times_incorrect += 1

        # Update timestamps
        timestamp = update_timestamp or datetime.utcnow()
        card_state.last_seen_at = timestamp
        card_state.last_quiz_at = timestamp

        return MasteryUpdateResult(
            card_id=card_state.card_id,
            old_score=old_score,
            new_score=new_score,
            delta=delta,
            contribution_weight=contribution,
            was_correct=is_correct,
            alpha_used=alpha,
        )

    def update_batch(
        self,
        card_states: dict[str, CardState],
        question: Question,
        is_correct: bool,
        pace: Literal["slow", "medium", "fast"] = "medium",
        update_timestamp: Optional[datetime] = None,
    ) -> list[MasteryUpdateResult]:
        """Update mastery scores for all cards linked to a question.

        This is the typical use case: when a learner answers a question,
        all linked cards have their mastery scores updated.

        Args:
            card_states: Map of card_id -> CardState
            question: The answered question
            is_correct: Whether answer was correct
            pace: Learner's pace preference
            update_timestamp: Optional timestamp for the update

        Returns:
            List of update results for each affected card
        """
        results = []
        timestamp = update_timestamp or datetime.utcnow()

        for card_id in question.linked_card_ids:
            if card_id in card_states:
                result = self.update_single(
                    card_states[card_id],
                    question,
                    is_correct,
                    pace,
                    timestamp,
                )
                results.append(result)

        return results

    def calculate_expected_change(
        self,
        question: Question,
        card_id: str,
        pace: Literal["slow", "medium", "fast"] = "medium",
    ) -> tuple[float, float]:
        """Calculate expected score change for correct/incorrect answers.

        Useful for simulations and UI previews.

        Args:
            question: The question
            card_id: The card ID
            pace: Pace preference

        Returns:
            Tuple of (change_if_correct, change_if_incorrect)
        """
        alpha = self.config.get_alpha(pace)
        contribution = question.get_contribution_weight(card_id)

        change_correct = alpha * contribution * 1.0
        change_incorrect = alpha * contribution * -1.0

        return (change_correct, change_incorrect)

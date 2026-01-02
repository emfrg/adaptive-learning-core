"""Time-based decay in mastery scores (Equations 7-8 from paper).

Implements Ebbinghaus's Forgetting Curve to simulate natural memory decay.
This encourages periodic review of material, aiding knowledge consolidation.

Equation 7:
    M(c) = M(c) × e^{-β × λ × τ}

Equation 8 (Decay Multiplier λ):
    λ = 0   if τ < 4 hours (no decay)
    λ = 1   if τ ≥ 4 hours and same calendar date
    λ = 2   if τ ≥ 4 hours and different calendar date
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from adaptive_learning_core.models.card import CardState


@dataclass(frozen=True)
class DecayConfig:
    """Configuration for knowledge decay.

    Attributes:
        beta: Base decay parameter (β)
        threshold_hours: Hours before decay begins
        same_day_multiplier: λ when same calendar date
        different_day_multiplier: λ when different calendar date
    """

    beta: float = 0.1
    threshold_hours: float = 4.0
    same_day_multiplier: float = 1.0
    different_day_multiplier: float = 2.0


@dataclass
class DecayResult:
    """Result of a decay calculation.

    Attributes:
        card_id: The card that was decayed
        old_score: Previous mastery score
        new_score: Decayed mastery score
        decay_multiplier: The λ value used
        hours_elapsed: Hours since last interaction
        decay_applied: Whether decay was actually applied
        decay_factor: The e^{-β×λ×τ} factor applied
    """

    card_id: str
    old_score: float
    new_score: float
    decay_multiplier: float
    hours_elapsed: float
    decay_applied: bool
    decay_factor: float

    @property
    def score_lost(self) -> float:
        """Amount of mastery score lost to decay."""
        return self.old_score - self.new_score

    @property
    def retention_rate(self) -> float:
        """Percentage of score retained after decay."""
        if self.old_score == 0:
            return 1.0
        return self.new_score / self.old_score


class KnowledgeDecay:
    """Applies time-based decay to mastery scores.

    Implements Equations 7-8 from Section 3.2.6:

        M(c) = M(c) × e^{-β × λ × τ}

    The decay is designed to reflect Ebbinghaus's Forgetting Curve,
    where memory retention diminishes rapidly after initial learning.

    The decay multiplier λ considers:
    - No decay if less than 4 hours have passed
    - Slower decay if revisiting on the same calendar day
    - Faster decay if revisiting on a different calendar day

    Example:
        >>> decay = KnowledgeDecay()
        >>> result = decay.apply_decay(card_state)
        >>> print(f"Score decayed from {result.old_score:.3f} to {result.new_score:.3f}")
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        """Initialize with optional custom configuration.

        Args:
            config: Custom decay configuration, or None for defaults
        """
        self.config = config or DecayConfig()

    def calculate_decay_multiplier(
        self,
        last_interaction: datetime,
        current_time: datetime,
    ) -> float:
        """Calculate the decay multiplier λ based on time context.

        Implements Equation 8:
            λ = 0 if τ < 4 hours
            λ = 1 if τ ≥ 4 hours and same date
            λ = 2 if τ ≥ 4 hours and different date

        Args:
            last_interaction: When the card was last seen
            current_time: Current timestamp

        Returns:
            The decay multiplier λ
        """
        elapsed = current_time - last_interaction
        hours_elapsed = elapsed.total_seconds() / 3600

        # No decay if less than threshold hours
        if hours_elapsed < self.config.threshold_hours:
            return 0.0

        # Check if same calendar date
        same_date = last_interaction.date() == current_time.date()

        if same_date:
            return self.config.same_day_multiplier
        else:
            return self.config.different_day_multiplier

    def calculate_decay_factor(
        self,
        hours_elapsed: float,
        decay_multiplier: float,
    ) -> float:
        """Calculate the decay factor e^{-β × λ × τ}.

        Args:
            hours_elapsed: Time since last interaction (τ)
            decay_multiplier: The λ value

        Returns:
            The decay factor to multiply mastery score by
        """
        if decay_multiplier == 0:
            return 1.0  # No decay

        exponent = -self.config.beta * decay_multiplier * hours_elapsed
        return math.exp(exponent)

    def apply_decay(
        self,
        card_state: CardState,
        current_time: Optional[datetime] = None,
        in_place: bool = True,
    ) -> DecayResult:
        """Apply time-based decay to a card's mastery score.

        Implements Equation 7: M(c) = M(c) × e^{-β × λ × τ}

        Args:
            card_state: The card state to decay
            current_time: Current time (defaults to now)
            in_place: Whether to modify card_state (True) or just calculate

        Returns:
            DecayResult with old/new scores and decay info
        """
        current_time = current_time or datetime.utcnow()

        # If never seen, no decay to apply
        if card_state.last_seen_at is None:
            return DecayResult(
                card_id=card_state.card_id,
                old_score=card_state.mastery_score,
                new_score=card_state.mastery_score,
                decay_multiplier=0.0,
                hours_elapsed=0.0,
                decay_applied=False,
                decay_factor=1.0,
            )

        # Calculate elapsed time
        elapsed = current_time - card_state.last_seen_at
        hours_elapsed = elapsed.total_seconds() / 3600

        # Calculate decay multiplier (λ)
        lambda_mult = self.calculate_decay_multiplier(
            card_state.last_seen_at,
            current_time,
        )

        old_score = card_state.mastery_score

        # Calculate decay factor
        decay_factor = self.calculate_decay_factor(hours_elapsed, lambda_mult)

        # Apply decay: M(c) = M(c) × e^{-β × λ × τ}
        if lambda_mult > 0:
            new_score = old_score * decay_factor
            decay_applied = True

            if in_place:
                card_state.mastery_score = new_score
        else:
            new_score = old_score
            decay_applied = False

        return DecayResult(
            card_id=card_state.card_id,
            old_score=old_score,
            new_score=new_score,
            decay_multiplier=lambda_mult,
            hours_elapsed=hours_elapsed,
            decay_applied=decay_applied,
            decay_factor=decay_factor,
        )

    def apply_decay_batch(
        self,
        card_states: list[CardState],
        current_time: Optional[datetime] = None,
        in_place: bool = True,
    ) -> list[DecayResult]:
        """Apply decay to multiple cards.

        Typically called at the start of a learning session to
        decay all cards based on time since last interaction.

        Args:
            card_states: List of card states
            current_time: Current time (defaults to now)
            in_place: Whether to modify card_states

        Returns:
            List of decay results
        """
        current_time = current_time or datetime.utcnow()
        return [
            self.apply_decay(state, current_time, in_place)
            for state in card_states
        ]

    def preview_decay(
        self,
        mastery_score: float,
        hours_from_now: float,
        same_day: bool = False,
    ) -> float:
        """Preview what a mastery score would be after a certain time.

        Useful for simulations and UI displays.

        Args:
            mastery_score: Current mastery score
            hours_from_now: Hours until the preview time
            same_day: Whether it would be the same calendar day

        Returns:
            The projected mastery score after decay
        """
        if hours_from_now < self.config.threshold_hours:
            return mastery_score

        lambda_mult = (
            self.config.same_day_multiplier
            if same_day
            else self.config.different_day_multiplier
        )

        decay_factor = self.calculate_decay_factor(hours_from_now, lambda_mult)
        return mastery_score * decay_factor

    def time_to_threshold(
        self,
        current_score: float,
        target_score: float,
        same_day: bool = False,
    ) -> Optional[float]:
        """Calculate hours until mastery decays to a target score.

        Useful for determining when a card needs review.

        Args:
            current_score: Current mastery score
            target_score: Target score to decay to
            same_day: Whether to use same-day decay rate

        Returns:
            Hours until decay reaches target, or None if impossible
        """
        if target_score >= current_score or target_score <= 0 or current_score <= 0:
            return None

        lambda_mult = (
            self.config.same_day_multiplier
            if same_day
            else self.config.different_day_multiplier
        )

        if lambda_mult == 0:
            return None  # No decay

        # Solve: target = current × e^{-β×λ×τ} for τ
        # ln(target/current) = -β×λ×τ
        # τ = -ln(target/current) / (β×λ)
        ratio = target_score / current_score
        hours = -math.log(ratio) / (self.config.beta * lambda_mult)

        return max(0, hours + self.config.threshold_hours)

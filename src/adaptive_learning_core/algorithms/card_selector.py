"""Card selection algorithm for learning cycles (Equations 4-5 from paper).

Implements probability-based card selection that prioritizes cards
with lower mastery scores while ensuring new cards aren't neglected.

Equation 4 (Card Selection Probability):
    P(c) = (1 - M(c) + γ) / Z

    Where:
    - M(c) is the mastery score
    - γ is a prior probability for unseen cards
    - Z is the normalization constant

Equation 5 (Gamma Boosting):
    γ_new = γ × δ  (when quiz score > θ)

    This encourages introducing new material when the learner
    is performing well.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from adaptive_learning_core.models.card import CardState


@dataclass(frozen=True)
class CardSelectionConfig:
    """Configuration for card selection.

    Attributes:
        gamma: Prior probability constant for unseen cards (γ)
        delta: Boosting factor for gamma (δ)
        score_threshold: Quiz score threshold θ for boosting
        default_selection_size: Default number of cards to select per cycle
    """

    gamma: float = 0.1
    delta: float = 1.5
    score_threshold: float = 0.8
    default_selection_size: int = 5


@dataclass
class CardSelectionResult:
    """Result of card selection.

    Attributes:
        selected_card_ids: IDs of the selected cards
        selection_probabilities: Probability for each valid card
        gamma_used: The γ value used for selection
        total_valid_cards: Number of valid (non-ignored) cards
        num_unseen_cards: Number of cards never seen before
    """

    selected_card_ids: list[str]
    selection_probabilities: dict[str, float]
    gamma_used: float
    total_valid_cards: int
    num_unseen_cards: int

    @property
    def num_selected(self) -> int:
        """Number of cards selected."""
        return len(self.selected_card_ids)


class CardSelector:
    """Selects cards for the next learning cycle.

    Implements Rule 1 from Section 3.2.5:
    - Cards with lower mastery scores have higher selection probability
    - A prior γ ensures unseen cards aren't neglected
    - γ is boosted when the learner performs well (score > θ)

    The design promotes uniform progression across all cards by
    focusing practice on weaker areas.

    Example:
        >>> selector = CardSelector()
        >>> result = selector.select_cards(card_states, n=5)
        >>> print(f"Selected: {result.selected_card_ids}")
    """

    def __init__(self, config: Optional[CardSelectionConfig] = None):
        """Initialize the selector with optional custom config.

        Args:
            config: Custom configuration, or None for defaults
        """
        self.config = config or CardSelectionConfig()

    def calculate_probabilities(
        self,
        card_states: list[CardState],
        gamma: Optional[float] = None,
    ) -> dict[str, float]:
        """Calculate selection probability for each card.

        Implements Equation 4: P(c) = (1 - M(c) + γ) / Z

        Args:
            card_states: List of valid (non-ignored) card states
            gamma: Prior probability (uses config default if None)

        Returns:
            Dictionary mapping card_id -> selection probability
        """
        gamma = gamma if gamma is not None else self.config.gamma

        # Filter out ignored cards
        valid_states = [s for s in card_states if not s.is_ignored]

        if not valid_states:
            return {}

        # Calculate raw probabilities: (1 - M(c)) + γ
        raw_probs: dict[str, float] = {}
        for state in valid_states:
            # Higher probability for lower mastery scores
            raw_probs[state.card_id] = (1 - state.mastery_score) + gamma

        # Normalize: Z = Σ((1 - M(c)) + γ)
        z = sum(raw_probs.values())

        if z == 0:
            # Edge case: all cards at max mastery and gamma=0
            # Return uniform distribution
            uniform_prob = 1.0 / len(valid_states)
            return {card_id: uniform_prob for card_id in raw_probs}

        # P(c) = (1 - M(c) + γ) / Z
        return {card_id: prob / z for card_id, prob in raw_probs.items()}

    def maybe_boost_gamma(
        self,
        current_gamma: float,
        quiz_score: float,
    ) -> float:
        """Conditionally boost gamma based on quiz performance.

        Implements Equation 5: γ_new = γ × δ when score > θ

        This encourages the introduction of new material when
        the learner is performing well.

        Args:
            current_gamma: Current gamma value
            quiz_score: Score from most recent quiz (0-1)

        Returns:
            Possibly boosted gamma value
        """
        if quiz_score > self.config.score_threshold:
            return current_gamma * self.config.delta
        return current_gamma

    def select_cards(
        self,
        card_states: list[CardState],
        n: Optional[int] = None,
        gamma: Optional[float] = None,
        previous_quiz_score: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> CardSelectionResult:
        """Select cards for the next learning cycle.

        Args:
            card_states: All card states for the module
            n: Number of cards to select (uses config default)
            gamma: Prior probability (uses config default)
            previous_quiz_score: Score from last quiz (for γ boosting)
            random_state: Random seed for reproducibility

        Returns:
            CardSelectionResult with selected cards and probabilities
        """
        n = n if n is not None else self.config.default_selection_size
        gamma = gamma if gamma is not None else self.config.gamma

        # Maybe boost gamma based on previous performance
        if previous_quiz_score is not None:
            gamma = self.maybe_boost_gamma(gamma, previous_quiz_score)

        # Filter valid cards
        valid_states = [s for s in card_states if not s.is_ignored]
        num_unseen = sum(1 for s in valid_states if s.is_unseen)

        # Calculate probabilities
        probabilities = self.calculate_probabilities(valid_states, gamma)

        if not probabilities:
            return CardSelectionResult(
                selected_card_ids=[],
                selection_probabilities={},
                gamma_used=gamma,
                total_valid_cards=0,
                num_unseen_cards=0,
            )

        # Set random state for reproducibility
        rng = np.random.default_rng(random_state)

        # Sample cards based on probabilities
        card_ids = list(probabilities.keys())
        probs = np.array([probabilities[cid] for cid in card_ids])

        # Normalize in case of floating point errors
        probs = probs / probs.sum()

        # Sample without replacement
        n_to_select = min(n, len(card_ids))
        selected_indices = rng.choice(
            len(card_ids),
            size=n_to_select,
            replace=False,
            p=probs,
        )

        selected_ids = [card_ids[i] for i in selected_indices]

        return CardSelectionResult(
            selected_card_ids=selected_ids,
            selection_probabilities=probabilities,
            gamma_used=gamma,
            total_valid_cards=len(valid_states),
            num_unseen_cards=num_unseen,
        )

    def get_unseen_cards(
        self,
        card_states: list[CardState],
    ) -> list[CardState]:
        """Get cards that haven't been seen yet.

        Unseen cards have M(c) = 0 and times_seen = 0.

        Args:
            card_states: All card states

        Returns:
            List of unseen card states
        """
        return [s for s in card_states if not s.is_ignored and s.is_unseen]

    def get_weak_cards(
        self,
        card_states: list[CardState],
        threshold: float = 0.4,
    ) -> list[CardState]:
        """Get cards with mastery below a threshold.

        Useful for identifying cards that need extra practice.

        Args:
            card_states: All card states
            threshold: Mastery threshold (default 0.4 = Beginner)

        Returns:
            List of weak card states, sorted by mastery (lowest first)
        """
        weak = [
            s
            for s in card_states
            if not s.is_ignored and s.mastery_score < threshold
        ]
        return sorted(weak, key=lambda s: s.mastery_score)

    def get_cards_by_stereotype(
        self,
        card_states: list[CardState],
        stereotype_level: int,
    ) -> list[CardState]:
        """Get cards at a specific stereotype level.

        Args:
            card_states: All card states
            stereotype_level: Target level (0-4)

        Returns:
            List of cards at that stereotype level
        """
        return [
            s
            for s in card_states
            if not s.is_ignored and s.stereotype_level == stereotype_level
        ]

"""Question selection algorithm for quizzes (Rules 2 & 3 from paper).

Implements question filtering and selection based on:
- Target Mastery Level (only questions at or below this difficulty)
- Linked cards (only questions linked to selected cards)
- Learner stereotype matching (appropriate difficulty for current level)

Rules 2 & 3 from Section 3.2.5:
- Questions are filtered based on Target Mastery Level T
- Only questions with difficulty level S(q) ≤ T are considered
- Only questions linked to cards in the current Learning Cycle are included
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from adaptive_learning_core.models.card import CardState
from adaptive_learning_core.models.question import Question


@dataclass(frozen=True)
class QuestionSelectionConfig:
    """Configuration for question selection.

    Attributes:
        default_quiz_size: Default number of questions per quiz
        difficulty_tolerance: How far above stereotype to allow (0 = exact match)
        prefer_unasked: Whether to prefer questions not yet asked
        balance_difficulty: Whether to balance difficulty distribution
    """

    default_quiz_size: int = 5
    difficulty_tolerance: int = 1  # Allow 1 level above current stereotype
    prefer_unasked: bool = True
    balance_difficulty: bool = True


@dataclass
class QuestionSelectionResult:
    """Result of question selection.

    Attributes:
        selected_question_ids: IDs of selected questions
        questions_by_difficulty: Count of questions at each Bloom level
        total_eligible: Total questions that passed filters
        filtered_by_target: Questions filtered by target mastery
        filtered_by_cards: Questions filtered by card linkage
    """

    selected_question_ids: list[str]
    questions_by_difficulty: dict[int, int]
    total_eligible: int
    filtered_by_target: int
    filtered_by_cards: int

    @property
    def num_selected(self) -> int:
        """Number of questions selected."""
        return len(self.selected_question_ids)


class QuestionSelector:
    """Selects appropriate questions for a quiz.

    Implements Rules 2 & 3 from Section 3.2.5:

    Rule 2 (Target Mastery Filter):
        Only include questions where difficulty ≤ Target Mastery Level.
        If learner's target is "Advanced" (level 3), no questions at
        "Expert" difficulty will be shown.

    Rule 3 (Card Linkage Filter):
        Only include questions linked to cards in the current
        Learning Cycle. This ensures questions are relevant to
        the material just studied.

    Additional features:
    - Matches question difficulty to learner's current stereotype
    - Balances difficulty distribution for varied practice
    - Prefers questions not yet answered (if configured)

    Example:
        >>> selector = QuestionSelector()
        >>> result = selector.select_questions(
        ...     all_questions=questions,
        ...     card_states=card_states,
        ...     selected_card_ids=["c1", "c2", "c3"],
        ...     target_mastery_level=3,
        ...     n=5,
        ... )
        >>> print(f"Selected: {result.selected_question_ids}")
    """

    def __init__(self, config: Optional[QuestionSelectionConfig] = None):
        """Initialize the selector with optional custom config.

        Args:
            config: Custom configuration, or None for defaults
        """
        self.config = config or QuestionSelectionConfig()

    def filter_by_target_mastery(
        self,
        questions: list[Question],
        target_mastery_level: int,
    ) -> list[Question]:
        """Filter questions by target mastery level.

        Implements Rule 2: Only questions with S(q) ≤ T are considered.

        Args:
            questions: List of all questions
            target_mastery_level: Target stereotype level (0-4)

        Returns:
            Questions at or below the target level
        """
        return [q for q in questions if q.difficulty_level <= target_mastery_level]

    def filter_by_linked_cards(
        self,
        questions: list[Question],
        selected_card_ids: set[str],
    ) -> list[Question]:
        """Filter questions by linked cards.

        Implements Rule 3: Only questions linked to selected cards.

        Args:
            questions: List of questions
            selected_card_ids: Set of card IDs in current Learning Cycle

        Returns:
            Questions that are linked to at least one selected card
        """
        return [
            q
            for q in questions
            if any(card_id in selected_card_ids for card_id in q.linked_card_ids)
        ]

    def filter_by_stereotype_match(
        self,
        questions: list[Question],
        card_states: dict[str, CardState],
        tolerance: Optional[int] = None,
    ) -> list[Question]:
        """Filter questions appropriate for the learner's current level.

        For each question, checks if its difficulty matches the learner's
        stereotype for the linked cards (with tolerance).

        Args:
            questions: List of questions
            card_states: Map of card_id -> CardState
            tolerance: How many levels above stereotype to allow

        Returns:
            Questions matching the learner's current level
        """
        tolerance = tolerance if tolerance is not None else self.config.difficulty_tolerance
        result = []

        for question in questions:
            # Get the learner's level for linked cards
            linked_stereotypes = []
            for card_id in question.linked_card_ids:
                if card_id in card_states:
                    linked_stereotypes.append(card_states[card_id].stereotype_level)

            if not linked_stereotypes:
                # No linked cards with state, include by default
                result.append(question)
                continue

            # Use average stereotype level of linked cards
            avg_stereotype = sum(linked_stereotypes) / len(linked_stereotypes)
            max_appropriate_level = int(avg_stereotype) + tolerance

            # Include if question difficulty is appropriate
            if question.difficulty_level <= max_appropriate_level:
                result.append(question)

        return result

    def score_question(
        self,
        question: Question,
        card_states: dict[str, CardState],
        answered_question_ids: Optional[set[str]] = None,
    ) -> float:
        """Score a question for selection priority.

        Higher scores are better. Considers:
        - Whether question has been answered before
        - How well difficulty matches current stereotype
        - Question quality score (if available)

        Args:
            question: The question to score
            card_states: Map of card_id -> CardState
            answered_question_ids: Set of previously answered question IDs

        Returns:
            Priority score (higher = more likely to select)
        """
        score = 1.0

        # Prefer unasked questions
        if self.config.prefer_unasked and answered_question_ids:
            if question.id not in answered_question_ids:
                score += 2.0

        # Calculate average stereotype for linked cards
        linked_stereotypes = []
        for card_id in question.linked_card_ids:
            if card_id in card_states:
                linked_stereotypes.append(card_states[card_id].stereotype_level)

        if linked_stereotypes:
            avg_stereotype = sum(linked_stereotypes) / len(linked_stereotypes)
            # Prefer questions matching current level
            diff = abs(question.difficulty_level - avg_stereotype)
            score += max(0, 1.0 - (diff * 0.3))

        # Boost by quality score if available
        if question.quality_score is not None:
            score += question.quality_score

        return score

    def select_questions(
        self,
        all_questions: list[Question],
        card_states: dict[str, CardState],
        selected_card_ids: list[str],
        target_mastery_level: int,
        n: Optional[int] = None,
        answered_question_ids: Optional[set[str]] = None,
        random_state: Optional[int] = None,
    ) -> QuestionSelectionResult:
        """Select questions for a quiz.

        Applies filtering rules and then selects appropriate questions:
        1. Filter by target mastery level (Rule 2)
        2. Filter by linked cards (Rule 3)
        3. Score and select best questions

        Args:
            all_questions: All available questions
            card_states: Map of card_id -> CardState
            selected_card_ids: Cards in current Learning Cycle
            target_mastery_level: Target stereotype level (0-4)
            n: Number of questions to select
            answered_question_ids: Previously answered question IDs
            random_state: Random seed for reproducibility

        Returns:
            QuestionSelectionResult with selected questions
        """
        n = n if n is not None else self.config.default_quiz_size
        selected_card_set = set(selected_card_ids)

        # Track filtering statistics
        total_questions = len(all_questions)

        # Rule 2: Filter by target mastery
        filtered = self.filter_by_target_mastery(all_questions, target_mastery_level)
        filtered_by_target = total_questions - len(filtered)

        # Rule 3: Filter by linked cards
        pre_card_filter = len(filtered)
        filtered = self.filter_by_linked_cards(filtered, selected_card_set)
        filtered_by_cards = pre_card_filter - len(filtered)

        # Additional filtering by stereotype match (optional)
        filtered = self.filter_by_stereotype_match(filtered, card_states)

        total_eligible = len(filtered)

        if not filtered:
            return QuestionSelectionResult(
                selected_question_ids=[],
                questions_by_difficulty={},
                total_eligible=0,
                filtered_by_target=filtered_by_target,
                filtered_by_cards=filtered_by_cards,
            )

        # Score all eligible questions
        scored = [
            (q, self.score_question(q, card_states, answered_question_ids))
            for q in filtered
        ]

        # Set random state
        rng = np.random.default_rng(random_state)

        # Select questions
        if self.config.balance_difficulty:
            selected = self._select_balanced(scored, n, rng)
        else:
            selected = self._select_by_score(scored, n, rng)

        # Count difficulty distribution
        difficulty_counts: dict[int, int] = {}
        for q in selected:
            level = q.difficulty_level
            difficulty_counts[level] = difficulty_counts.get(level, 0) + 1

        return QuestionSelectionResult(
            selected_question_ids=[q.id for q in selected],
            questions_by_difficulty=difficulty_counts,
            total_eligible=total_eligible,
            filtered_by_target=filtered_by_target,
            filtered_by_cards=filtered_by_cards,
        )

    def _select_by_score(
        self,
        scored_questions: list[tuple[Question, float]],
        n: int,
        rng: np.random.Generator,
    ) -> list[Question]:
        """Select questions by score with some randomness.

        Args:
            scored_questions: List of (question, score) tuples
            n: Number to select
            rng: Random number generator

        Returns:
            Selected questions
        """
        if len(scored_questions) <= n:
            return [q for q, _ in scored_questions]

        # Sort by score (descending)
        sorted_questions = sorted(scored_questions, key=lambda x: x[1], reverse=True)

        # Take top candidates (more than n to add variety)
        pool_size = min(n * 2, len(sorted_questions))
        pool = sorted_questions[:pool_size]

        # Convert scores to probabilities
        scores = np.array([s for _, s in pool])
        scores = scores - scores.min() + 0.1  # Ensure positive
        probs = scores / scores.sum()

        # Sample without replacement
        indices = rng.choice(len(pool), size=n, replace=False, p=probs)
        return [pool[i][0] for i in indices]

    def _select_balanced(
        self,
        scored_questions: list[tuple[Question, float]],
        n: int,
        rng: np.random.Generator,
    ) -> list[Question]:
        """Select questions with balanced difficulty distribution.

        Args:
            scored_questions: List of (question, score) tuples
            n: Number to select
            rng: Random number generator

        Returns:
            Selected questions with varied difficulties
        """
        # Group by difficulty level
        by_difficulty: dict[int, list[tuple[Question, float]]] = {}
        for q, score in scored_questions:
            level = q.difficulty_level
            if level not in by_difficulty:
                by_difficulty[level] = []
            by_difficulty[level].append((q, score))

        selected: list[Question] = []
        levels = sorted(by_difficulty.keys())

        if not levels:
            return []

        # Round-robin selection across difficulty levels
        while len(selected) < n:
            made_selection = False
            for level in levels:
                if len(selected) >= n:
                    break
                pool = by_difficulty[level]
                if pool:
                    # Select highest scored question from this level
                    pool.sort(key=lambda x: x[1], reverse=True)
                    q, _ = pool.pop(0)
                    selected.append(q)
                    made_selection = True

            if not made_selection:
                break  # No more questions available

        return selected

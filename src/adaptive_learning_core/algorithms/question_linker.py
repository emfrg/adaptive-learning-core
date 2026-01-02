"""Question-to-Card linking using cosine similarity (Equations 1-2 from paper).

Links generated questions to source cards to enable:
1. Targeted mastery updates based on quiz performance
2. Question selection based on card mastery levels

Equation 1 (Cosine Similarity):
    cos_sim(q, c) = (q · c) / (|q| × |c|)

Equation 2 (Softmax Contribution Weights):
    C_{c,q} = exp(cos_sim(q, c)) / Σ_{c' ∈ top-k} exp(cos_sim(q, c'))
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QuestionLinkConfig:
    """Configuration for question-card linking.

    Attributes:
        top_k: Number of top cards to link to each question
        min_similarity: Minimum similarity threshold for linking
    """

    top_k: int = 3
    min_similarity: float = 0.3


@dataclass
class LinkResult:
    """Result of linking a question to cards.

    Attributes:
        question_id: The question that was linked
        linked_cards: Card IDs that were linked
        similarities: Cosine similarity for each linked card
        contribution_weights: Softmax weights C_{c,q} for each card
    """

    question_id: str
    linked_cards: list[str]
    similarities: dict[str, float]
    contribution_weights: dict[str, float]

    @property
    def num_linked(self) -> int:
        """Number of cards linked."""
        return len(self.linked_cards)

    @property
    def primary_card(self) -> Optional[str]:
        """Card with highest contribution weight."""
        if not self.contribution_weights:
            return None
        return max(self.contribution_weights, key=self.contribution_weights.get)

    @property
    def max_similarity(self) -> float:
        """Maximum similarity score."""
        if not self.similarities:
            return 0.0
        return max(self.similarities.values())


class QuestionLinker:
    """Links questions to source cards using cosine similarity.

    This is essential for the adaptive system because:
    1. When a question is answered, we need to know which cards to update
    2. The contribution weight determines how much each card's score changes

    Implements Equations 1-2 from Section 3.2.3:

    Equation 1 - Cosine similarity:
        cos_sim(q, c) = (q · c) / (|q| × |c|)

    Equation 2 - Softmax contribution weights:
        C_{c,q} = exp(cos_sim(q, c)) / Σ_{c' ∈ top-k} exp(cos_sim(q, c'))

    Example:
        >>> linker = QuestionLinker()
        >>> result = linker.link_question(
        ...     question_id="q1",
        ...     question_embedding=q_emb,
        ...     card_embeddings={"c1": c1_emb, "c2": c2_emb}
        ... )
        >>> print(f"Linked to: {result.linked_cards}")
    """

    def __init__(self, config: Optional[QuestionLinkConfig] = None):
        """Initialize the linker with optional custom config.

        Args:
            config: Custom configuration, or None for defaults
        """
        self.config = config or QuestionLinkConfig()

    def cosine_similarity(
        self,
        vec_a: NDArray[np.float64],
        vec_b: NDArray[np.float64],
    ) -> float:
        """Calculate cosine similarity between two vectors.

        Implements Equation 1: cos_sim(q, c) = (q · c) / (|q| × |c|)

        Args:
            vec_a: First vector
            vec_b: Second vector

        Returns:
            Cosine similarity in range [-1, 1]
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def compute_contribution_weights(
        self,
        similarities: dict[str, float],
    ) -> dict[str, float]:
        """Compute softmax contribution weights from similarities.

        Implements Equation 2:
            C_{c,q} = exp(cos_sim(q, c)) / Σ_{c' ∈ top-k} exp(cos_sim(q, c'))

        The softmax ensures weights sum to 1, allowing proper
        distribution of mastery updates across linked cards.

        Args:
            similarities: Dictionary of card_id -> similarity score

        Returns:
            Dictionary of card_id -> contribution weight
        """
        if not similarities:
            return {}

        # Compute exp(similarity) for each card
        # Use a stable softmax (subtract max for numerical stability)
        max_sim = max(similarities.values())
        exp_sims = {
            card_id: np.exp(sim - max_sim) for card_id, sim in similarities.items()
        }

        # Sum for normalization
        z = sum(exp_sims.values())

        if z == 0:
            # Edge case: uniform weights
            uniform = 1.0 / len(similarities)
            return {card_id: uniform for card_id in similarities}

        # Softmax weights
        return {card_id: exp_sim / z for card_id, exp_sim in exp_sims.items()}

    def link_question(
        self,
        question_id: str,
        question_embedding: NDArray[np.float64],
        card_embeddings: dict[str, NDArray[np.float64]],
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> LinkResult:
        """Link a question to its most relevant cards.

        Args:
            question_id: Unique identifier for the question
            question_embedding: Vector embedding of the question
            card_embeddings: Dictionary of card_id -> embedding
            top_k: Override for number of cards to link
            min_similarity: Override for minimum similarity threshold

        Returns:
            LinkResult with linked cards and contribution weights
        """
        top_k = top_k if top_k is not None else self.config.top_k
        min_similarity = (
            min_similarity
            if min_similarity is not None
            else self.config.min_similarity
        )

        # Calculate similarity to all cards
        all_similarities: dict[str, float] = {}
        for card_id, card_emb in card_embeddings.items():
            sim = self.cosine_similarity(question_embedding, card_emb)
            all_similarities[card_id] = sim

        # Sort by similarity (descending)
        sorted_cards = sorted(
            all_similarities.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Take top-k that meet minimum threshold
        top_k_cards = [
            (card_id, sim)
            for card_id, sim in sorted_cards[:top_k]
            if sim >= min_similarity
        ]

        # Build result
        linked_cards = [card_id for card_id, _ in top_k_cards]
        similarities = {card_id: sim for card_id, sim in top_k_cards}
        contribution_weights = self.compute_contribution_weights(similarities)

        return LinkResult(
            question_id=question_id,
            linked_cards=linked_cards,
            similarities=similarities,
            contribution_weights=contribution_weights,
        )

    def link_questions_batch(
        self,
        questions: list[tuple[str, NDArray[np.float64]]],
        card_embeddings: dict[str, NDArray[np.float64]],
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> list[LinkResult]:
        """Link multiple questions to cards.

        Efficient batch processing for initial question linking.

        Args:
            questions: List of (question_id, embedding) tuples
            card_embeddings: Dictionary of card embeddings
            top_k: Override for number of cards to link
            min_similarity: Override for minimum similarity threshold

        Returns:
            List of LinkResults
        """
        return [
            self.link_question(q_id, q_emb, card_embeddings, top_k, min_similarity)
            for q_id, q_emb in questions
        ]

    def find_most_relevant_card(
        self,
        question_embedding: NDArray[np.float64],
        card_embeddings: dict[str, NDArray[np.float64]],
    ) -> Optional[tuple[str, float]]:
        """Find the single most relevant card for a question.

        Useful for quick lookups or when only the primary
        card is needed.

        Args:
            question_embedding: Question vector
            card_embeddings: Dictionary of card embeddings

        Returns:
            Tuple of (card_id, similarity) or None if no cards
        """
        if not card_embeddings:
            return None

        best_card: Optional[str] = None
        best_sim = float("-inf")

        for card_id, card_emb in card_embeddings.items():
            sim = self.cosine_similarity(question_embedding, card_emb)
            if sim > best_sim:
                best_sim = sim
                best_card = card_id

        if best_card is None:
            return None

        return (best_card, best_sim)

    def compute_similarity_matrix(
        self,
        question_embeddings: dict[str, NDArray[np.float64]],
        card_embeddings: dict[str, NDArray[np.float64]],
    ) -> dict[str, dict[str, float]]:
        """Compute full similarity matrix between questions and cards.

        Useful for analysis and visualization.

        Args:
            question_embeddings: Dictionary of question embeddings
            card_embeddings: Dictionary of card embeddings

        Returns:
            Nested dict: question_id -> card_id -> similarity
        """
        matrix: dict[str, dict[str, float]] = {}

        for q_id, q_emb in question_embeddings.items():
            matrix[q_id] = {}
            for c_id, c_emb in card_embeddings.items():
                matrix[q_id][c_id] = self.cosine_similarity(q_emb, c_emb)

        return matrix

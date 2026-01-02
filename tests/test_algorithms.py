"""Tests for core adaptive learning algorithms."""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone

from adaptive_learning_core.models.card import Card, CardState, LearnerStereotype
from adaptive_learning_core.models.question import Question, BloomLevel
from adaptive_learning_core.algorithms.mastery_update import MasteryUpdater, MasteryUpdateConfig
from adaptive_learning_core.algorithms.knowledge_decay import KnowledgeDecay, DecayConfig
from adaptive_learning_core.algorithms.card_selector import CardSelector, CardSelectionConfig
from adaptive_learning_core.algorithms.question_linker import QuestionLinker, QuestionLinkConfig


class TestMasteryUpdate:
    """Tests for mastery update algorithm (Equation 3)."""

    @pytest.fixture
    def updater(self):
        return MasteryUpdater(config=MasteryUpdateConfig(
            alpha_slow=0.05,
            alpha_medium=0.10,
            alpha_fast=0.15,
        ))

    @pytest.fixture
    def card_state(self):
        return CardState(
            card_id="card1",
            learner_id="user1",
            mastery_score=0.5,
        )

    @pytest.fixture
    def question(self):
        return Question(
            id="q1",
            module_id="mod1",
            stem="What is X?",
            correct_answer="A",
            distractors=["B", "C", "D"],
            explanation="Because...",
            bloom_level=BloomLevel.REMEMBER,
            linked_card_ids=["card1"],
            card_contribution_weights={"card1": 1.0},
        )

    def test_correct_answer_increases_mastery(self, updater, card_state, question):
        """Correct answer should increase mastery score."""
        result = updater.update_single(
            card_state=card_state,
            question=question,
            is_correct=True,
            pace="medium",
        )
        assert result.new_score > result.old_score
        assert result.was_correct is True
        assert result.delta == 1.0

    def test_incorrect_answer_decreases_mastery(self, updater, card_state, question):
        """Incorrect answer should decrease mastery score."""
        result = updater.update_single(
            card_state=card_state,
            question=question,
            is_correct=False,
            pace="medium",
        )
        assert result.new_score < result.old_score
        assert result.was_correct is False
        assert result.delta == -1.0

    def test_mastery_clamped_to_range(self, updater, question):
        """Mastery should stay in [0, 1] range."""
        # Test upper bound
        high_state = CardState(card_id="card1", learner_id="user1", mastery_score=0.99)
        result = updater.update_single(high_state, question, is_correct=True, pace="fast")
        assert result.new_score <= 1.0

        # Test lower bound
        low_state = CardState(card_id="card1", learner_id="user1", mastery_score=0.01)
        result = updater.update_single(low_state, question, is_correct=False, pace="fast")
        assert result.new_score >= 0.0

    def test_pace_affects_update_rate(self, updater, question):
        """Different paces should have different update magnitudes."""
        results = {}
        for pace in ["slow", "medium", "fast"]:
            state = CardState(card_id="card1", learner_id="user1", mastery_score=0.5)
            result = updater.update_single(state, question, is_correct=True, pace=pace)
            results[pace] = result.score_change

        assert results["slow"] < results["medium"] < results["fast"]


class TestKnowledgeDecay:
    """Tests for knowledge decay algorithm (Equations 7-8)."""

    @pytest.fixture
    def decay(self):
        return KnowledgeDecay(config=DecayConfig(
            beta=0.1,
            threshold_hours=4.0,
            same_day_multiplier=1.0,
            different_day_multiplier=2.0,
        ))

    def test_no_decay_within_threshold(self, decay):
        """No decay should be applied within threshold hours."""
        now = datetime.now(timezone.utc)
        two_hours_ago = now - timedelta(hours=2)

        state = CardState(
            card_id="card1",
            learner_id="user1",
            mastery_score=0.8,
            last_seen_at=two_hours_ago,
        )

        result = decay.apply_decay(state, current_time=now, in_place=False)
        assert result.new_score == result.old_score
        assert result.decay_applied is False
        assert result.decay_multiplier == 0.0

    def test_decay_applied_after_threshold(self, decay):
        """Decay should be applied after threshold hours."""
        now = datetime.now(timezone.utc)
        two_days_ago = now - timedelta(hours=48)

        state = CardState(
            card_id="card1",
            learner_id="user1",
            mastery_score=0.8,
            last_seen_at=two_days_ago,
        )

        result = decay.apply_decay(state, current_time=now, in_place=False)
        assert result.new_score < result.old_score
        assert result.decay_applied is True
        assert result.decay_multiplier == 2.0  # Different day

    def test_no_decay_if_never_seen(self, decay):
        """No decay for cards that have never been seen."""
        state = CardState(
            card_id="card1",
            learner_id="user1",
            mastery_score=0.0,
            last_seen_at=None,
        )

        result = decay.apply_decay(state, in_place=False)
        assert result.decay_applied is False

    def test_in_place_modification(self, decay):
        """In-place mode should modify the original state."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(hours=24)

        state = CardState(
            card_id="card1",
            learner_id="user1",
            mastery_score=0.8,
            last_seen_at=yesterday,
        )
        original_score = state.mastery_score

        decay.apply_decay(state, current_time=now, in_place=True)
        assert state.mastery_score < original_score


class TestCardSelector:
    """Tests for card selection algorithm (Equations 4-5)."""

    @pytest.fixture
    def selector(self):
        return CardSelector(config=CardSelectionConfig(
            gamma=0.1,
            delta=1.5,
            score_threshold=0.8,
            default_selection_size=5,
        ))

    @pytest.fixture
    def card_states(self):
        return [
            CardState(card_id="c1", learner_id="u1", mastery_score=0.1),  # Low mastery
            CardState(card_id="c2", learner_id="u1", mastery_score=0.5),  # Medium
            CardState(card_id="c3", learner_id="u1", mastery_score=0.9),  # High mastery
        ]

    def test_lower_mastery_higher_probability(self, selector, card_states):
        """Cards with lower mastery should have higher selection probability."""
        probs = selector.calculate_probabilities(card_states)

        assert probs["c1"] > probs["c2"] > probs["c3"]

    def test_probabilities_sum_to_one(self, selector, card_states):
        """Selection probabilities should sum to 1."""
        probs = selector.calculate_probabilities(card_states)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_gamma_boosting(self, selector):
        """Gamma should be boosted when quiz score > threshold."""
        initial_gamma = 0.1
        boosted = selector.maybe_boost_gamma(initial_gamma, quiz_score=0.9)
        assert boosted == initial_gamma * selector.config.delta

        not_boosted = selector.maybe_boost_gamma(initial_gamma, quiz_score=0.7)
        assert not_boosted == initial_gamma

    def test_selection_respects_n(self, selector, card_states):
        """Should select exactly n cards."""
        result = selector.select_cards(card_states, n=2, random_state=42)
        assert len(result.selected_card_ids) == 2

    def test_selection_is_deterministic_with_seed(self, selector, card_states):
        """Same random seed should give same selection."""
        result1 = selector.select_cards(card_states, n=2, random_state=42)
        result2 = selector.select_cards(card_states, n=2, random_state=42)
        assert result1.selected_card_ids == result2.selected_card_ids


class TestQuestionLinker:
    """Tests for question-card linking (Equations 1-2)."""

    @pytest.fixture
    def linker(self):
        return QuestionLinker(config=QuestionLinkConfig(
            top_k=3,
            min_similarity=0.3,
        ))

    @pytest.fixture
    def embeddings(self):
        return {
            "card1": np.array([1.0, 0.0, 0.0]),
            "card2": np.array([0.0, 1.0, 0.0]),
            "card3": np.array([0.0, 0.0, 1.0]),
        }

    def test_cosine_similarity_identical(self, linker):
        """Identical vectors should have similarity 1."""
        v = np.array([1.0, 2.0, 3.0])
        sim = linker.cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self, linker):
        """Orthogonal vectors should have similarity 0."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        sim = linker.cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_softmax_contribution_weights(self, linker):
        """Contribution weights should sum to 1."""
        similarities = {"c1": 0.9, "c2": 0.5, "c3": 0.2}
        weights = linker.compute_contribution_weights(similarities)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_linking_finds_similar_card(self, linker, embeddings):
        """Should link question to most similar card."""
        question_emb = np.array([0.95, 0.05, 0.0])  # Similar to card1

        result = linker.link_question(
            question_id="q1",
            question_embedding=question_emb,
            card_embeddings=embeddings,
        )

        assert result.primary_card == "card1"
        assert result.contribution_weights["card1"] > result.contribution_weights.get("card2", 0)

    def test_min_similarity_threshold(self, linker, embeddings):
        """Cards below min similarity should not be linked."""
        # Query orthogonal to all cards
        question_emb = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)

        linker_strict = QuestionLinker(config=QuestionLinkConfig(
            top_k=3,
            min_similarity=0.9,  # Very high threshold
        ))

        result = linker_strict.link_question(
            question_id="q1",
            question_embedding=question_emb,
            card_embeddings=embeddings,
        )

        # Should link to fewer cards due to high threshold
        assert len(result.linked_cards) < 3

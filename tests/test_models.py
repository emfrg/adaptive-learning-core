"""Tests for data models."""

import pytest
from datetime import datetime, timezone

from adaptive_learning_core.models.card import Card, CardState, LearnerStereotype
from adaptive_learning_core.models.question import Question, BloomLevel
from adaptive_learning_core.models.quiz import Quiz, QuizAnswer, QuizResult
from adaptive_learning_core.models.module import Module, ModuleProgress, ModuleStatus
from adaptive_learning_core.models.learner import LearnerProfile, PacePreference, LearningGoal


class TestLearnerStereotype:
    """Tests for LearnerStereotype enum."""

    @pytest.mark.parametrize("score,expected", [
        (0.0, LearnerStereotype.NOVICE),
        (0.15, LearnerStereotype.NOVICE),
        (0.2, LearnerStereotype.BEGINNER),
        (0.35, LearnerStereotype.BEGINNER),
        (0.4, LearnerStereotype.INTERMEDIATE),
        (0.55, LearnerStereotype.INTERMEDIATE),
        (0.6, LearnerStereotype.ADVANCED),
        (0.75, LearnerStereotype.ADVANCED),
        (0.8, LearnerStereotype.EXPERT),
        (1.0, LearnerStereotype.EXPERT),
    ])
    def test_from_mastery_score(self, score, expected):
        """Mastery score should map to correct stereotype."""
        result = LearnerStereotype.from_mastery_score(score)
        assert result == expected

    def test_stereotype_levels(self):
        """Stereotypes should have correct numeric levels."""
        assert LearnerStereotype.NOVICE.level == 0
        assert LearnerStereotype.BEGINNER.level == 1
        assert LearnerStereotype.INTERMEDIATE.level == 2
        assert LearnerStereotype.ADVANCED.level == 3
        assert LearnerStereotype.EXPERT.level == 4


class TestCard:
    """Tests for Card model."""

    def test_card_creation(self):
        """Card should be created with required fields."""
        card = Card(
            id="card1",
            module_id="mod1",
            text="This is the card content.",
        )
        assert card.id == "card1"
        assert card.module_id == "mod1"
        assert card.text == "This is the card content."

    def test_card_equality(self):
        """Cards with same ID should be equal."""
        card1 = Card(id="card1", module_id="mod1", text="Content 1")
        card2 = Card(id="card1", module_id="mod1", text="Content 2")
        assert card1 == card2

    def test_card_hash(self):
        """Cards should be hashable by ID."""
        card1 = Card(id="card1", module_id="mod1", text="Content")
        card2 = Card(id="card2", module_id="mod1", text="Content")

        card_set = {card1, card2}
        assert len(card_set) == 2


class TestCardState:
    """Tests for CardState model."""

    def test_stereotype_property(self):
        """CardState should compute stereotype from mastery."""
        state = CardState(card_id="c1", learner_id="u1", mastery_score=0.5)
        assert state.stereotype == LearnerStereotype.INTERMEDIATE

    def test_is_unseen(self):
        """is_unseen should be True for new cards."""
        unseen = CardState(card_id="c1", learner_id="u1", mastery_score=0.0, times_seen=0)
        assert unseen.is_unseen is True

        seen = CardState(card_id="c1", learner_id="u1", mastery_score=0.3, times_seen=1)
        assert seen.is_unseen is False

    def test_accuracy_calculation(self):
        """Accuracy should be computed correctly."""
        state = CardState(
            card_id="c1",
            learner_id="u1",
            times_correct=7,
            times_incorrect=3,
        )
        assert state.accuracy == 0.7

    def test_accuracy_zero_attempts(self):
        """Accuracy should be 0 with no attempts."""
        state = CardState(card_id="c1", learner_id="u1")
        assert state.accuracy == 0.0


class TestBloomLevel:
    """Tests for BloomLevel enum."""

    def test_bloom_ordering(self):
        """Bloom levels should be ordered by cognitive complexity."""
        assert BloomLevel.REMEMBER < BloomLevel.UNDERSTAND
        assert BloomLevel.UNDERSTAND < BloomLevel.APPLY
        assert BloomLevel.APPLY < BloomLevel.ANALYZE
        assert BloomLevel.ANALYZE < BloomLevel.EVALUATE
        assert BloomLevel.EVALUATE < BloomLevel.CREATE

    @pytest.mark.parametrize("level,expected_stereotype", [
        (BloomLevel.REMEMBER, 0),    # Novice
        (BloomLevel.UNDERSTAND, 1),  # Beginner
        (BloomLevel.APPLY, 2),       # Intermediate
        (BloomLevel.ANALYZE, 3),     # Advanced
        (BloomLevel.EVALUATE, 4),    # Expert
        (BloomLevel.CREATE, 4),      # Expert
    ])
    def test_to_stereotype_level(self, level, expected_stereotype):
        """Bloom level should map to correct stereotype level."""
        assert level.to_stereotype_level() == expected_stereotype

    def test_action_verbs(self):
        """Each level should have action verbs."""
        for level in BloomLevel:
            verbs = level.action_verbs
            assert isinstance(verbs, list)
            assert len(verbs) > 0


class TestQuestion:
    """Tests for Question model."""

    @pytest.fixture
    def question(self):
        return Question(
            id="q1",
            module_id="mod1",
            stem="What is photosynthesis?",
            correct_answer="Converting light to chemical energy",
            distractors=["Breathing", "Eating", "Sleeping"],
            explanation="Photosynthesis is the process...",
            bloom_level=BloomLevel.UNDERSTAND,
            linked_card_ids=["c1", "c2"],
            card_contribution_weights={"c1": 0.7, "c2": 0.3},
        )

    def test_question_creation(self, question):
        """Question should be created with required fields."""
        assert question.id == "q1"
        assert question.bloom_level == BloomLevel.UNDERSTAND
        assert len(question.distractors) == 3

    def test_all_options(self, question):
        """all_options should include correct answer and distractors."""
        options = question.all_options
        assert len(options) == 4
        assert question.correct_answer in options
        for d in question.distractors:
            assert d in options

    def test_difficulty_level(self, question):
        """difficulty_level should map from bloom level."""
        assert question.difficulty_level == 1  # UNDERSTAND -> Beginner

    def test_get_contribution_weight(self, question):
        """Should return correct contribution weight."""
        assert question.get_contribution_weight("c1") == 0.7
        assert question.get_contribution_weight("c2") == 0.3
        assert question.get_contribution_weight("c3") == 0.0  # Not linked


class TestQuiz:
    """Tests for Quiz and QuizResult models."""

    def test_quiz_answer(self):
        """QuizAnswer should track correctness."""
        answer = QuizAnswer(
            question_id="q1",
            selected_answer="A",
            is_correct=True,
            time_spent_seconds=15.5,
        )
        assert answer.is_correct is True
        assert answer.time_spent_seconds == 15.5

    def test_quiz_result_score(self):
        """QuizResult should compute score correctly."""
        from datetime import datetime

        answers = [
            QuizAnswer(question_id="q1", selected_answer="A", is_correct=True),
            QuizAnswer(question_id="q2", selected_answer="B", is_correct=True),
            QuizAnswer(question_id="q3", selected_answer="C", is_correct=False),
            QuizAnswer(question_id="q4", selected_answer="D", is_correct=True),
        ]
        result = QuizResult(
            id="result1",
            quiz_id="quiz1",
            learner_id="user1",
            module_id="mod1",
            answers=answers,
            started_at=datetime.utcnow(),
        )
        assert result.score == 0.75
        assert result.num_correct == 3
        assert result.num_questions == 4


class TestModule:
    """Tests for Module and ModuleProgress models."""

    def test_module_creation(self):
        """Module should be created with required fields."""
        module = Module(
            id="mod1",
            name="Biology 101",
            description="Introduction to biology",
            card_ids=["c1", "c2", "c3"],
        )
        assert module.id == "mod1"
        assert module.name == "Biology 101"
        assert len(module.card_ids) == 3

    def test_module_progress(self):
        """ModuleProgress should track completion."""
        progress = ModuleProgress(
            module_id="mod1",
            learner_id="user1",
            num_cards_learned=8,
            num_cards_total=10,
            status=ModuleStatus.IN_PROGRESS,
        )
        assert progress.progress_percentage == 80.0
        assert progress.status == ModuleStatus.IN_PROGRESS


class TestLearnerProfile:
    """Tests for LearnerProfile model."""

    def test_learner_creation(self):
        """LearnerProfile should be created with preferences."""
        learner = LearnerProfile(
            id="user1",
            default_pace=PacePreference.MEDIUM,
            default_goal=LearningGoal.COMPREHEND,
        )
        assert learner.id == "user1"
        assert learner.default_pace == PacePreference.MEDIUM
        assert learner.default_goal == LearningGoal.COMPREHEND

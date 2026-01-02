"""Tests for LangGraph agents."""

import pytest
from datetime import datetime, timezone, timedelta

from adaptive_learning_core.agents.state import LearningState, QuestionGenState
from adaptive_learning_core.agents.nodes import (
    initialize_session,
    apply_decay,
    select_cards,
    check_progress,
    should_continue_learning,
)
from adaptive_learning_core.agents.learning_agent import (
    LearningAgent,
    create_learning_graph,
)
from adaptive_learning_core.models.card import Card, CardState
from adaptive_learning_core.models.question import Question, BloomLevel


class TestLearningState:
    """Tests for LearningState TypedDict."""

    def test_state_creation(self):
        """Should create valid state dict."""
        state = LearningState(
            session_id="session_123",
            learner_id="user_1",
            module_id="mod_1",
            cycle_count=0,
            should_continue=True,
        )
        assert state["session_id"] == "session_123"
        assert state["cycle_count"] == 0


class TestNodes:
    """Tests for LangGraph node functions."""

    @pytest.fixture
    def cards(self):
        return [
            Card(id="c1", module_id="mod1", text="Card 1 content"),
            Card(id="c2", module_id="mod1", text="Card 2 content"),
            Card(id="c3", module_id="mod1", text="Card 3 content"),
        ]

    @pytest.fixture
    def card_states(self):
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(hours=24)
        return {
            "c1": CardState(card_id="c1", learner_id="u1", mastery_score=0.3, last_seen_at=yesterday),
            "c2": CardState(card_id="c2", learner_id="u1", mastery_score=0.7, last_seen_at=yesterday),
            "c3": CardState(card_id="c3", learner_id="u1", mastery_score=0.1, last_seen_at=yesterday),
        }

    @pytest.fixture
    def initial_state(self, cards, card_states):
        return LearningState(
            cards=cards,
            card_states=card_states,
            learner_id="user_1",
            module_id="mod_1",
            pace="medium",
            cards_per_cycle=2,
            questions_per_quiz=3,
            max_cycles=5,
            exam_threshold=0.8,
        )

    def test_initialize_session(self, initial_state):
        """Initialize node should set up session state."""
        result = initialize_session(initial_state)

        assert "session_id" in result
        assert result["session_id"].startswith("session_")
        assert result["cycle_count"] == 0
        assert result["decay_applied"] is False
        assert result["should_continue"] is True

    def test_apply_decay(self, initial_state):
        """Decay node should apply time-based decay."""
        result = apply_decay(initial_state)

        assert result["decay_applied"] is True
        assert "decay_results" in result
        assert len(result["decay_results"]) == 3  # 3 cards

    def test_select_cards(self, initial_state):
        """Select cards node should choose cards for study."""
        result = select_cards(initial_state)

        assert "selected_card_ids" in result
        assert len(result["selected_card_ids"]) == 2  # cards_per_cycle

    def test_check_progress_continue(self, initial_state):
        """Check progress should continue when below threshold."""
        state = {**initial_state, "cycle_count": 0}
        result = check_progress(state)

        assert result["cycle_count"] == 1
        assert result["should_continue"] is True
        assert result["exam_recommended"] is False

    def test_check_progress_max_cycles(self, initial_state):
        """Check progress should stop at max cycles."""
        state = {**initial_state, "cycle_count": 4}  # max_cycles - 1
        result = check_progress(state)

        assert result["cycle_count"] == 5
        assert result["should_continue"] is False

    def test_check_progress_exam_recommended(self):
        """Should recommend exam when most cards above threshold."""
        high_mastery_states = {
            "c1": CardState(card_id="c1", learner_id="u1", mastery_score=0.9),
            "c2": CardState(card_id="c2", learner_id="u1", mastery_score=0.85),
            "c3": CardState(card_id="c3", learner_id="u1", mastery_score=0.82),
        }
        state = LearningState(
            card_states=high_mastery_states,
            cycle_count=2,
            max_cycles=10,
            exam_threshold=0.8,
        )
        result = check_progress(state)

        assert result["exam_recommended"] is True
        assert result["should_continue"] is False

    def test_should_continue_routing(self):
        """Routing function should return correct next node."""
        continue_state = LearningState(should_continue=True)
        assert should_continue_learning(continue_state) == "select_cards"

        stop_state = LearningState(should_continue=False)
        assert should_continue_learning(stop_state) == "end_session"


class TestLearningGraph:
    """Tests for the complete learning graph."""

    def test_graph_creation(self):
        """Graph should be created with all nodes."""
        graph = create_learning_graph()

        expected_nodes = [
            "initialize",
            "apply_decay",
            "select_cards",
            "select_questions",
            "await_answers",
            "process_answers",
            "update_mastery",
            "check_progress",
            "end_session",
        ]

        for node in expected_nodes:
            assert node in graph.nodes

    def test_graph_edges(self):
        """Graph should have correct edges."""
        graph = create_learning_graph()

        # Check that key edges exist
        edges = graph.edges
        # The graph should have edges connecting the nodes
        assert len(edges) > 0


class TestLearningAgent:
    """Tests for LearningAgent class."""

    @pytest.fixture
    def cards(self):
        return [
            Card(id="c1", module_id="mod1", text="Content 1"),
            Card(id="c2", module_id="mod1", text="Content 2"),
        ]

    @pytest.fixture
    def card_states(self):
        return {
            "c1": CardState(card_id="c1", learner_id="u1", mastery_score=0.3),
            "c2": CardState(card_id="c2", learner_id="u1", mastery_score=0.5),
        }

    @pytest.fixture
    def questions(self):
        return [
            Question(
                id="q1",
                module_id="mod1",
                stem="Question 1?",
                correct_answer="A",
                distractors=["B", "C", "D"],
                explanation="Because...",
                bloom_level=BloomLevel.REMEMBER,
                linked_card_ids=["c1"],
            ),
            Question(
                id="q2",
                module_id="mod1",
                stem="Question 2?",
                correct_answer="X",
                distractors=["Y", "Z", "W"],
                explanation="Because...",
                bloom_level=BloomLevel.UNDERSTAND,
                linked_card_ids=["c2"],
            ),
        ]

    def test_agent_initialization(self):
        """Agent should initialize without LLM."""
        agent = LearningAgent()
        assert agent.llm is None
        assert agent.compiled is not None

    def test_create_session(self, cards, card_states, questions):
        """Should create valid session state."""
        agent = LearningAgent()
        state = agent.create_session(
            cards=cards,
            card_states=card_states,
            questions=questions,
            learner_id="user_1",
            module_id="mod_1",
        )

        assert state["learner_id"] == "user_1"
        assert state["module_id"] == "mod_1"
        assert len(state["cards"]) == 2
        assert len(state["questions"]) == 2

    def test_get_progress(self, cards, card_states, questions):
        """Should compute progress summary."""
        agent = LearningAgent()
        state = agent.create_session(
            cards=cards,
            card_states=card_states,
            questions=questions,
            learner_id="user_1",
        )
        state["session_id"] = "test_session"
        state["cycle_count"] = 3
        state["last_quiz_score"] = 0.8

        progress = agent.get_progress(state)

        assert progress["session_id"] == "test_session"
        assert progress["cycle_count"] == 3
        assert progress["last_quiz_score"] == 0.8
        assert progress["total_cards"] == 2

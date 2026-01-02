"""LangGraph-based Learning Session Agent.

This agent orchestrates the complete adaptive learning flow as a state machine:

    ┌─────────────────────────────────────────────────────────────┐
    │                    LEARNING SESSION GRAPH                     │
    └─────────────────────────────────────────────────────────────┘

    START
      │
      ▼
    ┌─────────────┐
    │ initialize  │
    └─────┬───────┘
          │
          ▼
    ┌─────────────┐
    │ apply_decay │ ◄── Eq. 7-8: M(c) = M(c) × e^{-βλτ}
    └─────┬───────┘
          │
          ▼
    ┌─────────────┐
    │select_cards │ ◄── Eq. 4-5: P(c) = (1 - M(c) + γ) / Z
    └─────┬───────┘
          │
          ▼
    ┌──────────────────┐
    │ select_questions │ ◄── Rules 2 & 3
    └─────┬────────────┘
          │
          ▼
    ┌───────────────┐
    │ await_answers │ ◄── Human-in-the-loop
    └─────┬─────────┘
          │
          ▼
    ┌──────────────────┐
    │ process_answers  │
    └─────┬────────────┘
          │
          ▼
    ┌────────────────┐
    │ update_mastery │ ◄── Eq. 3: M(c) = M(c) + α × C_{c,q} × δ
    └─────┬──────────┘
          │
          ▼
    ┌────────────────┐
    │ check_progress │ ◄── Rule 4: Exam recommendation
    └─────┬──────────┘
          │
          ├──────────────────┐
          │ should_continue  │
          ▼                  ▼
    [select_cards]      ┌─────────────┐
          ▲             │ end_session │
          │             └─────┬───────┘
          │                   │
          └───────────────────┘
                              │
                              ▼
                            END
"""

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from adaptive_learning_core.agents.state import LearningState
from adaptive_learning_core.agents.nodes import (
    initialize_session,
    apply_decay,
    select_cards,
    select_questions,
    await_answers,
    process_answers,
    update_mastery,
    check_progress,
    end_session,
    should_continue_learning,
)
from adaptive_learning_core.models.card import Card, CardState
from adaptive_learning_core.models.question import Question


def create_learning_graph() -> StateGraph:
    """Create the learning session state graph.

    Returns:
        Compiled StateGraph for the learning session
    """
    # Create the graph
    graph = StateGraph(LearningState)

    # Add nodes
    graph.add_node("initialize", initialize_session)
    graph.add_node("apply_decay", apply_decay)
    graph.add_node("select_cards", select_cards)
    graph.add_node("select_questions", select_questions)
    graph.add_node("await_answers", await_answers)
    graph.add_node("process_answers", process_answers)
    graph.add_node("update_mastery", update_mastery)
    graph.add_node("check_progress", check_progress)
    graph.add_node("end_session", end_session)

    # Add edges
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "apply_decay")
    graph.add_edge("apply_decay", "select_cards")
    graph.add_edge("select_cards", "select_questions")
    graph.add_edge("select_questions", "await_answers")
    graph.add_edge("await_answers", "process_answers")
    graph.add_edge("process_answers", "update_mastery")
    graph.add_edge("update_mastery", "check_progress")

    # Conditional edge: continue or end
    graph.add_conditional_edges(
        "check_progress",
        should_continue_learning,
        {
            "select_cards": "select_cards",
            "end_session": "end_session",
        }
    )

    graph.add_edge("end_session", END)

    return graph


class LearningAgent:
    """High-level agent for managing learning sessions.

    Uses LangGraph for state management and LangChain for LLM interactions.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = LearningAgent(llm=llm)
        >>>
        >>> # Initialize session
        >>> state = agent.create_session(
        ...     cards=cards,
        ...     card_states=card_states,
        ...     questions=questions,
        ...     learner_id="user_123",
        ... )
        >>>
        >>> # Run until human input needed
        >>> state = agent.run_until_input(state)
        >>>
        >>> # Submit answers
        >>> state = agent.submit_answers(state, answers)
        >>>
        >>> # Continue
        >>> state = agent.run_until_input(state)
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        checkpointer: Optional[Any] = None,
    ):
        """Initialize the learning agent.

        Args:
            llm: LangChain chat model for LLM operations
            checkpointer: LangGraph checkpointer for state persistence
        """
        self.llm = llm
        self.checkpointer = checkpointer or MemorySaver()

        # Create and compile the graph
        self.graph = create_learning_graph()
        self.compiled = self.graph.compile(checkpointer=self.checkpointer)

    def create_session(
        self,
        cards: list[Card],
        card_states: dict[str, CardState],
        questions: list[Question],
        learner_id: str,
        module_id: str = "default",
        pace: str = "medium",
        target_mastery_level: int = 3,
        cards_per_cycle: int = 5,
        questions_per_quiz: int = 5,
        max_cycles: int = 10,
        exam_threshold: float = 0.8,
    ) -> LearningState:
        """Create initial state for a learning session.

        Args:
            cards: List of cards in the module
            card_states: Current mastery states for cards
            questions: Available questions
            learner_id: Learner identifier
            module_id: Module identifier
            pace: Learning pace (slow/medium/fast)
            target_mastery_level: Target stereotype level (0-4)
            cards_per_cycle: Cards per learning cycle
            questions_per_quiz: Questions per quiz
            max_cycles: Maximum cycles per session
            exam_threshold: Mastery threshold for exam recommendation

        Returns:
            Initial LearningState
        """
        return LearningState(
            cards=cards,
            card_states=card_states,
            questions=questions,
            learner_id=learner_id,
            module_id=module_id,
            pace=pace,
            target_mastery_level=target_mastery_level,
            cards_per_cycle=cards_per_cycle,
            questions_per_quiz=questions_per_quiz,
            max_cycles=max_cycles,
            exam_threshold=exam_threshold,
            cycle_count=0,
            selected_card_ids=[],
            quiz_questions=[],
            quiz_answers=[],
            last_quiz_score=None,
            decay_applied=False,
            exam_recommended=False,
            should_continue=True,
            session_complete=False,
            messages=[],
        )

    def run(
        self,
        initial_state: LearningState,
        thread_id: str = "default",
    ) -> LearningState:
        """Run the learning session graph to completion.

        Note: This will block at await_answers node waiting for input.
        For interactive use, use run_until_input() instead.

        Args:
            initial_state: Starting state
            thread_id: Thread ID for checkpointing

        Returns:
            Final state after execution
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self.compiled.invoke(initial_state, config)
        return result

    def run_until_input(
        self,
        state: LearningState,
        thread_id: str = "default",
    ) -> LearningState:
        """Run the graph until human input is needed.

        Stops at the await_answers node.

        Args:
            state: Current state
            thread_id: Thread ID for checkpointing

        Returns:
            State after running to await_answers
        """
        config = {"configurable": {"thread_id": thread_id}}

        # Run through the graph
        for event in self.compiled.stream(state, config):
            # Get the last state
            for node_name, node_output in event.items():
                state = {**state, **node_output}

            # Stop at await_answers
            if "await_answers" in event:
                break

        return state

    def submit_answers(
        self,
        state: LearningState,
        answers: list[dict[str, Any]],
        thread_id: str = "default",
    ) -> LearningState:
        """Submit quiz answers and continue processing.

        Args:
            state: Current state (at await_answers)
            answers: List of answer dicts with question_id and selected_answer
            thread_id: Thread ID for checkpointing

        Returns:
            Updated state after processing answers
        """
        from adaptive_learning_core.models.quiz import QuizAnswer

        # Convert to QuizAnswer objects
        quiz_answers = []
        for ans in answers:
            # Find the question
            question = next(
                (q for q in state.get("quiz_questions", [])
                 if q.id == ans["question_id"]),
                None
            )
            if question:
                quiz_answers.append(QuizAnswer(
                    question_id=ans["question_id"],
                    selected_answer=ans["selected_answer"],
                    is_correct=ans["selected_answer"] == question.correct_answer,
                ))

        # Update state with answers
        state = {**state, "quiz_answers": quiz_answers}

        # Continue from process_answers
        config = {"configurable": {"thread_id": thread_id}}

        # Resume execution
        for event in self.compiled.stream(state, config, stream_mode="values"):
            state = event

        return state

    def get_quiz_questions(self, state: LearningState) -> list[Question]:
        """Get the current quiz questions from state.

        Args:
            state: Current state

        Returns:
            List of questions for the quiz
        """
        return state.get("quiz_questions", [])

    def get_progress(self, state: LearningState) -> dict[str, Any]:
        """Get progress summary from state.

        Args:
            state: Current state

        Returns:
            Progress summary dict
        """
        card_states = state.get("card_states", {})
        exam_threshold = state.get("exam_threshold", 0.8)

        total_cards = len(card_states)
        above_threshold = sum(
            1 for s in card_states.values()
            if s.mastery_score >= exam_threshold
        )

        return {
            "session_id": state.get("session_id"),
            "cycle_count": state.get("cycle_count", 0),
            "last_quiz_score": state.get("last_quiz_score"),
            "total_cards": total_cards,
            "cards_above_threshold": above_threshold,
            "progress_percentage": above_threshold / total_cards * 100 if total_cards else 0,
            "exam_recommended": state.get("exam_recommended", False),
            "session_complete": state.get("session_complete", False),
        }

    def get_messages(self, state: LearningState) -> list[Any]:
        """Get all messages from the session.

        Args:
            state: Current state

        Returns:
            List of LangChain messages
        """
        return state.get("messages", [])

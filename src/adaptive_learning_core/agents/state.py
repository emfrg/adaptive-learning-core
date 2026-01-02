"""LangGraph state definitions for adaptive learning agents.

States are TypedDicts that flow through the graph nodes.
They carry all data needed for the learning session.
"""

from typing import Annotated, Any, Literal, Optional, TypedDict
from operator import add

from adaptive_learning_core.models.card import Card, CardState
from adaptive_learning_core.models.question import Question
from adaptive_learning_core.models.quiz import QuizAnswer


class LearningState(TypedDict, total=False):
    """State for the learning session agent.

    This state flows through the LangGraph nodes:
    apply_decay -> select_cards -> generate_quiz -> await_answers ->
    update_mastery -> check_progress -> (loop or end)
    """

    # Session configuration
    session_id: str
    learner_id: str
    module_id: str
    pace: Literal["slow", "medium", "fast"]
    target_mastery_level: int
    cards_per_cycle: int
    questions_per_quiz: int
    max_cycles: int
    exam_threshold: float

    # Card data
    cards: list[Card]
    card_states: dict[str, CardState]

    # Question pool
    questions: list[Question]
    card_embeddings: dict[str, list[float]]
    question_embeddings: dict[str, list[float]]

    # Current cycle state
    cycle_count: int
    selected_card_ids: list[str]
    quiz_questions: list[Question]
    quiz_answers: list[QuizAnswer]
    last_quiz_score: Optional[float]

    # Decay tracking
    decay_applied: bool
    decay_results: list[dict[str, Any]]

    # Mastery updates
    mastery_updates: Annotated[list[dict[str, Any]], add]

    # Progress tracking
    exam_recommended: bool
    should_continue: bool
    session_complete: bool

    # Error handling
    error: Optional[str]

    # Messages for LangChain integration
    messages: Annotated[list[Any], add]


class QuestionGenState(TypedDict, total=False):
    """State for the question generation agent.

    RAG pipeline state:
    load_documents -> chunk -> embed -> retrieve -> generate -> validate
    """

    # Input
    module_id: str
    source_text: str
    source_documents: list[dict[str, Any]]
    target_bloom_level: int
    num_questions: int

    # Chunking
    chunks: list[dict[str, Any]]
    chunk_embeddings: list[list[float]]

    # Retrieval
    query: str
    retrieved_chunks: list[dict[str, Any]]

    # Generation
    generated_questions: list[Question]
    generation_attempts: int

    # Validation
    validated_questions: list[Question]
    rejected_questions: list[dict[str, Any]]

    # Error handling
    error: Optional[str]

    # Messages
    messages: Annotated[list[Any], add]


class EvaluationState(TypedDict, total=False):
    """State for the evaluation agent.

    Evaluates question quality using:
    IRT discrimination + LLM-as-judge + RAGAS metrics
    """

    # Input
    question: Question
    context: str
    response_data: Optional[list[int]]
    ability_estimates: Optional[list[float]]

    # IRT results
    irt_discrimination: Optional[float]
    irt_difficulty: Optional[float]
    irt_quality_category: Optional[str]

    # LLM judge results
    llm_scores: dict[str, float]
    llm_reasoning: dict[str, str]

    # RAGAS results
    faithfulness_score: Optional[float]
    answer_relevancy_score: Optional[float]
    context_relevancy_score: Optional[float]

    # Combined
    overall_score: float
    quality_grade: str
    passed: bool
    recommendations: list[str]

    # Messages
    messages: Annotated[list[Any], add]

"""LangGraph-based agents for adaptive learning orchestration.

This module provides stateful agents using LangGraph for:
1. Learning Session Agent - Orchestrates the complete learning flow
2. Question Generation Agent - RAG pipeline for generating questions
3. Evaluation Agent - Quality assessment of generated content
4. Smart Chunker Agent - Filters and reformats text chunks
5. Paraphraser Agent - Adjusts chunk readability (scaffolding)

The agents implement the paper's algorithms as a state machine:
    START -> apply_decay -> select_cards -> generate_quiz ->
    await_answers -> update_mastery -> check_progress ->
    (continue | recommend_exam | END)
"""

from adaptive_learning_core.agents.state import (
    LearningState,
    QuestionGenState,
)
from adaptive_learning_core.agents.learning_agent import (
    LearningAgent,
    create_learning_graph,
)
from adaptive_learning_core.agents.question_gen_agent import (
    QuestionGenAgent,
    create_question_gen_graph,
)
from adaptive_learning_core.agents.smart_chunker_agent import (
    SmartChunkerAgent,
    SmartChunkerState,
    create_smart_chunker_graph,
)
from adaptive_learning_core.agents.paraphraser_agent import (
    ParaphraserAgent,
    ParaphraserState,
    ParaphraserConfig,
    ReadabilityLevel,
)

__all__ = [
    # State
    "LearningState",
    "QuestionGenState",
    "SmartChunkerState",
    "ParaphraserState",
    # Agents
    "LearningAgent",
    "create_learning_graph",
    "QuestionGenAgent",
    "create_question_gen_graph",
    "SmartChunkerAgent",
    "create_smart_chunker_graph",
    "ParaphraserAgent",
    "ParaphraserConfig",
    "ReadabilityLevel",
]

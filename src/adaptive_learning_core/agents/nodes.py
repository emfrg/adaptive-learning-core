"""LangGraph node functions for adaptive learning.

Each node function takes the current state and returns updates.
These nodes implement the paper's algorithms within the LangGraph framework.
"""

from datetime import datetime, timezone
from typing import Any
import uuid

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage

from adaptive_learning_core.agents.state import LearningState, QuestionGenState
from adaptive_learning_core.algorithms.knowledge_decay import KnowledgeDecay
from adaptive_learning_core.algorithms.card_selector import CardSelector
from adaptive_learning_core.algorithms.mastery_update import MasteryUpdater
from adaptive_learning_core.algorithms.question_linker import QuestionLinker
from adaptive_learning_core.algorithms.question_selector import QuestionSelector
from adaptive_learning_core.models.card import CardState
from adaptive_learning_core.config import DEFAULT_CONFIG


# =============================================================================
# Learning Session Nodes
# =============================================================================


def initialize_session(state: LearningState) -> dict[str, Any]:
    """Initialize a new learning session.

    Sets up default values and generates session ID.
    """
    return {
        "session_id": state.get("session_id") or f"session_{uuid.uuid4().hex[:12]}",
        "cycle_count": 0,
        "decay_applied": False,
        "exam_recommended": False,
        "should_continue": True,
        "session_complete": False,
        "mastery_updates": [],
        "messages": [
            AIMessage(content="Learning session initialized. Applying knowledge decay...")
        ],
    }


def apply_decay(state: LearningState) -> dict[str, Any]:
    """Apply time-based knowledge decay (Equations 7-8).

    M(c) = M(c) × e^{-β × λ × τ}

    This node applies decay to all cards based on time since last interaction.
    """
    decay = KnowledgeDecay(config=DEFAULT_CONFIG.decay)
    current_time = datetime.now(timezone.utc)

    card_states = state.get("card_states", {})
    decay_results = []

    for card_id, card_state in card_states.items():
        result = decay.apply_decay(
            card_state=card_state,
            current_time=current_time,
            in_place=True,
        )
        decay_results.append({
            "card_id": result.card_id,
            "old_score": result.old_score,
            "new_score": result.new_score,
            "hours_elapsed": result.hours_elapsed,
            "decay_applied": result.decay_applied,
        })

    decayed_count = sum(1 for r in decay_results if r["decay_applied"])

    return {
        "card_states": card_states,
        "decay_applied": True,
        "decay_results": decay_results,
        "messages": [
            AIMessage(content=f"Decay applied to {decayed_count} cards. Selecting cards for study...")
        ],
    }


def select_cards(state: LearningState) -> dict[str, Any]:
    """Select cards for the learning cycle (Equations 4-5).

    P(c) = (1 - M(c) + γ) / Z

    Selects cards probabilistically, prioritizing lower mastery.
    """
    selector = CardSelector(config=DEFAULT_CONFIG.selection)

    card_states = state.get("card_states", {})
    cards_per_cycle = state.get("cards_per_cycle", 5)
    last_score = state.get("last_quiz_score")

    state_list = list(card_states.values())
    result = selector.select_cards(
        card_states=state_list,
        n=cards_per_cycle,
        previous_quiz_score=last_score,
    )

    return {
        "selected_card_ids": result.selected_card_ids,
        "messages": [
            AIMessage(content=f"Selected {len(result.selected_card_ids)} cards: {result.selected_card_ids}")
        ],
    }


def select_questions(state: LearningState) -> dict[str, Any]:
    """Select questions matching the selected cards and learner level.

    Implements Rules 2 & 3 from the paper:
    - Filter by difficulty_level ≤ target_mastery_level
    - Match questions to selected cards
    """
    selector = QuestionSelector(config=DEFAULT_CONFIG.question_selection)

    questions = state.get("questions", [])
    selected_card_ids = state.get("selected_card_ids", [])
    card_states = state.get("card_states", {})
    questions_per_quiz = state.get("questions_per_quiz", 5)
    target_level = state.get("target_mastery_level", 3)

    # Get learner stereotype from average mastery
    avg_mastery = np.mean([
        card_states[cid].mastery_score
        for cid in selected_card_ids
        if cid in card_states
    ]) if selected_card_ids else 0.0

    result = selector.select_questions(
        questions=questions,
        selected_card_ids=selected_card_ids,
        learner_mastery=avg_mastery,
        target_level=target_level,
        n_questions=questions_per_quiz,
    )

    return {
        "quiz_questions": result.selected_questions,
        "messages": [
            AIMessage(content=f"Selected {len(result.selected_questions)} questions for quiz")
        ],
    }


def await_answers(state: LearningState) -> dict[str, Any]:
    """Wait for learner to answer quiz questions.

    This is a human-in-the-loop node. In practice, this would
    pause and wait for user input through the LangGraph API.
    """
    quiz_questions = state.get("quiz_questions", [])

    return {
        "messages": [
            AIMessage(content=f"Quiz ready with {len(quiz_questions)} questions. Waiting for answers...")
        ],
    }


def process_answers(state: LearningState) -> dict[str, Any]:
    """Process quiz answers and calculate score.

    Computes the quiz score from submitted answers.
    """
    quiz_questions = state.get("quiz_questions", [])
    quiz_answers = state.get("quiz_answers", [])

    if not quiz_answers:
        return {
            "last_quiz_score": 0.0,
            "messages": [AIMessage(content="No answers received.")],
        }

    # Calculate score
    correct = sum(1 for a in quiz_answers if a.is_correct)
    total = len(quiz_answers)
    score = correct / total if total > 0 else 0.0

    return {
        "last_quiz_score": score,
        "messages": [
            AIMessage(content=f"Quiz completed: {correct}/{total} correct ({score:.0%})")
        ],
    }


def update_mastery(state: LearningState) -> dict[str, Any]:
    """Update mastery scores based on quiz performance (Equation 3).

    M(c) = M(c) + α × C_{c,q} × δ

    Updates mastery for all cards linked to answered questions.
    """
    updater = MasteryUpdater(config=DEFAULT_CONFIG.mastery)

    card_states = state.get("card_states", {})
    quiz_questions = state.get("quiz_questions", [])
    quiz_answers = state.get("quiz_answers", [])
    pace = state.get("pace", "medium")

    updates = []
    for answer in quiz_answers:
        # Find the question
        question = next(
            (q for q in quiz_questions if q.id == answer.question_id),
            None
        )
        if not question:
            continue

        # Update mastery for linked cards
        results = updater.update_batch(
            card_states=card_states,
            question=question,
            is_correct=answer.is_correct,
            pace=pace,
        )

        for r in results:
            updates.append({
                "card_id": r.card_id,
                "old_score": r.old_score,
                "new_score": r.new_score,
                "was_correct": r.was_correct,
            })

    return {
        "card_states": card_states,
        "mastery_updates": updates,
        "messages": [
            AIMessage(content=f"Updated mastery for {len(updates)} card-question pairs")
        ],
    }


def check_progress(state: LearningState) -> dict[str, Any]:
    """Check learning progress and decide next action.

    Implements Rule 4: Recommend exam when sufficient cards
    have reached the target mastery level.
    """
    card_states = state.get("card_states", {})
    exam_threshold = state.get("exam_threshold", 0.8)
    max_cycles = state.get("max_cycles", 10)
    cycle_count = state.get("cycle_count", 0) + 1

    # Count cards above threshold
    total_cards = len(card_states)
    above_threshold = sum(
        1 for s in card_states.values()
        if s.mastery_score >= exam_threshold and not s.is_ignored
    )

    coverage = above_threshold / total_cards if total_cards > 0 else 0.0
    exam_recommended = coverage >= 0.8

    # Check if should continue
    should_continue = (
        cycle_count < max_cycles
        and not exam_recommended
    )

    message = f"Cycle {cycle_count} complete. {above_threshold}/{total_cards} cards above threshold ({coverage:.0%})"
    if exam_recommended:
        message += " - Module exam recommended!"
    elif not should_continue:
        message += " - Maximum cycles reached."

    return {
        "cycle_count": cycle_count,
        "exam_recommended": exam_recommended,
        "should_continue": should_continue,
        "session_complete": not should_continue,
        "messages": [AIMessage(content=message)],
    }


def end_session(state: LearningState) -> dict[str, Any]:
    """End the learning session and summarize results."""
    cycle_count = state.get("cycle_count", 0)
    exam_recommended = state.get("exam_recommended", False)
    card_states = state.get("card_states", {})

    avg_mastery = np.mean([s.mastery_score for s in card_states.values()]) if card_states else 0.0

    summary = f"Session complete after {cycle_count} cycles. Average mastery: {avg_mastery:.2f}"
    if exam_recommended:
        summary += " You're ready for the module exam!"

    return {
        "session_complete": True,
        "messages": [AIMessage(content=summary)],
    }


# =============================================================================
# Question Generation Nodes
# =============================================================================


def chunk_documents(state: QuestionGenState) -> dict[str, Any]:
    """Chunk source documents for RAG.

    Uses LangChain text splitters for chunking.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    source_text = state.get("source_text", "")
    source_docs = state.get("source_documents", [])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    # Chunk from raw text
    if source_text:
        text_chunks = splitter.split_text(source_text)
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "id": f"chunk_{i}",
                "text": chunk,
                "source": "text",
                "index": i,
            })

    # Chunk from documents
    for doc in source_docs:
        doc_text = doc.get("content", doc.get("text", ""))
        doc_chunks = splitter.split_text(doc_text)
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                "id": f"doc_{doc.get('id', 'unknown')}_{i}",
                "text": chunk,
                "source": doc.get("source", "document"),
                "index": i,
            })

    return {
        "chunks": chunks,
        "messages": [AIMessage(content=f"Created {len(chunks)} chunks from source material")],
    }


def embed_chunks(state: QuestionGenState) -> dict[str, Any]:
    """Embed chunks for retrieval.

    Uses sentence-transformers for embeddings.
    """
    from adaptive_learning_core.rag.embeddings import SentenceTransformerEmbeddings

    chunks = state.get("chunks", [])

    if not chunks:
        return {"chunk_embeddings": [], "messages": [AIMessage(content="No chunks to embed")]}

    embedder = SentenceTransformerEmbeddings()
    texts = [c["text"] for c in chunks]
    result = embedder.embed_batch(texts)

    return {
        "chunk_embeddings": result.embeddings,
        "messages": [AIMessage(content=f"Embedded {len(chunks)} chunks")],
    }


def retrieve_context(state: QuestionGenState) -> dict[str, Any]:
    """Retrieve relevant chunks for question generation."""
    chunks = state.get("chunks", [])
    embeddings = state.get("chunk_embeddings", [])
    query = state.get("query", "")
    target_bloom = state.get("target_bloom_level", 1)

    if not chunks or not embeddings:
        return {"retrieved_chunks": chunks[:5]}  # Fallback: use first 5

    # Simple retrieval: use all chunks for now
    # In production, would use vector similarity
    retrieved = chunks[:min(10, len(chunks))]

    return {
        "retrieved_chunks": retrieved,
        "messages": [AIMessage(content=f"Retrieved {len(retrieved)} relevant chunks")],
    }


def generate_questions(state: QuestionGenState) -> dict[str, Any]:
    """Generate questions from retrieved context using LLM.

    This node uses the LangChain chat model for generation.
    """
    # This is a placeholder - actual generation happens in the agent
    # which has access to the LLM
    return {
        "generation_attempts": state.get("generation_attempts", 0) + 1,
        "messages": [AIMessage(content="Generating questions from context...")],
    }


def validate_questions(state: QuestionGenState) -> dict[str, Any]:
    """Validate generated questions for quality."""
    generated = state.get("generated_questions", [])

    validated = []
    rejected = []

    for q in generated:
        # Basic validation
        is_valid = (
            len(q.stem) >= 10
            and len(q.correct_answer) >= 1
            and len(q.distractors) >= 2
        )

        if is_valid:
            validated.append(q)
        else:
            rejected.append({
                "question_id": q.id,
                "reason": "Failed basic validation",
            })

    return {
        "validated_questions": validated,
        "rejected_questions": rejected,
        "messages": [
            AIMessage(content=f"Validated {len(validated)} questions, rejected {len(rejected)}")
        ],
    }


# =============================================================================
# Routing Functions
# =============================================================================


def should_continue_learning(state: LearningState) -> str:
    """Determine the next node in the learning graph."""
    if state.get("should_continue", False):
        return "select_cards"
    else:
        return "end_session"


def should_retry_generation(state: QuestionGenState) -> str:
    """Determine if question generation should be retried."""
    attempts = state.get("generation_attempts", 0)
    generated = state.get("generated_questions", [])
    target = state.get("num_questions", 5)

    if len(generated) >= target:
        return "validate"
    elif attempts < 3:
        return "generate"
    else:
        return "validate"  # Give up and validate what we have

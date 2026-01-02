"""Prompts for RAG-based question generation."""

from adaptive_learning_core.rag.prompts.question_gen import (
    QUESTION_GENERATION_PROMPT,
    BLOOM_TARGETED_PROMPT,
)
from adaptive_learning_core.rag.prompts.distractor_gen import (
    DISTRACTOR_GENERATION_PROMPT,
    DISTRACTOR_IMPROVEMENT_PROMPT,
)

__all__ = [
    "QUESTION_GENERATION_PROMPT",
    "BLOOM_TARGETED_PROMPT",
    "DISTRACTOR_GENERATION_PROMPT",
    "DISTRACTOR_IMPROVEMENT_PROMPT",
]

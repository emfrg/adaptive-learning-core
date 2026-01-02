"""RAG (Retrieval-Augmented Generation) pipeline for question generation.

This module provides tools for:
1. Chunking learning materials (PDF/text) into cards
2. Generating embeddings for semantic similarity
3. Generating MCQ questions from cards using LLMs
4. Linking questions to source cards

The pipeline follows the approach from:
"Automated Evaluation of Retrieval-Augmented Language Models
with Task-Specific Exam Generation" (Guinet et al., ICML 2024)
"""

from adaptive_learning_core.rag.chunker import (
    ChunkingConfig,
    Chunk,
    TextChunker,
    TokenChunker,
    SentenceChunker,
    SemanticChunker,
)
from adaptive_learning_core.rag.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddings,
    EmbeddingResult,
)
from adaptive_learning_core.rag.retriever import (
    VectorRetriever,
    RetrievalResult,
    RetrievalConfig,
)
from adaptive_learning_core.rag.question_generator import (
    QuestionGenerator,
    QuestionGenerationConfig,
    GeneratedQuestion,
)

__all__ = [
    # Chunking
    "ChunkingConfig",
    "Chunk",
    "TextChunker",
    "TokenChunker",
    "SentenceChunker",
    "SemanticChunker",
    # Embeddings
    "EmbeddingProvider",
    "SentenceTransformerEmbeddings",
    "EmbeddingResult",
    # Retrieval
    "VectorRetriever",
    "RetrievalResult",
    "RetrievalConfig",
    # Question Generation
    "QuestionGenerator",
    "QuestionGenerationConfig",
    "GeneratedQuestion",
]

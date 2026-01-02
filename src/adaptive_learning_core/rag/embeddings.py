"""Embedding generation for semantic similarity.

Provides a generic interface for embedding providers and a
concrete implementation using sentence-transformers.

Embeddings are used for:
1. Question-to-card linking (Equation 1: cosine similarity)
2. Semantic retrieval of relevant cards
3. Similarity-based clustering of content
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.

    Attributes:
        id: Identifier for the embedded text
        embedding: The embedding vector
        text: The original text (for reference)
        model_name: Name of the model used
        dimension: Dimension of the embedding
    """

    id: str
    embedding: NDArray[np.float64]
    text: str
    model_name: str

    @property
    def dimension(self) -> int:
        """Dimension of the embedding vector."""
        return len(self.embedding)

    def cosine_similarity(self, other: "EmbeddingResult") -> float:
        """Calculate cosine similarity with another embedding.

        Implements Equation 1: cos_sim(a, b) = (a · b) / (|a| × |b|)

        Args:
            other: Another EmbeddingResult

        Returns:
            Cosine similarity in range [-1, 1]
        """
        dot_product = np.dot(self.embedding, other.embedding)
        norm_a = np.linalg.norm(self.embedding)
        norm_b = np.linalg.norm(other.embedding)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers.

    Implementations can use different backends:
    - sentence-transformers (local)
    - OpenAI embeddings API
    - Google Vertex AI embeddings
    - etc.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model being used."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embedding vectors produced."""
        pass

    @abstractmethod
    def embed(self, text: str, text_id: Optional[str] = None) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: The text to embed
            text_id: Optional identifier for the text

        Returns:
            EmbeddingResult with the embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str],
        text_ids: Optional[list[str]] = None,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        More efficient than calling embed() multiple times
        for batch processing.

        Args:
            texts: List of texts to embed
            text_ids: Optional list of identifiers

        Returns:
            List of EmbeddingResults
        """
        pass

    def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a query for retrieval.

        Some models use different embeddings for queries vs documents.
        Override this method if your model has this distinction.

        Args:
            query: The query text

        Returns:
            EmbeddingResult optimized for query matching
        """
        return self.embed(query, text_id="query")


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Embedding provider using sentence-transformers library.

    Uses locally-run transformer models for generating embeddings.
    Good balance of quality and cost (free, runs locally).

    Recommended models:
    - all-MiniLM-L6-v2: Fast, 384 dimensions
    - all-mpnet-base-v2: Better quality, 768 dimensions
    - multi-qa-mpnet-base-dot-v1: Optimized for Q&A

    Example:
        >>> embedder = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        >>> result = embedder.embed("What is machine learning?")
        >>> print(f"Dimension: {result.dimension}")
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ):
        """Initialize the embedding provider.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on ("cpu", "cuda", etc.)
            cache_folder: Where to cache downloaded models
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbeddings. "
                "Install with: pip install sentence-transformers"
            )

        self._model_name = model_name
        self._model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        """Name of the embedding model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        return self._dimension

    def embed(self, text: str, text_id: Optional[str] = None) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: The text to embed
            text_id: Optional identifier

        Returns:
            EmbeddingResult with the embedding vector
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        return EmbeddingResult(
            id=text_id or "",
            embedding=embedding.astype(np.float64),
            text=text,
            model_name=self._model_name,
        )

    def embed_batch(
        self,
        texts: list[str],
        text_ids: Optional[list[str]] = None,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Uses batch processing for efficiency.

        Args:
            texts: List of texts to embed
            text_ids: Optional list of identifiers

        Returns:
            List of EmbeddingResults
        """
        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(texts))]

        embeddings = self._model.encode(texts, convert_to_numpy=True)

        results = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            results.append(
                EmbeddingResult(
                    id=text_ids[i],
                    embedding=embedding.astype(np.float64),
                    text=text,
                    model_name=self._model_name,
                )
            )
        return results

    def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a query for retrieval.

        sentence-transformers doesn't distinguish queries from documents,
        so this just calls embed().

        Args:
            query: The query text

        Returns:
            EmbeddingResult for the query
        """
        return self.embed(query, text_id="query")


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wrapper that adds caching to any embedding provider.

    Useful for avoiding redundant API calls or computation
    when the same texts are embedded multiple times.

    Example:
        >>> base_embedder = SentenceTransformerEmbeddings()
        >>> cached = CachedEmbeddingProvider(base_embedder)
        >>> # Second call uses cache
        >>> result1 = cached.embed("Hello world")
        >>> result2 = cached.embed("Hello world")  # From cache
    """

    def __init__(
        self,
        base_provider: EmbeddingProvider,
        max_cache_size: int = 10000,
    ):
        """Initialize with a base provider.

        Args:
            base_provider: The underlying embedding provider
            max_cache_size: Maximum number of cached embeddings
        """
        self._base = base_provider
        self._cache: dict[str, EmbeddingResult] = {}
        self._max_cache_size = max_cache_size

    @property
    def model_name(self) -> str:
        """Name of the underlying model."""
        return self._base.model_name

    @property
    def dimension(self) -> int:
        """Dimension of embeddings."""
        return self._base.dimension

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Simple hash-based key
        return str(hash(text))

    def embed(self, text: str, text_id: Optional[str] = None) -> EmbeddingResult:
        """Embed with caching.

        Args:
            text: Text to embed
            text_id: Optional identifier

        Returns:
            EmbeddingResult (from cache if available)
        """
        cache_key = self._get_cache_key(text)

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Update the ID if provided
            if text_id:
                return EmbeddingResult(
                    id=text_id,
                    embedding=cached.embedding,
                    text=cached.text,
                    model_name=cached.model_name,
                )
            return cached

        # Compute and cache
        result = self._base.embed(text, text_id)

        if len(self._cache) < self._max_cache_size:
            self._cache[cache_key] = result

        return result

    def embed_batch(
        self,
        texts: list[str],
        text_ids: Optional[list[str]] = None,
    ) -> list[EmbeddingResult]:
        """Batch embed with caching.

        Only computes embeddings for texts not in cache.

        Args:
            texts: Texts to embed
            text_ids: Optional identifiers

        Returns:
            List of EmbeddingResults
        """
        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(texts))]

        results: list[Optional[EmbeddingResult]] = [None] * len(texts)
        texts_to_compute: list[tuple[int, str, str]] = []  # (index, text, id)

        # Check cache first
        for i, (text, text_id) in enumerate(zip(texts, text_ids)):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                results[i] = EmbeddingResult(
                    id=text_id,
                    embedding=cached.embedding,
                    text=cached.text,
                    model_name=cached.model_name,
                )
            else:
                texts_to_compute.append((i, text, text_id))

        # Compute missing embeddings
        if texts_to_compute:
            new_texts = [t[1] for t in texts_to_compute]
            new_ids = [t[2] for t in texts_to_compute]
            new_results = self._base.embed_batch(new_texts, new_ids)

            for (idx, text, _), result in zip(texts_to_compute, new_results):
                results[idx] = result
                # Cache if space available
                cache_key = self._get_cache_key(text)
                if len(self._cache) < self._max_cache_size:
                    self._cache[cache_key] = result

        return [r for r in results if r is not None]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Current number of cached embeddings."""
        return len(self._cache)

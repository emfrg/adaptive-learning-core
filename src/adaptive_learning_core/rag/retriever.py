"""Vector similarity retrieval for RAG pipeline.

Retrieves relevant cards/chunks based on semantic similarity
to a query. Used for finding context for question generation
and for linking questions to source material.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from adaptive_learning_core.rag.embeddings import EmbeddingProvider, EmbeddingResult


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for vector retrieval.

    Attributes:
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold
        include_scores: Whether to include similarity scores
        normalize_scores: Whether to normalize scores to [0, 1]
    """

    top_k: int = 5
    min_similarity: float = 0.3
    include_scores: bool = True
    normalize_scores: bool = False


@dataclass
class RetrievalResult:
    """A single retrieval result.

    Attributes:
        id: ID of the retrieved document/card
        text: The retrieved text
        similarity: Cosine similarity score
        rank: Rank in the result list (0-indexed)
        metadata: Additional metadata from the source
    """

    id: str
    text: str
    similarity: float
    rank: int
    metadata: dict = field(default_factory=dict)

    @property
    def is_relevant(self) -> bool:
        """Whether this result passes typical relevance threshold."""
        return self.similarity >= 0.5


class VectorRetriever:
    """Retrieves documents by vector similarity.

    Uses cosine similarity (Equation 1 from the paper) to find
    the most similar documents to a query.

    Supports:
    - Single query retrieval
    - Batch query retrieval
    - Filtering by metadata
    - Similarity thresholds

    Example:
        >>> retriever = VectorRetriever(embedding_provider)
        >>> retriever.add_documents(cards)
        >>> results = retriever.retrieve("What is photosynthesis?", top_k=3)
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: Optional[RetrievalConfig] = None,
    ):
        """Initialize the retriever.

        Args:
            embedding_provider: Provider for generating embeddings
            config: Retrieval configuration
        """
        self.embedder = embedding_provider
        self.config = config or RetrievalConfig()

        # Document storage
        self._documents: dict[str, str] = {}  # id -> text
        self._embeddings: dict[str, NDArray[np.float64]] = {}  # id -> embedding
        self._metadata: dict[str, dict] = {}  # id -> metadata

    @property
    def document_count(self) -> int:
        """Number of documents in the index."""
        return len(self._documents)

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
        embedding: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Add a single document to the index.

        Args:
            doc_id: Unique document identifier
            text: Document text
            metadata: Optional metadata dictionary
            embedding: Pre-computed embedding (computed if not provided)
        """
        self._documents[doc_id] = text
        self._metadata[doc_id] = metadata or {}

        if embedding is not None:
            self._embeddings[doc_id] = embedding
        else:
            result = self.embedder.embed(text, text_id=doc_id)
            self._embeddings[doc_id] = result.embedding

    def add_documents(
        self,
        documents: list[tuple[str, str, Optional[dict]]],
        embeddings: Optional[list[NDArray[np.float64]]] = None,
    ) -> None:
        """Add multiple documents to the index.

        More efficient than adding documents one by one as it
        uses batch embedding.

        Args:
            documents: List of (id, text, metadata) tuples
            embeddings: Optional pre-computed embeddings
        """
        if embeddings is not None:
            # Use provided embeddings
            for (doc_id, text, metadata), embedding in zip(documents, embeddings):
                self._documents[doc_id] = text
                self._metadata[doc_id] = metadata or {}
                self._embeddings[doc_id] = embedding
        else:
            # Batch compute embeddings
            texts = [text for _, text, _ in documents]
            ids = [doc_id for doc_id, _, _ in documents]
            results = self.embedder.embed_batch(texts, ids)

            for (doc_id, text, metadata), result in zip(documents, results):
                self._documents[doc_id] = text
                self._metadata[doc_id] = metadata or {}
                self._embeddings[doc_id] = result.embedding

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index.

        Args:
            doc_id: ID of document to remove

        Returns:
            True if document was removed, False if not found
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            del self._embeddings[doc_id]
            del self._metadata[doc_id]
            return True
        return False

    def _cosine_similarity(
        self,
        query_embedding: NDArray[np.float64],
        doc_embedding: NDArray[np.float64],
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Implements Equation 1: cos_sim(q, d) = (q · d) / (|q| × |d|)

        Args:
            query_embedding: Query vector
            doc_embedding: Document vector

        Returns:
            Cosine similarity in range [-1, 1]
        """
        dot_product = np.dot(query_embedding, doc_embedding)
        norm_q = np.linalg.norm(query_embedding)
        norm_d = np.linalg.norm(doc_embedding)

        if norm_q == 0 or norm_d == 0:
            return 0.0

        return float(dot_product / (norm_q * norm_d))

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents similar to query.

        Args:
            query: Query text
            top_k: Number of results (overrides config)
            min_similarity: Minimum similarity (overrides config)
            filter_metadata: Only return docs matching this metadata

        Returns:
            List of RetrievalResults, sorted by similarity
        """
        if not self._documents:
            return []

        top_k = top_k if top_k is not None else self.config.top_k
        min_similarity = (
            min_similarity if min_similarity is not None else self.config.min_similarity
        )

        # Get query embedding
        query_result = self.embedder.embed_query(query)
        query_embedding = query_result.embedding

        # Calculate similarities
        similarities: list[tuple[str, float]] = []
        for doc_id, doc_embedding in self._embeddings.items():
            # Apply metadata filter
            if filter_metadata:
                doc_meta = self._metadata.get(doc_id, {})
                if not all(doc_meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            sim = self._cosine_similarity(query_embedding, doc_embedding)
            if sim >= min_similarity:
                similarities.append((doc_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top_k
        top_results = similarities[:top_k]

        # Normalize scores if configured
        if self.config.normalize_scores and top_results:
            max_sim = top_results[0][1]
            min_sim_in_results = top_results[-1][1]
            range_sim = max_sim - min_sim_in_results
            if range_sim > 0:
                top_results = [
                    (doc_id, (sim - min_sim_in_results) / range_sim)
                    for doc_id, sim in top_results
                ]

        # Build results
        results = []
        for rank, (doc_id, sim) in enumerate(top_results):
            results.append(
                RetrievalResult(
                    id=doc_id,
                    text=self._documents[doc_id],
                    similarity=sim,
                    rank=rank,
                    metadata=self._metadata.get(doc_id, {}),
                )
            )

        return results

    def retrieve_by_embedding(
        self,
        embedding: NDArray[np.float64],
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents using a pre-computed embedding.

        Useful when you already have an embedding and want to
        avoid recomputing it.

        Args:
            embedding: Query embedding vector
            top_k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of RetrievalResults
        """
        if not self._documents:
            return []

        top_k = top_k if top_k is not None else self.config.top_k
        min_similarity = (
            min_similarity if min_similarity is not None else self.config.min_similarity
        )

        # Calculate similarities
        similarities: list[tuple[str, float]] = []
        for doc_id, doc_embedding in self._embeddings.items():
            sim = self._cosine_similarity(embedding, doc_embedding)
            if sim >= min_similarity:
                similarities.append((doc_id, sim))

        # Sort and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]

        # Build results
        results = []
        for rank, (doc_id, sim) in enumerate(top_results):
            results.append(
                RetrievalResult(
                    id=doc_id,
                    text=self._documents[doc_id],
                    similarity=sim,
                    rank=rank,
                    metadata=self._metadata.get(doc_id, {}),
                )
            )

        return results

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> list[list[RetrievalResult]]:
        """Retrieve documents for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of results per query
            min_similarity: Minimum similarity threshold

        Returns:
            List of result lists (one per query)
        """
        # Batch embed queries
        query_results = self.embedder.embed_batch(
            queries, [f"query_{i}" for i in range(len(queries))]
        )

        all_results = []
        for query_result in query_results:
            results = self.retrieve_by_embedding(
                query_result.embedding,
                top_k=top_k,
                min_similarity=min_similarity,
            )
            all_results.append(results)

        return all_results

    def get_document(self, doc_id: str) -> Optional[str]:
        """Get document text by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document text or None if not found
        """
        return self._documents.get(doc_id)

    def get_embedding(self, doc_id: str) -> Optional[NDArray[np.float64]]:
        """Get document embedding by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Embedding vector or None if not found
        """
        return self._embeddings.get(doc_id)

    def get_all_embeddings(self) -> dict[str, NDArray[np.float64]]:
        """Get all document embeddings.

        Returns:
            Dictionary of doc_id -> embedding
        """
        return self._embeddings.copy()

    def clear(self) -> None:
        """Clear all documents from the index."""
        self._documents.clear()
        self._embeddings.clear()
        self._metadata.clear()

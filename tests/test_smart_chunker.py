"""Tests for SmartChunkerAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from adaptive_learning_core.agents.smart_chunker_agent import (
    SmartChunkerAgent,
    SmartChunkerState,
    create_smart_chunker_graph,
    filter_rejected,
    should_continue_classification,
    REJECTED_CATEGORIES,
)
from adaptive_learning_core.rag.chunker import Chunk


class TestSmartChunkerState:
    """Tests for SmartChunkerState TypedDict."""

    def test_state_creation(self):
        """Should create valid state dict."""
        state = SmartChunkerState(
            input_chunks=[],
            batch_size=5,
            classification_results=[],
            current_batch_index=0,
        )
        assert state["batch_size"] == 5
        assert state["current_batch_index"] == 0


class TestFilterRejectedNode:
    """Tests for filter_rejected node function."""

    def test_filter_accepts_substantive(self):
        """Should accept chunks classified as substantive."""
        state = SmartChunkerState(
            input_chunks=[
                {"id": "c1", "text": "Photosynthesis is the process..."},
                {"id": "c2", "text": "Mitosis involves cell division..."},
            ],
            classification_results=[
                {"chunk_id": "c1", "category": "substantive", "confidence": 0.95, "reason": "Educational content"},
                {"chunk_id": "c2", "category": "substantive", "confidence": 0.9, "reason": "Biology concept"},
            ],
        )

        result = filter_rejected(state)

        assert len(result["accepted_chunks"]) == 2
        assert len(result["rejected_chunks"]) == 0
        assert len(result["rejection_reasons"]) == 0

    def test_filter_rejects_references(self):
        """Should reject chunks classified as references."""
        state = SmartChunkerState(
            input_chunks=[
                {"id": "c1", "text": "References:\n1. Smith, J. (2020)..."},
            ],
            classification_results=[
                {"chunk_id": "c1", "category": "reference", "confidence": 0.98, "reason": "Bibliography section"},
            ],
        )

        result = filter_rejected(state)

        assert len(result["accepted_chunks"]) == 0
        assert len(result["rejected_chunks"]) == 1
        assert "c1" in result["rejection_reasons"]

    def test_filter_rejects_toc(self):
        """Should reject table of contents."""
        state = SmartChunkerState(
            input_chunks=[
                {"id": "c1", "text": "Table of Contents\n1. Introduction...1\n2. Methods...15"},
            ],
            classification_results=[
                {"chunk_id": "c1", "category": "table_of_contents", "confidence": 0.99, "reason": "TOC"},
            ],
        )

        result = filter_rejected(state)

        assert len(result["accepted_chunks"]) == 0
        assert len(result["rejected_chunks"]) == 1

    def test_filter_rejects_all_junk_categories(self):
        """Should reject all junk categories."""
        chunks = []
        classifications = []

        for i, category in enumerate(REJECTED_CATEGORIES):
            chunk_id = f"c{i}"
            chunks.append({"id": chunk_id, "text": f"Content for {category}"})
            classifications.append({
                "chunk_id": chunk_id,
                "category": category,
                "confidence": 0.9,
                "reason": f"Detected as {category}",
            })

        state = SmartChunkerState(
            input_chunks=chunks,
            classification_results=classifications,
        )

        result = filter_rejected(state)

        assert len(result["accepted_chunks"]) == 0
        assert len(result["rejected_chunks"]) == len(REJECTED_CATEGORIES)

    def test_filter_accepts_unclassified_by_default(self):
        """Should accept chunks that weren't classified (safe fallback)."""
        state = SmartChunkerState(
            input_chunks=[
                {"id": "c1", "text": "Some content"},
            ],
            classification_results=[],  # No classification results
        )

        result = filter_rejected(state)

        assert len(result["accepted_chunks"]) == 1
        assert len(result["rejected_chunks"]) == 0

    def test_filter_mixed_content(self):
        """Should correctly separate mixed content."""
        state = SmartChunkerState(
            input_chunks=[
                {"id": "c1", "text": "Educational content about biology"},
                {"id": "c2", "text": "References:\n1. Author..."},
                {"id": "c3", "text": "More educational content"},
                {"id": "c4", "text": "Table of Contents..."},
            ],
            classification_results=[
                {"chunk_id": "c1", "category": "substantive", "confidence": 0.9, "reason": "Educational"},
                {"chunk_id": "c2", "category": "reference", "confidence": 0.95, "reason": "Bibliography"},
                {"chunk_id": "c3", "category": "substantive", "confidence": 0.85, "reason": "Educational"},
                {"chunk_id": "c4", "category": "table_of_contents", "confidence": 0.98, "reason": "TOC"},
            ],
        )

        result = filter_rejected(state)

        assert len(result["accepted_chunks"]) == 2
        assert len(result["rejected_chunks"]) == 2

        accepted_ids = [c["id"] for c in result["accepted_chunks"]]
        assert "c1" in accepted_ids
        assert "c3" in accepted_ids


class TestShouldContinueClassification:
    """Tests for the routing function."""

    def test_continue_when_more_batches(self):
        """Should continue when more batches need processing."""
        state = SmartChunkerState(
            input_chunks=[{"id": f"c{i}", "text": "..."} for i in range(15)],
            batch_size=5,
            current_batch_index=1,  # Only processed 1 batch out of 3
        )

        result = should_continue_classification(state)
        assert result == "classify_chunks"

    def test_filter_when_all_batches_done(self):
        """Should move to filter when all batches processed."""
        state = SmartChunkerState(
            input_chunks=[{"id": f"c{i}", "text": "..."} for i in range(10)],
            batch_size=5,
            current_batch_index=2,  # Processed all 2 batches
        )

        result = should_continue_classification(state)
        assert result == "filter_rejected"

    def test_filter_when_no_chunks(self):
        """Should move to filter immediately if no chunks."""
        state = SmartChunkerState(
            input_chunks=[],
            batch_size=5,
            current_batch_index=0,
        )

        result = should_continue_classification(state)
        assert result == "filter_rejected"


class TestSmartChunkerGraph:
    """Tests for graph creation."""

    def test_graph_creation(self):
        """Graph should be created with all nodes."""
        mock_llm = MagicMock()
        graph = create_smart_chunker_graph(mock_llm, batch_size=5)

        expected_nodes = [
            "classify_chunks",
            "filter_rejected",
            "reformat_chunks",
        ]

        for node in expected_nodes:
            assert node in graph.nodes


class TestSmartChunkerAgent:
    """Tests for SmartChunkerAgent class."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(
                id="c1",
                text="Photosynthesis is the process by which plants convert sunlight into energy.",
                source_id="doc1",
                start_index=0,
                end_index=70,
                metadata={"page": 1},
            ),
            Chunk(
                id="c2",
                text="References:\n1. Smith, J. (2020). Biology Textbook. Publisher.",
                source_id="doc1",
                start_index=70,
                end_index=140,
                metadata={"page": 50},
            ),
            Chunk(
                id="c3",
                text="Table of Contents\n1. Introduction...1\n2. Methods...15",
                source_id="doc1",
                start_index=140,
                end_index=200,
                metadata={"page": 0},
            ),
        ]

    def test_agent_requires_llm(self):
        """Agent should raise error without LLM."""
        with pytest.raises(TypeError):
            SmartChunkerAgent()

    def test_agent_initialization_with_llm(self):
        """Agent should initialize with LLM."""
        mock_llm = MagicMock()
        agent = SmartChunkerAgent(llm=mock_llm, batch_size=5)

        assert agent.llm == mock_llm
        assert agent.batch_size == 5
        assert agent.compiled is not None

    def test_chunk_to_dict_conversion(self):
        """Should correctly convert Chunk to dict."""
        mock_llm = MagicMock()
        agent = SmartChunkerAgent(llm=mock_llm)

        chunk = Chunk(
            id="c1",
            text="Test content",
            source_id="doc1",
            start_index=0,
            end_index=12,
            metadata={"key": "value"},
        )

        result = agent._chunk_to_dict(chunk)

        assert result["id"] == "c1"
        assert result["text"] == "Test content"
        assert result["source_id"] == "doc1"
        assert result["start_index"] == 0
        assert result["end_index"] == 12
        assert result["metadata"]["key"] == "value"

    def test_dict_to_chunk_conversion(self):
        """Should correctly convert dict back to Chunk."""
        mock_llm = MagicMock()
        agent = SmartChunkerAgent(llm=mock_llm)

        data = {
            "id": "c1",
            "text": "Test content",
            "source_id": "doc1",
            "start_index": 0,
            "end_index": 12,
            "metadata": {"key": "value"},
        }

        result = agent._dict_to_chunk(data)

        assert isinstance(result, Chunk)
        assert result.id == "c1"
        assert result.text == "Test content"
        assert result.source_id == "doc1"
        assert result.start_index == 0
        assert result.end_index == 12
        assert result.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_process_empty_chunks(self):
        """Should handle empty chunk list."""
        mock_llm = MagicMock()
        agent = SmartChunkerAgent(llm=mock_llm)

        result, stats = await agent.process_with_stats([])

        assert result == []
        assert stats["total_input"] == 0
        assert stats["accepted"] == 0
        assert stats["rejected"] == 0


class TestStatistics:
    """Tests for statistics generation."""

    def test_stats_structure(self):
        """Stats should have expected structure."""
        from adaptive_learning_core.agents.smart_chunker_agent import _build_stats

        state = SmartChunkerState(
            input_chunks=[{"id": "c1"}, {"id": "c2"}, {"id": "c3"}],
            accepted_chunks=[{"id": "c1"}],
            rejected_chunks=[{"id": "c2"}, {"id": "c3"}],
            classification_results=[
                {"chunk_id": "c1", "category": "substantive"},
                {"chunk_id": "c2", "category": "reference"},
                {"chunk_id": "c3", "category": "table_of_contents"},
            ],
            rejection_reasons={"c2": "Bibliography", "c3": "TOC"},
        )

        reformatted = [{"id": "c1", "metadata": {"was_reformatted": True}}]

        stats = _build_stats(state, reformatted)

        assert stats["total_input"] == 3
        assert stats["accepted"] == 1
        assert stats["rejected"] == 2
        assert stats["reformatted"] == 1
        assert "rejection_breakdown" in stats
        assert stats["rejection_breakdown"]["reference"] == 1
        assert stats["rejection_breakdown"]["table_of_contents"] == 1

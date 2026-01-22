"""Tests for ParaphraserAgent scaffolding."""

import pytest

from adaptive_learning_core.agents.paraphraser_agent import (
    ParaphraserAgent,
    ParaphraserState,
    ParaphraserConfig,
    ReadabilityLevel,
    create_paraphraser_graph,
)
from adaptive_learning_core.rag.chunker import Chunk


class TestReadabilityLevel:
    """Tests for ReadabilityLevel enum."""

    def test_enum_values(self):
        """Should have expected enum values."""
        assert ReadabilityLevel.ORIGINAL.value == "original"
        assert ReadabilityLevel.SIMPLIFIED.value == "simplified"
        assert ReadabilityLevel.DETAILED.value == "detailed"
        assert ReadabilityLevel.ACADEMIC.value == "academic"
        assert ReadabilityLevel.CONVERSATIONAL.value == "conversational"

    def test_enum_from_string(self):
        """Should create enum from string value."""
        level = ReadabilityLevel("simplified")
        assert level == ReadabilityLevel.SIMPLIFIED


class TestParaphraserConfig:
    """Tests for ParaphraserConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = ParaphraserConfig()

        assert config.level == ReadabilityLevel.ORIGINAL
        assert config.preserve_technical_terms is True
        assert config.target_grade_level is None
        assert config.custom_instructions is None

    def test_custom_config(self):
        """Should accept custom values."""
        config = ParaphraserConfig(
            level=ReadabilityLevel.SIMPLIFIED,
            preserve_technical_terms=False,
            target_grade_level=8,
            custom_instructions="Use simple words",
        )

        assert config.level == ReadabilityLevel.SIMPLIFIED
        assert config.preserve_technical_terms is False
        assert config.target_grade_level == 8
        assert config.custom_instructions == "Use simple words"

    def test_to_dict(self):
        """Should serialize to dict."""
        config = ParaphraserConfig(
            level=ReadabilityLevel.ACADEMIC,
            target_grade_level=12,
        )

        result = config.to_dict()

        assert result["level"] == "academic"
        assert result["preserve_technical_terms"] is True
        assert result["target_grade_level"] == 12

    def test_from_dict(self):
        """Should deserialize from dict."""
        data = {
            "level": "conversational",
            "preserve_technical_terms": False,
            "target_grade_level": 6,
            "custom_instructions": "Be friendly",
        }

        config = ParaphraserConfig.from_dict(data)

        assert config.level == ReadabilityLevel.CONVERSATIONAL
        assert config.preserve_technical_terms is False
        assert config.target_grade_level == 6
        assert config.custom_instructions == "Be friendly"

    def test_from_dict_defaults(self):
        """Should use defaults for missing keys."""
        data = {}

        config = ParaphraserConfig.from_dict(data)

        assert config.level == ReadabilityLevel.ORIGINAL
        assert config.preserve_technical_terms is True


class TestParaphraserState:
    """Tests for ParaphraserState TypedDict."""

    def test_state_creation(self):
        """Should create valid state dict."""
        state = ParaphraserState(
            input_chunks=[],
            config=ParaphraserConfig().to_dict(),
            processing_complete=False,
        )

        assert state["config"]["level"] == "original"
        assert state["processing_complete"] is False


class TestParaphraserGraph:
    """Tests for graph creation."""

    def test_graph_creation_without_llm(self):
        """Graph should be created without LLM (scaffolding)."""
        graph = create_paraphraser_graph(llm=None)

        assert "paraphrase_chunks" in graph.nodes

    def test_graph_creation_with_llm(self):
        """Graph should be created with LLM."""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        graph = create_paraphraser_graph(llm=mock_llm)

        assert "paraphrase_chunks" in graph.nodes


class TestParaphraserAgent:
    """Tests for ParaphraserAgent class."""

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
                text="Mitochondria are the powerhouses of the cell.",
                source_id="doc1",
                start_index=70,
                end_index=115,
                metadata={"page": 2},
            ),
        ]

    def test_agent_initialization_without_llm(self):
        """Agent should initialize without LLM (scaffolding)."""
        agent = ParaphraserAgent()

        assert agent.llm is None
        assert agent.config.level == ReadabilityLevel.ORIGINAL
        assert agent.compiled is not None

    def test_agent_initialization_with_config(self):
        """Agent should accept custom config."""
        config = ParaphraserConfig(level=ReadabilityLevel.SIMPLIFIED)
        agent = ParaphraserAgent(config=config)

        assert agent.config.level == ReadabilityLevel.SIMPLIFIED

    @pytest.mark.asyncio
    async def test_process_returns_unchanged_for_original(self, sample_chunks):
        """Should return chunks unchanged when level is ORIGINAL."""
        agent = ParaphraserAgent()

        result = await agent.process(sample_chunks)

        # Should return same chunks (possibly new objects, same content)
        assert len(result) == len(sample_chunks)
        for original, processed in zip(sample_chunks, result):
            assert processed.id == original.id
            assert processed.text == original.text

    @pytest.mark.asyncio
    async def test_process_with_stats_original(self, sample_chunks):
        """Should return correct stats for ORIGINAL level."""
        agent = ParaphraserAgent()

        result, stats = await agent.process_with_stats(sample_chunks)

        assert stats["total_input"] == 2
        assert stats["total_output"] == 2
        assert stats["level"] == "original"
        assert stats["was_modified"] is False

    @pytest.mark.asyncio
    async def test_process_empty_chunks(self):
        """Should handle empty chunk list."""
        agent = ParaphraserAgent()

        result, stats = await agent.process_with_stats([])

        assert result == []
        assert stats["total_input"] == 0
        assert stats["total_output"] == 0

    @pytest.mark.asyncio
    async def test_process_with_level_override(self, sample_chunks):
        """Should accept level override parameter."""
        agent = ParaphraserAgent()  # Default: ORIGINAL

        # Override to SIMPLIFIED (scaffolding - still returns unchanged)
        result, stats = await agent.process_with_stats(
            sample_chunks,
            level=ReadabilityLevel.SIMPLIFIED,
        )

        assert stats["level"] == "simplified"
        # Scaffolding: was_modified is False since paraphrasing not implemented
        assert stats["was_modified"] is False

    @pytest.mark.asyncio
    async def test_metadata_preservation(self, sample_chunks):
        """Should preserve original metadata."""
        agent = ParaphraserAgent()

        result = await agent.process(sample_chunks)

        for original, processed in zip(sample_chunks, result):
            assert processed.metadata.get("page") == original.metadata.get("page")

    def test_chunk_to_dict_conversion(self):
        """Should correctly convert Chunk to dict."""
        agent = ParaphraserAgent()

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
        assert result["metadata"]["key"] == "value"

    def test_dict_to_chunk_conversion(self):
        """Should correctly convert dict back to Chunk."""
        agent = ParaphraserAgent()

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


class TestParaphraserScaffolding:
    """Tests to verify scaffolding behavior."""

    @pytest.mark.asyncio
    async def test_non_original_levels_return_unchanged(self):
        """Non-ORIGINAL levels should still return unchanged (scaffolding)."""
        config = ParaphraserConfig(level=ReadabilityLevel.SIMPLIFIED)
        agent = ParaphraserAgent(config=config)

        chunks = [
            Chunk(id="c1", text="Complex text", source_id="doc", start_index=0, end_index=12),
        ]

        result, stats = await agent.process_with_stats(chunks)

        # Scaffolding: content unchanged even for non-ORIGINAL levels
        assert result[0].text == "Complex text"
        assert stats["was_modified"] is False

    @pytest.mark.asyncio
    async def test_scaffolding_metadata_note(self):
        """Non-ORIGINAL levels should add scaffolding note to metadata."""
        config = ParaphraserConfig(level=ReadabilityLevel.DETAILED)
        agent = ParaphraserAgent(config=config)

        chunks = [
            Chunk(id="c1", text="Some text", source_id="doc", start_index=0, end_index=9),
        ]

        result = await agent.process(chunks)

        # Should have metadata indicating paraphrasing not implemented
        assert result[0].metadata.get("was_paraphrased") is False
        assert "not yet implemented" in result[0].metadata.get("paraphrase_note", "").lower()

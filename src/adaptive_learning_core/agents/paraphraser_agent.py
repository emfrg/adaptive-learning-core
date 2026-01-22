"""Paraphraser Agent for adjusting chunk readability.

This agent adjusts the readability and complexity level of text chunks.
Currently scaffolding - returns chunks unchanged by default.

Future implementation will support:
- Simplified mode: Reduce complexity, shorter sentences
- Detailed mode: Expand with more context and examples
- Academic mode: Formal academic tone
- Conversational mode: Informal, easy to read
"""

from dataclasses import dataclass, field
from enum import Enum
from operator import add
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from adaptive_learning_core.rag.chunker import Chunk


# =============================================================================
# Configuration
# =============================================================================


class ReadabilityLevel(str, Enum):
    """Target readability level for content."""

    ORIGINAL = "original"  # No changes (default)
    SIMPLIFIED = "simplified"  # Reduce complexity, shorter sentences
    DETAILED = "detailed"  # Expand with more context/examples
    ACADEMIC = "academic"  # Formal academic tone
    CONVERSATIONAL = "conversational"  # Informal, easy to read


@dataclass
class ParaphraserConfig:
    """Configuration for the paraphraser agent.

    Attributes:
        level: Target readability level
        preserve_technical_terms: Keep technical terms unchanged
        target_grade_level: Target reading grade level (e.g., 8 for 8th grade)
        custom_instructions: Custom paraphrasing instructions
    """

    level: ReadabilityLevel = ReadabilityLevel.ORIGINAL
    preserve_technical_terms: bool = True
    target_grade_level: int | None = None
    custom_instructions: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for state serialization."""
        return {
            "level": self.level.value,
            "preserve_technical_terms": self.preserve_technical_terms,
            "target_grade_level": self.target_grade_level,
            "custom_instructions": self.custom_instructions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParaphraserConfig":
        """Create config from dict."""
        return cls(
            level=ReadabilityLevel(data.get("level", "original")),
            preserve_technical_terms=data.get("preserve_technical_terms", True),
            target_grade_level=data.get("target_grade_level"),
            custom_instructions=data.get("custom_instructions"),
        )


# =============================================================================
# State Definition
# =============================================================================


class ParaphraserState(TypedDict, total=False):
    """State for the paraphraser agent.

    Flow: START -> paraphrase_chunks -> END
    """

    # Input
    input_chunks: list[dict[str, Any]]
    config: dict[str, Any]  # Serialized ParaphraserConfig

    # Processing
    paraphrased_chunks: list[dict[str, Any]]
    processing_complete: bool

    # Output
    stats: dict[str, Any]
    error: Optional[str]
    messages: Annotated[list[Any], add]


# =============================================================================
# Prompt Templates (for future implementation)
# =============================================================================

# These prompts are defined but not used yet - scaffolding for future
SIMPLIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at simplifying educational content while preserving all key information.

Rewrite the text to be easier to understand:
- Use shorter sentences
- Replace complex words with simpler alternatives
- Break down complex concepts into steps
- Keep all factual information intact
{preserve_terms_instruction}
{grade_level_instruction}

Return the simplified text only, no explanations.""",
        ),
        ("human", "{text}"),
    ]
)

DETAILED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at expanding educational content with helpful context.

Expand the text with:
- Additional context and background
- Helpful examples or analogies
- Clearer explanations of complex points
- Keep the core information intact, just add supporting detail
{preserve_terms_instruction}

Return the expanded text only, no explanations.""",
        ),
        ("human", "{text}"),
    ]
)

ACADEMIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at writing in formal academic style.

Rewrite the text in academic style:
- Formal tone and vocabulary
- Precise technical language
- Clear logical structure
- Keep all factual information intact
{preserve_terms_instruction}

Return the academic version only, no explanations.""",
        ),
        ("human", "{text}"),
    ]
)

CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at making educational content engaging and conversational.

Rewrite the text in a conversational style:
- Friendly, approachable tone
- Natural language flow
- Engaging without being unprofessional
- Keep all factual information intact
{preserve_terms_instruction}

Return the conversational version only, no explanations.""",
        ),
        ("human", "{text}"),
    ]
)


# =============================================================================
# Node Functions
# =============================================================================


def create_paraphrase_node(llm: BaseChatModel | None):
    """Create paraphrasing node.

    Currently returns chunks unchanged (scaffolding).

    Args:
        llm: LangChain chat model (optional for scaffolding)

    Returns:
        Async node function
    """

    async def paraphrase_chunks(state: ParaphraserState) -> dict[str, Any]:
        """Paraphrase chunks based on config.

        Currently no-op - returns chunks unchanged.
        """
        input_chunks = state.get("input_chunks", [])
        config_dict = state.get("config", {})
        config = ParaphraserConfig.from_dict(config_dict)

        # Fast path: if ORIGINAL level, return unchanged
        if config.level == ReadabilityLevel.ORIGINAL:
            return {
                "paraphrased_chunks": input_chunks,
                "processing_complete": True,
                "stats": {
                    "total_input": len(input_chunks),
                    "total_output": len(input_chunks),
                    "level": config.level.value,
                    "was_modified": False,
                },
                "messages": [
                    AIMessage(content=f"Paraphrasing skipped (level=ORIGINAL): {len(input_chunks)} chunks")
                ],
            }

        # Future: Implement actual paraphrasing based on config.level
        # For now, return unchanged with a note
        paraphrased = []
        for chunk in input_chunks:
            paraphrased_chunk = {
                **chunk,
                "metadata": {
                    **chunk.get("metadata", {}),
                    "paraphrase_level": config.level.value,
                    "was_paraphrased": False,  # Not implemented yet
                    "paraphrase_note": "Paraphrasing not yet implemented",
                },
            }
            paraphrased.append(paraphrased_chunk)

        return {
            "paraphrased_chunks": paraphrased,
            "processing_complete": True,
            "stats": {
                "total_input": len(input_chunks),
                "total_output": len(paraphrased),
                "level": config.level.value,
                "was_modified": False,  # Not implemented yet
            },
            "messages": [
                AIMessage(
                    content=f"Paraphrasing scaffolding: {len(paraphrased)} chunks (level={config.level.value}, not yet implemented)"
                )
            ],
        }

    return paraphrase_chunks


# =============================================================================
# Graph Creation
# =============================================================================


def create_paraphraser_graph(llm: BaseChatModel | None = None) -> StateGraph:
    """Create the paraphraser state graph.

    Flow: START -> paraphrase_chunks -> END

    Args:
        llm: LangChain chat model (optional for scaffolding)

    Returns:
        StateGraph for paraphrasing
    """
    graph = StateGraph(ParaphraserState)

    # Add nodes
    graph.add_node("paraphrase_chunks", create_paraphrase_node(llm))

    # Add edges
    graph.add_edge(START, "paraphrase_chunks")
    graph.add_edge("paraphrase_chunks", END)

    return graph


# =============================================================================
# Agent Class
# =============================================================================


class ParaphraserAgent:
    """Agent for adjusting chunk readability.

    Currently returns chunks unchanged (scaffolding).
    Future: Implements actual paraphrasing based on config.

    Example:
        >>> from adaptive_learning_core.agents import ParaphraserAgent, ParaphraserConfig, ReadabilityLevel
        >>>
        >>> # Default: no changes
        >>> agent = ParaphraserAgent()
        >>> result = await agent.process(chunks)  # Returns unchanged
        >>>
        >>> # Future: simplified readability
        >>> config = ParaphraserConfig(level=ReadabilityLevel.SIMPLIFIED)
        >>> agent = ParaphraserAgent(llm=llm, config=config)
        >>> result = await agent.process(chunks)
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        config: ParaphraserConfig | None = None,
        checkpointer: Any | None = None,
    ):
        """Initialize the paraphraser agent.

        Args:
            llm: LangChain chat model (optional for scaffolding)
            config: Paraphraser configuration
            checkpointer: Optional LangGraph checkpointer
        """
        self.llm = llm
        self.config = config or ParaphraserConfig()
        self.checkpointer = checkpointer or MemorySaver()

        # Create and compile graph
        self.graph = create_paraphraser_graph(llm)
        self.compiled = self.graph.compile(checkpointer=self.checkpointer)

    async def process(
        self,
        chunks: list[Chunk],
        level: ReadabilityLevel | None = None,
        thread_id: str = "default",
    ) -> list[Chunk]:
        """Process chunks through paraphrasing.

        Args:
            chunks: Input chunks
            level: Override readability level (optional)
            thread_id: Thread ID for checkpointing

        Returns:
            Paraphrased chunks (currently unchanged)
        """
        processed, _ = await self.process_with_stats(chunks, level, thread_id)
        return processed

    async def process_with_stats(
        self,
        chunks: list[Chunk],
        level: ReadabilityLevel | None = None,
        thread_id: str = "default",
    ) -> tuple[list[Chunk], dict[str, Any]]:
        """Process chunks and return statistics.

        Args:
            chunks: Input chunks
            level: Override readability level (optional)
            thread_id: Thread ID for checkpointing

        Returns:
            Tuple of (processed chunks, statistics dict)
        """
        if not chunks:
            return [], {
                "total_input": 0,
                "total_output": 0,
                "level": (level or self.config.level).value,
                "was_modified": False,
            }

        # Use override level or config level
        effective_config = self.config
        if level is not None:
            effective_config = ParaphraserConfig(
                level=level,
                preserve_technical_terms=self.config.preserve_technical_terms,
                target_grade_level=self.config.target_grade_level,
                custom_instructions=self.config.custom_instructions,
            )

        # Fast path: if ORIGINAL level, skip graph entirely
        if effective_config.level == ReadabilityLevel.ORIGINAL:
            return chunks, {
                "total_input": len(chunks),
                "total_output": len(chunks),
                "level": ReadabilityLevel.ORIGINAL.value,
                "was_modified": False,
            }

        # Convert Chunk objects to dicts for state
        input_chunks = [self._chunk_to_dict(c) for c in chunks]

        initial_state = ParaphraserState(
            input_chunks=input_chunks,
            config=effective_config.to_dict(),
            paraphrased_chunks=[],
            processing_complete=False,
            stats={},
            messages=[],
        )

        config = {"configurable": {"thread_id": thread_id}}
        result = await self.compiled.ainvoke(initial_state, config)

        # Convert dicts back to Chunk objects
        processed_chunks = [
            self._dict_to_chunk(c) for c in result.get("paraphrased_chunks", [])
        ]

        stats = result.get("stats", {})

        return processed_chunks, stats

    def _chunk_to_dict(self, chunk: Chunk) -> dict[str, Any]:
        """Convert Chunk to dict for state."""
        return {
            "id": chunk.id,
            "text": chunk.text,
            "source_id": chunk.source_id,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "metadata": chunk.metadata.copy() if chunk.metadata else {},
        }

    def _dict_to_chunk(self, data: dict[str, Any]) -> Chunk:
        """Convert dict back to Chunk."""
        return Chunk(
            id=data["id"],
            text=data["text"],
            source_id=data["source_id"],
            start_index=data["start_index"],
            end_index=data["end_index"],
            metadata=data.get("metadata", {}),
        )

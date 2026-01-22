"""Smart Chunker Agent for filtering and reformatting text chunks.

This agent processes raw text chunks from PDFs to create high-quality learning cards by:
1. Filtering out non-useful chunks (references, TOC, acknowledgments, etc.)
2. Reformatting useful chunks to improve clarity and structure
3. Preserving all substantive detail (no information loss)
"""

import json
from operator import add
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from adaptive_learning_core.rag.chunker import Chunk


# =============================================================================
# State Definition
# =============================================================================


class SmartChunkerState(TypedDict, total=False):
    """State for the smart chunker agent.

    Flow: START -> classify_chunks -> filter_rejected -> reformat_chunks -> END
    """

    # Input
    input_chunks: list[dict[str, Any]]
    batch_size: int

    # Classification
    classification_results: list[dict[str, Any]]  # {chunk_id, category, confidence, reason}
    current_batch_index: int

    # Filtering
    accepted_chunks: list[dict[str, Any]]
    rejected_chunks: list[dict[str, Any]]
    rejection_reasons: dict[str, str]

    # Reformatting
    reformatted_chunks: list[dict[str, Any]]

    # Output
    processing_complete: bool
    stats: dict[str, Any]
    error: Optional[str]

    # Messages for debugging
    messages: Annotated[list[Any], add]


# =============================================================================
# Prompt Templates
# =============================================================================

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at analyzing educational content to identify what is useful for learning.

Classify each chunk into one of these categories:
- substantive: Core educational content that teaches concepts, explains ideas, provides examples, or contains important information
- reference: Bibliography, citations, works cited, references sections
- table_of_contents: Chapter listings, section headers with page numbers, navigation lists
- acknowledgment: Thank you notes, dedications, author bios, preface acknowledgments
- index: Alphabetical index entries, glossary page references
- boilerplate: Copyright notices, publisher info, ISBN, legal disclaimers
- navigation: Page numbers alone, chapter markers without content, headers/footers
- empty: Whitespace only, minimal text with no educational meaning

For each chunk, provide:
1. category: The classification category (one of the above)
2. confidence: Your confidence score (0.0-1.0)
3. reason: Brief explanation for the classification (1 sentence)

Return as a JSON array with objects containing chunk_id, category, confidence, and reason.""",
        ),
        (
            "human",
            """Classify these {count} chunks:

{chunks_json}

Return a JSON array:""",
        ),
    ]
)

REFORMAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are cleaning up educational content for a learning card.

Your job is to FIX formatting issues, NOT rewrite or restructure the content.

DO:
- Fix OCR artifacts and broken words/sentences
- Remove redundant whitespace
- Output as markdown
- Add minimal formatting where it helps (a bullet point or **bold** here and there)
- Remove citation markers like (5), [5], [Smith 2020], superscript numbers, and footnote references
- Remove end-of-text reference lists or bibliographies

DO NOT:
- Rewrite or paraphrase the text
- Add structure that wasn't there (don't create bullet lists from paragraphs)
- Remove or summarize educational content
- Change technical terms, formulas, or equations

EXCEPTION for citations: Keep attributions that add educational value.
- REMOVE: "The mitochondria is the powerhouse of the cell [5]."
- KEEP: "As Plato argued in *The Republic*, justice is..."
- KEEP: "Einstein's famous equation E=mc²..."

The goal is to remove reference noise while preserving meaningful attributions.

Return a JSON object with chunk_id and reformatted_text.""",
        ),
        (
            "human",
            """Clean up this chunk:

ID: {chunk_id}

Original text:
---
{text}
---

Return JSON with chunk_id and reformatted_text:""",
        ),
    ]
)

# Categories that should be rejected (not useful for learning)
REJECTED_CATEGORIES = {
    "reference",
    "table_of_contents",
    "acknowledgment",
    "index",
    "boilerplate",
    "navigation",
    "empty",
}


# =============================================================================
# Node Functions
# =============================================================================


def create_classify_node(llm: BaseChatModel, batch_size: int = 5):
    """Create classification node with LLM.

    Processes chunks in batches for efficiency.

    Args:
        llm: LangChain chat model
        batch_size: Number of chunks to classify per LLM call

    Returns:
        Async node function
    """
    parser = JsonOutputParser()

    async def classify_chunks(state: SmartChunkerState) -> dict[str, Any]:
        """Classify chunks as useful or junk."""
        input_chunks = state.get("input_chunks", [])
        batch_size_val = state.get("batch_size", batch_size)
        current_batch = state.get("current_batch_index", 0)

        # Calculate batch boundaries
        start_idx = current_batch * batch_size_val
        end_idx = min(start_idx + batch_size_val, len(input_chunks))
        batch = input_chunks[start_idx:end_idx]

        if not batch:
            return {
                "processing_complete": True,
                "messages": [AIMessage(content="No more chunks to classify")],
            }

        # Prepare batch for LLM - truncate text for classification
        chunks_for_llm = []
        for c in batch:
            chunks_for_llm.append(
                {
                    "chunk_id": c["id"],
                    "text": c["text"][:1500],  # Truncate for classification
                }
            )
        chunks_json = json.dumps(chunks_for_llm, indent=2)

        chain = CLASSIFICATION_PROMPT | llm | parser

        try:
            results = await chain.ainvoke(
                {
                    "count": len(batch),
                    "chunks_json": chunks_json,
                }
            )

            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]

            existing_results = state.get("classification_results", [])

            return {
                "classification_results": existing_results + results,
                "current_batch_index": current_batch + 1,
                "messages": [
                    AIMessage(
                        content=f"Classified batch {current_batch + 1}: {len(results)} chunks"
                    )
                ],
            }
        except Exception as e:
            return {
                "error": f"Classification error: {e}",
                "current_batch_index": current_batch + 1,  # Move forward even on error
                "messages": [AIMessage(content=f"Classification error in batch {current_batch + 1}: {e}")],
            }

    return classify_chunks


def filter_rejected(state: SmartChunkerState) -> dict[str, Any]:
    """Filter chunks based on classification results.

    Non-LLM node - pure logic.
    """
    input_chunks = state.get("input_chunks", [])
    classifications = state.get("classification_results", [])

    # Create lookup by chunk_id
    class_lookup = {c.get("chunk_id"): c for c in classifications if "chunk_id" in c}

    accepted = []
    rejected = []
    rejection_reasons: dict[str, str] = {}

    for chunk in input_chunks:
        chunk_id = chunk["id"]
        classification = class_lookup.get(chunk_id)

        if classification is None:
            # Not classified - accept by default (safe fallback)
            accepted.append(chunk)
        elif classification.get("category") in REJECTED_CATEGORIES:
            rejected.append(chunk)
            rejection_reasons[chunk_id] = classification.get("reason", "Classified as non-educational")
        else:
            accepted.append(chunk)

    return {
        "accepted_chunks": accepted,
        "rejected_chunks": rejected,
        "rejection_reasons": rejection_reasons,
        "messages": [
            AIMessage(content=f"Filtered: {len(accepted)} accepted, {len(rejected)} rejected")
        ],
    }


def create_reformat_node(llm: BaseChatModel):
    """Create reformatting node with LLM.

    Processes accepted chunks to improve formatting.

    Args:
        llm: LangChain chat model

    Returns:
        Async node function
    """
    parser = JsonOutputParser()

    async def reformat_chunks(state: SmartChunkerState) -> dict[str, Any]:
        """Reformat accepted chunks for better readability."""
        accepted = state.get("accepted_chunks", [])

        if not accepted:
            return {
                "reformatted_chunks": [],
                "processing_complete": True,
                "stats": _build_stats(state, []),
                "messages": [AIMessage(content="No chunks to reformat")],
            }

        reformatted = []
        chain = REFORMAT_PROMPT | llm | parser

        for chunk in accepted:
            try:
                result = await chain.ainvoke(
                    {
                        "chunk_id": chunk["id"],
                        "text": chunk["text"],
                    }
                )

                # Create reformatted chunk preserving all metadata
                reformatted_chunk = {
                    **chunk,
                    "text": result.get("reformatted_text", chunk["text"]),
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "original_text": chunk["text"],
                        "was_reformatted": True,
                    },
                }
                reformatted.append(reformatted_chunk)

            except Exception as e:
                # On error, keep original unchanged
                reformatted_chunk = {
                    **chunk,
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "reformat_error": str(e),
                        "was_reformatted": False,
                    },
                }
                reformatted.append(reformatted_chunk)

        stats = _build_stats(state, reformatted)

        return {
            "reformatted_chunks": reformatted,
            "processing_complete": True,
            "stats": stats,
            "messages": [AIMessage(content=f"Reformatted {len(reformatted)} chunks")],
        }

    return reformat_chunks


def _build_stats(state: SmartChunkerState, reformatted: list[dict[str, Any]]) -> dict[str, Any]:
    """Build statistics dictionary."""
    classifications = state.get("classification_results", [])
    rejected_ids = {c["id"] for c in state.get("rejected_chunks", [])}

    # Count rejections by category
    rejection_breakdown: dict[str, int] = {}
    for c in classifications:
        if c.get("chunk_id") in rejected_ids:
            category = c.get("category", "unknown")
            rejection_breakdown[category] = rejection_breakdown.get(category, 0) + 1

    return {
        "total_input": len(state.get("input_chunks", [])),
        "accepted": len(state.get("accepted_chunks", [])),
        "rejected": len(state.get("rejected_chunks", [])),
        "reformatted": sum(1 for c in reformatted if c.get("metadata", {}).get("was_reformatted")),
        "reformat_errors": sum(1 for c in reformatted if "reformat_error" in c.get("metadata", {})),
        "rejection_breakdown": rejection_breakdown,
        "rejection_reasons": list(state.get("rejection_reasons", {}).values()),
    }


def should_continue_classification(state: SmartChunkerState) -> str:
    """Determine if more classification batches are needed."""
    input_chunks = state.get("input_chunks", [])
    batch_size = state.get("batch_size", 5)
    current_batch = state.get("current_batch_index", 0)

    total_batches = (len(input_chunks) + batch_size - 1) // batch_size if input_chunks else 0

    if current_batch >= total_batches:
        return "filter_rejected"
    else:
        return "classify_chunks"


# =============================================================================
# Graph Creation
# =============================================================================


def create_smart_chunker_graph(llm: BaseChatModel, batch_size: int = 5) -> StateGraph:
    """Create the smart chunker state graph.

    Flow:
        START -> classify_chunks -> (loop or filter) -> reformat_chunks -> END

    Args:
        llm: LangChain chat model
        batch_size: Number of chunks per classification batch

    Returns:
        StateGraph for smart chunking
    """
    graph = StateGraph(SmartChunkerState)

    # Add nodes
    graph.add_node("classify_chunks", create_classify_node(llm, batch_size))
    graph.add_node("filter_rejected", filter_rejected)
    graph.add_node("reformat_chunks", create_reformat_node(llm))

    # Add edges
    graph.add_edge(START, "classify_chunks")

    # Conditional: continue classifying or move to filter
    graph.add_conditional_edges(
        "classify_chunks",
        should_continue_classification,
        {
            "classify_chunks": "classify_chunks",  # More batches
            "filter_rejected": "filter_rejected",  # All classified
        },
    )

    graph.add_edge("filter_rejected", "reformat_chunks")
    graph.add_edge("reformat_chunks", END)

    return graph


# =============================================================================
# Agent Class
# =============================================================================


class SmartChunkerAgent:
    """Agent for filtering and reformatting text chunks.

    Filters out non-educational content (references, TOC, acknowledgments)
    and reformats remaining chunks into optimal learning card format.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> agent = SmartChunkerAgent(llm=llm, batch_size=5)
        >>>
        >>> chunks = chunker.chunk(document_text, "doc_1")
        >>> processed = await agent.process(chunks)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        batch_size: int = 5,
        checkpointer: Optional[Any] = None,
    ):
        """Initialize the smart chunker agent.

        Args:
            llm: LangChain chat model (required)
            batch_size: Number of chunks to classify per LLM call
            checkpointer: Optional LangGraph checkpointer
        """
        self.llm = llm
        self.batch_size = batch_size
        self.checkpointer = checkpointer or MemorySaver()

        # Create and compile graph
        self.graph = create_smart_chunker_graph(llm, batch_size)
        self.compiled = self.graph.compile(checkpointer=self.checkpointer)

    async def process(
        self,
        chunks: list[Chunk],
        thread_id: str = "default",
    ) -> list[Chunk]:
        """Process chunks, filtering and reformatting.

        Args:
            chunks: List of Chunk objects from a chunker
            thread_id: Thread ID for checkpointing

        Returns:
            List of processed Chunk objects (filtered and reformatted)
        """
        processed, _ = await self.process_with_stats(chunks, thread_id)
        return processed

    async def process_with_stats(
        self,
        chunks: list[Chunk],
        thread_id: str = "default",
    ) -> tuple[list[Chunk], dict[str, Any]]:
        """Process chunks and return statistics.

        Args:
            chunks: List of Chunk objects
            thread_id: Thread ID for checkpointing

        Returns:
            Tuple of (processed chunks, statistics dict)
            Statistics include: total_input, accepted, rejected,
            reformatted, rejection_breakdown, rejection_reasons
        """
        if not chunks:
            return [], {
                "total_input": 0,
                "accepted": 0,
                "rejected": 0,
                "reformatted": 0,
                "reformat_errors": 0,
                "rejection_breakdown": {},
                "rejection_reasons": [],
            }

        # Convert Chunk objects to dicts for state
        input_chunks = [self._chunk_to_dict(c) for c in chunks]

        initial_state = SmartChunkerState(
            input_chunks=input_chunks,
            batch_size=self.batch_size,
            classification_results=[],
            accepted_chunks=[],
            rejected_chunks=[],
            rejection_reasons={},
            reformatted_chunks=[],
            current_batch_index=0,
            processing_complete=False,
            stats={},
            messages=[],
        )

        config = {"configurable": {"thread_id": thread_id}}
        result = await self.compiled.ainvoke(initial_state, config)

        # Convert dicts back to Chunk objects
        processed_chunks = [
            self._dict_to_chunk(c) for c in result.get("reformatted_chunks", [])
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

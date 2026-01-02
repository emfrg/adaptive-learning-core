"""PDF/text chunking strategies for creating learning cards.

Provides multiple chunking strategies for splitting learning materials
into appropriately sized chunks (cards) for the adaptive learning system.

Chunking Strategies:
1. TokenChunker - Fixed token size with overlap
2. SentenceChunker - Sentence-based with configurable window
3. SemanticChunker - Section/paragraph based (semantic boundaries)
"""

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from adaptive_learning_core.models.card import Card


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for text chunking.

    Attributes:
        chunk_size: Target size of each chunk (tokens or sentences)
        chunk_overlap: Overlap between consecutive chunks
        min_chunk_size: Minimum chunk size to keep
        max_chunk_size: Maximum chunk size before forced split
        separator: Text separator for joining (token-based)
    """

    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    separator: str = " "


@dataclass
class Chunk:
    """A chunk of text extracted from source material.

    Attributes:
        id: Unique identifier for this chunk
        text: The chunk text content
        source_id: ID of the source document
        start_index: Character start position in source
        end_index: Character end position in source
        metadata: Additional metadata (section title, page number, etc.)
    """

    id: str
    text: str
    source_id: str
    start_index: int
    end_index: int
    metadata: dict = field(default_factory=dict)

    def to_card(self, module_id: str) -> Card:
        """Convert this chunk to a Card for the learning system.

        Args:
            module_id: The module this card belongs to

        Returns:
            Card instance
        """
        return Card(
            id=self.id,
            module_id=module_id,
            text=self.text,
            source_document=self.source_id,
            metadata=self.metadata,
        )

    @property
    def token_count_estimate(self) -> int:
        """Rough estimate of token count (words / 0.75)."""
        return int(len(self.text.split()) / 0.75)


class TextChunker(ABC):
    """Abstract base class for text chunking strategies."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize with optional configuration.

        Args:
            config: Chunking configuration, or None for defaults
        """
        self.config = config or ChunkingConfig()

    @abstractmethod
    def chunk(self, text: str, source_id: str) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: The text to chunk
            source_id: Identifier for the source document

        Returns:
            List of Chunk objects
        """
        pass

    def _generate_chunk_id(self, source_id: str, index: int) -> str:
        """Generate a unique ID for a chunk.

        Args:
            source_id: Source document ID
            index: Chunk index within the document

        Returns:
            Unique chunk ID
        """
        return f"{source_id}_chunk_{index}_{uuid.uuid4().hex[:8]}"


class TokenChunker(TextChunker):
    """Chunks text by approximate token count.

    Uses word count as a proxy for tokens (assuming ~0.75 tokens per word
    for English text). Creates overlapping chunks for context preservation.

    Example:
        >>> chunker = TokenChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
        >>> chunks = chunker.chunk(long_text, "doc1")
    """

    def chunk(self, text: str, source_id: str) -> list[Chunk]:
        """Split text into token-based chunks with overlap.

        Args:
            text: The text to chunk
            source_id: Source document ID

        Returns:
            List of Chunk objects
        """
        words = text.split()
        if not words:
            return []

        # Convert token counts to word counts (rough approximation)
        words_per_chunk = int(self.config.chunk_size * 0.75)
        overlap_words = int(self.config.chunk_overlap * 0.75)
        min_words = int(self.config.min_chunk_size * 0.75)

        chunks = []
        start_word_idx = 0
        chunk_index = 0

        while start_word_idx < len(words):
            end_word_idx = min(start_word_idx + words_per_chunk, len(words))

            # Get chunk words
            chunk_words = words[start_word_idx:end_word_idx]

            # Skip if too small (unless it's the last chunk)
            if len(chunk_words) < min_words and chunks:
                break

            chunk_text = self.config.separator.join(chunk_words)

            # Calculate character positions
            start_char = len(self.config.separator.join(words[:start_word_idx]))
            if start_word_idx > 0:
                start_char += len(self.config.separator)
            end_char = start_char + len(chunk_text)

            chunk = Chunk(
                id=self._generate_chunk_id(source_id, chunk_index),
                text=chunk_text,
                source_id=source_id,
                start_index=start_char,
                end_index=end_char,
                metadata={"chunk_index": chunk_index, "method": "token"},
            )
            chunks.append(chunk)

            # Move start position with overlap
            start_word_idx = end_word_idx - overlap_words
            if start_word_idx >= end_word_idx:
                break
            chunk_index += 1

        return chunks


class SentenceChunker(TextChunker):
    """Chunks text by sentences with a sliding window.

    Splits text into sentences and groups them to approximately
    meet the target chunk size. This preserves sentence boundaries
    for better semantic coherence.

    Example:
        >>> chunker = SentenceChunker(ChunkingConfig(chunk_size=500))
        >>> chunks = chunker.chunk(long_text, "doc1")
    """

    # Sentence-ending patterns
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Basic sentence splitting
        sentences = self.SENTENCE_PATTERN.split(text)
        # Clean up
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, source_id: str) -> list[Chunk]:
        """Split text into sentence-based chunks.

        Args:
            text: The text to chunk
            source_id: Source document ID

        Returns:
            List of Chunk objects
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Estimate words per sentence for chunking
        target_words = int(self.config.chunk_size * 0.75)
        overlap_words = int(self.config.chunk_overlap * 0.75)

        chunks = []
        current_sentences: list[str] = []
        current_word_count = 0
        chunk_index = 0
        char_position = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            # Check if adding this sentence exceeds max size
            if (
                current_word_count + sentence_word_count > target_words
                and current_sentences
            ):
                # Create chunk from current sentences
                chunk_text = " ".join(current_sentences)
                chunk = Chunk(
                    id=self._generate_chunk_id(source_id, chunk_index),
                    text=chunk_text,
                    source_id=source_id,
                    start_index=char_position,
                    end_index=char_position + len(chunk_text),
                    metadata={
                        "chunk_index": chunk_index,
                        "method": "sentence",
                        "sentence_count": len(current_sentences),
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

                # Keep overlap sentences
                overlap_sentences = []
                overlap_count = 0
                for s in reversed(current_sentences):
                    if overlap_count + len(s.split()) <= overlap_words:
                        overlap_sentences.insert(0, s)
                        overlap_count += len(s.split())
                    else:
                        break

                current_sentences = overlap_sentences
                current_word_count = overlap_count
                char_position += len(chunk_text) - len(" ".join(overlap_sentences))

            current_sentences.append(sentence)
            current_word_count += sentence_word_count

        # Handle remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = Chunk(
                id=self._generate_chunk_id(source_id, chunk_index),
                text=chunk_text,
                source_id=source_id,
                start_index=char_position,
                end_index=char_position + len(chunk_text),
                metadata={
                    "chunk_index": chunk_index,
                    "method": "sentence",
                    "sentence_count": len(current_sentences),
                },
            )
            chunks.append(chunk)

        return chunks


class SemanticChunker(TextChunker):
    """Chunks text by semantic boundaries (sections, paragraphs).

    Attempts to split at natural document boundaries like:
    - Section headers
    - Paragraph breaks
    - Topic transitions

    Falls back to sentence-based chunking if sections are too large.

    Example:
        >>> chunker = SemanticChunker()
        >>> chunks = chunker.chunk(textbook_content, "textbook_ch1")
    """

    # Patterns for detecting section boundaries
    SECTION_PATTERNS = [
        re.compile(r"\n#{1,6}\s+.+\n"),  # Markdown headers
        re.compile(r"\n[A-Z][^.]*:\s*\n"),  # Title-case headers with colon
        re.compile(r"\n\d+\.\d*\s+[A-Z].+\n"),  # Numbered sections (1.1 Title)
        re.compile(r"\n\n\n+"),  # Multiple newlines (section breaks)
    ]

    PARAGRAPH_PATTERN = re.compile(r"\n\n+")

    def _find_section_breaks(self, text: str) -> list[int]:
        """Find positions of section breaks in text.

        Args:
            text: Text to analyze

        Returns:
            Sorted list of break positions
        """
        breaks = set()
        for pattern in self.SECTION_PATTERNS:
            for match in pattern.finditer(text):
                breaks.add(match.start())
        return sorted(breaks)

    def _find_paragraph_breaks(self, text: str) -> list[int]:
        """Find positions of paragraph breaks.

        Args:
            text: Text to analyze

        Returns:
            Sorted list of break positions
        """
        breaks = []
        for match in self.PARAGRAPH_PATTERN.finditer(text):
            breaks.append(match.start())
        return breaks

    def chunk(self, text: str, source_id: str) -> list[Chunk]:
        """Split text at semantic boundaries.

        First tries section breaks, then paragraph breaks,
        falling back to sentence-based chunking for large sections.

        Args:
            text: The text to chunk
            source_id: Source document ID

        Returns:
            List of Chunk objects
        """
        # Find all potential break points
        section_breaks = self._find_section_breaks(text)
        paragraph_breaks = self._find_paragraph_breaks(text)

        # Combine and sort breaks
        all_breaks = sorted(set(section_breaks + paragraph_breaks + [0, len(text)]))

        chunks = []
        chunk_index = 0
        sentence_chunker = SentenceChunker(self.config)

        for i in range(len(all_breaks) - 1):
            start = all_breaks[i]
            end = all_breaks[i + 1]
            section_text = text[start:end].strip()

            if not section_text:
                continue

            word_count = len(section_text.split())
            max_words = int(self.config.max_chunk_size * 0.75)
            min_words = int(self.config.min_chunk_size * 0.75)

            # If section is too large, use sentence chunker
            if word_count > max_words:
                sub_chunks = sentence_chunker.chunk(section_text, source_id)
                for sc in sub_chunks:
                    sc.metadata["parent_section_start"] = start
                    sc.metadata["method"] = "semantic_sentence"
                    sc.start_index += start
                    sc.end_index += start
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

            # If section is reasonable size, use as-is
            elif word_count >= min_words:
                is_section = start in section_breaks
                chunk = Chunk(
                    id=self._generate_chunk_id(source_id, chunk_index),
                    text=section_text,
                    source_id=source_id,
                    start_index=start,
                    end_index=end,
                    metadata={
                        "chunk_index": chunk_index,
                        "method": "semantic",
                        "is_section": is_section,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

            # If section is too small, try to merge with next
            # (handled by continuing to next iteration)

        # Merge small chunks if needed
        return self._merge_small_chunks(chunks)

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge chunks that are too small.

        Args:
            chunks: List of chunks to potentially merge

        Returns:
            List with small chunks merged
        """
        if len(chunks) <= 1:
            return chunks

        min_words = int(self.config.min_chunk_size * 0.75)
        max_words = int(self.config.max_chunk_size * 0.75)

        merged = []
        current_text = ""
        current_start = 0
        current_metadata: dict = {}

        for chunk in chunks:
            word_count = len(chunk.text.split())
            current_word_count = len(current_text.split()) if current_text else 0

            # If current + new exceeds max, save current and start new
            if current_text and current_word_count + word_count > max_words:
                merged.append(
                    Chunk(
                        id=self._generate_chunk_id(chunk.source_id, len(merged)),
                        text=current_text.strip(),
                        source_id=chunk.source_id,
                        start_index=current_start,
                        end_index=current_start + len(current_text),
                        metadata={**current_metadata, "merged": True},
                    )
                )
                current_text = chunk.text
                current_start = chunk.start_index
                current_metadata = chunk.metadata.copy()

            # If current is too small, append this chunk
            elif current_word_count < min_words:
                if current_text:
                    current_text += "\n\n" + chunk.text
                else:
                    current_text = chunk.text
                    current_start = chunk.start_index
                    current_metadata = chunk.metadata.copy()

            # Otherwise, save current and start new
            else:
                if current_text:
                    merged.append(
                        Chunk(
                            id=self._generate_chunk_id(chunk.source_id, len(merged)),
                            text=current_text.strip(),
                            source_id=chunk.source_id,
                            start_index=current_start,
                            end_index=current_start + len(current_text),
                            metadata=current_metadata,
                        )
                    )
                current_text = chunk.text
                current_start = chunk.start_index
                current_metadata = chunk.metadata.copy()

        # Don't forget the last chunk
        if current_text:
            merged.append(
                Chunk(
                    id=self._generate_chunk_id(chunks[0].source_id, len(merged)),
                    text=current_text.strip(),
                    source_id=chunks[0].source_id,
                    start_index=current_start,
                    end_index=current_start + len(current_text),
                    metadata=current_metadata,
                )
            )

        return merged

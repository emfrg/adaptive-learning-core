"""Bloom's Taxonomy classifier for question difficulty.

Uses LLM to classify questions into Bloom's cognitive levels,
enabling adaptive question selection based on learner stereotype.

Two classifiers are provided:
1. BloomClassifier - LLM-based, more accurate
2. HeuristicBloomClassifier - Rule-based, for testing/fallback
"""

from dataclasses import dataclass
from typing import Optional, Protocol

from adaptive_learning_core.models.question import BloomLevel
from adaptive_learning_core.bloom.prompts import BLOOM_CLASSIFICATION_PROMPT
from adaptive_learning_core.bloom.taxonomy import BLOOM_LEVEL_INFO


class LLMProvider(Protocol):
    """Protocol for LLM providers.

    Any LLM implementation must support this interface
    for use with the BloomClassifier.
    """

    async def complete(self, prompt: str) -> str:
        """Generate completion for a prompt.

        Args:
            prompt: The prompt to complete

        Returns:
            The LLM's response text
        """
        ...


@dataclass
class ClassificationResult:
    """Result of a Bloom's Taxonomy classification.

    Attributes:
        level: The classified Bloom level
        confidence: Confidence score (0-1), if available
        raw_response: The raw LLM response (for debugging)
        method: Classification method used ("llm" or "heuristic")
    """

    level: BloomLevel
    confidence: Optional[float] = None
    raw_response: Optional[str] = None
    method: str = "llm"

    @property
    def level_name(self) -> str:
        """Human-readable level name."""
        return BLOOM_LEVEL_INFO[self.level].name

    @property
    def level_description(self) -> str:
        """Description of what this level tests."""
        return BLOOM_LEVEL_INFO[self.level].description


class BloomClassifier:
    """Classifies questions according to Bloom's Taxonomy using LLM.

    Uses an LLM to analyze question stems and determine their
    cognitive complexity level (Remember through Create).

    This enables the adaptive system to match question difficulty
    to learner stereotype levels.

    Example:
        >>> classifier = BloomClassifier(llm_provider)
        >>> result = await classifier.classify(
        ...     "What is the capital of France?",
        ...     "Paris"
        ... )
        >>> print(f"Level: {result.level_name}")  # "Remember"
    """

    def __init__(self, llm: LLMProvider):
        """Initialize the classifier with an LLM provider.

        Args:
            llm: An LLM provider implementing the LLMProvider protocol
        """
        self.llm = llm

    def _build_prompt(self, question_stem: str, answer: str) -> str:
        """Build classification prompt.

        Args:
            question_stem: The question text
            answer: The correct answer

        Returns:
            Formatted prompt for the LLM
        """
        return BLOOM_CLASSIFICATION_PROMPT.format(
            question=question_stem,
            answer=answer,
        )

    def _parse_response(self, response: str) -> BloomLevel:
        """Parse LLM response to BloomLevel.

        Handles various response formats:
        - "LEVEL: <level_name>"
        - Just the level name
        - Level number

        Args:
            response: Raw LLM response

        Returns:
            Parsed BloomLevel
        """
        response_lower = response.lower().strip()

        # Try to match level names
        level_keywords: dict[BloomLevel, list[str]] = {
            BloomLevel.REMEMBER: ["remember", "recall", "1", "level 1", "level: remember"],
            BloomLevel.UNDERSTAND: [
                "understand",
                "comprehend",
                "2",
                "level 2",
                "level: understand",
            ],
            BloomLevel.APPLY: ["apply", "3", "level 3", "level: apply"],
            BloomLevel.ANALYZE: [
                "analyze",
                "analyse",
                "4",
                "level 4",
                "level: analyze",
                "level: analyse",
            ],
            BloomLevel.EVALUATE: ["evaluate", "5", "level 5", "level: evaluate"],
            BloomLevel.CREATE: [
                "create",
                "synthesize",
                "6",
                "level 6",
                "level: create",
            ],
        }

        for level, keywords in level_keywords.items():
            if any(kw in response_lower for kw in keywords):
                return level

        # Default to UNDERSTAND if unclear (middle ground)
        return BloomLevel.UNDERSTAND

    async def classify(
        self,
        question_stem: str,
        correct_answer: str,
    ) -> ClassificationResult:
        """Classify a question's cognitive level.

        Args:
            question_stem: The question text
            correct_answer: The correct answer

        Returns:
            ClassificationResult with the determined level
        """
        prompt = self._build_prompt(question_stem, correct_answer)
        response = await self.llm.complete(prompt)
        level = self._parse_response(response)

        return ClassificationResult(
            level=level,
            raw_response=response,
            method="llm",
        )

    async def classify_batch(
        self,
        questions: list[tuple[str, str]],  # (stem, answer)
    ) -> list[ClassificationResult]:
        """Classify multiple questions.

        Currently processes sequentially. Could be optimized
        with batch API calls or parallel processing.

        Args:
            questions: List of (stem, answer) tuples

        Returns:
            List of ClassificationResults
        """
        results = []
        for stem, answer in questions:
            result = await self.classify(stem, answer)
            results.append(result)
        return results


class HeuristicBloomClassifier:
    """Rule-based Bloom classifier using keyword matching.

    Useful for testing or when LLM is not available.
    Less accurate than LLM-based classification but fast
    and deterministic.

    The classifier looks for action verbs and question patterns
    that are characteristic of each Bloom level.

    Example:
        >>> classifier = HeuristicBloomClassifier()
        >>> result = classifier.classify("Define photosynthesis.")
        >>> print(result.level_name)  # "Remember"
    """

    # Keywords that suggest each level (from Bloom's Taxonomy)
    LEVEL_KEYWORDS: dict[BloomLevel, list[str]] = {
        BloomLevel.REMEMBER: [
            "what is",
            "define",
            "list",
            "name",
            "identify",
            "which of the following",
            "who",
            "when",
            "where",
            "recall",
            "state",
            "match",
            "label",
            "select",
            "recognize",
        ],
        BloomLevel.UNDERSTAND: [
            "explain",
            "describe",
            "summarize",
            "interpret",
            "what does",
            "why does",
            "how does",
            "meaning",
            "paraphrase",
            "classify",
            "discuss",
            "distinguish",
            "predict",
            "translate",
        ],
        BloomLevel.APPLY: [
            "calculate",
            "solve",
            "use",
            "demonstrate",
            "how would you",
            "apply",
            "implement",
            "compute",
            "show",
            "illustrate",
            "execute",
            "employ",
            "practice",
        ],
        BloomLevel.ANALYZE: [
            "compare",
            "contrast",
            "analyze",
            "analyse",
            "differentiate",
            "what is the relationship",
            "examine",
            "distinguish",
            "investigate",
            "categorize",
            "deconstruct",
            "correlate",
            "outline",
        ],
        BloomLevel.EVALUATE: [
            "evaluate",
            "judge",
            "assess",
            "critique",
            "what is the best",
            "which is most",
            "justify",
            "defend",
            "prioritize",
            "recommend",
            "rate",
            "validate",
            "argue",
        ],
        BloomLevel.CREATE: [
            "design",
            "create",
            "develop",
            "propose",
            "what would happen if",
            "formulate",
            "construct",
            "generate",
            "compose",
            "invent",
            "synthesize",
            "plan",
            "produce",
        ],
    }

    def classify(self, question_stem: str) -> ClassificationResult:
        """Classify using keyword matching.

        Checks for keywords from highest to lowest level,
        returning the first match. This prioritizes higher-order
        thinking skills when multiple keywords are present.

        Args:
            question_stem: The question text

        Returns:
            ClassificationResult with the determined level
        """
        stem_lower = question_stem.lower()

        # Check from highest to lowest level
        for level in reversed(BloomLevel):
            keywords = self.LEVEL_KEYWORDS[level]
            if any(kw in stem_lower for kw in keywords):
                return ClassificationResult(
                    level=level,
                    method="heuristic",
                )

        # Default to Remember for basic fact questions
        return ClassificationResult(
            level=BloomLevel.REMEMBER,
            method="heuristic",
        )

    def classify_with_confidence(
        self,
        question_stem: str,
    ) -> ClassificationResult:
        """Classify with a confidence score based on keyword matches.

        The confidence is based on:
        - Number of keywords matched at the determined level
        - Absence of keywords from other levels

        Args:
            question_stem: The question text

        Returns:
            ClassificationResult with confidence score
        """
        stem_lower = question_stem.lower()

        # Count matches at each level
        matches: dict[BloomLevel, int] = {}
        for level in BloomLevel:
            keywords = self.LEVEL_KEYWORDS[level]
            matches[level] = sum(1 for kw in keywords if kw in stem_lower)

        # Find level with most matches
        if not any(matches.values()):
            return ClassificationResult(
                level=BloomLevel.REMEMBER,
                confidence=0.3,  # Low confidence for default
                method="heuristic",
            )

        best_level = max(matches, key=lambda l: matches[l])
        best_count = matches[best_level]
        total_matches = sum(matches.values())

        # Confidence based on how dominant the best level is
        if total_matches > 0:
            confidence = best_count / total_matches
            # Boost confidence if multiple keywords at same level
            if best_count > 1:
                confidence = min(1.0, confidence + 0.1)
        else:
            confidence = 0.3

        return ClassificationResult(
            level=best_level,
            confidence=round(confidence, 2),
            method="heuristic",
        )

    def classify_batch(
        self,
        questions: list[str],
    ) -> list[ClassificationResult]:
        """Classify multiple questions.

        Args:
            questions: List of question stems

        Returns:
            List of ClassificationResults
        """
        return [self.classify(q) for q in questions]

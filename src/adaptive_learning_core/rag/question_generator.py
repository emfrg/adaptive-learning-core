"""RAG-based question generation from learning materials.

Inspired by auto-rag-eval (Guinet et al., 2024) but adapted for
educational content with Bloom's Taxonomy integration.

Pipeline:
1. Take a card (chunk of learning material)
2. Generate question stem + correct answer using LLM
3. Generate plausible distractors
4. Classify according to Bloom's Taxonomy
5. Validate question quality
"""

import json
import re
import uuid
from dataclasses import dataclass
from typing import Optional, Protocol

from adaptive_learning_core.models.card import Card
from adaptive_learning_core.models.question import BloomLevel, Question
from adaptive_learning_core.bloom.taxonomy import BLOOM_LEVEL_INFO
from adaptive_learning_core.rag.prompts.question_gen import (
    QUESTION_GENERATION_PROMPT,
    BLOOM_TARGETED_PROMPT,
)


class LLMProvider(Protocol):
    """Protocol for LLM providers.

    Any LLM implementation must support this interface
    for use with the QuestionGenerator.
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
class QuestionGenerationConfig:
    """Configuration for question generation.

    Attributes:
        questions_per_chunk: How many questions to generate per card
        num_distractors: Number of wrong answers per question
        target_bloom_levels: Specific Bloom levels to target (None = all)
        include_explanation: Whether to include explanations
        max_retries: Maximum retries on parse failures
        min_question_length: Minimum question stem length
        max_question_length: Maximum question stem length
    """

    questions_per_chunk: int = 3
    num_distractors: int = 3
    target_bloom_levels: Optional[list[BloomLevel]] = None
    include_explanation: bool = True
    max_retries: int = 2
    min_question_length: int = 10
    max_question_length: int = 500


@dataclass
class GeneratedQuestion:
    """Raw generated question before validation.

    Attributes:
        stem: The question text
        correct_answer: The correct answer
        distractors: List of wrong answers
        explanation: Explanation of why the answer is correct
        target_bloom_level: Intended Bloom level (if targeted)
        source_card_id: ID of the source card
    """

    stem: str
    correct_answer: str
    distractors: list[str]
    explanation: str
    target_bloom_level: Optional[BloomLevel]
    source_card_id: str

    def to_question(
        self,
        question_id: Optional[str] = None,
        difficulty_level: Optional[int] = None,
    ) -> Question:
        """Convert to a Question model instance.

        Args:
            question_id: Optional ID (generated if not provided)
            difficulty_level: Optional difficulty override

        Returns:
            Question instance ready for the learning system
        """
        qid = question_id or f"q_{uuid.uuid4().hex[:12]}"

        # Use Bloom level as difficulty if available
        if difficulty_level is None and self.target_bloom_level:
            difficulty_level = self.target_bloom_level.value
        elif difficulty_level is None:
            difficulty_level = BloomLevel.UNDERSTAND.value  # Default

        return Question(
            id=qid,
            stem=self.stem,
            correct_answer=self.correct_answer,
            distractors=self.distractors,
            difficulty_level=difficulty_level,
            bloom_level=self.target_bloom_level or BloomLevel.UNDERSTAND,
            linked_card_ids=[self.source_card_id],
            explanation=self.explanation if self.explanation else None,
        )

    @property
    def is_valid(self) -> bool:
        """Check if this question passes basic validation."""
        return (
            len(self.stem) >= 10
            and len(self.correct_answer) > 0
            and len(self.distractors) >= 2
            and all(d != self.correct_answer for d in self.distractors)
        )


class QuestionGenerator:
    """Generates multiple-choice questions from learning material chunks.

    Pipeline:
    1. Take a card (chunk of learning material)
    2. Generate question stem + correct answer using LLM
    3. Generate plausible distractors
    4. Classify according to Bloom's Taxonomy
    5. Validate question quality

    Based on the exam generation approach from:
    "Automated Evaluation of Retrieval-Augmented Language Models
    with Task-Specific Exam Generation" (Guinet et al., ICML 2024)

    Example:
        >>> generator = QuestionGenerator(llm_provider)
        >>> questions = await generator.generate_from_card(card)
        >>> for q in questions:
        ...     print(f"Q: {q.stem}")
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: Optional[QuestionGenerationConfig] = None,
    ):
        """Initialize the question generator.

        Args:
            llm: LLM provider for generation
            config: Generation configuration
        """
        self.llm = llm
        self.config = config or QuestionGenerationConfig()

    def _build_generation_prompt(
        self,
        card: Card,
        target_bloom: Optional[BloomLevel] = None,
    ) -> str:
        """Build the prompt for question generation.

        Args:
            card: Source card with learning material
            target_bloom: Specific Bloom level to target

        Returns:
            Formatted prompt string
        """
        if target_bloom:
            bloom_info = BLOOM_LEVEL_INFO[target_bloom]
            return BLOOM_TARGETED_PROMPT.format(
                content=card.text,
                bloom_level=bloom_info.name,
                bloom_description=bloom_info.description,
                action_verbs=", ".join(bloom_info.action_verbs[:5]),
                num_distractors=self.config.num_distractors,
            )
        else:
            return QUESTION_GENERATION_PROMPT.format(
                content=card.text,
                num_questions=self.config.questions_per_chunk,
                num_distractors=self.config.num_distractors,
            )

    def _parse_generated_questions(
        self,
        response: str,
        card: Card,
        target_bloom: Optional[BloomLevel] = None,
    ) -> list[GeneratedQuestion]:
        """Parse LLM response into question objects.

        Expected JSON format:
        {
            "questions": [
                {
                    "stem": "...",
                    "correct_answer": "...",
                    "distractors": ["...", "...", "..."],
                    "explanation": "..."
                }
            ]
        }

        Args:
            response: Raw LLM response
            card: Source card
            target_bloom: Target Bloom level (if any)

        Returns:
            List of parsed GeneratedQuestion objects
        """
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return []

        try:
            data = json.loads(json_match.group())
            questions = data.get("questions", [])

            return [
                GeneratedQuestion(
                    stem=q["stem"],
                    correct_answer=q["correct_answer"],
                    distractors=q["distractors"],
                    explanation=q.get("explanation", ""),
                    target_bloom_level=target_bloom
                    or self._parse_bloom_from_response(q),
                    source_card_id=card.id,
                )
                for q in questions
                if self._validate_raw_question(q)
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    def _parse_bloom_from_response(
        self,
        question_data: dict,
    ) -> Optional[BloomLevel]:
        """Extract Bloom level from question data if present.

        Args:
            question_data: Parsed question dictionary

        Returns:
            BloomLevel if found, None otherwise
        """
        bloom_str = question_data.get("bloom_level", "")
        if not bloom_str:
            return None

        bloom_str = bloom_str.upper().strip()
        for level in BloomLevel:
            if level.name == bloom_str:
                return level
        return None

    def _validate_raw_question(self, q: dict) -> bool:
        """Basic validation of question structure.

        Args:
            q: Question dictionary from parsed JSON

        Returns:
            True if question passes validation
        """
        return (
            isinstance(q.get("stem"), str)
            and len(q["stem"]) >= self.config.min_question_length
            and len(q["stem"]) <= self.config.max_question_length
            and isinstance(q.get("correct_answer"), str)
            and len(q["correct_answer"]) > 0
            and isinstance(q.get("distractors"), list)
            and len(q["distractors"]) >= 2
        )

    async def generate_from_card(
        self,
        card: Card,
        target_bloom_levels: Optional[list[BloomLevel]] = None,
    ) -> list[GeneratedQuestion]:
        """Generate questions from a single card.

        Args:
            card: The source card (learning material chunk)
            target_bloom_levels: Specific levels to target (overrides config)

        Returns:
            List of generated questions
        """
        target_levels = target_bloom_levels or self.config.target_bloom_levels
        all_questions: list[GeneratedQuestion] = []

        # If targeting specific Bloom levels, generate for each
        if target_levels:
            for level in target_levels:
                for attempt in range(self.config.max_retries + 1):
                    prompt = self._build_generation_prompt(card, level)
                    response = await self.llm.complete(prompt)
                    questions = self._parse_generated_questions(response, card, level)
                    if questions:
                        all_questions.extend(questions)
                        break
        else:
            # Generate mixed difficulty questions
            for attempt in range(self.config.max_retries + 1):
                prompt = self._build_generation_prompt(card)
                response = await self.llm.complete(prompt)
                questions = self._parse_generated_questions(response, card)
                if questions:
                    all_questions = questions
                    break

        return all_questions

    async def generate_from_cards(
        self,
        cards: list[Card],
        target_bloom_levels: Optional[list[BloomLevel]] = None,
    ) -> list[GeneratedQuestion]:
        """Generate questions from multiple cards.

        Args:
            cards: List of source cards
            target_bloom_levels: Specific levels to target

        Returns:
            List of all generated questions
        """
        all_questions: list[GeneratedQuestion] = []
        for card in cards:
            questions = await self.generate_from_card(card, target_bloom_levels)
            all_questions.extend(questions)
        return all_questions

    async def generate_balanced(
        self,
        cards: list[Card],
        questions_per_level: int = 1,
        levels: Optional[list[BloomLevel]] = None,
    ) -> list[GeneratedQuestion]:
        """Generate questions balanced across Bloom levels.

        Creates an even distribution of questions across the
        specified Bloom levels from the available cards.

        Args:
            cards: Source cards
            questions_per_level: How many questions at each level
            levels: Which levels to include (default: all)

        Returns:
            List of generated questions, balanced by level
        """
        levels = levels or list(BloomLevel)
        all_questions: list[GeneratedQuestion] = []

        # Distribute cards across levels
        for i, level in enumerate(levels):
            # Use different cards for different levels (round-robin)
            card_idx = i % len(cards)
            card = cards[card_idx]

            for _ in range(questions_per_level):
                questions = await self.generate_from_card(card, [level])
                if questions:
                    all_questions.append(questions[0])

        return all_questions

    def convert_to_questions(
        self,
        generated: list[GeneratedQuestion],
    ) -> list[Question]:
        """Convert generated questions to Question model instances.

        Filters invalid questions and assigns IDs.

        Args:
            generated: List of GeneratedQuestion objects

        Returns:
            List of valid Question model instances
        """
        questions = []
        for i, gq in enumerate(generated):
            if gq.is_valid:
                q = gq.to_question(question_id=f"q_{uuid.uuid4().hex[:12]}")
                questions.append(q)
        return questions

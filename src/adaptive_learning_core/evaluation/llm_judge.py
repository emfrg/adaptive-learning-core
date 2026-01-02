"""LLM-as-a-Judge evaluation for question quality.

Uses LLMs to evaluate generated questions across multiple dimensions:
1. Clarity: Is the question clear and unambiguous?
2. Correctness: Is the correct answer actually correct?
3. Difficulty: Is the difficulty level appropriate?
4. Distractors: Are the wrong answers plausible?
5. Relevance: Is the question relevant to the source material?

This approach is more nuanced than traditional metrics like ROUGE,
capturing semantic quality that string matching cannot.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

from adaptive_learning_core.models.question import Question


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(self, prompt: str) -> str:
        """Generate completion for a prompt."""
        ...


class QualityDimension(str, Enum):
    """Dimensions of question quality evaluated by the judge."""

    CLARITY = "clarity"
    CORRECTNESS = "correctness"
    DIFFICULTY = "difficulty"
    DISTRACTORS = "distractors"
    RELEVANCE = "relevance"
    EDUCATIONAL_VALUE = "educational_value"


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for LLM judge evaluation.

    Attributes:
        dimensions: Which dimensions to evaluate
        scale: Rating scale (e.g., 1-5 or 1-10)
        include_reasoning: Whether to request reasoning
        strict_mode: Whether to require all criteria met
        temperature: LLM temperature for evaluation
    """

    dimensions: tuple[QualityDimension, ...] = (
        QualityDimension.CLARITY,
        QualityDimension.CORRECTNESS,
        QualityDimension.DISTRACTORS,
        QualityDimension.RELEVANCE,
    )
    scale: int = 5
    include_reasoning: bool = True
    strict_mode: bool = False
    temperature: float = 0.0


@dataclass
class JudgmentResult:
    """Result of LLM judge evaluation.

    Attributes:
        question_id: ID of the evaluated question
        scores: Score per dimension
        overall_score: Aggregated overall score
        reasoning: Reasoning per dimension
        passed: Whether question passes quality threshold
        raw_response: Raw LLM response
    """

    question_id: str
    scores: dict[QualityDimension, float] = field(default_factory=dict)
    overall_score: float = 0.0
    reasoning: dict[QualityDimension, str] = field(default_factory=dict)
    passed: bool = True
    raw_response: Optional[str] = None

    @property
    def mean_score(self) -> float:
        """Mean score across all dimensions."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    @property
    def min_score(self) -> float:
        """Minimum score across dimensions."""
        if not self.scores:
            return 0.0
        return min(self.scores.values())

    @property
    def weak_dimensions(self) -> list[QualityDimension]:
        """Dimensions with below-average scores."""
        if not self.scores:
            return []
        mean = self.mean_score
        return [dim for dim, score in self.scores.items() if score < mean]


# Evaluation prompts
JUDGE_PROMPT = """You are an expert educational content evaluator. Evaluate the following multiple-choice question on a scale of 1 to {scale}.

QUESTION:
{question_stem}

CORRECT ANSWER:
{correct_answer}

DISTRACTORS (wrong answers):
{distractors}

SOURCE MATERIAL:
{context}

Evaluate on these dimensions:
{dimensions}

For each dimension, provide:
1. A score from 1 to {scale}
2. Brief reasoning for the score

SCORING GUIDELINES:
- 1: Very poor, unacceptable
- 2: Poor, needs significant improvement
- 3: Acceptable, but could be better
- 4: Good, minor improvements possible
- 5: Excellent, high quality

OUTPUT FORMAT (JSON):
{{
    "scores": {{
        "clarity": <score>,
        "correctness": <score>,
        "distractors": <score>,
        "relevance": <score>
    }},
    "reasoning": {{
        "clarity": "Brief explanation",
        "correctness": "Brief explanation",
        "distractors": "Brief explanation",
        "relevance": "Brief explanation"
    }},
    "overall_assessment": "Brief overall assessment",
    "recommendations": ["Improvement suggestion 1", "..."]
}}

Evaluate now:"""

DIMENSION_DESCRIPTIONS = {
    QualityDimension.CLARITY: "Is the question clear, unambiguous, and grammatically correct?",
    QualityDimension.CORRECTNESS: "Is the marked correct answer actually correct and complete?",
    QualityDimension.DIFFICULTY: "Is the difficulty level appropriate for the intended audience?",
    QualityDimension.DISTRACTORS: "Are the distractors plausible but clearly incorrect?",
    QualityDimension.RELEVANCE: "Is the question relevant to and grounded in the source material?",
    QualityDimension.EDUCATIONAL_VALUE: "Does the question test meaningful understanding?",
}


class LLMJudge:
    """Uses LLM to evaluate question quality.

    Provides structured evaluation across multiple quality dimensions
    with reasoning and recommendations.

    Example:
        >>> judge = LLMJudge(llm_provider)
        >>> result = await judge.evaluate(question, source_text)
        >>> if not result.passed:
        ...     print(f"Quality issues: {result.weak_dimensions}")
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: Optional[JudgeConfig] = None,
    ):
        """Initialize the LLM judge.

        Args:
            llm: LLM provider for evaluation
            config: Judge configuration
        """
        self.llm = llm
        self.config = config or JudgeConfig()

    def _build_prompt(
        self,
        question: Question,
        context: str,
    ) -> str:
        """Build the evaluation prompt.

        Args:
            question: Question to evaluate
            context: Source material

        Returns:
            Formatted prompt
        """
        # Format distractors
        distractors_str = "\n".join(
            f"- {d}" for d in question.distractors
        )

        # Format dimensions to evaluate
        dimensions_str = "\n".join(
            f"- {dim.value.upper()}: {DIMENSION_DESCRIPTIONS[dim]}"
            for dim in self.config.dimensions
        )

        return JUDGE_PROMPT.format(
            scale=self.config.scale,
            question_stem=question.stem,
            correct_answer=question.correct_answer,
            distractors=distractors_str,
            context=context,
            dimensions=dimensions_str,
        )

    def _parse_response(
        self,
        response: str,
        question_id: str,
    ) -> JudgmentResult:
        """Parse LLM response into JudgmentResult.

        Args:
            response: Raw LLM response
            question_id: Question identifier

        Returns:
            Parsed JudgmentResult
        """
        result = JudgmentResult(
            question_id=question_id,
            raw_response=response,
        )

        # Try to extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            # Fallback: couldn't parse, give neutral scores
            for dim in self.config.dimensions:
                result.scores[dim] = self.config.scale / 2
            result.passed = False
            return result

        try:
            data = json.loads(json_match.group())

            # Extract scores
            scores_data = data.get("scores", {})
            for dim in self.config.dimensions:
                dim_key = dim.value.lower()
                if dim_key in scores_data:
                    score = float(scores_data[dim_key])
                    result.scores[dim] = min(max(1, score), self.config.scale)
                else:
                    result.scores[dim] = self.config.scale / 2

            # Extract reasoning
            reasoning_data = data.get("reasoning", {})
            for dim in self.config.dimensions:
                dim_key = dim.value.lower()
                if dim_key in reasoning_data:
                    result.reasoning[dim] = reasoning_data[dim_key]

            # Calculate overall score
            result.overall_score = result.mean_score

            # Determine pass/fail
            if self.config.strict_mode:
                # All dimensions must be above threshold
                threshold = self.config.scale * 0.6
                result.passed = all(
                    score >= threshold for score in result.scores.values()
                )
            else:
                # Overall average must be above threshold
                result.passed = result.overall_score >= self.config.scale * 0.6

        except (json.JSONDecodeError, KeyError, ValueError):
            for dim in self.config.dimensions:
                result.scores[dim] = self.config.scale / 2
            result.passed = False

        return result

    async def evaluate(
        self,
        question: Question,
        context: str,
    ) -> JudgmentResult:
        """Evaluate a single question.

        Args:
            question: Question to evaluate
            context: Source material for the question

        Returns:
            JudgmentResult with scores and reasoning
        """
        prompt = self._build_prompt(question, context)
        response = await self.llm.complete(prompt)
        return self._parse_response(response, question.id)

    async def evaluate_batch(
        self,
        questions: list[tuple[Question, str]],
    ) -> list[JudgmentResult]:
        """Evaluate multiple questions.

        Args:
            questions: List of (question, context) tuples

        Returns:
            List of JudgmentResults
        """
        results = []
        for question, context in questions:
            result = await self.evaluate(question, context)
            results.append(result)
        return results

    async def evaluate_distractor(
        self,
        question_stem: str,
        correct_answer: str,
        distractor: str,
        context: str,
    ) -> dict[str, Any]:
        """Evaluate a single distractor.

        Args:
            question_stem: The question text
            correct_answer: The correct answer
            distractor: The distractor to evaluate
            context: Source material

        Returns:
            Dictionary with distractor evaluation
        """
        prompt = f"""Evaluate this distractor for a multiple-choice question.

QUESTION: {question_stem}
CORRECT ANSWER: {correct_answer}
DISTRACTOR TO EVALUATE: {distractor}

CONTEXT: {context}

Evaluate:
1. Is it clearly incorrect? (should be wrong)
2. Is it plausible? (could a learner reasonably choose it)
3. Is it distinct from the correct answer?

OUTPUT (JSON):
{{
    "is_incorrect": true/false,
    "is_plausible": true/false,
    "is_distinct": true/false,
    "quality_score": 1-5,
    "reasoning": "Brief explanation"
}}
"""
        response = await self.llm.complete(prompt)

        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "is_incorrect": True,
            "is_plausible": True,
            "is_distinct": True,
            "quality_score": 3,
            "reasoning": "Could not parse evaluation",
        }

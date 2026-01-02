"""RAGAS-style metrics for RAG quality evaluation.

RAGAS (Retrieval Augmented Generation Assessment) provides metrics for
evaluating the quality of RAG pipelines:

1. Faithfulness: Is the generated content grounded in the source?
2. Answer Relevancy: Does the answer address the question?
3. Context Relevancy: Is the retrieved context relevant?
4. Context Precision: How precise is the context selection?

These metrics are particularly useful for evaluating generated questions
to ensure they are well-grounded in the source material.

Reference: https://github.com/explodinggradients/ragas
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

import numpy as np


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(self, prompt: str) -> str:
        """Generate completion for a prompt."""
        ...


class RAGASMetric(str, Enum):
    """RAGAS evaluation metrics."""

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_RELEVANCY = "context_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


@dataclass(frozen=True)
class RAGASConfig:
    """Configuration for RAGAS evaluation.

    Attributes:
        metrics: Which metrics to compute
        use_llm_for_relevancy: Use LLM for relevancy scoring
        chunk_size: Size for chunking long contexts
    """

    metrics: tuple[RAGASMetric, ...] = (
        RAGASMetric.FAITHFULNESS,
        RAGASMetric.ANSWER_RELEVANCY,
        RAGASMetric.CONTEXT_RELEVANCY,
    )
    use_llm_for_relevancy: bool = True
    chunk_size: int = 500


@dataclass
class RAGASResult:
    """Result of RAGAS evaluation.

    Attributes:
        question_id: ID of evaluated question
        scores: Score per metric (0-1)
        details: Detailed breakdown per metric
        overall_score: Weighted average of metrics
    """

    question_id: str
    scores: dict[RAGASMetric, float] = field(default_factory=dict)
    details: dict[RAGASMetric, dict] = field(default_factory=dict)
    overall_score: float = 0.0

    @property
    def is_faithful(self) -> bool:
        """Whether the content is faithful to source."""
        return self.scores.get(RAGASMetric.FAITHFULNESS, 0) >= 0.7

    @property
    def is_relevant(self) -> bool:
        """Whether the content is relevant."""
        answer_rel = self.scores.get(RAGASMetric.ANSWER_RELEVANCY, 0)
        context_rel = self.scores.get(RAGASMetric.CONTEXT_RELEVANCY, 0)
        return (answer_rel + context_rel) / 2 >= 0.6


# Evaluation prompts
FAITHFULNESS_PROMPT = """Given the following context and statement, determine if the statement is supported by the context.

CONTEXT:
{context}

STATEMENT:
{statement}

Is this statement supported by the context? Answer with:
- SUPPORTED: The statement is directly supported by the context
- PARTIALLY: The statement is partially supported
- NOT_SUPPORTED: The statement is not supported or contradicts the context

Answer (one word):"""

RELEVANCY_PROMPT = """Rate the relevancy of the following answer to the question on a scale of 0 to 1.

QUESTION:
{question}

ANSWER:
{answer}

CONTEXT:
{context}

Consider:
1. Does the answer address the question?
2. Is the answer complete?
3. Is the answer accurate based on the context?

Provide a score from 0 (completely irrelevant) to 1 (perfectly relevant).

Score:"""

CONTEXT_RELEVANCY_PROMPT = """Rate how relevant this context is for answering the question.

QUESTION:
{question}

CONTEXT:
{context}

Rate from 0 (completely irrelevant) to 1 (highly relevant).

Score:"""


class RAGASEvaluator:
    """Evaluates RAG quality using RAGAS-style metrics.

    Provides evaluation of:
    - Faithfulness: Is the generated content grounded in source?
    - Answer Relevancy: Does the answer address the question?
    - Context Relevancy: Is the context appropriate?

    Example:
        >>> evaluator = RAGASEvaluator(llm_provider)
        >>> result = await evaluator.evaluate(
        ...     question="What is X?",
        ...     answer="X is Y",
        ...     context="Source material about X..."
        ... )
        >>> print(f"Faithfulness: {result.scores[RAGASMetric.FAITHFULNESS]:.2f}")
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        config: Optional[RAGASConfig] = None,
    ):
        """Initialize the RAGAS evaluator.

        Args:
            llm: LLM provider for evaluation
            config: Evaluation configuration
        """
        self.llm = llm
        self.config = config or RAGASConfig()

    async def _evaluate_faithfulness(
        self,
        statement: str,
        context: str,
    ) -> tuple[float, dict]:
        """Evaluate faithfulness of a statement to context.

        Args:
            statement: Statement to verify
            context: Source context

        Returns:
            Tuple of (score, details)
        """
        if not self.llm:
            # Fallback to simple overlap
            return self._simple_faithfulness(statement, context)

        prompt = FAITHFULNESS_PROMPT.format(
            context=context,
            statement=statement,
        )

        response = await self.llm.complete(prompt)
        response_lower = response.lower().strip()

        if "supported" in response_lower and "not" not in response_lower:
            score = 1.0
            support_level = "supported"
        elif "partial" in response_lower:
            score = 0.5
            support_level = "partial"
        else:
            score = 0.0
            support_level = "not_supported"

        return score, {"support_level": support_level, "raw_response": response}

    def _simple_faithfulness(
        self,
        statement: str,
        context: str,
    ) -> tuple[float, dict]:
        """Simple word overlap faithfulness check.

        Args:
            statement: Statement to verify
            context: Source context

        Returns:
            Tuple of (score, details)
        """
        statement_words = set(statement.lower().split())
        context_words = set(context.lower().split())

        if not statement_words:
            return 0.0, {"method": "word_overlap", "overlap": 0}

        overlap = len(statement_words & context_words) / len(statement_words)
        return overlap, {"method": "word_overlap", "overlap": overlap}

    async def _evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> tuple[float, dict]:
        """Evaluate how relevant the answer is to the question.

        Args:
            question: The question asked
            answer: The provided answer
            context: Source context

        Returns:
            Tuple of (score, details)
        """
        if not self.llm:
            return self._simple_relevancy(question, answer)

        prompt = RELEVANCY_PROMPT.format(
            question=question,
            answer=answer,
            context=context,
        )

        response = await self.llm.complete(prompt)

        # Try to extract score
        try:
            # Look for a number
            import re

            numbers = re.findall(r"[0-9.]+", response)
            if numbers:
                score = float(numbers[0])
                score = min(max(0, score), 1)  # Clamp to [0, 1]
            else:
                score = 0.5
        except ValueError:
            score = 0.5

        return score, {"raw_response": response}

    def _simple_relevancy(
        self,
        question: str,
        answer: str,
    ) -> tuple[float, dict]:
        """Simple relevancy check using word overlap.

        Args:
            question: The question
            answer: The answer

        Returns:
            Tuple of (score, details)
        """
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())

        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why"}
        q_words -= stop_words
        a_words -= stop_words

        if not q_words:
            return 0.5, {"method": "word_overlap"}

        overlap = len(q_words & a_words) / len(q_words)
        return min(1.0, overlap * 2), {"method": "word_overlap", "overlap": overlap}

    async def _evaluate_context_relevancy(
        self,
        question: str,
        context: str,
    ) -> tuple[float, dict]:
        """Evaluate how relevant the context is to the question.

        Args:
            question: The question
            context: Retrieved context

        Returns:
            Tuple of (score, details)
        """
        if not self.llm:
            return self._simple_relevancy(question, context)

        prompt = CONTEXT_RELEVANCY_PROMPT.format(
            question=question,
            context=context,
        )

        response = await self.llm.complete(prompt)

        try:
            import re

            numbers = re.findall(r"[0-9.]+", response)
            if numbers:
                score = float(numbers[0])
                score = min(max(0, score), 1)
            else:
                score = 0.5
        except ValueError:
            score = 0.5

        return score, {"raw_response": response}

    async def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        question_id: Optional[str] = None,
    ) -> RAGASResult:
        """Evaluate a question-answer pair with context.

        Args:
            question: The question text
            answer: The answer text
            context: Source context
            question_id: Optional question identifier

        Returns:
            RAGASResult with scores for each metric
        """
        result = RAGASResult(question_id=question_id or "unknown")

        for metric in self.config.metrics:
            if metric == RAGASMetric.FAITHFULNESS:
                # Evaluate faithfulness of the answer
                score, details = await self._evaluate_faithfulness(answer, context)
                result.scores[metric] = score
                result.details[metric] = details

            elif metric == RAGASMetric.ANSWER_RELEVANCY:
                score, details = await self._evaluate_answer_relevancy(
                    question, answer, context
                )
                result.scores[metric] = score
                result.details[metric] = details

            elif metric == RAGASMetric.CONTEXT_RELEVANCY:
                score, details = await self._evaluate_context_relevancy(
                    question, context
                )
                result.scores[metric] = score
                result.details[metric] = details

        # Calculate overall score
        if result.scores:
            result.overall_score = np.mean(list(result.scores.values()))

        return result

    async def evaluate_question(
        self,
        question_stem: str,
        correct_answer: str,
        context: str,
        question_id: Optional[str] = None,
    ) -> RAGASResult:
        """Evaluate a generated question against its source.

        Specifically designed for evaluating generated MCQ questions.

        Args:
            question_stem: The question text
            correct_answer: The correct answer
            context: Source material the question was generated from
            question_id: Optional identifier

        Returns:
            RAGASResult with quality metrics
        """
        # For questions, we evaluate:
        # 1. Faithfulness: Is the question grounded in the context?
        # 2. Answer Relevancy: Does the correct answer match the question?
        # 3. Context Relevancy: Is the context relevant to the question?

        result = RAGASResult(question_id=question_id or "unknown")

        # Faithfulness: Is the question-answer pair grounded in context?
        statement = f"Question: {question_stem} Answer: {correct_answer}"
        f_score, f_details = await self._evaluate_faithfulness(statement, context)
        result.scores[RAGASMetric.FAITHFULNESS] = f_score
        result.details[RAGASMetric.FAITHFULNESS] = f_details

        # Answer Relevancy: Does the answer properly address the question?
        ar_score, ar_details = await self._evaluate_answer_relevancy(
            question_stem, correct_answer, context
        )
        result.scores[RAGASMetric.ANSWER_RELEVANCY] = ar_score
        result.details[RAGASMetric.ANSWER_RELEVANCY] = ar_details

        # Context Relevancy: Is the source context appropriate?
        cr_score, cr_details = await self._evaluate_context_relevancy(
            question_stem, context
        )
        result.scores[RAGASMetric.CONTEXT_RELEVANCY] = cr_score
        result.details[RAGASMetric.CONTEXT_RELEVANCY] = cr_details

        # Overall score
        result.overall_score = np.mean(list(result.scores.values()))

        return result

    async def evaluate_batch(
        self,
        items: list[tuple[str, str, str, Optional[str]]],
    ) -> list[RAGASResult]:
        """Evaluate multiple items.

        Args:
            items: List of (question, answer, context, id) tuples

        Returns:
            List of RAGASResults
        """
        results = []
        for question, answer, context, qid in items:
            result = await self.evaluate(question, answer, context, qid)
            results.append(result)
        return results

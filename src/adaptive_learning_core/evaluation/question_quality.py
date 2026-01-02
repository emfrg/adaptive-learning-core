"""Combined question quality pipeline.

Aggregates multiple evaluation methods into a single quality score:
1. IRT Discrimination: How well the question differentiates learners
2. LLM-as-Judge: Semantic quality evaluation
3. RAGAS: Faithfulness and relevance to source material

The combined pipeline provides:
- Overall quality grade (A-F)
- Automatic filtering of low-quality questions
- Actionable improvement recommendations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from adaptive_learning_core.models.question import Question
from adaptive_learning_core.irt.discrimination import (
    DiscriminationAnalyzer,
    DiscriminationResult,
    QualityCategory,
)
from adaptive_learning_core.evaluation.llm_judge import (
    LLMJudge,
    JudgmentResult,
    QualityDimension,
)
from adaptive_learning_core.evaluation.ragas_eval import (
    RAGASEvaluator,
    RAGASResult,
    RAGASMetric,
)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(self, prompt: str) -> str:
        """Generate completion for a prompt."""
        ...


class QualityGrade(str, Enum):
    """Overall quality grade for a question."""

    A = "A"  # Excellent - use without changes
    B = "B"  # Good - minor improvements possible
    C = "C"  # Acceptable - use with caution
    D = "D"  # Poor - needs revision
    F = "F"  # Fail - should not be used


@dataclass(frozen=True)
class QualityConfig:
    """Configuration for quality pipeline.

    Attributes:
        use_irt: Whether to use IRT discrimination
        use_llm_judge: Whether to use LLM-as-judge
        use_ragas: Whether to use RAGAS metrics
        min_responses_for_irt: Minimum responses needed for IRT
        weight_irt: Weight for IRT score
        weight_llm: Weight for LLM judge score
        weight_ragas: Weight for RAGAS score
        grade_thresholds: Score thresholds for grades
    """

    use_irt: bool = True
    use_llm_judge: bool = True
    use_ragas: bool = True
    min_responses_for_irt: int = 10
    weight_irt: float = 0.3
    weight_llm: float = 0.4
    weight_ragas: float = 0.3
    grade_thresholds: tuple[float, ...] = (0.9, 0.8, 0.7, 0.5)  # A, B, C, D


@dataclass
class QualityResult:
    """Combined quality evaluation result.

    Attributes:
        question_id: Question identifier
        overall_score: Combined weighted score (0-1)
        grade: Letter grade
        irt_result: IRT discrimination result if computed
        llm_result: LLM judge result if computed
        ragas_result: RAGAS result if computed
        passed: Whether question passes quality threshold
        recommendations: Improvement recommendations
    """

    question_id: str
    overall_score: float = 0.0
    grade: QualityGrade = QualityGrade.C
    irt_result: Optional[DiscriminationResult] = None
    llm_result: Optional[JudgmentResult] = None
    ragas_result: Optional[RAGASResult] = None
    passed: bool = True
    recommendations: list[str] = field(default_factory=list)

    @property
    def component_scores(self) -> dict[str, float]:
        """Individual component scores."""
        scores = {}
        if self.irt_result:
            # Normalize discrimination to 0-1 (typical range 0-2.5)
            scores["irt"] = min(1.0, self.irt_result.discrimination / 2.5)
        if self.llm_result:
            # LLM score is already 1-5, normalize to 0-1
            scores["llm"] = self.llm_result.overall_score / 5.0
        if self.ragas_result:
            scores["ragas"] = self.ragas_result.overall_score
        return scores

    @property
    def needs_improvement(self) -> bool:
        """Whether the question needs improvement."""
        return self.grade in (QualityGrade.D, QualityGrade.F)


class QuestionQualityPipeline:
    """Combined quality evaluation pipeline.

    Aggregates IRT discrimination, LLM-as-judge, and RAGAS metrics
    into a single quality assessment.

    Example:
        >>> pipeline = QuestionQualityPipeline(llm_provider)
        >>> result = await pipeline.evaluate(
        ...     question=question,
        ...     context=source_text,
        ...     responses=response_data,  # Optional
        ...     abilities=ability_estimates,  # Optional
        ... )
        >>> print(f"Grade: {result.grade.value}, Score: {result.overall_score:.2f}")
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        config: Optional[QualityConfig] = None,
    ):
        """Initialize the quality pipeline.

        Args:
            llm: LLM provider for LLM-based evaluations
            config: Pipeline configuration
        """
        self.config = config or QualityConfig()
        self.llm = llm

        # Initialize components
        self.discrimination_analyzer = DiscriminationAnalyzer()

        if llm:
            self.llm_judge = LLMJudge(llm)
            self.ragas_evaluator = RAGASEvaluator(llm)
        else:
            self.llm_judge = None
            self.ragas_evaluator = RAGASEvaluator()  # Will use simple methods

    def _calculate_grade(self, score: float) -> QualityGrade:
        """Calculate letter grade from numeric score.

        Args:
            score: Score between 0 and 1

        Returns:
            QualityGrade
        """
        thresholds = self.config.grade_thresholds
        if score >= thresholds[0]:
            return QualityGrade.A
        elif score >= thresholds[1]:
            return QualityGrade.B
        elif score >= thresholds[2]:
            return QualityGrade.C
        elif score >= thresholds[3]:
            return QualityGrade.D
        else:
            return QualityGrade.F

    def _aggregate_recommendations(
        self,
        irt_result: Optional[DiscriminationResult],
        llm_result: Optional[JudgmentResult],
        ragas_result: Optional[RAGASResult],
    ) -> list[str]:
        """Aggregate recommendations from all sources.

        Args:
            irt_result: IRT result
            llm_result: LLM judge result
            ragas_result: RAGAS result

        Returns:
            Combined list of recommendations
        """
        recommendations = []

        # IRT recommendations
        if irt_result and irt_result.recommendations:
            recommendations.extend(irt_result.recommendations[:2])

        # LLM recommendations (from weak dimensions)
        if llm_result and llm_result.weak_dimensions:
            for dim in llm_result.weak_dimensions[:2]:
                if dim == QualityDimension.CLARITY:
                    recommendations.append("Improve question clarity and wording")
                elif dim == QualityDimension.DISTRACTORS:
                    recommendations.append("Improve distractor plausibility")
                elif dim == QualityDimension.RELEVANCE:
                    recommendations.append("Ensure question is grounded in source")

        # RAGAS recommendations
        if ragas_result:
            if ragas_result.scores.get(RAGASMetric.FAITHFULNESS, 1) < 0.7:
                recommendations.append(
                    "Question may not be well-grounded in source material"
                )
            if ragas_result.scores.get(RAGASMetric.ANSWER_RELEVANCY, 1) < 0.7:
                recommendations.append(
                    "Correct answer may not properly address the question"
                )

        return recommendations

    async def evaluate(
        self,
        question: Question,
        context: str,
        responses: Optional[NDArray[np.int_]] = None,
        abilities: Optional[NDArray[np.float64]] = None,
    ) -> QualityResult:
        """Evaluate question quality using all available methods.

        Args:
            question: Question to evaluate
            context: Source material
            responses: Optional response data for IRT
            abilities: Optional ability estimates for IRT

        Returns:
            QualityResult with combined assessment
        """
        result = QualityResult(question_id=question.id)
        scores = []
        weights = []

        # 1. IRT Discrimination (if data available)
        if (
            self.config.use_irt
            and responses is not None
            and abilities is not None
            and len(responses) >= self.config.min_responses_for_irt
        ):
            irt_result = self.discrimination_analyzer.analyze_item(
                item_id=question.id,
                responses=responses,
                abilities=abilities,
            )
            result.irt_result = irt_result

            # Normalize discrimination to 0-1
            irt_score = min(1.0, irt_result.discrimination / 2.5)
            scores.append(irt_score)
            weights.append(self.config.weight_irt)

        # 2. LLM-as-Judge
        if self.config.use_llm_judge and self.llm_judge:
            llm_result = await self.llm_judge.evaluate(question, context)
            result.llm_result = llm_result

            llm_score = llm_result.overall_score / 5.0
            scores.append(llm_score)
            weights.append(self.config.weight_llm)

        # 3. RAGAS
        if self.config.use_ragas and self.ragas_evaluator:
            ragas_result = await self.ragas_evaluator.evaluate_question(
                question_stem=question.stem,
                correct_answer=question.correct_answer,
                context=context,
                question_id=question.id,
            )
            result.ragas_result = ragas_result

            scores.append(ragas_result.overall_score)
            weights.append(self.config.weight_ragas)

        # Calculate weighted average
        if scores:
            total_weight = sum(weights)
            result.overall_score = sum(
                s * w for s, w in zip(scores, weights)
            ) / total_weight
        else:
            result.overall_score = 0.5  # Default neutral score

        # Determine grade
        result.grade = self._calculate_grade(result.overall_score)

        # Pass/fail determination
        result.passed = result.grade in (QualityGrade.A, QualityGrade.B, QualityGrade.C)

        # Aggregate recommendations
        result.recommendations = self._aggregate_recommendations(
            result.irt_result,
            result.llm_result,
            result.ragas_result,
        )

        return result

    async def evaluate_batch(
        self,
        questions: list[tuple[Question, str]],
        response_matrix: Optional[NDArray[np.int_]] = None,
        abilities: Optional[NDArray[np.float64]] = None,
    ) -> list[QualityResult]:
        """Evaluate multiple questions.

        Args:
            questions: List of (question, context) tuples
            response_matrix: Optional N x M response matrix
            abilities: Optional ability estimates

        Returns:
            List of QualityResults
        """
        results = []
        for i, (question, context) in enumerate(questions):
            # Get responses for this question if available
            item_responses = None
            if response_matrix is not None and i < response_matrix.shape[1]:
                item_responses = response_matrix[:, i]

            result = await self.evaluate(
                question=question,
                context=context,
                responses=item_responses,
                abilities=abilities,
            )
            results.append(result)

        return results

    def filter_quality_questions(
        self,
        results: list[QualityResult],
        min_grade: QualityGrade = QualityGrade.C,
    ) -> list[str]:
        """Filter questions that meet quality threshold.

        Args:
            results: List of quality results
            min_grade: Minimum acceptable grade

        Returns:
            List of question IDs that pass
        """
        grade_order = [QualityGrade.A, QualityGrade.B, QualityGrade.C, QualityGrade.D, QualityGrade.F]
        min_index = grade_order.index(min_grade)

        return [
            r.question_id
            for r in results
            if grade_order.index(r.grade) <= min_index
        ]

    def get_summary_statistics(
        self,
        results: list[QualityResult],
    ) -> dict:
        """Get summary statistics for evaluated questions.

        Args:
            results: List of quality results

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}

        scores = [r.overall_score for r in results]
        grades = [r.grade for r in results]

        # Count by grade
        grade_counts = {grade: 0 for grade in QualityGrade}
        for grade in grades:
            grade_counts[grade] += 1

        return {
            "total_questions": len(results),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "grade_distribution": {g.value: c for g, c in grade_counts.items()},
            "pass_rate": sum(1 for r in results if r.passed) / len(results),
            "needs_improvement_count": sum(
                1 for r in results if r.needs_improvement
            ),
        }

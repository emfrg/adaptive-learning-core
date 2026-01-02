"""Discrimination analysis for question quality evaluation.

Provides tools to analyze how well questions differentiate between
learners of different ability levels.

Discrimination Index Interpretation:
- a > 1.5: Excellent discrimination (highly recommended)
- 1.0 < a < 1.5: Good discrimination (acceptable)
- 0.5 < a < 1.0: Moderate discrimination (review recommended)
- 0.0 < a < 0.5: Poor discrimination (needs revision)
- a < 0.0: Negative discrimination (problematic - should be removed)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from adaptive_learning_core.irt.models import (
    IRTModel,
    ItemParameters,
    Model2PL,
)


class QualityCategory(str, Enum):
    """Quality categories based on discrimination."""

    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    PROBLEMATIC = "problematic"


@dataclass
class DiscriminationResult:
    """Result of discrimination analysis for an item.

    Attributes:
        item_id: Item identifier
        discrimination: IRT discrimination parameter (a)
        difficulty: IRT difficulty parameter (b)
        point_biserial: Point-biserial correlation
        proportion_correct: Proportion of correct responses
        category: Quality category
        recommendations: Improvement recommendations
    """

    item_id: str
    discrimination: float
    difficulty: float
    point_biserial: float
    proportion_correct: float
    category: QualityCategory
    recommendations: list[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        """Whether the item is acceptable for use."""
        return self.category in (QualityCategory.EXCELLENT, QualityCategory.GOOD)

    @property
    def needs_review(self) -> bool:
        """Whether the item needs review."""
        return self.category in (QualityCategory.MODERATE, QualityCategory.POOR)

    @property
    def should_remove(self) -> bool:
        """Whether the item should be removed."""
        return self.category == QualityCategory.PROBLEMATIC


class DiscriminationAnalyzer:
    """Analyzes question discrimination for quality evaluation.

    Provides both classical (point-biserial) and IRT-based
    discrimination indices.

    Example:
        >>> analyzer = DiscriminationAnalyzer()
        >>> result = analyzer.analyze_item(
        ...     item_id="q1",
        ...     responses=responses,
        ...     abilities=abilities
        ... )
        >>> if not result.is_acceptable:
        ...     print(f"Review needed: {result.recommendations}")
    """

    # Thresholds for discrimination categories
    EXCELLENT_THRESHOLD = 1.5
    GOOD_THRESHOLD = 1.0
    MODERATE_THRESHOLD = 0.5
    POOR_THRESHOLD = 0.0

    def __init__(self, model: Optional[IRTModel] = None):
        """Initialize the analyzer.

        Args:
            model: IRT model to use (defaults to 2PL)
        """
        self.model = model or Model2PL()

    def _calculate_point_biserial(
        self,
        responses: NDArray[np.int_],
        abilities: NDArray[np.float64],
    ) -> float:
        """Calculate point-biserial correlation.

        The point-biserial correlation measures the relationship
        between item scores (0/1) and total ability.

        Args:
            responses: Binary responses (0 or 1)
            abilities: Continuous ability estimates

        Returns:
            Point-biserial correlation coefficient
        """
        if len(responses) < 3:
            return 0.0

        # Separate abilities by response
        correct_abilities = abilities[responses == 1]
        incorrect_abilities = abilities[responses == 0]

        if len(correct_abilities) == 0 or len(incorrect_abilities) == 0:
            return 0.0

        # Point-biserial formula
        mean_correct = np.mean(correct_abilities)
        mean_incorrect = np.mean(incorrect_abilities)
        sd_total = np.std(abilities)

        if sd_total == 0:
            return 0.0

        p = np.mean(responses)
        q = 1 - p

        if p == 0 or q == 0:
            return 0.0

        r_pb = ((mean_correct - mean_incorrect) / sd_total) * np.sqrt(p * q)
        return float(r_pb)

    def _categorize_discrimination(self, discrimination: float) -> QualityCategory:
        """Categorize discrimination value.

        Args:
            discrimination: Discrimination parameter

        Returns:
            Quality category
        """
        if discrimination >= self.EXCELLENT_THRESHOLD:
            return QualityCategory.EXCELLENT
        elif discrimination >= self.GOOD_THRESHOLD:
            return QualityCategory.GOOD
        elif discrimination >= self.MODERATE_THRESHOLD:
            return QualityCategory.MODERATE
        elif discrimination >= self.POOR_THRESHOLD:
            return QualityCategory.POOR
        else:
            return QualityCategory.PROBLEMATIC

    def _generate_recommendations(
        self,
        result: DiscriminationResult,
    ) -> list[str]:
        """Generate improvement recommendations.

        Args:
            result: Partial discrimination result

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if result.category == QualityCategory.PROBLEMATIC:
            recommendations.append(
                "CRITICAL: Negative discrimination detected. "
                "Low-ability learners are outperforming high-ability learners. "
                "This item should be reviewed for ambiguity or errors."
            )
            recommendations.append(
                "Consider removing this item or completely rewriting it."
            )

        elif result.category == QualityCategory.POOR:
            recommendations.append(
                "This item does not differentiate well between ability levels."
            )
            if result.proportion_correct > 0.9:
                recommendations.append(
                    "The item may be too easy. Consider increasing difficulty."
                )
            elif result.proportion_correct < 0.1:
                recommendations.append(
                    "The item may be too hard. Consider decreasing difficulty."
                )
            recommendations.append(
                "Review distractors to ensure they are plausible."
            )

        elif result.category == QualityCategory.MODERATE:
            recommendations.append(
                "Item has acceptable but not optimal discrimination."
            )
            recommendations.append(
                "Consider improving distractor quality to better differentiate learners."
            )

        # Difficulty-based recommendations
        if result.difficulty > 2.0:
            recommendations.append(
                f"High difficulty (b={result.difficulty:.2f}). "
                "Only advanced learners will likely answer correctly."
            )
        elif result.difficulty < -2.0:
            recommendations.append(
                f"Low difficulty (b={result.difficulty:.2f}). "
                "Most learners will answer correctly regardless of ability."
            )

        return recommendations

    def analyze_item(
        self,
        item_id: str,
        responses: NDArray[np.int_],
        abilities: NDArray[np.float64],
    ) -> DiscriminationResult:
        """Analyze discrimination for a single item.

        Args:
            item_id: Item identifier
            responses: Array of binary responses
            abilities: Array of ability estimates

        Returns:
            DiscriminationResult with analysis
        """
        # Calculate proportion correct
        p_correct = float(np.mean(responses))

        # Calculate point-biserial
        r_pb = self._calculate_point_biserial(responses, abilities)

        # Estimate IRT parameters
        from adaptive_learning_core.irt.estimator import ItemEstimator

        estimator = ItemEstimator(self.model)
        params = estimator.estimate_item_2pl(
            item_id,
            responses.tolist(),
            abilities.tolist(),
        )

        # Categorize
        category = self._categorize_discrimination(params.discrimination)

        # Create result
        result = DiscriminationResult(
            item_id=item_id,
            discrimination=params.discrimination,
            difficulty=params.difficulty,
            point_biserial=r_pb,
            proportion_correct=p_correct,
            category=category,
        )

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def analyze_batch(
        self,
        response_matrix: NDArray[np.int_],
        item_ids: list[str],
        abilities: NDArray[np.float64],
    ) -> list[DiscriminationResult]:
        """Analyze discrimination for multiple items.

        Args:
            response_matrix: N x M response matrix
            item_ids: List of M item IDs
            abilities: Array of N ability estimates

        Returns:
            List of DiscriminationResults
        """
        results = []
        n_items = response_matrix.shape[1]

        for j in range(n_items):
            responses = response_matrix[:, j]
            result = self.analyze_item(item_ids[j], responses, abilities)
            results.append(result)

        return results

    def get_summary_statistics(
        self,
        results: list[DiscriminationResult],
    ) -> dict:
        """Get summary statistics for a set of items.

        Args:
            results: List of discrimination results

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}

        discriminations = [r.discrimination for r in results]
        difficulties = [r.difficulty for r in results]

        # Count by category
        category_counts = {cat: 0 for cat in QualityCategory}
        for r in results:
            category_counts[r.category] += 1

        return {
            "total_items": len(results),
            "mean_discrimination": float(np.mean(discriminations)),
            "std_discrimination": float(np.std(discriminations)),
            "min_discrimination": float(np.min(discriminations)),
            "max_discrimination": float(np.max(discriminations)),
            "mean_difficulty": float(np.mean(difficulties)),
            "std_difficulty": float(np.std(difficulties)),
            "categories": {cat.value: count for cat, count in category_counts.items()},
            "acceptable_rate": sum(
                1 for r in results if r.is_acceptable
            ) / len(results),
            "problematic_items": [r.item_id for r in results if r.should_remove],
        }

    def flag_problematic_items(
        self,
        results: list[DiscriminationResult],
    ) -> list[str]:
        """Get list of items that should be reviewed or removed.

        Args:
            results: List of discrimination results

        Returns:
            List of problematic item IDs
        """
        return [r.item_id for r in results if not r.is_acceptable]

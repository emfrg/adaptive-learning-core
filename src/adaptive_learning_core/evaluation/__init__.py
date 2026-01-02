"""Evaluation module for question and content quality.

Provides modern evaluation approaches:
1. LLM-as-Judge: Use LLMs to evaluate question quality
2. RAGAS: RAG-specific metrics for faithfulness and relevance
3. Combined Pipeline: Aggregate scores for filtering

This module does NOT use ROUGE - instead focusing on
semantic evaluation methods that better capture quality.
"""

from adaptive_learning_core.evaluation.llm_judge import (
    LLMJudge,
    JudgeConfig,
    JudgmentResult,
    QualityDimension,
)
from adaptive_learning_core.evaluation.ragas_eval import (
    RAGASEvaluator,
    RAGASConfig,
    RAGASResult,
    RAGASMetric,
)
from adaptive_learning_core.evaluation.question_quality import (
    QuestionQualityPipeline,
    QualityConfig,
    QualityResult,
    QualityGrade,
)

__all__ = [
    # LLM Judge
    "LLMJudge",
    "JudgeConfig",
    "JudgmentResult",
    "QualityDimension",
    # RAGAS
    "RAGASEvaluator",
    "RAGASConfig",
    "RAGASResult",
    "RAGASMetric",
    # Quality Pipeline
    "QuestionQualityPipeline",
    "QualityConfig",
    "QualityResult",
    "QualityGrade",
]

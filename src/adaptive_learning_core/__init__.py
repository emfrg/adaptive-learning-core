"""Adaptive Learning Core - Algorithms for personalized e-learning systems.

This library implements the core algorithms for an adaptive learning system based on:
- Bloom's Taxonomy for question difficulty classification
- Mastery Learning for learner progression tracking
- Item Response Theory (IRT) for question quality estimation
- LangGraph for agent orchestration
- LangChain for LLM operations and RAG

Key Equations from the Paper:
- Eq. 1: cos_sim(q, c) = (q · c) / (|q| × |c|) - Question-card similarity
- Eq. 2: C_{c,q} = exp(sim) / Σexp(sim) - Softmax contribution weights
- Eq. 3: M(c) = M(c) + α × C_{c,q} × δ - Mastery update rule
- Eq. 4: P(c) = (1 - M(c) + γ) / Z - Card selection probability
- Eq. 5: γ_new = γ × δ - Gamma boosting
- Eq. 7-8: M(c) = M(c) × e^{-βλτ} - Knowledge decay

Key components:
- models: Data models for cards, questions, quizzes, modules, and learners
- algorithms: Core adaptive algorithms (mastery update, decay, card/question selection)
- bloom: Bloom's Taxonomy classification
- agents: LangGraph-based orchestration agents
- rag: LangChain-based RAG pipeline
- irt: Item Response Theory models (1PL, 2PL, 3PL) for question evaluation
- evaluation: LangChain-based quality evaluation (LLM-as-judge, RAGAS)
"""

__version__ = "0.1.0"

# =============================================================================
# Models
# =============================================================================
from adaptive_learning_core.models.card import Card, CardState, LearnerStereotype
from adaptive_learning_core.models.question import Question, BloomLevel
from adaptive_learning_core.models.quiz import Quiz, QuizResult, QuizAnswer
from adaptive_learning_core.models.module import Module, ModuleProgress, ModuleStatus
from adaptive_learning_core.models.learner import (
    LearnerProfile,
    PacePreference,
    LearningGoal,
)

# =============================================================================
# Configuration
# =============================================================================
from adaptive_learning_core.config import (
    AdaptiveLearningConfig,
    MasteryConfig,
    DecayConfig,
    SelectionConfig,
    QuestionLinkConfig,
    BloomConfig,
    IRTConfig,
    EvaluationConfig,
    DEFAULT_CONFIG,
)

# =============================================================================
# Core Algorithms
# =============================================================================
from adaptive_learning_core.algorithms.mastery_update import (
    MasteryUpdater,
    MasteryUpdateConfig,
    MasteryUpdateResult,
)
from adaptive_learning_core.algorithms.knowledge_decay import (
    KnowledgeDecay,
    DecayConfig as DecayAlgoConfig,
    DecayResult,
)
from adaptive_learning_core.algorithms.card_selector import (
    CardSelector,
    CardSelectionConfig,
    CardSelectionResult,
)
from adaptive_learning_core.algorithms.question_linker import (
    QuestionLinker,
    QuestionLinkConfig as LinkConfig,
    LinkResult,
)
from adaptive_learning_core.algorithms.question_selector import (
    QuestionSelector,
    QuestionSelectionConfig,
    QuestionSelectionResult,
)

# =============================================================================
# Bloom's Taxonomy
# =============================================================================
from adaptive_learning_core.bloom.taxonomy import (
    BloomLevelInfo,
    BLOOM_LEVEL_INFO,
    get_bloom_info,
    get_action_verbs,
    get_stereotype_for_bloom,
    get_bloom_for_stereotype,
)
from adaptive_learning_core.bloom.classifier import (
    BloomClassifier,
    HeuristicBloomClassifier,
    ClassificationResult,
)

# =============================================================================
# RAG Pipeline (LangChain-based)
# =============================================================================
from adaptive_learning_core.rag.chunker import (
    TextChunker,
    TokenChunker,
    SentenceChunker,
    SemanticChunker,
    Chunk,
    ChunkingConfig,
)
from adaptive_learning_core.rag.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddings,
    CachedEmbeddingProvider,
    EmbeddingResult,
)
from adaptive_learning_core.rag.retriever import (
    VectorRetriever,
    RetrievalResult,
    RetrievalConfig,
)
from adaptive_learning_core.rag.question_generator import (
    QuestionGenerator,
    QuestionGenerationConfig,
    GeneratedQuestion,
)

# =============================================================================
# LangGraph Agents
# =============================================================================
from adaptive_learning_core.agents.state import (
    LearningState,
    QuestionGenState,
    EvaluationState,
)
from adaptive_learning_core.agents.learning_agent import (
    LearningAgent,
    create_learning_graph,
)
from adaptive_learning_core.agents.question_gen_agent import (
    QuestionGenAgent,
    create_question_gen_graph,
)

# =============================================================================
# Item Response Theory (IRT)
# =============================================================================
from adaptive_learning_core.irt.models import (
    IRTModel,
    Rasch1PL,
    Model2PL,
    Model3PL,
    ItemParameters,
    ResponsePattern,
)
from adaptive_learning_core.irt.estimator import (
    IRTEstimator,
    EstimationConfig,
    EstimationResult,
    ItemEstimator,
    AbilityEstimator,
)
from adaptive_learning_core.irt.discrimination import (
    DiscriminationAnalyzer,
    DiscriminationResult,
    QualityCategory,
)

# =============================================================================
# Evaluation (LangChain-based)
# =============================================================================
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

# =============================================================================
# Exports
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Models
    "Card",
    "CardState",
    "LearnerStereotype",
    "Question",
    "BloomLevel",
    "Quiz",
    "QuizResult",
    "QuizAnswer",
    "Module",
    "ModuleProgress",
    "ModuleStatus",
    "LearnerProfile",
    "PacePreference",
    "LearningGoal",
    # Config
    "AdaptiveLearningConfig",
    "MasteryConfig",
    "DecayConfig",
    "SelectionConfig",
    "QuestionLinkConfig",
    "BloomConfig",
    "IRTConfig",
    "EvaluationConfig",
    "DEFAULT_CONFIG",
    # Algorithms
    "MasteryUpdater",
    "MasteryUpdateConfig",
    "MasteryUpdateResult",
    "KnowledgeDecay",
    "DecayAlgoConfig",
    "DecayResult",
    "CardSelector",
    "CardSelectionConfig",
    "CardSelectionResult",
    "QuestionLinker",
    "LinkConfig",
    "LinkResult",
    "QuestionSelector",
    "QuestionSelectionConfig",
    "QuestionSelectionResult",
    # Bloom
    "BloomLevelInfo",
    "BLOOM_LEVEL_INFO",
    "get_bloom_info",
    "get_action_verbs",
    "get_stereotype_for_bloom",
    "get_bloom_for_stereotype",
    "BloomClassifier",
    "HeuristicBloomClassifier",
    "ClassificationResult",
    # RAG
    "TextChunker",
    "TokenChunker",
    "SentenceChunker",
    "SemanticChunker",
    "Chunk",
    "ChunkingConfig",
    "EmbeddingProvider",
    "SentenceTransformerEmbeddings",
    "CachedEmbeddingProvider",
    "EmbeddingResult",
    "VectorRetriever",
    "RetrievalResult",
    "RetrievalConfig",
    "QuestionGenerator",
    "QuestionGenerationConfig",
    "GeneratedQuestion",
    # LangGraph Agents
    "LearningState",
    "QuestionGenState",
    "EvaluationState",
    "LearningAgent",
    "create_learning_graph",
    "QuestionGenAgent",
    "create_question_gen_graph",
    # IRT
    "IRTModel",
    "Rasch1PL",
    "Model2PL",
    "Model3PL",
    "ItemParameters",
    "ResponsePattern",
    "IRTEstimator",
    "EstimationConfig",
    "EstimationResult",
    "ItemEstimator",
    "AbilityEstimator",
    "DiscriminationAnalyzer",
    "DiscriminationResult",
    "QualityCategory",
    # Evaluation
    "LLMJudge",
    "JudgeConfig",
    "JudgmentResult",
    "QualityDimension",
    "RAGASEvaluator",
    "RAGASConfig",
    "RAGASResult",
    "RAGASMetric",
    "QuestionQualityPipeline",
    "QualityConfig",
    "QualityResult",
    "QualityGrade",
]

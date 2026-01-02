"""Core algorithms for the adaptive learning system.

This module implements the mathematical foundations from the research paper:
- Mastery Update (Eq. 3): How mastery scores change based on quiz performance
- Knowledge Decay (Eq. 7-8): How mastery decreases over time (Ebbinghaus)
- Card Selection (Eq. 4-5): Probability-based selection for learning cycles
- Question Linking (Eq. 1-2): Cosine similarity for card-question relationships
- Question Selection (Rules 2-3): Filtering questions by learner level
"""

from adaptive_learning_core.algorithms.mastery_update import (
    MasteryUpdater,
    MasteryUpdateConfig,
    MasteryUpdateResult,
)
from adaptive_learning_core.algorithms.knowledge_decay import (
    KnowledgeDecay,
    DecayConfig,
    DecayResult,
)
from adaptive_learning_core.algorithms.card_selector import (
    CardSelector,
    CardSelectionConfig,
    CardSelectionResult,
)
from adaptive_learning_core.algorithms.question_linker import (
    QuestionLinker,
    QuestionLinkConfig,
    LinkResult,
)
from adaptive_learning_core.algorithms.question_selector import (
    QuestionSelector,
    QuestionSelectionConfig,
    QuestionSelectionResult,
)

__all__ = [
    # Mastery Update
    "MasteryUpdater",
    "MasteryUpdateConfig",
    "MasteryUpdateResult",
    # Knowledge Decay
    "KnowledgeDecay",
    "DecayConfig",
    "DecayResult",
    # Card Selection
    "CardSelector",
    "CardSelectionConfig",
    "CardSelectionResult",
    # Question Linking
    "QuestionLinker",
    "QuestionLinkConfig",
    "LinkResult",
    # Question Selection
    "QuestionSelector",
    "QuestionSelectionConfig",
    "QuestionSelectionResult",
]

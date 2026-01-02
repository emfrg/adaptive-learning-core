"""Bloom's Taxonomy module for cognitive level classification.

This module provides tools for classifying educational content according to
Bloom's Taxonomy, enabling adaptive question selection based on learner level.

Bloom's Taxonomy Levels (from paper Section 3.2.2):
    1. Remember - Recall facts and basic concepts
    2. Understand - Explain ideas or concepts
    3. Apply - Use information in new situations
    4. Analyze - Draw connections among ideas
    5. Evaluate - Justify a stand or decision
    6. Create - Produce new or original work

The taxonomy maps to learner stereotypes:
    - Novice: Remember (level 1)
    - Beginner: Understand (level 2)
    - Intermediate: Apply (level 3)
    - Advanced: Analyze (level 4)
    - Expert: Evaluate/Create (levels 5-6)
"""

from adaptive_learning_core.bloom.taxonomy import (
    BloomLevel,
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
from adaptive_learning_core.bloom.prompts import (
    BLOOM_CLASSIFICATION_PROMPT,
    BLOOM_ANALYSIS_PROMPT,
)

__all__ = [
    # Taxonomy
    "BloomLevel",
    "BloomLevelInfo",
    "BLOOM_LEVEL_INFO",
    "get_bloom_info",
    "get_action_verbs",
    "get_stereotype_for_bloom",
    "get_bloom_for_stereotype",
    # Classifier
    "BloomClassifier",
    "HeuristicBloomClassifier",
    "ClassificationResult",
    # Prompts
    "BLOOM_CLASSIFICATION_PROMPT",
    "BLOOM_ANALYSIS_PROMPT",
]

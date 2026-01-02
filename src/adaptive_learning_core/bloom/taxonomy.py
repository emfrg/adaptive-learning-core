"""Bloom's Taxonomy definitions and utilities.

Provides comprehensive information about Bloom's Taxonomy cognitive levels
and their mapping to learner stereotypes in the adaptive learning system.

From the paper (Section 3.2.2):
    "We classify learners into five distinct stereotypes based on
    demonstrated mastery: Novice, Beginner, Intermediate, Advanced,
    and Expert. These align with Bloom's revised taxonomy."
"""

from dataclasses import dataclass
from typing import Optional

# Re-export BloomLevel from models
from adaptive_learning_core.models.question import BloomLevel
from adaptive_learning_core.models.card import LearnerStereotype


@dataclass(frozen=True)
class BloomLevelInfo:
    """Comprehensive information about a Bloom's Taxonomy level.

    Attributes:
        level: The BloomLevel enum value
        name: Human-readable name
        description: What this cognitive level involves
        action_verbs: Verbs typically used in questions at this level
        question_starters: Common ways to start questions at this level
        typical_stereotype: The learner stereotype this level maps to
        cognitive_process: The underlying cognitive process
    """

    level: BloomLevel
    name: str
    description: str
    action_verbs: tuple[str, ...]
    question_starters: tuple[str, ...]
    typical_stereotype: LearnerStereotype
    cognitive_process: str


# Comprehensive mapping of Bloom levels to detailed information
BLOOM_LEVEL_INFO: dict[BloomLevel, BloomLevelInfo] = {
    BloomLevel.REMEMBER: BloomLevelInfo(
        level=BloomLevel.REMEMBER,
        name="Remember",
        description="Retrieve relevant knowledge from long-term memory. "
        "Recognize and recall facts, terms, basic concepts, and answers.",
        action_verbs=(
            "define",
            "list",
            "name",
            "identify",
            "recall",
            "state",
            "recognize",
            "match",
            "label",
            "select",
        ),
        question_starters=(
            "What is",
            "Who is",
            "Where is",
            "When did",
            "Which of the following",
            "Define",
            "List",
            "Name",
            "Identify",
        ),
        typical_stereotype=LearnerStereotype.NOVICE,
        cognitive_process="Retrieval of stored information",
    ),
    BloomLevel.UNDERSTAND: BloomLevelInfo(
        level=BloomLevel.UNDERSTAND,
        name="Understand",
        description="Construct meaning from instructional messages. "
        "Interpret, classify, summarize, infer, compare, and explain.",
        action_verbs=(
            "explain",
            "describe",
            "summarize",
            "interpret",
            "classify",
            "compare",
            "contrast",
            "paraphrase",
            "discuss",
            "distinguish",
        ),
        question_starters=(
            "Explain how",
            "Describe the",
            "What is the meaning of",
            "Why does",
            "How would you summarize",
            "What is the difference between",
            "Compare and contrast",
        ),
        typical_stereotype=LearnerStereotype.BEGINNER,
        cognitive_process="Construction of meaning from information",
    ),
    BloomLevel.APPLY: BloomLevelInfo(
        level=BloomLevel.APPLY,
        name="Apply",
        description="Carry out or use a procedure in a given situation. "
        "Execute and implement knowledge in new situations.",
        action_verbs=(
            "apply",
            "use",
            "solve",
            "demonstrate",
            "calculate",
            "implement",
            "execute",
            "compute",
            "show",
            "illustrate",
        ),
        question_starters=(
            "How would you use",
            "Calculate the",
            "Solve the following",
            "Apply the concept to",
            "Demonstrate how",
            "What would happen if",
            "How would you implement",
        ),
        typical_stereotype=LearnerStereotype.INTERMEDIATE,
        cognitive_process="Using procedures in new situations",
    ),
    BloomLevel.ANALYZE: BloomLevelInfo(
        level=BloomLevel.ANALYZE,
        name="Analyze",
        description="Break material into parts and determine relationships. "
        "Differentiate, organize, and attribute component parts.",
        action_verbs=(
            "analyze",
            "examine",
            "differentiate",
            "distinguish",
            "compare",
            "investigate",
            "categorize",
            "deconstruct",
            "outline",
            "correlate",
        ),
        question_starters=(
            "Analyze the",
            "What is the relationship between",
            "How does X compare to Y",
            "Examine the",
            "What are the components of",
            "What evidence supports",
            "Distinguish between",
        ),
        typical_stereotype=LearnerStereotype.ADVANCED,
        cognitive_process="Breaking down and relating components",
    ),
    BloomLevel.EVALUATE: BloomLevelInfo(
        level=BloomLevel.EVALUATE,
        name="Evaluate",
        description="Make judgments based on criteria and standards. "
        "Check, critique, and defend conclusions and decisions.",
        action_verbs=(
            "evaluate",
            "assess",
            "judge",
            "critique",
            "justify",
            "defend",
            "prioritize",
            "recommend",
            "rate",
            "validate",
        ),
        question_starters=(
            "Evaluate the",
            "What is the best",
            "Which approach is most effective",
            "Justify your answer",
            "Assess the validity of",
            "Critique the",
            "What criteria would you use",
        ),
        typical_stereotype=LearnerStereotype.EXPERT,
        cognitive_process="Making judgments based on criteria",
    ),
    BloomLevel.CREATE: BloomLevelInfo(
        level=BloomLevel.CREATE,
        name="Create",
        description="Put elements together to form a coherent whole. "
        "Generate, plan, or produce new ideas and products.",
        action_verbs=(
            "create",
            "design",
            "develop",
            "formulate",
            "propose",
            "construct",
            "generate",
            "compose",
            "invent",
            "synthesize",
        ),
        question_starters=(
            "Design a",
            "Create a",
            "Develop a plan for",
            "How would you construct",
            "Propose a solution for",
            "What would you invent to",
            "Formulate a hypothesis about",
        ),
        typical_stereotype=LearnerStereotype.EXPERT,
        cognitive_process="Producing new or original work",
    ),
}


def get_bloom_info(level: BloomLevel) -> BloomLevelInfo:
    """Get detailed information about a Bloom level.

    Args:
        level: The Bloom level to look up

    Returns:
        BloomLevelInfo with full details
    """
    return BLOOM_LEVEL_INFO[level]


def get_action_verbs(level: BloomLevel) -> tuple[str, ...]:
    """Get action verbs typically used for questions at this level.

    Args:
        level: The Bloom level

    Returns:
        Tuple of action verbs
    """
    return BLOOM_LEVEL_INFO[level].action_verbs


def get_stereotype_for_bloom(level: BloomLevel) -> LearnerStereotype:
    """Get the learner stereotype that maps to a Bloom level.

    This implements the mapping from Section 3.2.2 of the paper.

    Args:
        level: The Bloom level

    Returns:
        Corresponding LearnerStereotype
    """
    return BLOOM_LEVEL_INFO[level].typical_stereotype


def get_bloom_for_stereotype(
    stereotype: LearnerStereotype,
) -> tuple[BloomLevel, ...]:
    """Get appropriate Bloom levels for a learner stereotype.

    Returns all Bloom levels that are appropriate for a given
    learner stereotype (at or below their level).

    Args:
        stereotype: The learner stereotype

    Returns:
        Tuple of appropriate BloomLevels (ordered low to high)
    """
    # Map stereotype to maximum appropriate Bloom level
    max_levels: dict[LearnerStereotype, BloomLevel] = {
        LearnerStereotype.NOVICE: BloomLevel.REMEMBER,
        LearnerStereotype.BEGINNER: BloomLevel.UNDERSTAND,
        LearnerStereotype.INTERMEDIATE: BloomLevel.APPLY,
        LearnerStereotype.ADVANCED: BloomLevel.ANALYZE,
        LearnerStereotype.EXPERT: BloomLevel.CREATE,
    }

    max_level = max_levels[stereotype]

    # Return all levels up to and including max
    return tuple(level for level in BloomLevel if level <= max_level)


def bloom_level_from_string(name: str) -> Optional[BloomLevel]:
    """Parse a Bloom level from a string name.

    Handles various formats:
    - Full name: "Remember", "REMEMBER"
    - Number: "1", "2", etc.
    - Level prefix: "Level 1", "level 2"

    Args:
        name: String representation of the level

    Returns:
        BloomLevel or None if not parseable
    """
    name_lower = name.lower().strip()

    # Try by name
    name_map = {
        "remember": BloomLevel.REMEMBER,
        "recall": BloomLevel.REMEMBER,
        "understand": BloomLevel.UNDERSTAND,
        "comprehend": BloomLevel.UNDERSTAND,
        "apply": BloomLevel.APPLY,
        "analyze": BloomLevel.ANALYZE,
        "analyse": BloomLevel.ANALYZE,  # British spelling
        "evaluate": BloomLevel.EVALUATE,
        "create": BloomLevel.CREATE,
        "synthesize": BloomLevel.CREATE,
    }

    if name_lower in name_map:
        return name_map[name_lower]

    # Try by number
    for level in BloomLevel:
        if str(level.value) in name_lower:
            return level
        if f"level {level.value}" in name_lower:
            return level

    return None

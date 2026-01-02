"""Learner models for user profiles and preferences.

A LearnerProfile captures the learner's preferences and settings
from the Calibration Phase (Section 3.1.2).

Key preferences:
- Learning Goal: Target mastery level (Bloom's Taxonomy level)
- Pace Preference: Determines α (Mastery Progression Rate)
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from adaptive_learning_core.models.question import BloomLevel


class PacePreference(str, Enum):
    """Learner's preferred learning pace.

    From Section 3.1.2 (Calibration Phase):
    - Slow: Careful study, frequent testing, slower progression
    - Medium: Balanced pace with thorough understanding
    - Fast: Rapid progress, higher challenge tolerance

    Each pace maps to a different α (Mastery Progression Rate) value.
    """

    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"

    @property
    def description(self) -> str:
        """Human-readable description of the pace."""
        descriptions = {
            "slow": (
                "I like to take my time, carefully studying each piece of "
                "information to fully grasp the details and testing myself often, "
                "even if it means progressing more slowly."
            ),
            "medium": (
                "I prefer a steady pace, balancing speed with thorough understanding."
            ),
            "fast": (
                "I enjoy making rapid progress and don't mind quickly moving "
                "through material, even if it means the challenge is greater."
            ),
        }
        return descriptions[self.value]


class LearningGoal(str, Enum):
    """Learning goal corresponding to Bloom's Taxonomy levels.

    From Section 3.1.2 (Calibration Phase):
    Learners select their desired cognitive complexity level,
    which determines the Target Mastery Level for the module.
    """

    MEMORIZE = "memorize"  # Remember - Level 1
    COMPREHEND = "comprehend"  # Understand - Level 2
    USE = "use"  # Apply - Level 3
    EXAMINE = "examine"  # Analyze - Level 4
    ASSESS = "assess"  # Evaluate - Level 5
    DEVELOP = "develop"  # Create - Level 6

    @property
    def description(self) -> str:
        """Human-readable description from the calibration questionnaire."""
        descriptions = {
            "memorize": "I want to memorize key facts and basic concepts.",
            "comprehend": (
                "I want to comprehend the material and explain ideas in my own words."
            ),
            "use": "I want to use the information to solve problems or perform tasks.",
            "examine": (
                "I want to examine relationships and break down concepts into parts."
            ),
            "assess": (
                "I want to critically assess concepts and make judgments based on criteria."
            ),
            "develop": (
                "I want to develop new ideas or products using the concepts learned."
            ),
        }
        return descriptions[self.value]

    def to_bloom_level(self) -> BloomLevel:
        """Convert learning goal to corresponding Bloom level."""
        level_map = {
            LearningGoal.MEMORIZE: BloomLevel.REMEMBER,
            LearningGoal.COMPREHEND: BloomLevel.UNDERSTAND,
            LearningGoal.USE: BloomLevel.APPLY,
            LearningGoal.EXAMINE: BloomLevel.ANALYZE,
            LearningGoal.ASSESS: BloomLevel.EVALUATE,
            LearningGoal.DEVELOP: BloomLevel.CREATE,
        }
        return level_map[self]

    def to_stereotype_level(self) -> int:
        """Convert to target stereotype level (0-4).

        Remember/Understand -> Novice/Beginner
        Apply -> Intermediate
        Analyze -> Advanced
        Evaluate/Create -> Expert
        """
        return self.to_bloom_level().to_stereotype_level()


class LearnerProfile(BaseModel):
    """Profile for a learner in the adaptive learning system.

    Contains identification, preferences, and global statistics.
    Module-specific progress is tracked in ModuleProgress.

    Attributes:
        id: Unique learner identifier
        display_name: Optional display name
        email: Optional email address
        default_pace: Default pace preference for new modules
        default_goal: Default learning goal for new modules
        created_at: Account creation timestamp
        last_active_at: Last activity timestamp
        total_modules_started: Number of modules started
        total_modules_completed: Number of modules completed
        total_learning_time_minutes: Total time spent learning
        total_questions_answered: Total questions answered across all modules
        overall_accuracy: Overall accuracy rate
    """

    id: str = Field(..., description="Unique learner identifier")
    display_name: Optional[str] = Field(None, description="Optional display name")
    email: Optional[str] = Field(None, description="Optional email address")

    # Default preferences (can be overridden per module)
    default_pace: PacePreference = Field(
        default=PacePreference.MEDIUM,
        description="Default pace preference for new modules",
    )
    default_goal: LearningGoal = Field(
        default=LearningGoal.COMPREHEND,
        description="Default learning goal for new modules",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: Optional[datetime] = Field(None, description="Last activity")

    # Global statistics
    total_modules_started: int = Field(
        default=0, description="Number of modules started"
    )
    total_modules_completed: int = Field(
        default=0, description="Number of modules completed"
    )
    total_learning_time_minutes: float = Field(
        default=0.0, description="Total time spent learning"
    )
    total_questions_answered: int = Field(
        default=0, description="Total questions answered"
    )
    total_correct_answers: int = Field(
        default=0, description="Total correct answers"
    )

    @property
    def overall_accuracy(self) -> float:
        """Overall accuracy rate across all modules."""
        if self.total_questions_answered == 0:
            return 0.0
        return self.total_correct_answers / self.total_questions_answered

    @property
    def completion_rate(self) -> float:
        """Module completion rate."""
        if self.total_modules_started == 0:
            return 0.0
        return self.total_modules_completed / self.total_modules_started

    def __hash__(self) -> int:
        """Allow learner profiles to be used in sets/dicts."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Two profiles are equal if they have the same ID."""
        if not isinstance(other, LearnerProfile):
            return NotImplemented
        return self.id == other.id


class CalibrationResponse(BaseModel):
    """Response from the Calibration Phase questionnaire.

    Captures the learner's responses during module setup,
    which determine the learning parameters for that module.

    From Section 3.1.2:
    1. Learning Goals Setting (determines Target Mastery Level)
    2. Pace Preference Specification (determines α parameter)
    3. Previous Knowledge Assessment (initial mastery scores)
    """

    learner_id: str = Field(..., description="The learner's ID")
    module_id: str = Field(..., description="The module being calibrated")

    # Question 1: Learning Goal
    learning_goal: LearningGoal = Field(
        ..., description="Selected learning goal from questionnaire"
    )

    # Question 2: Pace Preference
    pace_preference: PacePreference = Field(
        ..., description="Selected pace preference"
    )

    # Question 3: Previous Knowledge Assessment results
    initial_exam_completed: bool = Field(
        default=False,
        description="Whether the initial assessment was completed",
    )
    initial_exam_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Score on initial knowledge assessment",
    )
    initial_exam_questions_answered: int = Field(
        default=0,
        description="Number of questions answered in initial assessment",
    )

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)

    @property
    def target_mastery_level(self) -> int:
        """Get the target mastery level (0-4) from the learning goal."""
        return self.learning_goal.to_stereotype_level()

    @property
    def target_bloom_level(self) -> BloomLevel:
        """Get the target Bloom level from the learning goal."""
        return self.learning_goal.to_bloom_level()

    @property
    def is_complete(self) -> bool:
        """Check if calibration is complete."""
        return self.completed_at is not None

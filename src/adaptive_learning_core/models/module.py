"""Module models for learning material organization.

A Module represents a collection of learning material centered around
a specific topic or subject area (e.g., one PDF textbook or chapter).

Based on Section 3.2.2 of the research paper (Knowledge Model):
- Module consists of Cards (material excerpts)
- Module Mastery is determined by individual Card mastery scores
- Target Mastery Level is set during Calibration Phase
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from adaptive_learning_core.models.card import LearnerStereotype


class ModuleStatus(str, Enum):
    """Status of a module for a learner.

    Tracks the overall progression state of a module.
    """

    NOT_STARTED = "not_started"  # No learning sessions yet
    IN_PROGRESS = "in_progress"  # Active learning
    READY_FOR_EXAM = "ready_for_exam"  # All cards meet target mastery
    COMPLETED = "completed"  # Module exam passed


class Module(BaseModel):
    """A collection of learning material.

    Represents a unit of study (e.g., a textbook, chapter, or course module).
    Contains Cards generated from the source material.

    Attributes:
        id: Unique module identifier
        name: Human-readable module name
        description: Optional module description
        source_filename: Original PDF filename
        source_hash: Hash of source file for change detection
        card_ids: List of card IDs belonging to this module
        question_ids: List of generated question IDs
        created_at: When the module was created
        updated_at: Last update timestamp
    """

    id: str = Field(..., description="Unique module identifier")
    name: str = Field(..., description="Human-readable module name")
    description: Optional[str] = Field(None, description="Optional module description")

    # Source material
    source_filename: Optional[str] = Field(
        None, description="Original PDF filename"
    )
    source_hash: Optional[str] = Field(
        None, description="Hash of source file for change detection"
    )

    # Content references
    card_ids: list[str] = Field(
        default_factory=list,
        description="List of card IDs belonging to this module",
    )
    question_ids: list[str] = Field(
        default_factory=list,
        description="List of generated question IDs",
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def num_cards(self) -> int:
        """Number of cards in this module."""
        return len(self.card_ids)

    @property
    def num_questions(self) -> int:
        """Number of questions generated for this module."""
        return len(self.question_ids)

    @property
    def has_content(self) -> bool:
        """Check if the module has any cards."""
        return self.num_cards > 0

    @property
    def has_questions(self) -> bool:
        """Check if questions have been generated."""
        return self.num_questions > 0

    def __hash__(self) -> int:
        """Allow modules to be used in sets/dicts by their ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Two modules are equal if they have the same ID."""
        if not isinstance(other, Module):
            return NotImplemented
        return self.id == other.id


class ModuleProgress(BaseModel):
    """Progress tracking for a learner on a specific module.

    Aggregates Card-level mastery into module-level statistics.
    Tracks calibration settings and progression toward target mastery.

    Based on Section 3.2.2:
    - A Card is "learned" if its Mastery Level reaches Target Mastery Level
    - Module is complete when all valid Cards are learned

    Attributes:
        module_id: Reference to the module
        learner_id: Reference to the learner
        status: Current module status
        target_mastery_level: Target stereotype level (0-4) from Calibration
        calibration_completed: Whether calibration phase was completed
        num_cards_total: Total number of cards in module
        num_cards_ignored: Number of cards ignored by learner
        num_cards_learned: Number of cards meeting target mastery
        average_mastery: Average mastery score across all valid cards
        learning_sessions_count: Number of learning sessions completed
        quizzes_completed: Number of quizzes completed
        total_questions_answered: Total questions answered
        total_correct_answers: Total correct answers
        started_at: When learner started this module
        last_activity_at: Last interaction timestamp
        completed_at: When module was completed (if applicable)
    """

    module_id: str = Field(..., description="Reference to the module")
    learner_id: str = Field(..., description="Reference to the learner")

    # Status
    status: ModuleStatus = Field(
        default=ModuleStatus.NOT_STARTED,
        description="Current module status",
    )

    # Calibration settings
    target_mastery_level: int = Field(
        default=4,  # Expert by default
        ge=0,
        le=4,
        description="Target stereotype level (0-4) from Calibration Phase",
    )
    calibration_completed: bool = Field(
        default=False,
        description="Whether calibration phase was completed",
    )

    # Card statistics
    num_cards_total: int = Field(
        default=0, description="Total number of cards in module"
    )
    num_cards_ignored: int = Field(
        default=0, description="Number of cards ignored by learner"
    )
    num_cards_learned: int = Field(
        default=0, description="Number of cards meeting target mastery"
    )
    num_cards_seen: int = Field(
        default=0, description="Number of cards seen at least once"
    )

    # Mastery statistics
    average_mastery: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average mastery score across all valid cards",
    )

    # Activity tracking
    learning_sessions_count: int = Field(
        default=0, description="Number of learning sessions completed"
    )
    quizzes_completed: int = Field(
        default=0, description="Number of quizzes completed"
    )
    total_questions_answered: int = Field(
        default=0, description="Total questions answered"
    )
    total_correct_answers: int = Field(
        default=0, description="Total correct answers"
    )

    # Timestamps
    started_at: Optional[datetime] = Field(
        None, description="When learner started this module"
    )
    last_activity_at: Optional[datetime] = Field(
        None, description="Last interaction timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None, description="When module was completed"
    )

    @property
    def num_cards_valid(self) -> int:
        """Number of valid (non-ignored) cards."""
        return self.num_cards_total - self.num_cards_ignored

    @property
    def progress_percentage(self) -> float:
        """Progress toward module completion as percentage.

        Based on cards learned / valid cards.
        """
        if self.num_cards_valid == 0:
            return 0.0
        return (self.num_cards_learned / self.num_cards_valid) * 100

    @property
    def coverage_percentage(self) -> float:
        """Percentage of cards that have been seen at least once."""
        if self.num_cards_valid == 0:
            return 0.0
        return (self.num_cards_seen / self.num_cards_valid) * 100

    @property
    def overall_accuracy(self) -> float:
        """Overall accuracy across all quiz answers."""
        if self.total_questions_answered == 0:
            return 0.0
        return self.total_correct_answers / self.total_questions_answered

    @property
    def target_stereotype(self) -> LearnerStereotype:
        """Get the target stereotype from the level."""
        level_map = {
            0: LearnerStereotype.NOVICE,
            1: LearnerStereotype.BEGINNER,
            2: LearnerStereotype.INTERMEDIATE,
            3: LearnerStereotype.ADVANCED,
            4: LearnerStereotype.EXPERT,
        }
        return level_map.get(self.target_mastery_level, LearnerStereotype.EXPERT)

    @property
    def is_ready_for_exam(self) -> bool:
        """Check if module is ready for the final exam (Rule 4).

        From Equation 6: ∀c ∈ V, M(c) ≥ T
        All valid cards must meet target mastery level.
        """
        return (
            self.num_cards_valid > 0
            and self.num_cards_learned >= self.num_cards_valid
        )

    @property
    def is_completed(self) -> bool:
        """Check if module has been completed."""
        return self.status == ModuleStatus.COMPLETED

    def __hash__(self) -> int:
        """Allow progress objects to be used in sets/dicts."""
        return hash((self.module_id, self.learner_id))

    def __eq__(self, other: object) -> bool:
        """Two progress objects are equal if same module and learner."""
        if not isinstance(other, ModuleProgress):
            return NotImplemented
        return (
            self.module_id == other.module_id
            and self.learner_id == other.learner_id
        )

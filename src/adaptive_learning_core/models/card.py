"""Card and CardState models for learning material representation.

A Card represents an excerpt of learning material (e.g., a chunk from a PDF).
CardState tracks a learner's mastery and interaction history with a specific card.

Based on Section 3.2.1 of the research paper:
- Mastery Score M(c) ∈ [0, 1] represents understanding
- Learner Stereotypes map continuous scores to discrete levels
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LearnerStereotype(str, Enum):
    """Learner stereotypes based on mastery level.

    Mapping from Section 3.2.1:
        - Novice: M(c) < 0.2
        - Beginner: 0.2 ≤ M(c) < 0.4
        - Intermediate: 0.4 ≤ M(c) < 0.6
        - Advanced: 0.6 ≤ M(c) < 0.8
        - Expert: M(c) ≥ 0.8

    These stereotypes guide the adaptive system in selecting
    appropriate questions for the student.
    """

    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

    @classmethod
    def from_mastery_score(cls, score: float) -> "LearnerStereotype":
        """Map a continuous mastery score to a learner stereotype.

        Args:
            score: Mastery score in [0, 1]

        Returns:
            The corresponding LearnerStereotype
        """
        if score < 0.2:
            return cls.NOVICE
        elif score < 0.4:
            return cls.BEGINNER
        elif score < 0.6:
            return cls.INTERMEDIATE
        elif score < 0.8:
            return cls.ADVANCED
        else:
            return cls.EXPERT

    @property
    def level(self) -> int:
        """Get the numeric level (0-4) for this stereotype."""
        level_map = {
            LearnerStereotype.NOVICE: 0,
            LearnerStereotype.BEGINNER: 1,
            LearnerStereotype.INTERMEDIATE: 2,
            LearnerStereotype.ADVANCED: 3,
            LearnerStereotype.EXPERT: 4,
        }
        return level_map[self]

    @property
    def min_mastery(self) -> float:
        """Get the minimum mastery score for this stereotype."""
        min_map = {
            LearnerStereotype.NOVICE: 0.0,
            LearnerStereotype.BEGINNER: 0.2,
            LearnerStereotype.INTERMEDIATE: 0.4,
            LearnerStereotype.ADVANCED: 0.6,
            LearnerStereotype.EXPERT: 0.8,
        }
        return min_map[self]


class Card(BaseModel):
    """A card representing an excerpt of learning material.

    Each card maps to a specific chunk of the source material (e.g., PDF).
    Cards are the fundamental unit in the knowledge model.

    Attributes:
        id: Unique identifier for the card
        module_id: Parent module identifier
        text: The excerpt text content
        title: Optional section title
        source_page: Source PDF page number
        start_char: Start character position in source
        end_char: End character position in source
        embedding: Vector embedding for similarity calculations
        created_at: Timestamp of card creation
    """

    id: str = Field(..., description="Unique identifier for the card")
    module_id: str = Field(..., description="Parent module identifier")

    # Content
    text: str = Field(..., description="The excerpt text content")
    title: Optional[str] = Field(None, description="Optional section title")
    source_page: Optional[int] = Field(None, description="Source PDF page number")
    start_char: Optional[int] = Field(None, description="Start character in source")
    end_char: Optional[int] = Field(None, description="End character in source")

    # Embeddings (computed lazily)
    embedding: Optional[list[float]] = Field(
        None, description="Vector embedding for similarity calculations"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": False}  # Allow embedding updates

    def __hash__(self) -> int:
        """Allow cards to be used in sets/dicts by their ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Two cards are equal if they have the same ID."""
        if not isinstance(other, Card):
            return NotImplemented
        return self.id == other.id


class CardState(BaseModel):
    """State of a card for a specific learner.

    Tracks mastery score and interaction history.
    This is the student model component from Section 3.2.1.

    Attributes:
        card_id: Reference to the card
        learner_id: Reference to the learner
        mastery_score: Continuous mastery score M(c) ∈ [0, 1]
        times_seen: Number of times card was shown
        times_correct: Correct answers on related questions
        times_incorrect: Incorrect answers on related questions
        last_seen_at: Last interaction timestamp
        last_quiz_at: Last quiz timestamp
        is_ignored: User chose to ignore this card
        is_learned: Mastery level reached target
    """

    card_id: str = Field(..., description="Reference to the card")
    learner_id: str = Field(..., description="Reference to the learner")

    # Mastery tracking
    mastery_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Continuous mastery score M(c) ∈ [0, 1]",
    )

    # Interaction tracking
    times_seen: int = Field(default=0, description="Number of times card was shown")
    times_correct: int = Field(
        default=0, description="Correct answers on related questions"
    )
    times_incorrect: int = Field(
        default=0, description="Incorrect answers on related questions"
    )

    # Temporal tracking (for decay function)
    last_seen_at: Optional[datetime] = Field(
        None, description="Last interaction timestamp"
    )
    last_quiz_at: Optional[datetime] = Field(None, description="Last quiz timestamp")

    # Status
    is_ignored: bool = Field(
        default=False, description="User chose to ignore this card"
    )
    is_learned: bool = Field(
        default=False, description="Mastery level reached target"
    )

    @property
    def stereotype(self) -> LearnerStereotype:
        """Map continuous mastery score to discrete learner stereotype.

        Returns:
            The LearnerStereotype corresponding to the current mastery score
        """
        return LearnerStereotype.from_mastery_score(self.mastery_score)

    @property
    def stereotype_level(self) -> int:
        """Get the numeric level (0-4) for the current stereotype."""
        return self.stereotype.level

    @property
    def is_unseen(self) -> bool:
        """Check if this card has never been seen."""
        return self.times_seen == 0 and self.mastery_score == 0.0

    @property
    def total_attempts(self) -> int:
        """Get total number of quiz attempts on this card."""
        return self.times_correct + self.times_incorrect

    @property
    def accuracy(self) -> float:
        """Get accuracy rate for this card, or 0 if no attempts."""
        if self.total_attempts == 0:
            return 0.0
        return self.times_correct / self.total_attempts

    def __hash__(self) -> int:
        """Allow card states to be used in sets/dicts."""
        return hash((self.card_id, self.learner_id))

    def __eq__(self, other: object) -> bool:
        """Two card states are equal if they reference the same card and learner."""
        if not isinstance(other, CardState):
            return NotImplemented
        return self.card_id == other.card_id and self.learner_id == other.learner_id

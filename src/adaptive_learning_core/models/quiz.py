"""Quiz models for assessment and results tracking.

A Quiz is a collection of questions presented to the learner.
QuizResult captures the learner's responses and performance.

After a quiz is submitted:
1. The system updates mastery scores (Eq. 3)
2. Card selection probabilities are recalculated (Eq. 4)
3. Gamma may be boosted if score > threshold (Eq. 5)
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class QuizAnswer(BaseModel):
    """A single answer to a quiz question.

    Attributes:
        question_id: Reference to the question
        selected_answer: The answer the learner selected
        is_correct: Whether the answer was correct
        time_spent_seconds: Time spent on this question (optional)
    """

    question_id: str = Field(..., description="Reference to the question")
    selected_answer: str = Field(..., description="The answer selected by learner")
    is_correct: bool = Field(..., description="Whether the answer was correct")
    time_spent_seconds: Optional[float] = Field(
        None, description="Time spent on this question in seconds"
    )


class Quiz(BaseModel):
    """A collection of questions for assessment.

    A Quiz is generated as part of a Learning Cycle, containing
    questions appropriate for the learner's current mastery levels.

    Attributes:
        id: Unique quiz identifier
        module_id: Parent module
        learner_id: The learner taking the quiz
        learning_cycle_id: The learning cycle this quiz belongs to
        question_ids: List of question IDs in this quiz
        card_ids: Cards covered by this quiz
        created_at: When the quiz was generated
        is_module_exam: Whether this is a full module exam
    """

    id: str = Field(..., description="Unique quiz identifier")
    module_id: str = Field(..., description="Parent module")
    learner_id: str = Field(..., description="The learner taking the quiz")
    learning_cycle_id: Optional[str] = Field(
        None, description="The learning cycle this quiz belongs to"
    )

    # Questions
    question_ids: list[str] = Field(
        ..., description="List of question IDs in this quiz"
    )

    # Cards covered
    card_ids: list[str] = Field(
        default_factory=list,
        description="Cards covered by this quiz (via question links)",
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_module_exam: bool = Field(
        default=False,
        description="Whether this is a full module exam (vs regular quiz)",
    )

    @property
    def num_questions(self) -> int:
        """Number of questions in this quiz."""
        return len(self.question_ids)

    def __hash__(self) -> int:
        """Allow quizzes to be used in sets/dicts by their ID."""
        return hash(self.id)


class QuizResult(BaseModel):
    """Result of a completed quiz.

    Captures all answers and calculates performance metrics.
    Used to update mastery scores via the Mastery Update Rule (Eq. 3).

    Attributes:
        id: Unique result identifier
        quiz_id: Reference to the quiz
        learner_id: The learner who took the quiz
        module_id: Parent module
        answers: List of answers to each question
        started_at: When the learner started the quiz
        submitted_at: When the quiz was submitted
        total_time_seconds: Total time spent on the quiz
    """

    id: str = Field(..., description="Unique result identifier")
    quiz_id: str = Field(..., description="Reference to the quiz")
    learner_id: str = Field(..., description="The learner who took the quiz")
    module_id: str = Field(..., description="Parent module")

    # Answers
    answers: list[QuizAnswer] = Field(
        ..., description="List of answers to each question"
    )

    # Timing
    started_at: datetime = Field(..., description="When the learner started")
    submitted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the quiz was submitted",
    )

    @computed_field  # type: ignore[misc]
    @property
    def total_time_seconds(self) -> float:
        """Total time spent on the quiz in seconds."""
        return (self.submitted_at - self.started_at).total_seconds()

    @computed_field  # type: ignore[misc]
    @property
    def num_questions(self) -> int:
        """Number of questions answered."""
        return len(self.answers)

    @computed_field  # type: ignore[misc]
    @property
    def num_correct(self) -> int:
        """Number of correct answers."""
        return sum(1 for a in self.answers if a.is_correct)

    @computed_field  # type: ignore[misc]
    @property
    def num_incorrect(self) -> int:
        """Number of incorrect answers."""
        return sum(1 for a in self.answers if not a.is_correct)

    @computed_field  # type: ignore[misc]
    @property
    def score(self) -> float:
        """Quiz score as a fraction (0-1).

        This score is used in Equation 5 to determine gamma boosting:
        If score > θ (threshold), then γ_new = γ × δ
        """
        if self.num_questions == 0:
            return 0.0
        return self.num_correct / self.num_questions

    @computed_field  # type: ignore[misc]
    @property
    def score_percentage(self) -> float:
        """Quiz score as a percentage (0-100)."""
        return self.score * 100

    def get_answer_for_question(self, question_id: str) -> Optional[QuizAnswer]:
        """Get the answer for a specific question.

        Args:
            question_id: The question ID to look up

        Returns:
            The QuizAnswer if found, None otherwise
        """
        for answer in self.answers:
            if answer.question_id == question_id:
                return answer
        return None

    def get_correct_question_ids(self) -> list[str]:
        """Get IDs of questions answered correctly."""
        return [a.question_id for a in self.answers if a.is_correct]

    def get_incorrect_question_ids(self) -> list[str]:
        """Get IDs of questions answered incorrectly."""
        return [a.question_id for a in self.answers if not a.is_correct]

    def exceeds_threshold(self, threshold: float = 0.8) -> bool:
        """Check if score exceeds the boosting threshold.

        Used to determine if gamma should be boosted (Eq. 5).

        Args:
            threshold: The score threshold θ (default 0.8)

        Returns:
            True if score > threshold
        """
        return self.score > threshold

    def __hash__(self) -> int:
        """Allow results to be used in sets/dicts by their ID."""
        return hash(self.id)

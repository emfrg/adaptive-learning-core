"""Question models with Bloom's Taxonomy integration.

Questions are multiple-choice items generated from learning material,
classified according to Bloom's Taxonomy cognitive levels.

Based on Section 3.2.3 of the research paper:
- Questions are linked to cards using cosine similarity (Eq. 1)
- Contribution weights are computed via softmax (Eq. 2)
- Bloom levels 5 & 6 both map to Expert stereotype
"""

from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field


class BloomLevel(IntEnum):
    """Bloom's Taxonomy cognitive levels (revised, 2001).

    Ordered from lowest to highest cognitive complexity.
    These levels map to question difficulty in the adaptive system.

    The 6 levels are:
        1. Remember - Recall facts and basic concepts
        2. Understand - Explain ideas or concepts
        3. Apply - Use information in new situations
        4. Analyze - Draw connections among ideas
        5. Evaluate - Justify a decision or course of action
        6. Create - Produce new or original work
    """

    REMEMBER = 1
    UNDERSTAND = 2
    APPLY = 3
    ANALYZE = 4
    EVALUATE = 5
    CREATE = 6

    @property
    def description(self) -> str:
        """Human-readable description of the cognitive level."""
        descriptions = {
            1: "Recall facts and basic concepts",
            2: "Explain ideas or concepts",
            3: "Use information in new situations",
            4: "Draw connections among ideas",
            5: "Justify a decision or course of action",
            6: "Produce new or original work",
        }
        return descriptions[self.value]

    @property
    def action_verbs(self) -> list[str]:
        """Common action verbs associated with this level.

        Useful for:
        - Generating questions at specific levels
        - Classifying existing questions
        """
        verbs = {
            1: ["define", "list", "recall", "identify", "name", "state", "match"],
            2: [
                "explain",
                "describe",
                "summarize",
                "interpret",
                "classify",
                "paraphrase",
            ],
            3: [
                "apply",
                "demonstrate",
                "solve",
                "use",
                "implement",
                "execute",
                "calculate",
            ],
            4: [
                "analyze",
                "compare",
                "contrast",
                "differentiate",
                "examine",
                "organize",
            ],
            5: [
                "evaluate",
                "judge",
                "critique",
                "justify",
                "assess",
                "defend",
                "argue",
            ],
            6: [
                "create",
                "design",
                "develop",
                "formulate",
                "construct",
                "produce",
                "invent",
            ],
        }
        return verbs[self.value]

    def to_stereotype_level(self) -> int:
        """Map Bloom level to learner stereotype level.

        From Section 3.2.3: Levels 5 and 6 both map to Expert.

        Mapping:
            - Remember (1) -> Novice (0)
            - Understand (2) -> Beginner (1)
            - Apply (3) -> Intermediate (2)
            - Analyze (4) -> Advanced (3)
            - Evaluate (5) -> Expert (4)
            - Create (6) -> Expert (4)

        Returns:
            Stereotype level 0-4
        """
        if self.value <= 4:
            return self.value - 1
        else:
            return 4  # Both 5 and 6 -> Expert

    @classmethod
    def from_stereotype_level(cls, level: int) -> "BloomLevel":
        """Get the maximum Bloom level appropriate for a stereotype.

        Args:
            level: Stereotype level (0-4)

        Returns:
            The highest appropriate BloomLevel
        """
        level_map = {
            0: cls.REMEMBER,
            1: cls.UNDERSTAND,
            2: cls.APPLY,
            3: cls.ANALYZE,
            4: cls.CREATE,  # Expert can handle all levels
        }
        return level_map.get(level, cls.REMEMBER)


class DistractorQuality(BaseModel):
    """Quality metrics for a distractor (wrong answer).

    Good distractors should be:
    - Plausible (seem like they could be correct)
    - Unambiguous (clearly incorrect upon reflection)
    - Similar in format to the correct answer
    """

    text: str = Field(..., description="The distractor text")
    is_plausible: bool = Field(..., description="Distractor seems reasonable")
    is_unambiguous: bool = Field(..., description="Clearly incorrect")
    similarity_to_correct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity to correct answer (0-1)",
    )
    rationale: Optional[str] = Field(
        None, description="Why this is a good/bad distractor"
    )


class Question(BaseModel):
    """A multiple-choice question generated from learning material.

    Integrates Bloom's Taxonomy for difficulty classification
    and links to source cards via cosine similarity (Eq. 1-2).

    Attributes:
        id: Unique question identifier
        module_id: Parent module
        stem: The question text
        correct_answer: The correct answer
        distractors: Incorrect answer options
        explanation: Explanation of the correct answer
        bloom_level: Cognitive complexity level
        linked_card_ids: Cards this question relates to
        card_contribution_weights: Contribution weight C_{c,q} for each card
        embedding: Vector embedding for similarity calculations
        difficulty_estimate: Estimated difficulty from IRT (b parameter)
        discrimination_estimate: How well question differentiates (a parameter)
        guessing_estimate: Guessing parameter for 3PL model (c parameter)
        generator_model: LLM used to generate the question
        quality_score: Combined quality score from evaluation
    """

    id: str = Field(..., description="Unique question identifier")
    module_id: str = Field(..., description="Parent module")

    # Question content
    stem: str = Field(..., description="The question text")
    correct_answer: str = Field(..., description="The correct answer")
    distractors: list[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="Incorrect answer options",
    )
    explanation: str = Field(..., description="Explanation of the correct answer")

    # Classification
    bloom_level: BloomLevel = Field(..., description="Cognitive complexity level")

    # Source linking (Eq. 1-2 from paper)
    linked_card_ids: list[str] = Field(
        default_factory=list,
        description="Cards this question relates to (by ID)",
    )
    card_contribution_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Contribution weight C_{c,q} for each linked card",
    )

    # Embedding for similarity calculations
    embedding: Optional[list[float]] = Field(
        None, description="Vector embedding of the question"
    )

    # IRT parameters (estimated from response data)
    difficulty_estimate: Optional[float] = Field(
        None,
        description="Estimated difficulty (b) from IRT, typically in [-3, 3]",
    )
    discrimination_estimate: Optional[float] = Field(
        None,
        description="Discrimination (a) - how well question differentiates abilities",
    )
    guessing_estimate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Guessing parameter (c) for 3PL model",
    )

    # Generation metadata
    generator_model: Optional[str] = Field(
        None, description="LLM model used to generate this question"
    )
    generation_prompt_version: Optional[str] = Field(
        None, description="Version of the prompt template used"
    )

    # Quality metrics
    quality_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Combined quality score from evaluation pipeline",
    )

    @property
    def all_options(self) -> list[str]:
        """All answer options (correct + distractors) in order."""
        return [self.correct_answer] + self.distractors

    @property
    def num_options(self) -> int:
        """Number of answer options."""
        return 1 + len(self.distractors)

    @property
    def difficulty_level(self) -> int:
        """Numeric difficulty level (0-4) for stereotype matching.

        Maps Bloom level to stereotype level for adaptive selection.
        """
        return self.bloom_level.to_stereotype_level()

    @property
    def is_linked(self) -> bool:
        """Check if this question has been linked to cards."""
        return len(self.linked_card_ids) > 0

    @property
    def has_irt_params(self) -> bool:
        """Check if IRT parameters have been estimated."""
        return self.difficulty_estimate is not None

    def get_contribution_weight(self, card_id: str) -> float:
        """Get the contribution weight for a specific card.

        If the card is not linked, returns 0.
        If linked but weight not set, returns equal weight.
        """
        if card_id not in self.linked_card_ids:
            return 0.0
        return self.card_contribution_weights.get(
            card_id, 1.0 / max(len(self.linked_card_ids), 1)
        )

    def __hash__(self) -> int:
        """Allow questions to be used in sets/dicts by their ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Two questions are equal if they have the same ID."""
        if not isinstance(other, Question):
            return NotImplemented
        return self.id == other.id

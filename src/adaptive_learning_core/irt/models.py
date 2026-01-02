"""Item Response Theory (IRT) models.

Implements the three major IRT models used in educational assessment:

1PL (Rasch) Model:
    P(X=1|θ,b) = 1 / (1 + e^(-(θ-b)))
    - Only considers difficulty (b)
    - Assumes equal discrimination for all items

2PL Model:
    P(X=1|θ,a,b) = 1 / (1 + e^(-a(θ-b)))
    - Adds discrimination parameter (a)
    - Different items can differentiate ability differently

3PL Model:
    P(X=1|θ,a,b,c) = c + (1-c) / (1 + e^(-a(θ-b)))
    - Adds guessing parameter (c)
    - Accounts for correct answers by chance
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class ItemParameters:
    """Parameters for an IRT item (question).

    Attributes:
        item_id: Unique identifier for the item
        difficulty: Difficulty parameter (b), typically -3 to +3
        discrimination: Discrimination parameter (a), typically 0.5 to 2.5
        guessing: Guessing/pseudo-chance parameter (c), typically 0 to 0.35
        upper_asymptote: Upper asymptote (d), typically 1.0
    """

    item_id: str
    difficulty: float = 0.0  # b parameter
    discrimination: float = 1.0  # a parameter
    guessing: float = 0.0  # c parameter
    upper_asymptote: float = 1.0  # d parameter (rarely used)

    def __post_init__(self):
        # Validate parameters
        if self.discrimination <= 0:
            raise ValueError(f"discrimination must be positive, got {self.discrimination}")
        if not 0 <= self.guessing < 1:
            raise ValueError(f"guessing must be in [0, 1), got {self.guessing}")
        if not 0 < self.upper_asymptote <= 1:
            raise ValueError(f"upper_asymptote must be in (0, 1], got {self.upper_asymptote}")

    @property
    def is_rasch(self) -> bool:
        """Whether this item follows Rasch (1PL) constraints."""
        return self.discrimination == 1.0 and self.guessing == 0.0

    @property
    def is_2pl(self) -> bool:
        """Whether this item follows 2PL constraints (no guessing)."""
        return self.guessing == 0.0


@dataclass
class ResponsePattern:
    """A learner's response pattern across items.

    Attributes:
        learner_id: Identifier for the learner
        responses: Dictionary of item_id -> response (0 or 1)
        ability_estimate: Estimated ability if computed
    """

    learner_id: str
    responses: dict[str, int] = field(default_factory=dict)
    ability_estimate: Optional[float] = None

    @property
    def total_correct(self) -> int:
        """Total number of correct responses."""
        return sum(self.responses.values())

    @property
    def total_items(self) -> int:
        """Total number of items responded to."""
        return len(self.responses)

    @property
    def proportion_correct(self) -> float:
        """Proportion of correct responses."""
        if self.total_items == 0:
            return 0.0
        return self.total_correct / self.total_items


class IRTModel(ABC):
    """Abstract base class for IRT models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model (e.g., '1PL', '2PL', '3PL')."""
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """Number of parameters per item."""
        pass

    @abstractmethod
    def probability(
        self,
        theta: float,
        params: ItemParameters,
    ) -> float:
        """Calculate probability of correct response.

        Args:
            theta: Learner ability parameter
            params: Item parameters

        Returns:
            Probability of correct response P(X=1|θ)
        """
        pass

    def probability_batch(
        self,
        theta: float,
        params_list: list[ItemParameters],
    ) -> NDArray[np.float64]:
        """Calculate probabilities for multiple items.

        Args:
            theta: Learner ability
            params_list: List of item parameters

        Returns:
            Array of probabilities
        """
        return np.array([self.probability(theta, p) for p in params_list])

    def information(
        self,
        theta: float,
        params: ItemParameters,
    ) -> float:
        """Calculate Fisher information at given ability.

        Information measures how well the item discriminates
        at a particular ability level.

        Args:
            theta: Ability level
            params: Item parameters

        Returns:
            Fisher information I(θ)
        """
        p = self.probability(theta, params)
        q = 1 - p
        if p <= 0 or q <= 0:
            return 0.0
        return (params.discrimination ** 2) * p * q

    def log_likelihood(
        self,
        theta: float,
        params: ItemParameters,
        response: int,
    ) -> float:
        """Calculate log-likelihood of a response.

        Args:
            theta: Ability level
            params: Item parameters
            response: Actual response (0 or 1)

        Returns:
            Log-likelihood ln(P(X=response|θ))
        """
        p = self.probability(theta, params)
        p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0)

        if response == 1:
            return np.log(p)
        else:
            return np.log(1 - p)

    def expected_score(
        self,
        theta: float,
        params_list: list[ItemParameters],
    ) -> float:
        """Calculate expected score (sum of probabilities).

        Args:
            theta: Ability level
            params_list: List of item parameters

        Returns:
            Expected number of correct responses
        """
        return sum(self.probability(theta, p) for p in params_list)

    def test_information(
        self,
        theta: float,
        params_list: list[ItemParameters],
    ) -> float:
        """Calculate total test information at ability level.

        Args:
            theta: Ability level
            params_list: List of item parameters

        Returns:
            Total Fisher information
        """
        return sum(self.information(theta, p) for p in params_list)


class Rasch1PL(IRTModel):
    """Rasch (1-Parameter Logistic) Model.

    P(X=1|θ,b) = 1 / (1 + e^(-(θ-b)))

    Only the difficulty parameter (b) varies across items.
    Discrimination is fixed at 1.0 and guessing is 0.

    The Rasch model assumes all items are equally good at
    discriminating between ability levels.

    Example:
        >>> model = Rasch1PL()
        >>> params = ItemParameters(item_id="q1", difficulty=0.5)
        >>> p = model.probability(theta=1.0, params=params)
        >>> print(f"P(correct) = {p:.3f}")  # 0.622
    """

    @property
    def model_name(self) -> str:
        return "1PL"

    @property
    def num_parameters(self) -> int:
        return 1

    def probability(
        self,
        theta: float,
        params: ItemParameters,
    ) -> float:
        """Calculate probability using Rasch model.

        P(X=1|θ,b) = 1 / (1 + e^(-(θ-b)))

        Args:
            theta: Learner ability
            params: Item parameters (only difficulty used)

        Returns:
            Probability of correct response
        """
        exponent = -(theta - params.difficulty)
        return 1.0 / (1.0 + np.exp(exponent))

    def information(
        self,
        theta: float,
        params: ItemParameters,
    ) -> float:
        """Calculate Rasch model information.

        For Rasch: I(θ) = P(θ) * Q(θ) = P(θ) * (1-P(θ))

        Args:
            theta: Ability level
            params: Item parameters

        Returns:
            Fisher information
        """
        p = self.probability(theta, params)
        return p * (1 - p)


class Model2PL(IRTModel):
    """2-Parameter Logistic Model.

    P(X=1|θ,a,b) = 1 / (1 + e^(-a(θ-b)))

    Parameters:
    - a: Discrimination - how well the item differentiates ability levels
    - b: Difficulty - the ability level where P=0.5

    Higher discrimination means the probability changes more
    rapidly around the difficulty level.

    Example:
        >>> model = Model2PL()
        >>> params = ItemParameters(
        ...     item_id="q1",
        ...     difficulty=0.0,
        ...     discrimination=1.5
        ... )
        >>> p = model.probability(theta=1.0, params=params)
        >>> print(f"P(correct) = {p:.3f}")
    """

    @property
    def model_name(self) -> str:
        return "2PL"

    @property
    def num_parameters(self) -> int:
        return 2

    def probability(
        self,
        theta: float,
        params: ItemParameters,
    ) -> float:
        """Calculate probability using 2PL model.

        P(X=1|θ,a,b) = 1 / (1 + e^(-a(θ-b)))

        Args:
            theta: Learner ability
            params: Item parameters

        Returns:
            Probability of correct response
        """
        exponent = -params.discrimination * (theta - params.difficulty)
        return 1.0 / (1.0 + np.exp(exponent))


class Model3PL(IRTModel):
    """3-Parameter Logistic Model.

    P(X=1|θ,a,b,c) = c + (1-c) / (1 + e^(-a(θ-b)))

    Parameters:
    - a: Discrimination
    - b: Difficulty
    - c: Guessing (pseudo-chance) - lower asymptote

    The guessing parameter accounts for correct answers by chance,
    especially relevant for multiple-choice questions.

    For 4-option MCQ, c is often around 0.25.

    Example:
        >>> model = Model3PL()
        >>> params = ItemParameters(
        ...     item_id="q1",
        ...     difficulty=0.0,
        ...     discrimination=1.5,
        ...     guessing=0.25
        ... )
        >>> p = model.probability(theta=-3.0, params=params)
        >>> print(f"P(correct) ≈ {params.guessing}")  # Low ability, still guessing
    """

    @property
    def model_name(self) -> str:
        return "3PL"

    @property
    def num_parameters(self) -> int:
        return 3

    def probability(
        self,
        theta: float,
        params: ItemParameters,
    ) -> float:
        """Calculate probability using 3PL model.

        P(X=1|θ,a,b,c) = c + (1-c) / (1 + e^(-a(θ-b)))

        Args:
            theta: Learner ability
            params: Item parameters

        Returns:
            Probability of correct response
        """
        c = params.guessing
        exponent = -params.discrimination * (theta - params.difficulty)
        return c + (1 - c) / (1.0 + np.exp(exponent))

    def information(
        self,
        theta: float,
        params: ItemParameters,
    ) -> float:
        """Calculate 3PL information function.

        The 3PL information is reduced compared to 2PL due to
        the guessing parameter.

        Args:
            theta: Ability level
            params: Item parameters

        Returns:
            Fisher information
        """
        p = self.probability(theta, params)
        p_star = (p - params.guessing) / (1 - params.guessing)
        q_star = 1 - p_star

        if p <= params.guessing or p_star <= 0 or q_star <= 0:
            return 0.0

        # I(θ) = a² * (Q*/P) * P*² for 3PL
        return (params.discrimination ** 2) * (q_star / p) * (p_star ** 2)


def get_model(model_type: str) -> IRTModel:
    """Factory function to get an IRT model by name.

    Args:
        model_type: One of "1PL", "2PL", "3PL"

    Returns:
        Corresponding IRT model instance

    Raises:
        ValueError: If model_type is unknown
    """
    models = {
        "1PL": Rasch1PL,
        "2PL": Model2PL,
        "3PL": Model3PL,
    }

    if model_type.upper() not in models:
        raise ValueError(f"Unknown model type: {model_type}. Use: {list(models.keys())}")

    return models[model_type.upper()]()

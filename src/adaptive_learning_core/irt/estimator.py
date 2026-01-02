"""IRT parameter estimation using Maximum Likelihood.

Provides estimators for:
1. Item parameters (difficulty, discrimination, guessing)
2. Learner ability (theta)

Uses iterative optimization methods including:
- Newton-Raphson for ability estimation
- Joint Maximum Likelihood (JML) for item calibration
- Expected A Posteriori (EAP) for Bayesian estimation
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy import stats

from adaptive_learning_core.irt.models import (
    IRTModel,
    ItemParameters,
    ResponsePattern,
    Rasch1PL,
    Model2PL,
    Model3PL,
)


@dataclass(frozen=True)
class EstimationConfig:
    """Configuration for IRT estimation.

    Attributes:
        model_type: IRT model to use ("1PL", "2PL", "3PL")
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance
        theta_range: Range for ability parameter (-4 to 4 typical)
        difficulty_range: Range for difficulty parameter
        discrimination_bounds: Bounds for discrimination parameter
        guessing_bounds: Bounds for guessing parameter
        prior_theta_mean: Prior mean for theta (Bayesian)
        prior_theta_sd: Prior SD for theta (Bayesian)
    """

    model_type: str = "2PL"
    max_iterations: int = 100
    tolerance: float = 1e-4
    theta_range: tuple[float, float] = (-4.0, 4.0)
    difficulty_range: tuple[float, float] = (-4.0, 4.0)
    discrimination_bounds: tuple[float, float] = (0.2, 3.0)
    guessing_bounds: tuple[float, float] = (0.0, 0.35)
    prior_theta_mean: float = 0.0
    prior_theta_sd: float = 1.0


@dataclass
class EstimationResult:
    """Result of parameter estimation.

    Attributes:
        parameters: Estimated parameters
        standard_errors: Standard errors of estimates
        log_likelihood: Final log-likelihood
        converged: Whether estimation converged
        iterations: Number of iterations used
        method: Estimation method used
    """

    parameters: dict[str, float]
    standard_errors: dict[str, float] = field(default_factory=dict)
    log_likelihood: float = 0.0
    converged: bool = True
    iterations: int = 0
    method: str = "MLE"


class AbilityEstimator:
    """Estimates learner ability from response patterns.

    Provides multiple estimation methods:
    - MLE: Maximum Likelihood Estimation
    - EAP: Expected A Posteriori (Bayesian)
    - MAP: Maximum A Posteriori (Bayesian)

    Example:
        >>> estimator = AbilityEstimator(model=Model2PL())
        >>> ability = estimator.estimate(response_pattern, item_params)
        >>> print(f"Estimated ability: {ability:.2f}")
    """

    def __init__(
        self,
        model: IRTModel,
        config: Optional[EstimationConfig] = None,
    ):
        """Initialize the ability estimator.

        Args:
            model: IRT model to use
            config: Estimation configuration
        """
        self.model = model
        self.config = config or EstimationConfig()

    def _log_likelihood(
        self,
        theta: float,
        responses: dict[str, int],
        item_params: dict[str, ItemParameters],
    ) -> float:
        """Calculate log-likelihood for given theta.

        Args:
            theta: Ability parameter
            responses: Item responses
            item_params: Item parameters

        Returns:
            Log-likelihood value
        """
        ll = 0.0
        for item_id, response in responses.items():
            if item_id in item_params:
                ll += self.model.log_likelihood(
                    theta, item_params[item_id], response
                )
        return ll

    def _neg_log_likelihood(
        self,
        theta: float,
        responses: dict[str, int],
        item_params: dict[str, ItemParameters],
    ) -> float:
        """Negative log-likelihood for minimization."""
        return -self._log_likelihood(theta, responses, item_params)

    def estimate_mle(
        self,
        responses: dict[str, int],
        item_params: dict[str, ItemParameters],
    ) -> EstimationResult:
        """Estimate ability using Maximum Likelihood.

        Args:
            responses: Item responses
            item_params: Item parameters

        Returns:
            EstimationResult with estimated theta
        """
        # Handle edge cases
        if not responses:
            return EstimationResult(
                parameters={"theta": 0.0},
                converged=False,
                method="MLE",
            )

        # All correct or all incorrect
        n_correct = sum(responses.values())
        n_total = len(responses)

        if n_correct == 0:
            return EstimationResult(
                parameters={"theta": self.config.theta_range[0]},
                converged=True,
                method="MLE_boundary",
            )
        if n_correct == n_total:
            return EstimationResult(
                parameters={"theta": self.config.theta_range[1]},
                converged=True,
                method="MLE_boundary",
            )

        # Optimize
        result = optimize.minimize_scalar(
            lambda t: self._neg_log_likelihood(t, responses, item_params),
            bounds=self.config.theta_range,
            method="bounded",
        )

        # Calculate standard error (from information)
        theta_hat = result.x
        info = sum(
            self.model.information(theta_hat, item_params[iid])
            for iid in responses
            if iid in item_params
        )
        se = 1.0 / np.sqrt(info) if info > 0 else np.nan

        return EstimationResult(
            parameters={"theta": theta_hat},
            standard_errors={"theta": se},
            log_likelihood=-result.fun,
            converged=result.success,
            iterations=result.nfev,
            method="MLE",
        )

    def estimate_eap(
        self,
        responses: dict[str, int],
        item_params: dict[str, ItemParameters],
        n_quadrature: int = 40,
    ) -> EstimationResult:
        """Estimate ability using Expected A Posteriori (Bayesian).

        Uses Gauss-Hermite quadrature for numerical integration.

        Args:
            responses: Item responses
            item_params: Item parameters
            n_quadrature: Number of quadrature points

        Returns:
            EstimationResult with estimated theta
        """
        # Generate quadrature points (normal prior)
        points, weights = np.polynomial.hermite.hermgauss(n_quadrature)

        # Transform points for N(0, 1) prior
        points = points * np.sqrt(2) * self.config.prior_theta_sd + self.config.prior_theta_mean
        weights = weights / np.sqrt(np.pi)

        # Calculate likelihood at each point
        likelihoods = np.zeros(n_quadrature)
        for i, theta in enumerate(points):
            ll = self._log_likelihood(theta, responses, item_params)
            likelihoods[i] = np.exp(ll)

        # EAP = E[θ|X] = Σ θ_i * L(X|θ_i) * w_i / Σ L(X|θ_i) * w_i
        weighted_likes = likelihoods * weights
        denom = np.sum(weighted_likes)

        if denom <= 0:
            return EstimationResult(
                parameters={"theta": 0.0},
                converged=False,
                method="EAP",
            )

        theta_hat = np.sum(points * weighted_likes) / denom

        # Posterior variance
        var = np.sum(((points - theta_hat) ** 2) * weighted_likes) / denom
        se = np.sqrt(var)

        return EstimationResult(
            parameters={"theta": theta_hat},
            standard_errors={"theta": se},
            converged=True,
            method="EAP",
        )

    def estimate(
        self,
        pattern: ResponsePattern,
        item_params: dict[str, ItemParameters],
        method: str = "MLE",
    ) -> float:
        """Estimate ability for a response pattern.

        Args:
            pattern: Response pattern
            item_params: Item parameters
            method: Estimation method ("MLE" or "EAP")

        Returns:
            Estimated ability
        """
        if method.upper() == "EAP":
            result = self.estimate_eap(pattern.responses, item_params)
        else:
            result = self.estimate_mle(pattern.responses, item_params)

        pattern.ability_estimate = result.parameters["theta"]
        return pattern.ability_estimate


class ItemEstimator:
    """Estimates item parameters from response data.

    Uses Joint Maximum Likelihood (JML) or Marginal Maximum
    Likelihood (MML) for calibration.

    Example:
        >>> estimator = ItemEstimator(model=Model2PL())
        >>> params = estimator.estimate_item(responses, abilities)
    """

    def __init__(
        self,
        model: IRTModel,
        config: Optional[EstimationConfig] = None,
    ):
        """Initialize the item estimator.

        Args:
            model: IRT model to use
            config: Estimation configuration
        """
        self.model = model
        self.config = config or EstimationConfig()

    def estimate_difficulty(
        self,
        responses: list[int],
        abilities: list[float],
        discrimination: float = 1.0,
    ) -> float:
        """Estimate difficulty given responses and abilities.

        Uses the logit of proportion correct as initial estimate,
        then refines with MLE.

        Args:
            responses: List of responses (0 or 1)
            abilities: Corresponding learner abilities
            discrimination: Fixed discrimination (for 1PL)

        Returns:
            Estimated difficulty parameter
        """
        if not responses:
            return 0.0

        p_correct = np.mean(responses)
        if p_correct == 0:
            return self.config.difficulty_range[1]
        if p_correct == 1:
            return self.config.difficulty_range[0]

        # Initial estimate from logit
        avg_ability = np.mean(abilities)
        logit_p = np.log(p_correct / (1 - p_correct))
        initial_b = avg_ability - logit_p / discrimination

        # Refine with MLE
        def neg_ll(b):
            ll = 0.0
            for response, theta in zip(responses, abilities):
                params = ItemParameters(
                    item_id="temp",
                    difficulty=b,
                    discrimination=discrimination,
                )
                ll += self.model.log_likelihood(theta, params, response)
            return -ll

        result = optimize.minimize_scalar(
            neg_ll,
            bounds=self.config.difficulty_range,
            method="bounded",
        )

        return result.x

    def estimate_discrimination(
        self,
        responses: list[int],
        abilities: list[float],
        difficulty: float,
    ) -> float:
        """Estimate discrimination given difficulty.

        Args:
            responses: List of responses
            abilities: Corresponding abilities
            difficulty: Fixed difficulty parameter

        Returns:
            Estimated discrimination parameter
        """
        if not responses:
            return 1.0

        def neg_ll(a):
            if a <= 0:
                return 1e10
            ll = 0.0
            for response, theta in zip(responses, abilities):
                params = ItemParameters(
                    item_id="temp",
                    difficulty=difficulty,
                    discrimination=a,
                )
                ll += self.model.log_likelihood(theta, params, response)
            return -ll

        result = optimize.minimize_scalar(
            neg_ll,
            bounds=self.config.discrimination_bounds,
            method="bounded",
        )

        return result.x

    def estimate_item_2pl(
        self,
        item_id: str,
        responses: list[int],
        abilities: list[float],
    ) -> ItemParameters:
        """Estimate 2PL parameters for an item.

        Args:
            item_id: Item identifier
            responses: List of responses
            abilities: Corresponding abilities

        Returns:
            Estimated ItemParameters
        """

        def neg_ll(params):
            a, b = params
            if a <= 0:
                return 1e10
            ll = 0.0
            item_params = ItemParameters(
                item_id=item_id,
                difficulty=b,
                discrimination=a,
            )
            for response, theta in zip(responses, abilities):
                ll += self.model.log_likelihood(theta, item_params, response)
            return -ll

        # Initial estimates
        p_correct = np.mean(responses) if responses else 0.5
        avg_ability = np.mean(abilities) if abilities else 0.0
        initial_b = avg_ability - np.log(max(0.01, p_correct) / max(0.01, 1 - p_correct))
        initial_a = 1.0

        result = optimize.minimize(
            neg_ll,
            x0=[initial_a, initial_b],
            bounds=[
                self.config.discrimination_bounds,
                self.config.difficulty_range,
            ],
            method="L-BFGS-B",
        )

        a_hat, b_hat = result.x

        return ItemParameters(
            item_id=item_id,
            difficulty=b_hat,
            discrimination=max(0.2, a_hat),
        )


class IRTEstimator:
    """Combined estimator for IRT calibration.

    Coordinates item and ability estimation in an iterative
    fashion (Joint Maximum Likelihood).

    Example:
        >>> estimator = IRTEstimator(config)
        >>> item_params = estimator.calibrate(response_matrix)
    """

    def __init__(self, config: Optional[EstimationConfig] = None):
        """Initialize the IRT estimator.

        Args:
            config: Estimation configuration
        """
        self.config = config or EstimationConfig()

        # Select model
        if self.config.model_type == "1PL":
            self.model = Rasch1PL()
        elif self.config.model_type == "2PL":
            self.model = Model2PL()
        else:
            self.model = Model3PL()

        self.ability_estimator = AbilityEstimator(self.model, self.config)
        self.item_estimator = ItemEstimator(self.model, self.config)

    def calibrate(
        self,
        response_matrix: NDArray[np.int_],
        item_ids: list[str],
        initial_abilities: Optional[NDArray[np.float64]] = None,
    ) -> dict[str, ItemParameters]:
        """Calibrate item parameters from a response matrix.

        Uses alternating estimation:
        1. Estimate abilities given item parameters
        2. Estimate item parameters given abilities
        3. Repeat until convergence

        Args:
            response_matrix: N x M matrix (N learners, M items)
            item_ids: List of M item IDs
            initial_abilities: Optional initial ability estimates

        Returns:
            Dictionary of item_id -> ItemParameters
        """
        n_learners, n_items = response_matrix.shape

        if len(item_ids) != n_items:
            raise ValueError("item_ids must match number of columns")

        # Initialize
        if initial_abilities is not None:
            abilities = initial_abilities.copy()
        else:
            # Use proportion correct as initial ability
            abilities = np.array([
                stats.norm.ppf(max(0.01, min(0.99, np.mean(row))))
                for row in response_matrix
            ])

        # Initialize item parameters
        item_params = {
            iid: ItemParameters(
                item_id=iid,
                difficulty=0.0,
                discrimination=1.0,
            )
            for iid in item_ids
        }

        prev_ll = -np.inf

        for iteration in range(self.config.max_iterations):
            # E-step: Estimate item parameters
            for j, item_id in enumerate(item_ids):
                responses = response_matrix[:, j].tolist()
                ability_list = abilities.tolist()

                if self.config.model_type == "1PL":
                    b = self.item_estimator.estimate_difficulty(
                        responses, ability_list, discrimination=1.0
                    )
                    item_params[item_id] = ItemParameters(
                        item_id=item_id,
                        difficulty=b,
                        discrimination=1.0,
                    )
                else:
                    item_params[item_id] = self.item_estimator.estimate_item_2pl(
                        item_id, responses, ability_list
                    )

            # M-step: Estimate abilities
            for i in range(n_learners):
                responses = {
                    item_ids[j]: int(response_matrix[i, j])
                    for j in range(n_items)
                    if not np.isnan(response_matrix[i, j])
                }
                result = self.ability_estimator.estimate_mle(responses, item_params)
                abilities[i] = result.parameters["theta"]

            # Check convergence
            ll = 0.0
            for i in range(n_learners):
                for j in range(n_items):
                    if not np.isnan(response_matrix[i, j]):
                        ll += self.model.log_likelihood(
                            abilities[i],
                            item_params[item_ids[j]],
                            int(response_matrix[i, j]),
                        )

            if abs(ll - prev_ll) < self.config.tolerance:
                break

            prev_ll = ll

        return item_params

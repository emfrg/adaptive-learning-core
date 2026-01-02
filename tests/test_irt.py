"""Tests for Item Response Theory models."""

import pytest
import numpy as np

from adaptive_learning_core.irt.models import (
    Rasch1PL,
    Model2PL,
    Model3PL,
    ItemParameters,
    ResponsePattern,
)
from adaptive_learning_core.irt.discrimination import (
    DiscriminationAnalyzer,
    QualityCategory,
)


class TestItemParameters:
    """Tests for ItemParameters dataclass."""

    def test_valid_parameters(self):
        """Valid parameters should create ItemParameters."""
        params = ItemParameters(
            item_id="q1",
            difficulty=0.0,
            discrimination=1.0,
            guessing=0.25,
        )
        assert params.difficulty == 0.0
        assert params.discrimination == 1.0
        assert params.guessing == 0.25

    def test_invalid_discrimination(self):
        """Negative discrimination should raise error."""
        with pytest.raises(ValueError):
            ItemParameters(item_id="q1", discrimination=-0.5)

    def test_invalid_guessing(self):
        """Guessing >= 1 should raise error."""
        with pytest.raises(ValueError):
            ItemParameters(item_id="q1", guessing=1.0)

    def test_is_rasch(self):
        """Should identify Rasch-compliant items."""
        rasch_item = ItemParameters(item_id="q1", discrimination=1.0, guessing=0.0)
        assert rasch_item.is_rasch is True

        non_rasch = ItemParameters(item_id="q1", discrimination=1.5, guessing=0.0)
        assert non_rasch.is_rasch is False


class TestRasch1PL:
    """Tests for Rasch 1PL model."""

    @pytest.fixture
    def model(self):
        return Rasch1PL()

    @pytest.fixture
    def item(self):
        return ItemParameters(item_id="q1", difficulty=0.0)

    def test_probability_at_difficulty(self, model, item):
        """P(theta=b) should be 0.5."""
        prob = model.probability(theta=0.0, params=item)
        assert abs(prob - 0.5) < 1e-6

    def test_probability_increases_with_ability(self, model, item):
        """Higher ability should give higher probability."""
        low_ability = model.probability(theta=-1.0, params=item)
        high_ability = model.probability(theta=1.0, params=item)
        assert high_ability > low_ability

    def test_probability_bounds(self, model, item):
        """Probability should be in (0, 1)."""
        for theta in np.linspace(-3, 3, 10):
            prob = model.probability(theta, item)
            assert 0 < prob < 1

    def test_information_at_difficulty(self, model, item):
        """Information should be maximum at theta=b for 1PL."""
        info_at_b = model.information(theta=0.0, params=item)
        info_away = model.information(theta=2.0, params=item)
        assert info_at_b > info_away


class TestModel2PL:
    """Tests for 2PL model."""

    @pytest.fixture
    def model(self):
        return Model2PL()

    def test_discrimination_affects_slope(self, model):
        """Higher discrimination should give steeper ICC."""
        low_disc = ItemParameters(item_id="q1", difficulty=0.0, discrimination=0.5)
        high_disc = ItemParameters(item_id="q1", difficulty=0.0, discrimination=2.0)

        # Probability difference for same theta change
        theta_low, theta_high = -0.5, 0.5

        diff_low = (
            model.probability(theta_high, low_disc) -
            model.probability(theta_low, low_disc)
        )
        diff_high = (
            model.probability(theta_high, high_disc) -
            model.probability(theta_low, high_disc)
        )

        assert diff_high > diff_low  # Higher disc = steeper slope

    def test_reduces_to_1pl(self, model):
        """2PL with a=1 should equal 1PL."""
        rasch = Rasch1PL()
        item = ItemParameters(item_id="q1", difficulty=0.5, discrimination=1.0)

        for theta in np.linspace(-2, 2, 5):
            p_2pl = model.probability(theta, item)
            p_1pl = rasch.probability(theta, item)
            assert abs(p_2pl - p_1pl) < 1e-6


class TestModel3PL:
    """Tests for 3PL model."""

    @pytest.fixture
    def model(self):
        return Model3PL()

    def test_guessing_parameter(self, model):
        """Low ability should still have P >= c."""
        item = ItemParameters(
            item_id="q1",
            difficulty=0.0,
            discrimination=1.0,
            guessing=0.25,
        )

        # Very low ability
        prob = model.probability(theta=-10.0, params=item)
        assert prob >= 0.25  # Should be close to guessing parameter

    def test_reduces_to_2pl(self, model):
        """3PL with c=0 should equal 2PL."""
        m2pl = Model2PL()
        item = ItemParameters(item_id="q1", difficulty=0.0, discrimination=1.5, guessing=0.0)

        for theta in np.linspace(-2, 2, 5):
            p_3pl = model.probability(theta, item)
            p_2pl = m2pl.probability(theta, item)
            assert abs(p_3pl - p_2pl) < 1e-6


class TestResponsePattern:
    """Tests for ResponsePattern dataclass."""

    def test_response_statistics(self):
        """Should compute response statistics correctly."""
        pattern = ResponsePattern(
            learner_id="user1",
            responses={"q1": 1, "q2": 1, "q3": 0, "q4": 1, "q5": 0},
        )
        assert pattern.total_correct == 3
        assert pattern.total_items == 5
        assert pattern.proportion_correct == 0.6


class TestDiscriminationAnalyzer:
    """Tests for question discrimination analysis."""

    @pytest.fixture
    def analyzer(self):
        return DiscriminationAnalyzer()

    def test_discrimination_categories(self, analyzer):
        """Discrimination should map to quality categories."""
        # High discrimination = excellent question
        result = analyzer._categorize_discrimination(1.6)
        assert result == QualityCategory.EXCELLENT

        # Low discrimination = poor question
        result = analyzer._categorize_discrimination(0.3)
        assert result == QualityCategory.POOR

        # Negative = problematic
        result = analyzer._categorize_discrimination(-0.5)
        assert result == QualityCategory.PROBLEMATIC

    def test_analyze_item(self, analyzer):
        """Should analyze item discrimination from responses."""
        # High-ability learners get it right, low-ability wrong
        responses = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
        abilities = np.array([1.5, 1.2, 0.8, 0.5, -0.5, -0.8, -1.0, -1.2, 0.3, -0.2])

        result = analyzer.analyze_item(
            item_id="q1",
            responses=responses,
            abilities=abilities,
        )

        assert result.item_id == "q1"
        assert result.discrimination > 0  # Should be positive
        assert result.proportion_correct == 0.5  # 5/10 correct

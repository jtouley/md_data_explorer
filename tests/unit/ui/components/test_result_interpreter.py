"""
Test Result Interpreter - Phase 5

Tests for plain-language interpretation of statistical results with:
- Value mappings for readable labels
- CI crossing 1 warnings
- Sample size warnings
- "Associated with" wording (not causation)
"""

from clinical_analytics.ui.components.result_interpreter import ResultInterpreter


class TestInterpretOddsRatio:
    """Test odds ratio interpretation with enhancements."""

    def test_interpret_odds_ratio_uses_associated_with_wording(self):
        """Test that interpretation uses 'associated with' wording (not causation)."""
        # Arrange
        or_value = 1.5
        ci_lower = 1.2
        ci_upper = 1.8
        p_value = 0.01
        variable_name = "Treatment A"

        # Act
        interpretation = ResultInterpreter.interpret_odds_ratio(
            or_value=or_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            variable_name=variable_name,
        )

        # Assert
        assert "associated with" in interpretation.lower()
        assert "causes" not in interpretation.lower()
        assert "causation" not in interpretation.lower()

    def test_interpret_odds_ratio_ci_crosses_one_warning(self):
        """Test that CI crossing 1 generates uncertainty warning."""
        # Arrange
        or_value = 1.2
        ci_lower = 0.8  # Crosses 1
        ci_upper = 1.6
        p_value = 0.05
        variable_name = "Risk Factor"

        # Act
        interpretation = ResultInterpreter.interpret_odds_ratio(
            or_value=or_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            variable_name=variable_name,
        )

        # Assert
        assert "confidence interval crosses 1" in interpretation.lower()
        assert "uncertainty" in interpretation.lower()

    def test_interpret_odds_ratio_small_sample_size_warning(self):
        """Test that small sample size (n<30) generates warning."""
        # Arrange
        or_value = 1.5
        ci_lower = 1.2
        ci_upper = 1.8
        p_value = 0.01
        variable_name = "Treatment"
        sample_size = 25  # Small sample

        # Act
        interpretation = ResultInterpreter.interpret_odds_ratio(
            or_value=or_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            variable_name=variable_name,
            sample_size=sample_size,
        )

        # Assert
        assert "small sample size" in interpretation.lower()
        assert "n=25" in interpretation
        assert "caution" in interpretation.lower()

    def test_interpret_odds_ratio_moderate_sample_size_note(self):
        """Test that moderate sample size (30<=n<100) generates note."""
        # Arrange
        or_value = 1.5
        ci_lower = 1.2
        ci_upper = 1.8
        p_value = 0.01
        variable_name = "Treatment"
        sample_size = 75  # Moderate sample

        # Act
        interpretation = ResultInterpreter.interpret_odds_ratio(
            or_value=or_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            variable_name=variable_name,
            sample_size=sample_size,
        )

        # Assert
        assert "moderate sample size" in interpretation.lower()
        assert "n=75" in interpretation
        assert "consider larger studies" in interpretation.lower()

    def test_interpret_odds_ratio_large_sample_no_warning(self):
        """Test that large sample size (n>=100) has no warning."""
        # Arrange
        or_value = 1.5
        ci_lower = 1.2
        ci_upper = 1.8
        p_value = 0.01
        variable_name = "Treatment"
        sample_size = 200  # Large sample

        # Act
        interpretation = ResultInterpreter.interpret_odds_ratio(
            or_value=or_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            variable_name=variable_name,
            sample_size=sample_size,
        )

        # Assert
        assert "small sample size" not in interpretation.lower()
        assert "moderate sample size" not in interpretation.lower()
        assert "n=200" not in interpretation

    def test_interpret_odds_ratio_value_mapping_used(self):
        """Test that value mappings are used when provided (future enhancement)."""
        # Arrange
        or_value = 1.5
        ci_lower = 1.2
        ci_upper = 1.8
        p_value = 0.01
        variable_name = "Treatment"
        value_mapping = {"1": "Yes", "2": "No"}

        # Act
        interpretation = ResultInterpreter.interpret_odds_ratio(
            or_value=or_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            variable_name=variable_name,
            value_mapping=value_mapping,
        )

        # Assert
        # Currently value_mapping is accepted but not yet used in display
        # This test verifies the parameter is accepted without error
        assert "Treatment" in interpretation
        assert "associated with" in interpretation.lower()

    def test_interpret_odds_ratio_not_significant_with_ci_crossing(self):
        """Test non-significant result with CI crossing 1 shows both warnings."""
        # Arrange
        or_value = 1.1
        ci_lower = 0.8  # Crosses 1
        ci_upper = 1.4
        p_value = 0.15  # Not significant
        variable_name = "Risk Factor"
        sample_size = 20  # Small sample

        # Act
        interpretation = ResultInterpreter.interpret_odds_ratio(
            or_value=or_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            variable_name=variable_name,
            sample_size=sample_size,
        )

        # Assert
        assert "not significantly associated" in interpretation.lower()
        assert "confidence interval crosses 1" in interpretation.lower()
        assert "small sample size" in interpretation.lower()

    def test_interpret_odds_ratio_significant_increases(self):
        """Test significant OR > 1 uses 'associated with' wording."""
        # Arrange
        or_value = 2.5
        ci_lower = 1.8
        ci_upper = 3.2
        p_value = 0.001
        variable_name = "Treatment A"

        # Act
        interpretation = ResultInterpreter.interpret_odds_ratio(
            or_value=or_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            variable_name=variable_name,
        )

        # Assert
        assert "strongly increases" in interpretation.lower()
        assert "associated with" in interpretation.lower()
        assert "increased odds" in interpretation.lower()

    def test_interpret_odds_ratio_significant_decreases(self):
        """Test significant OR < 1 uses 'associated with' wording."""
        # Arrange
        or_value = 0.4
        ci_lower = 0.2
        ci_upper = 0.6
        p_value = 0.001
        variable_name = "Treatment B"

        # Act
        interpretation = ResultInterpreter.interpret_odds_ratio(
            or_value=or_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            variable_name=variable_name,
        )

        # Assert
        assert "strongly decreases" in interpretation.lower()
        assert "associated with" in interpretation.lower()
        assert "decreased odds" in interpretation.lower()


class TestInterpretMeanDifference:
    """Test mean difference interpretation."""

    def test_interpret_mean_difference_uses_display_names(self):
        """Test that interpretation can use display names for groups."""
        # Arrange
        mean_diff = 5.2
        ci_lower = 2.1
        ci_upper = 8.3
        p_value = 0.01
        group1 = "Treatment A"
        group2 = "Treatment B"
        outcome_name = "Blood Pressure"

        # Act
        interpretation = ResultInterpreter.interpret_mean_difference(
            mean_diff=mean_diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            group1=group1,
            group2=group2,
            outcome_name=outcome_name,
        )

        # Assert
        assert group1 in interpretation
        assert group2 in interpretation
        assert outcome_name in interpretation
        assert "significant difference" in interpretation.lower()

    def test_interpret_mean_difference_not_significant(self):
        """Test non-significant mean difference interpretation."""
        # Arrange
        mean_diff = 1.5
        ci_lower = -0.5
        ci_upper = 3.5
        p_value = 0.15
        group1 = "Group A"
        group2 = "Group B"
        outcome_name = "Outcome"

        # Act
        interpretation = ResultInterpreter.interpret_mean_difference(
            mean_diff=mean_diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            group1=group1,
            group2=group2,
            outcome_name=outcome_name,
        )

        # Assert
        assert "no significant difference" in interpretation.lower()
        assert "could easily be due to chance" in interpretation.lower()

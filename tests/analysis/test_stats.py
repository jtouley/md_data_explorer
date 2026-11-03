"""
Tests for statistical analysis module.
"""

import numpy as np
import pandas as pd
import pytest

from clinical_analytics.analysis.stats import run_logistic_regression


class TestStats:
    """Test suite for statistical analysis functions."""

    def test_run_logistic_regression(self):
        """Test running logistic regression."""
        # Create test data
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "outcome": np.random.binomial(1, 0.3, n),
                "predictor1": np.random.normal(0, 1, n),
                "predictor2": np.random.normal(0, 1, n),
            }
        )

        model, summary_df = run_logistic_regression(df, outcome_col="outcome", predictors=["predictor1", "predictor2"])

        assert model is not None
        assert isinstance(summary_df, pd.DataFrame)
        assert "Odds Ratio" in summary_df.columns
        assert "CI Lower" in summary_df.columns
        assert "CI Upper" in summary_df.columns
        assert "P-Value" in summary_df.columns

    def test_run_logistic_regression_single_predictor(self):
        """Test logistic regression with single predictor."""
        df = pd.DataFrame({"outcome": [0, 1, 0, 1, 0, 1], "age": [20, 30, 40, 50, 60, 70]})

        model, summary_df = run_logistic_regression(df, outcome_col="outcome", predictors=["age"])

        assert model is not None
        assert len(summary_df) == 2  # Intercept + age

    def test_run_logistic_regression_with_nulls(self):
        """Rows with a null outcome or predictor are dropped before fitting."""
        # Arrange: non-separable data with nulls scattered across columns
        rng = np.random.default_rng(7)
        n = 120
        df = pd.DataFrame(
            {
                "outcome": rng.binomial(1, 0.4, n).astype(float),
                "predictor1": rng.normal(0, 1, n),
                "predictor2": rng.normal(0, 1, n),
            }
        )
        df.loc[[0, 5], "outcome"] = None
        df.loc[[1, 9], "predictor1"] = None
        df.loc[[2], "predictor2"] = None

        # Act
        model, summary_df = run_logistic_regression(df, outcome_col="outcome", predictors=["predictor1", "predictor2"])

        # Assert
        assert int(model.nobs) == n - 5
        assert list(summary_df.index) == ["Intercept", "predictor1", "predictor2"]

    @pytest.mark.parametrize(
        ("frame", "message"),
        [
            (
                pd.DataFrame({"outcome": [0] * 10 + [1] * 10, "marker": list(range(10)) + list(range(20, 30))}),
                "separation",
            ),
            (
                pd.DataFrame(
                    {
                        "outcome": [0, 1] * 15,
                        "age": list(range(30, 60)),
                        "age_months": [a * 12 for a in range(30, 60)],
                    }
                ),
                "collinearity",
            ),
        ],
        ids=["perfect_separation", "perfect_collinearity"],
    )
    def test_run_logistic_regression_notEstimable_raisesWithReason(self, frame, message):
        """Separation and collinearity are refused instead of returning meaningless odds ratios."""
        # Arrange
        from clinical_analytics.analysis.stats import ModelNotEstimableError

        predictors = [c for c in frame.columns if c != "outcome"]

        # Act / Assert
        with pytest.raises(ModelNotEstimableError, match=message):
            run_logistic_regression(frame, outcome_col="outcome", predictors=predictors)

    def test_run_logistic_regression_columnNamesWithSpacesAndCodes_fitAndKeepNames(self):
        """Clinical headers with spaces and code lists are quoted in the formula."""
        # Arrange
        rng = np.random.default_rng(3)
        n = 150
        df = pd.DataFrame(
            {
                "Relapse within 1 year? Yes: 1 No: 0": rng.binomial(1, 0.35, n),
                "Systolic BP (mmHg)": rng.normal(125, 12, n),
                "CD4 Count": rng.normal(600, 150, n),
            }
        )

        # Act
        _, summary_df = run_logistic_regression(
            df,
            outcome_col="Relapse within 1 year? Yes: 1 No: 0",
            predictors=["Systolic BP (mmHg)", "CD4 Count"],
        )

        # Assert
        assert list(summary_df.index) == ["Intercept", "Systolic BP (mmHg)", "CD4 Count"]

    def test_run_logistic_regression_all_nulls(self):
        """Test logistic regression with all nulls raises error."""
        df = pd.DataFrame({"outcome": [None, None, None], "predictor1": [1, 2, 3]})

        with pytest.raises(ValueError, match="No data remaining"):
            run_logistic_regression(df, outcome_col="outcome", predictors=["predictor1"])

    def test_run_logistic_regression_empty_dataframe(self):
        """Test logistic regression with empty dataframe raises error."""
        df = pd.DataFrame({"outcome": [], "predictor1": []})

        with pytest.raises(ValueError, match="No data remaining"):
            run_logistic_regression(df, outcome_col="outcome", predictors=["predictor1"])

    def test_run_logistic_regression_results_format(self):
        """Test that results are properly formatted."""
        # Use more realistic data to avoid perfect separation
        df = pd.DataFrame(
            {
                "outcome": [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
                "sex": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            }
        )

        model, summary_df = run_logistic_regression(df, outcome_col="outcome", predictors=["age", "sex"])

        # Check all required columns exist
        assert "Odds Ratio" in summary_df.columns
        assert "CI Lower" in summary_df.columns
        assert "CI Upper" in summary_df.columns
        assert "P-Value" in summary_df.columns

        # Check values are numeric
        assert summary_df["Odds Ratio"].dtype in [np.float64, np.float32]
        assert summary_df["P-Value"].dtype in [np.float64, np.float32]

        # Check values are reasonable (exclude intercept which can be 0 or inf in edge cases)
        predictor_rows = summary_df.index != "Intercept"
        assert (summary_df.loc[predictor_rows, "Odds Ratio"] > 0).all()
        assert (summary_df["P-Value"] >= 0).all()
        assert (summary_df["P-Value"] <= 1).all()

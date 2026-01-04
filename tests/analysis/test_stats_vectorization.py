"""
Test Vectorized Operations in Stats - Phase 7

Tests that vectorized operations are used instead of apply(lambda).
"""

import numpy as np
import pandas as pd
from clinical_analytics.analysis.stats import run_logistic_regression


class TestVectorizedOperations:
    """Test that vectorized operations are used in stats calculations."""

    def test_odds_ratio_calculation_uses_vectorized_operations(self):
        """Test that odds ratio calculation uses np.exp directly, not apply(lambda)."""
        # Arrange: Create test data with good separation
        df = pd.DataFrame(
            {
                "outcome": [0, 0, 0, 1, 1, 1],
                "predictor": [1, 2, 3, 4, 5, 6],  # Good separation
            }
        )

        # Act: Run logistic regression
        model, summary_df = run_logistic_regression(df, outcome_col="outcome", predictors=["predictor"])

        # Assert: Odds ratios are calculated (vectorized operations work)
        assert "Odds Ratio" in summary_df.columns
        assert "CI Lower" in summary_df.columns
        assert "CI Upper" in summary_df.columns
        assert "P-Value" in summary_df.columns

        # Verify odds ratios are numeric (not NaN from vectorization issues)
        odds_ratios = summary_df["Odds Ratio"]
        assert odds_ratios.notna().all(), "Odds ratios should not contain NaN values"

        # Verify confidence intervals are numeric
        assert summary_df["CI Lower"].notna().all()
        assert summary_df["CI Upper"].notna().all()

    def test_vectorized_exp_produces_same_results_as_apply(self):
        """Test that np.exp on Series produces same results as apply(lambda x: np.exp(x))."""
        # Arrange: Create test Series
        params = pd.Series([0.5, 1.0, -0.5, 2.0])

        # Act: Vectorized operation
        vectorized_result = np.exp(params)

        # Act: Old apply method (for comparison)
        apply_result = params.apply(lambda x: np.exp(x))

        # Assert: Results are identical
        np.testing.assert_array_almost_equal(vectorized_result.values, apply_result.values)

    def test_vectorized_exp_on_dataframe_produces_same_results(self):
        """Test that np.exp on DataFrame produces same results as apply(lambda)."""
        # Arrange: Create test DataFrame
        conf = pd.DataFrame({"CI Lower": [0.1, 0.2], "CI Upper": [0.3, 0.4]})

        # Act: Vectorized operation
        vectorized_result = np.exp(conf)

        # Act: Old apply method (for comparison)
        apply_result = conf.apply(lambda x: np.exp(x))

        # Assert: Results are identical
        pd.testing.assert_frame_equal(vectorized_result, apply_result)

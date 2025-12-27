"""
Tests for statistical analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from clinical_analytics.analysis.stats import run_logistic_regression


class TestStats:
    """Test suite for statistical analysis functions."""

    def test_run_logistic_regression(self):
        """Test running logistic regression."""
        # Create test data
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'outcome': np.random.binomial(1, 0.3, n),
            'predictor1': np.random.normal(0, 1, n),
            'predictor2': np.random.normal(0, 1, n)
        })

        model, summary_df = run_logistic_regression(
            df,
            outcome_col='outcome',
            predictors=['predictor1', 'predictor2']
        )

        assert model is not None
        assert isinstance(summary_df, pd.DataFrame)
        assert 'Odds Ratio' in summary_df.columns
        assert 'CI Lower' in summary_df.columns
        assert 'CI Upper' in summary_df.columns
        assert 'P-Value' in summary_df.columns

    def test_run_logistic_regression_single_predictor(self):
        """Test logistic regression with single predictor."""
        df = pd.DataFrame({
            'outcome': [0, 1, 0, 1, 0, 1],
            'age': [20, 30, 40, 50, 60, 70]
        })

        model, summary_df = run_logistic_regression(
            df,
            outcome_col='outcome',
            predictors=['age']
        )

        assert model is not None
        assert len(summary_df) == 2  # Intercept + age

    def test_run_logistic_regression_with_nulls(self):
        """Test logistic regression handles nulls correctly."""
        df = pd.DataFrame({
            'outcome': [0, 1, 0, 1, None, 1],
            'predictor1': [1, 2, None, 4, 5, 6],
            'predictor2': [10, 20, 30, None, 50, 60]
        })

        # Should drop rows with nulls
        model, summary_df = run_logistic_regression(
            df,
            outcome_col='outcome',
            predictors=['predictor1', 'predictor2']
        )

        assert model is not None
        assert isinstance(summary_df, pd.DataFrame)

    def test_run_logistic_regression_all_nulls(self):
        """Test logistic regression with all nulls raises error."""
        df = pd.DataFrame({
            'outcome': [None, None, None],
            'predictor1': [1, 2, 3]
        })

        with pytest.raises(ValueError, match="No data remaining"):
            run_logistic_regression(
                df,
                outcome_col='outcome',
                predictors=['predictor1']
            )

    def test_run_logistic_regression_empty_dataframe(self):
        """Test logistic regression with empty dataframe raises error."""
        df = pd.DataFrame({
            'outcome': [],
            'predictor1': []
        })

        with pytest.raises(ValueError, match="No data remaining"):
            run_logistic_regression(
                df,
                outcome_col='outcome',
                predictors=['predictor1']
            )

    def test_run_logistic_regression_results_format(self):
        """Test that results are properly formatted."""
        df = pd.DataFrame({
            'outcome': [0, 1, 0, 1, 0, 1, 0, 1],
            'age': [20, 30, 40, 50, 60, 70, 80, 90],
            'sex': [0, 1, 0, 1, 0, 1, 0, 1]
        })

        model, summary_df = run_logistic_regression(
            df,
            outcome_col='outcome',
            predictors=['age', 'sex']
        )

        # Check all required columns exist
        assert 'Odds Ratio' in summary_df.columns
        assert 'CI Lower' in summary_df.columns
        assert 'CI Upper' in summary_df.columns
        assert 'P-Value' in summary_df.columns

        # Check values are numeric
        assert summary_df['Odds Ratio'].dtype in [np.float64, np.float32]
        assert summary_df['P-Value'].dtype in [np.float64, np.float32]

        # Check values are reasonable
        assert (summary_df['Odds Ratio'] > 0).all()
        assert (summary_df['P-Value'] >= 0).all()
        assert (summary_df['P-Value'] <= 1).all()


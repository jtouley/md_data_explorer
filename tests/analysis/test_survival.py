"""
Tests for survival analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from clinical_analytics.analysis.survival import (
    run_kaplan_meier,
    run_cox_regression,
    run_logrank_test,
    calculate_median_survival
)


class TestSurvivalAnalysis:
    """Test suite for survival analysis functions."""

    def test_run_kaplan_meier_single_cohort(self):
        """Test Kaplan-Meier analysis for single cohort."""
        # Create test survival data
        df = pd.DataFrame({
            'duration': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'event': [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
        })

        kmf, summary = run_kaplan_meier(
            df,
            duration_col='duration',
            event_col='event'
        )

        assert kmf is not None
        assert isinstance(summary, pd.DataFrame)
        assert 'time' in summary.columns
        assert 'survival_probability' in summary.columns
        assert 'ci_lower' in summary.columns
        assert 'ci_upper' in summary.columns

    def test_run_kaplan_meier_stratified(self):
        """Test Kaplan-Meier analysis with stratification."""
        df = pd.DataFrame({
            'duration': [10, 20, 30, 40, 50, 60, 70, 80],
            'event': [1, 1, 0, 1, 0, 1, 1, 0],
            'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        })

        kmf, summary = run_kaplan_meier(
            df,
            duration_col='duration',
            event_col='event',
            group_col='group'
        )

        assert kmf is not None
        assert isinstance(summary, pd.DataFrame)
        assert 'group' in summary.columns
        assert len(summary['group'].unique()) == 2

    def test_run_kaplan_meier_with_nulls(self):
        """Test Kaplan-Meier handles nulls correctly."""
        df = pd.DataFrame({
            'duration': [10, 20, None, 40, 50],
            'event': [1, None, 0, 1, 1]
        })

        kmf, summary = run_kaplan_meier(
            df,
            duration_col='duration',
            event_col='event'
        )

        assert kmf is not None
        assert isinstance(summary, pd.DataFrame)

    def test_run_kaplan_meier_all_nulls(self):
        """Test Kaplan-Meier with all nulls raises error."""
        df = pd.DataFrame({
            'duration': [None, None, None],
            'event': [None, None, None]
        })

        with pytest.raises(ValueError, match="No data remaining"):
            run_kaplan_meier(
                df,
                duration_col='duration',
                event_col='event'
            )

    def test_run_cox_regression(self):
        """Test Cox regression analysis."""
        df = pd.DataFrame({
            'duration': [10, 20, 30, 40, 50, 60, 70, 80],
            'event': [1, 1, 0, 1, 0, 1, 1, 0],
            'age': [45, 50, 55, 60, 65, 70, 75, 80],
            'treatment': [0, 1, 0, 1, 0, 1, 0, 1]
        })

        cph, summary_df = run_cox_regression(
            df,
            duration_col='duration',
            event_col='event',
            covariates=['age', 'treatment']
        )

        assert cph is not None
        assert isinstance(summary_df, pd.DataFrame)
        assert 'hazard_ratio' in summary_df.columns
        assert 'hr_ci_lower' in summary_df.columns
        assert 'hr_ci_upper' in summary_df.columns
        assert 'p' in summary_df.columns

    def test_run_cox_regression_categorical(self):
        """Test Cox regression with categorical variables."""
        df = pd.DataFrame({
            'duration': [10, 20, 30, 40, 50],
            'event': [1, 1, 0, 1, 0],
            'treatment': ['A', 'B', 'A', 'B', 'A']
        })

        cph, summary_df = run_cox_regression(
            df,
            duration_col='duration',
            event_col='event',
            covariates=['treatment']
        )

        assert cph is not None
        assert isinstance(summary_df, pd.DataFrame)

    def test_run_cox_regression_with_nulls(self):
        """Test Cox regression handles nulls correctly."""
        df = pd.DataFrame({
            'duration': [10, 20, None, 40, 50],
            'event': [1, None, 0, 1, 1],
            'age': [45, 50, 55, None, 60]
        })

        cph, summary_df = run_cox_regression(
            df,
            duration_col='duration',
            event_col='event',
            covariates=['age']
        )

        assert cph is not None
        assert isinstance(summary_df, pd.DataFrame)

    def test_run_cox_regression_all_nulls(self):
        """Test Cox regression with all nulls raises error."""
        df = pd.DataFrame({
            'duration': [None, None, None],
            'event': [None, None, None],
            'age': [None, None, None]
        })

        with pytest.raises(ValueError, match="No data remaining"):
            run_cox_regression(
                df,
                duration_col='duration',
                event_col='event',
                covariates=['age']
            )

    def test_run_logrank_test_two_groups(self):
        """Test log-rank test with two groups."""
        df = pd.DataFrame({
            'duration': [10, 20, 30, 40, 50, 60, 70, 80],
            'event': [1, 1, 0, 1, 0, 1, 1, 0],
            'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        })

        result = run_logrank_test(
            df,
            duration_col='duration',
            event_col='event',
            group_col='group'
        )

        assert isinstance(result, dict)
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'groups' in result
        assert result['n_groups'] == 2
        assert result['test_type'] == 'two_group_logrank'

    def test_run_logrank_test_multiple_groups(self):
        """Test log-rank test with multiple groups."""
        df = pd.DataFrame({
            'duration': [10, 20, 30, 40, 50, 60, 70, 80, 90],
            'event': [1, 1, 0, 1, 0, 1, 1, 0, 1],
            'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        })

        result = run_logrank_test(
            df,
            duration_col='duration',
            event_col='event',
            group_col='group'
        )

        assert isinstance(result, dict)
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert result['n_groups'] == 3
        assert result['test_type'] == 'multivariate_logrank'
        assert 'degrees_of_freedom' in result

    def test_run_logrank_test_single_group(self):
        """Test log-rank test with single group raises error."""
        df = pd.DataFrame({
            'duration': [10, 20, 30],
            'event': [1, 1, 0],
            'group': ['A', 'A', 'A']
        })

        with pytest.raises(ValueError, match="Need at least 2 groups"):
            run_logrank_test(
                df,
                duration_col='duration',
                event_col='event',
                group_col='group'
            )

    def test_calculate_median_survival_single_cohort(self):
        """Test calculating median survival for single cohort."""
        df = pd.DataFrame({
            'duration': [10, 20, 30, 40, 50, 60, 70, 80],
            'event': [1, 1, 0, 1, 0, 1, 1, 0]
        })

        result = calculate_median_survival(
            df,
            duration_col='duration',
            event_col='event'
        )

        assert isinstance(result, pd.DataFrame)
        assert 'group' in result.columns
        assert 'median_survival' in result.columns
        assert 'n' in result.columns
        assert 'n_events' in result.columns

    def test_calculate_median_survival_stratified(self):
        """Test calculating median survival with stratification."""
        df = pd.DataFrame({
            'duration': [10, 20, 30, 40, 50, 60],
            'event': [1, 1, 0, 1, 0, 1],
            'group': ['A', 'A', 'A', 'B', 'B', 'B']
        })

        result = calculate_median_survival(
            df,
            duration_col='duration',
            event_col='event',
            group_col='group'
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two groups
        assert 'group' in result.columns
        assert 'median_survival' in result.columns


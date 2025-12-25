"""
Survival Analysis Module - Time-to-event analysis for clinical data.

Provides Kaplan-Meier survival curves and Cox proportional hazards regression.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


def run_kaplan_meier(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: Optional[str] = None,
    alpha: float = 0.05
) -> Tuple[KaplanMeierFitter, pd.DataFrame]:
    """
    Perform Kaplan-Meier survival analysis.

    Args:
        df: DataFrame with survival data
        duration_col: Column name for time-to-event or censoring
        event_col: Column name for event indicator (1=event, 0=censored)
        group_col: Optional column for stratification
        alpha: Significance level for confidence intervals

    Returns:
        Tuple of (fitted model, summary DataFrame)
    """
    # Clean data
    data = df[[duration_col, event_col]].copy()
    if group_col:
        data[group_col] = df[group_col]

    data = data.dropna()

    if len(data) == 0:
        raise ValueError("No data remaining after dropping nulls")

    # Fit Kaplan-Meier
    kmf = KaplanMeierFitter(alpha=alpha)

    if group_col is None:
        # Single cohort
        kmf.fit(
            durations=data[duration_col],
            event_observed=data[event_col],
            label="Overall"
        )

        summary = pd.DataFrame({
            'time': kmf.survival_function_.index,
            'survival_probability': kmf.survival_function_['Overall'],
            'ci_lower': kmf.confidence_interval_['Overall_lower_0.95'],
            'ci_upper': kmf.confidence_interval_['Overall_upper_0.95']
        })

    else:
        # Stratified by group
        summaries = []

        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]

            kmf.fit(
                durations=group_data[duration_col],
                event_observed=group_data[event_col],
                label=str(group)
            )

            group_summary = pd.DataFrame({
                'group': group,
                'time': kmf.survival_function_.index,
                'survival_probability': kmf.survival_function_[str(group)],
                'ci_lower': kmf.confidence_interval_[f'{group}_lower_0.95'],
                'ci_upper': kmf.confidence_interval_[f'{group}_upper_0.95']
            })

            summaries.append(group_summary)

        summary = pd.concat(summaries, ignore_index=True)

    return kmf, summary


def run_cox_regression(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: List[str],
    alpha: float = 0.05
) -> Tuple[CoxPHFitter, pd.DataFrame]:
    """
    Perform Cox proportional hazards regression.

    Args:
        df: DataFrame with survival data
        duration_col: Column name for time-to-event or censoring
        event_col: Column name for event indicator (1=event, 0=censored)
        covariates: List of covariate column names
        alpha: Significance level

    Returns:
        Tuple of (fitted model, summary DataFrame with hazard ratios)
    """
    # Prepare data
    model_cols = [duration_col, event_col] + covariates
    data = df[model_cols].copy()

    # Handle categorical variables - convert to dummies
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c in covariates]

    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Drop missing values
    data = data.dropna()

    if len(data) == 0:
        raise ValueError("No data remaining after cleaning")

    # Fit Cox model
    cph = CoxPHFitter(alpha=alpha)
    cph.fit(data, duration_col=duration_col, event_col=event_col)

    # Extract summary with hazard ratios
    summary_df = cph.summary

    # Add hazard ratios explicitly
    summary_df['hazard_ratio'] = np.exp(summary_df['coef'])
    summary_df['hr_ci_lower'] = np.exp(summary_df['coef lower 95%'])
    summary_df['hr_ci_upper'] = np.exp(summary_df['coef upper 95%'])

    # Reorder columns for clarity
    summary_df = summary_df[[
        'hazard_ratio',
        'hr_ci_lower',
        'hr_ci_upper',
        'p',
        'coef',
        'se(coef)',
        'z'
    ]]

    summary_df = summary_df.round(4)

    return cph, summary_df


def run_logrank_test(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str
) -> dict:
    """
    Perform log-rank test to compare survival curves between groups.

    Args:
        df: DataFrame with survival data
        duration_col: Column name for time-to-event
        event_col: Column name for event indicator
        group_col: Column name for group variable

    Returns:
        Dictionary with test results
    """
    data = df[[duration_col, event_col, group_col]].dropna()

    if len(data) == 0:
        raise ValueError("No data after cleaning")

    groups = data[group_col].unique()

    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for log-rank test")

    elif len(groups) == 2:
        # Two-group log-rank test
        group_a = data[data[group_col] == groups[0]]
        group_b = data[data[group_col] == groups[1]]

        result = logrank_test(
            durations_A=group_a[duration_col],
            durations_B=group_b[duration_col],
            event_observed_A=group_a[event_col],
            event_observed_B=group_b[event_col]
        )

        return {
            'test_statistic': float(result.test_statistic),
            'p_value': float(result.p_value),
            'groups': [str(g) for g in groups],
            'n_groups': 2,
            'test_type': 'two_group_logrank'
        }

    else:
        # Multi-group log-rank test
        result = multivariate_logrank_test(
            event_durations=data[duration_col],
            groups=data[group_col],
            event_observed=data[event_col]
        )

        return {
            'test_statistic': float(result.test_statistic),
            'p_value': float(result.p_value),
            'groups': [str(g) for g in groups],
            'n_groups': len(groups),
            'test_type': 'multivariate_logrank',
            'degrees_of_freedom': result.degrees_of_freedom
        }


def calculate_median_survival(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate median survival time with confidence intervals.

    Args:
        df: DataFrame with survival data
        duration_col: Column name for time-to-event
        event_col: Column name for event indicator
        group_col: Optional group column for stratification

    Returns:
        DataFrame with median survival times and CIs
    """
    kmf = KaplanMeierFitter()
    results = []

    if group_col is None:
        data = df[[duration_col, event_col]].dropna()

        kmf.fit(durations=data[duration_col], event_observed=data[event_col])

        results.append({
            'group': 'Overall',
            'median_survival': kmf.median_survival_time_,
            'ci_lower': kmf.confidence_interval_survival_function_.iloc[0, 0]
            if not kmf.confidence_interval_survival_function_.empty else None,
            'ci_upper': kmf.confidence_interval_survival_function_.iloc[0, 1]
            if not kmf.confidence_interval_survival_function_.empty else None,
            'n': len(data),
            'n_events': int(data[event_col].sum())
        })

    else:
        data = df[[duration_col, event_col, group_col]].dropna()

        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]

            kmf.fit(
                durations=group_data[duration_col],
                event_observed=group_data[event_col]
            )

            results.append({
                'group': str(group),
                'median_survival': kmf.median_survival_time_,
                'n': len(group_data),
                'n_events': int(group_data[event_col].sum())
            })

    return pd.DataFrame(results)

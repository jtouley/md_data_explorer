"""
Pure compute functions for analysis - Polars-first, return serializable dicts.

These functions have no UI dependencies and can be tested independently.
"""

from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from clinical_analytics.ui.components.question_engine import AnalysisContext


def _try_convert_to_numeric(series: pl.Series) -> pl.Series | None:
    """
    Try to convert a string series to numeric.

    Handles:
    - European comma format: "-1,8" -> -1.8
    - Empty strings and whitespace -> null
    - Non-numeric values -> null

    Returns None if conversion fails or results in all nulls.
    """
    if series.dtype != pl.Utf8:
        return None

    try:
        # Clean the values: replace comma with dot, strip whitespace
        cleaned = (
            series.str.strip_chars().str.replace(",", ".").str.replace_all(r"^\s*$", "")  # Empty strings to empty
        )

        # Try to cast to float
        numeric = cleaned.cast(pl.Float64, strict=False)

        # Check if we got any valid values
        if numeric.null_count() == len(numeric):
            return None

        return numeric
    except Exception:
        return None


def compute_descriptive_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Compute descriptive analysis using Polars-native operations.

    If context.primary_variable is set to a specific column, focus on that column.
    If set to "all" or None, describe all columns.

    Returns serializable dict (no Polars objects).
    """
    # Check if we're focusing on a specific variable
    primary_var = context.primary_variable
    focus_on_single = primary_var and primary_var != "all" and primary_var in df.columns

    if focus_on_single:
        # Focused analysis on a single variable
        col = primary_var
        series = df[col]
        row_count = df.height

        # Try to convert string columns to numeric (handles European comma format, etc.)
        numeric_series = None
        if series.dtype == pl.Utf8:
            numeric_series = _try_convert_to_numeric(series)

        # Use converted series if successful, otherwise original
        analysis_series = numeric_series if numeric_series is not None else series
        is_numeric = analysis_series.dtype in (pl.Int64, pl.Float64)

        non_null_count = row_count - analysis_series.null_count()

        result = {
            "type": "descriptive",
            "focused_variable": col,
            "row_count": row_count,
            "non_null_count": non_null_count,
            "null_count": analysis_series.null_count(),
            "null_pct": (analysis_series.null_count() / row_count * 100) if row_count > 0 else 0.0,
        }

        # Compute stats based on dtype
        if is_numeric:
            # Numeric variable - compute mean, median, std, min, max
            clean_series = analysis_series.drop_nulls()
            if len(clean_series) > 0:
                result["mean"] = float(clean_series.mean())
                result["median"] = float(clean_series.median())
                result["std"] = float(clean_series.std()) if len(clean_series) > 1 else 0.0
                result["min"] = float(clean_series.min())
                result["max"] = float(clean_series.max())
                result["is_numeric"] = True

                # Add headline answer for the query
                result["headline"] = f"The average {col} is **{result['mean']:.2f}**"
            else:
                result["headline"] = f"No valid data for {col}"
                result["is_numeric"] = True
        else:
            # Categorical variable - show value counts
            value_counts = series.value_counts().sort("count", descending=True).head(10)
            result["value_counts"] = value_counts.to_dicts()
            result["is_numeric"] = False
            result["unique_count"] = series.n_unique()

            # Headline for categorical
            top_value = value_counts[0] if len(value_counts) > 0 else None
            if top_value:
                top_val_name = top_value[col]
                top_val_count = top_value["count"]
                result["headline"] = f"Most common {col}: **{top_val_name}** ({top_val_count} occurrences)"
            else:
                result["headline"] = f"No data for {col}"

        return result

    # Full dataset analysis (original behavior for "all")
    row_count = df.height
    col_count = df.width
    total_cells = row_count * col_count
    null_count = df.null_count().sum_horizontal().item()
    missing_pct = (null_count / total_cells * 100) if total_cells > 0 else 0.0

    # Summary statistics for numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in (pl.Int64, pl.Float64)]
    desc_stats_dict = []
    if numeric_cols:
        desc_stats = df.select(numeric_cols).describe()
        desc_stats_dict = desc_stats.to_dicts()

    # Categorical summary (first 10 columns)
    categorical_cols = [col for col in df.columns if df[col].dtype == pl.Utf8][:10]
    categorical_summary = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts().sort("count", descending=True).head(5)
        categorical_summary[col] = value_counts.to_dicts()

    return {
        "type": "descriptive",
        "row_count": row_count,
        "column_count": col_count,
        "missing_pct": missing_pct,
        "summary_stats": desc_stats_dict,
        "categorical_summary": categorical_summary,
    }


def _compute_headline_answer(
    group_means: dict[str, float], outcome_col: str, group_col: str, query_direction: str = "lowest"
) -> dict[str, Any]:
    """
    Compute headline answer for comparison results.

    Args:
        group_means: Dict mapping group name to mean value
        outcome_col: Name of outcome variable
        group_col: Name of grouping variable
        query_direction: "lowest" or "highest"

    Returns:
        Dict with headline_group, headline_value, headline_text
    """
    if not group_means:
        return {}

    if query_direction == "lowest":
        best_group = min(group_means, key=group_means.get)
    else:
        best_group = max(group_means, key=group_means.get)

    best_value = group_means[best_group]

    # Format the headline text
    headline_text = f"**{best_group}** had the {query_direction} {outcome_col} (mean: {best_value:.1f})"

    return {
        "headline_group": str(best_group),
        "headline_value": best_value,
        "headline_text": headline_text,
        "headline_direction": query_direction,
    }


def _try_numeric_conversion(series: pl.Series) -> tuple[pl.Series | None, bool]:
    """
    Try to convert a string series to numeric, handling common patterns.

    Returns (converted_series, success).
    """
    if series.dtype in (pl.Int64, pl.Float64):
        return series, True

    if series.dtype != pl.Utf8:
        return None, False

    try:
        # Try direct cast first
        numeric = series.cast(pl.Float64)
        if numeric.null_count() < series.len() * 0.5:  # Less than 50% nulls
            return numeric, True
    except Exception:
        pass

    try:
        # Try extracting numeric part (handles "<20" -> 20, ">100" -> 100)
        numeric = series.str.extract(r"(\d+\.?\d*)").cast(pl.Float64)
        if numeric.null_count() < series.len() * 0.5:
            return numeric, True
    except Exception:
        pass

    return None, False


def compute_comparison_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Compute comparison analysis using Polars-native operations.

    Returns serializable dict (no Polars objects) including headline_answer.
    """
    outcome_col = context.primary_variable
    group_col = context.grouping_variable

    # Clean data
    analysis_df = df.select([outcome_col, group_col]).drop_nulls()

    if analysis_df.height < 2:
        return {"type": "comparison", "error": "Not enough data for comparison"}

    # Determine appropriate test
    outcome_dtype = analysis_df[outcome_col].dtype
    outcome_numeric = outcome_dtype in (pl.Int64, pl.Float64)

    groups = analysis_df[group_col].unique().to_list()
    n_groups = len(groups)

    if n_groups < 2:
        return {"type": "comparison", "error": "Need at least 2 groups for comparison"}

    # Run appropriate test
    if outcome_numeric:
        if n_groups == 2:
            # T-test
            group1_data = analysis_df.filter(pl.col(group_col) == groups[0])[outcome_col].to_numpy()
            group2_data = analysis_df.filter(pl.col(group_col) == groups[1])[outcome_col].to_numpy()

            statistic, p_value = stats.ttest_ind(group1_data, group2_data)

            # Compute means and std for interpretation
            mean1 = float(np.mean(group1_data))
            mean2 = float(np.mean(group2_data))
            std1 = float(np.std(group1_data, ddof=1))
            std2 = float(np.std(group2_data, ddof=1))
            n1 = len(group1_data)
            n2 = len(group2_data)

            mean_diff = mean1 - mean2
            se_diff = np.sqrt((std1**2 / n1) + (std2**2 / n2))
            ci_lower = mean_diff - 1.96 * se_diff
            ci_upper = mean_diff + 1.96 * se_diff

            # Compute headline answer
            group_means = {str(groups[0]): mean1, str(groups[1]): mean2}
            headline = _compute_headline_answer(group_means, outcome_col, group_col, "lowest")

            return {
                "type": "comparison",
                "test_type": "t_test",
                "outcome_col": outcome_col,
                "group_col": group_col,
                "groups": groups,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "group1_mean": mean1,
                "group2_mean": mean2,
                "mean_diff": float(mean_diff),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "group_means": group_means,
                **headline,
            }

        else:
            # ANOVA
            group_data = [analysis_df.filter(pl.col(group_col) == g)[outcome_col].to_numpy() for g in groups]
            statistic, p_value = stats.f_oneway(*group_data)

            # Group means
            group_means = {
                str(g): float(analysis_df.filter(pl.col(group_col) == g)[outcome_col].mean()) for g in groups
            }

            # Compute headline answer
            headline = _compute_headline_answer(group_means, outcome_col, group_col, "lowest")

            return {
                "type": "comparison",
                "test_type": "anova",
                "outcome_col": outcome_col,
                "group_col": group_col,
                "groups": groups,
                "n_groups": n_groups,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "group_means": group_means,
                **headline,
            }

    else:
        # Chi-square test for categorical outcome
        # Build contingency table using Polars
        contingency = (
            analysis_df.group_by([outcome_col, group_col])
            .agg(pl.len().alias("count"))
            .pivot(index=outcome_col, columns=group_col, values="count", aggregate_function="sum")
            .fill_null(0)
        )

        # Convert to numpy for scipy
        contingency_np = contingency.select([c for c in contingency.columns if c != outcome_col]).to_numpy()

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_np)

        # Store contingency as dict for serialization
        contingency_dict = contingency.to_dicts()

        # Try to compute group means if outcome can be converted to numeric
        # This allows answering "which group had the lowest X" for pseudo-numeric data
        headline: dict[str, Any] = {}
        group_means: dict[str, float] = {}
        numeric_outcome, success = _try_numeric_conversion(analysis_df[outcome_col])

        if success and numeric_outcome is not None:
            # Compute mean per group using the numeric conversion
            analysis_with_numeric = analysis_df.with_columns(numeric_outcome.alias("_numeric_outcome"))

            for g in groups:
                group_mean = (
                    analysis_with_numeric.filter(pl.col(group_col) == g).select("_numeric_outcome").mean().item()
                )
                if group_mean is not None:
                    group_means[str(g)] = float(group_mean)

            if group_means:
                headline = _compute_headline_answer(group_means, outcome_col, group_col, "lowest")

        return {
            "type": "comparison",
            "test_type": "chi_square",
            "outcome_col": outcome_col,
            "group_col": group_col,
            "chi2": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof),
            "contingency": contingency_dict,
            "contingency_index_col": outcome_col,
            "group_means": group_means if group_means else None,
            **headline,
        }


def compute_analysis_by_type(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Route to appropriate compute function based on intent.

    Args:
        df: Polars DataFrame (cohort data)
        context: AnalysisContext with intent and variables

    Returns:
        Serializable dict with analysis results
    """
    from clinical_analytics.ui.components.question_engine import AnalysisIntent

    if context.inferred_intent == AnalysisIntent.DESCRIBE:
        return compute_descriptive_analysis(df, context)
    elif context.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
        return compute_comparison_analysis(df, context)
    elif context.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
        return compute_predictor_analysis(df, context)
    elif context.inferred_intent == AnalysisIntent.EXAMINE_SURVIVAL:
        return compute_survival_analysis(df, context)
    elif context.inferred_intent == AnalysisIntent.EXPLORE_RELATIONSHIPS:
        return compute_relationship_analysis(df, context)
    else:
        return {"type": "unknown", "error": f"Unknown intent: {context.inferred_intent}"}


def compute_predictor_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Compute predictor analysis (logistic regression).

    Returns serializable dict (no Polars objects).
    """
    import pandas as pd  # Only for legacy run_logistic_regression

    outcome_col = context.primary_variable
    predictors = context.predictor_variables

    # Prepare data
    analysis_cols = [outcome_col] + predictors
    analysis_df = df.select(analysis_cols).drop_nulls()

    if analysis_df.height < 10:
        return {"type": "predictor", "error": "Need at least 10 complete observations"}

    # Check outcome type
    outcome_data = analysis_df[outcome_col].drop_nulls()
    n_unique = outcome_data.n_unique()

    if n_unique != 2:
        return {
            "type": "predictor",
            "error": f"Outcome has {n_unique} unique values. Only binary outcomes are supported.",
        }

    # Convert to pandas for legacy function (PANDAS EXCEPTION)
    # TODO: Refactor run_logistic_regression to use Polars
    analysis_df_pd = analysis_df.to_pandas()

    # Handle categorical predictors
    categorical_cols = analysis_df_pd.select_dtypes(include=["object", "category"]).columns.tolist()
    if outcome_col in categorical_cols:
        categorical_cols.remove(outcome_col)

    if categorical_cols:
        analysis_df_pd = pd.get_dummies(analysis_df_pd, columns=categorical_cols, drop_first=True)
        new_predictors = [c for c in analysis_df_pd.columns if c != outcome_col]
    else:
        new_predictors = predictors

    # Run logistic regression (legacy function)
    from clinical_analytics.analysis.stats import run_logistic_regression

    model, summary_df = run_logistic_regression(analysis_df_pd, outcome_col, new_predictors)

    # Convert summary to serializable format
    summary_dict = summary_df.to_dict(orient="index")

    # Extract significant predictors
    significant = summary_df[summary_df["P-Value"] < 0.05]
    significant_predictors = [
        {
            "variable": var,
            "odds_ratio": float(summary_df.loc[var, "Odds Ratio"]),
            "p_value": float(summary_df.loc[var, "P-Value"]),
            "ci_lower": float(summary_df.loc[var, "CI Lower"]),
            "ci_upper": float(summary_df.loc[var, "CI Upper"]),
        }
        for var in significant.index
        if var != "Intercept"
    ]

    return {
        "type": "predictor",
        "outcome_col": outcome_col,
        "summary": summary_dict,
        "significant_predictors": significant_predictors,
        "schema": {col: str(dtype) for col, dtype in summary_df.dtypes.items()},
    }


def compute_survival_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Compute survival analysis.

    Returns serializable dict (no Polars objects).
    """

    time_col = context.time_variable
    event_col = context.event_variable

    # Prepare data
    analysis_df = df.select([time_col, event_col]).drop_nulls()

    if analysis_df.height < 10:
        return {"type": "survival", "error": "Need at least 10 complete observations"}

    # Get unique event values
    unique_vals = sorted(analysis_df[event_col].unique().to_list())

    if len(unique_vals) != 2:
        return {
            "type": "survival",
            "error": f"Event variable must be binary (has {len(unique_vals)} values)",
            "unique_values": unique_vals,
        }

    # Convert to pandas for legacy function (PANDAS EXCEPTION)
    # TODO: Refactor run_kaplan_meier to use Polars
    analysis_df_pd = analysis_df.to_pandas()

    # Run Kaplan-Meier (legacy function)
    from clinical_analytics.analysis.survival import run_kaplan_meier

    kmf, summary_df = run_kaplan_meier(
        analysis_df_pd,
        duration_col=time_col,
        event_col=event_col,
        group_col=context.grouping_variable,
    )

    # Convert summary to serializable format
    summary_dict = summary_df.to_dict(orient="records")

    # Median survival
    median_survival = float(kmf.median_survival_time_) if not np.isnan(kmf.median_survival_time_) else None

    return {
        "type": "survival",
        "time_col": time_col,
        "event_col": event_col,
        "summary": summary_dict,
        "median_survival": median_survival,
        "unique_values": unique_vals,
    }


def compute_relationship_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Compute relationship analysis (correlations).

    Returns serializable dict (no Polars objects).
    """
    variables = context.predictor_variables

    if len(variables) < 2:
        return {"type": "relationship", "error": "Need at least 2 variables to examine relationships"}

    # Calculate correlations
    analysis_df = df.select(variables).drop_nulls()

    if analysis_df.height < 3:
        return {"type": "relationship", "error": "Need at least 3 observations"}

    # Cast to numeric for correlation
    numeric_df = analysis_df.select([pl.col(v).cast(pl.Float64) for v in variables])

    # Compute correlation matrix using Polars
    corr_data = []
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i <= j:
                if i == j:
                    corr_val = 1.0
                else:
                    # Compute correlation using Polars
                    corr_result = numeric_df.select(pl.corr(var1, var2))
                    corr_val = float(corr_result.item()) if corr_result.height > 0 else 0.0
                corr_data.append({"var1": var1, "var2": var2, "correlation": corr_val})

    # Find strong correlations
    strong_correlations = [
        item for item in corr_data if item["var1"] != item["var2"] and abs(item["correlation"]) >= 0.5
    ]

    return {
        "type": "relationship",
        "variables": variables,
        "correlations": corr_data,
        "strong_correlations": strong_correlations,
    }

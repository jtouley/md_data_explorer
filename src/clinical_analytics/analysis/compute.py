"""
Pure compute functions for analysis - Polars-first, return serializable dicts.

These functions have no UI dependencies and can be tested independently.
"""

from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from clinical_analytics.core.query_plan import FilterSpec
from clinical_analytics.ui.components.question_engine import AnalysisContext


def _normalize_column_name(name: str) -> str:
    """
    Normalize column name for matching: collapse whitespace, lowercase, strip.

    This handles cases where column names have inconsistent whitespace:
    - "DEXA Score          (T score)" -> "dexa score (t score)"
    - "Age " -> "age"
    - "CD4  Count" -> "cd4 count"

    Args:
        name: Original column name

    Returns:
        Normalized name for comparison
    """
    import re

    # Collapse multiple spaces to single space, strip, lowercase
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    return normalized


def _find_matching_column(target_name: str, available_columns: list[str]) -> str | None:
    """
    Find matching column using robust matching strategy.

    Strategy (in order):
    1. Exact match (case-sensitive)
    2. Exact match after normalization (whitespace/case-insensitive)
    3. Substring match (normalized)
    4. Fuzzy match (simple Levenshtein-like, normalized)

    Args:
        target_name: Column name to find
        available_columns: List of available column names

    Returns:
        Matching column name from available_columns, or None if no match
    """
    # Strategy 1: Exact match
    if target_name in available_columns:
        return target_name

    # Strategy 2: Normalized exact match
    target_normalized = _normalize_column_name(target_name)
    for col in available_columns:
        if _normalize_column_name(col) == target_normalized:
            return col

    # Strategy 3: Substring match (normalized) - but require key terms for specificity
    # Extract key terms from target (e.g., "t score", "z score", "viral load")
    key_terms_in_target = []
    important_terms = ["t score", "z score", "viral load", "cd4", "age", "regimen"]
    for term in important_terms:
        if term in target_normalized:
            key_terms_in_target.append(term)

    # If target has specific key terms, require at least one to match
    if key_terms_in_target:
        for col in available_columns:
            col_normalized = _normalize_column_name(col)
            # Check if all key terms from target are in column
            if all(term in col_normalized for term in key_terms_in_target):
                return col
        # If no exact key term match, don't use substring matching (too risky)
    else:
        # No key terms - use general substring matching
        for col in available_columns:
            col_normalized = _normalize_column_name(col)
            if target_normalized in col_normalized or col_normalized in target_normalized:
                return col

    # Strategy 4: Simple fuzzy match (character overlap ratio)
    # This is a lightweight alternative to full Levenshtein
    best_match = None
    best_score = 0.0
    target_chars = set(target_normalized.replace(" ", ""))

    for col in available_columns:
        col_normalized = _normalize_column_name(col)
        col_chars = set(col_normalized.replace(" ", ""))

        if not target_chars or not col_chars:
            continue

        # Jaccard similarity on character sets
        intersection = len(target_chars & col_chars)
        union = len(target_chars | col_chars)
        score = intersection / union if union > 0 else 0.0

        # Boost score if key terms match (e.g., "t score" in both)
        key_terms = ["t score", "z score", "dexa", "viral load", "cd4"]
        for term in key_terms:
            if term in target_normalized and term in col_normalized:
                score += 0.3
                break

        if score > best_score and score > 0.5:  # Minimum threshold
            best_score = score
            best_match = col

    return best_match


def _try_convert_to_numeric(series: pl.Series) -> pl.Series | None:
    """
    Try to convert a string series to numeric.

    Handles:
    - European comma format: "-1,8" -> -1.8 (comma as decimal separator)
    - US thousands format: "1,234.5" -> 1234.5 (comma as thousands separator)
    - Below detection limit: "<20" -> 20 (uses upper bound, extracts numeric part)
    - Above detection limit: ">100" -> 100 (uses lower bound, extracts numeric part)
    - Empty strings and whitespace -> null
    - Non-numeric values -> null

    Returns None if conversion fails or results in all nulls.
    """
    if series.dtype != pl.Utf8:
        return None

    try:
        # Step 1: Extract numeric part (handles "<20", ">100", etc.)
        # This pattern extracts digits, commas, dots, and minus signs
        # It will extract "20" from "<20", "1234.5" from "1,234.5", etc.
        extracted = series.str.extract(r"([\d,\.\-]+)", 1)

        # Step 2: Clean based on format detection
        # Strategy: Use map_elements for complex logic that Polars string ops can't handle easily
        def clean_numeric(s: str | None) -> str | None:
            if s is None:
                return None
            s = s.strip()
            if not s:
                return None

            # Check if it looks like US format (has both comma and dot)
            # In US format: "1,234.5" -> comma is thousands, dot is decimal -> remove comma
            if "," in s and "." in s:
                # US format: remove commas (thousands separator)
                return s.replace(",", "")
            # Check if it looks like European format (comma but no dot)
            # In European format: "1,8" -> comma is decimal -> replace with dot
            elif "," in s and "." not in s:
                # European format: replace comma with dot (decimal separator)
                return s.replace(",", ".")
            # No comma, use as-is
            return s

        # Apply cleaning using map_elements
        cleaned = extracted.map_elements(clean_numeric, return_dtype=pl.Utf8)

        # Step 3: Try to cast to float
        numeric = cleaned.cast(pl.Float64, strict=False)

        # Step 4: Check if we got any valid values
        if numeric.null_count() == len(numeric):
            # Fallback: try direct conversion on original series
            # Remove all commas (assume thousands separator) and try again
            cleaned_direct = (
                series.str.strip_chars()
                .str.replace(",", "")  # Remove commas
                .str.replace_all(r"^\s*$", "")  # Empty strings
            )
            numeric = cleaned_direct.cast(pl.Float64, strict=False)
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
    import structlog

    logger = structlog.get_logger()

    # Store original count before applying filters
    original_count = df.height

    # Apply filters from QueryPlan if present
    filters_applied = []
    if context.query_plan and context.query_plan.filters:
        df = _apply_filters(df, context.query_plan.filters)
        filters_applied = [f.__dict__ for f in context.query_plan.filters]
    elif context.filters:
        # Fallback to context.filters for backward compatibility
        df = _apply_filters(df, context.filters)
        filters_applied = [f.__dict__ for f in context.filters]

    filtered_count = df.height

    # Check if we're focusing on a specific variable
    primary_var = context.primary_variable
    focus_on_single = False
    matched_col = None

    if primary_var and primary_var != "all":
        # Use robust column matching to handle whitespace/normalization differences
        matched_col = _find_matching_column(primary_var, df.columns)
        focus_on_single = matched_col is not None

        logger.debug(
            "compute_descriptive_check",
            primary_var=primary_var,
            matched_col=matched_col,
            focus_on_single=focus_on_single,
            df_columns=df.columns[:5] if len(df.columns) > 5 else df.columns,
        )

        if not focus_on_single:
            logger.warning(
                "compute_descriptive_column_not_found",
                primary_var=primary_var,
                available_columns=list(df.columns),
            )
            # Return error result instead of falling through to "all"
            cols_preview = list(df.columns[:10])
            cols_str = ", ".join(cols_preview)
            if len(df.columns) > 10:
                cols_str += "..."
            return {
                "type": "descriptive",
                "error": f"Column '{primary_var}' not found in dataset. Available columns: {cols_str}",
                "requested_column": primary_var,
                "available_columns": list(df.columns),
            }

    if focus_on_single and matched_col:
        # Focused analysis on a single variable
        col = matched_col
        series = df[col]
        row_count = df.height

        logger.info(
            "compute_descriptive_focused",
            col=col,
            original_dtype=str(series.dtype),
            row_count=row_count,
        )

        # Try to convert string columns to numeric (handles European comma format, etc.)
        numeric_series = None
        if series.dtype == pl.Utf8:
            numeric_series = _try_convert_to_numeric(series)
            logger.debug(
                "compute_descriptive_conversion",
                col=col,
                conversion_success=numeric_series is not None,
            )

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
            "original_count": original_count,
            "filtered_count": filtered_count,
            "filters_applied": filters_applied,
            "filter_description": _format_filter_description(
                context.query_plan.filters if context.query_plan else context.filters
            )
            if (context.query_plan and context.query_plan.filters) or context.filters
            else "",
        }

        # Compute stats based on dtype
        if is_numeric:
            # Numeric variable - compute mean, median, std, min, max
            clean_series = analysis_series.drop_nulls()
            if len(clean_series) > 0:
                mean_val = clean_series.mean()
                median_val = clean_series.median()
                std_val = clean_series.std() if len(clean_series) > 1 else 0.0
                min_val = clean_series.min()
                max_val = clean_series.max()
                result["mean"] = float(mean_val) if mean_val is not None else 0.0  # type: ignore[arg-type]
                result["median"] = float(median_val) if median_val is not None else 0.0  # type: ignore[arg-type]
                result["std"] = float(std_val) if std_val is not None else 0.0  # type: ignore[arg-type]
                result["min"] = float(min_val) if min_val is not None else 0.0  # type: ignore[arg-type]
                result["max"] = float(max_val) if max_val is not None else 0.0  # type: ignore[arg-type]
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
        "original_count": original_count,
        "filtered_count": filtered_count,
        "filters_applied": filters_applied,
        "filter_description": _format_filter_description(
            context.query_plan.filters if context.query_plan else context.filters
        )
        if (context.query_plan and context.query_plan.filters) or context.filters
        else "",
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
        best_group = min(group_means, key=lambda k: group_means.get(k, float("inf")))
    else:
        best_group = max(group_means, key=lambda k: group_means.get(k, float("-inf")))

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

    if outcome_col is None or group_col is None:
        return {"type": "comparison", "error": "Missing required columns for comparison"}

    # Clean data
    analysis_df = df.select([outcome_col, group_col]).drop_nulls()

    if analysis_df.height < 2:
        return {"type": "comparison", "error": "Not enough data for comparison"}

    # CRITICAL FIX: Try numeric conversion FIRST, before deciding test type
    outcome_series = analysis_df[outcome_col]
    numeric_outcome = None
    outcome_is_numeric = False

    # Check if already numeric
    if outcome_series.dtype in (pl.Int64, pl.Float64):
        outcome_is_numeric = True
        numeric_outcome = outcome_series
    else:
        # Try to convert string columns to numeric (handles "<20", "120", etc.)
        numeric_outcome = _try_convert_to_numeric(outcome_series)
        if numeric_outcome is not None:
            outcome_is_numeric = True
            # Replace outcome column with numeric version
            analysis_df = analysis_df.with_columns(numeric_outcome.alias(outcome_col))

    # Determine appropriate test
    outcome_numeric = outcome_is_numeric

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

            # Group means - compute directly as dict comprehension
            anova_group_means: dict[str, float] = {}
            for g in groups:
                mean_value = analysis_df.filter(pl.col(group_col) == g)[outcome_col].mean()
                # Handle various return types from .mean() - use numpy conversion for safety
                if mean_value is not None:
                    # Convert to numpy scalar first, then float
                    anova_group_means[str(g)] = float(np.asarray(mean_value).item())
                else:
                    anova_group_means[str(g)] = 0.0

            # Compute headline answer
            headline = _compute_headline_answer(anova_group_means, outcome_col, group_col, "lowest")

            return {
                "type": "comparison",
                "test_type": "anova",
                "outcome_col": outcome_col,
                "group_col": group_col,
                "groups": groups,
                "n_groups": n_groups,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "group_means": anova_group_means,
                **headline,
            }

    else:
        # Chi-square test for categorical outcome
        # Build contingency table using Polars
        contingency = (
            analysis_df.group_by([outcome_col, group_col])
            .agg(pl.len().alias("count"))
            .pivot(on=group_col, index=outcome_col, values="count")
            .fill_null(0)
        )

        # Convert to numpy for scipy
        contingency_np = contingency.select([c for c in contingency.columns if c != outcome_col]).to_numpy()

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_np)

        # Store contingency as dict for serialization
        contingency_dict = contingency.to_dicts()

        # Try to compute group means if outcome can be converted to numeric
        # This allows answering "which group had the lowest X" for pseudo-numeric data
        headline: dict[str, Any] = {}  # type: ignore[no-redef]
        group_means: dict[str, float] = {}  # type: ignore[no-redef]
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


def _format_filter_description(filters: list[FilterSpec] | None) -> str:
    """
    Convert filter list to human-readable description.

    Args:
        filters: List of FilterSpec objects (or None)

    Returns:
        Human-readable filter description string
    """
    if not filters:
        return ""

    op_text = {
        "==": "equals",
        "!=": "does not equal",
        "<": "less than",
        ">": "greater than",
        "<=": "at most",
        ">=": "at least",
        "IN": "in",
        "NOT_IN": "not in",
    }

    descriptions = []
    for f in filters:
        op = op_text.get(f.operator, f.operator)
        descriptions.append(f"{f.column} {op} {f.value}")

    return "; ".join(descriptions)


def _apply_filters(df: pl.DataFrame, filters: list[FilterSpec]) -> pl.DataFrame:
    """
    Apply filter conditions to DataFrame.

    Args:
        df: Input DataFrame
        filters: List of FilterSpec objects

    Returns:
        Filtered DataFrame
    """
    if not filters:
        return df

    # Operator dispatch table
    filtered_df = df
    for filter_spec in filters:
        if filter_spec.column not in df.columns:
            continue  # Skip if column not found

        # Apply null exclusion if requested (default: yes)
        if filter_spec.exclude_nulls:
            filtered_df = filtered_df.filter(pl.col(filter_spec.column).is_not_null())

        # Get column dtype to check for type mismatches
        col_dtype = df.schema[filter_spec.column]

        # Apply operator with type safety
        try:
            if filter_spec.operator == "==":
                # Check for type mismatch: numeric column with string value
                if col_dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.Float64, pl.Float32):
                    if isinstance(filter_spec.value, str):
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Type mismatch: column '{filter_spec.column}' is {col_dtype} "
                            f"but filter value is string '{filter_spec.value}'. Skipping filter."
                        )
                        continue
                filtered_df = filtered_df.filter(pl.col(filter_spec.column) == filter_spec.value)
            elif filter_spec.operator == "!=":
                # Check for type mismatch
                if col_dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.Float64, pl.Float32):
                    if isinstance(filter_spec.value, str):
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Type mismatch: column '{filter_spec.column}' is {col_dtype} "
                            f"but filter value is string '{filter_spec.value}'. Skipping filter."
                        )
                        continue
                filtered_df = filtered_df.filter(pl.col(filter_spec.column) != filter_spec.value)
            elif filter_spec.operator == "<":
                filtered_df = filtered_df.filter(pl.col(filter_spec.column) < filter_spec.value)
            elif filter_spec.operator == ">":
                filtered_df = filtered_df.filter(pl.col(filter_spec.column) > filter_spec.value)
            elif filter_spec.operator == "<=":
                filtered_df = filtered_df.filter(pl.col(filter_spec.column) <= filter_spec.value)
            elif filter_spec.operator == ">=":
                filtered_df = filtered_df.filter(pl.col(filter_spec.column) >= filter_spec.value)
            elif filter_spec.operator == "IN":
                if isinstance(filter_spec.value, list):
                    filtered_df = filtered_df.filter(pl.col(filter_spec.column).is_in(filter_spec.value))
            elif filter_spec.operator == "NOT_IN":
                if isinstance(filter_spec.value, list):
                    filtered_df = filtered_df.filter(~pl.col(filter_spec.column).is_in(filter_spec.value))
        except pl.exceptions.ComputeError as e:
            # Log the error and skip this filter rather than crashing
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Error applying filter {filter_spec.column} {filter_spec.operator} {filter_spec.value}: {e}. "
                "Skipping filter."
            )
            continue

    return filtered_df


def compute_count_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Compute count analysis - returns total count, optionally grouped.

    Args:
        df: Polars DataFrame (cohort data)
        context: AnalysisContext with intent and variables

    Returns:
        Serializable dict with count results
    """
    # Apply filters from QueryPlan if present
    if context.query_plan and context.query_plan.filters:
        df = _apply_filters(df, context.query_plan.filters)
    elif context.filters:
        df = _apply_filters(df, context.filters)

    row_count = df.height

    # Check if query asks for "most" (e.g., "which statin was most prescribed?")
    is_most_query = False
    query_text = getattr(context, "query_text", None) or ""

    if query_text:
        query_lower = query_text.lower()
        is_most_query = "most" in query_lower and ("which" in query_lower or "what" in query_lower)

    # If grouping variable is specified, count by group
    if context.grouping_variable and context.grouping_variable in df.columns:
        group_col = context.grouping_variable
        counts = df.group_by(group_col).agg(pl.len().alias("count")).sort("count", descending=True)
        counts_dict = counts.to_dicts()

        # For "most" queries, return only the top result
        if is_most_query and len(counts_dict) > 0:
            counts_dict = [counts_dict[0]]  # Only top result

        # Create headline
        total = sum(item["count"] for item in counts_dict)
        if is_most_query and len(counts_dict) > 0:
            top_group = counts_dict[0]
            headline = f"**{top_group[group_col]}** with {top_group['count']} patients"
        else:
            headline = f"Total count: **{total}**"
            if len(counts_dict) > 0:
                top_group = counts_dict[0]
                headline += f" (largest group: {top_group[group_col]} with {top_group['count']})"

        return {
            "type": "count",
            "total_count": total,
            "grouped_by": group_col,
            "group_counts": counts_dict,
            "headline": headline,
            "is_most_query": is_most_query,  # Flag for rendering
        }
    else:
        # Simple total count
        return {
            "type": "count",
            "total_count": row_count,
            "headline": f"Total count: **{row_count}**",
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
    elif context.inferred_intent == AnalysisIntent.COUNT:
        return compute_count_analysis(df, context)
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

    if outcome_col is None:
        return {"type": "predictor", "error": "No outcome variable specified"}

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

    if time_col is None or event_col is None:
        return {"type": "survival", "error": "Time and event variables required for survival analysis"}

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

    # Normalize variable names to match actual dataframe columns
    # Handle cases where NLU extracted names don't exactly match column names (e.g., "BMI" vs "BMI ")
    actual_columns = df.columns
    normalized_variables = []
    missing_variables = []

    for var in variables:
        # Use the same robust column matching as compute_descriptive_analysis
        matched_col = _find_matching_column(var, actual_columns)
        if matched_col:
            normalized_variables.append(matched_col)
        else:
            missing_variables.append(var)

    if missing_variables:
        cols_preview = actual_columns[:10]
        cols_str = ", ".join(cols_preview)
        if len(actual_columns) > 10:
            cols_str += "..."
        return {
            "type": "relationship",
            "error": f"Variables not found in data: {', '.join(missing_variables)}. Available columns: {cols_str}",
        }

    if len(normalized_variables) < 2:
        return {"type": "relationship", "error": "Need at least 2 valid variables to examine relationships"}

    # Filter to only numeric variables (correlation requires numeric data)
    # Check which variables are numeric or can be converted to numeric
    numeric_variables = []
    non_numeric_variables = []

    for var in normalized_variables:
        try:
            # Try to get the column and check its type
            col_data = df.select(pl.col(var))
            dtype = col_data.schema[var]

            # Check if already numeric
            numeric_types = (
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            )
            if dtype in numeric_types:
                numeric_variables.append(var)
            else:
                # Try to convert to numeric (for string columns that might be numeric)
                # Test on a sample to see if conversion is possible
                sample = col_data.head(100)
                try:
                    # Try casting to float (if it works, column is numeric)
                    sample.select(pl.col(var).cast(pl.Float64, strict=True))
                    numeric_variables.append(var)
                except Exception:
                    # Cannot convert to numeric
                    non_numeric_variables.append(var)
        except Exception:
            # Column doesn't exist or other error - skip it
            non_numeric_variables.append(var)

    if len(numeric_variables) < 2:
        numeric_vars_str = ", ".join(numeric_variables) if numeric_variables else "none"
        excluded_str = ", ".join(non_numeric_variables[:5])
        if len(non_numeric_variables) > 5:
            excluded_str += "..."

        return {
            "type": "relationship",
            "error": (
                f"Need at least 2 numeric variables for correlation analysis. "
                f"Found {len(numeric_variables)} numeric variable(s): {numeric_vars_str}. "
                f"Non-numeric variables excluded: {excluded_str}"
            ),
            "numeric_variables": numeric_variables,
            "non_numeric_variables": non_numeric_variables,
        }

    # Calculate correlations using only numeric variables
    analysis_df = df.select(numeric_variables).drop_nulls()

    if analysis_df.height < 3:
        return {"type": "relationship", "error": "Need at least 3 observations with complete data"}

    # Cast to numeric for correlation (should all be numeric now, but ensure Float64)
    numeric_df = analysis_df.select([pl.col(v).cast(pl.Float64) for v in numeric_variables])

    # Compute correlation matrix using Polars
    corr_data = []
    for i, var1 in enumerate(numeric_variables):
        for j, var2 in enumerate(numeric_variables):
            if i <= j:
                if i == j:
                    corr_val = 1.0
                else:
                    # Compute correlation using Polars
                    corr_result = numeric_df.select(pl.corr(var1, var2))
                    corr_val = float(corr_result.item()) if corr_result.height > 0 else 0.0
                corr_data.append({"var1": var1, "var2": var2, "correlation": corr_val})

    # Find strong correlations (|r| >= 0.5)
    strong_correlations = [
        item
        for item in corr_data
        if item["var1"] != item["var2"] and abs(float(item["correlation"])) >= 0.5  # type: ignore[arg-type]
    ]

    # Find moderate correlations (0.3 <= |r| < 0.5)
    moderate_correlations = [
        item
        for item in corr_data
        if item["var1"] != item["var2"] and 0.3 <= abs(float(item["correlation"])) < 0.5  # type: ignore[arg-type]
    ]

    return {
        "type": "relationship",
        "variables": numeric_variables,  # Return only numeric variables used
        "non_numeric_excluded": non_numeric_variables,  # Inform user which variables were excluded
        "correlations": corr_data,
        "strong_correlations": strong_correlations,
        "moderate_correlations": moderate_correlations,
        "n_observations": analysis_df.height,
    }

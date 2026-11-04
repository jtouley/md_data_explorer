"""
Pure compute functions for analysis - Polars-first, return serializable dicts.

These functions have no UI dependencies and can be tested independently.
"""

from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from clinical_analytics.analysis.variable_types import binary_event_level, classify_variable, code_key, is_numeric_dtype
from clinical_analytics.core.analysis_types import AnalysisContext
from clinical_analytics.core.query_plan import FilterSpec


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


def _drop_missing(df: pl.DataFrame) -> pl.DataFrame:
    """Rows with no null and no NaN/inf in any column.

    Polars drop_nulls keeps NaN, and a NaN outcome turns every mean and p-value into NaN while
    n_excluded stays 0. Callers count the dropped rows as excluded.
    """
    finite = [pl.col(name).is_finite() for name, dtype in df.schema.items() if dtype.is_float()]
    complete = df.drop_nulls()
    return complete.filter(pl.all_horizontal(finite)) if finite else complete


# Spread at or below this fraction of the values' magnitude is float noise, not variation:
# 0.1 + 0.2 and 0.3 differ by ~1e-16 relative and must count as one constant value.
_CONSTANT_REL_TOL = 1e-9


def _finite_or_none(value: float) -> float | None:
    """An effect estimate, or None when it is inf/NaN (zero cell, overflow): never a non-finite number."""
    number = float(value)
    return number if np.isfinite(number) else None


_EFFECT_COLUMNS = ("Odds Ratio", "CI Lower", "CI Upper")


def _constant_groups(groups: list[Any], group_data: list[np.ndarray]) -> list[str]:
    """Groups whose values do not vary beyond float noise (range <= tolerance x magnitude)."""
    return [
        str(g)
        for g, d in zip(groups, group_data, strict=True)
        if float(np.ptp(d)) <= _CONSTANT_REL_TOL * float(np.max(np.abs(d)))
    ]


_MISSING_TOKENS = ("", "na", "n/a", "nan", "null", "none", "missing", ".", "-", "?")
_NUMERIC_TEXT = r"^\s*[<>]?\s*-?\d[\d,]*(\.\d+)?\s*$"


class AmbiguousNumberFormatError(ValueError):
    """Text numbers whose comma could be a decimal or a thousands separator."""


def _numeric_text_or_none(series: pl.Series) -> pl.Series | None:
    """Numbers stored as text, or None when any non-missing value is not a plain number.

    "Grade 2" or "1/2" stay text (categorical). "<20" reads as 20. Missing tokens ("n/a", ".")
    become null. Commas: "1,8" is a decimal comma and "1,234.5" / "12,345,678" use thousands
    separators; "1,234" with no other evidence could be either, so it raises
    AmbiguousNumberFormatError instead of guessing.
    """
    cleaned = series.str.strip_chars()
    is_missing = cleaned.is_null() | cleaned.str.to_lowercase().is_in(list(_MISSING_TOKENS))
    text = pl.select(pl.when(is_missing).then(None).otherwise(cleaned)).to_series()
    present = text.drop_nulls()
    if present.len() == 0 or not present.str.contains(_NUMERIC_TEXT).all():
        return None
    bare = text.str.replace(r"^[<>]\s*", "")
    with_comma = bare.drop_nulls().filter(bare.drop_nulls().str.contains(","))
    if with_comma.len() == 0:
        return bare.cast(pl.Float64, strict=False)
    thousands = r"^-?\d{1,3}(,\d{3})+(\.\d+)?$"
    if with_comma.str.contains(thousands).all():
        unambiguous = (
            with_comma.str.contains(r"\.").any()
            or with_comma.str.contains(r",\d{3},").any()
            or bare.drop_nulls().str.contains(r"^-?\d+\.\d+$").any()
        )
        if not unambiguous:
            raise AmbiguousNumberFormatError(
                f"{series.name} has values like '{with_comma[0]}' where the comma could mean thousands or a "
                "decimal point. Store the column as numbers, or use a dot for decimals."
            )
        return bare.str.replace_all(",", "").cast(pl.Float64, strict=False)
    if with_comma.str.contains(r"^-?\d+,\d+$").all() and not bare.drop_nulls().str.contains(r"\.").any():
        return bare.str.replace(",", ".").cast(pl.Float64, strict=False)
    raise AmbiguousNumberFormatError(
        f"{series.name} mixes comma formats (e.g. '{with_comma[0]}'), so its numbers cannot be read reliably."
    )


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

    df, applied = _apply_context_filters(df, context)
    filters_applied = [f.__dict__ for f in applied]

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
        # Mean/SD only for continuous measures; 0/1 flags and coded categories get frequencies.
        is_numeric = classify_variable(analysis_series) == "continuous"

        # NaN/inf are missing too: the mean below is over the finite values, and the counts match it.
        if is_numeric_dtype(analysis_series.dtype):
            non_null_count = _drop_missing(analysis_series.to_frame()).height
        else:
            non_null_count = row_count - analysis_series.null_count()
        missing_count = row_count - non_null_count

        result = {
            "type": "descriptive",
            "focused_variable": col,
            "row_count": row_count,
            "non_null_count": non_null_count,
            "null_count": missing_count,
            "null_pct": (missing_count / row_count * 100) if row_count > 0 else 0.0,
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
            clean_series = _drop_missing(analysis_series.to_frame()).to_series()
            if clean_series.len() > 0:
                mean_val = clean_series.mean()
                median_val = clean_series.median()
                std_val = clean_series.std() if len(clean_series) > 1 else None  # SD is undefined for one value
                min_val = clean_series.min()
                max_val = clean_series.max()
                result["mean"] = float(mean_val) if mean_val is not None else 0.0  # type: ignore[arg-type]
                result["median"] = float(median_val) if median_val is not None else 0.0  # type: ignore[arg-type]
                result["std"] = float(std_val) if std_val is not None else None  # type: ignore[arg-type]
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
    numeric_cols = [col for col in df.columns if classify_variable(df[col]) == "continuous"]
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
    if is_numeric_dtype(series.dtype):
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
    Compare an outcome across groups, choosing the test from what the outcome measures.

    - continuous outcome, 2 groups: Welch t-test; the 95% CI uses the same Welch standard
      error and Welch-Satterthwaite degrees of freedom as the p-value
    - continuous outcome, >2 groups: one-way ANOVA
    - binary or categorical outcome: chi-square, or Fisher's exact test for a 2x2 table with
      an expected count below 5; binary outcomes also report per-group proportions

    Rows missing the outcome or group (null, NaN or infinite) are excluded and counted
    (n_excluded). Refused with an error, never reported as a result: a Welch test where either
    group's values do not vary (its p-value is a numerical limit and its interval undefined),
    and an ANOVA with a group under 2 patients or no variation within any group (F is inf or NaN).

    Fisher's odds ratio is the odds of the event level in groups[1] over groups[0] (groups in
    sorted order, groups[0] the reference, as in the logistic model's indicators). It does not
    depend on what the non-event level is called, and is omitted when the event level is unknown.
    Returns serializable dict (no Polars objects).
    """
    df, _ = _apply_context_filters(df, context)

    outcome_col = context.primary_variable
    group_col = context.grouping_variable

    if outcome_col is None or group_col is None:
        return {"type": "comparison", "error": "Missing required columns for comparison"}

    n_in_scope = df.height
    analysis_df = _drop_missing(df.select([outcome_col, group_col]))
    n_used = analysis_df.height
    n_excluded = n_in_scope - n_used

    if n_used < 2:
        return {"type": "comparison", "error": "Not enough data for comparison"}

    outcome_series = analysis_df[outcome_col]
    if outcome_series.dtype == pl.Utf8:
        try:
            converted = _numeric_text_or_none(outcome_series)
        except AmbiguousNumberFormatError as exc:
            return {"type": "comparison", "error": str(exc)}
        if converted is not None:
            # Missing-value tokens ("n/a", ".") became null: exclude and count them.
            analysis_df = _drop_missing(analysis_df.with_columns(converted.alias(outcome_col)))
            n_used = analysis_df.height
            n_excluded = n_in_scope - n_used
            if n_used < 2:
                return {"type": "comparison", "error": "Not enough data for comparison"}
            outcome_series = analysis_df[outcome_col]
    outcome_kind = classify_variable(outcome_series)
    if outcome_series.n_unique() < 2:
        return {
            "type": "comparison",
            "error": f"{outcome_col} has only one value in the rows analysed, so there is nothing to compare.",
        }

    groups = analysis_df[group_col].unique().sort().to_list()  # deterministic group order
    n_groups = len(groups)
    if n_groups < 2:
        return {"type": "comparison", "error": "Need at least 2 groups for comparison"}

    group_sizes = {str(g): int(n) for g, n in analysis_df.group_by(group_col).len().sort(group_col).iter_rows()}
    base: dict[str, Any] = {
        "type": "comparison",
        "outcome_col": outcome_col,
        "group_col": group_col,
        "outcome_kind": outcome_kind,
        "groups": groups,
        "n_groups": n_groups,
        "group_sizes": group_sizes,
        "n_used": n_used,
        "n_excluded": n_excluded,
    }

    if outcome_kind == "continuous":
        values = analysis_df.with_columns(pl.col(outcome_col).cast(pl.Float64))
        group_data = [values.filter(pl.col(group_col) == g)[outcome_col].to_numpy() for g in groups]
        group_means = {str(g): float(np.mean(d)) for g, d in zip(groups, group_data, strict=True)}
        headline = _compute_headline_answer(group_means, outcome_col, group_col, "lowest")
        if n_groups == 2:
            a, b = group_data
            if min(len(a), len(b)) < 2:
                return {**base, "error": "Each group needs at least 2 observations for a t-test"}
            constant = _constant_groups(groups, group_data)
            if constant:
                # Zero within-group variance: scipy returns p=0 with t=inf and the Welch df is NaN.
                return {
                    **base,
                    "error": f"{outcome_col} values do not vary within a group ({', '.join(constant)}), so a "
                    "t-test p-value and confidence interval cannot be calculated.",
                }
            statistic, p_value = stats.ttest_ind(a, b, equal_var=False)
            var_a, var_b = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
            se_diff = float(np.sqrt(var_a + var_b))
            welch_df = float((var_a + var_b) ** 2 / (var_a**2 / (len(a) - 1) + var_b**2 / (len(b) - 1)))
            mean_diff = float(np.mean(a) - np.mean(b))
            t_crit = float(stats.t.ppf(0.975, welch_df))
            return {
                **base,
                "test_type": "welch_t_test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "df": welch_df,
                "group1_mean": float(np.mean(a)),
                "group2_mean": float(np.mean(b)),
                "mean_diff": mean_diff,
                "ci_lower": mean_diff - t_crit * se_diff,
                "ci_upper": mean_diff + t_crit * se_diff,
                "group_means": group_means,
                **headline,
            }
        small = [str(g) for g, d in zip(groups, group_data, strict=True) if len(d) < 2]
        if small:
            return {**base, "error": f"Each group needs at least 2 observations for ANOVA ({', '.join(small)})"}
        if len(_constant_groups(groups, group_data)) == n_groups:
            # No within-group variation: scipy returns F=inf and p=0.
            return {
                **base,
                "error": f"{outcome_col} values do not vary within any {group_col} group, so an ANOVA p-value "
                "cannot be calculated.",
            }
        statistic, p_value = stats.f_oneway(*group_data)
        return {
            **base,
            "test_type": "anova",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "group_means": group_means,
            **headline,
        }

    # Binary or categorical outcome: contingency table (rows = outcome levels, columns = groups).
    outcome_levels = analysis_df[outcome_col].unique().sort().to_list()
    table = np.array(
        [
            [analysis_df.filter((pl.col(outcome_col) == lvl) & (pl.col(group_col) == g)).height for g in groups]
            for lvl in outcome_levels
        ]
    )
    contingency = [
        {outcome_col: str(lvl), **{str(g): int(table[i, j]) for j, g in enumerate(groups)}}
        for i, lvl in enumerate(outcome_levels)
    ]
    expected = stats.contingency.expected_freq(table)
    result: dict[str, Any] = {
        **base,
        "contingency": contingency,
        "contingency_index_col": outcome_col,
        "min_expected_count": float(expected.min()),
    }
    event_row: int | None = None
    if outcome_kind == "binary":
        event = binary_event_level(analysis_df[outcome_col])
        if event is not None and event in outcome_levels:
            event_row = outcome_levels.index(event)
            proportions = {str(g): float(table[event_row, j] / table[:, j].sum()) for j, g in enumerate(groups)}
            result.update({"event_level": code_key(event), "group_proportions": proportions})
    if table.shape == (2, 2) and expected.min() < 5:
        _, p_value = stats.fisher_exact(table)  # two-sided p does not depend on row or column order
        result.update({"test_type": "fisher_exact", "p_value": float(p_value)})
        if event_row is not None:
            # Rows: event, non-event. Columns: groups[1], groups[0]. scipy's (a*d)/(b*c) is then
            # the odds of the event in groups[1] relative to groups[0].
            oriented = table[[event_row, 1 - event_row]][:, [1, 0]]
            odds_ratio, _ = stats.fisher_exact(oriented)
            result.update(
                {
                    "odds_ratio": _finite_or_none(odds_ratio),  # None for a zero cell (inf)
                    "odds_ratio_group": str(groups[1]),
                    "odds_ratio_reference": str(groups[0]),
                }
            )
        return result
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    result.update({"test_type": "chi_square", "chi2": float(chi2), "p_value": float(p_value), "dof": int(dof)})
    if expected.min() < 5:
        result["warning"] = "Some expected counts are below 5; the chi-square p-value may be unreliable."
    return result


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


def _apply_context_filters(df: pl.DataFrame, context: AnalysisContext) -> tuple[pl.DataFrame, list[FilterSpec]]:
    """
    Apply the question's filters (QueryPlan first, legacy context.filters as fallback).

    Every compute_* analysis must call this so filtered questions are answered on the filtered rows.

    Returns:
        (filtered DataFrame, filters that were applied)
    """
    filters = (context.query_plan.filters if context.query_plan and context.query_plan.filters else None) or (
        context.filters or []
    )
    if not filters:
        return df, []
    return _apply_filters(df, filters), list(filters)


def compute_count_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Compute count analysis - returns total count, optionally grouped.

    Args:
        df: Polars DataFrame (cohort data)
        context: AnalysisContext with intent and variables

    Returns:
        Serializable dict with count results
    """
    df, _ = _apply_context_filters(df, context)

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
    from clinical_analytics.core.analysis_types import AnalysisIntent

    # DESCRIBE / COUNT resolve fuzzy column names themselves; the others select columns directly,
    # so an unknown name must become a renderable error instead of a ColumnNotFoundError.
    result_types = {
        AnalysisIntent.COMPARE_GROUPS: "comparison",
        AnalysisIntent.FIND_PREDICTORS: "predictor",
        AnalysisIntent.EXAMINE_SURVIVAL: "survival",
        AnalysisIntent.EXPLORE_RELATIONSHIPS: "relationship",
    }
    if context.inferred_intent in result_types:
        referenced = [
            context.primary_variable,
            context.grouping_variable,
            context.time_variable,
            context.event_variable,
            *(context.predictor_variables or []),
        ]
        missing = [col for col in referenced if col is not None and col not in df.columns]
        if missing:
            return {
                "type": result_types[context.inferred_intent],
                "error": f"Column(s) not found in dataset: {missing}",
                "available_columns": df.columns,
            }

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


def _needs_indicators(series: pl.Series) -> bool:
    kind = classify_variable(series)
    if series.dtype == pl.Utf8 or kind == "categorical":
        return True
    return kind == "binary" and not set(series.drop_nulls().unique().to_list()) <= {0, 1}


def _code_text(series: pl.Series) -> pl.Series:
    """Codes as text for indicator names: 2.0 -> '2' so names match the header's codes."""
    if series.dtype.is_float() and bool((series.drop_nulls() % 1 == 0).all()):
        return series.cast(pl.Int64).cast(pl.Utf8)
    return series.cast(pl.Utf8)


def _ambiguous_event_message(column: str, series: pl.Series) -> str:
    values = ", ".join(str(v) for v in sorted(series.drop_nulls().unique().to_list(), key=str))
    return (
        f"Cannot tell which value of {column} is the event (values: {values}). Recode it to 0/1, "
        f"or name the codes in the column header, e.g. '{column} 1: Yes 2: No'."
    )


def compute_predictor_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Logistic regression of a binary outcome on the named predictors.

    The outcome is recoded explicitly to 0/1 and the modelled event level is reported, so a
    1/2-coded or text outcome is never fitted on raw codes. Coded categorical predictors
    (e.g. "1: Normal 2: Osteopenia") enter as indicator variables, not as numbers. Perfect
    separation is refused rather than reported with meaningless odds ratios.

    Returns serializable dict (no Polars objects).
    """
    df, _ = _apply_context_filters(df, context)

    import pandas as pd  # PANDAS EXCEPTION: statsmodels formula API needs a pandas frame

    from clinical_analytics.analysis.stats import ModelNotEstimableError, run_logistic_regression

    outcome_col = context.primary_variable
    predictors = context.predictor_variables

    if outcome_col is None:
        return {"type": "predictor", "error": "No outcome variable specified"}
    if not predictors:
        # An empty right-hand side would reach the regression as "outcome ~" and fail to parse.
        return {
            "type": "predictor",
            "error": f"No predictors specified for {outcome_col}. Name them, e.g. 'do age and treatment predict "
            f"{outcome_col}?'",
        }

    n_in_scope = df.height
    analysis_df = _drop_missing(df.select([outcome_col, *predictors]))
    n_used = analysis_df.height
    base: dict[str, Any] = {
        "type": "predictor",
        "outcome_col": outcome_col,
        "n_used": n_used,
        "n_excluded": n_in_scope - n_used,
    }

    if n_used < 10:
        return {**base, "error": "Need at least 10 complete observations"}

    outcome_series = analysis_df[outcome_col]
    n_unique = outcome_series.n_unique()
    if n_unique != 2:
        return {
            **base,
            "error": f"Outcome has {n_unique} unique values. Only binary outcomes are supported.",
        }
    event_level = binary_event_level(outcome_series)
    if event_level is None:
        return {**base, "error": _ambiguous_event_message(outcome_col, outcome_series)}
    analysis_df = analysis_df.with_columns(
        pl.col(outcome_col).eq(pl.lit(event_level)).cast(pl.Int64).alias(outcome_col)
    )

    # Booleans fit as 0/1. Coded categoricals, text, and binary codes other than 0/1 (e.g. 1/2)
    # become indicator variables named "<column>_<code>", never fitted on raw codes.
    analysis_df = analysis_df.with_columns(
        [pl.col(p).cast(pl.Int64) for p in predictors if analysis_df[p].dtype == pl.Boolean]
    )
    categorical = [p for p in predictors if _needs_indicators(analysis_df[p])]
    analysis_df_pd = analysis_df.with_columns([_code_text(analysis_df[c]) for c in categorical]).to_pandas()
    if categorical:
        analysis_df_pd = pd.get_dummies(analysis_df_pd, columns=categorical, drop_first=True, dtype=int)
    model_predictors = [c for c in analysis_df_pd.columns if c != outcome_col]

    try:
        _, summary_df = run_logistic_regression(analysis_df_pd, outcome_col, model_predictors)
    except ModelNotEstimableError as exc:
        return {**base, "error": str(exc), "event_level": code_key(event_level)}

    summary_dict = {
        var: {col: (_finite_or_none(val) if col in _EFFECT_COLUMNS else val) for col, val in row.items()}
        for var, row in summary_df.to_dict(orient="index").items()
    }
    significant = summary_df[summary_df["P-Value"] < 0.05]
    significant_predictors = [
        {
            "variable": var,
            "odds_ratio": _finite_or_none(summary_df.loc[var, "Odds Ratio"]),
            "p_value": float(summary_df.loc[var, "P-Value"]),
            "ci_lower": _finite_or_none(summary_df.loc[var, "CI Lower"]),
            "ci_upper": _finite_or_none(summary_df.loc[var, "CI Upper"]),
        }
        for var in significant.index
        if var != "Intercept"
    ]
    # Strongest = smallest p-value; ties keep model order.
    significant_predictors.sort(key=lambda row: row["p_value"])

    return {
        **base,
        "event_level": code_key(event_level),
        "summary": summary_dict,
        "significant_predictors": significant_predictors,
        "schema": {col: str(dtype) for col, dtype in summary_df.dtypes.items()},
    }


def compute_survival_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Kaplan-Meier survival, overall and by group, with a log-rank test across groups.

    The event column is recoded explicitly to 0/1 (event level reported). Medians are from
    each group's own fit; the overall median is from an ungrouped fit.

    Returns serializable dict (no Polars objects).
    """
    df, _ = _apply_context_filters(df, context)

    from lifelines import KaplanMeierFitter

    from clinical_analytics.analysis.survival import run_kaplan_meier, run_logrank_test

    time_col = context.time_variable
    event_col = context.event_variable
    group_col = context.grouping_variable

    if time_col is None or event_col is None:
        return {"type": "survival", "error": "Time and event variables required for survival analysis"}

    columns = [time_col, event_col] + ([group_col] if group_col else [])
    n_in_scope = df.height
    analysis_df = _drop_missing(df.select(columns))
    n_used = analysis_df.height
    base: dict[str, Any] = {
        "type": "survival",
        "time_col": time_col,
        "event_col": event_col,
        "group_col": group_col,
        "n_used": n_used,
        "n_excluded": n_in_scope - n_used,
    }

    if n_used < 10:
        return {**base, "error": "Need at least 10 complete observations"}

    unique_vals = sorted(analysis_df[event_col].unique().to_list(), key=str)
    if len(unique_vals) != 2:
        return {
            **base,
            "error": f"Event variable must be binary (has {len(unique_vals)} values)",
            "unique_values": unique_vals,
        }
    event_level = binary_event_level(analysis_df[event_col])
    if event_level is None:
        return {**base, "error": _ambiguous_event_message(event_col, analysis_df[event_col])}
    analysis_df = analysis_df.with_columns(pl.col(event_col).eq(pl.lit(event_level)).cast(pl.Int64).alias(event_col))
    n_events = int(analysis_df[event_col].sum())

    # PANDAS EXCEPTION: lifelines takes pandas inputs
    analysis_df_pd = analysis_df.to_pandas()

    overall = KaplanMeierFitter().fit(analysis_df_pd[time_col], analysis_df_pd[event_col])
    median_survival = None if np.isinf(overall.median_survival_time_) else float(overall.median_survival_time_)
    _, summary_df = run_kaplan_meier(analysis_df_pd, duration_col=time_col, event_col=event_col, group_col=group_col)

    result: dict[str, Any] = {
        **base,
        "event_level": code_key(event_level),
        "n_events": n_events,
        "summary": summary_df.to_dict(orient="records"),
        "median_survival": median_survival,
        "unique_values": unique_vals,
    }
    if group_col:
        group_medians: dict[str, float | None] = {}
        group_sizes: dict[str, int] = {}
        for group, rows in analysis_df_pd.groupby(group_col, sort=True):
            fit = KaplanMeierFitter().fit(rows[time_col], rows[event_col])
            group_medians[str(group)] = (
                None if np.isinf(fit.median_survival_time_) else float(fit.median_survival_time_)
            )
            group_sizes[str(group)] = len(rows)
        result.update({"group_medians": group_medians, "group_sizes": group_sizes})
        if len(group_sizes) >= 2:
            logrank = run_logrank_test(analysis_df_pd, time_col, event_col, group_col)
            result.update({"logrank_p_value": logrank["p_value"], "logrank_statistic": logrank["test_statistic"]})
    return result


def compute_relationship_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict[str, Any]:
    """
    Compute relationship analysis (correlations).

    Returns serializable dict (no Polars objects).
    """
    df, _ = _apply_context_filters(df, context)

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
    analysis_df = _drop_missing(df.select(numeric_variables))

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

"""
Dynamic Analysis Page - Question-Driven Analytics

Ask questions, get answers. No statistical jargon - just tell me what you want to know.
"""

import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import streamlit as st
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import from config (single source of truth)
# Import analysis compute functions (pure, no UI dependencies)
from clinical_analytics.analysis.compute import compute_analysis_by_type
from clinical_analytics.core.column_parser import parse_column_name
from clinical_analytics.core.nl_query_config import AUTO_EXECUTE_CONFIDENCE_THRESHOLD
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
from clinical_analytics.ui.components.question_engine import (
    AnalysisContext,
    AnalysisIntent,
    QuestionEngine,
)
from clinical_analytics.ui.components.result_interpreter import ResultInterpreter
from clinical_analytics.ui.config import MULTI_TABLE_ENABLED
from clinical_analytics.ui.messages import (
    COLLISION_SUGGESTION_WARNING,
    LOW_CONFIDENCE_WARNING,
)


# TypedDict schemas for chat transcript and pending state
class ChatMessage(TypedDict):
    """Schema for chat transcript messages."""

    role: str  # "user" or "assistant"
    text: str  # Message text
    run_key: str | None  # Run key for assistant messages (None for user messages)
    status: str  # "pending", "completed", "error"
    created_at: float  # Unix timestamp


class Pending(TypedDict):
    """Schema for pending computation state."""

    run_key: str  # Run key for pending computation
    context: AnalysisContext  # Analysis context for computation
    query_text: str  # Original query text
    query_plan: object | None  # QueryPlan object (if available)


# Page config
st.set_page_config(page_title="Ask Questions | Clinical Analytics", page_icon="💬", layout="wide")
# Prevent emoji rendering artifacts in chat messages
# Ensure chat message icons persist across reruns
st.markdown(
    """
    <style>
    .stChatMessage {
        border-bottom: none !important;
    }
    .stChatMessage:not(:last-child) {
        margin-bottom: 0.5rem;
    }
    /* Ensure chat message avatars persist */
    .stChatMessage [data-testid="stAvatar"] {
        display: block !important;
        visibility: visible !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Structured logging
logger = structlog.get_logger()

# Constants
MAX_STORED_RESULTS_PER_DATASET = 5

# Caching Strategy (Phase 3):
# - Cohorts: NOT cached via st.cache_data (handled via session_state with lifecycle management)
# - Alias index: Already lazy in SemanticLayer (built once per instance)
# - Small intermediates: Can be cached with @st.cache_data if needed (profiling, metadata)
# - Cache keys: Must include dataset_version (upload_id/file_hash) as explicit function arguments
# - Do NOT access st.session_state inside cached functions


@st.cache_resource(show_spinner="Loading semantic layer...")
def get_cached_semantic_layer(dataset_version: str, _dataset):
    """
    Get semantic layer with caching (Phase 1.2 - PR20 P0 Fix).

    IMPORTANT: Uses @st.cache_resource (not @st.cache_data) because:
    - SemanticLayer contains non-picklable objects (DB connections, Ibis expressions)
    - cache_data requires pickling, which fails with DuckDB/Ibis backends
    - cache_resource stores objects in memory without serialization

    Cache key includes dataset_version to invalidate on dataset changes.
    Dataset object passed with _ prefix (not hashed, used for computation only).

    Args:
        dataset_version: Dataset version identifier (for cache key)
        _dataset: Dataset object (not hashed, just used to call get_semantic_layer)

    Returns:
        Semantic layer instance

    Raises:
        ValueError: If semantic layer not available
    """
    return _dataset.get_semantic_layer()


def normalize_query(q: str | None) -> str:
    """
    Normalize query text: collapse whitespace, lowercase, strip.

    This is the single source of truth for query normalization.
    All queries must be normalized immediately after st.chat_input().

    Args:
        q: Raw query text (may be None)

    Returns:
        Normalized query string (lowercase, single spaces, stripped)
    """
    if q is None:
        return ""
    # Collapse whitespace, lowercase, strip
    return " ".join(q.strip().split()).lower()


def canonicalize_scope(scope: dict | None) -> dict:
    """
    Canonicalize semantic scope dict for stable hashing.

    - Drops None values
    - Sorts dictionary keys
    - Sorts list values
    - Ensures stable JSON serialization

    Args:
        scope: Semantic scope dict (may be None)

    Returns:
        Canonicalized scope dict (stable, sorted, no Nones)
    """
    if scope is None:
        return {}

    canonical = {}
    for key in sorted(scope.keys()):
        value = scope[key]
        if value is None:
            continue  # Drop None values
        if isinstance(value, list):
            canonical[key] = sorted(value)  # Sort lists for stability
        else:
            canonical[key] = value

    return canonical


def generate_run_key(
    dataset_version: str,
    query_text: str,  # Must be already normalized, never None
    context: AnalysisContext,
    scope: dict | None = None,
) -> str:
    """
    Generate stable run key for idempotency.

    Canonicalizes inputs to ensure same query + variables + scope = same key.

    Args:
        dataset_version: Dataset version identifier
        query_text: Normalized query text (must be pre-normalized, never None)
        context: Analysis context with variables
        scope: Semantic scope dict (optional, will be canonicalized)

    Returns:
        SHA256 hash of canonicalized payload
    """
    # Extract material context variables (only those that affect computation)
    material_vars = {
        "primary": context.primary_variable or "",
        "grouping": context.grouping_variable or "",
        "predictors": sorted(context.predictor_variables or []),
        "time": context.time_variable or "",
        "event": context.event_variable or "",
    }

    # Build payload with canonicalized scope
    payload = {
        "dataset_version": dataset_version,
        "query": query_text,  # Already normalized by caller
        "intent": context.inferred_intent.value if context.inferred_intent else "UNKNOWN",
        "vars": material_vars,
        "scope": canonicalize_scope(scope),  # Include canonicalized scope
    }

    # Stable JSON serialization
    payload_str = json.dumps(payload, sort_keys=True)
    run_key = hashlib.sha256(payload_str.encode()).hexdigest()

    return run_key


def remember_run(dataset_version: str, run_key: str) -> None:
    """
    Remember this run in history for dataset version.

    O(1) eviction: Proactively delete evicted result when deque reaches maxlen,
    instead of scanning all session_state keys.

    Note: Store history as list[str] (not deque) to avoid serialization quirks.
    Convert to deque locally for LRU logic.
    """
    hist_key = f"run_history_{dataset_version}"
    hist_list = st.session_state.get(hist_key, [])

    # Convert to deque for LRU logic (local only, not stored)
    hist = deque(hist_list, maxlen=MAX_STORED_RESULTS_PER_DATASET)

    # Capture what will be evicted BEFORE any modifications
    evicted_key = None
    if len(hist) == MAX_STORED_RESULTS_PER_DATASET and run_key not in hist:
        evicted_key = hist[0]  # Oldest will be evicted

    # De-dupe: move existing key to end (LRU behavior)
    if run_key in hist:
        hist.remove(run_key)
    hist.append(run_key)

    # Store back as list (deque not serializable in session_state)
    st.session_state[hist_key] = list(hist)

    # Delete evicted result immediately (O(1) instead of O(n) scan)
    if evicted_key:
        result_key = f"analysis_result:{dataset_version}:{evicted_key}"
        if result_key in st.session_state:
            del st.session_state[result_key]
            logger.info(
                "evicted_old_result",
                dataset_version=dataset_version,
                evicted_run_key=evicted_key,
            )


def cleanup_old_results(dataset_version: str) -> None:
    """
    Safety net: Remove any orphaned results not in history.

    Note: Most cleanup happens proactively in remember_run() (O(1)).
    This function is a safety net for edge cases (e.g., manual deletions).
    """
    hist_key = f"run_history_{dataset_version}"
    hist_list = st.session_state.get(hist_key, [])

    if not hist_list:
        return

    # Keep only results in history (using dataset-scoped keys)
    keep_keys = {f"analysis_result:{dataset_version}:{rk}" for rk in hist_list}

    # Remove any dataset-scoped result keys not in keep set
    result_prefix = f"analysis_result:{dataset_version}:"
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith(result_prefix) and key not in keep_keys]

    for key in keys_to_remove:
        del st.session_state[key]

    if keys_to_remove:
        logger.info(
            "cleaned_orphaned_results",
            dataset_version=dataset_version,
            count=len(keys_to_remove),
        )


def _render_add_alias_ui(
    unknown_term: str,
    available_columns: list[str],
    upload_id: str,
    dataset_version: str,
    semantic_layer,
) -> None:
    """
    Render "Add alias?" UI for unknown terms (Phase 2 - ADR003).

    Args:
        unknown_term: Term that wasn't recognized
        available_columns: List of available column names
        upload_id: Upload identifier
        dataset_version: Dataset version
        semantic_layer: SemanticLayer instance
    """
    with st.expander(f"💡 Add alias for '{unknown_term}'?"):
        st.info(f"**Unknown term**: '{unknown_term}'")
        st.caption("Map this term to a column to use it in future queries.")

        selected_column = st.selectbox(
            "Map to column:",
            options=["(Select column)"] + available_columns,
            key=f"add_alias_{unknown_term}_{dataset_version}",
        )

        if st.button("Save Alias", key=f"save_alias_{unknown_term}_{dataset_version}"):
            if selected_column and selected_column != "(Select column)":
                try:
                    semantic_layer.add_user_alias(unknown_term, selected_column, upload_id, dataset_version)
                    st.success(f"✅ Alias saved! '{unknown_term}' now maps to '{selected_column}'. Retry your query.")
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ Error saving alias: {e}")
                except Exception as e:
                    logger.error(f"Error adding user alias: {e}", exc_info=True)
                    st.error(f"❌ Unexpected error: {e}")
            else:
                st.warning("Please select a column to map to.")


def clear_all_results(dataset_version: str) -> None:
    """
    Clear all results and history for this dataset version.

    Dataset-scoped: Only clears results for this specific dataset_version,
    not global results. Uses dataset-scoped result keys for trivial cleanup.
    """
    hist_key = f"run_history_{dataset_version}"

    # Clear history
    if hist_key in st.session_state:
        del st.session_state[hist_key]

    # Clear all dataset-scoped results (trivial with scoped keys)
    result_prefix = f"analysis_result:{dataset_version}:"
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith(result_prefix)]

    for key in keys_to_remove:
        del st.session_state[key]

    # Clear dataset-scoped last_run_key
    dataset_last_run_key = f"last_run_key:{dataset_version}"
    if dataset_last_run_key in st.session_state:
        del st.session_state[dataset_last_run_key]

    logger.info(
        "cleared_all_results",
        dataset_version=dataset_version,
        result_count=len(keys_to_remove),
    )


# PANDAS EXCEPTION: Required for Streamlit st.dataframe display
# TODO: Remove when Streamlit supports Polars natively
def render_descriptive_analysis(result: dict, query_text: str | None = None) -> None:
    """
    Render descriptive analysis from serializable dict.

    Args:
        result: Analysis result dict
        query_text: Original query text (to determine if user explicitly asked for summary)
    """
    # Check for error results first
    if "error" in result:
        st.error(f"❌ **Analysis Error**: {result['error']}")
        if "available_columns" in result:
            cols_preview = result["available_columns"][:20]
            cols_str = ", ".join(cols_preview)
            if len(result["available_columns"]) > 20:
                cols_str += "..."
            st.info(f"💡 **Available columns**: {cols_str}")

        # Phase 2: Add "Add alias?" UI for unknown terms
        if "unknown_term" in result:
            _render_add_alias_ui(
                unknown_term=result["unknown_term"],
                available_columns=result.get("available_columns", []),
                upload_id=result.get("upload_id"),
                dataset_version=result.get("dataset_version"),
                semantic_layer=result.get("semantic_layer"),
            )
        return

    # Check if this is a focused single-variable analysis
    if "focused_variable" in result:
        _render_focused_descriptive(result)
        return

    # Full dataset analysis
    # Show breakdown if filters were applied
    if result.get("filters_applied"):
        st.markdown("### Data Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Matching Criteria", f"{result.get('filtered_count', result['row_count']):,}")
        with col2:
            original = result.get("original_count", result["row_count"])
            filtered = result.get("filtered_count", result["row_count"])
            excluded = original - filtered
            st.metric("Excluded", f"{excluded:,}")

        if result.get("filter_description"):
            st.caption(f"**Filters:** {result['filter_description']}")

        st.divider()

    # Only show "Your Data at a Glance" if user explicitly asked for summary/overview
    # Check if query contains summary-related keywords
    is_explicit_summary = False
    if query_text:
        query_lower = query_text.lower()
        summary_keywords = ["summary", "overview", "describe all", "show all", "data at a glance", "general statistics"]
        is_explicit_summary = any(keyword in query_lower for keyword in summary_keywords)

    # If not explicitly requested, show a more focused message
    if not is_explicit_summary:
        st.info(
            "💡 **Tip**: Ask about a specific variable for more detailed analysis, "
            "or say 'show summary' for an overview."
        )
        return

    # User explicitly asked for summary - show full overview
    st.markdown("<h2>📊 Your Data at a Glance</h2>", unsafe_allow_html=True)

    # Overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", f"{result['row_count']:,}")
    with col2:
        st.metric("Variables", result["column_count"])
    with col3:
        st.metric("Data Completeness", f"{100 - result['missing_pct']:.1f}%")

    # Summary statistics (use HTML to avoid Streamlit auto-anchor generation)
    st.markdown("<h3>Summary Statistics</h3>", unsafe_allow_html=True)

    if result["summary_stats"]:
        st.markdown("**Numeric Variables:**")
        # Convert to pandas for display
        # TODO: Future enhancement - use seaborn for styled tables/heatmaps
        # e.g., sns.heatmap(summary_df, annot=True, fmt='.2f') for correlation matrices
        summary_df = pd.DataFrame(result["summary_stats"])
        st.dataframe(summary_df)

    if result["categorical_summary"]:
        st.markdown("**Categorical Variables:**")
        for col, value_counts in result["categorical_summary"].items():
            st.markdown(f"**{col}:**")
            for item in value_counts:
                value = item[col]
                count = item["count"]
                pct = (count / result["row_count"]) * 100 if result["row_count"] > 0 else 0.0
                st.write(f"  - {value}: {count} ({pct:.1f}%)")


def _render_focused_descriptive(result: dict) -> None:
    """Render focused single-variable descriptive analysis."""
    var_name = result["focused_variable"]

    # Show breakdown if filters were applied
    if result.get("filters_applied"):
        st.markdown("### Data Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Matching Criteria", f"{result.get('filtered_count', result['row_count']):,}")
        with col2:
            original = result.get("original_count", result["row_count"])
            filtered = result.get("filtered_count", result["row_count"])
            excluded = original - filtered
            st.metric("Excluded", f"{excluded:,}")

        if result.get("filter_description"):
            st.caption(f"**Filters:** {result['filter_description']}")

        st.divider()

    # Headline answer first!
    if "headline" in result:
        st.info(f"📋 **Answer:** {result['headline']}")

    st.markdown(f"## 📊 Analysis: {var_name}")

    # Data quality metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{result['row_count']:,}")
    with col2:
        st.metric("Valid Values", f"{result['non_null_count']:,}")
    with col3:
        st.metric("Missing", f"{result['null_pct']:.1f}%")

    if result.get("is_numeric"):
        # Numeric variable stats
        st.markdown("### Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Mean", f"{result.get('mean', 0):.2f}")
        with col2:
            st.metric("Median", f"{result.get('median', 0):.2f}")
        with col3:
            st.metric("Std Dev", f"{result.get('std', 0):.2f}")
        with col4:
            st.metric("Min", f"{result.get('min', 0):.2f}")
        with col5:
            st.metric("Max", f"{result.get('max', 0):.2f}")
    else:
        # Categorical variable - show value distribution
        st.markdown("### Value Distribution")
        if result.get("value_counts"):
            for item in result["value_counts"]:
                value = item[var_name]
                count = item["count"]
                pct = (count / result["row_count"]) * 100 if result["row_count"] > 0 else 0.0
                st.write(f"  - **{value}**: {count} ({pct:.1f}%)")


def render_count_analysis(result: dict) -> None:
    """Render count analysis inline."""
    # If grouped, show group breakdown
    if "grouped_by" in result and result.get("group_counts"):
        group_col = result["grouped_by"]
        is_most_query = result.get("is_most_query", False)

        # Get code-to-label mapping for the grouping column
        code_to_label = {}
        try:
            from clinical_analytics.core.column_parser import parse_column_name

            column_meta = parse_column_name(group_col)
            if column_meta.value_mapping:
                code_to_label = column_meta.value_mapping
        except Exception:
            pass  # If parsing fails, just use codes as-is

        # For "most" queries, show only top result with label if available
        if is_most_query and len(result["group_counts"]) > 0:
            top_item = result["group_counts"][0]
            group_value = top_item[group_col]
            count = top_item["count"]

            # Try to map code to label
            display_value = group_value
            if code_to_label:
                # Convert group_value to string for lookup
                value_str = str(group_value)
                if value_str in code_to_label:
                    label = code_to_label[value_str]
                    display_value = f"{label} ({value_str})"

            st.metric("Most Prescribed", f"{display_value}: {count:,}")
        else:
            # Show all groups
            for item in result["group_counts"]:
                group_value = item[group_col]
                count = item["count"]
                pct = (count / result["total_count"]) * 100 if result["total_count"] > 0 else 0.0

                # Try to map code to label
                display_value = group_value
                if code_to_label:
                    value_str = str(group_value)
                    if value_str in code_to_label:
                        label = code_to_label[value_str]
                        display_value = f"{label} ({value_str})"

                st.write(f"- **{display_value}**: {count:,} ({pct:.1f}%)")
    else:
        # Simple total count
        st.metric("Total Count", f"{result['total_count']:,}")


# PANDAS EXCEPTION: Required for Streamlit st.dataframe display
# TODO: Remove when Streamlit supports Polars natively
def render_comparison_analysis(result: dict) -> None:
    """Render comparison analysis from serializable dict inline."""
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown("#### 📈 Group Comparison")

    outcome_col = result["outcome_col"]
    group_col = result["group_col"]
    test_type = result["test_type"]

    # Headline already shown above in chat message, skip duplicate

    if test_type == "t_test":
        groups = result["groups"]
        p_value = result["p_value"]

        st.markdown("**Results:**")
        st.markdown(f"Comparing {outcome_col} between {groups[0]} and {groups[1]}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{groups[0]} Average", f"{result['group1_mean']:.2f}")
        with col2:
            st.metric(f"{groups[1]} Average", f"{result['group2_mean']:.2f}")
        with col3:
            p_interp = ResultInterpreter.interpret_p_value(p_value)
            st.metric("Difference", f"{p_interp['significance']} {p_interp['emoji']}")

        st.markdown("**Interpretation:**")
        interpretation = ResultInterpreter.interpret_mean_difference(
            mean_diff=result["mean_diff"],
            ci_lower=result["ci_lower"],
            ci_upper=result["ci_upper"],
            p_value=p_value,
            group1=str(groups[0]),
            group2=str(groups[1]),
            outcome_name=outcome_col,
        )
        st.markdown(interpretation)

    elif test_type == "anova":
        n_groups = result["n_groups"]
        p_value = result["p_value"]

        st.markdown("### Results")
        st.markdown(f"**Comparing {outcome_col} across {n_groups} groups**")

        p_interp = ResultInterpreter.interpret_p_value(p_value)
        st.metric("Result", f"{p_interp['significance']} {p_interp['emoji']}")

        if p_interp["is_significant"]:
            st.success("✅ The groups differ significantly")
            st.markdown("**Group Averages:**")
            for g, mean_val in result["group_means"].items():
                st.write(f"- {g}: {mean_val:.2f}")
        else:
            st.info("ℹ️ No significant difference between groups")

    elif test_type == "chi_square":
        p_value = result["p_value"]

        st.markdown("### Results")
        st.markdown(f"**Association between {outcome_col} and {group_col}**")

        p_interp = ResultInterpreter.interpret_p_value(p_value)
        st.metric("Result", f"{p_interp['significance']} {p_interp['emoji']}")

        # Show group means if available (for pseudo-numeric categorical data)
        if result.get("group_means"):
            st.markdown("### Group Averages")
            cols = st.columns(min(len(result["group_means"]), 4))
            for idx, (group, mean_val) in enumerate(sorted(result["group_means"].items(), key=lambda x: x[1])):
                with cols[idx % len(cols)]:
                    st.metric(group, f"{mean_val:.1f}")

        st.markdown("### Distribution")
        # Reconstruct contingency table for display
        contingency_list = result["contingency"]
        index_col = result["contingency_index_col"]
        contingency_df = pd.DataFrame(contingency_list)
        contingency_df = contingency_df.set_index(index_col)
        st.dataframe(contingency_df)

        if p_interp["is_significant"]:
            st.success(f"✅ {outcome_col} distribution differs significantly across {group_col} groups")
        else:
            st.info(f"ℹ️ {outcome_col} distribution is similar across groups")


# PANDAS EXCEPTION: Required for Streamlit st.dataframe display
# TODO: Remove when Streamlit supports Polars natively
def render_predictor_analysis(result: dict) -> None:
    """Render predictor analysis from serializable dict."""
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown("## 🎯 Finding Predictors")

    outcome_col = result["outcome_col"]
    st.markdown("### Results")
    st.markdown(f"**What predicts {outcome_col}?**")

    # Show significant predictors with enhanced interpretation
    significant = result["significant_predictors"]
    sample_size = result.get("sample_size")

    if len(significant) > 0:
        st.success(f"✅ Found {len(significant)} significant predictor(s)")

        st.markdown("### 📖 What does this mean?")
        for pred in significant:
            var = pred["variable"]
            or_val = pred["odds_ratio"]
            ci_lower = pred.get("ci_lower", 0.0)
            ci_upper = pred.get("ci_upper", 0.0)
            p_val = pred["p_value"]

            # Get value mapping if available (from column metadata)
            value_mapping = None
            try:
                meta = parse_column_name(var)
                if meta.value_mapping:
                    value_mapping = meta.value_mapping
            except Exception:
                pass

            # Use enhanced interpretation with value mapping and warnings
            interpretation = ResultInterpreter.interpret_odds_ratio(
                or_value=or_val,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                p_value=p_val,
                variable_name=var,
                value_mapping=value_mapping,
                sample_size=sample_size,
            )
            st.markdown(interpretation)
            st.divider()

    else:
        st.info("ℹ️ No significant predictors found at p<0.05")

    # Full results in expander
    with st.expander("📊 Detailed Results"):
        # Reconstruct summary DataFrame for display
        summary_df = pd.DataFrame.from_dict(result["summary"], orient="index")
        summary_df = summary_df.astype(result["schema"], errors="ignore")
        st.dataframe(
            summary_df.style.format(
                {
                    "Odds Ratio": "{:.3f}",
                    "CI Lower": "{:.3f}",
                    "CI Upper": "{:.3f}",
                    "P-Value": "{:.4f}",
                }
            )
        )


# PANDAS EXCEPTION: Required for Streamlit st.dataframe display and matplotlib
# TODO: Remove when Streamlit supports Polars natively
def render_survival_analysis(result: dict) -> None:
    """Render survival analysis from serializable dict."""
    if "error" in result:
        st.error(result["error"])
        if "unique_values" in result:
            st.info(f"Event values found: {result['unique_values']}")
        return

    st.markdown("## ⏱️ Survival Analysis")

    st.markdown("### Results")

    # Reconstruct summary DataFrame for plotting
    summary_df = pd.DataFrame(result["summary"])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary_df["time"], summary_df["survival_probability"], linewidth=2)
    ax.fill_between(summary_df["time"], summary_df["ci_lower"], summary_df["ci_upper"], alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Survival Curve")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Median survival
    median_survival = result["median_survival"]
    st.metric(
        "Median Survival Time",
        f"{median_survival:.1f}" if median_survival is not None else "Not reached",
    )


# PANDAS EXCEPTION: Required for Streamlit st.dataframe display and seaborn
# TODO: Remove when Streamlit supports Polars natively
def render_relationship_analysis(result: dict) -> None:
    """Render relationship analysis from serializable dict with clean, user-friendly output."""
    if "error" in result:
        st.error(f"❌ **Analysis Error**: {result['error']}")
        if "numeric_variables" in result:
            st.info(f"✅ **Numeric variables found**: {', '.join(result['numeric_variables'][:5])}")
        if "non_numeric_variables" in result:
            excluded = result["non_numeric_variables"][:5]
            st.info(f"ℹ️ **Non-numeric variables excluded** (cannot compute correlations): {', '.join(excluded)}")
        return

    # Show summary info
    n_obs = result.get("n_observations", 0)
    variables = result["variables"]
    non_numeric_excluded = result.get("non_numeric_excluded", [])

    # Header with key info
    st.markdown("### 🔗 Variable Relationships")

    # Show what was analyzed
    st.markdown(f"**Analyzed {len(variables)} numeric variable(s)** across **{n_obs} observations**")

    # Warn if variables were excluded
    if non_numeric_excluded:
        excluded_preview = ", ".join(non_numeric_excluded[:3])
        if len(non_numeric_excluded) > 3:
            excluded_preview += f" and {len(non_numeric_excluded) - 3} more"
        st.info(f"ℹ️ **Note**: {excluded_preview} were excluded (text/categorical variables cannot be correlated)")

    # Get correlations
    strong_correlations = result.get("strong_correlations", [])
    moderate_correlations = result.get("moderate_correlations", [])
    corr_data = result.get("correlations", [])

    # Show key findings first (strong correlations)
    if strong_correlations:
        st.markdown("#### 📊 Key Findings")
        for item in strong_correlations:
            var1 = item["var1"]
            var2 = item["var2"]
            corr_val = item["correlation"]

            # Clean up variable names for display
            var1_clean = var1.split(":")[0].strip() if ":" in var1 else var1
            var2_clean = var2.split(":")[0].strip() if ":" in var2 else var2

            direction = "increases together" if corr_val > 0 else "increases as the other decreases"
            strength = "strongly" if abs(corr_val) >= 0.7 else "moderately"

            st.markdown(f"• **{var1_clean}** and **{var2_clean}** {strength} {direction} (correlation: {corr_val:.2f})")
    elif moderate_correlations:
        st.markdown("#### 📊 Findings")
        st.info("No strong relationships found, but some moderate relationships exist:")
        for item in moderate_correlations[:5]:  # Show top 5
            var1 = item["var1"]
            var2 = item["var2"]
            corr_val = item["correlation"]

            var1_clean = var1.split(":")[0].strip() if ":" in var1 else var1
            var2_clean = var2.split(":")[0].strip() if ":" in var2 else var2

            direction = "positive" if corr_val > 0 else "negative"
            st.markdown(f"• **{var1_clean}** ↔ **{var2_clean}**: {direction} (r={corr_val:.2f})")
    else:
        st.info("ℹ️ No strong or moderate correlations found between the analyzed variables.")

    # Optional: Show correlation matrix in expander (less prominent)
    if len(variables) <= 6 and corr_data:  # Only show for small number of variables
        with st.expander("📈 View Correlation Matrix", expanded=False):
            # Reconstruct correlation matrix for heatmap
            corr_matrix = pd.DataFrame(index=variables, columns=variables)
            for item in corr_data:
                corr_matrix.loc[item["var1"], item["var2"]] = item["correlation"]
                corr_matrix.loc[item["var2"], item["var1"]] = item["correlation"]  # Symmetric
            corr_matrix = corr_matrix.astype(float)

            # Simple heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            import seaborn as sns

            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)


def render_analysis_by_type(result: dict, intent: AnalysisIntent, query_text: str | None = None) -> None:
    """Route to appropriate render function based on result type."""
    result_type = result.get("type")

    if result_type == "descriptive":
        render_descriptive_analysis(result, query_text=query_text)
    elif result_type == "comparison":
        render_comparison_analysis(result)
    elif result_type == "predictor":
        render_predictor_analysis(result)
    elif result_type == "survival":
        render_survival_analysis(result)
    elif result_type == "relationship":
        render_relationship_analysis(result)
    elif result_type == "count":
        render_count_analysis(result)
    else:
        st.error(f"Unknown result type: {result_type}")
        if "error" in result:
            st.error(result["error"])


def render_chat(dataset_version: str, cohort: pl.DataFrame) -> None:
    """
    Render entire chat transcript from session_state (no side effects).

    This is the ONLY function that calls st.chat_message(). All chat rendering
    goes through this function to ensure transcript-driven rendering.

    Args:
        dataset_version: Dataset version for retrieving results
        cohort: Cohort data (for Trust UI in render_result)
    """
    chat: list[ChatMessage] = st.session_state.get("chat", [])

    for message in chat:
        role = message["role"]
        text = message["text"]
        run_key = message.get("run_key")
        status = message.get("status", "completed")

        with st.chat_message(role):
            if role == "user":
                # User message - just display text
                st.write(text)
            elif role == "assistant":
                # Assistant message - render result
                if status == "pending":
                    st.info("💭 Thinking...")
                elif status == "error":
                    st.error(f"❌ Error: {text}")
                elif status == "completed" and run_key:
                    # Load result from session_state
                    result_key = f"analysis_result:{dataset_version}:{run_key}"
                    if result_key in st.session_state:
                        result = st.session_state[result_key]

                        # Reconstruct context from result (minimal, for rendering)
                        # Note: Full context not stored in transcript for memory efficiency
                        # We only need intent for rendering
                        intent_str = result.get("intent", "DESCRIBE")
                        try:
                            intent_enum = AnalysisIntent(intent_str)
                        except (ValueError, KeyError):
                            intent_enum = AnalysisIntent.DESCRIBE

                        # Create minimal context for rendering
                        context = AnalysisContext(inferred_intent=intent_enum)

                        # Render result (idempotent, no side effects)
                        render_result(
                            result=result,
                            context=context,
                            run_key=run_key,
                            query_text=text,
                            query_plan=None,  # Not available from transcript
                            cohort=None,  # Not available for cached results
                            dataset_version=dataset_version,
                        )
                    else:
                        st.warning("⚠️ Result not found in cache")
                else:
                    # Fallback: just display text
                    st.write(text)


def get_or_compute_result(
    cohort: pl.DataFrame,
    context: AnalysisContext,
    run_key: str,
    dataset_version: str,
    query_text: str,
) -> dict:
    """
    Get cached result or compute new one (pure computation, no UI).

    Args:
        cohort: Polars DataFrame with patient data
        context: Analysis context with variables
        run_key: Run key for this analysis
        dataset_version: Dataset version identifier
        query_text: Original query text

    Returns:
        Analysis result dict (serializable)
    """
    # Use dataset-scoped result key
    result_key = f"analysis_result:{dataset_version}:{run_key}"

    # Check if already computed
    if result_key in st.session_state:
        logger.info(
            "analysis_result_cached",
            run_key=run_key,
            dataset_version=dataset_version,
            intent_type=context.inferred_intent.value,
        )
        return st.session_state[result_key]

    # Not computed - compute now
    logger.info(
        "analysis_execution_start",
        intent=context.inferred_intent.value,
        primary_variable=context.primary_variable,
        grouping_variable=context.grouping_variable,
        predictor_variables=context.predictor_variables,
        dataset_version=dataset_version,
        confidence=getattr(context, "confidence", 0.0),
    )

    # Convert cohort to Polars if needed (defensive)
    if isinstance(cohort, pd.DataFrame):
        cohort_pl = pl.from_pandas(cohort)
    else:
        cohort_pl = cohort

    # Compute analysis (pure function, no UI dependencies)
    logger.debug(
        "analysis_computation_start",
        run_key=run_key,
        cohort_shape=(cohort_pl.height, cohort_pl.width),
        intent_type=context.inferred_intent.value,
    )
    result = compute_analysis_by_type(cohort_pl, context)

    logger.info(
        "analysis_computation_complete",
        run_key=run_key,
        result_type=result.get("type", "unknown"),
        has_error="error" in result,
        dataset_version=dataset_version,
    )

    # Store result (serializable format)
    st.session_state[result_key] = result
    st.session_state[f"last_run_key:{dataset_version}"] = run_key

    # Remember this run in history (O(1) eviction happens here)
    remember_run(dataset_version, run_key)

    return result


def render_result(
    result: dict,
    context: AnalysisContext,
    run_key: str,
    query_text: str,
    query_plan=None,
    cohort: pl.DataFrame | None = None,
    dataset_version: str | None = None,
) -> None:
    """
    Render analysis result (pure rendering, no computation, no side effects).

    This function is idempotent - can be called multiple times with same inputs.
    It only reads from session_state and creates UI widgets. No mutations.

    Args:
        result: Analysis result dict
        context: Analysis context
        run_key: Run key for this analysis
        query_text: Original query text
        query_plan: QueryPlan object (if available)
        cohort: Cohort data (for Trust UI, optional)
        dataset_version: Dataset version (for Trust UI, optional)
    """
    # Render main results inline (no expander, no extra headers)
    render_analysis_by_type(result, context.inferred_intent, query_text=query_text)

    # Show interpretation inline (compact, directly under results)
    if query_plan:
        _render_interpretation_inline_compact(query_plan)

    # ADR003 Phase 1: Trust UI (verify source patients, patient-level export)
    # Only show if cohort and dataset_version are provided (fresh computation)
    if query_plan and cohort is not None and dataset_version is not None:
        from clinical_analytics.ui.components.trust_ui import TrustUI

        TrustUI.render_verification(
            query_plan=query_plan,
            result=result,
            cohort=cohort,
            dataset_version=dataset_version,
            query_text=query_text,
        )
    elif query_plan:
        st.info("ℹ️ Trust UI only available for fresh computations (not cached results)")

    # Suggest follow-up questions (only for current results, not history)
    # Use a session state flag to prevent duplicate rendering in same cycle
    follow_ups_key = f"followups_rendered_{run_key}"
    if not st.session_state.get(follow_ups_key, False):
        _suggest_follow_ups(context, result, run_key, render_context="current")
        st.session_state[follow_ups_key] = True


def execute_analysis_with_idempotency(
    cohort: pl.DataFrame,
    context: AnalysisContext,
    run_key: str,
    dataset_version: str,
    query_text: str,
    query_plan=None,
) -> None:
    """
    Execute analysis with idempotency guard and result persistence.

    Checks if result already exists in session_state. If yes, renders from cache.
    If no, computes, stores, and renders.
    """
    # Use dataset-scoped result key for trivial cleanup
    result_key = f"analysis_result:{dataset_version}:{run_key}"

    # Check if already computed
    if result_key in st.session_state:
        logger.info(
            "analysis_result_cached",
            run_key=run_key,
            dataset_version=dataset_version,
            intent_type=context.inferred_intent.value,
        )
        # Render inline in chat message style (Phase 3.5)
        result = st.session_state[result_key]

        # Render main results inline (no expander, no extra headers)
        render_analysis_by_type(result, context.inferred_intent, query_text=query_text)

        # Show interpretation inline (compact, directly under results)
        if query_plan:
            _render_interpretation_inline_compact(query_plan)

        # ADR003 Phase 1: Trust UI (verify source patients, patient-level export)
        # Note: cohort not available in cached path, skip trust UI for cached results
        # Trust UI will be shown for fresh computations only
        if query_plan:
            st.info("ℹ️ Trust UI only available for fresh computations (not cached results)")

        # Suggest follow-up questions (only for current results, not history)
        # Use a session state flag to prevent duplicate rendering in same cycle
        follow_ups_key = f"followups_rendered_{run_key}"
        if not st.session_state.get(follow_ups_key, False):
            _suggest_follow_ups(context, result, run_key, render_context="current")
            st.session_state[follow_ups_key] = True
        return

    # Not computed - compute and store
    with st.spinner("Running analysis..."):
        # Log execution start for observability (logger already defined at module level)
        logger.info(
            "analysis_execution_start",
            intent=context.inferred_intent.value,
            primary_variable=context.primary_variable,
            grouping_variable=context.grouping_variable,
            predictor_variables=context.predictor_variables,
            dataset_version=dataset_version,
            confidence=getattr(context, "confidence", 0.0),
        )

        # Convert cohort to Polars if needed (defensive)
        if isinstance(cohort, pd.DataFrame):
            cohort_pl = pl.from_pandas(cohort)
        else:
            cohort_pl = cohort

        # Compute analysis (pure function, no UI dependencies)
        logger.debug(
            "analysis_computation_start",
            run_key=run_key,
            cohort_shape=(cohort_pl.height, cohort_pl.width),
            intent_type=context.inferred_intent.value,
        )
        result = compute_analysis_by_type(cohort_pl, context)

        logger.info(
            "analysis_computation_complete",
            run_key=run_key,
            result_type=result.get("type", "unknown"),
            has_error="error" in result,
            dataset_version=dataset_version,
        )

        # Store result (serializable format)
        st.session_state[result_key] = result
        st.session_state[f"last_run_key:{dataset_version}"] = run_key

        # Remember this run in history (O(1) eviction happens here)
        remember_run(dataset_version, run_key)

        # Add to conversation history (lightweight storage per ADR001)
        query_text = query_text or getattr(context, "research_question", "")
        headline = result.get("headline") or result.get("headline_text") or "Analysis completed"
        filters_applied = []
        if context.query_plan and context.query_plan.filters:
            filters_applied = [f.__dict__ for f in context.query_plan.filters]
        elif context.filters:
            filters_applied = [f.__dict__ for f in context.filters]

        # Ensure conversation history exists (don't reset if it already exists)
        if "conversation_history" not in st.session_state:
            st.session_state["conversation_history"] = []

        # Add entry (lightweight: headline, not full result dict)
        # Limit to last 20 queries to prevent memory bloat
        max_conversation_history = 20
        st.session_state["conversation_history"].append(
            {
                "query": query_text,
                "intent": context.inferred_intent.value if context.inferred_intent else "UNKNOWN",
                "headline": headline,
                "run_key": run_key,
                "timestamp": time.time(),
                "filters_applied": filters_applied,
            }
        )

        # Limit history size (keep last N entries)
        if len(st.session_state["conversation_history"]) > max_conversation_history:
            st.session_state["conversation_history"] = st.session_state["conversation_history"][
                -max_conversation_history:
            ]

        logger.info(
            "analysis_result_stored",
            run_key=run_key,
            dataset_version=dataset_version,
        )

        # Render inline in chat message style (Phase 3.5)
        logger.debug("analysis_rendering_start", run_key=run_key)

        # Render main results inline (no expander, no extra headers)
        render_analysis_by_type(result, context.inferred_intent, query_text=query_text)

        # Show interpretation inline (compact, directly under results)
        if query_plan:
            _render_interpretation_inline_compact(query_plan)

        # ADR003 Phase 1: Trust UI (verify source patients, patient-level export)
        if query_plan:
            from clinical_analytics.ui.components.trust_ui import TrustUI

            TrustUI.render_verification(
                query_plan=query_plan,
                result=result,
                cohort=cohort_pl,
                dataset_version=dataset_version,
                query_text=query_text,
            )

        # Suggest follow-up questions (only for current results, not history)
        # Use a session state flag to prevent duplicate rendering in same cycle
        follow_ups_key = f"followups_rendered_{run_key}"
        if not st.session_state.get(follow_ups_key, False):
            _suggest_follow_ups(context, result, run_key, render_context="current")
            st.session_state[follow_ups_key] = True

        logger.info("analysis_rendering_complete", run_key=run_key)

        # Don't rerun - results are already rendered inline, conversation history will show on next query


def _render_confirmation_ui(
    query_plan,
    failure_reason: str,
    confidence: float,
    threshold: float,
    dataset_version: str,
) -> None:
    """
    Render confirmation UI when confidence or completeness gating fails (ADR003 Phase 3).

    Shows the QueryPlan details, confidence score, and requires explicit user confirmation
    before execution.
    """
    from clinical_analytics.core.query_plan import QueryPlan

    if not isinstance(query_plan, QueryPlan):
        return

    st.warning("⚠️ **Execution requires confirmation**")

    # Show failure reason
    st.info(f"**Reason:** {failure_reason}")

    # Show QueryPlan details
    with st.expander("📋 Query Plan Details", expanded=True):
        st.write(f"**Intent:** {query_plan.intent}")
        if query_plan.metric:
            st.write(f"**Metric:** {query_plan.metric}")
        if query_plan.group_by:
            st.write(f"**Group By:** {query_plan.group_by}")
        if query_plan.filters:
            st.write("**Filters:**")
            for f in query_plan.filters:
                st.write(f"  - {f.column} {f.operator} {f.value}")
        st.write(f"**Confidence:** {confidence:.0%} (threshold: {threshold:.0%})")
        if query_plan.explanation:
            st.write(f"**Explanation:** {query_plan.explanation}")

    # Confirmation button
    if st.button("✅ Confirm and Run", key=f"confirm_execution_{dataset_version}", type="primary"):
        st.session_state[f"confirmed_execution_{dataset_version}"] = True
        st.rerun()


def _render_interpretation_inline_compact(query_plan) -> None:
    """Render compact interpretation inline directly under results."""
    confidence = query_plan.confidence

    # Show compact confidence badge
    if confidence >= 0.75:
        st.caption(f"✓ High confidence ({confidence:.0%})")
    elif confidence >= 0.5:
        st.caption(f"⚠ Moderate confidence ({confidence:.0%})")
    else:
        st.caption(f"⚠ Low confidence ({confidence:.0%})")

    # Show compact interpretation (only if low confidence or has filters)
    if confidence < 0.75 or query_plan.filters:
        interpretation_parts = []
        if query_plan.group_by:
            interpretation_parts.append(f"grouped by {query_plan.group_by}")
        if query_plan.filters:
            filter_text = ", ".join([f"{f.column} {f.operator} {f.value}" for f in query_plan.filters[:2]])
            if len(query_plan.filters) > 2:
                filter_text += f" (+{len(query_plan.filters) - 2} more)"
            interpretation_parts.append(f"filters: {filter_text}")
        if interpretation_parts:
            st.caption(f"Interpreted: {', '.join(interpretation_parts)}")

    logger.info(
        "confidence_displayed",
        confidence=confidence,
        intent=query_plan.intent,
        has_filters=len(query_plan.filters) > 0,
        has_explanation=bool(query_plan.explanation),
    )


def _render_interpretation_and_confidence(query_plan, result: dict) -> None:
    """
    Show transparent interpretation and confidence inline with results.

    This replaces blocking confirmation gates with data-driven transparency:
    - Always execute regardless of confidence
    - Show what was interpreted
    - Display confidence level with appropriate visual indicator
    - Explain why confidence might be low (if applicable)
    """
    from clinical_analytics.core.query_plan import QueryPlan

    if not isinstance(query_plan, QueryPlan):
        return

    st.divider()
    st.markdown("### 🔍 Interpretation & Confidence")

    # Show confidence badge with appropriate visual indicator
    confidence = query_plan.confidence
    if confidence >= 0.75:
        st.success(f"✅ **High confidence** ({confidence:.0%})")
    elif confidence >= 0.5:
        st.warning(f"⚠️ **Moderate confidence** ({confidence:.0%})")
    else:
        st.error(f"⚠️ **Low confidence** ({confidence:.0%})")

    # Show what we interpreted (collapsible, expanded for low confidence)
    expanded = confidence < 0.75
    with st.expander("How I interpreted your question", expanded=expanded):
        st.markdown(f"**Intent:** `{query_plan.intent}`")

        if query_plan.metric:
            st.markdown(f"**Metric:** `{query_plan.metric}`")

        if query_plan.group_by:
            st.markdown(f"**Group by:** `{query_plan.group_by}`")

        if query_plan.filters:
            st.markdown("**Filters applied:**")
            for f in query_plan.filters:
                op_text = {
                    "==": "equals",
                    "!=": "does not equal",
                    "<": "less than",
                    ">": "greater than",
                    "<=": "at most",
                    ">=": "at least",
                    "IN": "in",
                    "NOT_IN": "not in",
                }.get(f.operator, f.operator)
                st.markdown(f"- `{f.column}` {op_text} `{f.value}`")

        # Show explanation (why confidence is what it is)
        if query_plan.explanation:
            st.info(f"**Note:** {query_plan.explanation}")

    # Log confidence for analytics (helps identify improvement opportunities)
    logger.info(
        "confidence_displayed",
        confidence=confidence,
        intent=query_plan.intent,
        has_filters=len(query_plan.filters) > 0,
        has_explanation=bool(query_plan.explanation),
    )


def _suggest_follow_ups(
    context: AnalysisContext, result: dict, run_key: str | None = None, render_context: str = "current"
) -> None:
    """
    Suggest natural follow-up questions based on current result.

    Args:
        context: Analysis context
        result: Analysis result dict
        run_key: Run key for this analysis
        render_context: Context identifier ("current" or "history") to ensure unique button keys
    """
    suggestions = []

    # Generate suggestions based on intent and result
    if context.inferred_intent == AnalysisIntent.DESCRIBE:
        if context.primary_variable and context.primary_variable != "all":
            # Suggest comparing this variable
            suggestions.append(f"Compare {context.primary_variable} by treatment group")
            suggestions.append(f"What predicts {context.primary_variable}?")
        else:
            # No specific variable - suggest exploring
            suggestions.append("Compare outcomes by treatment group")
            suggestions.append("What predicts the outcome?")

    elif context.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
        if context.grouping_variable and context.primary_variable:
            # Suggest deeper analysis
            suggestions.append(f"What else affects {context.primary_variable}?")
            suggestions.append(f"Show distribution of {context.primary_variable} by {context.grouping_variable}")
        if context.primary_variable:
            suggestions.append(f"Describe {context.primary_variable} in detail")

    elif context.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
        if context.primary_variable:
            # Suggest comparing predictors
            suggestions.append(f"Compare {context.primary_variable} by the strongest predictor")
            suggestions.append(f"Describe {context.primary_variable} in detail")

    elif context.inferred_intent == AnalysisIntent.COUNT:
        # Suggest grouping or filtering
        if context.grouping_variable:
            suggestions.append(f"Filter {context.grouping_variable} and count again")
        else:
            suggestions.append("Break down the count by a grouping variable")
        if context.filters or (context.query_plan and context.query_plan.filters):
            suggestions.append("Remove filters and count all records")

    elif context.inferred_intent == AnalysisIntent.EXAMINE_SURVIVAL:
        if context.primary_variable:
            suggestions.append(f"Compare survival by {context.primary_variable}")
        suggestions.append("What predicts survival time?")

    elif context.inferred_intent == AnalysisIntent.EXPLORE_RELATIONSHIPS:
        if context.predictor_variables:
            second_var = context.predictor_variables[1] if len(context.predictor_variables) > 1 else "outcome"
            suggestions.append(f"Compare {context.predictor_variables[0]} by {second_var}")

    # Render suggestions as compact buttons (only if we have suggestions)
    if suggestions:
        st.markdown("**💡 You might also ask:**")
        # Use columns for compact layout
        cols = st.columns(min(len(suggestions), 2))
        for idx, suggestion in enumerate(suggestions[:4]):  # Limit to 4 suggestions
            col = cols[idx % len(cols)]
            with col:
                # Include run_key and render_context in button key to ensure uniqueness
                # This prevents StreamlitDuplicateElementKey errors when same suggestion appears in different contexts
                # Use sanitized run_key (not hash) to prevent collisions while keeping it readable
                # Sanitize run_key to remove special characters that might cause issues
                run_key_safe = (run_key or "default").replace(":", "_").replace("/", "_").replace("-", "_")[:30]
                suggestion_hash = hash(suggestion) % 1000000
                # Use render_context to distinguish between current result and history rendering
                # Create a unique key that combines all these elements
                button_key = f"followup_{render_context}_{run_key_safe}_{idx}_{suggestion_hash}"
                if st.button(suggestion, key=button_key, use_container_width=True):
                    # Store suggestion to prefill chat input and rerun to process it
                    st.session_state["prefilled_query"] = suggestion
                    st.rerun()


def get_dataset_version(dataset, dataset_choice: str) -> str:
    """
    Get stable dataset version identifier for caching and lifecycle management.

    All datasets are uploaded datasets, so use upload_id as version.
    """
    # All datasets are uploaded datasets: use upload_id as version
    return dataset_choice  # This is the upload_id


def main():
    st.title("💬 Ask Questions")
    st.markdown("""
    Ask questions about your data in plain English. I'll figure out the right analysis and explain the results.
    """)

    # Dataset selection
    st.sidebar.header("Data Selection")

    # Load datasets - only user uploads
    dataset_display_names = {}
    uploaded_datasets = {}
    try:
        uploads = UploadedDatasetFactory.list_available_uploads()
        for upload in uploads:
            upload_id = upload["upload_id"]
            dataset_name = upload.get("dataset_name", upload_id)
            display_name = f"📤 {dataset_name}"
            dataset_display_names[display_name] = upload_id
            uploaded_datasets[upload_id] = upload
    except Exception as e:
        st.sidebar.warning(f"Could not load uploaded datasets: {e}")

    if not dataset_display_names:
        st.error("No datasets found! Please upload data using the 'Add Your Data' page.")
        st.info("👈 Go to **Add Your Data** to upload your first dataset")
        return

    dataset_choice_display = st.sidebar.selectbox("Choose Dataset", list(dataset_display_names.keys()))
    dataset_choice = dataset_display_names[dataset_choice_display]

    # Initialize chat transcript and pending state (Phase 2)
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    if "pending" not in st.session_state:
        st.session_state["pending"] = None

    # Detect dataset change and clear conversation history/context/chat
    if "last_dataset_choice" not in st.session_state:
        st.session_state["last_dataset_choice"] = dataset_choice
    elif st.session_state["last_dataset_choice"] != dataset_choice:
        # Dataset changed - clear conversation history, analysis context, and chat
        if "conversation_history" in st.session_state:
            st.session_state["conversation_history"] = []
        if "analysis_context" in st.session_state:
            del st.session_state["analysis_context"]
        # Clear chat transcript on dataset change
        st.session_state["chat"] = []
        st.session_state["pending"] = None
        st.session_state["last_dataset_choice"] = dataset_choice

    # Load dataset (always uploaded)
    with st.spinner(f"Loading {dataset_choice_display}..."):
        try:
            dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
            dataset.load()
            cohort_pd = dataset.get_cohort()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

    # Convert to Polars for compute functions
    cohort = pl.from_pandas(cohort_pd)
    cohort_pl = cohort  # Alias for clarity in Phase 2

    # Get dataset version for lifecycle management
    dataset_version = get_dataset_version(dataset, dataset_choice)

    # Show Semantic Scope in sidebar
    with st.sidebar.expander("🔍 Semantic Scope", expanded=False):
        st.markdown("**V1 Cohort-First Mode**")

        # Cohort table status
        st.markdown(f"✅ **Cohort Table**: {cohort.height:,} rows")

        # Multi-table status
        if MULTI_TABLE_ENABLED:
            st.markdown("⚠️ **Multi-Table**: Experimental")
        else:
            st.markdown("⏸️ **Multi-Table**: Disabled (V2)")

        # Detected grain
        grain = "patient_level"  # Default for V1
        st.markdown(f"📊 **Grain**: {grain}")

        # Outcome column (if detected)
        outcome_cols = [
            c for c in cohort.columns if "outcome" in c.lower() or "death" in c.lower() or "mortality" in c.lower()
        ]
        if outcome_cols:
            st.markdown(f"🎯 **Outcome**: `{outcome_cols[0]}`")
        else:
            st.markdown("🎯 **Outcome**: Not specified")

        # Show column count
        st.caption(f"{cohort.width} columns available")

    # Conversation management in sidebar - Phase 3.4 Remove Buttons
    with st.sidebar:
        st.markdown("### Conversation")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state["conversation_history"] = []
            st.session_state["analysis_context"] = None
            st.session_state["intent_signal"] = None
            st.rerun()

    st.divider()

    # Initialize session state for context
    if "analysis_context" not in st.session_state:
        st.session_state["analysis_context"] = None
        st.session_state["intent_signal"] = None

    # Initialize conversation history (lightweight storage per ADR001 - keep for backward compat)
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []

    # Render chat transcript (Phase 2: transcript-driven rendering)
    # This is the ONLY place where chat messages are rendered
    # Fixes empty emoji tiles by rendering from persistent state, not control flow
    render_chat(dataset_version=dataset_version, cohort=cohort_pl)

    # Check for semantic layer availability (show message if not available)
    # Phase 3: Use cached semantic layer for performance
    try:
        semantic_layer = get_cached_semantic_layer(dataset_version, dataset)
    except ValueError:
        # Semantic layer not available
        st.info("Natural language queries are only available for datasets with semantic layers.")
        # Still show chat input but it will fail gracefully
        semantic_layer = None

    # Handle analysis execution if we have a context ready
    if st.session_state.get("intent_signal") is not None:
        # We have intent, now gather details
        context = st.session_state["analysis_context"]

        logger.info(
            "page_render_analysis_configuration",
            intent_signal=st.session_state.get("intent_signal"),
            intent_type=context.inferred_intent.value if context else None,
            is_complete=context.is_complete_for_intent() if context else False,
            dataset_version=get_dataset_version(dataset, dataset_choice),
        )

        st.divider()

        # Ask follow-up questions based on intent
        if context.inferred_intent == AnalysisIntent.DESCRIBE:
            # Only default to "all" if no specific variable was requested
            if not context.primary_variable:
                context.primary_variable = "all"

        elif context.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
            if not context.primary_variable:
                context.primary_variable = QuestionEngine.select_primary_variable(
                    cohort, context, "What do you want to compare?"
                )

            if context.primary_variable and not context.grouping_variable:
                context.grouping_variable = QuestionEngine.select_grouping_variable(
                    cohort, exclude=[context.primary_variable]
                )

        elif context.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
            if not context.primary_variable:
                context.primary_variable = QuestionEngine.select_primary_variable(
                    cohort, context, "What outcome do you want to predict?"
                )

            if context.primary_variable and not context.predictor_variables:
                context.predictor_variables = QuestionEngine.select_predictor_variables(
                    cohort, exclude=[context.primary_variable], min_vars=1
                )

        elif context.inferred_intent == AnalysisIntent.EXAMINE_SURVIVAL:
            if not context.time_variable or not context.event_variable:
                context.time_variable, context.event_variable = QuestionEngine.select_time_variables(cohort)

        elif context.inferred_intent == AnalysisIntent.EXPLORE_RELATIONSHIPS:
            # For CORRELATIONS, NLU may extract variables as primary_variable + grouping_variable
            # OR as predictor_variables. If we have primary+grouping but not enough predictors, use those.
            if len(context.predictor_variables) < 2:
                # Check if we have primary + grouping variables from NLU extraction
                if context.primary_variable and context.grouping_variable:
                    # Merge primary and grouping with existing predictor_variables (avoiding duplicates)
                    existing = set(context.predictor_variables)
                    if context.primary_variable not in existing:
                        context.predictor_variables.append(context.primary_variable)
                    if context.grouping_variable not in existing:
                        context.predictor_variables.append(context.grouping_variable)
                elif not context.primary_variable and not context.grouping_variable:
                    # No variables extracted by NLU - ask user to select
                    context.predictor_variables = QuestionEngine.select_predictor_variables(
                        cohort, exclude=[], min_vars=2
                    )

        # Update context in session state
        st.session_state["analysis_context"] = context

        # Show progress (only if missing info - removed "I have everything" message)
        missing = context.get_missing_info()
        if missing:
            st.divider()
            QuestionEngine.render_progress_indicator(context)

        # If complete, check confidence gating (ADR003 Phase 3)
        if context.is_complete_for_intent():
            # Get QueryPlan if available (preferred), otherwise fallback to context
            query_plan = getattr(context, "query_plan", None)

            # Get confidence from QueryPlan or context (default to 0.0 if missing - fail closed)
            if query_plan:
                confidence = query_plan.confidence
                # Normalize query before generating run_key
                raw_query = getattr(context, "research_question", "")
                normalized_query = normalize_query(raw_query)
                run_key = query_plan.run_key or generate_run_key(dataset_version, normalized_query, context)
            else:
                confidence = getattr(context, "confidence", 0.0)
                # Normalize query before generating run_key
                raw_query = getattr(context, "research_question", "")
                normalized_query = normalize_query(raw_query)
                run_key = generate_run_key(dataset_version, normalized_query, context)

            logger.info(
                "analysis_execution_triggered",
                intent_type=context.inferred_intent.value,
                confidence=confidence,
                threshold=AUTO_EXECUTE_CONFIDENCE_THRESHOLD,
                run_key=run_key,
                dataset_version=dataset_version,
                query=getattr(context, "research_question", ""),
            )

            # ADR003 Phase 3: Use execute_query_plan() for confidence and completeness gating
            # semantic_layer is already available from line 1291
            if query_plan and semantic_layer:
                query_text = getattr(context, "research_question", "")
                execution_result = semantic_layer.execute_query_plan(
                    query_plan, confidence_threshold=AUTO_EXECUTE_CONFIDENCE_THRESHOLD, query_text=query_text
                )

                if execution_result.get("requires_confirmation"):
                    # Gate failed - show confirmation UI
                    st.divider()
                    _render_confirmation_ui(
                        query_plan,
                        execution_result["failure_reason"],
                        confidence,
                        AUTO_EXECUTE_CONFIDENCE_THRESHOLD,
                        dataset_version,
                    )
                elif execution_result.get("success"):
                    # Gate passed - execute analysis
                    st.divider()
                    # Update run_key from execution result if provided
                    if execution_result.get("run_key"):
                        run_key = execution_result["run_key"]
                    execute_analysis_with_idempotency(cohort, context, run_key, dataset_version, query_text, query_plan)
                else:
                    # Execution failed - show error
                    st.error(f"Execution failed: {execution_result.get('failure_reason', 'Unknown error')}")
            else:
                # Fallback: No QueryPlan or semantic_layer - use old path (for backward compatibility)
                # But still check confidence threshold
                if confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD:
                    st.divider()
                    execute_analysis_with_idempotency(
                        cohort, context, run_key, dataset_version, getattr(context, "research_question", ""), query_plan
                    )
                else:
                    st.divider()
                    threshold_pct = f"{AUTO_EXECUTE_CONFIDENCE_THRESHOLD:.0%}"
                    st.warning(
                        f"⚠️ **Low confidence** ({confidence:.0%}) - below threshold ({threshold_pct}). "
                        "Please refine your question or confirm execution."
                    )
                    if st.button("Confirm and Run Anyway", key=f"confirm_low_confidence_{run_key}"):
                        execute_analysis_with_idempotency(
                            cohort,
                            context,
                            run_key,
                            dataset_version,
                            getattr(context, "research_question", ""),
                            query_plan,
                        )

        else:
            # Context not complete - show variable selection UI (for missing required variables)
            st.warning(LOW_CONFIDENCE_WARNING)

            # Helper to get display name for a column
            # Note: parse_column_name() doesn't require semantic layer, so no check needed
            def get_display_name(canonical_name: str) -> str:
                """Get display name for a column, falling back to canonical if parsing fails."""
                try:
                    meta = parse_column_name(canonical_name)
                    return meta.display_name
                except Exception:
                    return canonical_name

            # Show detected variables with display names and editable selectors
            st.markdown("### Detected Variables")

            available_cols = [c for c in cohort.columns if c not in ["patient_id", "time_zero"]]

            # Primary variable
            if context.inferred_intent in [AnalysisIntent.COMPARE_GROUPS, AnalysisIntent.FIND_PREDICTORS]:
                primary_display = get_display_name(context.primary_variable) if context.primary_variable else None
                primary_index = (
                    available_cols.index(context.primary_variable) if context.primary_variable in available_cols else 0
                )
                selected_primary = st.selectbox(
                    "**Primary Variable** (what you want to measure/compare):",
                    options=available_cols,
                    index=primary_index if primary_index < len(available_cols) else 0,
                    key=f"low_conf_primary_{dataset_version}",
                    help=f"Detected: {primary_display or context.primary_variable}"
                    if context.primary_variable
                    else None,
                )
                context.primary_variable = selected_primary

            # Grouping variable (for comparisons)
            if context.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
                grouping_display = get_display_name(context.grouping_variable) if context.grouping_variable else None
                grouping_index = (
                    available_cols.index(context.grouping_variable)
                    if context.grouping_variable in available_cols
                    else 0
                )
                exclude_primary = [context.primary_variable] if context.primary_variable else []
                grouping_options = [c for c in available_cols if c not in exclude_primary]
                selected_grouping = st.selectbox(
                    "**Grouping Variable** (groups to compare):",
                    options=grouping_options,
                    index=min(grouping_index, len(grouping_options) - 1)
                    if context.grouping_variable in grouping_options
                    else 0,
                    key=f"low_conf_grouping_{dataset_version}",
                    help=f"Detected: {grouping_display or context.grouping_variable}"
                    if context.grouping_variable
                    else None,
                )
                context.grouping_variable = selected_grouping

            # Predictor variables (for regression)
            if context.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
                predictor_display = (
                    [get_display_name(p) for p in context.predictor_variables] if context.predictor_variables else []
                )
                exclude_primary = [context.primary_variable] if context.primary_variable else []
                predictor_options = [c for c in available_cols if c not in exclude_primary]
                selected_predictors = st.multiselect(
                    "**Predictor Variables** (what might affect the outcome):",
                    options=predictor_options,
                    default=context.predictor_variables if context.predictor_variables else [],
                    key=f"low_conf_predictors_{dataset_version}",
                    help=f"Detected: {', '.join(predictor_display) if predictor_display else 'None'}",
                )
                context.predictor_variables = selected_predictors

            # Time and event variables (for survival)
            if context.inferred_intent == AnalysisIntent.EXAMINE_SURVIVAL:
                col1, col2 = st.columns(2)
                with col1:
                    time_display = get_display_name(context.time_variable) if context.time_variable else None
                    time_index = (
                        available_cols.index(context.time_variable) if context.time_variable in available_cols else 0
                    )
                    selected_time = st.selectbox(
                        "**Time Variable**:",
                        options=available_cols,
                        index=time_index if time_index < len(available_cols) else 0,
                        key=f"low_conf_time_{dataset_version}",
                        help=f"Detected: {time_display or context.time_variable}" if context.time_variable else None,
                    )
                    context.time_variable = selected_time
                with col2:
                    event_options = [c for c in available_cols if c != context.time_variable]
                    event_display = get_display_name(context.event_variable) if context.event_variable else None
                    event_index = (
                        event_options.index(context.event_variable) if context.event_variable in event_options else 0
                    )
                    selected_event = st.selectbox(
                        "**Event Variable**:",
                        options=event_options,
                        index=event_index if event_index < len(event_options) else 0,
                        key=f"low_conf_event_{dataset_version}",
                        help=f"Detected: {event_display or context.event_variable}" if context.event_variable else None,
                    )
                    context.event_variable = selected_event

            # Show collision suggestions if available
            if context.match_suggestions:
                st.warning(COLLISION_SUGGESTION_WARNING)
                for query_term, suggestions in context.match_suggestions.items():
                    suggestion_display = [get_display_name(s) for s in suggestions]
                    selected = st.selectbox(
                        f"**'{query_term}'** matches multiple columns. Which one did you mean?",
                        options=["(Select one)"] + suggestions,
                        key=f"collision_{query_term}_{dataset_version}",
                        help=f"Options: {', '.join(suggestion_display)}",
                    )
                    if selected and selected != "(Select one)":
                        # Update context with selected column based on intent
                        if not context.primary_variable:
                            context.primary_variable = selected
                        elif context.inferred_intent == AnalysisIntent.COMPARE_GROUPS and not context.grouping_variable:
                            context.grouping_variable = selected
                        st.info(f"✅ Selected: {get_display_name(selected)}")

            # Phase 2: Show "Add alias?" UI for unknown terms (no suggestions)
            # Check if there are any terms in the query that have no matches
            if hasattr(context, "unknown_terms") and context.unknown_terms:
                for unknown_term in context.unknown_terms:
                    # Get semantic layer and upload info
                    upload_id = dataset.upload_id if hasattr(dataset, "upload_id") else None
                    semantic_layer = dataset.semantic if hasattr(dataset, "semantic") else None

                    if semantic_layer and upload_id:
                        _render_add_alias_ui(
                            unknown_term=unknown_term,
                            available_columns=available_cols,
                            upload_id=upload_id,
                            dataset_version=dataset_version,
                            semantic_layer=semantic_layer,
                        )

            # Update context in session state after edits
            st.session_state["analysis_context"] = context

            # Auto-execute after variable selection (no confirmation button needed)
            # User has already made their selection, so proceed with execution
            st.info("✅ Variables selected. Executing analysis...")
            st.rerun()

        # Action buttons removed - replaced with sidebar conversation management (Phase 3.4)

    # Always show query input at bottom (sticky) - Phase 3.3 UI Redesign
    # This provides a conversational interface for follow-up questions
    # Check for prefilled query from follow-up suggestions
    prefilled = st.session_state.get("prefilled_query", "")

    # If prefilled query exists, use it and clear it (user clicked a follow-up button)
    if prefilled:
        query = prefilled
        # Clear prefilled immediately to prevent reuse
        del st.session_state["prefilled_query"]
    else:
        query = st.chat_input("Ask a question about your data...")

    if query:
        # Handle new query from chat input
        try:
            # Get semantic layer (Phase 3: cached for performance)
            semantic_layer = get_cached_semantic_layer(dataset_version, dataset)

            # Get dataset identifiers for structured logging
            dataset_id = dataset.name if hasattr(dataset, "name") else None
            upload_id = dataset.upload_id if hasattr(dataset, "upload_id") else None

            logger.info(
                "chat_input_query_received",
                query=query,
                dataset_id=dataset_id,
                upload_id=upload_id,
                dataset_version=dataset_version,
            )

            # Parse query directly (without UI components from ask_free_form_question)
            from clinical_analytics.core.nl_query_engine import NLQueryEngine

            nl_engine = NLQueryEngine(semantic_layer)
            query_intent = nl_engine.parse_query(query, dataset_id=dataset_id, upload_id=upload_id)

            if query_intent:
                # Convert QueryIntent to AnalysisContext (similar to ask_free_form_question logic)
                context = AnalysisContext()

                # Map intent type
                intent_map = {
                    "DESCRIBE": AnalysisIntent.DESCRIBE,
                    "COMPARE_GROUPS": AnalysisIntent.COMPARE_GROUPS,
                    "FIND_PREDICTORS": AnalysisIntent.FIND_PREDICTORS,
                    "SURVIVAL": AnalysisIntent.EXAMINE_SURVIVAL,
                    "CORRELATIONS": AnalysisIntent.EXPLORE_RELATIONSHIPS,
                    "COUNT": AnalysisIntent.COUNT,
                }
                context.inferred_intent = intent_map.get(query_intent.intent_type, AnalysisIntent.UNKNOWN)

                # Map variables
                context.research_question = query
                context.query_text = query  # Store original query for "most" detection
                context.primary_variable = query_intent.primary_variable
                context.grouping_variable = query_intent.grouping_variable
                context.predictor_variables = query_intent.predictor_variables
                context.time_variable = query_intent.time_variable
                context.event_variable = query_intent.event_variable

                # Copy filters from QueryIntent to AnalysisContext
                context.filters = query_intent.filters

                # Convert QueryIntent to QueryPlan and store in context
                if dataset_version and hasattr(nl_engine, "_intent_to_plan"):
                    context.query_plan = nl_engine._intent_to_plan(query_intent, dataset_version)
                    # Use QueryPlan confidence
                    context.confidence = context.query_plan.confidence
                else:
                    # Fallback: use QueryIntent confidence
                    context.confidence = query_intent.confidence

                # Set flags based on intent
                context.compare_groups = query_intent.intent_type == "COMPARE_GROUPS"
                context.find_predictors = query_intent.intent_type == "FIND_PREDICTORS"
                context.time_to_event = query_intent.intent_type == "SURVIVAL"

                # Store context
                st.session_state["analysis_context"] = context
                st.session_state["intent_signal"] = "nl_parsed"

                # If context is complete, execute immediately and render inline
                if context.is_complete_for_intent():
                    # Get QueryPlan if available (preferred), otherwise fallback to context
                    query_plan = getattr(context, "query_plan", None)

                    # Normalize query before generating run_key
                    normalized_query = normalize_query(query)

                    # Get confidence from QueryPlan or context (default to 0.0 if missing)
                    if query_plan:
                        confidence = query_plan.confidence
                        run_key = query_plan.run_key or generate_run_key(dataset_version, normalized_query, context)
                    else:
                        confidence = getattr(context, "confidence", 0.0)
                        run_key = generate_run_key(dataset_version, normalized_query, context)

                    logger.info(
                        "chat_input_analysis_execution_triggered",
                        intent_type=context.inferred_intent.value,
                        confidence=confidence,
                        run_key=run_key,
                        dataset_version=dataset_version,
                        query=query,
                    )

                    # Phase 2: Append messages to transcript and rerun
                    # Add user message to chat
                    user_msg: ChatMessage = {
                        "role": "user",
                        "text": query,
                        "run_key": None,
                        "status": "completed",
                        "created_at": time.time(),
                    }
                    st.session_state["chat"].append(user_msg)

                    # Compute result (no UI, pure computation)
                    result = get_or_compute_result(
                        cohort=cohort_pl,
                        context=context,
                        run_key=run_key,
                        dataset_version=dataset_version,
                        query_text=query,
                    )

                    # Add result to conversation history (backward compat)
                    headline = result.get("headline") or result.get("headline_text") or "Analysis completed"
                    filters_applied = []
                    if context.query_plan and context.query_plan.filters:
                        filters_applied = [f.__dict__ for f in context.query_plan.filters]
                    elif context.filters:
                        filters_applied = [f.__dict__ for f in context.filters]

                    max_conversation_history = 20
                    st.session_state["conversation_history"].append(
                        {
                            "query": query,
                            "intent": context.inferred_intent.value if context.inferred_intent else "UNKNOWN",
                            "headline": headline,
                            "run_key": run_key,
                            "timestamp": time.time(),
                            "filters_applied": filters_applied,
                        }
                    )
                    if len(st.session_state["conversation_history"]) > max_conversation_history:
                        st.session_state["conversation_history"] = st.session_state["conversation_history"][
                            -max_conversation_history:
                        ]

                    # Add assistant message to chat
                    assistant_msg: ChatMessage = {
                        "role": "assistant",
                        "text": query,  # Store query for reference
                        "run_key": run_key,
                        "status": "completed",
                        "created_at": time.time(),
                    }
                    st.session_state["chat"].append(assistant_msg)

                    # Rerun to render chat transcript
                    st.rerun()
                else:
                    # Context not complete - need variable selection, rerun to show UI
                    st.rerun()
            else:
                st.error("I couldn't understand your question. Please try rephrasing.")

        except ValueError:
            # Semantic layer not available
            st.error("Natural language queries are only available for datasets with semantic layers.")
        except Exception as e:
            logger.error("chat_input_query_error", query=query, error=str(e), exc_info=True)
            st.error(f"Error processing your question: {e}")


if __name__ == "__main__":
    main()

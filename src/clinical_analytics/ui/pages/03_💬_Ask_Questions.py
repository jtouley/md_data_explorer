"""
Dynamic Analysis Page - Question-Driven Analytics

Ask questions, get answers. No statistical jargon - just tell me what you want to know.
"""

import hashlib
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Literal, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import streamlit as st
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import from config (single source of truth)
from clinical_analytics.core.column_parser import parse_column_name
from clinical_analytics.core.nl_query_config import AUTO_EXECUTE_CONFIDENCE_THRESHOLD
from clinical_analytics.ui.components.dataset_loader import render_dataset_selector
from clinical_analytics.ui.components.question_engine import (
    AnalysisContext,
    AnalysisIntent,
    QuestionEngine,
)
from clinical_analytics.ui.components.result_interpreter import ResultInterpreter
from clinical_analytics.ui.messages import (
    COLLISION_SUGGESTION_WARNING,
    LOW_CONFIDENCE_WARNING,
)


# TypedDict schemas for chat transcript and pending state
class ChatMessage(TypedDict, total=False):
    """
    Schema for chat transcript messages.

    Phase 9 enhancements (Staff Feedback):
    - Added query_text, assistant_text for better history UX
    - Added intent, confidence for display without recompute
    - Made fields optional (total=False) for backward compatibility

    Phase 12 type improvements (Staff Feedback):
    - Use Literal types for role and status (better type safety)
    """

    # Core fields (Phase 12: Literal types for better type safety)
    role: Literal["user", "assistant"]  # Message role
    text: str  # Message text (user query or assistant response)
    run_key: str | None  # Run key for assistant messages (None for user messages)
    status: Literal["pending", "completed", "error"]  # Message status
    created_at: float  # Unix timestamp

    # Phase 9 additions (optional for backward compat)
    query_text: str | None  # Original user query that produced this result
    assistant_text: str | None  # What was displayed to user (may differ from text)
    intent: str | None  # Intent classification (e.g., "DESCRIBE", "COMPARE_GROUPS")
    confidence: float | None  # Confidence score (0.0-1.0)


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
    Canonicalize semantic scope dict for stable hashing (Phase 1.4 - Recursive).

    - Drops None values recursively
    - Sorts dictionary keys recursively
    - Sorts list values recursively
    - Ensures stable JSON serialization

    PR25: Improved to handle common edge cases (enums, dataclasses).
    Limitations: Complex objects (dataclasses with nested objects) should be
    converted to dicts before calling this function.

    Args:
        scope: Semantic scope dict (may be None)

    Returns:
        Canonicalized scope dict (stable, sorted, no Nones)

    Raises:
        TypeError: If scope contains non-serializable objects (enums/dataclasses
                   should be converted to primitives before calling)
    """
    if scope is None:
        return {}

    canonical = {}
    for key in sorted(scope.keys()):
        value = scope[key]
        if value is None:
            continue  # Drop None values
        elif isinstance(value, dict):
            # Recursively canonicalize nested dicts
            nested_canonical = canonicalize_scope(value)
            if nested_canonical:  # Only add non-empty dicts
                canonical[key] = nested_canonical
        elif isinstance(value, list):
            # Sort lists, recursively canonicalize list items if they are dicts
            sorted_list = []
            for item in value:
                if isinstance(item, dict):
                    sorted_list.append(canonicalize_scope(item))
                else:
                    # PR25: Handle enums and other objects with .value or .name attributes
                    if hasattr(item, "value"):
                        sorted_list.append(item.value)
                    elif hasattr(item, "name"):
                        sorted_list.append(item.name)
                    else:
                        sorted_list.append(item)
            # Sort the list (works for primitives, dicts as JSON strings for comparison)
            try:
                canonical[key] = sorted(sorted_list, key=lambda x: str(x))
            except TypeError as e:
                # PR25: If sorting fails, raise with helpful error message
                raise TypeError(
                    f"Scope contains non-serializable value for key '{key}': {type(value).__name__}. "
                    "Convert enums/dataclasses to primitives (use .value or .name) before canonicalizing."
                ) from e
        else:
            # PR25: Handle enums and other objects with .value or .name attributes
            if hasattr(value, "value"):
                canonical[key] = value.value
            elif hasattr(value, "name"):
                canonical[key] = value.name
            else:
                canonical[key] = value

    return canonical


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

    PR25: Now called proactively to enforce lifecycle invariants.
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
        logger.debug("cleanup_orphaned_result", dataset_version=dataset_version, removed_key=key)


def _evict_old_execution_cache(dataset_version: str) -> None:
    """
    Evict old execution cache entries to prevent unbounded growth (PR25).

    Execution cache (exec_result:{dataset_version}:{query_hash}) can grow unbounded
    if users run many different queries. This function limits cache size to
    MAX_STORED_RESULTS_PER_DATASET entries per dataset.

    Args:
        dataset_version: Dataset version to evict cache for
    """
    exec_prefix = f"exec_result:{dataset_version}:"
    exec_keys = [key for key in st.session_state.keys() if key.startswith(exec_prefix)]

    if len(exec_keys) <= MAX_STORED_RESULTS_PER_DATASET:
        return

    # Sort by access time (if available) or FIFO eviction
    # For now, use simple FIFO: keep last N entries based on insertion order
    # (Streamlit session_state preserves insertion order)
    keys_to_evict = exec_keys[:-MAX_STORED_RESULTS_PER_DATASET]

    for key in keys_to_evict:
        del st.session_state[key]
        logger.debug("evicted_execution_cache", dataset_version=dataset_version, evicted_key=key)


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

    # ADR009 Phase 2: Render LLM-generated query interpretation
    if query_plan:
        _render_query_interpretation(query_plan)

    # ADR009 Phase 1: Render LLM-generated follow-up questions
    if query_plan and query_plan.follow_ups:
        _render_llm_follow_ups(query_plan, run_key)


def execute_analysis_with_idempotency(
    cohort: pl.DataFrame,
    context: AnalysisContext,
    run_key: str,
    dataset_version: str,
    query_text: str,
    query_plan=None,
    execution_result: dict | None = None,
    semantic_layer=None,
) -> None:
    """
    Execute analysis with idempotency guard and result persistence.

    Checks if result already exists in session_state. If yes, renders from cache.
    If no, computes, stores, and renders.

    Phase 3.1: Added execution_result and semantic_layer parameters to avoid re-execution.
    If execution_result is provided (from execute_query_plan), uses its DataFrame
    instead of re-executing via compute_analysis_by_type().

    Args:
        cohort: Cohort data (for Trust UI)
        context: Analysis context
        run_key: Run key for idempotency
        dataset_version: Dataset version
        query_text: Query text
        query_plan: QueryPlan object (if available)
        execution_result: Result from execute_query_plan() (Phase 3.1)
        semantic_layer: SemanticLayer instance for formatting (Phase 3.1)
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

        # PR25: Enforce remember_run() invariant - must be called even for cached results
        # This ensures history tracking is consistent regardless of cache hits
        remember_run(dataset_version, run_key)

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

        # PR25: _suggest_follow_ups() removed (disabled partial feature)
        # Follow-ups will be added via LLM-generated QueryPlan.follow_ups in future
        return

    # Not computed - compute and store
    with st.spinner("Running analysis..."):
        # Phase 3.1: Use execution_result from execute_query_plan() if provided
        if execution_result and semantic_layer:
            # Execution already happened via execute_query_plan() - format its result
            logger.info(
                "analysis_using_execution_result",
                run_key=run_key,
                dataset_version=dataset_version,
                intent_type=context.inferred_intent.value,
            )

            # Format result via semantic layer (no re-execution, just formatting)
            result = semantic_layer.format_execution_result(execution_result, context)
        else:
            # Phase 3.1: No legacy path - execution_result is required
            # All queries must go through execute_query_plan() first
            logger.error(
                "legacy_execution_path_blocked",
                message="execute_analysis_with_idempotency requires execution_result from execute_query_plan()",
                run_key=run_key,
                dataset_version=dataset_version,
            )
            raise ValueError(
                "Phase 3.1: Legacy execution path blocked. "
                "execute_analysis_with_idempotency requires execution_result parameter. "
                "All queries must go through semantic_layer.execute_query_plan() first."
            )

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
                cohort=cohort,
                dataset_version=dataset_version,
                query_text=query_text,
            )

        # PR25: _suggest_follow_ups() removed (disabled partial feature)
        # Follow-ups will be added via LLM-generated QueryPlan.follow_ups in future

        logger.info("analysis_rendering_complete", run_key=run_key)

        # Don't rerun - results are already rendered inline, conversation history will show on next query


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


def _render_thinking_indicator(steps: list[dict[str, Any]]) -> None:
    """
    Render progressive thinking steps from core layer (Phase 2.5.1).

    Displays step-by-step progress of query execution, showing query plan interpretation,
    validation status, execution progress, and completion/failure state.

    Args:
        steps: List of step dicts from execute_query_plan() result.
              Each step has:
              - "status": str - One of "processing", "completed", "error"
              - "text": str - Step description (e.g., "Interpreting query")
              - "details": dict - Step-specific details (intent, metric, warnings, etc.)
    """
    if not steps:
        return

    # Determine final status from last step
    last_step = steps[-1]
    status_label = "🤔 Processing your question..."
    status_state = "running"
    expanded = True  # Default to expanded

    if last_step["status"] == "completed":
        status_label = "✅ Query complete!"
        status_state = "complete"
        expanded = False  # Auto-collapse when completed
    elif last_step["status"] == "error":
        status_label = "❌ Query failed"
        status_state = "error"
        expanded = True  # Keep expanded for errors

    with st.status(status_label, expanded=expanded, state=status_state):
        for step in steps:
            # Render step text
            st.write(f"**{step['text']}**")

            # Render step details if available
            if step.get("details"):
                details = step["details"]
                if step["text"] == "Interpreting query":
                    # Show query plan interpretation
                    if details.get("intent"):
                        st.write(f"- Intent: `{details['intent']}`")
                    if details.get("metric"):
                        st.write(f"- Analyzing: `{details['metric']}`")
                    if details.get("group_by"):
                        st.write(f"- Grouped by: `{details['group_by']}`")
                    if details.get("filter_count", 0) > 0:
                        st.write(f"- Filters: {details['filter_count']} condition(s)")
                elif step["text"] == "Validating plan":
                    if details.get("has_warnings"):
                        st.write(f"- ⚠️ {details.get('warning_count', 0)} warning(s) detected")
                elif step["text"] == "Executing query":
                    st.write(f"- Run key: `{details.get('run_key', 'N/A')[:16]}...`")
                elif step["text"] == "Query complete":
                    st.write(f"- Result: {details.get('result_rows', 0)} row(s)")
                elif step["text"] == "Query failed":
                    st.write(f"- Error: {details.get('error', 'Unknown error')}")


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


# ADR009 Phase 2: Query Interpretation
def _render_query_interpretation(plan) -> None:
    """
    Render LLM-generated query interpretation (ADR009 Phase 2).

    Displays human-readable explanation of what the query is asking and why
    the confidence score is what it is. Helps users understand how their
    question was parsed without needing to inspect technical QueryPlan fields.

    Args:
        plan: QueryPlan with interpretation and confidence_explanation fields
    """
    if not plan.interpretation and not plan.confidence_explanation:
        return  # No interpretation to render

    # Only show interpretation section if we have content
    if plan.interpretation:
        st.info(f"**Understanding your question:** {plan.interpretation}")

    # Show confidence explanation if confidence is not high or if explanation exists
    if plan.confidence_explanation and plan.confidence < 0.9:
        st.caption(f"*Confidence note:* {plan.confidence_explanation}")


# ADR009 Phase 1: LLM-Generated Follow-Ups
def _render_llm_follow_ups(plan, run_key: str) -> None:
    """
    Render LLM-generated follow-up questions (ADR009 Phase 1).

    Displays context-aware follow-up questions as clickable buttons that prefill
    the query input. Follow-ups are treated as suggestions, not endorsements.

    Args:
        plan: QueryPlan with follow_ups and follow_up_explanation fields
        run_key: Current query run key for logging
    """
    if not plan.follow_ups:
        return  # No follow-ups to render

    st.markdown("---")
    st.subheader("💡 Explore Further")

    if plan.follow_up_explanation:
        st.caption(plan.follow_up_explanation)

    # Render follow-up questions as buttons in columns
    cols = st.columns(min(len(plan.follow_ups), 3))
    for idx, follow_up in enumerate(plan.follow_ups):
        col_idx = idx % 3
        with cols[col_idx]:
            # Use button with unique key
            button_key = f"followup_{run_key}_{idx}"
            if st.button(
                follow_up,
                key=button_key,
                use_container_width=True,
                help="Click to explore this question",
            ):
                # Prefill query input with follow-up question
                st.session_state["prefilled_query"] = follow_up
                st.rerun()

    logger.info(
        "llm_followups_rendered",
        run_key=run_key,
        follow_up_count=len(plan.follow_ups),
        has_explanation=bool(plan.follow_up_explanation),
    )


def get_dataset_version(dataset, dataset_choice: str) -> str:
    """
    Get stable dataset version identifier for caching and lifecycle management.

    All datasets are uploaded datasets, so use upload_id as version.
    """
    # All datasets are uploaded datasets: use upload_id as version
    return dataset_choice  # This is the upload_id


def main():
    """
    Ask Questions Page - Question-Driven Analytics with NL Query Engine.

    ## Session State Machine

    This page manages complex state transitions for natural language query processing.
    Understanding this state machine is critical for debugging and maintenance.

    ### State Keys

    1. **`analysis_context`** (AnalysisContext | None)
       - Current analysis context (intent, variables, filters, etc.)
       - Set during NL parsing, used during execution
       - Cleared on dataset change or new query

    2. **`intent_signal`** (str | None)
       - State transition signal: None → "nl_parsed" → "executed"
       - Drives rendering and execution flow
       - Fragile: Order of operations matters!

    3. **`use_nl_query`** (bool)
       - True: NL query mode (QueryPlan-driven execution)
       - False: Legacy manual mode (direct column selection)
       - Set by user via radio button

    4. **`chat`** (list[ChatMessage])
       - Chat transcript with run keys for result lookup
       - Persists across reruns for conversation history

    5. **`pending`** (Pending | None)
       - Pending computation state (run_key, context, query_plan)
       - Set when user clicks "Execute" on parsed query
       - Cleared after execution completes

    ### State Transitions

    ```
    None (initial state)
      ↓ [User enters query + parses]
    "nl_parsed" (query parsed, awaiting confirmation)
      ↓ [User clicks "Execute this analysis"]
    "executed" (execution complete, results rendered)
      ↓ [User enters new query]
    None (reset to initial state)
    ```

    ### Fragility Notes

    **WARNING**: This state machine is fragile and order-dependent!

    1. **Rerun Invalidation**: Streamlit reruns invalidate assumptions
       - Widget state changes trigger full page rerun
       - State transitions must be idempotent
       - Side effects (execute, store results) must be guarded

    2. **Order Matters**:
       - `intent_signal` must be checked BEFORE rendering widgets
       - Execution must happen BEFORE chat rendering
       - Chat rendering must happen LAST (uses stored results)

    3. **Dataset Change Edge Case**:
       - Dataset change clears ALL state (context, chat, pending)
       - But widget values persist across dataset changes!
       - Must explicitly clear widget-dependent state

    ### Future Refactor (ADR008)

    This state management should be extracted into a dedicated StateMachine class:
    - Explicit state enum (IDLE, PARSED, EXECUTING, COMPLETED)
    - Transition validation (prevent invalid transitions)
    - Side effect isolation (execute only on valid transitions)
    - Testing: State machine unit tests separate from UI tests

    See ADR008 for full refactor plan.
    """
    st.title("💬 Ask Questions")
    st.markdown("""
    Ask questions about your data in plain English. I'll figure out the right analysis and explain the results.
    """)

    # Dataset selection (Phase 8.2: Use reusable component)
    result = render_dataset_selector(show_semantic_scope=True)
    if result is None:
        return  # No datasets available (error message already shown)

    dataset, cohort_pd, dataset_choice, dataset_version = result

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

    # Convert to Polars for compute functions (page-specific logic)
    cohort = pl.from_pandas(cohort_pd)
    cohort_pl = cohort  # Alias for clarity in Phase 2

    # Conversation management in sidebar - Phase 3.4 Remove Buttons
    with st.sidebar:
        st.markdown("### Conversation")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state["conversation_history"] = []
            st.session_state["analysis_context"] = None
            st.session_state["intent_signal"] = None
            st.rerun()

    st.divider()

    # ============================================================================
    # STATE MACHINE DOCUMENTATION (Per Staff Engineer Feedback)
    # ============================================================================
    # The Ask Questions page uses Streamlit session state as a mini state machine
    # to manage analysis context and execution flow. This is fragile but acceptable for MVP.
    #
    # State Keys:
    #   - analysis_context: AnalysisContext | None
    #       Stores the current analysis configuration (variables, intent, filters)
    #       Set when NL query is parsed or user answers clarifying questions
    #   - intent_signal: "nl_parsed" | None
    #       Signals that NL parsing completed and context is ready
    #       Used to trigger analysis execution flow
    #   - use_nl_query: bool (legacy, may be removed in future)
    #       Legacy flag for NL query mode (kept for backward compatibility)
    #
    # Allowed Transitions:
    #   1. None -> "nl_parsed"
    #      Trigger: User submits NL query, parsing succeeds
    #      Action: Set analysis_context, set intent_signal="nl_parsed"
    #      Location: After successful NL query parsing (~line 1990)
    #
    #   2. "nl_parsed" -> None
    #      Trigger: User changes dataset, clears conversation, or resets
    #      Action: Clear analysis_context, set intent_signal=None
    #      Location: Clear conversation button (~line 1543), dataset change detection
    #
    #   3. "nl_parsed" -> "executed" (implicit)
    #      Trigger: Context is complete, analysis executes
    #      Action: Execute analysis, render results, keep context for follow-ups
    #      Location: Analysis execution block (~line 1574-1700)
    #
    # Fragility Notes:
    #   - Order matters: Must check intent_signal before accessing analysis_context
    #   - Reruns can invalidate assumptions if state is modified mid-cycle
    #   - Future contributors will break this accidentally without explicit docs
    #   - State is not validated - invalid states can cause silent failures
    #
    # Future Refactor (Post-MVP, see ADR008):
    #   - Extract to StateManager class with explicit transition methods
    #   - Add state validation and transition guards
    #   - Use state machine library (e.g., transitions) for robustness
    # ============================================================================

    # Initialize session state for context
    # PR25: Initialize state machine state with validation
    if "analysis_context" not in st.session_state:
        st.session_state["analysis_context"] = None
        st.session_state["intent_signal"] = None

    # PR25: Proactive cleanup - enforce lifecycle invariants
    cleanup_old_results(dataset_version)

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
    # PR25: State machine validation - enforce documented transitions
    intent_signal = st.session_state.get("intent_signal")
    if intent_signal is not None:
        # Validate state machine invariant: intent_signal requires analysis_context
        context = st.session_state.get("analysis_context")
        if context is None:
            logger.error(
                "state_machine_invariant_violation",
                message="intent_signal set but analysis_context is None",
                intent_signal=intent_signal,
            )
            # Recover by clearing invalid state
            st.session_state["intent_signal"] = None
            st.warning("⚠️ Analysis state was invalid. Please try your query again.")
            return
        # We have intent, now gather details

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
            else:
                confidence = getattr(context, "confidence", 0.0)

            # Phase 1.1.5: Don't generate run_key before execution - semantic layer will generate it
            # We'll get run_key from execution_result after execution

            logger.info(
                "analysis_execution_triggered",
                intent_type=context.inferred_intent.value,
                confidence=confidence,
                threshold=AUTO_EXECUTE_CONFIDENCE_THRESHOLD,
                dataset_version=dataset_version,
                query=getattr(context, "research_question", ""),
            )

            # ADR003 Phase 3: Use execute_query_plan() for confidence and completeness gating
            # semantic_layer is already available from line 1291
            if query_plan and semantic_layer:
                query_text = getattr(context, "research_question", "")

                # INVARIANT ENFORCEMENT (PR25): Normalize query text before passing to execute_query_plan()
                # _generate_run_key() enforces normalization contract - must normalize here
                normalized_query = normalize_query(query_text)

                # Reject empty queries at ingestion (PR25 - invariant enforcement)
                if not normalized_query:
                    logger.warning("empty_query_rejected", original_query=query_text)
                    st.warning("Please enter a non-empty query.")
                    return

                # Phase 2.4: Cache execution results to prevent duplicate execute_query_plan() calls
                # Use stable sha256 digest (not hash()) for deterministic cache keys across sessions
                # run_key is only available after execution, so we use query hash for pre-execution cache lookup
                query_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]
                exec_cache_key = f"exec_result:{dataset_version}:{query_hash}"

                # Check if user requested force rerun
                force_rerun_key = f"force_rerun:{dataset_version}"
                force_rerun = st.session_state.get(force_rerun_key, False)

                # Try to get cached execution result (unless forcing rerun)
                execution_result = None
                if not force_rerun and exec_cache_key in st.session_state:
                    execution_result = st.session_state[exec_cache_key]
                    logger.debug(
                        "query_execution_cache_hit", cache_key=exec_cache_key, query_preview=normalized_query[:50]
                    )

                # Phase 2.5.1: Progressive thinking indicator for query execution
                if execution_result is None:
                    # Execute query (core layer generates step information)
                    # INVARIANT: Pass normalized_query (not raw query_text) to enforce contract
                    execution_result = semantic_layer.execute_query_plan(
                        query_plan, confidence_threshold=AUTO_EXECUTE_CONFIDENCE_THRESHOLD, query_text=normalized_query
                    )

                    # Cache the execution result
                    st.session_state[exec_cache_key] = execution_result
                    logger.debug(
                        "query_execution_cache_miss", cache_key=exec_cache_key, query_preview=normalized_query[:50]
                    )

                    # PR25: Evict old execution cache entries (unbounded growth prevention)
                    _evict_old_execution_cache(dataset_version)

                # Phase 2.5.1: Render thinking indicator from core layer step data (UI only renders)
                if execution_result.get("steps"):
                    _render_thinking_indicator(execution_result["steps"])
                elif execution_result is not None:
                    # Cached result - show brief indicator (no steps available for cached results)
                    st.info("💡 Using cached result (click 'Re-run Query' below to refresh)")

                # Clear force_rerun flag after use
                if force_rerun:
                    st.session_state[force_rerun_key] = False

                # Phase 2.3: Always execute, show warnings inline (no gating)
                st.divider()

                # Display warnings if present - group in expander to avoid cluttering UI
                warnings = execution_result.get("warnings", [])
                if warnings:
                    # Categorize warnings by severity (info vs warning)
                    info_warnings = []
                    error_warnings = []
                    for warning in warnings:
                        warning_lower = warning.lower()
                        if (
                            "error" in warning_lower
                            or "failed" in warning_lower
                            or "validation failed" in warning_lower
                        ):
                            error_warnings.append(warning)
                        else:
                            info_warnings.append(warning)

                    # Group warnings in expander (auto-collapsed if only info warnings)
                    warning_count = len(warnings)
                    warning_text = f"message{'s' if warning_count != 1 else ''}"
                    with st.expander(
                        f"ℹ️ Query Information ({warning_count} {warning_text})",
                        expanded=len(error_warnings) > 0,
                    ):
                        # Show error-level warnings first
                        for warning in error_warnings:
                            st.error(f"❌ {warning}")
                        # Then info-level warnings
                        for warning in info_warnings:
                            st.info(f"ℹ️ {warning}")

                # Phase 1.1.5: Always use run_key from execution result (semantic layer generates it deterministically)
                run_key = execution_result.get("run_key")
                if not run_key:
                    raise ValueError("Execution result must include run_key - semantic layer should always generate it")

                # Proceed with analysis if successful
                if execution_result.get("success"):
                    execute_analysis_with_idempotency(
                        cohort,
                        context,
                        run_key,
                        dataset_version,
                        normalized_query,  # PR25: Pass normalized query (invariant enforcement)
                        query_plan,
                        execution_result,
                        semantic_layer,
                    )

                    # Phase 3.1: Add assistant message to chat if query came from chat input
                    # Check if last chat message is user message (indicates chat input was used)
                    chat = st.session_state.get("chat", [])
                    if chat and chat[-1]["role"] == "user" and chat[-1].get("run_key") is None:
                        # Query came from chat input - add assistant message with actual answer content
                        # Get formatted result headline/summary (not the query text)
                        result_key = f"analysis_result:{dataset_version}:{run_key}"
                        result = st.session_state.get(result_key, {})
                        assistant_text = result.get("headline") or result.get("headline_text") or "Analysis completed"

                        assistant_msg: ChatMessage = {
                            "role": "assistant",
                            "text": assistant_text,  # Store actual answer content, not query text
                            "run_key": run_key,
                            "status": "completed",
                            "created_at": time.time(),
                        }
                        st.session_state["chat"].append(assistant_msg)

                    # Phase 2.4: Add "Re-run Query" button for explicit re-execution
                    # Use stable hash for button key (same normalization as cache key)
                    query_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]
                    if st.button("🔄 Re-run Query", key=f"rerun_btn_{dataset_version}_{query_hash}"):
                        st.session_state[force_rerun_key] = True
                        st.rerun()
                else:
                    # Execution failed - show error with details from warnings
                    error_msg = "Execution failed"
                    if execution_result.get("warnings"):
                        error_msg += " - see warnings above for details"
                    st.error(f"❌ {error_msg}")
            else:
                # Phase 3.1: No fallback - QueryPlan is required
                st.error(
                    "❌ Cannot execute query without QueryPlan. "
                    "This indicates a problem with query parsing. Please try rephrasing your question."
                )
                logger.error(
                    "query_execution_failed_no_queryplan",
                    message="execute_query_plan requires QueryPlan - no fallback path available",
                    has_semantic_layer=semantic_layer is not None,
                    has_query_plan=query_plan is not None,
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
        # Phase 1.3: Normalize query and reject if empty
        normalized_query = normalize_query(query)
        if not normalized_query:
            # Empty query after normalization - reject silently (don't process, don't log, don't compute)
            logger.debug("empty_query_rejected_at_ingestion", raw_query=query)
            st.rerun()  # Rerun to clear the input without processing

        # Handle new query from chat input
        try:
            # Get semantic layer (Phase 3: cached for performance)
            semantic_layer = get_cached_semantic_layer(dataset_version, dataset)

            # Get dataset identifiers for structured logging
            dataset_id = dataset.name if hasattr(dataset, "name") else None
            upload_id = dataset.upload_id if hasattr(dataset, "upload_id") else None

            logger.info(
                "chat_input_query_received",
                query=normalized_query,  # Log normalized query
                dataset_id=dataset_id,
                upload_id=upload_id,
                dataset_version=dataset_version,
            )

            # Parse query directly (without UI components from ask_free_form_question)
            from clinical_analytics.core.nl_query_engine import NLQueryEngine

            nl_engine = NLQueryEngine(semantic_layer)
            query_intent = nl_engine.parse_query(normalized_query, dataset_id=dataset_id, upload_id=upload_id)

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

                # Phase 3.1: Chat handler should NOT execute - only parse and rerun
                # Main flow (lines 1630-1720) will handle execution after rerun
                if context.is_complete_for_intent():
                    # Just set state and rerun - main flow will handle execution
                    # Add user message to chat for display
                    user_msg: ChatMessage = {
                        "role": "user",
                        "text": query,
                        "run_key": None,
                        "status": "completed",
                        "created_at": time.time(),
                    }
                    st.session_state["chat"].append(user_msg)

                    # Rerun - main flow will execute query and render results
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

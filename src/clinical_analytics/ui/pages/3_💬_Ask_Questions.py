"""
Dynamic Analysis Page - Question-Driven Analytics

Ask questions, get answers. No statistical jargon - just tell me what you want to know.
"""

import hashlib
import json
import sys
from collections import deque
from pathlib import Path

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
from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
from clinical_analytics.ui.components.question_engine import (
    AnalysisContext,
    AnalysisIntent,
    QuestionEngine,
)
from clinical_analytics.ui.components.result_interpreter import ResultInterpreter
from clinical_analytics.ui.config import MULTI_TABLE_ENABLED
from clinical_analytics.ui.messages import (
    CLEAR_RESULTS,
    COLLISION_SUGGESTION_WARNING,
    CONFIRM_AND_RUN,
    LOW_CONFIDENCE_WARNING,
    NO_DATASETS_AVAILABLE,
    RESULTS_CLEARED,
    START_OVER,
)

# Page config
st.set_page_config(page_title="Ask Questions | Clinical Analytics", page_icon="💬", layout="wide")

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


def generate_run_key(
    dataset_version: str,
    query_text: str | None,
    context: AnalysisContext,
) -> str:
    """
    Generate stable run key for idempotency.

    Canonicalizes inputs to ensure same query + variables = same key.
    """
    # Normalize query text (collapse whitespace)
    # Handle None query_text (default to empty string)
    if query_text is None:
        query_text = ""
    normalized_query = " ".join(query_text.strip().split())

    # Canonicalize variables
    payload = {
        "dataset_version": dataset_version,
        "query": normalized_query,
        "intent": context.inferred_intent.value if context.inferred_intent else "UNKNOWN",
        "vars": {
            "primary": context.primary_variable or "",
            "grouping": context.grouping_variable or "",
            "predictors": sorted(context.predictor_variables or []),
            "time": context.time_variable or "",
            "event": context.event_variable or "",
        },
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
def render_descriptive_analysis(result: dict) -> None:
    """Render descriptive analysis from serializable dict."""
    # Check for error results first
    if "error" in result:
        st.error(f"❌ **Analysis Error**: {result['error']}")
        if "available_columns" in result:
            cols_preview = result["available_columns"][:20]
            cols_str = ", ".join(cols_preview)
            if len(result["available_columns"]) > 20:
                cols_str += "..."
            st.info(f"💡 **Available columns**: {cols_str}")
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

    st.markdown("## 📊 Your Data at a Glance")

    # Overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", f"{result['row_count']:,}")
    with col2:
        st.metric("Variables", result["column_count"])
    with col3:
        st.metric("Data Completeness", f"{100 - result['missing_pct']:.1f}%")

    # Summary statistics
    st.markdown("### Summary Statistics")

    if result["summary_stats"]:
        st.markdown("**Numeric Variables:**")
        # Convert to pandas for display
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
    """Render count analysis from serializable dict."""
    # Headline answer first!
    if "headline" in result:
        st.info(f"📋 **Answer:** {result['headline']}")

    st.markdown("## 📊 Count Analysis")

    # If grouped, show group breakdown
    if "grouped_by" in result and result.get("group_counts"):
        st.markdown(f"### Counts by {result['grouped_by']}")
        group_col = result["grouped_by"]
        for item in result["group_counts"]:
            group_value = item[group_col]
            count = item["count"]
            pct = (count / result["total_count"]) * 100 if result["total_count"] > 0 else 0.0
            st.write(f"  - **{group_value}**: {count} ({pct:.1f}%)")
    else:
        # Simple total count
        st.metric("Total Count", f"{result['total_count']:,}")


# PANDAS EXCEPTION: Required for Streamlit st.dataframe display
# TODO: Remove when Streamlit supports Polars natively
def render_comparison_analysis(result: dict) -> None:
    """Render comparison analysis from serializable dict."""
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown("## 📈 Group Comparison")

    outcome_col = result["outcome_col"]
    group_col = result["group_col"]
    test_type = result["test_type"]

    # Show headline answer if available (direct answer to the user's question)
    if "headline_text" in result:
        st.markdown("### 📋 Answer")
        st.info(result["headline_text"])

    if test_type == "t_test":
        groups = result["groups"]
        p_value = result["p_value"]

        st.markdown("### Results")
        st.markdown(f"**Comparing {outcome_col} between {groups[0]} and {groups[1]}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{groups[0]} Average", f"{result['group1_mean']:.2f}")
        with col2:
            st.metric(f"{groups[1]} Average", f"{result['group2_mean']:.2f}")
        with col3:
            p_interp = ResultInterpreter.interpret_p_value(p_value)
            st.metric("Difference", f"{p_interp['significance']} {p_interp['emoji']}")

        st.markdown("### What does this mean?")
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
    """Render relationship analysis from serializable dict."""
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown("## 🔗 Relationships Between Variables")

    variables = result["variables"]

    st.markdown("### Correlation Heatmap")

    # Reconstruct correlation matrix for heatmap
    corr_data = result["correlations"]
    corr_matrix = pd.DataFrame(index=variables, columns=variables)
    for item in corr_data:
        corr_matrix.loc[item["var1"], item["var2"]] = item["correlation"]
        corr_matrix.loc[item["var2"], item["var1"]] = item["correlation"]  # Symmetric
    corr_matrix = corr_matrix.astype(float)

    # Simple heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    import seaborn as sns

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("How Variables Relate")
    st.pyplot(fig)

    # Highlight strong correlations
    st.markdown("### Strong Relationships")

    strong_correlations = result["strong_correlations"]

    if strong_correlations:
        for item in strong_correlations:
            var1 = item["var1"]
            var2 = item["var2"]
            corr_val = item["correlation"]
            direction = "positive" if corr_val > 0 else "negative"
            strength = "strong" if abs(corr_val) >= 0.7 else "moderate"
            st.markdown(f"**{var1}** and **{var2}**: {strength} {direction} relationship (r={corr_val:.2f})")
    else:
        st.info("ℹ️ No strong correlations found (|r| >= 0.5)")


def render_analysis_by_type(result: dict, intent: AnalysisIntent) -> None:
    """Route to appropriate render function based on result type."""
    result_type = result.get("type")

    if result_type == "descriptive":
        render_descriptive_analysis(result)
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


def execute_analysis_with_idempotency(
    cohort: pl.DataFrame,
    context: AnalysisContext,
    run_key: str,
    dataset_version: str,
    query_text: str,
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
        render_analysis_by_type(st.session_state[result_key], context.inferred_intent)
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

        logger.info(
            "analysis_result_stored",
            run_key=run_key,
            dataset_version=dataset_version,
        )

        # Render
        logger.debug("analysis_rendering_start", run_key=run_key)
        render_analysis_by_type(result, context.inferred_intent)
        logger.info("analysis_rendering_complete", run_key=run_key)


def get_dataset_version(dataset, is_uploaded: bool, dataset_choice: str) -> str:
    """
    Get stable dataset version identifier for caching and lifecycle management.

    For uploaded datasets: use upload_id
    For built-in datasets: use dataset name + config hash (if available)
    """
    if is_uploaded:
        # Uploaded datasets: use upload_id as version
        return dataset_choice  # This is the upload_id
    else:
        # Built-in datasets: use dataset name (could be enhanced with config hash)
        return dataset_choice


def main():
    st.title("💬 Ask Questions")
    st.markdown("""
    Ask questions about your data in plain English. I'll figure out the right analysis and explain the results.
    """)

    # Dataset selection
    st.sidebar.header("Data Selection")

    # Load datasets
    available_datasets = DatasetRegistry.list_datasets()
    dataset_info = DatasetRegistry.get_all_dataset_info()

    dataset_display_names = {}
    for ds_name in available_datasets:
        info = dataset_info[ds_name]
        display_name = info["config"].get("display_name", ds_name.replace("_", "-").upper())
        dataset_display_names[display_name] = ds_name

    uploaded_datasets = {}
    try:
        uploads = UploadedDatasetFactory.list_available_uploads()
        for upload in uploads:
            upload_id = upload["upload_id"]
            dataset_name = upload.get("dataset_name", upload_id)
            display_name = f"📤 {dataset_name}"
            dataset_display_names[display_name] = upload_id
            uploaded_datasets[upload_id] = upload
    except Exception:
        pass

    if not dataset_display_names:
        st.error(NO_DATASETS_AVAILABLE)
        return

    dataset_choice_display = st.sidebar.selectbox("Choose Dataset", list(dataset_display_names.keys()))
    dataset_choice = dataset_display_names[dataset_choice_display]
    # Check if this is an uploaded dataset (multiple checks for robustness)
    is_uploaded = (
        dataset_choice in uploaded_datasets
        or dataset_choice_display.startswith("📤")
        or dataset_choice not in available_datasets
    )

    # Load dataset
    with st.spinner(f"Loading {dataset_choice_display}..."):
        try:
            if is_uploaded:
                # For uploaded datasets, use the factory (requires upload_id)
                dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
                dataset.load()
            else:
                # For built-in datasets, use the registry
                dataset = DatasetRegistry.get_dataset(dataset_choice)
                dataset.validate()
                dataset.load()

            cohort_pd = dataset.get_cohort()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

    # Convert to Polars for compute functions
    cohort = pl.from_pandas(cohort_pd)

    # Get dataset version for lifecycle management
    dataset_version = get_dataset_version(dataset, is_uploaded, dataset_choice)

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

    st.divider()

    # Initialize session state for context
    if "analysis_context" not in st.session_state:
        st.session_state["analysis_context"] = None
        st.session_state["intent_signal"] = None
        st.session_state["use_nl_query"] = True  # Default to NL query first

    # Step 1: Ask question (NL or structured)
    if st.session_state["intent_signal"] is None:
        # Try free-form NL query first
        if st.session_state["use_nl_query"]:
            try:
                # Get semantic layer using contract pattern
                semantic_layer = dataset.get_semantic_layer()

                # Get dataset identifiers for structured logging
                dataset_id = dataset.name if hasattr(dataset, "name") else None
                upload_id = dataset.upload_id if hasattr(dataset, "upload_id") else None

                logger.info(
                    "page_render_query_input",
                    dataset_id=dataset_id,
                    upload_id=upload_id,
                    dataset_version=get_dataset_version(dataset, is_uploaded, dataset_choice),
                    use_nl_query=True,
                )

                context = QuestionEngine.ask_free_form_question(
                    semantic_layer,
                    dataset_id=dataset_id,
                    upload_id=upload_id,
                    dataset_version=get_dataset_version(dataset, is_uploaded, dataset_choice),
                )

                if context:
                    # Successfully parsed NL query
                    logger.info(
                        "nl_query_parsed_successfully",
                        query=getattr(context, "research_question", ""),
                        intent_type=context.inferred_intent.value,
                        confidence=context.confidence,
                        is_complete=context.is_complete_for_intent(),
                        dataset_id=dataset_id,
                        upload_id=upload_id,
                    )
                    st.session_state["analysis_context"] = context
                    st.session_state["intent_signal"] = "nl_parsed"
                    logger.debug("page_rerun_triggered", reason="nl_query_parsed")
                    st.rerun()

            except ValueError:
                # Semantic layer not available
                st.info("Natural language queries are only available for datasets with semantic layers.")
                st.session_state["use_nl_query"] = False
                st.rerun()
                return
            except Exception as e:
                st.error(f"Error parsing natural language query: {e}")
                st.session_state["use_nl_query"] = False

            # Show option to use structured questions instead
            st.divider()
            st.markdown("### Or use structured questions")
            if st.button("💬 Use structured questions instead", help="Choose from predefined question types"):
                st.session_state["use_nl_query"] = False
                st.rerun()

        else:
            # Use structured questions
            intent_signal = QuestionEngine.ask_initial_question(cohort)

            if intent_signal:
                if intent_signal == "help":
                    st.divider()
                    help_answers = QuestionEngine.ask_help_questions(cohort)

                    # Map help answers to intent
                    if help_answers.get("has_time"):
                        intent_signal = "survival"
                    elif help_answers.get("has_outcome"):
                        intent_signal = help_answers.get("approach", "predict")
                    else:
                        intent_signal = "describe"

                st.session_state["intent_signal"] = intent_signal
                st.session_state["analysis_context"] = QuestionEngine.build_context_from_intent(intent_signal, cohort)
                st.rerun()

            # Show option to go back to NL query
            st.divider()
            if st.button("🔙 Try natural language query instead"):
                st.session_state["use_nl_query"] = True
                st.rerun()

    else:
        # We have intent, now gather details
        context = st.session_state["analysis_context"]

        logger.info(
            "page_render_analysis_configuration",
            intent_signal=st.session_state.get("intent_signal"),
            intent_type=context.inferred_intent.value if context else None,
            is_complete=context.is_complete_for_intent() if context else False,
            dataset_version=get_dataset_version(dataset, is_uploaded, dataset_choice),
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
            if len(context.predictor_variables) < 2:
                context.predictor_variables = QuestionEngine.select_predictor_variables(cohort, exclude=[], min_vars=2)

        # Update context in session state
        st.session_state["analysis_context"] = context

        # Show progress
        st.divider()
        QuestionEngine.render_progress_indicator(context)

        # If complete, auto-execute or show confirmation
        if context.is_complete_for_intent():
            # Get QueryPlan if available (preferred), otherwise fallback to context
            query_plan = getattr(context, "query_plan", None)

            # Get confidence from QueryPlan or context (default to 0.0 if missing - fail closed)
            if query_plan:
                confidence = query_plan.confidence
                run_key = query_plan.run_key or generate_run_key(
                    dataset_version, getattr(context, "research_question", ""), context
                )
            else:
                confidence = getattr(context, "confidence", 0.0)
                query_text = getattr(context, "research_question", "")
                run_key = generate_run_key(dataset_version, query_text, context)

            # Check if user has confirmed (for low confidence cases)
            confirmation_key = f"confirmed_analysis:{dataset_version}:{run_key}"
            user_confirmed = st.session_state.get(confirmation_key, False)

            # Auto-execute if high confidence OR user confirmed
            should_auto_execute = confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD or user_confirmed

            logger.info(
                "analysis_execution_decision",
                intent_type=context.inferred_intent.value,
                confidence=confidence,
                threshold=AUTO_EXECUTE_CONFIDENCE_THRESHOLD,
                user_confirmed=user_confirmed,
                should_auto_execute=should_auto_execute,
                run_key=run_key,
                dataset_version=dataset_version,
                query=getattr(context, "research_question", ""),
            )

            if should_auto_execute:
                # Auto-execute with idempotency guard
                st.divider()
                logger.info("analysis_execution_triggered", run_key=run_key, dataset_version=dataset_version)
                execute_analysis_with_idempotency(
                    cohort, context, run_key, dataset_version, getattr(context, "research_question", "")
                )

            else:
                # Low confidence: show QueryPlan confirmation UI
                if query_plan:
                    with st.chat_message("assistant"):
                        st.warning(f"Confidence: {query_plan.confidence:.0%} - Please confirm interpretation")

                        # Show QueryPlan details
                        with st.expander("🔍 How I interpreted your question", expanded=True):
                            st.json(
                                {
                                    "intent": query_plan.intent,
                                    "metric": query_plan.metric,
                                    "group_by": query_plan.group_by,
                                    "filters": [
                                        {
                                            "column": f.column,
                                            "operator": f.operator,
                                            "value": f.value,
                                            "exclude_nulls": f.exclude_nulls,
                                        }
                                        for f in query_plan.filters
                                    ],
                                    "explanation": query_plan.explanation,
                                }
                            )

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Confirm and Run", key=f"confirm_{run_key}", use_container_width=True):
                                st.session_state[confirmation_key] = True
                                st.rerun()
                        with col2:
                            if st.button("Cancel", key=f"cancel_{run_key}", use_container_width=True):
                                st.session_state[confirmation_key] = False
                                st.session_state["analysis_context"] = None
                                st.rerun()
                else:
                    # Fallback: Low confidence without QueryPlan - show detected variables
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
                        primary_display = (
                            get_display_name(context.primary_variable) if context.primary_variable else None
                        )
                        primary_index = (
                            available_cols.index(context.primary_variable)
                            if context.primary_variable in available_cols
                            else 0
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
                        grouping_display = (
                            get_display_name(context.grouping_variable) if context.grouping_variable else None
                        )
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
                            [get_display_name(p) for p in context.predictor_variables]
                            if context.predictor_variables
                            else []
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
                                available_cols.index(context.time_variable)
                                if context.time_variable in available_cols
                                else 0
                            )
                            selected_time = st.selectbox(
                                "**Time Variable**:",
                                options=available_cols,
                                index=time_index if time_index < len(available_cols) else 0,
                                key=f"low_conf_time_{dataset_version}",
                                help=f"Detected: {time_display or context.time_variable}"
                                if context.time_variable
                                else None,
                            )
                            context.time_variable = selected_time
                        with col2:
                            event_options = [c for c in available_cols if c != context.time_variable]
                            event_display = get_display_name(context.event_variable) if context.event_variable else None
                            event_index = (
                                event_options.index(context.event_variable)
                                if context.event_variable in event_options
                                else 0
                            )
                            selected_event = st.selectbox(
                                "**Event Variable**:",
                                options=event_options,
                                index=event_index if event_index < len(event_options) else 0,
                                key=f"low_conf_event_{dataset_version}",
                                help=f"Detected: {event_display or context.event_variable}"
                                if context.event_variable
                                else None,
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
                                elif (
                                    context.inferred_intent == AnalysisIntent.COMPARE_GROUPS
                                    and not context.grouping_variable
                                ):
                                    context.grouping_variable = selected
                                st.info(f"✅ Selected: {get_display_name(selected)}")

                    # Update context in session state after edits
                    st.session_state["analysis_context"] = context

                    # Confirmation button
                    if st.button(CONFIRM_AND_RUN, type="primary", use_container_width=True):
                        st.session_state[confirmation_key] = True
                        st.rerun()

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(START_OVER, use_container_width=True):
                st.session_state["analysis_context"] = None
                st.session_state["intent_signal"] = None
                st.rerun()
        with col2:
            if st.button(CLEAR_RESULTS, use_container_width=True):
                clear_all_results(dataset_version)
                st.success(RESULTS_CLEARED)
                st.rerun()


if __name__ == "__main__":
    main()

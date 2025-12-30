"""
Clinical Analytics Platform - Streamlit UI

Interactive interface for exploring and analyzing clinical datasets.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging ONCE at entry point (not in pages)
from clinical_analytics.ui.logging_config import configure_logging

configure_logging()

# Initialize Ollama LLM service (self-contained, like DuckDB)
from clinical_analytics.ui.ollama_init import initialize_ollama  # noqa: E402

ollama_status = initialize_ollama()
if not ollama_status["ready"]:
    # Log warning but don't block app startup
    import logging  # noqa: E402

    logger = logging.getLogger(__name__)
    logger.info(f"Ollama initialization: {ollama_status['message']}")

# Imports after logging config (intentional - logging must be configured first)
from clinical_analytics.analysis.stats import run_logistic_regression  # noqa: E402
from clinical_analytics.core.profiling import DataProfiler  # noqa: E402
from clinical_analytics.core.schema import UnifiedCohort  # noqa: E402
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory  # noqa: E402
from clinical_analytics.ui.config import V1_MVP_MODE  # noqa: E402
from clinical_analytics.ui.helpers import require_outcome  # noqa: E402


def display_data_profiling(cohort: pd.DataFrame, dataset_name: str):
    """Display data profiling tab with quality metrics."""
    st.subheader("Data Quality Profile")

    try:
        # Generate profile using DataProfiler
        profiler = DataProfiler(cohort)
        profile = profiler.generate_profile()

        # Overview Section
        with st.expander("📋 Dataset Overview", expanded=True):
            overview = profile["overview"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{overview['n_rows']:,}")
            with col2:
                st.metric("Columns", overview["n_columns"])
            with col3:
                st.metric("Memory Usage (MB)", f"{overview['memory_usage_mb']:.2f}")

        # Missing Data Section
        with st.expander("❓ Missing Data Analysis", expanded=True):
            missing = profile["missing_data"]
            if missing["total_missing_cells"] > 0:
                st.warning(
                    f"Total missing values: {missing['total_missing_cells']:,} ({missing['pct_missing_overall']:.2f}%)"
                )

                # Show columns with missing data
                if missing["columns_with_missing"]:
                    missing_df = pd.DataFrame.from_dict(missing["columns_with_missing"], orient="index")
                    missing_df.index.name = "Column"
                    missing_df = missing_df.reset_index()
                    st.dataframe(missing_df)
            else:
                st.success("No missing values detected!")

        # Numeric Features Section
        with st.expander("🔢 Numeric Features"):
            numeric = profile["numeric_features"]
            if numeric:
                for col, stats in numeric.items():
                    st.markdown(f"**{col}**")
                    cols = st.columns(6)
                    cols[0].metric("Mean", f"{stats['mean']:.2f}")
                    cols[1].metric("Std", f"{stats['std']:.2f}")
                    cols[2].metric("Min", f"{stats['min']:.2f}")
                    cols[3].metric("25%", f"{stats['q25']:.2f}")
                    cols[4].metric("Median", f"{stats['median']:.2f}")
                    cols[5].metric("Max", f"{stats['max']:.2f}")
                    st.divider()
            else:
                st.info("No numeric features found")

        # Categorical Features Section
        with st.expander("📊 Categorical Features"):
            categorical = profile["categorical_features"]
            if categorical:
                for col, stats in categorical.items():
                    st.markdown(f"**{col}**")
                    cols = st.columns(3)
                    cols[0].metric("Unique Values", stats["n_unique"])
                    cols[1].metric("Most Common", str(stats["mode"]) if stats["mode"] else "N/A")
                    cols[2].metric("Mode %", f"{stats['pct_mode']:.1f}%")

                    # Show top values if available
                    if stats["top_values"]:
                        with st.expander(f"Top values for {col}"):
                            top_df = pd.DataFrame(list(stats["top_values"].items()), columns=["Value", "Count"])
                            st.dataframe(top_df)
                    st.divider()
            else:
                st.info("No categorical features found")

        # Data Quality Score
        with st.expander("✅ Data Quality Assessment", expanded=True):
            quality = profile["data_quality"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Quality Score", f"{quality['quality_score']:.1f}/100")
            with col2:
                st.metric("Issues Detected", quality["n_issues"])

            st.markdown("**Quality Issues:**")
            if quality["issues"]:
                for issue in quality["issues"]:
                    severity = issue.get("severity", "info")
                    message = issue["message"]
                    if severity == "warning":
                        st.warning(f"⚠️ {message}")
                    else:
                        st.info(f"ℹ️ {message}")
            else:
                st.success("✅ No major quality issues detected!")

    except Exception as e:
        st.error(f"Error generating data profile: {str(e)}")
        st.exception(e)


def display_statistical_analysis(cohort: pd.DataFrame, dataset_name: str):
    """Display statistical analysis tab with regression capabilities."""
    st.subheader("Logistic Regression Analysis")

    # Get available predictors (exclude required schema columns)
    available_predictors = [col for col in cohort.columns if col not in UnifiedCohort.REQUIRED_COLUMNS]

    if available_predictors:
        selected_predictors = st.multiselect(
            "Select Predictor Variables",
            available_predictors,
            default=available_predictors[: min(3, len(available_predictors))],
        )

        if selected_predictors and st.button("Run Logistic Regression"):
            try:
                # Check if outcome exists (required for logistic regression)
                require_outcome(cohort, "Logistic regression")

                # Prepare data for analysis
                analysis_data = prepare_analysis_data(cohort, selected_predictors)

                if analysis_data is not None:
                    model, summary_df = run_logistic_regression(
                        analysis_data, UnifiedCohort.OUTCOME, selected_predictors
                    )

                    st.subheader("Regression Results")

                    # Display summary table
                    st.dataframe(summary_df)

                    # Export regression results
                    col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 3])
                    with col_exp1:
                        csv_results = summary_df.to_csv(index=True)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv_results,
                            file_name=f"{dataset_name}_regression_results.csv",
                            mime="text/csv",
                        )
                    with col_exp2:
                        # Export full model summary as text
                        model_summary_text = str(model.summary())
                        st.download_button(
                            label="Download Full Summary",
                            data=model_summary_text,
                            file_name=f"{dataset_name}_model_summary.txt",
                            mime="text/plain",
                        )

                    # Display model statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Pseudo R²", f"{model.prsquared:.4f}")
                    with col2:
                        st.metric("Log-Likelihood", f"{model.llf:.2f}")

                    # Display model summary in expander
                    with st.expander("Full Model Summary"):
                        st.text(str(model.summary()))

            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                st.exception(e)
    else:
        st.info("No predictor variables available for analysis in this dataset.")


def main():
    st.set_page_config(page_title="Clinical Analytics Platform", page_icon="🏥", layout="wide")

    # V1 MVP: Redirect to Upload page (landing page)
    if V1_MVP_MODE:
        st.switch_page("pages/1_📤_Add_Your_Data.py")
        return

    # Development/testing mode: Keep existing dataset selection
    st.title("🏥 Clinical Analytics Platform")
    st.markdown("Multi-dataset clinical analytics with unified schema")

    # DYNAMIC DATASET DISCOVERY - Only user uploads
    # Only show uploaded datasets, no built-in datasets
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

    # Sidebar for dataset selection
    st.sidebar.header("Dataset Selection")

    if not dataset_display_names:
        st.error("No datasets found! Please upload data using the 'Add Your Data' page.")
        st.info("👈 Go to **Add Your Data** to upload your first dataset")
        return

    dataset_choice_display = st.sidebar.selectbox("Choose Dataset", list(dataset_display_names.keys()))

    # Get internal dataset name
    dataset_choice = dataset_display_names[dataset_choice_display]

    # All datasets are uploaded datasets now
    upload_info = uploaded_datasets[dataset_choice]
    st.sidebar.markdown("**Type:** User Upload")
    st.sidebar.markdown(f"**Uploaded:** {upload_info['upload_timestamp'][:10]}")
    st.sidebar.markdown(f"**Rows:** {upload_info.get('row_count', 'N/A'):,}")

    # Load selected dataset (always uploaded)
    with st.spinner(f"Loading {dataset_choice_display} dataset..."):
        try:
            dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
            dataset.load()
        except Exception as e:
            st.error(f"Failed to load uploaded dataset: {str(e)}")
            return

    # Display dataset info
    st.header(f"📊 {dataset_choice} Dataset")

    # Get cohort data
    try:
        cohort = dataset.get_cohort()

        if cohort.empty:
            st.warning("No data available in the selected dataset.")
            return

        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", len(cohort))
        with col2:
            if UnifiedCohort.OUTCOME in cohort.columns:
                outcome_rate = cohort[UnifiedCohort.OUTCOME].mean() * 100
                st.metric("Outcome Rate", f"{outcome_rate:.1f}%")
            else:
                st.metric("Outcome Rate", "N/A")
        with col3:
            st.metric("Features", len(cohort.columns) - len(UnifiedCohort.REQUIRED_COLUMNS))

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 Overview", "📊 Data Profiling", "📈 Statistical Analysis", "🔍 Query Builder"]
        )

        with tab1:
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(cohort.head(10))

            # Export Section
            st.subheader("📥 Export Data")
            col1, col2, col3 = st.columns([1, 1, 3])

            with col1:
                csv_data = cohort.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{dataset_choice}_cohort.csv",
                    mime="text/csv",
                )

            with col2:
                json_data = cohort.to_json(orient="records", indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{dataset_choice}_cohort.json",
                    mime="application/json",
                )

        with tab2:
            # Data Profiling Tab
            display_data_profiling(cohort, dataset_choice)

        with tab3:
            # Statistical Analysis Tab
            display_statistical_analysis(cohort, dataset_choice)

        with tab4:
            # Query Builder Tab - Config-driven!
            display_query_builder(dataset, dataset_choice)

    except Exception as e:
        st.error(f"Error loading cohort: {str(e)}")
        st.exception(e)


def load_dataset(dataset_name: str):
    """
    Load the selected dataset - only supports user uploads.
    """
    try:
        # All datasets are uploaded datasets
        dataset = UploadedDatasetFactory.create_dataset(dataset_name)

        # Validate and load
        if not dataset.validate():
            st.warning(f"{dataset_name} data not found. Please ensure data files are in the correct location.")
            return None

        dataset.load()
        return dataset

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.exception(e)
        return None


def display_query_builder(dataset, dataset_name: str):
    """Display config-driven query builder using semantic layer."""
    st.subheader("🔍 Query Builder")
    st.markdown("Build custom queries using config-defined metrics and dimensions. SQL generated behind the scenes!")

    # Get semantic layer using contract pattern
    try:
        semantic = dataset.get_semantic_layer()
    except ValueError as e:
        st.error(f"Dataset does not support query builder: {e}")
        return
    dataset_info = semantic.get_dataset_info()

    # Get available metrics and dimensions from config
    metrics = dataset_info.get("metrics", {})
    dimensions = dataset_info.get("dimensions", {})
    filters = dataset_info.get("filters", {})

    if not metrics and not dimensions:
        st.info("No metrics or dimensions defined in config for this dataset.")
        return

    # Metrics selection
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Metrics")
        available_metrics = list(metrics.keys())
        if available_metrics:
            selected_metrics = st.multiselect(
                "Select Metrics",
                available_metrics,
                help="Metrics are aggregated values (rates, counts, averages)",
            )
            # Show metric descriptions
            for metric_name in selected_metrics:
                metric_def = metrics[metric_name]
                st.caption(f"**{metric_def.get('label', metric_name)}**: {metric_def.get('description', '')}")
        else:
            st.info("No metrics defined in config")
            selected_metrics = []

    with col2:
        st.markdown("### Dimensions")
        available_dimensions = list(dimensions.keys())
        if available_dimensions:
            selected_dimensions = st.multiselect(
                "Group By (Dimensions)",
                available_dimensions,
                help="Dimensions are grouping variables (categorical or continuous)",
            )
            # Show dimension info
            for dim_name in selected_dimensions:
                dim_def = dimensions[dim_name]
                st.caption(f"**{dim_def.get('label', dim_name)}**: {dim_def.get('type', 'unknown')} type")
        else:
            st.info("No dimensions defined in config")
            selected_dimensions = []

    # Filters
    st.markdown("### Filters")
    filter_values = {}
    if filters and len(filters) > 0:
        filter_cols = st.columns(min(3, len(filters)))
        for idx, (filter_name, filter_def) in enumerate(filters.items()):
            with filter_cols[idx % len(filter_cols)]:
                filter_type = filter_def.get("type", "equals")
                description = filter_def.get("description", "")

                if filter_type == "equals":
                    # Boolean filter
                    filter_values[filter_name] = st.checkbox(
                        filter_def.get("description", filter_name),
                        value=dataset.config.get("default_filters", {}).get(filter_name, False),
                        help=description,
                    )
    else:
        st.info("No filters defined in config")

    # Execute query
    if st.button("🔍 Run Query", type="primary"):
        if not selected_metrics and not selected_dimensions:
            st.warning("Please select at least one metric or dimension")
            return

        try:
            with st.spinner("Generating SQL and executing query..."):
                # Use semantic layer to build and execute query
                results = semantic.query(
                    metrics=selected_metrics if selected_metrics else None,
                    dimensions=selected_dimensions if selected_dimensions else None,
                    filters=filter_values if filter_values else None,
                    show_sql=True,  # Show generated SQL for transparency
                )

                st.success(f"Query executed successfully! Returned {len(results)} rows")

                # Display results
                st.dataframe(results, width="stretch")

                # Export results
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"{dataset_name}_query_results.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            st.exception(e)

    # Show available config info
    with st.expander("📖 Available Metrics & Dimensions"):
        st.json(
            {
                "metrics": {k: {"label": v.get("label"), "type": v.get("type")} for k, v in metrics.items()},
                "dimensions": {k: {"label": v.get("label"), "type": v.get("type")} for k, v in dimensions.items()},
                "filters": {
                    k: {"type": v.get("type"), "description": v.get("description")} for k, v in filters.items()
                },
            }
        )


def prepare_analysis_data(cohort: pd.DataFrame, predictors: list) -> pd.DataFrame:
    """
    Prepare data for statistical analysis.

    Handles categorical variables and missing data.
    """
    # Select relevant columns - only include OUTCOME if it exists
    analysis_cols = predictors.copy()
    if UnifiedCohort.OUTCOME in cohort.columns:
        analysis_cols = [UnifiedCohort.OUTCOME] + analysis_cols

    # Check if we have any columns to analyze
    available_cols = [col for col in analysis_cols if col in cohort.columns]
    if not available_cols:
        st.error("No valid analysis columns found in dataset.")
        return None

    data = cohort[available_cols].copy()

    # Handle categorical variables - convert to dummy variables
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    if categorical_cols:
        st.info(f"Converting categorical variables to dummy variables: {', '.join(categorical_cols)}")
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Drop rows with missing values
    initial_rows = len(data)
    data = data.dropna()
    dropped_rows = initial_rows - len(data)

    if dropped_rows > 0:
        st.warning(
            f"Dropped {dropped_rows} rows with missing values ({dropped_rows / initial_rows * 100:.1f}% of data)"
        )

    if len(data) == 0:
        st.error("No data remaining after cleaning. Please check your data quality.")
        return None

    return data


if __name__ == "__main__":
    main()

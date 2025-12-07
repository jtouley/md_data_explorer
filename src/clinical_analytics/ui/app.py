"""
Clinical Analytics Platform - Streamlit UI

Interactive interface for exploring and analyzing clinical datasets.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.analysis.stats import run_logistic_regression
from clinical_analytics.core.schema import UnifiedCohort


def main():
    st.set_page_config(
        page_title="Clinical Analytics Platform",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Clinical Analytics Platform")
    st.markdown("Multi-dataset clinical analytics with unified schema")

    # DYNAMIC DATASET DISCOVERY - No more hardcoding!
    # Auto-discover all available datasets from registry
    available_datasets = DatasetRegistry.list_datasets()
    dataset_info = DatasetRegistry.get_all_dataset_info()

    # Build display names from config
    dataset_display_names = {}
    for ds_name in available_datasets:
        info = dataset_info[ds_name]
        display_name = info['config'].get('display_name', ds_name.replace('_', '-').upper())
        dataset_display_names[display_name] = ds_name

    # Sidebar for dataset selection
    st.sidebar.header("Dataset Selection")

    if not dataset_display_names:
        st.error("No datasets found! Please check your dataset implementations.")
        return

    dataset_choice_display = st.sidebar.selectbox(
        "Choose Dataset",
        list(dataset_display_names.keys())
    )

    # Get internal dataset name
    dataset_choice = dataset_display_names[dataset_choice_display]

    # Show dataset info in sidebar
    info = dataset_info[dataset_choice]
    st.sidebar.markdown(f"**Status:** {info['config'].get('status', 'unknown')}")
    st.sidebar.markdown(f"**Source:** {info['config'].get('source', 'N/A')}")

    # Load selected dataset using registry (no more if/else!)
    with st.spinner(f"Loading {dataset_choice_display} dataset..."):
        dataset = load_dataset(dataset_choice)

        if dataset is None:
            st.error(f"Failed to load {dataset_choice} dataset. Please check data availability.")
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
            outcome_rate = cohort[UnifiedCohort.OUTCOME].mean() * 100
            st.metric("Outcome Rate", f"{outcome_rate:.1f}%")
        with col3:
            st.metric("Features", len(cohort.columns) - len(UnifiedCohort.REQUIRED_COLUMNS))

        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(cohort.head(10))

        # Statistical Analysis Section
        st.header("📈 Statistical Analysis")

        # Get available predictors (exclude required schema columns)
        available_predictors = [
            col for col in cohort.columns
            if col not in UnifiedCohort.REQUIRED_COLUMNS
        ]

        if available_predictors:
            selected_predictors = st.multiselect(
                "Select Predictor Variables",
                available_predictors,
                default=available_predictors[:min(3, len(available_predictors))]
            )

            if selected_predictors and st.button("Run Logistic Regression"):
                try:
                    # Prepare data for analysis
                    analysis_data = prepare_analysis_data(cohort, selected_predictors)

                    if analysis_data is not None:
                        model, summary_df = run_logistic_regression(
                            analysis_data,
                            UnifiedCohort.OUTCOME,
                            selected_predictors
                        )

                        st.subheader("Regression Results")

                        # Display summary table
                        st.dataframe(summary_df)

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

    except Exception as e:
        st.error(f"Error loading cohort: {str(e)}")
        st.exception(e)


def load_dataset(dataset_name: str):
    """
    Load the selected dataset using registry factory method.

    NO MORE HARDCODED IF/ELSE! Registry auto-discovers and instantiates.
    """
    try:
        # Use registry factory - completely dynamic
        dataset = DatasetRegistry.get_dataset(dataset_name)

        # Validate and load
        if not dataset.validate():
            st.warning(f"{dataset_name} data not found. Please ensure data files are in the correct location.")
            return None

        dataset.load()
        return dataset

    except KeyError as e:
        st.error(f"Dataset '{dataset_name}' not found in registry: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.exception(e)
        return None


def prepare_analysis_data(cohort: pd.DataFrame, predictors: list) -> pd.DataFrame:
    """
    Prepare data for statistical analysis.

    Handles categorical variables and missing data.
    """
    # Select relevant columns
    analysis_cols = [UnifiedCohort.OUTCOME] + predictors
    data = cohort[analysis_cols].copy()

    # Handle categorical variables - convert to dummy variables
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_cols:
        st.info(f"Converting categorical variables to dummy variables: {', '.join(categorical_cols)}")
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Drop rows with missing values
    initial_rows = len(data)
    data = data.dropna()
    dropped_rows = initial_rows - len(data)

    if dropped_rows > 0:
        st.warning(f"Dropped {dropped_rows} rows with missing values ({dropped_rows/initial_rows*100:.1f}% of data)")

    if len(data) == 0:
        st.error("No data remaining after cleaning. Please check your data quality.")
        return None

    return data


if __name__ == "__main__":
    main()

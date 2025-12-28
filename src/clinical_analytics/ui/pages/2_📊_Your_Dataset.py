"""
Your Dataset - Overview and Summary

See your data at a glance: patient counts, outcomes, data quality.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
from clinical_analytics.ui.config import MULTI_TABLE_ENABLED

# Page config
st.set_page_config(page_title="Your Dataset | Clinical Analytics", page_icon="📊", layout="wide")


def main():
    st.title("📊 Your Dataset")
    st.markdown("""
    Overview of your dataset: patient counts, outcomes, and data quality.
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
        st.error("No datasets available. Please upload data first.")
        st.info("👈 Go to **Add Your Data** to upload your first dataset")
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

            cohort = dataset.get_cohort()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

    # Show Semantic Scope in sidebar
    with st.sidebar.expander("🔍 Semantic Scope", expanded=False):
        st.markdown("**V1 Cohort-First Mode**")

        # Cohort table status
        st.markdown(f"✅ **Cohort Table**: {len(cohort):,} rows")

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
        st.caption(f"{len(cohort.columns)} columns available")

    st.divider()

    # Dataset Overview Metrics
    st.markdown("## 📈 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", f"{len(cohort):,}")
    with col2:
        st.metric("Variables", len(cohort.columns))
    with col3:
        missing_pct = (cohort.isna().sum().sum() / cohort.size) * 100
        st.metric("Data Completeness", f"{100 - missing_pct:.1f}%")
    with col4:
        # Outcome prevalence if available
        if outcome_cols:
            outcome_col = outcome_cols[0]
            if pd.api.types.is_numeric_dtype(cohort[outcome_col]):
                prevalence = cohort[outcome_col].mean() * 100
                st.metric("Outcome Rate", f"{prevalence:.1f}%")
            else:
                st.metric("Outcome Rate", "N/A")
        else:
            st.metric("Outcome Rate", "N/A")

    # Show data quality warnings if available
    try:
        # Try to get semantic layer
        semantic = dataset.get_semantic_layer()
        warnings = semantic.get_data_quality_warnings()
        if warnings:
            st.divider()
            st.markdown("## ⚠️ Data Quality Warnings")
            st.warning(f"Found {len(warnings)} data quality warning(s)")
            with st.expander("View Warnings", expanded=True):
                for warning in warnings:
                    st.warning(f"**{warning['type']}**: {warning['message']}")
    except (ValueError, AttributeError):
        # Dataset doesn't support semantic layer or no warnings available
        pass

    # Summary statistics
    st.divider()
    st.markdown("## 📊 Summary Statistics")

    tab1, tab2 = st.tabs(["Numeric Variables", "Categorical Variables"])

    with tab1:
        numeric_cols = cohort.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            desc_stats = cohort[numeric_cols].describe().T
            st.dataframe(desc_stats, use_container_width=True)
        else:
            st.info("No numeric variables found")

    with tab2:
        categorical_cols = cohort.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            for col in categorical_cols[:10]:  # Limit to first 10
                value_counts = cohort[col].value_counts()
                st.markdown(f"**{col}**")
                for value, count in value_counts.head(5).items():
                    pct = count / len(cohort) * 100
                    st.write(f"  - {value}: {count} ({pct:.1f}%)")
                st.divider()
        else:
            st.info("No categorical variables found")

    # Data preview
    st.divider()
    st.markdown("## 🔍 Data Preview")
    st.dataframe(cohort.head(20), use_container_width=True)


if __name__ == "__main__":
    main()

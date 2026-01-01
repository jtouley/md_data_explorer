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

from clinical_analytics.ui.components.dataset_loader import render_dataset_selector

# Page config
st.set_page_config(page_title="Your Dataset | Clinical Analytics", page_icon="📊", layout="wide")


def main():
    st.title("📊 Your Dataset")
    st.markdown("""
    Overview of your dataset: patient counts, outcomes, and data quality.
    """)

    # Dataset selection (Phase 8.2: Use reusable component)
    result = render_dataset_selector(show_semantic_scope=True)
    if result is None:
        return  # No datasets available (error message already shown)

    dataset, cohort, dataset_choice, dataset_version = result

    # Detect outcome columns (needed for metrics below)
    outcome_cols = [
        c for c in cohort.columns if "outcome" in c.lower() or "death" in c.lower() or "mortality" in c.lower()
    ]

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
            st.dataframe(desc_stats, width="stretch")
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
    st.dataframe(cohort.head(20), width="stretch")


if __name__ == "__main__":
    main()

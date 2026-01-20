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
from clinical_analytics.ui.components.enrichment_integration import (
    EnrichmentService,
    format_enrichment_button_text,
    get_confidence_badge_color,
)
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

# Page config
st.set_page_config(page_title="Your Dataset | Clinical Analytics", page_icon="📊", layout="wide")

# Initialize storage and enrichment service
storage = UserDatasetStorage()
overlay_store_path = storage.upload_dir / "metadata"


def get_enrichment_service():
    """Get or create enrichment service."""
    from clinical_analytics.core.overlay_store import OverlayStore

    return EnrichmentService(overlay_store=OverlayStore(base_dir=overlay_store_path))


def render_enrichment_section(dataset, dataset_choice: str, dataset_version: str):
    """Render the metadata enrichment section."""
    st.divider()
    st.markdown("## 🧠 Metadata Enrichment")
    st.markdown(
        "Enhance your dataset metadata using AI-powered suggestions. "
        "Review and approve each suggestion before it's applied."
    )

    service = get_enrichment_service()
    upload_id = dataset_choice

    # Get current stats
    stats = service.get_enrichment_stats(upload_id, dataset_version)
    pending_count = stats["pending_count"]
    accepted_count = stats["accepted_count"]

    # Show stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pending Suggestions", pending_count)
    with col2:
        st.metric("Accepted", accepted_count)
    with col3:
        st.metric("Rejected", stats["rejected_count"])

    # Check if generating (stored in session state)
    is_generating = st.session_state.get("enrichment_generating", False)

    # Show trigger button or pending review
    if pending_count == 0 and not is_generating:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(
                format_enrichment_button_text(0, False),
                type="primary",
                key="enrich_trigger",
            ):
                st.session_state["enrichment_generating"] = True
                st.rerun()
        with col2:
            st.caption("AI will analyze your schema and suggest metadata improvements.")

    elif is_generating:
        with st.spinner("Generating metadata suggestions..."):
            try:
                # Get schema from dataset
                semantic = dataset.get_semantic_layer()
                inferred_schema = getattr(semantic, "inferred_schema", None)

                if inferred_schema is None:
                    # Reconstruct from semantic layer
                    from clinical_analytics.core.schema_inference import InferredSchema

                    inferred_schema = InferredSchema(
                        patient_id_column=semantic.config.get("column_mapping", {}).get("patient_id"),
                        outcome_columns=list(semantic.config.get("outcomes", {}).keys()),
                    )

                suggestions = service.trigger_enrichment(
                    upload_id=upload_id,
                    version=dataset_version,
                    schema=inferred_schema,
                )

                st.session_state["enrichment_generating"] = False

                if suggestions:
                    st.success(f"Generated {len(suggestions)} suggestions!")
                else:
                    st.info("No suggestions generated. Schema may already be well-described.")

                st.rerun()

            except Exception as e:
                st.session_state["enrichment_generating"] = False
                st.error(f"Error generating suggestions: {e}")

    else:
        # Show pending suggestions for review
        st.markdown("### Review Suggestions")
        pending = service.get_pending_suggestions(upload_id, dataset_version)

        for patch in pending:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 3, 1, 1])

                with col1:
                    st.markdown(f"**{patch.column}**")
                    if hasattr(patch, "operation"):
                        st.caption(patch.operation.value.replace("_", " ").title())

                with col2:
                    st.markdown(f"→ {patch.value}")
                    if hasattr(patch, "confidence") and patch.confidence:
                        color = get_confidence_badge_color(patch.confidence)
                        st.caption(f"Confidence: {patch.confidence:.0%} ({color})")

                with col3:
                    if st.button("✓", key=f"accept_{patch.patch_id}", help="Accept"):
                        service.accept_suggestion(upload_id, dataset_version, patch.patch_id, "user")
                        st.rerun()

                with col4:
                    if st.button("✗", key=f"reject_{patch.patch_id}", help="Reject"):
                        service.reject_suggestion(upload_id, dataset_version, patch.patch_id, "User rejected")
                        st.rerun()

                st.divider()

        # Batch actions
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Accept All", key="accept_all"):
                service.accept_all_suggestions(upload_id, dataset_version, "user")
                st.success("All suggestions accepted!")
                st.rerun()
        with col2:
            if st.button("Reject All", key="reject_all"):
                service.reject_all_suggestions(upload_id, dataset_version, "User rejected all")
                st.info("All suggestions rejected.")
                st.rerun()


def main():
    st.title("📊 Your Dataset")
    st.markdown(
        """
    Overview of your dataset: patient counts, outcomes, and data quality.
    """
    )

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

    # Metadata Enrichment Section (ADR011)
    render_enrichment_section(dataset, dataset_choice, dataset_version)

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

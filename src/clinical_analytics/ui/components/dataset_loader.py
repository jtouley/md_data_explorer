"""
Dataset Loader Component (Phase 8.2).

Reusable component for loading and selecting datasets across UI pages.
Eliminates 350-560 lines of duplication across 7 pages.

Usage:
    from clinical_analytics.ui.components.dataset_loader import render_dataset_selector

    # In your Streamlit page
    result = render_dataset_selector(show_semantic_scope=True)
    if result is None:
        return  # No datasets available

    dataset, cohort, dataset_choice, dataset_version = result
    # Now use dataset and cohort in your page
"""

import streamlit as st

from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
from clinical_analytics.ui.config import MULTI_TABLE_ENABLED


def render_dataset_selector(
    show_semantic_scope: bool = True,
    sidebar_header: str = "Data Selection",
) -> tuple | None:
    """
    Render dataset selector with loading and optional semantic scope display.

    This component encapsulates the common pattern of:
    1. Loading available datasets
    2. Rendering dataset selection widget
    3. Loading the selected dataset
    4. Optionally showing semantic scope information

    Args:
        show_semantic_scope: Whether to show the semantic scope expander (default: True)
        sidebar_header: Header text for the sidebar section (default: "Data Selection")

    Returns:
        tuple[dataset, cohort, dataset_choice, dataset_version] if successful
        None if no datasets available (and displays error message)

    Example:
        ```python
        result = render_dataset_selector()
        if result is None:
            return

        dataset, cohort, dataset_choice, dataset_version = result
        # Use dataset and cohort in your analysis
        ```
    """
    # Sidebar header
    st.sidebar.header(sidebar_header)

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

    # Check if any datasets available
    if not dataset_display_names:
        st.error("No datasets found! Please upload data using the 'Add Your Data' page.")
        st.info("👈 Go to **Add Your Data** to upload your first dataset")
        return None

    # Dataset selection widget
    dataset_choice_display = st.sidebar.selectbox("Choose Dataset", list(dataset_display_names.keys()))
    dataset_choice = dataset_display_names[dataset_choice_display]

    # Load dataset
    with st.spinner(f"Loading {dataset_choice_display}..."):
        try:
            dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
            dataset.load()
            cohort = dataset.get_cohort()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

    # Get dataset version for lifecycle management
    # All datasets are uploaded datasets: use upload_id as version
    dataset_version = dataset_choice  # This is the upload_id

    # Optionally show semantic scope
    if show_semantic_scope:
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

    return dataset, cohort, dataset_choice, dataset_version

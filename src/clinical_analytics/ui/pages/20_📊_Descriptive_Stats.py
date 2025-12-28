"""
Descriptive Statistics Page

Create Table 1 with patient characteristics and summary statistics.
No statistical jargon - just clear descriptions of your data.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Keep only lightweight imports at module scope
from clinical_analytics.core.schema import UnifiedCohort

# Heavy imports moved inside main() after gate

# Page config
st.set_page_config(page_title="Descriptive Statistics | Clinical Analytics", page_icon="📊", layout="wide")


def generate_table_one(df: pd.DataFrame, stratify_by: str = None) -> pd.DataFrame:
    """
    Generate Table 1 (demographic/characteristic table).

    Args:
        df: DataFrame with patient data
        stratify_by: Optional column to stratify by (e.g., treatment group)

    Returns:
        Formatted Table 1 as DataFrame
    """
    results = []

    # Get stratification groups
    if stratify_by and stratify_by in df.columns:
        groups = sorted(df[stratify_by].dropna().unique())
        overall = False
    else:
        groups = ["Overall"]
        stratify_by = None
        overall = True

    for col in df.columns:
        # Skip the stratification column itself
        if col == stratify_by:
            continue

        # Skip system columns
        if col in [UnifiedCohort.PATIENT_ID, UnifiedCohort.TIME_ZERO]:
            continue

        # Determine variable type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric variable
            n_unique = df[col].nunique()

            if n_unique == 2:
                # Binary numeric (0/1)
                var_type = "binary"
            elif n_unique <= 10:
                # Categorical numeric
                var_type = "categorical"
            else:
                # Continuous
                var_type = "continuous"
        else:
            # Non-numeric
            var_type = "categorical"

        # Generate statistics based on type
        if var_type == "continuous":
            row = {"Variable": col, "Type": "Continuous"}

            for group in groups:
                if overall:
                    data = df[col].dropna()
                else:
                    data = df[df[stratify_by] == group][col].dropna()

                if len(data) > 0:
                    mean = data.mean()
                    std = data.std()
                    row[str(group)] = f"{mean:.1f} ± {std:.1f}"
                else:
                    row[str(group)] = "N/A"

            results.append(row)

        elif var_type == "categorical" or var_type == "binary":
            # Get value counts
            values = df[col].dropna().unique()

            for value in sorted(values):
                row = {"Variable": f"{col}: {value}", "Type": "Categorical"}

                for group in groups:
                    if overall:
                        data = df[col]
                    else:
                        data = df[df[stratify_by] == group][col]

                    total = len(data.dropna())
                    count = (data == value).sum()
                    pct = (count / total * 100) if total > 0 else 0

                    row[str(group)] = f"{count} ({pct:.1f}%)"

                results.append(row)

    # Create DataFrame
    table_one = pd.DataFrame(results)

    # Add sample size row at top
    size_row = {"Variable": "N (Sample Size)", "Type": "Count"}
    for group in groups:
        if overall:
            size_row[str(group)] = str(len(df))
        else:
            size_row[str(group)] = str(len(df[df[stratify_by] == group]))

    table_one = pd.concat([pd.DataFrame([size_row]), table_one], ignore_index=True)

    return table_one


def main():
    # Gate: V1 MVP mode disables legacy pages
    # MUST run before any expensive operations
    from clinical_analytics.ui.config import V1_MVP_MODE

    if V1_MVP_MODE:
        st.info("🚧 This page is disabled in V1 MVP mode. Use the **Ask Questions** page for all analysis.")
        st.markdown("""
        **V1 MVP focuses on:**
        - Upload your data
        - Ask questions in natural language
        - Get answers with SQL preview

        All analysis is available through the Chat interface on the Ask Questions page.
        """)
        if st.button("Go to Ask Questions Page"):
            st.switch_page("pages/3_💬_Ask_Questions.py")
        st.stop()

    # NOW do heavy imports (after gate)
    from clinical_analytics.core.registry import DatasetRegistry
    from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory

    st.title("📊 Descriptive Statistics")
    st.markdown("""
    Create **Table 1** with patient characteristics - the foundation of any research paper.
    No statistical tests, just clear descriptions of your data.
    """)

    # Dataset selection in sidebar
    st.sidebar.header("Data Selection")

    # Load available datasets
    available_datasets = DatasetRegistry.list_datasets()
    dataset_info = DatasetRegistry.get_all_dataset_info()

    # Build display names
    dataset_display_names = {}
    for ds_name in available_datasets:
        info = dataset_info[ds_name]
        display_name = info["config"].get("display_name", ds_name.replace("_", "-").upper())
        dataset_display_names[display_name] = ds_name

    # Add uploaded datasets
    uploaded_datasets = {}
    uploaded_ids = set()  # Track upload IDs for detection
    try:
        uploads = UploadedDatasetFactory.list_available_uploads()
        for upload in uploads:
            upload_id = upload["upload_id"]
            dataset_name = upload.get("dataset_name", upload_id)
            display_name = f"📤 {dataset_name}"
            dataset_display_names[display_name] = upload_id
            uploaded_datasets[upload_id] = upload
            uploaded_ids.add(upload_id)
    except Exception as e:
        st.sidebar.warning(f"Could not load uploads: {e}")

    if not dataset_display_names:
        st.error("No datasets available. Please upload data first.")
        return

    dataset_choice_display = st.sidebar.selectbox("Choose Dataset", list(dataset_display_names.keys()))

    dataset_choice = dataset_display_names[dataset_choice_display]

    # Check if this is an uploaded dataset
    # Method 1: Check if in uploaded_datasets dict
    # Method 2: Check if display name starts with 📤
    # Method 3: Check if dataset_choice is an upload_id (UUID-like or matches upload pattern)
    is_uploaded = (
        dataset_choice in uploaded_datasets or dataset_choice_display.startswith("📤") or dataset_choice in uploaded_ids
    )

    # Load dataset
    with st.spinner(f"Loading {dataset_choice_display}..."):
        try:
            if is_uploaded:
                # For uploaded datasets, use the factory
                upload_id = dataset_choice  # dataset_choice is the upload_id for uploaded datasets
                try:
                    dataset = UploadedDatasetFactory.create_dataset(upload_id)
                    dataset.load()
                except Exception as e:
                    st.error(f"Error loading uploaded dataset: {str(e)}")
                    st.exception(e)
                    return
            else:
                # For built-in datasets, use the registry
                try:
                    dataset = DatasetRegistry.get_dataset(dataset_choice)
                    if not dataset.validate():
                        st.error("Dataset validation failed")
                        return
                    dataset.load()
                except KeyError:
                    # Dataset not found in registry - might be an uploaded dataset
                    st.error(f"Dataset '{dataset_choice}' not found in registry.")
                    st.info("💡 If this is an uploaded dataset, please refresh the page or check the upload status.")
                    return
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
                    st.exception(e)
                    return

            cohort = dataset.get_cohort()

            if cohort.empty:
                st.error("No data in dataset")
                return

        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.exception(e)
            return

    # Show data overview
    st.markdown("## 📋 Data Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", f"{len(cohort):,}")
    with col2:
        st.metric("Variables", len(cohort.columns))
    with col3:
        missing_pct = (cohort.isna().sum().sum() / cohort.size) * 100
        st.metric("Complete Data", f"{100 - missing_pct:.1f}%")

    # Configuration
    st.markdown("## ⚙️ Table Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Select variables to include
        available_vars = [c for c in cohort.columns if c not in [UnifiedCohort.PATIENT_ID, UnifiedCohort.TIME_ZERO]]

        selected_vars = st.multiselect(
            "Variables to Include",
            available_vars,
            default=available_vars,
            help="Select which variables to include in Table 1",
        )

    with col2:
        # Stratification option
        stratify_by = st.selectbox(
            "Stratify By (Optional)",
            ["None"] + list(cohort.columns),
            help="Split table by groups (e.g., treatment vs control)",
        )

        if stratify_by == "None":
            stratify_by = None

    # Generate Table 1
    if st.button("📊 Generate Table 1", type="primary"):
        if not selected_vars:
            st.warning("Please select at least one variable")
            return

        with st.spinner("Generating descriptive statistics..."):
            # Filter data to selected variables
            if stratify_by:
                analysis_df = cohort[[stratify_by] + selected_vars].copy()
            else:
                analysis_df = cohort[selected_vars].copy()

            # Generate table
            table_one = generate_table_one(analysis_df, stratify_by=stratify_by)

            st.success("✅ Table 1 generated successfully!")

            # Display table
            st.markdown("## 📄 Table 1: Patient Characteristics")

            st.dataframe(table_one, use_container_width=True, hide_index=True)

            # Export options
            st.markdown("### 📥 Export Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                # CSV export
                csv_data = table_one.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"table1_{dataset_choice}.csv",
                    mime="text/csv",
                )

            with col2:
                # Excel export
                try:
                    from io import BytesIO

                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        table_one.to_excel(writer, index=False, sheet_name="Table 1")
                    buffer.seek(0)

                    st.download_button(
                        label="Download Excel",
                        data=buffer,
                        file_name=f"table1_{dataset_choice}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except ImportError:
                    st.caption("Excel export requires openpyxl")

            with col3:
                # Formatted text for Word
                formatted_text = "Table 1. Patient Characteristics\n\n"
                formatted_text += table_one.to_string(index=False)

                st.download_button(
                    label="Download as Text",
                    data=formatted_text,
                    file_name=f"table1_{dataset_choice}.txt",
                    mime="text/plain",
                )

            # Methods text
            with st.expander("📝 Methods Section Text"):
                methods_text = """
**Statistical Analysis**

Descriptive statistics were calculated for all variables. Continuous variables are
presented as mean ± standard deviation. Categorical variables are presented as
frequencies and percentages.
"""
                if stratify_by:
                    methods_text += f" Data are stratified by {stratify_by}."

                methods_text += " All analyses were performed using Clinical Analytics Platform."

                st.markdown(methods_text)

                st.download_button(
                    label="Copy Methods Text",
                    data=methods_text,
                    file_name="methods_descriptive.txt",
                    mime="text/plain",
                )

            # Interpretation guide
            with st.expander("📖 How to Read Table 1"):
                st.markdown("""
                **Continuous Variables** (e.g., Age, Weight):
                - Shown as: Mean ± Standard Deviation
                - Example: "45.3 ± 12.1" means average age is 45.3 years,
                  with most patients between 33.2 and 57.4 years

                **Categorical Variables** (e.g., Sex, Treatment):
                - Shown as: Count (Percentage)
                - Example: "75 (60.0%)" means 75 patients (60% of total) have this characteristic

                **Sample Size (N)**:
                - Total number of patients in each group
                """)


if __name__ == "__main__":
    main()

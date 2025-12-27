"""
Upload Data Page

Self-service data upload for clinicians.
Upload CSV, Excel, or SPSS files without code or YAML configuration.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from clinical_analytics.ui.components.data_validator import DataQualityValidator
from clinical_analytics.ui.components.variable_detector import VariableTypeDetector
from clinical_analytics.ui.components.variable_mapper import VariableMappingWizard
from clinical_analytics.ui.storage.user_datasets import UploadSecurityValidator, UserDatasetStorage

# Page configuration
st.set_page_config(
    page_title="Upload Data | Clinical Analytics",
    page_icon="📤",
    layout="wide"
)

# Initialize storage
storage = UserDatasetStorage()


def render_upload_step():
    """Step 1: File Upload"""
    st.markdown("## 📤 Upload Your Data")
    st.markdown("""
    Upload your clinical dataset in CSV, Excel, SPSS, or ZIP format.

    **Supported formats:**
    - CSV (`.csv`) - Single table
    - Excel (`.xlsx`, `.xls`) - Single table
    - SPSS (`.sav`) - Single table
    - **ZIP (`.zip`) - Multi-table datasets** ⭐ NEW!

    **Requirements:**
    - File size: 1KB - 100MB
    - Must include patient ID column
    - Must include outcome variable

    **Multi-Table ZIP Format:**
    - ZIP file containing multiple CSV files
    - Tables will be automatically joined (e.g., MIMIC-IV demo)
    - Relationships detected via foreign keys
    """)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'sav', 'zip'],
        help="Maximum file size: 100MB. ZIP files for multi-table datasets.",
        key="file_uploader"
    )

    if uploaded_file is not None:
        # Show file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("Type", Path(uploaded_file.name).suffix.upper())

        # Validate file
        file_bytes = uploaded_file.getvalue()
        valid, error_msg = UploadSecurityValidator.validate(uploaded_file.name, file_bytes)

        if not valid:
            st.error(f"❌ {error_msg}")
            return None

        st.success("✅ File validation passed")

        # Check if ZIP file (multi-table)
        file_ext = Path(uploaded_file.name).suffix.lower()

        if file_ext == '.zip':
            # Handle multi-table ZIP upload
            st.info("🗂️ **Multi-table dataset detected!** ZIP file validated.")

            # Store ZIP upload details
            st.session_state['is_zip_upload'] = True
            st.session_state['uploaded_filename'] = uploaded_file.name
            st.session_state['uploaded_bytes'] = file_bytes

            st.success("✅ ZIP file ready for processing")
            st.info("💡 **Next:** Click 'Continue to Review' to process tables, detect relationships, and build unified cohort.")

            # Button to proceed to review step
            if st.button("Continue to Review ➡️", type="primary"):
                st.session_state['upload_step'] = 5
                st.rerun()

            return uploaded_file

        # Try to load preview for single-table files
        try:
            if file_ext == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext in {'.xlsx', '.xls'}:
                df = pd.read_excel(uploaded_file)
            elif file_ext == '.sav':
                import pyreadstat
                df, meta = pyreadstat.read_sav(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_ext}")
                return None

            # Store in session state (single-table)
            st.session_state['is_zip_upload'] = False
            st.session_state['uploaded_df'] = df
            st.session_state['uploaded_filename'] = uploaded_file.name
            st.session_state['uploaded_bytes'] = file_bytes
            st.session_state['upload_step'] = 2

            return df

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.exception(e)
            return None

    return None


def render_preview_step(df: pd.DataFrame):
    """Step 2: Data Preview & Quality Check"""
    st.markdown("## 👀 Data Preview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        missing_pct = (df.isna().sum().sum() / df.size) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory", f"{memory_mb:.1f} MB")

    # Show data preview
    st.markdown("### First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    # Data quality validation
    st.markdown("### 🔍 Data Quality Check")

    with st.spinner("Running quality checks..."):
        validation_result = DataQualityValidator.validate_complete(df)

    if validation_result['is_valid']:
        st.success("✅ Data quality check passed!")
    else:
        st.warning(f"⚠️ Found {validation_result['summary']['errors']} error(s) and {validation_result['summary']['warnings']} warning(s)")

    # Show issues
    if validation_result['issues']:
        with st.expander(f"View Issues ({len(validation_result['issues'])})"):
            for issue in validation_result['issues']:
                severity = issue['severity']
                message = issue['message']

                if severity == 'error':
                    st.error(f"❌ **{issue['type']}**: {message}")
                else:
                    st.warning(f"⚠️ **{issue['type']}**: {message}")

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Upload"):
            st.session_state['upload_step'] = 1
            st.rerun()
    with col2:
        # Can proceed even with warnings, but not with errors
        can_proceed = validation_result['is_valid'] or validation_result['summary']['errors'] == 0

        if st.button("Continue to Variable Detection ➡️", disabled=not can_proceed, type="primary"):
            st.session_state['upload_step'] = 3
            st.rerun()

    if not can_proceed:
        st.error("⚠️ Please fix critical errors before continuing")


def render_variable_detection_step(df: pd.DataFrame):
    """Step 3: Variable Type Detection"""
    st.markdown("## 🔬 Variable Type Detection")
    st.markdown("Automatically detecting variable types from your data...")

    with st.spinner("Analyzing variables..."):
        # Detect variable types
        variable_info = VariableTypeDetector.detect_all_variables(df)
        suggestions = VariableTypeDetector.suggest_schema_mapping(df)

    # Store in session state
    st.session_state['variable_info'] = variable_info
    st.session_state['suggestions'] = suggestions

    # Show detection results
    st.markdown("### Detection Results")

    # Group by type
    type_groups = {}
    for col, info in variable_info.items():
        var_type = info['type']
        if var_type not in type_groups:
            type_groups[var_type] = []
        type_groups[var_type].append((col, info))

    # Display in columns
    type_emojis = {
        'continuous': '📊',
        'categorical': '🏷️',
        'binary': '🔀',
        'datetime': '📅',
        'id': '🆔'
    }

    for var_type, cols_info in type_groups.items():
        with st.expander(f"{type_emojis.get(var_type, '📌')} {var_type.upper()} Variables ({len(cols_info)})"):
            for col, info in cols_info:
                st.markdown(f"**{col}**")

                metadata = info['metadata']
                missing_pct = info['missing_pct']

                col1, col2 = st.columns([3, 1])
                with col1:
                    # Show relevant metadata
                    if var_type == 'continuous':
                        st.caption(f"Range: {metadata['min']:.2f} to {metadata['max']:.2f}, Mean: {metadata['mean']:.2f}")
                    elif var_type in ['categorical', 'binary']:
                        values = metadata.get('values', [])
                        if len(values) <= 10:
                            st.caption(f"Values: {', '.join(str(v) for v in values)}")
                        else:
                            st.caption(f"{metadata['unique_count']} unique values")

                with col2:
                    if missing_pct > 0:
                        st.caption(f"⚠️ {missing_pct:.1f}% missing")
                    else:
                        st.caption("✅ Complete")

                st.divider()

    # Show suggestions
    if any(suggestions.values()):
        st.markdown("### 💡 Suggestions")

        col1, col2, col3 = st.columns(3)
        with col1:
            if suggestions.get('patient_id'):
                st.info(f"**Patient ID:** `{suggestions['patient_id']}`")
        with col2:
            if suggestions.get('outcome'):
                st.info(f"**Outcome:** `{suggestions['outcome']}`")
        with col3:
            if suggestions.get('time_zero'):
                st.info(f"**Time Variable:** `{suggestions['time_zero']}`")

    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Preview"):
            st.session_state['upload_step'] = 2
            st.rerun()
    with col2:
        if st.button("Continue to Mapping ➡️", type="primary"):
            st.session_state['upload_step'] = 4
            st.rerun()


def render_mapping_step(df: pd.DataFrame, variable_info: dict, suggestions: dict):
    """Step 4: Variable Mapping Wizard"""
    st.markdown("## 🗺️ Map Variables to Analysis Schema")

    # Render the wizard
    mapping = VariableMappingWizard.render_complete_wizard(
        df=df,
        variable_info=variable_info,
        suggestions=suggestions,
        key_prefix="upload"
    )

    if mapping:
        # Store mapping in session state
        st.session_state['variable_mapping'] = mapping

        st.divider()

        # Show mapping summary
        with st.expander("📋 Mapping Summary", expanded=True):
            st.markdown(f"**Patient ID:** `{mapping['patient_id']}`")
            st.markdown(f"**Outcome:** `{mapping['outcome']}`")

            if mapping['time_variables']:
                st.markdown("**Time Variables:**")
                for key, val in mapping['time_variables'].items():
                    if val:
                        st.markdown(f"- {key}: `{val}`")

            st.markdown(f"**Predictors:** {len(mapping['predictors'])} variables")
            if mapping['predictors']:
                st.caption(', '.join(mapping['predictors'][:10]) + ('...' if len(mapping['predictors']) > 10 else ''))

            if mapping['excluded']:
                st.markdown(f"**Excluded:** {len(mapping['excluded'])} variables")

        # Navigation
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Detection"):
                st.session_state['upload_step'] = 3
                st.rerun()
        with col2:
            if st.button("Continue to Review ➡️", type="primary"):
                st.session_state['upload_step'] = 5
                st.rerun()


def render_review_step(df: pd.DataFrame = None, mapping: dict = None, variable_info: dict = None):
    """Step 5: Final Review & Save"""
    st.markdown("## ✅ Review & Save Dataset")

    # Check if this is a ZIP upload (multi-table)
    is_zip = st.session_state.get('is_zip_upload', False)

    if is_zip:
        # Handle multi-table ZIP upload
        return render_zip_review_step()

    # Single-table workflow (existing logic)
    if df is None or mapping is None or variable_info is None:
        st.warning("Missing required data. Please start over.")
        st.session_state['upload_step'] = 1
        st.rerun()
        return

    # Run final validation with mapping
    patient_id_col = mapping['patient_id']
    outcome_col = mapping['outcome']

    validation_result = DataQualityValidator.validate_complete(
        df,
        id_column=patient_id_col,
        outcome_column=outcome_col
    )

    # Show validation results
    if validation_result['is_valid']:
        st.success("✅ All validation checks passed! Dataset is ready to use.")
    else:
        error_count = validation_result['summary']['errors']
        warning_count = validation_result['summary']['warnings']

        if error_count > 0:
            st.error(f"❌ {error_count} critical error(s) found. Please fix before saving.")
        else:
            st.warning(f"⚠️ {warning_count} warning(s) found. You can proceed, but be aware of these issues.")

    # Show final summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Analysis Variables", len(mapping['predictors']) + 1)  # +1 for outcome
    with col3:
        outcome_series = df[outcome_col].dropna()
        if len(outcome_series) > 0:
            if len(outcome_series.unique()) == 2:
                outcome_rate = (outcome_series.value_counts().iloc[0] / len(outcome_series)) * 100
                st.metric("Outcome Rate", f"{outcome_rate:.1f}%")

    # Dataset name input
    st.markdown("### 📝 Dataset Name")
    default_name = Path(st.session_state.get('uploaded_filename', 'dataset')).stem
    dataset_name = st.text_input(
        "Enter a name for this dataset",
        value=default_name,
        help="This name will be used to identify your dataset in the analysis interface"
    )

    # Show issues if any
    if validation_result['issues']:
        with st.expander(f"⚠️ View Validation Issues ({len(validation_result['issues'])})"):
            for issue in validation_result['issues']:
                severity_emoji = "❌" if issue['severity'] == 'error' else "⚠️"
                st.markdown(f"{severity_emoji} **{issue['type']}**: {issue['message']}")

    # Navigation and save
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Mapping"):
            st.session_state['upload_step'] = 4
            st.rerun()

    with col2:
        # Can only save if no critical errors and dataset name provided
        can_save = validation_result['summary']['errors'] == 0 and dataset_name.strip()

        if st.button("💾 Save Dataset", disabled=not can_save, type="primary"):
            # Prepare metadata
            metadata = {
                'dataset_name': dataset_name,
                'variable_types': variable_info,
                'variable_mapping': mapping,
                'validation_result': validation_result
            }

            # Save upload
            success, message, upload_id = storage.save_upload(
                file_bytes=st.session_state['uploaded_bytes'],
                original_filename=st.session_state['uploaded_filename'],
                metadata=metadata
            )

            if success:
                st.success(f"✅ {message}")
                st.balloons()

                st.markdown(f"""
                **Dataset saved successfully!**

                - **Upload ID:** `{upload_id}`
                - **Name:** {dataset_name}
                - **Rows:** {len(df):,}
                - **Variables:** {len(mapping['predictors']) + 1}

                You can now use this dataset in the main analysis interface.
                """)

                # Clear session state
                if st.button("Upload Another Dataset"):
                    for key in list(st.session_state.keys()):
                        if key.startswith('upload'):
                            del st.session_state[key]
                    st.rerun()
            else:
                st.error(f"❌ {message}")


def render_zip_review_step():
    """Step 5: Review & Save for Multi-Table ZIP Upload"""
    st.markdown("### 🗂️ Multi-Table Dataset Processing")

    # Dataset name input
    st.markdown("### 📝 Dataset Name")
    default_name = Path(st.session_state.get('uploaded_filename', 'dataset')).stem
    dataset_name = st.text_input(
        "Enter a name for this dataset",
        value=default_name,
        help="This name will be used to identify your dataset in the analysis interface"
    )

    # Process ZIP file
    if st.button("🚀 Process & Save Multi-Table Dataset", type="primary", disabled=not dataset_name.strip()):
        # Create progress tracking UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_expander = st.expander("📋 Processing Log", expanded=True)
        log_container = log_expander.container()


        def progress_callback(step, total_steps, message, details):
            """Update progress UI with current step information."""
            progress = step / total_steps if total_steps > 0 else 0
            progress_bar.progress(progress)
            status_text.info(f"🔄 {message}")

            # Add to log
            with log_container:
                if details:
                    if 'table_name' in details:
                        table_info = f"**{details['table_name']}**"
                        if 'rows' in details:
                            table_info += f" - {details['rows']:,} rows, {details['cols']} cols"
                        if 'progress' in details:
                            table_info += f" ({details['progress']})"
                        st.text(f"✓ {table_info}")
                    elif 'tables_found' in details:
                        st.text(f"📦 Found {details['tables_found']} tables in ZIP")
                        if 'table_names' in details:
                            st.text(f"   Tables: {', '.join(details['table_names'][:5])}" +
                                   (f" ... and {len(details['table_names']) - 5} more"
                                    if len(details['table_names']) > 5 else ""))
                    elif 'relationships' in details:
                        st.text(f"🔗 Detected {len(details['relationships'])} relationships")
                        for rel in details['relationships'][:3]:  # Show first 3
                            st.text(f"   • {rel}")
                        if len(details['relationships']) > 3:
                            st.text(f"   ... and {len(details['relationships']) - 3} more")
                    else:
                        st.text(f"→ {message}")
                else:
                    st.text(f"→ {message}")

        # Prepare metadata
        metadata = {
            'dataset_name': dataset_name
        }

        # Save ZIP upload (this processes everything)
        try:
            success, message, upload_id = storage.save_zip_upload(
                file_bytes=st.session_state['uploaded_bytes'],
                original_filename=st.session_state['uploaded_filename'],
                metadata=metadata,
                progress_callback=progress_callback
            )
        except Exception as e:
            import traceback
            with log_container:
                st.error(f"❌ Error during processing: {str(e)}")
                st.code(traceback.format_exc())
            status_text.error(f"❌ Processing failed: {str(e)}")
            success = False
            message = str(e)
            upload_id = None

        if success:
            progress_bar.progress(1.0)
            status_text.success(f"✅ {message}")
            with log_container:
                st.success("✅ Processing complete!")
            st.balloons()

            # Load metadata to show details
            upload_metadata = storage.get_upload_metadata(upload_id)
            if upload_metadata:
                st.markdown("### 📊 Processing Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tables Joined", upload_metadata.get('tables', []) and len(upload_metadata.get('tables', [])) or 0)
                with col2:
                    st.metric("Unified Rows", upload_metadata.get('row_count', 0))
                with col3:
                    st.metric("Total Columns", upload_metadata.get('column_count', 0))

                # Show detected relationships
                relationships = upload_metadata.get('relationships', [])
                if relationships:
                    with st.expander(f"🔗 Detected Relationships ({len(relationships)})"):
                        for rel in relationships:
                            st.code(rel)

                # Show tables
                tables = upload_metadata.get('tables', [])
                table_counts = upload_metadata.get('table_counts', {})
                if tables:
                    with st.expander(f"📋 Tables ({len(tables)})"):
                        for table in tables:
                            count = table_counts.get(table, 0)
                            st.markdown(f"- **{table}**: {count:,} rows")

                # Show inferred schema
                inferred_schema = upload_metadata.get('inferred_schema', {})
                if inferred_schema:
                    with st.expander("🔬 Inferred Schema"):
                        if inferred_schema.get('column_mapping'):
                            st.markdown("**Column Mappings:**")
                            for col, role in inferred_schema['column_mapping'].items():
                                st.markdown(f"- `{col}` → {role}")

                        if inferred_schema.get('outcomes'):
                            st.markdown("**Outcomes:**")
                            for outcome, config in inferred_schema['outcomes'].items():
                                st.markdown(f"- `{outcome}` ({config.get('type', 'unknown')})")

            st.markdown(f"""
            **Dataset saved successfully!**

            - **Upload ID:** `{upload_id}`
            - **Name:** {dataset_name}
            - **Format:** Multi-table (ZIP)

            You can now use this dataset in the main analysis interface.
            """)

            # Clear session state
            if st.button("Upload Another Dataset"):
                for key in list(st.session_state.keys()):
                    if key.startswith('upload') or key == 'is_zip_upload':
                        del st.session_state[key]
                st.rerun()
        else:
            st.error(f"❌ {message}")

    # Back button
    if st.button("⬅️ Back to Upload"):
        st.session_state['upload_step'] = 1
        st.rerun()


def main():
    """Main upload page logic"""
    st.title("📤 Upload Clinical Data")
    st.markdown("Self-service data upload - no coding required!")

    # Initialize session state
    if 'upload_step' not in st.session_state:
        st.session_state['upload_step'] = 1

    # Progress indicator
    steps = ["📤 Upload", "👀 Preview", "🔬 Detect", "🗺️ Map", "✅ Review"]
    current_step = st.session_state['upload_step']

    cols = st.columns(5)
    for idx, (col, step) in enumerate(zip(cols, steps), 1):
        with col:
            if idx < current_step:
                st.markdown(f"✅ {step}")
            elif idx == current_step:
                st.markdown(f"**▶️ {step}**")
            else:
                st.markdown(f"⚪ {step}")

    st.divider()

    # Render appropriate step
    if current_step == 1:
        render_upload_step()

    elif current_step == 2:
        if 'uploaded_df' in st.session_state:
            render_preview_step(st.session_state['uploaded_df'])
        else:
            st.warning("No data uploaded. Please start over.")
            st.session_state['upload_step'] = 1
            st.rerun()

    elif current_step == 3:
        if 'uploaded_df' in st.session_state:
            render_variable_detection_step(st.session_state['uploaded_df'])
        else:
            st.warning("No data uploaded. Please start over.")
            st.session_state['upload_step'] = 1
            st.rerun()

    elif current_step == 4:
        if all(k in st.session_state for k in ['uploaded_df', 'variable_info', 'suggestions']):
            render_mapping_step(
                st.session_state['uploaded_df'],
                st.session_state['variable_info'],
                st.session_state['suggestions']
            )
        else:
            st.warning("Missing required data. Please start over.")
            st.session_state['upload_step'] = 1
            st.rerun()

    elif current_step == 5:
        # Check if ZIP upload (skip to review directly)
        is_zip = st.session_state.get('is_zip_upload', False)

        if is_zip:
            # For ZIP files, go directly to review (skip preview/detection/mapping)
            render_review_step()
        elif all(k in st.session_state for k in ['uploaded_df', 'variable_mapping', 'variable_info']):
            render_review_step(
                st.session_state['uploaded_df'],
                st.session_state['variable_mapping'],
                st.session_state['variable_info']
            )
        else:
            st.warning("Missing required data. Please start over.")
            st.session_state['upload_step'] = 1
            st.rerun()


if __name__ == "__main__":
    main()

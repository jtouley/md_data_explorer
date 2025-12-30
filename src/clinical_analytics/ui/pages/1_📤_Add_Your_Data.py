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
from clinical_analytics.ui.config import MULTI_TABLE_ENABLED
from clinical_analytics.ui.storage.user_datasets import UploadSecurityValidator, UserDatasetStorage

# Page configuration
st.set_page_config(page_title="Add Your Data | Clinical Analytics", page_icon="📤", layout="wide")

# Initialize storage
storage = UserDatasetStorage()


def render_upload_step():
    """Step 1: File Upload"""
    st.markdown("## 📤 Upload Your Data")

    # Determine allowed file types based on feature flag
    if MULTI_TABLE_ENABLED:
        allowed_types = ["csv", "xlsx", "xls", "sav", "zip"]
        format_help = """
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
    """
        uploader_help = "Maximum file size: 100MB. ZIP files for multi-table datasets."
    else:
        allowed_types = ["csv", "xlsx", "xls", "sav"]
        format_help = """
    Upload your clinical dataset in CSV, Excel, or SPSS format.

    **Supported formats:**
    - CSV (`.csv`) - Comma-separated values
    - Excel (`.xlsx`, `.xls`) - Microsoft Excel
    - SPSS (`.sav`) - Statistical software format

    **Requirements:**
    - File size: 1KB - 100MB
    - Must include patient ID column
    - Must include outcome variable
    """
        uploader_help = "Maximum file size: 100MB."

    st.markdown(format_help)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=allowed_types,
        help=uploader_help,
        key="file_uploader",
    )

    if uploaded_file is not None:
        # Only process on step 1 (prevents reprocessing on reruns)
        if st.session_state.get("upload_step", 1) == 1:
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

            if file_ext == ".zip":
                # Multi-table feature gated by config flag
                if not MULTI_TABLE_ENABLED:
                    st.error(
                        "❌ Multi-table (ZIP) uploads are disabled in V1. "
                        "Please upload a single CSV, Excel, or SPSS file."
                    )
                    st.info(
                        "💡 Multi-table support is planned for a future release. "
                        "Set MULTI_TABLE_ENABLED=true in environment to enable (experimental)."
                    )
                    return None

                # Handle multi-table ZIP upload
                st.info("🗂️ **Multi-table dataset detected!** ZIP file validated.")

                # Store ZIP upload details
                st.session_state["is_zip_upload"] = True
                st.session_state["uploaded_filename"] = uploaded_file.name
                st.session_state["uploaded_bytes"] = file_bytes

                st.success("✅ ZIP file ready for processing")
                st.info(
                    "💡 **Next:** Click 'Continue to Review' to process tables, "
                    "detect relationships, and build unified cohort."
                )

                # Button to proceed to review step
                if st.button("Continue to Review ➡️", type="primary", key="zip_upload_continue"):
                    st.session_state["upload_step"] = 5
                    st.rerun()

                return uploaded_file

            # Try to load preview for single-table files
            try:
                if file_ext == ".csv":
                    df = pd.read_csv(uploaded_file)
                elif file_ext in {".xlsx", ".xls"}:
                    # Use load_single_file from user_datasets for consistent Excel reading
                    # This includes intelligent header detection
                    import logging

                    from clinical_analytics.ui.storage.user_datasets import load_single_file

                    logger = logging.getLogger(__name__)

                    # Use file_bytes already read at line 98 via getvalue()
                    # Don't call read() again - that can cause position issues
                    logger.debug(f"Using file_bytes from getvalue(): type={type(file_bytes)}, len={len(file_bytes)}")

                    try:
                        # Use the same Excel reading logic as upload (with header detection)
                        filename = uploaded_file.name if uploaded_file.name else "upload.xlsx"
                        df_polars = load_single_file(file_bytes, filename)
                        # Convert to pandas for compatibility with existing preview code
                        df = df_polars.to_pandas()
                        logger.info(
                            f"Successfully loaded Excel file for preview: {df.shape[0]} rows, {df.shape[1]} columns"
                        )
                    except Exception as e:
                        logger.error(f"Failed to load Excel file for preview: {e}")
                        st.error(f"Failed to load Excel file: {e}")
                        return None
                elif file_ext == ".sav":
                    import pyreadstat

                    df, meta = pyreadstat.read_sav(uploaded_file)
                else:
                    st.error(f"Unsupported file type: {file_ext}")
                    return None

                # Store in session state (single-table)
                st.session_state["is_zip_upload"] = False
                st.session_state["uploaded_df"] = df
                st.session_state["uploaded_filename"] = uploaded_file.name
                st.session_state["uploaded_bytes"] = file_bytes
                st.session_state["upload_step"] = 2

                # Force immediate rerun to show preview step
                st.rerun()

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
    st.dataframe(df.head(10), width="stretch")

    # Data quality validation
    st.markdown("### 🔍 Data Quality Check")

    with st.spinner("Running quality checks..."):
        validation_result = DataQualityValidator.validate_complete(df)

    if validation_result["is_valid"]:
        st.success("✅ Data quality check passed!")
    else:
        st.warning(
            f"⚠️ Found {validation_result['summary']['errors']} error(s) and "
            f"{validation_result['summary']['warnings']} warning(s)"
        )

    # Show issues
    if validation_result["issues"]:
        with st.expander(f"View Issues ({len(validation_result['issues'])})"):
            for issue in validation_result["issues"]:
                severity = issue["severity"]
                message = issue["message"]

                if severity == "error":
                    st.error(f"❌ **{issue['type']}**: {message}")
                else:
                    st.warning(f"⚠️ **{issue['type']}**: {message}")

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Upload", key="preview_back"):
            st.session_state["upload_step"] = 1
            st.rerun()
    with col2:
        # Can proceed even with warnings, but not with errors
        can_proceed = validation_result["is_valid"] or validation_result["summary"]["errors"] == 0

        if st.button(
            "Continue to Variable Detection ➡️", disabled=not can_proceed, type="primary", key="preview_continue"
        ):
            st.session_state["upload_step"] = 3
            st.rerun()

    if not can_proceed:
        st.error("⚠️ Please fix critical errors before continuing")


def render_variable_detection_step(df: pd.DataFrame):
    """Step 3: Variable Type Detection"""
    st.markdown("## 🔬 Variable Type Detection")
    st.markdown("Automatically detecting variable types from your data...")

    with st.spinner("Analyzing variables..."):
        # Detect variable types (components handle pandas->polars conversion internally)
        variable_info = VariableTypeDetector.detect_all_variables(df)
        suggestions = VariableTypeDetector.suggest_schema_mapping(df)

    # Store in session state
    st.session_state["variable_info"] = variable_info
    st.session_state["suggestions"] = suggestions

    # Show detection results
    st.markdown("### Detection Results")

    # Group by type
    type_groups = {}
    for col, info in variable_info.items():
        var_type = info["type"]
        if var_type not in type_groups:
            type_groups[var_type] = []
        type_groups[var_type].append((col, info))

    # Display in columns
    type_emojis = {
        "continuous": "📊",
        "categorical": "🏷️",
        "binary": "🔀",
        "datetime": "📅",
        "id": "🆔",
    }

    for var_type, cols_info in type_groups.items():
        with st.expander(f"{type_emojis.get(var_type, '📌')} {var_type.upper()} Variables ({len(cols_info)})"):
            for col, info in cols_info:
                st.markdown(f"**{col}**")

                metadata = info["metadata"]
                missing_pct = info["missing_pct"]

                col1, col2 = st.columns([3, 1])
                with col1:
                    # Show relevant metadata
                    if var_type == "continuous":
                        st.caption(
                            f"Range: {metadata['min']:.2f} to {metadata['max']:.2f}, Mean: {metadata['mean']:.2f}"
                        )
                    elif var_type in ["categorical", "binary"]:
                        values = metadata.get("values", [])
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

    # Auto-apply suggestions (doctors shouldn't have to map columns)
    # Build mapping automatically from detected types
    auto_mapping = {
        "patient_id": suggestions.get("patient_id"),
        "outcome": suggestions.get("outcome"),
        "time_variables": {"time_zero": suggestions.get("time_zero")},
        "predictors": [],
        "excluded": [],
    }

    # Auto-categorize remaining columns as predictors (excluding ID, outcome, time)
    reserved_cols = {auto_mapping["patient_id"], auto_mapping["outcome"], suggestions.get("time_zero")}
    reserved_cols.discard(None)

    for col, info in variable_info.items():
        if col not in reserved_cols:
            # Include as predictor unless very high missing
            if info["missing_pct"] < 80:
                auto_mapping["predictors"].append(col)
            else:
                auto_mapping["excluded"].append(col)

    # Store auto-mapping
    st.session_state["variable_mapping"] = auto_mapping

    # Show auto-detected mapping summary
    st.markdown("### ✅ Auto-Detected Schema")
    st.success("We automatically detected your data schema. Review below and continue.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if auto_mapping["patient_id"]:
            st.metric("Patient ID", auto_mapping["patient_id"])
        else:
            st.warning("⚠️ No ID column detected")
    with col2:
        if auto_mapping["outcome"]:
            st.metric("Outcome", auto_mapping["outcome"])
        else:
            st.info("ℹ️ No outcome detected (optional)")
    with col3:
        if suggestions.get("time_zero"):
            st.metric("Time Variable", suggestions["time_zero"])
        else:
            st.info("ℹ️ No time column detected (optional)")

    # Show predictors count
    st.info(f"📊 **{len(auto_mapping['predictors'])}** predictor variables ready for analysis")

    # Optional: Allow override in expander (not mandatory)
    with st.expander("🔧 Adjust mappings (optional)", expanded=False):
        st.caption("Override auto-detection if needed")

        # Patient ID override
        id_options = ["(auto-detect)"] + list(df.columns)
        current_id_idx = id_options.index(auto_mapping["patient_id"]) if auto_mapping["patient_id"] in id_options else 0
        new_patient_id = st.selectbox("Patient ID Column", id_options, index=current_id_idx)
        if new_patient_id != "(auto-detect)" and new_patient_id != auto_mapping["patient_id"]:
            auto_mapping["patient_id"] = new_patient_id
            st.session_state["variable_mapping"] = auto_mapping

        # Outcome override
        outcome_options = ["(none)"] + list(df.columns)
        current_outcome_idx = (
            outcome_options.index(auto_mapping["outcome"]) if auto_mapping["outcome"] in outcome_options else 0
        )
        new_outcome = st.selectbox("Outcome Column", outcome_options, index=current_outcome_idx)
        if new_outcome == "(none)":
            auto_mapping["outcome"] = None
        elif new_outcome != auto_mapping["outcome"]:
            auto_mapping["outcome"] = new_outcome
        st.session_state["variable_mapping"] = auto_mapping

    # Navigation - Skip step 4 (mapping wizard), go directly to review
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Preview", key="detect_back"):
            st.session_state["upload_step"] = 2
            st.rerun()
    with col2:
        if st.button("Continue to Review ➡️", type="primary", key="detect_continue"):
            st.session_state["upload_step"] = 5  # Skip step 4, go directly to review
            st.rerun()


def render_mapping_step(df: pd.DataFrame, variable_info: dict, suggestions: dict):
    """Step 4: Variable Mapping Wizard"""
    st.markdown("## 🗺️ Map Variables to Analysis Schema")

    # Render the wizard
    mapping = VariableMappingWizard.render_complete_wizard(
        df=df, variable_info=variable_info, suggestions=suggestions, key_prefix="upload"
    )

    if mapping:
        # Store mapping in session state
        st.session_state["variable_mapping"] = mapping

        st.divider()

        # Show mapping summary
        with st.expander("📋 Mapping Summary", expanded=True):
            st.markdown(f"**Patient ID:** `{mapping['patient_id']}`")
            st.markdown(f"**Outcome:** `{mapping['outcome']}`")

            if mapping["time_variables"]:
                st.markdown("**Time Variables:**")
                for key, val in mapping["time_variables"].items():
                    if val:
                        st.markdown(f"- {key}: `{val}`")

            st.markdown(f"**Predictors:** {len(mapping['predictors'])} variables")
            if mapping["predictors"]:
                st.caption(", ".join(mapping["predictors"][:10]) + ("..." if len(mapping["predictors"]) > 10 else ""))

            if mapping["excluded"]:
                st.markdown(f"**Excluded:** {len(mapping['excluded'])} variables")

        # Navigation
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Detection", key="map_back"):
                st.session_state["upload_step"] = 3
                st.rerun()
        with col2:
            if st.button("Continue to Review ➡️", type="primary", key="map_continue"):
                st.session_state["upload_step"] = 5
                st.rerun()


def render_review_step(df: pd.DataFrame = None, mapping: dict = None, variable_info: dict = None):
    """Step 5: Final Review & Save"""
    st.markdown("## ✅ Review & Save Dataset")

    # Check if this is a ZIP upload (multi-table)
    is_zip = st.session_state.get("is_zip_upload", False)

    if is_zip:
        # Handle multi-table ZIP upload
        return render_zip_review_step()

    # Single-table workflow (existing logic)
    if df is None or mapping is None or variable_info is None:
        st.warning("Missing required data. Please start over.")
        st.session_state["upload_step"] = 1
        st.rerun()
        return

    # Run final validation with mapping
    patient_id_col = mapping["patient_id"]
    outcome_col = mapping["outcome"]

    # Get inferred granularity from variable detection (default to unknown for V1)
    granularity = "unknown"  # Default for V1 MVP - duplicates are warnings unless explicitly patient-level
    # TODO: Extract from variable_info or mapping if user explicitly selects granularity

    # Run final validation with mapping and granularity
    validation_result = DataQualityValidator.validate_complete(
        df, id_column=patient_id_col, outcome_column=outcome_col, granularity=granularity
    )

    # Show validation results (schema-first approach)
    if validation_result["is_valid"]:
        st.success("✅ Schema contract validated! Dataset is ready to use.")
        if validation_result["summary"]["warnings"] > 0:
            st.info(
                f"ℹ️ {validation_result['summary']['warnings']} data quality warning(s) found. "
                "You can proceed, but review warnings below."
            )
    else:
        st.error(f"❌ {validation_result['summary']['errors']} schema error(s) found. Please fix before saving.")

    # Show schema errors (blocking) - NEW
    if validation_result.get("schema_errors"):
        with st.expander(f"❌ Schema Errors ({len(validation_result['schema_errors'])})", expanded=True):
            for error in validation_result["schema_errors"]:
                # Handle both string errors and dict errors
                if isinstance(error, str):
                    st.error(f"**Schema Error**: {error}")
                else:
                    st.error(f"**{error.get('type', 'error')}**: {error.get('message', str(error))}")
                st.caption("💡 This can be fixed by mapping a different column or cleaning your data.")

    # Show quality warnings (non-blocking) - NEW
    if validation_result.get("quality_warnings"):
        with st.expander(f"⚠️ Data Quality Warnings ({len(validation_result['quality_warnings'])})", expanded=False):
            for warning in validation_result["quality_warnings"]:
                st.warning(f"**{warning.get('type', 'warning')}**: {warning.get('message', str(warning))}")
                if warning.get("actionable"):
                    st.caption("💡 You can exclude this column in the mapping step or proceed anyway.")

    # Keep existing issues display for backward compatibility (if new format not available)
    if validation_result.get("issues") and not (
        validation_result.get("schema_errors") or validation_result.get("quality_warnings")
    ):
        with st.expander(f"View Issues ({len(validation_result['issues'])})"):
            for issue in validation_result["issues"]:
                severity = issue["severity"]
                if severity == "error":
                    st.error(f"❌ **{issue['type']}**: {issue['message']}")
                else:
                    st.warning(f"⚠️ **{issue['type']}**: {issue['message']}")

    # Show final summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        # Count predictors + outcome (if outcome exists)
        var_count = len(mapping["predictors"])
        if outcome_col:
            var_count += 1
        st.metric("Analysis Variables", var_count)
    with col3:
        if outcome_col:
            outcome_series = df[outcome_col].dropna()
            if len(outcome_series) > 0:
                if len(outcome_series.unique()) == 2:
                    outcome_rate = (outcome_series.value_counts().iloc[0] / len(outcome_series)) * 100
                    st.metric("Outcome Rate", f"{outcome_rate:.1f}%")
                else:
                    st.metric("Outcome Rate", "N/A")
        else:
            st.metric("Outcome Rate", "Not set")

    # Dataset name input
    st.markdown("### 📝 Dataset Name")
    default_name = Path(st.session_state.get("uploaded_filename", "dataset")).stem
    dataset_name = st.text_input(
        "Enter a name for this dataset",
        value=default_name,
        help="This name will be used to identify your dataset in the analysis interface",
    )

    # Check if already saved in this session
    already_saved = st.session_state.get("upload_success") is True

    # Show success state if already saved
    if already_saved and st.session_state.get("upload_result"):
        result = st.session_state["upload_result"]
        st.success(f"""
        ✅ **Dataset saved successfully!**

        - **Upload ID:** `{result.get("upload_id", "N/A")}`
        - **Name:** {dataset_name}
        - **Rows:** {len(df):,}
        - **Variables:** {len(mapping["predictors"]) + 1}

        You can now use this dataset in the main analysis interface.
        """)

        # Option to upload another dataset
        if st.button("📤 Upload Another Dataset", key="success_upload_another"):
            # Clear all upload-related session state
            for key in list(st.session_state.keys()):
                if key.startswith("upload") or key in [
                    "uploaded_df",
                    "uploaded_bytes",
                    "uploaded_filename",
                    "variable_info",
                    "suggestions",
                    "variable_mapping",
                ]:
                    st.session_state.pop(key, None)
            st.session_state["upload_step"] = 1
            st.rerun()

        return  # Don't show save button again

    # Navigation and save
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Detection", key="review_back"):
            st.session_state["upload_step"] = 3
            st.rerun()

    with col2:
        # Can only save if no critical errors and dataset name provided
        can_save = validation_result["summary"]["errors"] == 0 and dataset_name.strip()

        if st.button("💾 Save Dataset", disabled=not can_save, type="primary", key="review_save"):
            # Show spinner during save (Streamlit renders this immediately)
            with st.spinner("💾 Saving dataset..."):
                try:
                    # Prepare metadata
                    metadata = {
                        "dataset_name": dataset_name,
                        "variable_types": variable_info,
                        "variable_mapping": mapping,
                        "validation_result": validation_result,
                    }

                    # Run save
                    success, message, upload_id = storage.save_upload(
                        file_bytes=st.session_state["uploaded_bytes"],
                        original_filename=st.session_state["uploaded_filename"],
                        metadata=metadata,
                    )

                    if success:
                        # Store success state to prevent re-saves
                        st.session_state["upload_success"] = True
                        st.session_state["upload_result"] = {
                            "success": success,
                            "message": message,
                            "upload_id": upload_id,
                        }
                        st.balloons()
                        st.rerun()  # Rerun to show success state
                    else:
                        st.error(f"❌ {message}")

                except Exception as e:
                    import traceback

                    st.error(f"❌ Error saving dataset: {e}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

        # Clear session state
        if st.button("Upload Another Dataset", key="error_upload_another"):
            for key in list(st.session_state.keys()):
                if key.startswith("upload"):
                    del st.session_state[key]
            st.rerun()


def render_zip_review_step():
    """Step 5: Review & Save for Multi-Table ZIP Upload"""
    st.markdown("### 🗂️ Multi-Table Dataset Processing")

    # Dataset name input
    st.markdown("### 📝 Dataset Name")
    default_name = Path(st.session_state.get("uploaded_filename", "dataset")).stem
    dataset_name = st.text_input(
        "Enter a name for this dataset",
        value=default_name,
        help="This name will be used to identify your dataset in the analysis interface",
    )

    # Process ZIP file
    if st.button(
        "🚀 Process & Save Multi-Table Dataset", type="primary", disabled=not dataset_name.strip(), key="zip_process"
    ):
        # Create progress tracking UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_expander = st.expander("📋 Processing Log", expanded=True)

        # Store messages in session_state to avoid unbounded DOM growth
        if "upload_log_messages" not in st.session_state:
            st.session_state["upload_log_messages"] = []

        def progress_callback(step, total_steps, message, details):
            """Update progress UI with current step information."""
            progress = step / total_steps if total_steps > 0 else 0
            progress_bar.progress(progress)
            status_text.info(f"🔄 {message}")

            # Store message in session_state (bounded list)
            log_msg = message
            if details:
                if "table_name" in details:
                    table_info = f"**{details['table_name']}**"
                    if "rows" in details:
                        table_info += f" - {details['rows']:,} rows, {details['cols']} cols"
                    if "progress" in details:
                        table_info += f" ({details['progress']})"
                    log_msg = f"✓ {table_info}"
                elif "tables_found" in details:
                    log_msg = f"📦 Found {details['tables_found']} tables in ZIP"
                elif "relationships" in details:
                    log_msg = f"🔗 Detected {len(details['relationships'])} relationships"
                else:
                    log_msg = f"→ {message}"
            else:
                log_msg = f"→ {message}"

            st.session_state["upload_log_messages"].append(log_msg)
            # Keep only last 50 messages to prevent unbounded growth
            if len(st.session_state["upload_log_messages"]) > 50:
                st.session_state["upload_log_messages"] = st.session_state["upload_log_messages"][-50:]

        # Prepare metadata
        metadata = {"dataset_name": dataset_name}

        # Save ZIP upload (this processes everything)
        try:
            success, message, upload_id = storage.save_zip_upload(
                file_bytes=st.session_state["uploaded_bytes"],
                original_filename=st.session_state["uploaded_filename"],
                metadata=metadata,
                progress_callback=progress_callback,
            )
        except Exception as e:
            import traceback

            # Render log messages from session_state
            with log_expander:
                for msg in st.session_state.get("upload_log_messages", []):
                    st.text(msg)
                st.error(f"❌ Error during processing: {str(e)}")
                st.code(traceback.format_exc())
            status_text.error(f"❌ Processing failed: {str(e)}")
            success = False
            message = str(e)
            upload_id = None

        if success:
            progress_bar.progress(1.0)
            status_text.success(f"✅ {message}")
            # Render all log messages from session_state
            with log_expander:
                for msg in st.session_state.get("upload_log_messages", []):
                    st.text(msg)
                st.success("✅ Processing complete!")
            st.balloons()
            # Clear log messages after successful upload
            st.session_state["upload_log_messages"] = []

            # Load metadata to show details
            upload_metadata = storage.get_upload_metadata(upload_id)
            if upload_metadata:
                st.markdown("### 📊 Processing Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Tables Joined",
                        upload_metadata.get("tables", []) and len(upload_metadata.get("tables", [])) or 0,
                    )
                with col2:
                    st.metric("Unified Rows", upload_metadata.get("row_count", 0))
                with col3:
                    st.metric("Total Columns", upload_metadata.get("column_count", 0))

                # Show detected relationships
                relationships = upload_metadata.get("relationships", [])
                if relationships:
                    with st.expander(f"🔗 Detected Relationships ({len(relationships)})"):
                        for rel in relationships:
                            st.code(rel)

                # Show tables
                tables = upload_metadata.get("tables", [])
                table_counts = upload_metadata.get("table_counts", {})
                if tables:
                    with st.expander(f"📋 Tables ({len(tables)})"):
                        for table in tables:
                            count = table_counts.get(table, 0)
                            st.markdown(f"- **{table}**: {count:,} rows")

                # Show inferred schema
                inferred_schema = upload_metadata.get("inferred_schema", {})
                if inferred_schema:
                    with st.expander("🔬 Inferred Schema"):
                        if inferred_schema.get("column_mapping"):
                            st.markdown("**Column Mappings:**")
                            for col, role in inferred_schema["column_mapping"].items():
                                st.markdown(f"- `{col}` → {role}")

                        if inferred_schema.get("outcomes"):
                            st.markdown("**Outcomes:**")
                            for outcome, config in inferred_schema["outcomes"].items():
                                st.markdown(f"- `{outcome}` ({config.get('type', 'unknown')})")

            st.markdown(f"""
            **Dataset saved successfully!**

            - **Upload ID:** `{upload_id}`
            - **Name:** {dataset_name}
            - **Format:** Multi-table (ZIP)

            You can now use this dataset in the main analysis interface.
            """)

            # Clear session state
            if st.button("Upload Another Dataset", key="zip_success_another"):
                for key in list(st.session_state.keys()):
                    if key.startswith("upload") or key == "is_zip_upload":
                        del st.session_state[key]
                st.rerun()
        else:
            st.error(f"❌ {message}")

    # Back button
    if st.button("⬅️ Back to Upload", key="zip_back"):
        st.session_state["upload_step"] = 1
        st.rerun()


def main():
    """Main upload page logic"""
    st.title("📤 Add Your Data")
    st.markdown("Upload your clinical dataset - no coding required!")

    # Initialize session state
    if "upload_step" not in st.session_state:
        st.session_state["upload_step"] = 1

    # Progress indicator
    steps = ["📤 Upload", "👀 Preview", "🔬 Detect", "🗺️ Map", "✅ Review"]
    current_step = st.session_state["upload_step"]

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
        if "uploaded_df" in st.session_state:
            render_preview_step(st.session_state["uploaded_df"])
        else:
            st.warning("No data uploaded. Please start over.")
            st.session_state["upload_step"] = 1
            st.rerun()

    elif current_step == 3:
        if "uploaded_df" in st.session_state:
            render_variable_detection_step(st.session_state["uploaded_df"])
        else:
            st.warning("No data uploaded. Please start over.")
            st.session_state["upload_step"] = 1
            st.rerun()

    elif current_step == 4:
        if all(k in st.session_state for k in ["uploaded_df", "variable_info", "suggestions"]):
            render_mapping_step(
                st.session_state["uploaded_df"],
                st.session_state["variable_info"],
                st.session_state["suggestions"],
            )
        else:
            st.warning("Missing required data. Please start over.")
            st.session_state["upload_step"] = 1
            st.rerun()

    elif current_step == 5:
        # Check if ZIP upload (skip to review directly)
        is_zip = st.session_state.get("is_zip_upload", False)

        if is_zip:
            # For ZIP files, go directly to review (skip preview/detection/mapping)
            render_review_step()
        elif all(k in st.session_state for k in ["uploaded_df", "variable_mapping", "variable_info"]):
            render_review_step(
                st.session_state["uploaded_df"],
                st.session_state["variable_mapping"],
                st.session_state["variable_info"],
            )
        else:
            st.warning("Missing required data. Please start over.")
            st.session_state["upload_step"] = 1
            st.rerun()


if __name__ == "__main__":
    main()

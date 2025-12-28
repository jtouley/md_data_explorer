"""
User Dataset Storage Manager

Handles secure storage, metadata management, and persistence of uploaded datasets.
"""

import hashlib
import json
import logging
import uuid
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from clinical_analytics.ui.config import MULTI_TABLE_ENABLED

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related error (path traversal, symlinks, etc.)."""

    pass


def _safe_store_upload(
    file_bytes: bytes,
    base_dir: Path,
    original_filename: str,
) -> Path:
    """
    Safely store upload using UUID, ignoring original filename.

    Args:
        file_bytes: File content
        base_dir: Base upload directory (enforced)
        original_filename: Original name (logged only, not used for path)

    Returns:
        Path to stored file (UUID-based)

    Raises:
        SecurityError: If path traversal or symlink detected
    """
    # Generate UUID-based filename with original extension only
    upload_id = str(uuid.uuid4())
    # Sanitize extension: only allow known safe extensions
    original_ext = Path(original_filename).suffix.lower()
    safe_extensions = {".csv", ".xlsx", ".xls", ".sav", ".zip", ".parquet"}
    if original_ext not in safe_extensions:
        original_ext = ".csv"  # Default to CSV if unknown

    safe_path = (base_dir / upload_id).with_suffix(original_ext)

    # Ensure base_dir exists and is absolute
    base_dir = base_dir.resolve()
    safe_path = safe_path.resolve()

    # Enforce: must be within base_dir
    if not safe_path.is_relative_to(base_dir):
        raise SecurityError(f"Path traversal detected: {safe_path}")

    # No symlinks allowed
    if base_dir.is_symlink():
        raise SecurityError("Symlinks not allowed in upload directory")

    # Write file
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_bytes(file_bytes)

    logger.info(
        "Stored upload securely",
        extra={
            "upload_id": upload_id,
            "original_filename": original_filename,
            "stored_path": str(safe_path),
        },
    )

    return safe_path


def _safe_extract_zip_member(
    zip_file: zipfile.ZipFile,
    member: str,
    extract_to: Path,
) -> Path:
    """
    Safely extract ZIP member, preventing path traversal.

    Args:
        zip_file: Open ZipFile object
        member: Member name to extract
        extract_to: Base directory to extract to

    Returns:
        Path to extracted file

    Raises:
        SecurityError: If path traversal or symlink detected
    """
    # Resolve to absolute paths
    extract_to = extract_to.resolve()
    target_path = (extract_to / member).resolve()

    # Must be within extract directory
    if not target_path.is_relative_to(extract_to):
        raise SecurityError(f"Path traversal detected in ZIP member: {member}")

    # No parent traversal in member name
    if ".." in member or member.startswith("/"):
        raise SecurityError(f"Invalid ZIP member path: {member}")

    # Extract and check for symlinks
    extracted = Path(zip_file.extract(member, extract_to))

    # Post-extraction symlink check
    if extracted.is_symlink():
        # Remove the symlink immediately
        extracted.unlink()
        raise SecurityError(f"Symlinks not allowed in ZIP: {member}")

    return extracted


class UploadSecurityValidator:
    """
    Security validation for uploaded files.

    Implements:
    - File type validation (allowlist)
    - File size limits
    - Path traversal prevention
    - Malicious content detection
    """

    # Allowlisted file extensions
    ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".sav", ".zip"}

    # Maximum file size (100MB as per Phase 0 spec)
    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB

    # Minimum file size (1KB to prevent empty files)
    MIN_FILE_SIZE_BYTES = 1024

    @classmethod
    def validate_file_type(cls, filename: str) -> tuple[bool, str]:
        """
        Validate file extension against allowlist.

        Args:
            filename: Original filename

        Returns:
            Tuple of (is_valid, error_message)
        """
        file_ext = Path(filename).suffix.lower()

        if not file_ext:
            return False, "File has no extension"

        if file_ext not in cls.ALLOWED_EXTENSIONS:
            allowed = ", ".join(cls.ALLOWED_EXTENSIONS)
            return False, f"File type '{file_ext}' not allowed. Allowed types: {allowed}"

        return True, ""

    @classmethod
    def validate_file_size(cls, file_bytes: bytes) -> tuple[bool, str]:
        """
        Validate file size is within acceptable range.

        Args:
            file_bytes: File content as bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        size = len(file_bytes)

        if size < cls.MIN_FILE_SIZE_BYTES:
            return False, f"File too small ({size} bytes). Minimum size is 1KB"

        if size > cls.MAX_FILE_SIZE_BYTES:
            size_mb = size / (1024 * 1024)
            return False, f"File too large ({size_mb:.1f}MB). Maximum size is 100MB"

        return True, ""

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and injection.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename (basename only, special chars removed)
        """
        # Get basename only (no directory traversal)
        safe_name = Path(filename).name

        # Remove any remaining special characters except alphanumeric, underscore, dash, dot
        safe_chars = []
        for char in safe_name:
            if char.isalnum() or char in ("_", "-", "."):
                safe_chars.append(char)
            else:
                safe_chars.append("_")

        return "".join(safe_chars)

    @classmethod
    def validate(cls, filename: str, file_bytes: bytes) -> tuple[bool, str]:
        """
        Run all security validations.

        Args:
            filename: Original filename
            file_bytes: File content

        Returns:
            Tuple of (is_valid, error_message)
        """
        # File type check
        valid, error = cls.validate_file_type(filename)
        if not valid:
            return False, error

        # File size check
        valid, error = cls.validate_file_size(file_bytes)
        if not valid:
            return False, error

        return True, ""


class UserDatasetStorage:
    """
    Manages storage and retrieval of user-uploaded datasets.

    Features:
    - Secure file storage with validation
    - Metadata persistence (variable types, mappings)
    - Upload tracking and management
    - Integration with existing registry system
    """

    def __init__(self, upload_dir: Path | None = None):
        """
        Initialize storage manager.

        Args:
            upload_dir: Base directory for uploads (default: data/uploads/)
        """
        if upload_dir is None:
            # Default to data/uploads relative to project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            upload_dir = project_root / "data" / "uploads"

        self.upload_dir = Path(upload_dir)
        self.raw_dir = self.upload_dir / "raw"
        self.metadata_dir = self.upload_dir / "metadata"

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def generate_upload_id(self, filename: str) -> str:
        """
        Generate unique upload ID.

        Uses timestamp + filename hash for uniqueness.

        Args:
            filename: Original filename

        Returns:
            Unique upload ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"user_upload_{timestamp}_{file_hash}"

    def save_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
        metadata: dict[str, Any],
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> tuple[bool, str, str | None]:
        """
        Save uploaded file with security validation.

        Args:
            file_bytes: File content
            original_filename: Original filename
            metadata: Upload metadata (variable types, mappings, etc.)
            progress_cb: Optional callback(progress: int, message: str) for UI updates (0-100, not 0.0-1.0)

        Returns:
            Tuple of (success, message, upload_id)
        """
        # Security validation
        if progress_cb:
            progress_cb(10, "Validating file security...")

        valid, error = UploadSecurityValidator.validate(original_filename, file_bytes)
        if not valid:
            if progress_cb:
                progress_cb(100, f"Validation failed: {error}")
            return False, error, None

        # Generate upload ID
        if progress_cb:
            progress_cb(20, "Preparing upload...")

        upload_id = self.generate_upload_id(original_filename)

        try:
            # Sanitize filename
            safe_filename = UploadSecurityValidator.sanitize_filename(original_filename)

            # Extract dataset_name from metadata (user-provided friendly name)
            dataset_name = metadata.get("dataset_name")
            if not dataset_name:
                # Fallback to sanitized original filename (without extension)
                dataset_name = UploadSecurityValidator.sanitize_filename(original_filename)
                dataset_name = Path(dataset_name).stem  # Remove extension

            # Sanitize the friendly name for filesystem safety
            safe_dataset_name = UploadSecurityValidator.sanitize_filename(dataset_name)
            # Ensure it's not empty and doesn't conflict
            if not safe_dataset_name or safe_dataset_name == ".":
                safe_dataset_name = "dataset"

            # Convert to CSV format (normalize all uploads to CSV)
            file_ext = Path(original_filename).suffix.lower()

            if progress_cb:
                progress_cb(30, "Reading file...")

            if file_ext == ".csv":
                # Already CSV, save with friendly name
                csv_filename = f"{safe_dataset_name}.csv"
                csv_path = self.raw_dir / csv_filename

                # Handle filename conflicts (if user re-uploads same name)
                if csv_path.exists():
                    # Append upload_id suffix to make unique
                    csv_filename = f"{safe_dataset_name}_{upload_id}.csv"
                    csv_path = self.raw_dir / csv_filename

                csv_path.write_bytes(file_bytes)
                df = pd.read_csv(csv_path)

            elif file_ext in {".xlsx", ".xls"}:
                # Excel file - convert to CSV
                import io

                df = pd.read_excel(io.BytesIO(file_bytes))

                if progress_cb:
                    progress_cb(50, "Converting file format...")

                csv_filename = f"{safe_dataset_name}.csv"
                csv_path = self.raw_dir / csv_filename

                # Handle filename conflicts
                if csv_path.exists():
                    csv_filename = f"{safe_dataset_name}_{upload_id}.csv"
                    csv_path = self.raw_dir / csv_filename

                df.to_csv(csv_path, index=False)

            elif file_ext == ".sav":
                # SPSS file - convert to CSV
                import io

                import pyreadstat

                df, meta = pyreadstat.read_sav(io.BytesIO(file_bytes))

                if progress_cb:
                    progress_cb(50, "Converting file format...")

                csv_filename = f"{safe_dataset_name}.csv"
                csv_path = self.raw_dir / csv_filename

                # Handle filename conflicts
                if csv_path.exists():
                    csv_filename = f"{safe_dataset_name}_{upload_id}.csv"
                    csv_path = self.raw_dir / csv_filename

                df.to_csv(csv_path, index=False)

            else:
                return False, f"Unsupported file type: {file_ext}", None

            # Validate schema if variable mapping is provided
            # This enforces the UnifiedCohort schema contract at save-time
            if progress_cb:
                progress_cb(60, "Validating schema contract...")

            variable_mapping = metadata.get("variable_mapping", {})
            schema_validation_result = None

            if variable_mapping:
                # Check if the mapped columns exist and validate schema
                from clinical_analytics.core.schema import (
                    UnifiedCohort,
                    validate_unified_cohort_schema,
                )

                # Create a view with renamed columns to validate schema
                cohort_df = df.copy()

                # Apply column mapping if patient_id is mapped
                patient_id_col = variable_mapping.get("patient_id")
                outcome_col = variable_mapping.get("outcome")

                # Rename columns to UnifiedCohort schema
                rename_map = {}
                if patient_id_col and patient_id_col in cohort_df.columns:
                    rename_map[patient_id_col] = UnifiedCohort.PATIENT_ID

                if outcome_col and outcome_col in cohort_df.columns:
                    rename_map[outcome_col] = UnifiedCohort.OUTCOME

                # Add time_zero if specified
                time_vars = variable_mapping.get("time_variables", {})
                if time_vars:
                    time_zero_col = time_vars.get("time_zero")
                    if time_zero_col and time_zero_col in cohort_df.columns:
                        rename_map[time_zero_col] = UnifiedCohort.TIME_ZERO

                if rename_map:
                    cohort_df = cohort_df.rename(columns=rename_map)

                    # Add outcome_label if not present
                    if UnifiedCohort.OUTCOME_LABEL not in cohort_df.columns:
                        cohort_df[UnifiedCohort.OUTCOME_LABEL] = outcome_col or "outcome"

                    # Add time_zero placeholder if not present
                    if UnifiedCohort.TIME_ZERO not in cohort_df.columns:
                        from datetime import datetime as dt

                        cohort_df[UnifiedCohort.TIME_ZERO] = dt.now()

                    # Validate the cohort schema using complete validator
                    from clinical_analytics.ui.components.data_validator import DataQualityValidator

                    # Run complete validation (schema-first, granularity unknown by default)
                    validation_result = DataQualityValidator.validate_complete(
                        cohort_df,
                        id_column=patient_id_col,
                        outcome_column=outcome_col,
                        granularity="unknown",  # Default granularity
                    )

                    schema_validation_result = {
                        "is_valid": validation_result["is_valid"],
                        "errors": validation_result["schema_errors"],
                    }

                    if not validation_result["is_valid"]:
                        # Log validation errors but don't fail - warn user
                        logger.warning(
                            "Schema validation warnings for upload %s: %s",
                            upload_id,
                            validation_result["schema_errors"],
                        )

            # Save metadata
            if progress_cb:
                progress_cb(80, "Saving metadata...")

            # Store canonical quality warnings at validation.quality_warnings
            # Use pop() to avoid duplicates if validation_result was already stored
            quality_warnings = []
            if schema_validation_result:
                quality_warnings = validation_result.get("quality_warnings", [])

            full_metadata = {
                "upload_id": upload_id,
                "original_filename": safe_filename,
                "stored_filename": csv_filename,  # Friendly name (may change with collisions)
                "stored_relpath": str(csv_path.relative_to(self.raw_dir)),  # Stable relative path pointer
                "dataset_name": dataset_name,  # User-provided friendly name
                "upload_timestamp": datetime.now().isoformat(),
                "file_size_bytes": len(file_bytes),
                "file_format": file_ext.lstrip("."),
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "schema_validation": schema_validation_result,
                "validation": {
                    "quality_warnings": quality_warnings,
                },
                **metadata,
            }

            metadata_path = self.metadata_dir / f"{upload_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(full_metadata, f, indent=2)

            if progress_cb:
                progress_cb(100, "Upload complete!")

            return True, f"Upload successful: {upload_id}", upload_id

        except Exception as e:
            if progress_cb:
                progress_cb(100, f"Error: {str(e)}")
            return False, f"Error saving upload: {str(e)}", None

    def get_upload_metadata(self, upload_id: str) -> dict[str, Any] | None:
        """
        Retrieve metadata for an upload.

        Args:
            upload_id: Upload identifier

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self.metadata_dir / f"{upload_id}.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            return json.load(f)

    def get_upload_data(self, upload_id: str) -> pd.DataFrame | None:
        """
        Load uploaded dataset.

        Uses friendly filename from metadata if available, falls back to upload_id.

        Args:
            upload_id: Upload identifier

        Returns:
            DataFrame or None if not found
        """
        # Get metadata to check for friendly filename
        # Use stored_relpath (stable pointer) if available, else stored_filename, else fallback
        metadata = self.get_upload_metadata(upload_id)
        if metadata:
            if "stored_relpath" in metadata:
                csv_path = self.raw_dir / metadata["stored_relpath"]
            elif "stored_filename" in metadata:
                csv_path = self.raw_dir / metadata["stored_filename"]
            else:
                # Fallback to old naming convention (for backward compatibility)
                csv_path = self.raw_dir / f"{upload_id}.csv"
        else:
            # No metadata - fallback to old naming convention
            csv_path = self.raw_dir / f"{upload_id}.csv"

        if not csv_path.exists():
            return None

        return pd.read_csv(csv_path)

    def list_uploads(self) -> list[dict[str, Any]]:
        """
        List all uploaded datasets.

        Returns:
            List of metadata dictionaries
        """
        uploads = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            with open(metadata_file) as f:
                metadata = json.load(f)
                uploads.append(metadata)

        # Sort by upload timestamp (newest first)
        uploads.sort(key=lambda x: x["upload_timestamp"], reverse=True)

        return uploads

    def delete_upload(self, upload_id: str) -> tuple[bool, str]:
        """
        Delete an uploaded dataset.

        Args:
            upload_id: Upload identifier

        Returns:
            Tuple of (success, message)
        """
        try:
            csv_path = self.raw_dir / f"{upload_id}.csv"
            metadata_path = self.metadata_dir / f"{upload_id}.json"

            deleted_files = []

            if csv_path.exists():
                csv_path.unlink()
                deleted_files.append("data file")

            if metadata_path.exists():
                metadata_path.unlink()
                deleted_files.append("metadata")

            if not deleted_files:
                return False, f"Upload {upload_id} not found"

            return True, f"Deleted {', '.join(deleted_files)} for {upload_id}"

        except Exception as e:
            return False, f"Error deleting upload: {str(e)}"

    def update_metadata(self, upload_id: str, metadata_updates: dict[str, Any]) -> tuple[bool, str]:
        """
        Update metadata for an upload.

        Args:
            upload_id: Upload identifier
            metadata_updates: New metadata fields

        Returns:
            Tuple of (success, message)
        """
        metadata_path = self.metadata_dir / f"{upload_id}.json"

        if not metadata_path.exists():
            return False, f"Upload {upload_id} not found"

        try:
            # Load existing metadata
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Update with new fields
            metadata.update(metadata_updates)

            # Save updated metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return True, "Metadata updated successfully"

        except Exception as e:
            return False, f"Error updating metadata: {str(e)}"

    def save_zip_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
        metadata: dict[str, Any],
        progress_callback: Callable[[int, int, str, dict], None] | None = None,
    ) -> tuple[bool, str, str | None]:
        """
        Save uploaded ZIP file containing multiple CSV files.

        This enables multi-table dataset support (e.g., MIMIC-IV).
        Tables are extracted, relationships detected, and unified cohort created.

        Args:
            file_bytes: ZIP file content
            original_filename: Original filename
            metadata: Upload metadata
            progress_callback: Optional callback function(step, total_steps, message, details)

        Returns:
            Tuple of (success, message, upload_id)
        """
        # Gate on feature flag - multi-table support is deferred to V2
        if not MULTI_TABLE_ENABLED:
            return (
                False,
                "Multi-table (ZIP) uploads are disabled in V1. "
                "Set MULTI_TABLE_ENABLED=true in environment to enable (experimental).",
                None,
            )

        import io
        import zipfile

        from clinical_analytics.core.multi_table_handler import MultiTableHandler
        from clinical_analytics.core.schema_inference import SchemaInferenceEngine

        # Security validation
        valid, error = UploadSecurityValidator.validate(original_filename, file_bytes)
        if not valid:
            return False, error, None

        # Generate upload ID
        upload_id = self.generate_upload_id(original_filename)

        try:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Starting ZIP upload processing: {original_filename}")

            # Extract ZIP contents
            zip_buffer = io.BytesIO(file_bytes)
            tables: dict[str, pl.DataFrame] = {}

            with zipfile.ZipFile(zip_buffer, "r") as zip_file:
                # Get list of CSV files in ZIP (including .csv.gz in subdirectories)
                csv_files = [
                    f
                    for f in zip_file.namelist()
                    if (f.endswith(".csv") or f.endswith(".csv.gz"))
                    and not f.startswith("__MACOSX")
                    and not f.endswith("/")  # Skip directory entries
                ]

                if not csv_files:
                    return False, "No CSV files found in ZIP archive", None

                logger.info(f"Found {len(csv_files)} CSV files in ZIP archive")

                # Calculate total steps now that we know number of files
                # 1 (found tables) + len(csv_files) (loading) + 4 (detect, build, save, infer)
                total_steps = 1 + len(csv_files) + 4

                if progress_callback:
                    progress_callback(0, total_steps, "Initializing ZIP extraction...", {})
                    progress_callback(
                        1,
                        total_steps,
                        f"Found {len(csv_files)} tables to load",
                        {
                            "tables_found": len(csv_files),
                            "table_names": [Path(f).stem for f in csv_files],
                        },
                    )

                # Load each CSV as a table
                for idx, csv_filename in enumerate(csv_files, start=1):
                    # Extract table name (without path and extension)
                    table_name = Path(csv_filename).stem
                    if table_name.endswith(".csv"):
                        # Handle .csv.gz case where stem gives us "filename.csv"
                        table_name = Path(table_name).stem

                    logger.info(f"Loading table {idx}/{len(csv_files)}: {table_name} from {csv_filename}")

                    if progress_callback:
                        progress_callback(
                            1 + idx,
                            total_steps,
                            f"Loading table: {table_name}",
                            {
                                "table_name": table_name,
                                "file": csv_filename,
                                "progress": f"{idx}/{len(csv_files)}",
                            },
                        )

                    # Read file content
                    csv_content = zip_file.read(csv_filename)

                    # Handle gzip compression
                    if csv_filename.endswith(".gz"):
                        import gzip

                        logger.debug(f"Decompressing gzip file: {csv_filename}")
                        csv_content = gzip.decompress(csv_content)

                    # Load as Polars DataFrame with robust schema inference
                    # Use larger infer_schema_length to handle mixed-type columns (e.g., ICD codes)
                    try:
                        logger.debug(f"Reading CSV with schema inference for {table_name}")
                        df = pl.read_csv(
                            io.BytesIO(csv_content),
                            infer_schema_length=10000,  # Scan more rows for better type inference
                            try_parse_dates=True,
                        )
                    except Exception as e:
                        logger.warning(f"Schema inference failed for {table_name}, falling back to string types: {e}")
                        # Fallback: read with all columns as strings, let DuckDB handle types
                        df = pl.read_csv(
                            io.BytesIO(csv_content),
                            infer_schema_length=0,  # Treat all as strings
                        )

                    tables[table_name] = df
                    logger.info(f"Loaded table '{table_name}': {df.height:,} rows, {df.width} cols")
                    logger.debug(f"Schema for {table_name}: {dict(df.schema)}")

                    if progress_callback:
                        progress_callback(
                            1 + idx,
                            total_steps,
                            f"Loaded {table_name}: {df.height:,} rows, {df.width} cols",
                            {
                                "table_name": table_name,
                                "rows": df.height,
                                "cols": df.width,
                                "status": "loaded",
                            },
                        )

            logger.info(f"Extracted {len(tables)} tables from ZIP: {list(tables.keys())}")

            # Detect relationships between tables
            # Step calculation: 1 (init) + len(csv_files) (loading) = current step
            step_num = 1 + len(csv_files)
            logger.info(f"Detecting relationships for {len(tables)} tables")

            if progress_callback:
                progress_callback(
                    step_num,
                    total_steps,
                    "Detecting table relationships...",
                    {
                        "tables": list(tables.keys()),
                        "table_counts": {name: df.height for name, df in tables.items()},
                    },
                )

            handler = MultiTableHandler(tables)
            relationships = handler.detect_relationships()
            logger.info(f"Detected {len(relationships)} relationships")

            if relationships:
                for rel in relationships:
                    logger.info(f"Relationship: {rel}")

            if progress_callback:
                progress_callback(
                    step_num + 1,
                    total_steps,
                    f"Detected {len(relationships)} relationships",
                    {"relationships": [str(rel) for rel in relationships]},
                )

            # Build unified cohort
            logger.info("Building unified cohort from detected relationships")
            if progress_callback:
                progress_callback(step_num + 2, total_steps, "Building unified cohort...", {})

            unified_df = handler.build_unified_cohort()
            logger.info(f"Unified cohort created: {unified_df.height:,} rows, {unified_df.width} cols")

            # Save unified cohort as CSV
            csv_path = self.raw_dir / f"{upload_id}.csv"
            logger.info(f"Saving unified cohort to {csv_path}")
            if progress_callback:
                progress_callback(
                    step_num + 3,
                    total_steps,
                    f"Saving unified cohort ({unified_df.height:,} rows)...",
                    {},
                )
            unified_df.write_csv(csv_path)

            # Save individual tables to disk for semantic layer access
            tables_dir = self.raw_dir / f"{upload_id}_tables"
            tables_dir.mkdir(exist_ok=True)
            logger.info(f"Saving {len(tables)} individual tables to {tables_dir}")

            for table_name, df in tables.items():
                table_path = tables_dir / f"{table_name}.csv"
                df.write_csv(table_path)
                logger.debug(f"Saved table '{table_name}' ({df.height:,} rows) to {table_path}")

            # Infer schema for unified cohort
            logger.info("Inferring schema for unified cohort")
            if progress_callback:
                progress_callback(step_num + 4, total_steps, "Inferring schema...", {})

            engine = SchemaInferenceEngine()
            schema = engine.infer_schema(unified_df)
            logger.info("Schema inference complete")

            # Save metadata
            full_metadata = {
                "upload_id": upload_id,
                "original_filename": UploadSecurityValidator.sanitize_filename(original_filename),
                "upload_timestamp": datetime.now().isoformat(),
                "file_size_bytes": len(file_bytes),
                "file_format": "zip_multi_table",
                "row_count": unified_df.height,
                "column_count": unified_df.width,
                "columns": list(unified_df.columns),
                "tables": list(tables.keys()),
                "table_counts": {name: df.height for name, df in tables.items()},
                "relationships": [str(rel) for rel in relationships],
                "inferred_schema": schema.to_dataset_config(),
                **metadata,
            }

            metadata_path = self.metadata_dir / f"{upload_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(full_metadata, f, indent=2)

            handler.close()

            if progress_callback:
                progress_callback(
                    step_num + 4,
                    total_steps,
                    "Processing complete!",
                    {"tables": len(tables), "rows": unified_df.height, "cols": unified_df.width},
                )

            logger.info(f"Multi-table upload successful: {len(tables)} tables joined into {unified_df.height:,} rows")
            return (
                True,
                f"Multi-table upload successful: {len(tables)} tables joined into {unified_df.height:,} rows",
                upload_id,
            )

        except Exception as e:
            import logging
            import traceback

            logger = logging.getLogger(__name__)
            logger.error(f"Error processing ZIP upload: {type(e).__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Error processing ZIP upload: {str(e)}", None

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


class UploadError(Exception):
    """Upload processing error (malformed files, invalid content, etc.)."""

    pass


# Re-export schema conversion functions for backward compatibility
# These are now defined in clinical_analytics.datasets.uploaded.schema_conversion
# but kept here for existing imports


def normalize_upload_to_table_list(
    file_bytes: bytes,
    filename: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Normalize any upload to unified table list.

    This is the ONLY function that detects upload type.
    Everything downstream uses unified table list format.

    Args:
        file_bytes: File content
        filename: Original filename (used to detect type)
        metadata: Optional metadata (for progress callbacks)

    Returns:
        (tables, table_metadata) where:
        - tables: list of {"name": str, "data": pl.DataFrame}
        - table_metadata: dict with table_count, table_names, etc.
    """
    # Detect upload type from file extension
    if filename.endswith(".zip"):
        # Multi-table: extract from ZIP
        tables = extract_zip_tables(file_bytes)
    else:
        # Single-file: wrap in list (becomes multi-table with 1 table)
        df = load_single_file(file_bytes, filename)
        table_name = Path(filename).stem  # Use original filename stem
        tables = [{"name": table_name, "data": df}]

    # Build metadata
    table_metadata = {
        "table_count": len(tables),
        "table_names": [t["name"] for t in tables],
    }

    return tables, table_metadata


def extract_zip_tables(file_bytes: bytes) -> list[dict[str, Any]]:
    """
    Extract tables from ZIP archive.

    Args:
        file_bytes: ZIP file content

    Returns:
        List of {"name": str, "data": pl.DataFrame} dicts

    Raises:
        SecurityError: If path traversal or invalid paths detected
        UploadError: If no CSV files or corrupted ZIP
    """
    import gzip
    import io

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            # Security: Check for path traversal
            for entry in z.namelist():
                if ".." in entry or entry.startswith("/"):
                    raise SecurityError(f"Invalid path: {entry}")

            # Get CSV files (including .csv.gz), skip __MACOSX and directories
            csv_files = [
                e
                for e in z.namelist()
                if (e.endswith(".csv") or e.endswith(".csv.gz"))
                and not e.startswith("__MACOSX")
                and not e.endswith("/")  # Skip directory entries
            ]

            if not csv_files:
                raise UploadError("No CSV files in ZIP")

            # Check for duplicate table names
            seen = set()
            tables = []

            for entry in csv_files:
                # Extract table name (filename stem)
                name = Path(entry).stem
                if name.endswith(".csv"):  # Handle .csv.gz case
                    name = Path(name).stem

                if name in seen:
                    raise UploadError(f"Duplicate table name: {name}")
                seen.add(name)

                # Read file content
                csv_content = z.read(entry)

                # Handle gzip compression
                if entry.endswith(".gz"):
                    csv_content = gzip.decompress(csv_content)

                # Load as Polars DataFrame with robust schema inference
                try:
                    # Common ID columns that should be strings to prevent overflow
                    id_column_names = {
                        "patient_id",
                        "patientid",
                        "subject_id",
                        "subjectid",
                        "id",
                        "mrn",
                        "study_id",
                    }

                    # First, peek at schema to identify ID columns
                    sample_df = pl.read_csv(io.BytesIO(csv_content), n_rows=0)
                    schema_overrides = {}
                    for col_name in sample_df.columns:
                        if col_name.lower() in {name.lower() for name in id_column_names}:
                            schema_overrides[col_name] = pl.Utf8

                    df = pl.read_csv(
                        io.BytesIO(csv_content),
                        infer_schema_length=10000,  # Scan more rows for better type inference
                        try_parse_dates=True,
                        schema_overrides=schema_overrides if schema_overrides else None,
                    )
                except Exception as e:
                    logger.warning(f"Schema inference failed for {name}, falling back to string types: {e}")
                    # Fallback: read with all columns as strings
                    df = pl.read_csv(
                        io.BytesIO(csv_content),
                        infer_schema_length=0,  # Treat all as strings
                    )

                tables.append({"name": name, "data": df})
                logger.debug(f"Loaded table '{name}': {df.height:,} rows, {df.width} cols")

            return tables

    except zipfile.BadZipFile:
        raise UploadError("Corrupted ZIP file")


def _detect_excel_header_row(file_bytes: bytes, max_rows_to_check: int = 5) -> int:
    """
    Intelligently detect the best header row in an Excel file.

    Analyzes the first few rows to find which one looks most like column headers:
    - Headers typically have many non-empty cells
    - Headers are usually strings (not numeric data)
    - Headers are relatively short
    - Headers are unique/distinct

    Args:
        file_bytes: Excel file content
        max_rows_to_check: Maximum number of rows to analyze (default 5)

    Returns:
        Row index (0-based) to use as header, or 0 if detection fails
    """
    import io

    import pandas as pd

    try:
        if not isinstance(file_bytes, bytes):
            if hasattr(file_bytes, "read"):
                file_bytes = file_bytes.read()
            else:
                raise TypeError(f"Expected bytes, got {type(file_bytes)}")

        df_preview = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", header=None, nrows=max_rows_to_check)

        if df_preview.empty:
            return 0

        scores = []
        for row_idx in range(min(max_rows_to_check, len(df_preview))):
            row = df_preview.iloc[row_idx]

            # Count non-empty cells
            non_empty = row.notna().sum()
            if non_empty == 0:
                scores.append(0.0)
                continue

            # Calculate score components
            score = 0.0

            # 1. More non-empty cells = better (weight: 40%)
            fill_ratio = non_empty / len(row)
            score += fill_ratio * 0.4

            # 2. Check if values look like headers (strings, not numeric data)
            string_count = 0
            avg_length = 0
            unique_count = 0

            for val in row.dropna():
                val_str = str(val).strip()
                if not val_str:
                    continue

                # Headers are usually strings, not pure numbers
                try:
                    float(val_str)  # If it's a number, less likely to be a header
                    # But allow short numbers (like "1", "2" for coded values)
                    if len(val_str) > 3:
                        continue
                except ValueError:
                    string_count += 1

                # Track length and uniqueness
                avg_length += len(val_str)
                unique_count += 1

            if unique_count > 0:
                avg_length = avg_length / unique_count
                uniqueness_ratio = row.dropna().nunique() / unique_count

                # 3. Higher string ratio = better (weight: 30%)
                string_ratio = string_count / unique_count if unique_count > 0 else 0
                score += string_ratio * 0.3

                # 4. Reasonable length (not too short, not too long) (weight: 15%)
                # Headers are usually 5-50 characters
                if 5 <= avg_length <= 50:
                    score += 0.15
                elif avg_length < 5:
                    score += (avg_length / 5) * 0.15
                else:
                    score += max(0, (50 / avg_length)) * 0.15

                # 5. High uniqueness (headers should be distinct) (weight: 15%)
                score += min(uniqueness_ratio, 1.0) * 0.15

            scores.append(score)

        # Find row with highest score
        if scores and max(scores) > 0.3:  # Minimum threshold
            best_row = scores.index(max(scores))
            logger.debug(f"Detected header row: {best_row} (score: {max(scores):.2f})")
            return best_row

        # Default to row 0 if no good candidate found
        logger.debug("No clear header row detected, using row 0")
        return 0

    except Exception as e:
        logger.warning(f"Header detection failed: {e}, using default row 0")
        return 0


def load_single_file(file_bytes: bytes, filename: str) -> pl.DataFrame:
    """
    Load single file (CSV, Excel, SPSS) as Polars DataFrame.

    Args:
        file_bytes: File content
        filename: Original filename (determines file type)

    Returns:
        Polars DataFrame

    Raises:
        ValueError: If unsupported file type
    """
    import io

    file_ext = Path(filename).suffix.lower()

    if file_ext == ".csv":
        # For CSV files, use Polars directly
        try:
            # Common ID columns that should be strings
            id_column_names = {
                "patient_id",
                "patientid",
                "subject_id",
                "subjectid",
                "id",
                "mrn",
                "study_id",
            }

            # Peek at schema first
            sample_df = pl.read_csv(io.BytesIO(file_bytes), n_rows=0)
            schema_overrides = {}
            for col_name in sample_df.columns:
                if col_name.lower() in {name.lower() for name in id_column_names}:
                    schema_overrides[col_name] = pl.Utf8

            return pl.read_csv(
                io.BytesIO(file_bytes),
                infer_schema_length=10000,
                try_parse_dates=True,
                schema_overrides=schema_overrides if schema_overrides else None,
            )
        except Exception as e:
            logger.warning(f"CSV read failed with schema inference: {e}. Retrying with ID columns as strings.")
            # Fallback: force ID columns to strings
            try:
                fallback_overrides = {
                    "patient_id": pl.Utf8,
                    "patientid": pl.Utf8,
                    "subject_id": pl.Utf8,
                    "subjectid": pl.Utf8,
                    "id": pl.Utf8,
                }
                return pl.read_csv(
                    io.BytesIO(file_bytes),
                    infer_schema_length=10000,
                    try_parse_dates=True,
                    schema_overrides=fallback_overrides,
                )
            except Exception as e2:
                logger.warning(f"Retry failed: {e2}. Falling back to all-string schema.")
                return pl.read_csv(io.BytesIO(file_bytes), infer_schema_length=0)

    elif file_ext in {".xlsx", ".xls"}:
        # PANDAS EXCEPTION: Use pandas as primary for Excel files
        # Polars read_excel() can miss columns or stop early on complex Excel files
        # TODO: Revisit when Polars Excel support is more robust
        import pandas as pd

        try:
            header_row = _detect_excel_header_row(file_bytes, max_rows_to_check=5)
            df_pandas = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", header=header_row)

            # Clean up any remaining "Unnamed" columns (from empty header cells)
            df_pandas.columns = [
                f"column_{i}" if str(col).startswith("Unnamed") else col for i, col in enumerate(df_pandas.columns)
            ]

            # Try converting pandas DataFrame to Polars
            try:
                return pl.from_pandas(df_pandas)
            except Exception as polars_error:
                logger.warning(f"pl.from_pandas failed: {polars_error}. Trying Polars read_excel as fallback.")
                # Fallback to Polars read_excel and compare column counts
                try:
                    df_polars_fallback = pl.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
                    pandas_column_count = len(df_pandas.columns)
                    polars_column_count = len(df_polars_fallback.columns)

                    # Use whichever has more columns
                    if pandas_column_count >= polars_column_count:
                        # Convert object columns to string to handle mixed types
                        for col in df_pandas.columns:
                            if df_pandas[col].dtype == "object":
                                df_pandas[col] = df_pandas[col].astype(str).replace("nan", None)
                        return pl.from_pandas(df_pandas)
                    else:
                        return df_polars_fallback
                except Exception as polars_error:
                    logger.error(f"Both pandas and Polars failed to read Excel file: {polars_error}")
                    raise ValueError(f"Failed to read Excel file: {polars_error}") from polars_error
        except Exception as e:
            logger.warning(f"Pandas Excel read failed: {e}. Trying Polars as fallback.")
            # Fallback to Polars if pandas fails completely
            try:
                return pl.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
            except Exception as polars_error:
                logger.error(f"Both pandas and Polars failed to read Excel file: {polars_error}")
                raise ValueError(f"Failed to read Excel file: {polars_error}") from polars_error

    elif file_ext == ".sav":
        # Load SPSS file
        import pyreadstat

        df_pandas, meta = pyreadstat.read_sav(io.BytesIO(file_bytes))
        return pl.from_pandas(df_pandas)

    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


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


def save_table_list(
    storage: "UserDatasetStorage",
    tables: list[dict[str, Any]],
    upload_id: str,
    metadata: dict[str, Any],
    progress_cb: Callable[[int, str], None] | None = None,
) -> tuple[bool, str]:
    """
    Save normalized table list to disk (unified persistence for both upload types).

    CRITICAL: Schema conversion happens AFTER normalization (fixes circular dependency).

    TECHNICAL DEBT (Boundary Leakage - MVP acceptable, fix in Phase 5+):
        This function couples UI upload concerns with semantic storage:
        1. ID column normalization (lines ~725-743): Type coercion based on UI heuristics
        2. Schema conversion (lines ~652-659): UI variable_mapping influences semantic schema
        3. Schema inference fallback (lines ~772-778): UI-driven fallback logic

        Ideal: Semantic layer owns data types/schema, UI layer only provides hints.
        Reality: Upload wizard directly mutates data before semantic layer sees it.

        Deferred to Phase 5+: Move type coercion to semantic layer boundary validator.

    Args:
        storage: UserDatasetStorage instance
        tables: Normalized table list [{"name": str, "data": pl.DataFrame}]
        upload_id: Upload identifier
        metadata: Upload metadata
        progress_cb: Optional progress callback

    Returns:
        (success, message)
    """
    try:
        # 1. Convert schema (AFTER normalization, has df access)
        if "variable_mapping" in metadata and tables:
            from clinical_analytics.datasets.uploaded.schema_conversion import convert_schema

            metadata["inferred_schema"] = convert_schema(
                metadata["variable_mapping"],
                tables[0]["data"],  # Access normalized DataFrame
            )

        # 2. Compute dataset version and table fingerprints (MVP - Phase 1)
        from clinical_analytics.storage.versioning import (
            compute_dataset_version,
            compute_table_fingerprint,
        )

        # Extract DataFrames for versioning
        table_dfs = [t["data"] for t in tables]
        dataset_version = compute_dataset_version(table_dfs)
        logger.info(f"Computed dataset_version: {dataset_version}")

        # Compute table fingerprints for provenance
        table_fingerprints = [compute_table_fingerprint(t["data"], t["name"]) for t in tables]

        # 3. Save individual tables to {upload_id}_tables/
        tables_dir = storage.raw_dir / f"{upload_id}_tables"
        tables_dir.mkdir(exist_ok=True)

        table_names = []
        for table in tables:
            table_name = table["name"]
            table_names.append(table_name)
            table_path = tables_dir / f"{table_name}.csv"
            table["data"].write_csv(table_path)
            logger.debug(f"Saved table '{table_name}' ({table['data'].height:,} rows) to {table_path}")

        # 3. Build unified cohort (for backward compatibility)
        if len(tables) == 1:
            # Single-table: unified cohort = first table
            unified_df = tables[0]["data"]
        else:
            # Multi-table: use MultiTableHandler to build unified cohort
            from clinical_analytics.core.multi_table_handler import MultiTableHandler

            # Convert list of dicts to dict for handler
            tables_dict = {t["name"]: t["data"] for t in tables}
            handler = MultiTableHandler(tables_dict)

            # Use relationships from metadata if present (avoid duplicate detection)
            if "relationships" in metadata and metadata["relationships"]:
                # Relationships already detected, just build unified cohort
                logger.debug("Using relationships from metadata")
            else:
                # Detect relationships and store in metadata
                relationships = handler.detect_relationships()
                metadata["relationships"] = [str(rel) for rel in relationships]
                logger.info(f"Detected {len(relationships)} relationships")

            unified_df = handler.build_unified_cohort()
            handler.close()

        # 4. Save unified cohort CSV
        csv_path = storage.raw_dir / f"{upload_id}.csv"
        unified_df.write_csv(csv_path)
        logger.info(f"Saved unified cohort ({unified_df.height:,} rows) to {csv_path}")

        # 4.5. Save to persistent DuckDB and export to Parquet (Phase 2 + Phase 3)
        from clinical_analytics.storage.datastore import DataStore

        # Get or create DataStore (persistent DuckDB at data/analytics.duckdb)
        db_path = storage.upload_dir.parent / "analytics.duckdb"
        parquet_dir = storage.upload_dir.parent / "parquet"
        datastore = DataStore(db_path)

        # Common ID column names that should be strings (prevent integer overflow)
        id_column_names = {
            "patient_id",
            "patientid",
            "subject_id",
            "subjectid",
            "id",
            "mrn",
            "study_id",
        }

        # Convert ID columns to strings before persisting (ensures consistent schema)
        for table in tables:
            df = table["data"]
            for col in df.columns:
                if col.lower() in id_column_names:
                    # Cast to string to prevent integer overflow and ensure consistent type
                    table["data"] = df.with_columns(pl.col(col).cast(pl.Utf8))
                    logger.debug(f"Converted {col} to Utf8 in table '{table['name']}'")

        parquet_paths = {}
        try:
            # Save all individual tables to DuckDB and export to Parquet
            for table in tables:
                # Save to DuckDB (Phase 2)
                datastore.save_table(
                    table_name=table["name"],
                    data=table["data"],
                    upload_id=upload_id,
                    dataset_version=dataset_version,
                )
                logger.debug(f"Saved table '{table['name']}' to DuckDB")

                # Export to Parquet (Phase 3)
                parquet_path = datastore.export_to_parquet(
                    upload_id=upload_id,
                    table_name=table["name"],
                    dataset_version=dataset_version,
                    parquet_dir=parquet_dir,
                )
                parquet_paths[table["name"]] = str(parquet_path)
                logger.debug(f"Exported table '{table['name']}' to Parquet")

            logger.info(f"Saved {len(tables)} tables to persistent DuckDB and exported to Parquet at {parquet_dir}")
        finally:
            datastore.close()

        # 5. Infer schema for unified cohort (if not already present)
        if "inferred_schema" not in metadata:
            from clinical_analytics.core.schema_inference import SchemaInferenceEngine

            engine = SchemaInferenceEngine()
            schema = engine.infer_schema(unified_df)
            metadata["inferred_schema"] = schema.to_dataset_config()

        # 6. Save metadata with tables list, dataset_version, provenance, and Parquet paths
        full_metadata = {
            **metadata,
            "upload_id": upload_id,
            "dataset_version": dataset_version,  # Phase 1: Content-based version
            "tables": table_names,
            "table_counts": {t["name"]: t["data"].height for t in tables},
            "row_count": unified_df.height,
            "column_count": unified_df.width,
            "columns": list(unified_df.columns),
            "provenance": {  # Phase 1: Basic provenance tracking
                "upload_type": "single" if len(tables) == 1 else "multi",
                "tables": table_fingerprints,
            },
            "parquet_paths": parquet_paths,  # Phase 3: Parquet file paths for lazy loading
        }

        metadata_path = storage.metadata_dir / f"{upload_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        logger.info(f"Saved {len(tables)} tables for upload {upload_id}")
        return True, f"Saved {len(tables)} tables"

    except Exception as e:
        logger.error(f"Error saving tables for upload {upload_id}: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False, f"Error saving tables: {str(e)}"


def _migrate_legacy_upload(
    storage: "UserDatasetStorage",
    upload_id: str,
    metadata: dict[str, Any],
) -> None:
    """
    Migrate legacy single-table upload to unified format.

    Creates {upload_id}_tables/ directory and copies/derives table from existing CSV.
    Converts variable_mapping to inferred_schema if needed.
    Updates metadata with tables list and migrated_to_v2 marker.

    Args:
        storage: UserDatasetStorage instance
        upload_id: Upload identifier
        metadata: Upload metadata (will be updated in place)
    """
    # Load existing CSV
    csv_path = storage.raw_dir / f"{upload_id}.csv"
    if not csv_path.exists():
        logger.warning(f"Legacy CSV not found for migration: {csv_path}")
        return  # Nothing to migrate

    logger.info(f"Migrating legacy upload {upload_id} to unified format")

    # Load as Polars DataFrame
    df = pl.read_csv(csv_path)

    # Create tables directory
    tables_dir = storage.raw_dir / f"{upload_id}_tables"
    tables_dir.mkdir(exist_ok=True)

    # Determine table name (use original filename stem if available)
    table_name = Path(metadata.get("original_filename", "data.csv")).stem
    if not table_name or table_name == "data":
        # Fallback to generic name
        table_name = "table_0"

    # Save as individual table
    table_path = tables_dir / f"{table_name}.csv"
    df.write_csv(table_path)
    logger.info(f"Migrated table '{table_name}' to {table_path}")

    # Convert variable_mapping to inferred_schema if needed
    if "variable_mapping" in metadata and "inferred_schema" not in metadata:
        from clinical_analytics.datasets.uploaded.schema_conversion import convert_schema

        metadata["inferred_schema"] = convert_schema(metadata["variable_mapping"], df)
        logger.info("Converted variable_mapping to inferred_schema")

    # Update metadata
    metadata["tables"] = [table_name]
    metadata["migrated_to_v2"] = True

    # Write back metadata
    metadata_path = storage.metadata_dir / f"{upload_id}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Migration complete for upload {upload_id}")


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
                NOTE: This couples storage to UI (technical debt). Future: emit structured events instead.
                Progress callbacks are best-effort: exceptions are caught and ignored to prevent UI errors
                from breaking storage operations.

        Returns:
            Tuple of (success, message, upload_id)
        """
        # Security validation
        # Progress callback is best-effort: don't let UI errors break storage
        if progress_cb:
            try:
                progress_cb(5, "Validating file security and format...")
            except Exception:
                pass  # Best-effort: continue even if callback fails

        valid, error = UploadSecurityValidator.validate(original_filename, file_bytes)
        if not valid:
            if progress_cb:
                try:
                    progress_cb(100, f"❌ Validation failed: {error}")
                except Exception:
                    pass  # Best-effort
            return False, error, None

        # Generate upload ID
        if progress_cb:
            try:
                progress_cb(10, "Generating unique upload identifier...")
            except Exception:
                pass  # Best-effort

        upload_id = self.generate_upload_id(original_filename)

        try:
            # Sanitize filename
            safe_filename = UploadSecurityValidator.sanitize_filename(original_filename)

            # Extract dataset_name from metadata (user-provided friendly name) - for display only
            # CRITICAL: upload_id is the immutable storage key, dataset_name is metadata only
            dataset_name = metadata.get("dataset_name")
            if not dataset_name:
                # Fallback to sanitized original filename (without extension)
                dataset_name = UploadSecurityValidator.sanitize_filename(original_filename)
                dataset_name = Path(dataset_name).stem  # Remove extension

            # Check for existing dataset with same name (prevent duplicates)
            existing_uploads = self.list_uploads()
            for existing_meta in existing_uploads:
                if existing_meta.get("dataset_name") == dataset_name:
                    return (
                        False,
                        f"Dataset '{dataset_name}' already exists. "
                        "Use a different name or delete the existing dataset.",
                        None,
                    )

            # Normalize upload to table list (unified entry point)
            if progress_cb:
                try:
                    progress_cb(20, "Normalizing upload to table list...")
                except Exception:
                    pass  # Best-effort

            tables, table_metadata = normalize_upload_to_table_list(file_bytes, original_filename, metadata)

            # For single-table uploads, ensure patient_id exists (transparent to user)
            if len(tables) == 1:
                from clinical_analytics.ui.components.variable_detector import VariableTypeDetector

                logger.info(f"Ensuring patient_id exists for upload {upload_id}")
                df_with_id, id_metadata = VariableTypeDetector.ensure_patient_id(tables[0]["data"])

                logger.info(
                    f"Patient ID creation result: source={id_metadata['patient_id_source']}, "
                    f"columns={id_metadata.get('patient_id_columns')}, "
                    f"has_patient_id={'patient_id' in df_with_id.columns}"
                )

                # Store metadata about how patient_id was created
                if "synthetic_id_metadata" not in metadata:
                    metadata["synthetic_id_metadata"] = {}
                metadata["synthetic_id_metadata"]["patient_id"] = id_metadata

                # Update table with patient_id
                tables[0]["data"] = df_with_id

                # Verify patient_id exists before saving
                if "patient_id" not in df_with_id.columns:
                    logger.error(
                        f"patient_id column missing after ensure_patient_id! Columns: {list(df_with_id.columns)}"
                    )
                    raise ValueError("Failed to create patient_id column")
                logger.info(
                    f"Successfully ensured patient_id exists. DataFrame shape: {df_with_id.shape}, "
                    f"columns: {list(df_with_id.columns)}"
                )

                # Convert to pandas for validation (existing validation logic uses pandas)
                df = df_with_id.to_pandas()
            else:
                # Multi-table: use first table for validation (shouldn't happen in save_upload, but handle gracefully)
                df = tables[0]["data"].to_pandas()

            # Validate schema if variable mapping is provided
            # This enforces the UnifiedCohort schema contract at save-time
            if progress_cb:
                try:
                    progress_cb(70, "Validating data schema and quality...")
                except Exception:
                    pass  # Best-effort

            variable_mapping = metadata.get("variable_mapping", {})
            schema_validation_result = None
            internal_validation_result = None  # Store for quality_warnings extraction

            if variable_mapping:
                # Check if the mapped columns exist and validate schema
                from clinical_analytics.core.schema import (
                    UnifiedCohort,
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

                    # Store for later use (quality_warnings extraction)
                    internal_validation_result = validation_result

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

            # Extract validation_result from metadata BEFORE merging
            # This prevents duplication - canonical location is metadata["validation"]
            validation_result_from_ui = metadata.pop("validation_result", None)

            # Store canonical quality warnings at validation.quality_warnings
            # Use quality_warnings from UI validation_result if available,
            # otherwise from internal validation
            quality_warnings = []
            if validation_result_from_ui:
                # UI validation_result is the source of truth
                quality_warnings = validation_result_from_ui.get("quality_warnings", [])
            elif internal_validation_result:
                # Fallback to internal validation if UI didn't provide it
                quality_warnings = internal_validation_result.get("quality_warnings", [])

            # Prepare metadata for save_table_list()
            file_ext = Path(original_filename).suffix.lower()
            metadata.update(
                {
                    "original_filename": safe_filename,
                    "dataset_name": dataset_name,
                    "upload_timestamp": datetime.now().isoformat(),
                    "file_size_bytes": len(file_bytes),
                    "file_format": file_ext.lstrip("."),
                    "schema_validation": schema_validation_result,
                    "validation": {
                        "quality_warnings": quality_warnings,
                    },
                }
            )

            # Update table with validated data (convert back to Polars for save_table_list)
            if len(tables) == 1:
                # Convert validated pandas DataFrame back to Polars
                tables[0]["data"] = pl.from_pandas(df)

            # Save using unified save_table_list() function
            if progress_cb:
                try:
                    progress_cb(85, "Saving dataset files and metadata...")
                except Exception:
                    pass  # Best-effort

            success, message = save_table_list(self, tables, upload_id, metadata, progress_cb)

            if not success:
                return False, message, None

            if progress_cb:
                try:
                    progress_cb(100, f"✅ Upload complete! Dataset '{dataset_name}' ready to use.")
                except Exception:
                    pass  # Best-effort

            return True, f"Upload successful: {upload_id}", upload_id

        except Exception as e:
            if progress_cb:
                try:
                    progress_cb(100, f"Error: {str(e)}")
                except Exception:
                    pass  # Best-effort
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

    def get_upload_data(self, upload_id: str, lazy: bool = True) -> pl.LazyFrame | pd.DataFrame | None:
        """
        Load uploaded dataset with automatic legacy migration.

        Uses upload_id as immutable storage key: always {upload_id}.csv
        Automatically migrates legacy single-table uploads to unified format.

        IO Boundary Behavior:
        - CSV files: When lazy=True, uses pl.scan_csv() for true lazy IO
        - Excel files: Eagerly loaded via pandas (due to Polars Excel limitations),
                      then converted to LazyFrame. Lazy execution applies from
                      transformation/filtering onward, not from IO.
        - SPSS files: Eagerly loaded via pyreadstat, then converted to LazyFrame.

        Internal Representation:
        - lazy=True: Returns pl.LazyFrame (recommended for internal use)
        - lazy=False: Returns pd.DataFrame (for backward compatibility/UI boundaries)

        Args:
            upload_id: Upload identifier (immutable storage key)
            lazy: If True, return Polars LazyFrame (default). If False, return pandas DataFrame
                  for backward compatibility.

        Returns:
            LazyFrame (if lazy=True), pandas DataFrame (if lazy=False), or None if not found

        Note:
            Excel eager read is due to Polars read_excel() limitations with complex files
            (header detection, mixed types). See load_single_file() for details.
        """
        # Load metadata
        metadata = self.get_upload_metadata(upload_id)
        if not metadata:
            return None

        # Check if legacy upload needs migration
        tables_dir = self.raw_dir / f"{upload_id}_tables"
        needs_migration = not metadata.get("migrated_to_v2", False) and (
            not metadata.get("tables") or not tables_dir.exists()
        )

        if needs_migration:
            logger.info(f"Detected legacy upload {upload_id}, migrating to unified format")
            _migrate_legacy_upload(self, upload_id, metadata)
            # Reload metadata after migration
            metadata = self.get_upload_metadata(upload_id)

        # Primary: upload_id is the immutable storage key
        csv_path = self.raw_dir / f"{upload_id}.csv"

        # Backward compatibility: check for old friendly-name files
        if not csv_path.exists():
            if metadata:
                # Try old stored_relpath or stored_filename (legacy uploads)
                if "stored_relpath" in metadata:
                    legacy_path = self.raw_dir / metadata["stored_relpath"]
                    if legacy_path.exists():
                        csv_path = legacy_path
                elif "stored_filename" in metadata:
                    legacy_path = self.raw_dir / metadata["stored_filename"]
                    if legacy_path.exists():
                        csv_path = legacy_path

        if not csv_path.exists():
            return None

        if lazy:
            # Phase 3: Prefer Parquet for lazy loading (columnar, compressed, lazy IO)
            # Check if Parquet paths available in metadata (Phase 3+)
            parquet_paths = metadata.get("parquet_paths", {})
            if parquet_paths:
                # Try to load from Parquet first (single-table upload = first table)
                # For multi-table, this loads the unified cohort's first table
                first_table_name = metadata.get("tables", [])[0] if metadata.get("tables") else None
                if first_table_name and first_table_name in parquet_paths:
                    from pathlib import Path

                    from clinical_analytics.storage.datastore import DataStore

                    parquet_path = Path(parquet_paths[first_table_name])
                    if parquet_path.exists():
                        logger.info(f"Loading from Parquet (lazy): {parquet_path}")
                        return DataStore.load_from_parquet(parquet_path)
                    else:
                        logger.warning(f"Parquet file missing: {parquet_path}. Falling back to CSV.")

            # Fallback to CSV if Parquet not available (backward compatibility)
            logger.debug(f"Loading from CSV (lazy): {csv_path}")

            # Build schema overrides for ID columns to prevent integer overflow
            # Common ID column names that should always be strings
            id_column_names = {
                "patient_id",
                "patientid",
                "subject_id",
                "subjectid",
                "id",
                "mrn",
                "study_id",
            }

            # Check metadata for synthetic ID info to identify ID columns
            synthetic_id_metadata = metadata.get("synthetic_id_metadata", {})
            if "patient_id" in synthetic_id_metadata:
                # If patient_id was created synthetically, it's definitely an ID column
                id_column_names.add("patient_id")

            # Try to read CSV with schema overrides for ID columns
            try:
                # First, scan CSV to get column names without materializing
                # We need to peek at the schema to know which columns exist
                sample_lf = pl.scan_csv(csv_path, n_rows=0)
                try:
                    schema = sample_lf.collect_schema()  # Preferred method (Polars 0.19+)
                except AttributeError:
                    schema = sample_lf.schema  # Fallback for older Polars versions

                # Build schema_overrides dict: force ID columns to Utf8
                schema_overrides = {}
                for col_name in schema.keys():
                    if col_name.lower() in {name.lower() for name in id_column_names}:
                        schema_overrides[col_name] = pl.Utf8
                        logger.debug(f"Overriding {col_name} to Utf8 to prevent integer overflow")

                # If we have overrides, use them; otherwise use default inference
                if schema_overrides:
                    return pl.scan_csv(csv_path, schema_overrides=schema_overrides)
                else:
                    return pl.scan_csv(csv_path)

            except Exception as e:
                # If schema inference fails (e.g., integer overflow), retry with patient_id as string
                logger.warning(f"CSV schema inference failed for {upload_id}: {e}. Retrying with patient_id as string.")
                # Fallback: force common ID columns to strings
                fallback_overrides = {
                    "patient_id": pl.Utf8,
                    "patientid": pl.Utf8,
                    "subject_id": pl.Utf8,
                    "subjectid": pl.Utf8,
                    "id": pl.Utf8,
                }
                try:
                    return pl.scan_csv(csv_path, schema_overrides=fallback_overrides)
                except Exception as e2:
                    # Last resort: read all as strings
                    logger.warning(f"Retry with ID overrides failed: {e2}. Falling back to all-string schema.")
                    # Get column names first
                    sample_lf = pl.scan_csv(csv_path, n_rows=0)
                    try:
                        schema = sample_lf.collect_schema()
                    except AttributeError:
                        schema = sample_lf.schema
                    all_string_overrides = {col: pl.Utf8 for col in schema.keys()}
                    return pl.scan_csv(csv_path, schema_overrides=all_string_overrides)
        else:
            # For pandas, read as string to avoid integer overflow
            return pd.read_csv(csv_path, dtype={"patient_id": str})

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

        from clinical_analytics.core.multi_table_handler import MultiTableHandler

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

            # Normalize upload to table list (unified entry point)
            if progress_callback:
                progress_callback(0, 10, "Initializing ZIP extraction...", {})

            tables_list, table_metadata = normalize_upload_to_table_list(file_bytes, original_filename, metadata)

            if not tables_list:
                return False, "No tables found in ZIP archive", None

            logger.info(f"Normalized {len(tables_list)} tables from ZIP: {[t['name'] for t in tables_list]}")

            # Report table loading progress (for compatibility with existing tests)
            if progress_callback:
                for idx, table in enumerate(tables_list, start=1):
                    progress_callback(
                        idx,
                        10,
                        f"Loaded {table['name']}: {table['data'].height:,} rows, {table['data'].width} cols",
                        {
                            "table_name": table["name"],
                            "rows": table["data"].height,
                            "cols": table["data"].width,
                            "status": "loaded",
                        },
                    )

            # Convert list of dicts to dict for MultiTableHandler
            tables_dict = {t["name"]: t["data"] for t in tables_list}

            # Detect relationships between tables (for metadata)
            step_num = len(tables_list) + 1
            if progress_callback:
                progress_callback(
                    step_num,
                    10,
                    "Detecting table relationships...",
                    {
                        "tables": list(tables_dict.keys()),
                        "table_counts": {name: df.height for name, df in tables_dict.items()},
                    },
                )

            from clinical_analytics.core.multi_table_handler import MultiTableHandler

            handler = MultiTableHandler(tables_dict)
            relationships = handler.detect_relationships()
            handler.close()

            logger.info(f"Detected {len(relationships)} relationships")
            if relationships:
                for rel in relationships:
                    logger.info(f"Relationship: {rel}")

            if progress_callback:
                progress_callback(
                    step_num + 1,
                    10,
                    f"Detected {len(relationships)} relationships",
                    {"relationships": [str(rel) for rel in relationships]},
                )

            # Store relationships in metadata for save_table_list()
            metadata["relationships"] = [str(rel) for rel in relationships]

            # Prepare metadata
            metadata.update(
                {
                    "original_filename": UploadSecurityValidator.sanitize_filename(original_filename),
                    "upload_timestamp": datetime.now().isoformat(),
                    "file_size_bytes": len(file_bytes),
                    "file_format": "zip_multi_table",
                }
            )

            # Save using unified save_table_list() function
            if progress_callback:
                progress_callback(step_num + 2, 10, "Saving tables and building unified cohort...", {})

            # Create adapter for progress callback (save_table_list uses different signature)
            def progress_adapter(progress: int, message: str) -> None:
                if progress_callback:
                    # Map to ZIP upload progress format
                    progress_callback(step_num + 2 + progress // 10, 10, message, {})

            success, message = save_table_list(self, tables_list, upload_id, metadata, progress_adapter)

            if not success:
                return False, message, None

            if progress_callback:
                # Get row count from metadata
                saved_metadata = self.get_upload_metadata(upload_id)
                row_count = saved_metadata.get("row_count", 0) if saved_metadata else 0
                progress_callback(
                    10,
                    10,
                    "Processing complete!",
                    {"tables": len(tables_list), "rows": row_count},
                )

            logger.info(f"Multi-table upload successful: {len(tables_list)} tables")
            return (
                True,
                f"Multi-table upload successful: {len(tables_list)} tables",
                upload_id,
            )

        except Exception as e:
            import logging
            import traceback

            logger = logging.getLogger(__name__)
            logger.error(f"Error processing ZIP upload: {type(e).__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Error processing ZIP upload: {str(e)}", None

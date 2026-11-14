# SPDX-License-Identifier: MIT
"""
User Dataset Storage Manager

Handles secure storage, metadata management, and persistence of uploaded datasets.

Storage layout per dataset version (``STORAGE_LAYOUT``), all under the data home's ``data/``:

* ``uploads/raw/{upload_id}/{dataset_version}{ext}``: the uploaded bytes as received (raw landing,
  kept so a version can be re-staged or audited; never rewritten).
* ``parquet/{upload_id}_{table}_{dataset_version}.parquet``: each table, typed once at staging
  (``storage.staging``). For a multi-table upload, the unified cohort is
  ``parquet/{upload_id}__cohort_{dataset_version}.parquet``; a single-table upload's cohort is
  its one table file. This typed Parquet is the one source of truth: the semantic layer, the
  preview and the contract checks all read it.
* ``uploads/metadata/{upload_id}.json``: written last, as the commit point. Until it names a
  version, that version does not exist for readers.

Every file goes through ``storage.atomic`` (temp file, fsync, rename). Data files are named by
content version and never replaced; the cohort Parquet's name also covers the cohort logic and
relationships (``cohort_parquet_path``), and an existing file is only reused after its schema and
row count are checked. Metadata is committed by read-modify-write under the sidecar lock.
Uploads made before this layout (``{id}.csv``) are converted to it on first load
(``UserDatasetStorage.stored_dataset``).
"""

import hashlib
import io
import json
import threading
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Final

import polars as pl
import structlog

from clinical_analytics.core.schema import DataQualityError
from clinical_analytics.storage import atomic
from clinical_analytics.storage.metadata_lock import (
    METADATA_LOCK_TIMEOUT_S,
    FileLockTimeoutError,
    file_lock,
    metadata_lock_path,
)
from clinical_analytics.storage.metadata_lock import update_metadata as _update_metadata
from clinical_analytics.storage.staging import (
    ID_COLUMN_NAMES,
    StagedTable,
    StagingError,
    detect_header_row,
    read_csv_typed,
    read_excel_rows,
    read_excel_typed,
    settle_shared_column_types,
)
from clinical_analytics.ui.config import MULTI_TABLE_ENABLED

logger = structlog.get_logger()

# Defined once elsewhere (storage.staging, storage.metadata_lock); re-exported for old imports.
__all__ = [
    "ID_COLUMN_NAMES",
    "METADATA_LOCK_TIMEOUT_S",
    "FileLockTimeoutError",
    "_update_metadata",
    "file_lock",
    "metadata_lock_path",
]

# Cross-dataset dedup key: compute_dataset_version over the dataset as stored (its Parquet files),
# so it matches for identical data whatever Polars version or upload path produced it. It sits
# beside dataset_version, which is never rewritten (it is part of the persistence keys) and, for
# uploads made before PR #83, hashes Parquet bytes that name the Polars version.
CONTENT_HASH_KEY = "content_hash"
CONTENT_HASH_SCHEME_KEY = "content_hash_scheme"

# Metadata keys of the typed storage layout (see module docstring).
STORAGE_LAYOUT_KEY: Final = "storage_layout"
STORAGE_LAYOUT: Final = "typed_parquet_v1"
COHORT_PARQUET_KEY: Final = "cohort_parquet"
RAW_LANDING_KEY: Final = "raw_landing"
COLUMN_TYPES_KEY: Final = "column_types"
COLUMN_UNITS_KEY: Final = "column_units"
COHORT_BUILD_KEY: Final = "cohort_build"  # multi-table: tables left out of the cohort, and why
LEGACY_UNTYPED_KEY: Final = "legacy_tables_untyped"  # legacy tables kept as old Parquet (no CSV)
BASE_ACTIVE_VERSION_KEY: Final = "_base_active_version"  # active version when a save started
PARQUET_DIRNAME: Final = "parquet"
COHORT_FILE_STEM: Final = "_cohort"  # {upload_id}__cohort_{version}: no sanitized table name starts with "_"
PARQUET_COMPRESSION: Final = "zstd"


class SchemaDriftPolicy(Enum):
    """Policy for handling schema drift during overwrites."""

    REJECT = "reject"  # Reject overwrite if schema differs
    WARN = "warn"  # Allow overwrite but warn user about schema drift
    ALLOW = "allow"  # Allow overwrite silently (permissive)


class SecurityError(Exception):
    """Security-related error (path traversal, symlinks, etc.)."""

    pass


class UploadError(Exception):
    """Upload processing error (malformed files, invalid content, etc.)."""

    pass


class DatasetChangedError(UploadError):
    """Another save changed the dataset's active version while this one ran (retry the upload)."""


def active_version_id(metadata: dict[str, Any] | None) -> str | None:
    """The version readers see: the active ``version_history`` entry, else a legacy ``dataset_version``."""
    if not metadata:
        return None
    active = [v for v in metadata.get("version_history") or [] if v.get("is_active")]
    if active:
        return str(active[0].get("version"))
    legacy = metadata.get("dataset_version")
    return str(legacy) if legacy else None


ProgressCallback = Callable[[int, str], None]


def _notify_progress(progress_cb: ProgressCallback | None, progress: int, message: str) -> None:
    """Best-effort UI progress: a failing callback is logged (type only) and never stops the save."""
    if progress_cb is None:
        return
    try:
        progress_cb(progress, message)
    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError) as e:
        logger.warning("progress_callback_failed", progress=progress, error_type=type(e).__name__)


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
        - tables: list of {"name": str, "data": pl.DataFrame, "notes": column type notes,
          "units": units by column}
        - table_metadata: dict with table_count, table_names, etc.
    """
    # Detect upload type from file extension
    if filename.endswith(".zip"):
        # Multi-table: extract from ZIP
        tables = extract_zip_tables(file_bytes)

        # Extract documentation files from ZIP (Phase 1: ADR004)
        from io import BytesIO

        doc_files = extract_documentation_files(BytesIO(file_bytes))
        logger.info(f"Extracted {len(doc_files)} documentation files from ZIP")
    else:
        # Single-file: wrap in list (becomes multi-table with 1 table)
        staged = load_single_file_staged(file_bytes, filename)
        table_name = Path(filename).stem  # Use original filename stem
        tables = [{"name": table_name, "data": staged.data, "notes": staged.notes, "units": staged.units}]
        doc_files = []  # No docs for single-file uploads by default

    # Handle external PDF/documentation files (standalone upload)
    if metadata and "external_pdf_bytes" in metadata:
        from dataclasses import dataclass

        @dataclass
        class DocFile:
            """Documentation file info."""

            name: str
            path: str
            content: bytes

        # Convert external PDF bytes to DocFile format
        pdf_bytes = metadata["external_pdf_bytes"]
        pdf_filename = metadata.get("external_pdf_filename", "documentation.pdf")
        doc_files.append(DocFile(name=pdf_filename, path=pdf_filename, content=pdf_bytes))
        logger.info(f"Added external documentation file: {pdf_filename}")

    # Build metadata
    table_metadata = {
        "table_count": len(tables),
        "table_names": [t["name"] for t in tables],
        "doc_files": doc_files,  # Phase 1: Store doc files for processing
    }

    return tables, table_metadata


def extract_documentation_files(zip_file: Any) -> list[Any]:
    """
    Extract documentation files (PDF, Markdown, text) from ZIP archive.

    Args:
        zip_file: ZIP file (bytes, BytesIO, or file path)

    Returns:
        List of documentation file info objects with .name attribute

    Example:
        >>> zip_buffer = BytesIO(zip_data)
        >>> doc_files = extract_documentation_files(zip_buffer)
        >>> doc_names = [f.name for f in doc_files]
        ['README.md', 'dictionary.txt', 'study_protocol.pdf']
    """
    from dataclasses import dataclass

    @dataclass
    class DocFile:
        """Documentation file info."""

        name: str
        path: str
        content: bytes

    doc_extensions = {".pdf", ".md", ".txt"}
    doc_files = []

    try:
        with zipfile.ZipFile(zip_file) as zf:
            for file_info in zf.filelist:
                # Skip directories
                if file_info.is_dir():
                    continue

                # Check if file has documentation extension
                file_path = Path(file_info.filename)
                if file_path.suffix.lower() in doc_extensions:
                    content = zf.read(file_info.filename)
                    doc_files.append(DocFile(name=file_path.name, path=file_info.filename, content=content))

        return doc_files

    except zipfile.BadZipFile as e:
        logger.warning(f"Failed to extract documentation files from ZIP: {e}")
        return []


def extract_zip_tables(file_bytes: bytes) -> list[dict[str, Any]]:
    """
    Extract tables from ZIP archive.

    Args:
        file_bytes: ZIP file content

    Returns:
        List of {"name": str, "data": pl.DataFrame, "notes": ..., "units": ...} dicts

    Raises:
        SecurityError: If path traversal or invalid paths detected
        UploadError: If no CSV files, corrupted ZIP, or a table that cannot be staged
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
            tables: list[dict[str, Any]] = []

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

                # Typed over the whole file, column by column (storage.staging); no all-text fallback.
                try:
                    staged = read_csv_typed(csv_content)
                except StagingError as e:
                    raise UploadError(f"Table '{name}': {e}") from e
                df = staged.data
                tables.append({"name": name, "data": df, "notes": staged.notes, "units": staged.units})
                logger.debug(f"Loaded table '{name}': {df.height:,} rows, {df.width} cols")

            # Tables are typed one by one; a column shared by several (a join key such as ward)
            # must have one type in all of them before the cohort is built.
            settled, shared_notes = settle_shared_column_types({t["name"]: t["data"] for t in tables})
            for table in tables:
                table["data"] = settled[table["name"]]
                table["notes"] = (*table["notes"], *(n for name, n in shared_notes if name == table["name"]))
            return tables

    except zipfile.BadZipFile:
        raise UploadError("Corrupted ZIP file")


def _detect_excel_header_row(file_bytes: bytes, max_rows_to_check: int = 5) -> int:
    """
    Detect the header row of an Excel file (0-based); 0 when the file cannot be read.

    Scores the first rows for header-likeness (filled, textual, short, distinct); see
    ``storage.staging.detect_header_row``.

    Args:
        file_bytes: Excel file content (bytes or a readable file object)
        max_rows_to_check: Maximum number of rows to analyze (default 5)

    Returns:
        Row index (0-based) to use as header, or 0 if detection fails
    """
    if not isinstance(file_bytes, bytes):
        if hasattr(file_bytes, "read"):
            file_bytes = file_bytes.read()
        else:
            raise TypeError(f"Expected bytes, got {type(file_bytes)}")
    try:
        rows = read_excel_rows(file_bytes)
    except StagingError as e:
        logger.warning("excel_header_detection_failed", reason=str(e))
        return 0
    return detect_header_row(rows, max_rows_to_check)


def load_single_file_staged(file_bytes: bytes, filename: str) -> StagedTable:
    """
    Load a single file (CSV, Excel, SPSS) as one typed table with its staging notes.

    CSV and Excel are read as text and typed column by column over the whole file
    (``storage.staging``). SPSS files carry their own types and are read straight to Polars.

    Raises:
        ValueError: Unsupported file type.
        StagingError: The file cannot be read or has a row with no ID and no values.
    """
    file_ext = Path(filename).suffix.lower()
    if file_ext == ".csv":
        return read_csv_typed(file_bytes)
    if file_ext in {".xlsx", ".xls"}:
        # Openpyxl reads .xlsx only; an .xls file fails here with a clear StagingError.
        return read_excel_typed(file_bytes)
    if file_ext == ".sav":
        import pyreadstat

        df, _meta = pyreadstat.read_sav(io.BytesIO(file_bytes), output_format="polars")
        return StagedTable(data=df)
    raise ValueError(f"Unsupported file type: {file_ext}")


def load_single_file(file_bytes: bytes, filename: str) -> pl.DataFrame:
    """
    Load single file (CSV, Excel, SPSS) as a typed Polars DataFrame.

    Args:
        file_bytes: File content
        filename: Original filename (determines file type)

    Returns:
        Polars DataFrame

    Raises:
        ValueError: If unsupported file type (StagingError, a ValueError, if unreadable)
    """
    return load_single_file_staged(file_bytes, filename).data


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


def compute_schema_fingerprint(df: pl.DataFrame) -> str:
    """
    Compute deterministic schema fingerprint for a DataFrame.

    Uses SHA256 hash of canonicalized schema JSON (sorted keys) to ensure
    deterministic fingerprint over ordered columns/types.

    Args:
        df: Polars DataFrame

    Returns:
        Schema fingerprint (SHA256 hex digest)
    """
    # Extract schema as ordered list of (column_name, dtype_str)
    schema_list = [(col, str(dtype)) for col, dtype in df.schema.items()]

    # Sort by column name for determinism
    schema_list.sort(key=lambda x: x[0])

    # Create canonical JSON representation
    schema_dict = {"columns": schema_list}
    schema_json = json.dumps(schema_dict, sort_keys=True)

    # Compute SHA256 hash
    return hashlib.sha256(schema_json.encode()).hexdigest()


def detect_schema_drift(schema1: dict[str, Any], schema2: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """
    Detect schema drift between two schemas.

    Compares column names and types to detect additions, removals, and type changes.

    Args:
        schema1: First schema dict with 'columns' key (list of (name, type) tuples)
        schema2: Second schema dict with 'columns' key (list of (name, type) tuples)

    Returns:
        Tuple of (has_drift, drift_details) where drift_details contains:
        - added_columns: list of column names
        - removed_columns: list of column names
        - type_changes: dict of {column_name: (old_type, new_type)}
    """
    # Convert to dicts for easier comparison
    cols1 = {col: dtype for col, dtype in schema1["columns"]}
    cols2 = {col: dtype for col, dtype in schema2["columns"]}

    # Detect drift
    added_columns = [col for col in cols2 if col not in cols1]
    removed_columns = [col for col in cols1 if col not in cols2]
    type_changes: dict[str, tuple[Any, Any]] = {
        col: (cols1[col], cols2[col]) for col in cols1 if col in cols2 and cols1[col] != cols2[col]
    }

    has_drift = bool(added_columns or removed_columns or type_changes)

    drift_details: dict[str, Any] = {}
    if added_columns:
        drift_details["added_columns"] = added_columns
    if removed_columns:
        drift_details["removed_columns"] = removed_columns
    if type_changes:
        drift_details["type_changes"] = type_changes

    return has_drift, drift_details


def classify_schema_drift(old_schema: dict[str, Any], new_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Classify schema drift as 'additive', 'breaking', or 'none' (Phase 3).

    Uses detect_schema_drift() to identify changes, then classifies:
    - 'none': No changes
    - 'additive': Only new columns added (no removals, no type changes)
    - 'breaking': Removed columns, type changes, or mixed changes

    Args:
        old_schema: Old schema dict with 'columns' key (list of (name, type) tuples)
        new_schema: New schema dict with 'columns' key (list of (name, type) tuples)

    Returns:
        Dict with keys:
        - drift_type: "none" | "additive" | "breaking"
        - added_columns: list[str] (empty if none)
        - removed_columns: list[str] (empty if none)
        - type_changes: dict[str, tuple[str, str]] (empty if none)
    """
    has_drift, drift_details = detect_schema_drift(old_schema, new_schema)

    if not has_drift:
        return {
            "drift_type": "none",
            "added_columns": [],
            "removed_columns": [],
            "type_changes": {},
        }

    added_columns = drift_details.get("added_columns", [])
    removed_columns = drift_details.get("removed_columns", [])
    type_changes = drift_details.get("type_changes", {})

    # Classify drift type
    if removed_columns or type_changes:
        # Breaking: removals or type changes
        drift_type = "breaking"
    elif added_columns:
        # Additive: only additions
        drift_type = "additive"
    else:
        # No drift (shouldn't happen if has_drift is True, but defensive)
        drift_type = "none"

    return {
        "drift_type": drift_type,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "type_changes": type_changes,
    }


def apply_schema_drift_policy(
    drift_result: dict[str, Any],
    override: bool = False,
) -> tuple[bool, str, list[str]]:
    """
    Apply schema drift policy to determine if overwrite is allowed (Phase 3).

    Policy rules:
    - 'none': Always allowed
    - 'additive': Allowed without override
    - 'breaking': Blocked unless override=True

    Args:
        drift_result: Result from classify_schema_drift() with keys:
            - drift_type: "none" | "additive" | "breaking"
            - added_columns: list[str]
            - removed_columns: list[str]
            - type_changes: dict[str, tuple[str, str]]
        override: If True, allow breaking changes (with warnings)

    Returns:
        Tuple of (allowed: bool, message: str, warnings: list[str])
    """
    drift_type = drift_result.get("drift_type", "none")
    added_columns = drift_result.get("added_columns", [])
    removed_columns = drift_result.get("removed_columns", [])
    type_changes = drift_result.get("type_changes", {})

    warnings = []

    if drift_type == "none":
        return True, "No schema changes detected", []

    if drift_type == "additive":
        message = f"Additive schema changes detected: {len(added_columns)} new column(s) added"
        return True, message, []

    if override:
        # Allowed with override, but warn
        warning_parts = []
        if removed_columns:
            warning_parts.append(f"{len(removed_columns)} column(s) removed: {', '.join(removed_columns)}")
        if type_changes:
            warning_parts.append(f"{len(type_changes)} type change(s): {', '.join(type_changes.keys())}")
        if added_columns:
            warning_parts.append(f"{len(added_columns)} column(s) added: {', '.join(added_columns)}")

        warnings.append(f"Breaking schema changes overridden: {'; '.join(warning_parts)}")
        message = "Breaking schema changes allowed with override"
        return True, message, warnings
    else:
        # Blocked
        error_parts = []
        if removed_columns:
            error_parts.append(f"removed columns: {', '.join(removed_columns)}")
        if type_changes:
            error_parts.append(f"type changes: {', '.join(type_changes.keys())}")

        message = (
            f"Breaking schema changes detected ({'; '.join(error_parts)}). "
            "Overwrite blocked. Use override=True to proceed."
        )
        warnings.append(message)
        return False, message, warnings


def compute_stored_content_hash(metadata: dict[str, Any]) -> str:
    """
    Content hash of a dataset from its own stored Parquet files, in ``metadata["tables"]`` order.

    Args:
        metadata: Dataset metadata with ``tables`` and ``parquet_paths``

    Returns:
        16-character hex hash (storage.versioning.compute_dataset_version)

    Raises:
        ValueError: If a table has no stored Parquet path
        OSError, polars.exceptions.PolarsError: If a Parquet file is missing or unreadable
    """
    from clinical_analytics.storage.versioning import compute_dataset_version

    parquet_paths: dict[str, str] = metadata.get("parquet_paths") or {}
    table_names: list[str] = metadata.get("tables") or list(parquet_paths)
    missing = [name for name in table_names if name not in parquet_paths]
    if not table_names or missing:
        raise ValueError(f"No stored Parquet for tables {missing or '(none listed)'}")
    return compute_dataset_version([pl.scan_parquet(parquet_paths[name]).collect() for name in table_names])


def assert_metadata_invariants(metadata: dict[str, Any]) -> None:
    """
    Assert critical metadata invariants.

    Validates that metadata satisfies required structural constraints before
    operations that depend on version history (rollback, overwrite, etc.).

    Args:
        metadata: Dataset metadata to validate

    Raises:
        ValueError: If any invariant is violated

    Invariants:
    - version_history must exist and be non-empty
    - Exactly one version must be marked is_active=True
    - Active version must have valid tables structure
    """
    # Invariant 1: version_history must exist
    if "version_history" not in metadata:
        raise ValueError("Metadata missing required 'version_history' field")

    version_history = metadata["version_history"]

    # Invariant 2: version_history must be non-empty
    if not version_history:
        raise ValueError("version_history cannot be empty")

    # Invariant 3: Exactly one active version
    active_versions = [v for v in version_history if v.get("is_active", False)]
    if len(active_versions) == 0:
        raise ValueError("No active version found in version_history")
    if len(active_versions) > 1:
        raise ValueError(f"Multiple active versions found: {len(active_versions)} (should be exactly 1)")

    # Invariant 4: Active version has valid structure
    active_version = active_versions[0]
    if "tables" not in active_version:
        raise ValueError("Active version missing 'tables' field")
    if not isinstance(active_version["tables"], dict):
        raise ValueError("Active version 'tables' must be a dict")


def _parquet_dir(storage: "UserDatasetStorage") -> Path:
    return storage.upload_dir.parent / PARQUET_DIRNAME


def table_parquet_path(storage: "UserDatasetStorage", upload_id: str, table_name: str, dataset_version: str) -> Path:
    """Typed Parquet of one table of one version (the name DataStore used, so deletes still find it)."""
    from clinical_analytics.storage.datastore import _sanitize_table_name

    return _parquet_dir(storage) / f"{upload_id}_{_sanitize_table_name(table_name)}_{dataset_version}.parquet"


def cohort_file_key(dataset_version: str, relationships: list[str] | None) -> str:
    """Name key of a derived cohort: the table version, the cohort logic version and the joins.

    The cohort depends on MultiTableHandler's logic as well as on the tables, so a logic change
    (``COHORT_LOGIC_VERSION``) or other relationships give a new file and never reuse a stale one.
    """
    from clinical_analytics.core import multi_table_handler

    payload = json.dumps(
        [dataset_version, multi_table_handler.COHORT_LOGIC_VERSION, sorted(relationships or [])],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cohort_parquet_path(
    storage: "UserDatasetStorage", upload_id: str, dataset_version: str, relationships: list[str] | None
) -> Path:
    """Typed Parquet of a multi-table upload's unified cohort (named by ``cohort_file_key``)."""
    key = cohort_file_key(dataset_version, relationships)
    return _parquet_dir(storage) / f"{upload_id}_{COHORT_FILE_STEM}_{key}.parquet"


def _write_parquet(df: pl.DataFrame, path: Path) -> bool:
    """Write a versioned data file once (True), or reuse an existing one that matches (False).

    The file name is a content version, so an existing file should hold these rows; its schema
    and row count are checked before it is reused, so a name clash can never serve other data.

    Raises:
        UploadError: an existing file under this name holds other data.
    """
    wrote = atomic.atomic_write(
        path, lambda part: df.write_parquet(part, compression=PARQUET_COMPRESSION), exclusive=True
    )
    if not wrote:
        same_schema = pl.read_parquet_schema(path) == dict(df.schema)
        same_rows = same_schema and pl.scan_parquet(path).select(pl.len()).collect().item() == df.height
        if not (same_schema and same_rows):
            logger.error("existing_parquet_mismatch", file=path.name, same_schema=same_schema)
            raise UploadError(
                f"A stored file named like this version ({path.name}) does not match the data being saved. "
                "Delete the dataset and upload it again."
            )
    return wrote


def _check_unique_table_names(tables: list[dict[str, Any]]) -> None:
    """Fail fast when two tables would be stored under one file name (``Lab Results``, ``lab-results``).

    Raises:
        UploadError: names that sanitize alike, listed so the user can rename one.
    """
    from clinical_analytics.storage.datastore import _sanitize_table_name

    by_file: dict[str, list[str]] = {}
    for table in tables:
        by_file.setdefault(_sanitize_table_name(table["name"]), []).append(str(table["name"]))
    clashes = [names for names in by_file.values() if len(names) > 1]
    if clashes:
        listed = "; ".join(" and ".join(f"'{n}'" for n in names) for names in clashes)
        raise UploadError(f"Tables {listed} would be stored under the same name. Rename one and upload again.")


def _discard_uncommitted(storage: "UserDatasetStorage", upload_id: str, created: list[Path]) -> None:
    """After a failed save, remove the data files this save wrote (never files that existed before).

    Nothing references them (metadata was not committed), so they would only hold patient rows
    no one can see or delete. For an upload that was never committed at all, its derived
    contract and quality files go too.
    """
    import shutil

    from clinical_analytics.contracts.config import CONTRACTS_DIRNAME, DQ_DIRNAME

    log = logger.bind(upload_id=upload_id, files=len(created))
    for path in created:
        path.unlink(missing_ok=True)
    landing_dir = storage.raw_dir / upload_id
    if landing_dir.is_dir() and not any(landing_dir.iterdir()):
        landing_dir.rmdir()
    if not (storage.metadata_dir / f"{upload_id}.json").exists():
        for dirname in (CONTRACTS_DIRNAME, DQ_DIRNAME):
            shutil.rmtree(storage.upload_dir.parent / dirname / upload_id, ignore_errors=True)
    log.info("uncommitted_files_removed")


def describe_cohort_build(report: dict[str, Any] | None) -> list[str]:
    """One sentence per thing the multi-table cohort left out (from MultiTableHandler's report)."""
    if not report:
        return []
    lines = [
        f"{t['table']}: not in the cohort ({t['classification']} table): {t['reason']}."
        for t in report.get("excluded_tables", [])
    ]
    if report.get("null_key_note"):
        lines.append(f"{report['null_key_note']}.")
    lines.extend(
        f"{table}: latest-value columns not created: {why}."
        for table, why in (report.get("omitted_features") or {}).items()
    )
    return lines


def _cast_id_columns(df: pl.DataFrame) -> pl.DataFrame:
    """ID-named columns as text, all in one pass (staging already does this for uploads)."""
    id_cols = [c for c in df.columns if c.lower() in ID_COLUMN_NAMES and df.schema[c] != pl.String]
    return df.with_columns(pl.col(id_cols).cast(pl.Utf8)) if id_cols else df


# Exceptions whose messages this project writes for the user (never with cell values).
_USER_MESSAGE_ERRORS: Final = (StagingError, UploadError, SecurityError, DataQualityError)


def duplicate_name_message(dataset_name: str) -> str:
    """Why an upload under a name already in use is refused, and the two ways forward."""
    return (
        f"A dataset named '{dataset_name}' already exists. To keep both, upload this file under "
        "another name. To replace it, delete the existing dataset, then upload this file."
    )


def no_data_rows_message(filename: str, columns: int) -> str:
    """Why a file whose header has no rows under it is refused, and what to do (no cell values)."""
    read = f"a header of {columns} column{'s' if columns != 1 else ''} and no rows under it"
    if Path(filename).suffix.lower() in (".xlsx", ".xls"):
        return (
            f"No data rows were found in this file: only {read} could be read from its first sheet. "
            "Check that the data is on the first sheet, below its header. If it is, the workbook may "
            "declare a wrong size: open it, save the sheet as CSV, and upload the CSV."
        )
    return f"No data rows were found in this file: only {read} could be read. Add the data rows and upload it again."


def _upload_failure_reason(e: BaseException) -> str:
    """A reason for the user that never quotes cell values.

    Allow-list: only the project's own errors (``_USER_MESSAGE_ERRORS`` and their subclasses
    defined in this project) keep their message. Every other error, a plain ``ValueError`` or a
    library error such as ``pyarrow.ArrowInvalid`` ("Could not convert 'Jane Doe'"), is named by
    type only.
    """
    if isinstance(e, _USER_MESSAGE_ERRORS) and type(e).__module__.startswith("clinical_analytics."):
        return str(e)
    return type(e).__name__


def save_table_list(
    storage: "UserDatasetStorage",
    tables: list[dict[str, Any]],
    upload_id: str,
    metadata: dict[str, Any],
    progress_cb: Callable[[int, str], None] | None = None,
    raw_bytes: bytes | None = None,
) -> tuple[bool, str]:
    """
    Save normalized tables as one new dataset version (unified persistence for every upload type).

    Order (each step crash-safe; metadata last as the commit point, so a crash at any step leaves
    the previous version as the one readers see):

    1. Write the raw landing (``raw_bytes``) and each table's typed Parquet, named by version.
    2. Read the cohort back from its Parquet: everything after this uses that one frame.
    3. Build and probe the query view from that Parquet. If it cannot be built, the upload fails.
    4. Run the contract and quality checks on the same frame.
    5. Commit metadata: read-modify-write under the sidecar lock. If the active version changed
       since this save started (``BASE_ACTIVE_VERSION_KEY``), fail with a retryable error.

    Whatever the exception type, files this call wrote are removed unless the commit happened.

    Args:
        storage: UserDatasetStorage instance
        tables: Normalized table list [{"name": str, "data": pl.DataFrame, "notes"?, "units"?}]
        upload_id: Upload identifier
        metadata: Upload metadata
        progress_cb: Optional progress callback
        raw_bytes: The uploaded file as received (raw landing); None when there is none

    Returns:
        (success, message)
    """
    log = logger.bind(upload_id=upload_id, table_count=len(tables))
    created: list[Path] = []  # data files this call wrote; removed if the save does not commit
    committed = False
    try:
        _check_unique_table_names(tables)

        # 1. Extract documentation context (Phase 1: ADR004)
        # Skip extraction if doc_context already exists (idempotency safeguard)
        if "doc_files" in metadata and metadata["doc_files"]:
            if "doc_context" not in metadata:
                import tempfile

                from clinical_analytics.core.doc_parser import extract_context_from_docs

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    doc_paths = []
                    for doc_file in metadata["doc_files"]:
                        doc_temp_path = temp_path / doc_file.name
                        doc_temp_path.write_bytes(doc_file.content)
                        doc_paths.append(doc_temp_path)
                    doc_context = extract_context_from_docs(doc_paths)
                    metadata["doc_context"] = doc_context
                    log.info("doc_context_extracted", chars=len(doc_context), files=len(doc_paths))
            else:
                log.debug("doc_context_exists_skipped")
            # doc_files are not JSON serializable; only doc_context is kept
            metadata.pop("doc_files", None)

        # Processed into doc_files above; not JSON serializable
        metadata.pop("external_pdf_bytes", None)
        metadata.pop("external_pdf_filename", None)

        # 2. Convert schema (AFTER normalization, has df access)
        if "variable_mapping" in metadata and tables:
            from clinical_analytics.datasets.uploaded.schema_conversion import convert_schema

            metadata["inferred_schema"] = convert_schema(
                metadata["variable_mapping"],
                tables[0]["data"],
                metadata.get("synthetic_id_metadata"),
            )

        # 3. Dataset version and table fingerprints
        from clinical_analytics.storage.versioning import (
            CONTENT_HASH_SCHEME,
            compute_dataset_version,
            compute_table_fingerprint,
        )

        dataset_version = compute_dataset_version([t["data"] for t in tables])
        log = log.bind(dataset_version=dataset_version)
        table_fingerprints = [compute_table_fingerprint(t["data"], t["name"]) for t in tables]
        table_names = [t["name"] for t in tables]

        # 4. Unified cohort (single table: the table itself). A multi-table cohort reports the
        # tables it left out; a key that breaks one row per patient raises DataQualityError.
        cohort_build: dict[str, Any] | None = None
        if len(tables) == 1:
            unified_df = tables[0]["data"]
        else:
            from clinical_analytics.core.multi_table_handler import MultiTableHandler

            with MultiTableHandler({t["name"]: t["data"] for t in tables}) as handler:
                if not metadata.get("relationships"):
                    relationships = handler.detect_relationships()
                    metadata["relationships"] = [str(rel) for rel in relationships]
                    log.info("relationships_detected", count=len(relationships))
                build = handler.build_unified_cohort_with_report()
            unified_df = build.data
            cohort_build = {
                "anchor_table": build.anchor_table,
                "grain_key": build.grain_key,
                "excluded_tables": [
                    {"table": t.table_name, "classification": t.classification, "reason": t.reason}
                    for t in build.excluded_tables
                ],
                "null_key_note": build.null_key_note,
                "omitted_features": dict(build.omitted_features),
            }
            log.info("cohort_built", excluded_tables=len(build.excluded_tables))

        # 5. Raw landing + typed Parquet per table (versioned, never replaced)
        for table in tables:
            table["data"] = _cast_id_columns(table["data"])
        unified_df = _cast_id_columns(unified_df)

        raw_landing: Path | None = None
        if raw_bytes is not None:
            ext = Path(str(metadata.get("original_filename", ""))).suffix.lower() or ".bin"
            raw_landing = storage.raw_dir / upload_id / f"{dataset_version}{ext}"
            if atomic.atomic_write_bytes(raw_landing, raw_bytes, exclusive=True):
                created.append(raw_landing)

        parquet_paths: dict[str, str] = {}
        for table in tables:
            path = table_parquet_path(storage, upload_id, table["name"], dataset_version)
            if _write_parquet(table["data"], path):
                created.append(path)
            parquet_paths[table["name"]] = str(path)
        if len(tables) == 1:
            cohort_path = Path(parquet_paths[tables[0]["name"]])
        else:
            cohort_path = cohort_parquet_path(storage, upload_id, dataset_version, metadata.get("relationships"))
            if _write_parquet(unified_df, cohort_path):
                created.append(cohort_path)
        log.info("typed_parquet_written", tables=len(parquet_paths))

        # 6. From here on, the stored frame is the one source of truth (what queries will read).
        stored_df = pl.read_parquet(cohort_path)
        # The tables in memory are what was written (or a checked match of it): no re-read needed.
        content_hash = compute_dataset_version([t["data"] for t in tables])

        inferred: Any = None  # SchemaInferenceEngine result, computed at most once

        def inferred_schema() -> Any:
            nonlocal inferred
            if inferred is None:
                from clinical_analytics.core.schema_inference import SchemaInferenceEngine

                inferred = SchemaInferenceEngine().infer_schema(stored_df)
            return inferred

        if "inferred_schema" not in metadata:
            metadata["inferred_schema"] = inferred_schema().to_dataset_config()

        column_types = [n.as_dict() for t in tables for n in t.get("notes", ())]
        column_units: dict[str, str] = {}
        for t in tables:
            column_units.update(t.get("units") or {})

        canonical_tables = {
            t["name"]: {
                "parquet_path": parquet_paths[t["name"]],
                "row_count": t["data"].height,
                "column_count": t["data"].width,
                "schema_fingerprint": compute_schema_fingerprint(t["data"]),
            }
            for t in tables
        }

        # Version history entry and event (Phase 2/5); merged into the stored history at commit.
        is_overwrite = metadata.pop("_is_overwrite", False)
        base_active_version = metadata.pop(BASE_ACTIVE_VERSION_KEY, None)
        provided_history: list[dict[str, Any]] = metadata.pop("_existing_version_history", None) or []
        metadata.pop("_existing_events", None)  # re-read under the lock at commit
        now_utc = datetime.now(UTC)
        event_id = uuid.uuid4().hex
        timestamp = now_utc.isoformat().replace("+00:00", "Z")
        version_entry = {
            "version": dataset_version,
            "created_at": timestamp,
            "event_id": event_id,
            "event_type": "overwrite" if is_overwrite else "upload",
            "is_active": True,
            "tables": canonical_tables,
            COHORT_PARQUET_KEY: str(cohort_path),
            "source_filename": metadata.get("original_filename", ""),
        }
        event_entry: dict[str, Any] = {
            "event_id": event_id,
            "timestamp": timestamp,
            "event_type": "version_activated" if is_overwrite else "upload_created",
            "version": dataset_version,
            "upload_id": upload_id,
        }
        if is_overwrite and base_active_version:
            event_entry["previous_version"] = base_active_version

        full_metadata = {
            **metadata,
            "upload_id": upload_id,
            "dataset_version": dataset_version,
            CONTENT_HASH_KEY: content_hash,
            CONTENT_HASH_SCHEME_KEY: CONTENT_HASH_SCHEME,
            STORAGE_LAYOUT_KEY: STORAGE_LAYOUT,
            COHORT_PARQUET_KEY: str(cohort_path),
            RAW_LANDING_KEY: str(raw_landing) if raw_landing else None,
            COLUMN_TYPES_KEY: column_types,
            COLUMN_UNITS_KEY: column_units,
            COHORT_BUILD_KEY: cohort_build,
            "tables": table_names,
            "table_counts": {t["name"]: t["data"].height for t in tables},
            "row_count": stored_df.height,
            "column_count": stored_df.width,
            "columns": list(stored_df.columns),
            "provenance": {
                "upload_type": "single" if len(tables) == 1 else "multi",
                "tables": table_fingerprints,
            },
            "parquet_paths": parquet_paths,
            "alias_mappings": metadata.get("alias_mappings", {"user_aliases": {}, "system_aliases": {}}),
            "version_history": [*(dict(v, is_active=False) for v in provided_history), version_entry],
            "events": [event_entry],
        }

        # 7. Build and probe the query view from the stored Parquet, before anything is committed.
        from clinical_analytics.datasets.uploaded.definition import SemanticLayerBuildError, probe_query_view

        try:
            probe_query_view(storage, upload_id, full_metadata, expected_rows=stored_df.height)
        except SemanticLayerBuildError as e:
            log.error("upload_query_view_failed", reason=str(e))
            return False, f"The dataset could not be prepared for questions: {e}"

        # 8. Example questions (best effort: a failure here never blocks the upload)
        if "example_questions" not in metadata:
            from unittest.mock import MagicMock

            from clinical_analytics.core.question_generator import generate_upload_questions

            try:
                mock_semantic_layer = MagicMock()
                mock_semantic_layer.get_column_alias_index.return_value = {c: c for c in stored_df.columns}
                full_metadata["example_questions"] = generate_upload_questions(
                    semantic_layer=mock_semantic_layer,
                    inferred_schema=inferred_schema(),
                    doc_context=metadata.get("doc_context"),
                )
            except (ValueError, KeyError, TypeError, AttributeError, pl.exceptions.PolarsError) as e:
                log.warning("upload_questions_failed", error_type=type(e).__name__)

        # 9. Data contract + quality checks on the stored frame
        from clinical_analytics.contracts import service as contract_service

        quality_report = contract_service.check_upload_or_log(
            storage.upload_dir.parent,
            stored_df,
            upload_id=upload_id,
            dataset_version=dataset_version,
            metadata=full_metadata,
            overlay_dir=storage.metadata_dir,
        )
        if quality_report.status == "ran":
            full_metadata["quality"] = {
                "status": "ran",
                "run_ts": quality_report.run_ts,
                "failed": quality_report.failed,
                "warnings": quality_report.warnings,
            }
        else:
            full_metadata["quality"] = {"status": "not_run", "reason": quality_report.reason}

        # 10. Commit: metadata last, by read-modify-write under the lock. The stored history and
        # events are re-read here, so a concurrent change is merged, never overwritten.
        def commit(current: dict[str, Any]) -> bool:
            now_active = active_version_id(current)
            if now_active != base_active_version:
                log.warning("dataset_changed_during_save", expected=base_active_version, found=now_active)
                raise DatasetChangedError(
                    "The dataset changed while this upload was being saved (another upload finished first). "
                    "Upload again."
                )
            history = current.get("version_history") or provided_history
            merged = {
                **full_metadata,
                "version_history": sorted(
                    [*(dict(v, is_active=False) for v in history), version_entry],
                    key=lambda v: v.get("created_at", ""),
                ),
                "events": [*(current.get("events") or []), event_entry],
            }
            current.clear()
            current.update(merged)
            return True

        _update_metadata(storage.metadata_dir / f"{upload_id}.json", commit, missing_ok=True)
        committed = True
        log.info("dataset_version_committed", rows=stored_df.height, columns=stored_df.width)
        return True, f"Saved {len(tables)} tables"

    except (
        UploadError,
        StagingError,
        DataQualityError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        pl.exceptions.PolarsError,
    ) as e:
        log.error("save_tables_failed", error_type=type(e).__name__)
        if isinstance(e, DataQualityError):
            return False, f"The tables could not be combined into one row per patient: {e}"
        return False, f"Error saving tables: {_upload_failure_reason(e)}"
    finally:
        # Runs on every exit, whatever the exception type: nothing uncommitted stays on disk.
        if not committed:
            try:
                _discard_uncommitted(storage, upload_id, created)
            except OSError as cleanup_error:
                log.error("uncommitted_files_not_removed", error_type=type(cleanup_error).__name__)


_MIGRATION_LOCKS: dict[str, threading.Lock] = {}
_MIGRATION_LOCKS_GUARD = threading.Lock()


def _migration_lock(upload_id: str) -> threading.Lock:
    """Per-upload lock so concurrent readers never convert the same legacy upload twice."""
    with _MIGRATION_LOCKS_GUARD:
        return _MIGRATION_LOCKS.setdefault(upload_id, threading.Lock())


@dataclass(frozen=True)
class StoredDataset:
    """Where one dataset version's typed data is: Parquet files, or (legacy fallback) frames in memory.

    ``frames`` is set only for a legacy upload whose conversion could not be saved; the frames
    are staged exactly as the conversion would have written them.
    """

    metadata: dict[str, Any]
    cohort_path: Path | None
    table_paths: dict[str, Path] = field(default_factory=dict)
    cohort_frame: pl.DataFrame | None = None
    table_frames: dict[str, pl.DataFrame] = field(default_factory=dict)

    def scan_cohort(self) -> pl.LazyFrame:
        if self.cohort_frame is not None:
            return self.cohort_frame.lazy()
        if self.cohort_path is None:
            raise FileNotFoundError("No stored cohort for this dataset")
        return pl.scan_parquet(self.cohort_path)


def _legacy_source_path(storage: "UserDatasetStorage", upload_id: str, metadata: dict[str, Any]) -> Path | None:
    """The unified cohort file of an upload made before typed storage."""
    candidates = [storage.raw_dir / f"{upload_id}.csv"]
    for key in ("stored_relpath", "stored_filename"):
        if metadata.get(key):
            candidates.append(storage.raw_dir / str(metadata[key]))
    return next((p for p in candidates if p.is_file()), None)


def _stage_legacy_file(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return _cast_id_columns(read_excel_typed(path.read_bytes()).data)
    return _cast_id_columns(read_csv_typed(path.read_bytes()).data)


def _convert_legacy_upload(storage: "UserDatasetStorage", upload_id: str, metadata: dict[str, Any]) -> StoredDataset:
    """
    Convert an upload made before typed storage to typed Parquet, once, on first load.

    The unified CSV (what the semantic layer used to read) and every ``{id}_tables/*.csv`` are
    staged with the same typing as new uploads and written as versioned Parquet, so the tables
    and the cohort follow one typing regime. An old per-table Parquet (written by the pandas
    round trip) is reused only for a table with no CSV, and is listed in metadata under
    ``LEGACY_UNTYPED_KEY`` so the mismatch is visible. The new files are named by the version of
    the re-staged data (never the legacy version, which hashed other bytes and may name an old
    file). Metadata is updated last. If the Parquet or metadata cannot be written, the staged
    frames are served from memory and the fallback is logged.

    Raises:
        FileNotFoundError: no stored data for this upload.
        StagingError: the stored file cannot be staged.
    """
    log = logger.bind(upload_id=upload_id)
    source = _legacy_source_path(storage, upload_id, metadata)
    if source is None:
        raise FileNotFoundError(f"No stored data for upload {upload_id}")
    cohort = _stage_legacy_file(source)

    from clinical_analytics.storage.versioning import compute_dataset_version

    tables_dir = storage.raw_dir / f"{upload_id}_tables"
    old_parquet = {name: Path(p) for name, p in (metadata.get("parquet_paths") or {}).items() if Path(p).is_file()}
    table_frames: dict[str, pl.DataFrame] = {}
    untyped: dict[str, Path] = {}
    for name in metadata.get("tables") or []:
        csv_path = tables_dir / f"{name}.csv"
        if csv_path.is_file():
            table_frames[name] = _stage_legacy_file(csv_path)
        elif name in old_parquet:
            untyped[name] = old_parquet[name]
    version = compute_dataset_version([cohort, *table_frames.values()])

    try:
        cohort_path = cohort_parquet_path(storage, upload_id, version, metadata.get("relationships"))
        _write_parquet(cohort, cohort_path)
        table_paths = dict(untyped)
        for name, df in table_frames.items():
            table_paths[name] = table_parquet_path(storage, upload_id, name, version)
            _write_parquet(df, table_paths[name])

        def mutate(current: dict[str, Any]) -> bool:
            current[STORAGE_LAYOUT_KEY] = STORAGE_LAYOUT
            current[COHORT_PARQUET_KEY] = str(cohort_path)
            current["parquet_paths"] = {name: str(p) for name, p in table_paths.items()}
            current[LEGACY_UNTYPED_KEY] = sorted(untyped)
            current.setdefault("dataset_version", version)  # aliases are scoped by the original one
            current.setdefault("tables", [])
            if "variable_mapping" in current and "inferred_schema" not in current:
                from clinical_analytics.datasets.uploaded.schema_conversion import convert_schema

                current["inferred_schema"] = convert_schema(
                    current["variable_mapping"], cohort, synthetic_id_metadata=current.get("synthetic_id_metadata")
                )
            current["legacy_converted_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            return True

        updated = _update_metadata(storage.metadata_dir / f"{upload_id}.json", mutate)
    except (OSError, RuntimeError, UploadError) as e:
        log.warning("legacy_upload_fallback", reason=type(e).__name__, source=source.suffix)
        return StoredDataset(
            metadata=metadata,
            cohort_path=None,
            table_paths=untyped,
            cohort_frame=cohort,
            table_frames=table_frames,
        )
    if untyped:
        log.warning("legacy_tables_untyped", tables=len(untyped))
    log.info("legacy_upload_converted", tables=len(table_paths))
    return StoredDataset(metadata=updated, cohort_path=cohort_path, table_paths=table_paths)


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
            upload_dir: Base directory for uploads (default: the per-user data home,
                never the source tree; see core.data_home)
        """
        if upload_dir is None:
            from clinical_analytics.core.data_home import ensure_data_home, uploads_dir, warn_if_legacy_repo_data

            warn_if_legacy_repo_data(Path(__file__).parent.parent.parent.parent.parent)
            ensure_data_home()
            upload_dir = uploads_dir()

        self.upload_dir = Path(upload_dir)
        self.raw_dir = self.upload_dir / "raw"
        self.metadata_dir = self.upload_dir / "metadata"

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def generate_upload_id(self, filename: str) -> str:
        """
        Generate unique upload ID.

        Random (uuid4): timestamp + filename hash collided for two uploads of the same file name
        in one second, so one upload overwrote, and a delete removed, the other. The filename is
        not used, so ids leak nothing about it; ids match the delete id pattern.

        Args:
            filename: Original filename (unused; kept for the call signature)

        Returns:
            Unique upload ID, ``user_upload_<32 hex>``
        """
        return f"user_upload_{uuid.uuid4().hex}"

    def find_datasets_by_content_hash(
        self, content_hash: str, exclude_upload_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Find all datasets whose stored content hash (CONTENT_HASH_KEY) matches.

        Used for cross-dataset deduplication to warn users about duplicate content
        with different names. A dataset stored before the content hash existed, or under an
        older CONTENT_HASH_SCHEME, gets it backfilled from its own Parquet files first
        (_ensure_content_hash), so old and new uploads of the same data match.

        Args:
            content_hash: Hash from compute_stored_content_hash
            exclude_upload_id: Upload to leave out (the one just saved)

        Returns:
            List of metadata dicts for matching datasets
        """
        matching_datasets = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skipping unreadable metadata file %s: %s", metadata_file.name, type(e).__name__)
                continue
            if not isinstance(metadata, dict):
                continue

            metadata = self._ensure_content_hash(metadata_file, metadata)
            if exclude_upload_id is not None and metadata.get("upload_id") == exclude_upload_id:
                continue
            if metadata.get(CONTENT_HASH_KEY) == content_hash:
                matching_datasets.append(metadata)

        return matching_datasets

    def _ensure_content_hash(self, metadata_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Backfill CONTENT_HASH_KEY for a dataset stored before it existed (or under an older scheme).

        The hash is computed from the dataset's own stored Parquet and written next to the
        untouched dataset_version. Idempotent: a dataset that already has a current-scheme hash
        is returned as is. A dataset whose data cannot be read is logged and returned unchanged
        (it cannot be matched, but it must not block an upload). Logs ids and error types only,
        never row values.

        Args:
            metadata_path: The dataset's metadata JSON
            metadata: Its parsed contents

        Returns:
            Metadata, with the content hash when it could be computed
        """
        from clinical_analytics.storage.versioning import CONTENT_HASH_SCHEME

        if metadata.get(CONTENT_HASH_SCHEME_KEY) == CONTENT_HASH_SCHEME and metadata.get(CONTENT_HASH_KEY):
            return metadata
        upload_id = metadata.get("upload_id", metadata_path.stem)
        if not metadata.get("parquet_paths"):
            # Pre-Parquet legacy upload: nothing stored to hash until it is migrated and re-saved.
            logger.debug("No stored Parquet to backfill content_hash for upload %s", upload_id)
            return metadata

        try:
            content_hash = compute_stored_content_hash(metadata)
        except (OSError, ValueError, pl.exceptions.PolarsError) as e:
            logger.warning(
                "content_hash backfill skipped for upload %s: stored data unreadable (%s)",
                upload_id,
                type(e).__name__,
            )
            return metadata

        def mutate(current: dict[str, Any]) -> bool:
            if current.get("parquet_paths") != metadata.get("parquet_paths"):
                return False  # overwritten since we read it: that save stored its own hash; keep it
            current[CONTENT_HASH_KEY] = content_hash
            current[CONTENT_HASH_SCHEME_KEY] = CONTENT_HASH_SCHEME
            return True

        try:
            current = _update_metadata(metadata_path, mutate)
        except (RuntimeError, OSError) as e:
            logger.warning("content_hash_backfill_not_saved", upload_id=upload_id, error_type=type(e).__name__)
            return {**metadata, CONTENT_HASH_KEY: content_hash, CONTENT_HASH_SCHEME_KEY: CONTENT_HASH_SCHEME}

        logger.info("Backfilled content_hash for upload %s", upload_id)
        return dict(current)

    def save_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
        metadata: dict[str, Any],
        progress_cb: Callable[[int, str], None] | None = None,
        overwrite: bool = False,
        raise_staging_errors: bool = False,
    ) -> tuple[bool, str, str | None]:
        """
        Save uploaded file with security validation.

        Args:
            file_bytes: File content
            original_filename: Original filename
            metadata: Upload metadata (variable types, mappings, etc.)
            progress_cb: Optional callback(progress: int, message: str) for UI updates (0-100, not 0.0-1.0)
                NOTE: This couples storage to UI (technical debt). Future: emit structured events instead.
                Progress callbacks are best-effort: a failing callback is logged (``_notify_progress``)
                and never breaks the save.
            overwrite: If True, allow overwriting existing dataset with same name (Phase 4.2)
            raise_staging_errors: If True, a file that cannot be read raises ``StagingError``
                (its message is written for the user) instead of returning a failure, so the
                API can answer 400 rather than 500.

        Returns:
            Tuple of (success, message, upload_id)

        Raises:
            StagingError: only with ``raise_staging_errors``, when the file cannot be read.
        """
        # Security validation
        # Progress callback is best-effort: don't let UI errors break storage
        _notify_progress(progress_cb, 5, "Validating file security and format...")

        valid, error = UploadSecurityValidator.validate(original_filename, file_bytes)
        if not valid:
            _notify_progress(progress_cb, 100, f"❌ Validation failed: {error}")
            return False, error, None

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

            # Phase 4.2: Check for existing dataset with same name
            existing_uploads = self.list_uploads()
            existing_upload_id = None
            existing_version_history = None
            base_active_version: str | None = None  # active version when this save started

            # The bundled sample matches only the earlier sample (is_demo), never a clinician's
            # dataset that happens to share its display name; that one is left untouched.
            saving_demo = metadata.get("is_demo") is True
            for existing_meta in existing_uploads:
                if saving_demo:
                    same_dataset = existing_meta.get("is_demo") is True
                else:
                    same_dataset = existing_meta.get("dataset_name") == dataset_name
                if same_dataset:
                    if not overwrite:
                        # Reject duplicate without overwrite flag
                        return False, duplicate_name_message(dataset_name), None
                    else:
                        # Overwrite mode: preserve existing upload_id and version_history
                        existing_upload_id = existing_meta.get("upload_id")
                        existing_version_history = existing_meta.get("version_history", [])
                        base_active_version = active_version_id(existing_meta)

                        # Phase 4.2: Legacy dataset migration during overwrite
                        # If existing_version_history is empty but legacy metadata exists,
                        # create synthetic version entry
                        if len(existing_version_history) == 0:
                            # Legacy dataset: create synthetic version_history from top-level metadata
                            legacy_version = existing_meta.get("dataset_version")
                            legacy_created_at = existing_meta.get("created_at") or existing_meta.get("upload_timestamp")
                            legacy_tables = {}

                            # Reconstruct tables from parquet_paths or duckdb_tables if available
                            parquet_paths = existing_meta.get("parquet_paths", {})
                            if parquet_paths:
                                # Convert parquet_paths dict to canonical tables structure
                                for table_name, parquet_path in parquet_paths.items():
                                    legacy_tables[table_name] = {
                                        "parquet_path": parquet_path,
                                        "duckdb_table": existing_meta.get("duckdb_tables", {}).get(
                                            table_name, table_name
                                        ),
                                        "row_count": existing_meta.get("table_counts", {}).get(table_name, 0),
                                        "column_count": existing_meta.get("column_count", 0),
                                        "schema_fingerprint": existing_meta.get("schema_fingerprint", ""),
                                    }
                            else:
                                # Fallback: create minimal table entry
                                table_name = (
                                    existing_meta.get("tables", ["table_0"])[0]
                                    if existing_meta.get("tables")
                                    else "table_0"
                                )
                                legacy_tables[table_name] = {
                                    "parquet_path": "",  # Will be filled if parquet exists
                                    "duckdb_table": table_name,
                                    "row_count": existing_meta.get("row_count", 0),
                                    "column_count": existing_meta.get("column_count", 0),
                                    "schema_fingerprint": "",
                                }

                            synthetic_version_entry = {
                                "version": legacy_version or "legacy",
                                "created_at": (
                                    legacy_created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
                                ),
                                "event_id": uuid.uuid4().hex,
                                "event_type": "upload",
                                "is_active": False,  # Will be marked inactive when new version added
                                "tables": legacy_tables,
                                "source_filename": existing_meta.get("original_filename", ""),
                            }

                            existing_version_history = [synthetic_version_entry]
                            logger.info("Migrated legacy dataset to version_history (1 version)")

                        logger.info(
                            f"Overwriting existing dataset '{dataset_name}' "
                            f"(upload_id={existing_upload_id}, "
                            f"version_history={existing_version_history is not None}, "
                            f"{len(existing_version_history) if existing_version_history else 0} versions)"
                        )
                        break

            # Generate upload ID (or use existing if overwriting)
            _notify_progress(progress_cb, 10, "Generating unique upload identifier...")

            # Phase 4.2: Use existing upload_id if overwriting
            if existing_upload_id:
                upload_id = existing_upload_id
            else:
                upload_id = self.generate_upload_id(original_filename)

            # Normalize upload to table list (unified entry point)
            _notify_progress(progress_cb, 20, "Normalizing upload to table list...")

            tables, table_metadata = normalize_upload_to_table_list(file_bytes, original_filename, metadata)
            # A file with a header and no data rows answers nothing: refuse it before an ID is
            # added, so a row number on zero rows never makes it look "ready".
            if all(t["data"].height == 0 for t in tables):
                raise StagingError(no_data_rows_message(original_filename, tables[0]["data"].width if tables else 0))

            # Mark all existing versions as inactive (the drift check below reads stored metadata)
            if existing_version_history is not None:
                for version_entry in existing_version_history:
                    version_entry["is_active"] = False

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

                # Verify an ID exists before saving: patient_id, or the row number added when no
                # column identifies a patient (answers then count rows, never patients)
                has_row_number = id_metadata.get("patient_id_source") == "row_index"
                if "patient_id" not in df_with_id.columns and not has_row_number:
                    logger.error(
                        f"patient_id column missing after ensure_patient_id! Columns: {list(df_with_id.columns)}"
                    )
                    raise ValueError("Failed to create patient_id column")
                logger.info(
                    f"Ensured an ID column ({'row number' if has_row_number else 'patient_id'}). "
                    f"DataFrame shape: {df_with_id.shape}"
                )

            df = tables[0]["data"]

            # Overwrite: block breaking type drift against the stored version, after the same
            # normalization the save applies (ID columns as text), so an identical re-upload passes.
            if existing_version_history and overwrite and existing_upload_id:
                blocked = self._check_overwrite_drift(existing_upload_id, _cast_id_columns(df), metadata)
                if blocked is not None:
                    return False, blocked, None

            # Validate schema if variable mapping is provided
            # This enforces the UnifiedCohort schema contract at save-time
            _notify_progress(progress_cb, 70, "Validating data schema and quality...")

            variable_mapping = metadata.get("variable_mapping", {})
            schema_validation_result = None
            internal_validation_result = None  # Store for quality_warnings extraction

            if variable_mapping:
                # Check if the mapped columns exist and validate schema
                from clinical_analytics.core.schema import (
                    UnifiedCohort,
                )

                # Validate a renamed view of the typed frame (the stored frame is not changed)
                cohort_df = df

                patient_id_col = variable_mapping.get("patient_id")
                outcome_col = variable_mapping.get("outcome")

                rename_map = {}
                if patient_id_col and patient_id_col in cohort_df.columns:
                    rename_map[patient_id_col] = UnifiedCohort.PATIENT_ID

                if outcome_col and outcome_col in cohort_df.columns:
                    rename_map[outcome_col] = UnifiedCohort.OUTCOME

                time_vars = variable_mapping.get("time_variables", {})
                if time_vars:
                    time_zero_col = time_vars.get("time_zero")
                    if time_zero_col and time_zero_col in cohort_df.columns:
                        rename_map[time_zero_col] = UnifiedCohort.TIME_ZERO

                if rename_map:
                    # Drop pre-existing columns that already carry a canonical name but are not the
                    # mapped source (e.g. a raw "outcome" column when the user maps outcome -> "mortality").
                    colliding = [
                        target
                        for source, target in rename_map.items()
                        if target != source and target in cohort_df.columns and target not in rename_map
                    ]
                    if colliding:
                        cohort_df = cohort_df.drop(colliding)
                    cohort_df = cohort_df.rename({k: v for k, v in rename_map.items() if k != v})

                    if UnifiedCohort.OUTCOME_LABEL not in cohort_df.columns:
                        cohort_df = cohort_df.with_columns(
                            pl.lit(outcome_col or "outcome").alias(UnifiedCohort.OUTCOME_LABEL)
                        )
                    if UnifiedCohort.TIME_ZERO not in cohort_df.columns:
                        cohort_df = cohort_df.with_columns(pl.lit(datetime.now()).alias(UnifiedCohort.TIME_ZERO))

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
                    "source_file_sha256": hashlib.sha256(file_bytes).hexdigest(),
                    "file_format": file_ext.lstrip("."),
                    "schema_validation": schema_validation_result,
                    "validation": {
                        "quality_warnings": quality_warnings,
                    },
                }
            )

            # Phase 4.2: Pass existing_version_history for overwrite; the commit re-reads the stored
            # history and events under the metadata lock and checks the active version is unchanged.
            if existing_version_history is not None:
                metadata["_existing_version_history"] = existing_version_history
                metadata["_is_overwrite"] = True  # Phase 5: Mark as overwrite event
            metadata[BASE_ACTIVE_VERSION_KEY] = base_active_version

            # Save using unified save_table_list() function
            _notify_progress(progress_cb, 85, "Saving dataset files and metadata...")

            success, message = save_table_list(self, tables, upload_id, metadata, progress_cb, raw_bytes=file_bytes)

            if not success:
                return False, message, None

            # Phase 1: Check for duplicate content with different names (warn, don't block). Runs on
            # the stored content hash, after the save, so it compares data in the form every dataset
            # is stored in (patient_id added, ID columns as text), old uploads included.
            saved_hash = (self.get_upload_metadata(upload_id) or {}).get(CONTENT_HASH_KEY)
            duplicate_warning = None
            duplicate_datasets = (
                self.find_datasets_by_content_hash(saved_hash, exclude_upload_id=upload_id) if saved_hash else []
            )
            different_name_duplicates = [d for d in duplicate_datasets if d.get("dataset_name") != dataset_name]
            if different_name_duplicates:
                # Warn about duplicate content (UX polish - clinicians may intentionally duplicate)
                dup_names = [d.get("dataset_name") for d in different_name_duplicates if d.get("dataset_name")]
                duplicate_warning = (
                    f"⚠️ Warning: This file has the same content as existing dataset(s): "
                    f"{', '.join(str(n) for n in dup_names if n)}. "
                    f"Upload will proceed, but you may want to use the existing dataset instead."
                )
                logger.info("Duplicate content detected for upload %s: %d dataset(s)", upload_id, len(dup_names))

            _notify_progress(progress_cb, 100, f"✅ Upload complete! Dataset '{dataset_name}' ready to use.")

            # Phase 1: Include duplicate warning in success message (if any)
            success_message = f"Upload successful: {upload_id}"
            if duplicate_warning:
                success_message = f"{duplicate_warning} {success_message}"

            return True, success_message, upload_id

        except (UploadError, SecurityError, OSError, RuntimeError, ValueError, pl.exceptions.PolarsError) as e:
            if raise_staging_errors and isinstance(e, StagingError):
                logger.warning("save_upload_unreadable_file", error_type=type(e).__name__)
                raise
            logger.error("save_upload_failed", error_type=type(e).__name__)
            _notify_progress(progress_cb, 100, f"Error: {_upload_failure_reason(e)}")
            return False, f"Error saving upload: {_upload_failure_reason(e)}", None

    def _stored_cohort_schema(self, upload_id: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
        """Typed schema of the stored active version, read the way new uploads are typed.

        Returns None only when the dataset has no stored data at all (nothing to drift from).

        Raises:
            OSError, StagingError, polars.exceptions.PolarsError: stored data exists but is unreadable.
        """
        if metadata.get(STORAGE_LAYOUT_KEY) == STORAGE_LAYOUT:
            schema = pl.read_parquet_schema(metadata[COHORT_PARQUET_KEY])
        else:
            source = _legacy_source_path(self, upload_id, metadata)
            if source is None:
                return None
            schema = dict(_stage_legacy_file(source).schema)
        return {"columns": [(col, str(dtype)) for col, dtype in schema.items()]}

    def _check_overwrite_drift(self, upload_id: str, new_df: pl.DataFrame, metadata: dict[str, Any]) -> str | None:
        """Message blocking the overwrite on breaking schema drift, or None when it may proceed.

        Compares the new staged frame with the stored active version's typed schema. A stored
        version that exists but cannot be read refuses the overwrite (never skips the check).
        """
        log = logger.bind(upload_id=upload_id)
        existing = self.get_upload_metadata(upload_id) or {}
        try:
            old_schema = self._stored_cohort_schema(upload_id, existing)
        except (OSError, StagingError, pl.exceptions.PolarsError) as e:
            log.error("drift_check_stored_unreadable", error_type=type(e).__name__)
            return (
                "The stored version of this dataset could not be read to compare column types "
                f"({type(e).__name__}). Overwrite refused; delete the dataset and upload it again."
            )
        if old_schema is None:
            log.warning("drift_check_no_stored_data")
            return None
        new_schema = {"columns": [(col, str(dtype)) for col, dtype in new_df.schema.items()]}
        drift_result = classify_schema_drift(old_schema, new_schema)
        allowed, message, warnings = apply_schema_drift_policy(
            drift_result, override=metadata.get("schema_drift_override", False)
        )
        log.info("drift_check_done", drift_type=drift_result["drift_type"], allowed=allowed)
        if warnings:
            log.warning("drift_check_warnings", count=len(warnings))
        return None if allowed else message

    def stored_dataset(self, upload_id: str, metadata: dict[str, Any] | None = None) -> StoredDataset:
        """
        The typed data of a dataset's active version (the one source of truth for every reader).

        Uploads made before typed storage are converted on first call (``_convert_legacy_upload``).

        Args:
            upload_id: Upload identifier
            metadata: Metadata to use instead of the stored file (the upload probe, before commit)

        Raises:
            FileNotFoundError: no metadata, or the stored data is missing.
            StagingError: a legacy file cannot be staged.
        """
        if metadata is None:
            metadata = self.get_upload_metadata(upload_id)
            if metadata is None:
                raise FileNotFoundError(f"Upload {upload_id} not found")
        if metadata.get(STORAGE_LAYOUT_KEY) != STORAGE_LAYOUT:
            with _migration_lock(upload_id):
                current = self.get_upload_metadata(upload_id) or metadata
                if current.get(STORAGE_LAYOUT_KEY) != STORAGE_LAYOUT:
                    return _convert_legacy_upload(self, upload_id, current)
                metadata = current
        cohort_path = Path(metadata[COHORT_PARQUET_KEY])
        if not cohort_path.is_file():
            raise FileNotFoundError(f"Stored data missing for upload {upload_id}")
        table_paths = {name: Path(p) for name, p in (metadata.get("parquet_paths") or {}).items()}
        return StoredDataset(metadata=metadata, cohort_path=cohort_path, table_paths=table_paths)

    def get_active_version(self, upload_id: str) -> dict[str, Any] | None:
        """
        Get the currently active version entry for a dataset (Phase 7).

        Returns the version entry from version_history that has is_active=True.
        Used by query execution to determine which schema version to use.

        Defensive: Warns if multiple active versions detected (invariant violation).

        Args:
            upload_id: Upload identifier

        Returns:
            Active version entry dict, or None if dataset not found or no active version
        """
        metadata = self.get_upload_metadata(upload_id)
        if not metadata:
            return None

        version_history = metadata.get("version_history", [])
        if not version_history:
            return None

        # Find all active versions (should be exactly one)
        active_versions = [entry for entry in version_history if entry.get("is_active") is True]

        # Defensive: check for invariant violations
        if len(active_versions) == 0:
            logger.warning(f"No active version found for dataset {upload_id} (invariant violation)")
            return None
        elif len(active_versions) > 1:
            logger.error(
                f"Multiple active versions found for dataset {upload_id} "
                f"(invariant violation): {len(active_versions)} active versions. "
                f"Returning first one, but metadata is corrupted."
            )
            # Return first one but log the corruption
            result = active_versions[0]
            return dict(result) if isinstance(result, dict) else {}
        else:
            # Exactly one active version (expected case)
            result = active_versions[0]
            return dict(result) if isinstance(result, dict) else {}

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
            result = json.load(f)
            return dict(result) if isinstance(result, dict) else None

    def get_upload_data(self, upload_id: str, lazy: bool = True) -> pl.LazyFrame | None:
        """
        The dataset's typed cohort as a LazyFrame (the same Parquet the semantic layer reads).

        Legacy uploads are converted to typed Parquet on first call (see ``stored_dataset``).

        Args:
            upload_id: Upload identifier (immutable storage key)
            lazy: Backward-compatibility flag (ignored). Result is always LazyFrame.

        Returns:
            LazyFrame, or None if the upload or its stored data is not found
        """
        try:
            return self.stored_dataset(upload_id).scan_cohort()
        except FileNotFoundError:
            logger.warning("upload_data_not_found", upload_id=upload_id)
            return None

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
        Delete an uploaded dataset's files: raw data, tables, metadata, overlays, Parquet
        exports and DuckDB tables (see storage.dataset_deletion). API sessions are removed by
        DELETE /api/datasets/{id}.

        Args:
            upload_id: Upload identifier

        Returns:
            Tuple of (success, message)
        """
        from clinical_analytics.storage.dataset_deletion import (
            DatasetDeletionError,
            InvalidDatasetIdError,
            purge_dataset_files,
        )

        try:
            report = purge_dataset_files(upload_id, self.upload_dir)
        except (InvalidDatasetIdError, DatasetDeletionError) as e:
            return False, f"Error deleting upload: {e}"

        if not report.found_anything:
            return False, f"Upload {upload_id} not found"
        return True, (
            f"Deleted {len(report.removed_paths)} files and {len(report.dropped_tables)} tables for {upload_id}"
        )

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

        def mutate(current: dict[str, Any]) -> bool:
            current.update(metadata_updates)
            return True

        try:
            _update_metadata(metadata_path, mutate)
        except (RuntimeError, OSError, ValueError) as e:
            return False, f"Error updating metadata: {type(e).__name__}"
        return True, "Metadata updated successfully"

    def add_documentation_to_upload(
        self,
        upload_id: str,
        pdf_path: Path | bytes,
        re_infer_schema: bool = True,
    ) -> tuple[bool, str]:
        """
        Add documentation to existing upload and optionally re-run schema inference.

        Extracts doc_context from PDF and updates metadata. If re_infer_schema=True,
        re-runs schema inference with doc_context to enhance variable_types.

        Args:
            upload_id: Existing upload ID
            pdf_path: Path to PDF file or PDF bytes
            re_infer_schema: If True, re-run schema inference with doc_context

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

            # Extract doc_context from PDF
            import tempfile

            from clinical_analytics.core.doc_parser import extract_context_from_docs

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Handle both Path and bytes
                if isinstance(pdf_path, bytes):
                    # Save bytes to temp file
                    pdf_temp_path = temp_path / "documentation.pdf"
                    pdf_temp_path.write_bytes(pdf_path)
                    doc_paths = [pdf_temp_path]
                else:
                    # Path provided
                    if not pdf_path.exists():
                        return False, f"PDF file not found: {pdf_path}"
                    doc_paths = [pdf_path]

                # Extract text context from PDF
                doc_context = extract_context_from_docs(doc_paths)
                if not doc_context:
                    return False, "Failed to extract text from PDF (file may be empty or corrupted)"

            # Update metadata with doc_context
            metadata["doc_context"] = doc_context
            logger.info(f"Extracted doc_context ({len(doc_context)} chars) from PDF")

            # Optionally re-run schema inference with doc_context, on the stored typed cohort
            # (the same Parquet queries read; legacy uploads are converted first).
            if re_infer_schema:
                main_df = self.stored_dataset(upload_id, metadata).scan_cohort().collect()

                if main_df.width:
                    # Re-run schema inference with doc_context
                    from clinical_analytics.core.schema_inference import SchemaInferenceEngine

                    engine = SchemaInferenceEngine()
                    inferred_schema = engine.infer_schema(main_df, doc_context=doc_context)

                    # Update variable_types with enhanced schema
                    if inferred_schema.dictionary_metadata:
                        # Convert DictionaryMetadata to variable_types format
                        # This is a simplified version - full integration would use to_dataset_config()
                        if "variable_types" not in metadata:
                            metadata["variable_types"] = {}

                        # Update variable_types with codebooks and descriptions
                        for col_name, codebook in inferred_schema.dictionary_metadata.codebooks.items():
                            if col_name in metadata["variable_types"]:
                                if "codebook" not in metadata["variable_types"][col_name]:
                                    metadata["variable_types"][col_name]["codebook"] = codebook

                        for col_name, description in inferred_schema.dictionary_metadata.column_descriptions.items():
                            if col_name in metadata["variable_types"]:
                                if "description" not in metadata["variable_types"][col_name]:
                                    metadata["variable_types"][col_name]["description"] = description

                    logger.info("Re-ran schema inference with doc_context")

            doc_updates = {k: metadata[k] for k in ("doc_context", "variable_types") if k in metadata}

            def mutate(current: dict[str, Any]) -> bool:
                current.update(doc_updates)
                return True

            _update_metadata(metadata_path, mutate)
            return True, f"Documentation added successfully ({len(doc_context)} chars extracted)"

        except (RuntimeError, OSError, ValueError, UploadError, pl.exceptions.PolarsError) as e:
            logger.error("add_documentation_failed", upload_id=upload_id, error_type=type(e).__name__)
            return False, f"Error adding documentation: {type(e).__name__}"

    def save_zip_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
        metadata: dict[str, Any],
        progress_callback: Callable[[int, int, str, dict], None] | None = None,
        overwrite: bool = False,
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

        # Phase 9: Handle overwrite for ZIP uploads
        dataset_name = metadata.get("dataset_name")
        existing_upload_id = None
        existing_version_history = None
        base_active_version: str | None = None
        if overwrite and dataset_name:
            # Check if dataset with this name already exists
            existing_datasets = self.list_uploads()
            for dataset in existing_datasets:
                if dataset.get("dataset_name") == dataset_name:
                    existing_upload_id = dataset.get("upload_id")
                    if existing_upload_id and isinstance(existing_upload_id, str):
                        existing_meta = self.get_upload_metadata(existing_upload_id)
                    else:
                        existing_meta = None
                    if existing_meta:
                        existing_version_history = existing_meta.get("version_history", [])
                        base_active_version = active_version_id(existing_meta)
                        # Phase 4.2: Legacy dataset migration during overwrite
                        if len(existing_version_history) == 0:
                            # Legacy dataset: create synthetic version_history from top-level metadata
                            legacy_version = existing_meta.get("dataset_version")
                            legacy_created_at = existing_meta.get("created_at") or existing_meta.get("upload_timestamp")
                            legacy_tables = {}

                            # Reconstruct tables from parquet_paths or duckdb_tables if available
                            parquet_paths = existing_meta.get("parquet_paths", {})
                            if parquet_paths:
                                # Convert parquet_paths dict to canonical tables structure
                                for table_name, parquet_path in parquet_paths.items():
                                    legacy_tables[table_name] = {
                                        "parquet_path": parquet_path,
                                        "duckdb_table": existing_meta.get("duckdb_tables", {}).get(
                                            table_name, table_name
                                        ),
                                        "row_count": existing_meta.get("table_counts", {}).get(table_name, 0),
                                        "column_count": existing_meta.get("column_count", 0),
                                        "schema_fingerprint": existing_meta.get("schema_fingerprint", ""),
                                    }
                            else:
                                # Fallback: create minimal table entry
                                table_name = (
                                    existing_meta.get("tables", ["table_0"])[0]
                                    if existing_meta.get("tables")
                                    else "table_0"
                                )
                                legacy_tables[table_name] = {
                                    "parquet_path": "",
                                    "duckdb_table": table_name,
                                    "row_count": existing_meta.get("row_count", 0),
                                    "column_count": existing_meta.get("column_count", 0),
                                    "schema_fingerprint": "",
                                }

                            synthetic_version_entry = {
                                "version": legacy_version or "legacy",
                                "created_at": (
                                    legacy_created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
                                ),
                                "event_id": uuid.uuid4().hex,
                                "event_type": "upload",
                                "is_active": False,
                                "tables": legacy_tables,
                                "source_filename": existing_meta.get("original_filename", ""),
                            }

                            existing_version_history = [synthetic_version_entry]
                            logger.info("Migrated legacy dataset to version_history (1 version)")  # noqa: F823
                    break

        # Generate upload ID (or use existing if overwriting)
        if existing_upload_id and overwrite:
            upload_id = existing_upload_id
        else:
            upload_id = self.generate_upload_id(original_filename)

        try:
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

            with MultiTableHandler(tables_dict) as handler:
                relationships = handler.detect_relationships()

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
                    "source_file_sha256": hashlib.sha256(file_bytes).hexdigest(),
                    "file_format": "zip_multi_table",
                }
            )

            # Phase 1: Merge table_metadata (includes doc_files) into metadata
            # This ensures doc_files are available for save_table_list() to extract doc_context
            metadata.update(table_metadata)

            # Phase 9: Pass overwrite metadata to save_table_list
            if overwrite and existing_version_history is not None:
                metadata["_existing_version_history"] = existing_version_history
                metadata["_is_overwrite"] = True
            # The commit re-reads history and events under the lock and checks this version is still active.
            metadata[BASE_ACTIVE_VERSION_KEY] = base_active_version if existing_upload_id and overwrite else None

            # Save using unified save_table_list() function
            if progress_callback:
                progress_callback(step_num + 2, 10, "Saving tables and building unified cohort...", {})

            # Create adapter for progress callback (save_table_list uses different signature)
            def progress_adapter(progress: int, message: str) -> None:
                if progress_callback:
                    # Map to ZIP upload progress format
                    progress_callback(step_num + 2 + progress // 10, 10, message, {})

            success, message = save_table_list(
                self, tables_list, upload_id, metadata, progress_adapter, raw_bytes=file_bytes
            )

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
            notes = describe_cohort_build((self.get_upload_metadata(upload_id) or {}).get(COHORT_BUILD_KEY))
            message = f"Multi-table upload successful: {len(tables_list)} tables"
            return True, " ".join([message, *notes]), upload_id

        except (
            UploadError,
            SecurityError,
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            pl.exceptions.PolarsError,
        ) as e:
            # Type only, in logs and in the reply: exception text can quote cell values.
            logger.error("zip_upload_failed", error_type=type(e).__name__)
            return False, f"Error processing ZIP upload: {_upload_failure_reason(e)}", None

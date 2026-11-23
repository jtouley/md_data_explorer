# SPDX-License-Identifier: MIT
"""Dataset API routes for Electron UI.

Endpoints:
- GET  /api/datasets                      - List available datasets
- GET  /api/datasets/{dataset_id}         - Get dataset metadata
- GET  /api/datasets/{dataset_id}/preview - Get sample rows
- GET  /api/datasets/{dataset_id}/quality - Data-quality report for the active version
- POST /api/datasets/upload               - Upload a dataset file
- DELETE /api/datasets/{dataset_id}       - Delete a dataset and everything derived from it
"""

import asyncio
from pathlib import Path
from typing import Annotated, Any, Literal

import polars as pl
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from clinical_analytics.api.db.database import get_db
from clinical_analytics.api.dependencies import (
    dataset_deletion_guard,
    dataset_files_lock,
    invalidate_semantic_layer_cache,
    reject_if_deleting,
)
from clinical_analytics.api.models import database as db_models
from clinical_analytics.api.routes.queries import forget_query_service
from clinical_analytics.contracts.builder import logical_type
from clinical_analytics.contracts.config import POPULATION_ESTIMATES_FORM_VALUE, POPULATION_ESTIMATES_NOT_SAVED
from clinical_analytics.contracts.declarations import CONFIRMATION_TOKEN_CHARS, Declarations, FileDeclarationStore
from clinical_analytics.contracts.errors import (
    ContractDamagedError,
    ContractError,
    ContractStoreError,
    InvalidDeclarationKeyError,
)
from clinical_analytics.contracts.quality import QualityCheckResult, QualityNotRun, QualityReport
from clinical_analytics.contracts.semantics import DatasetSemantics
from clinical_analytics.contracts.service import check_upload
from clinical_analytics.contracts.store import latest_report
from clinical_analytics.core.log_redaction import error_fields, text_fingerprint
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
from clinical_analytics.storage.dataset_deletion import (
    DatasetDeletionError,
    DatasetStoreBusyError,
    InvalidDatasetIdError,
    purge_dataset_files,
    validate_dataset_id,
)
from clinical_analytics.ui.storage.user_datasets import (
    USER_FILE_ERRORS,
    UploadSecurityValidator,
    UserDatasetStorage,
    active_version_id,
    duplicate_name_message,
)

logger = structlog.get_logger()

router = APIRouter()


# ============================================================================
# Pydantic Schemas
# ============================================================================


class DatasetSummary(BaseModel):
    """Summary information for a dataset in listing."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., description="Display name")
    source: Literal["uploaded", "builtin"] = Field(..., description="Dataset source type")
    table_count: int = Field(1, description="Number of tables")
    row_count: int = Field(0, description="Total row count")
    created_at: str | None = Field(None, description="Creation timestamp")


class DatasetListResponse(BaseModel):
    """Response containing list of datasets."""

    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetSummary] = Field(..., description="List of datasets")
    total: int = Field(..., description="Total count")


class TableInfo(BaseModel):
    """Information about a single table in a dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Table name")
    row_count: int = Field(0, description="Row count")
    columns: list[str] = Field(default_factory=list, description="Column names")


class DatasetDetail(BaseModel):
    """Detailed information for a single dataset."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dataset_id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., description="Display name")
    source: Literal["uploaded", "builtin"] = Field(..., description="Dataset source type")
    tables: list[TableInfo] = Field(default_factory=list, description="Tables in dataset")
    column_schema: dict[str, str] = Field(
        default_factory=dict, serialization_alias="schema", description="Column to dtype mapping"
    )
    created_at: str | None = Field(None, description="Creation timestamp")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class DatasetPreview(BaseModel):
    """Preview of dataset rows."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="Dataset identifier")
    rows: list[dict[str, Any]] = Field(..., description="Sample rows as dicts")
    columns: list[str] = Field(..., description="Column names")
    total_rows: int = Field(..., description="Total row count in dataset")


# ============================================================================
# Endpoints
# ============================================================================


# Routes that read files or DuckDB are plain ``def``: FastAPI runs them in its threadpool, so a
# slow load (or a delete holding the store lock) cannot stall the event loop for other requests.
@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    """List all available datasets.

    Returns:
        List of dataset summaries for both uploaded and built-in datasets.
    """
    log = logger.bind()
    log.info("datasets_list_requested")

    datasets: list[DatasetSummary] = []

    # Get uploaded datasets
    try:
        uploads = UploadedDatasetFactory.list_available_uploads()
        for upload in uploads:
            datasets.append(
                DatasetSummary(
                    dataset_id=upload.get("upload_id", ""),
                    name=upload.get("dataset_name", upload.get("upload_id", "Unknown")),
                    source="uploaded",
                    table_count=len(upload.get("tables", [])) or 1,
                    row_count=upload.get("row_count", 0),
                    created_at=upload.get("upload_timestamp"),
                )
            )
    except Exception as e:
        log.warning("datasets_list_uploads_failed", **error_fields(e))

    # Future: Add built-in datasets from registry (deferred to Phase 1.1)

    log.info("datasets_list_completed", count=len(datasets))
    return DatasetListResponse(datasets=datasets, total=len(datasets))


@router.get("/datasets/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: str) -> DatasetDetail:
    """Get detailed information about a specific dataset.

    Args:
        dataset_id: Dataset identifier

    Returns:
        Full dataset metadata including schema and tables.

    Raises:
        HTTPException: 404 if dataset not found
    """
    log = logger.bind(dataset_id=dataset_id)
    log.info("dataset_get_requested")
    reject_if_deleting(dataset_id)

    try:
        # Try to load as uploaded dataset
        dataset = UploadedDatasetFactory.create_dataset(dataset_id)
        info = dataset.get_info()

        # Types as stored (the typed Parquet the queries read), as contract logical types
        columns = info.get("columns", [])
        schema = _stored_column_types(dataset, columns)

        # Build table info
        tables = [
            TableInfo(
                name="unified",
                row_count=info.get("row_count", 0),
                columns=columns,
            )
        ]

        # Dataset names can identify a study, so only the id is logged.
        log.info("dataset_get_completed")
        return DatasetDetail(
            dataset_id=dataset_id,
            name=info.get("name", dataset_id),
            source="uploaded",
            tables=tables,
            column_schema=schema,
            created_at=info.get("uploaded_at"),
            metadata=info,
        )

    except ValueError as e:
        log.warning("dataset_get_not_found", **error_fields(e))
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}") from e

    except Exception as e:
        log.error("dataset_get_failed", **error_fields(e))
        raise HTTPException(
            status_code=500, detail=f"Could not load this dataset ({type(e).__name__}). Try reloading it."
        ) from e


def _stored_column_types(dataset: Any, columns: list[str]) -> dict[str, str]:
    """Each column's stored type ("integer", "number", "string", "date", ...), read from the
    stored data's schema (no rows are read). A listed column missing from the store is
    "missing", so a stale column list never passes for a typed one. When the stored data is
    gone or unreadable every column is "missing": the details still open (the metadata is
    intact), and the failure is logged, never a 500."""
    from clinical_analytics.datasets.uploaded.definition import SemanticLayerBuildError

    try:
        dataset.load()
        stored = dataset.data.collect_schema()
    except (FileNotFoundError, SemanticLayerBuildError) as e:
        logger.warning("dataset_stored_types_unreadable", error_type=type(e).__name__)
        return dict.fromkeys(columns, "missing")
    names = columns or list(stored)
    return {col: logical_type(stored[col]) if col in stored else "missing" for col in names}


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    dataset_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Number of rows to return"),
) -> DatasetPreview:
    """Get a preview of dataset rows.

    Args:
        dataset_id: Dataset identifier
        limit: Maximum number of rows to return (1-100)

    Returns:
        Sample rows from the dataset.

    Raises:
        HTTPException: 404 if dataset not found
    """
    log = logger.bind(dataset_id=dataset_id, limit=limit)
    log.info("dataset_preview_requested")
    reject_if_deleting(dataset_id)

    try:
        # Load dataset
        storage = UserDatasetStorage()
        metadata = storage.get_upload_metadata(dataset_id)

        if not metadata:
            raise ValueError(f"Upload {dataset_id} not found")

        data = storage.get_upload_data(dataset_id, lazy=True)
        if data is None:
            raise ValueError(f"Dataset data not found for {dataset_id}")

        lf = data.lazy() if isinstance(data, pl.DataFrame) else data
        total_rows = lf.select(pl.len().alias("n")).collect()["n"][0]
        preview_df = lf.limit(limit).collect()

        rows = preview_df.to_dicts()
        columns = preview_df.columns

        log.info("dataset_preview_completed", row_count=len(rows), total=total_rows)
        return DatasetPreview(
            dataset_id=dataset_id,
            rows=rows,
            columns=columns,
            total_rows=total_rows,
        )

    except ValueError as e:
        log.warning("dataset_preview_not_found", **error_fields(e))
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}") from e

    except Exception as e:
        log.error("dataset_preview_failed", **error_fields(e))
        raise HTTPException(
            status_code=500, detail=f"Could not show a preview of this dataset ({type(e).__name__}). Try reloading it."
        ) from e


NO_CHECK_RECORDED = "no data check was recorded for this upload"
REPORT_UNAVAILABLE = "this upload's report could not be read"


def _upload_quality(storage: UserDatasetStorage, upload_id: str) -> QualityReport | QualityNotRun:
    """The result of this upload's own check, for the upload response.

    Only the report whose run time the upload recorded is returned: an older report on disk for
    the same version is not this upload's result, so a failed run shows as not run.
    """
    metadata = storage.get_upload_metadata(upload_id) or {}
    recorded = metadata.get("quality")
    version = metadata.get("dataset_version")
    if not isinstance(recorded, dict) or not isinstance(version, str) or not version:
        return QualityNotRun(reason=NO_CHECK_RECORDED)
    if recorded.get("status") == "not_run":
        return QualityNotRun(reason=str(recorded.get("reason") or NO_CHECK_RECORDED))
    try:
        report = latest_report(storage.upload_dir.parent, upload_id, version)
    except (ContractError, ValueError) as e:
        logger.error("dataset_upload_quality_unreadable", upload_id=upload_id, **error_fields(e))
        return QualityNotRun(reason=REPORT_UNAVAILABLE)
    if report is None or report.run_ts != recorded.get("run_ts"):
        logger.error("dataset_upload_quality_missing", upload_id=upload_id)
        return QualityNotRun(reason=REPORT_UNAVAILABLE)
    return report


def _backfill_quality(storage: UserDatasetStorage, dataset_id: str, version: str) -> QualityReport:
    """Build the contract and report under the dataset's files lock (see ``get_dataset_quality``).

    Raises:
        HTTPException: 409 if a delete started meanwhile, 404 if the dataset is gone.
    """
    log = logger.bind(dataset_id=dataset_id)
    data_dir = storage.upload_dir.parent
    with dataset_files_lock(dataset_id):
        reject_if_deleting(dataset_id)
        metadata = storage.get_upload_metadata(dataset_id)
        if not metadata or metadata.get("dataset_version") != version:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        report = latest_report(data_dir, dataset_id, version)  # another request may have backfilled
        if report is not None:
            return report
        log.info("dataset_quality_backfill")
        data = storage.get_upload_data(dataset_id, lazy=True)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Dataset data not found: {dataset_id}")
        return check_upload(
            data_dir,
            data.collect(),
            upload_id=dataset_id,
            dataset_version=version,
            metadata=metadata,
            overlay_dir=storage.metadata_dir,
        )


@router.get("/datasets/{dataset_id}/quality", response_model=QualityReport)
def get_dataset_quality(dataset_id: str) -> QualityReport:
    """Data-quality report for the dataset's active version (latest run).

    Uploads made before contracts existed (or whose upload check failed) have none: the contract
    and report are generated on first request (the contract is written once per version, so
    repeating the request is safe). The backfill holds the dataset's files lock, which a delete
    also takes around its purge, and re-checks the deleting mark and metadata under it, so a
    concurrent delete cannot leave contract or report files behind.

    Raises:
        HTTPException: 400 bad id, 404 unknown dataset, 409 while deleting, 500 if the check fails.
    """
    try:
        validate_dataset_id(dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    log = logger.bind(dataset_id=dataset_id)
    reject_if_deleting(dataset_id)

    storage = UserDatasetStorage()
    metadata = storage.get_upload_metadata(dataset_id)
    version = (metadata or {}).get("dataset_version")
    if not metadata or not version:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    data_dir = storage.upload_dir.parent
    try:
        report = latest_report(data_dir, dataset_id, version)
        if report is None:
            report = _backfill_quality(storage, dataset_id, version)
    except ContractDamagedError as e:
        log.error("dataset_quality_contract_damaged", **error_fields(e))
        raise HTTPException(status_code=500, detail="The saved data contract for this cohort is damaged.") from e
    except (ContractError, OSError, pl.exceptions.PolarsError) as e:
        log.error("dataset_quality_failed", **error_fields(e))
        raise HTTPException(
            status_code=500, detail=f"Could not check this dataset's quality ({type(e).__name__})."
        ) from e
    log.info("dataset_quality_completed", failed=report.failed, warnings=report.warnings)
    return report.model_copy(
        update={"facts": _facts(_contract_semantics(storage, dataset_id), _upload_declarations(storage, dataset_id))}
    )


class DatasetSemanticsResponse(BaseModel):
    """The active version's semantics (ADR017) with the user's confirmations applied."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_version: str
    semantics: dict[str, Any] = Field(..., description="The contract's semantics section (DatasetSemantics)")
    facts: list[str] = Field(default_factory=list, description="The key facts in clinical words")


@router.get("/datasets/{dataset_id}/semantics", response_model=DatasetSemanticsResponse)
def get_dataset_semantics(dataset_id: str) -> DatasetSemanticsResponse:
    """What the active version's file means, read from its data contract (``semantics_for``; a
    version stored before ADR017 gets the same inference pass over its stored data, read-only).

    Privacy: the response holds patient counts and evidence sentences. They are shown on this
    machine as they are (ADR015); anything that exports or packages them (a report, a model
    packet) must pass them through ``core.export_boundary`` first.

    Raises:
        HTTPException: 400 bad id, 404 unknown dataset or no semantics yet, 409 while deleting,
            500 when the contract or declarations can't be read.
    """
    from clinical_analytics.contracts.semantics import describe

    try:
        validate_dataset_id(dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    reject_if_deleting(dataset_id)
    storage = UserDatasetStorage()
    version = active_version_id(storage.get_upload_metadata(dataset_id))
    semantics = _contract_semantics(storage, dataset_id, declared=True)
    if version is None or semantics is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return DatasetSemanticsResponse(
        dataset_id=dataset_id,
        dataset_version=version,
        semantics=semantics.model_dump(mode="json"),
        facts=describe(semantics),
    )


# ============================================================================
# Upload (Phase 8)
# ============================================================================


class OneRowPerPatientQuestion(BaseModel):
    """ "Is each row one patient?", asked once after upload (``core.patient_rows``)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pending"] = Field(
        "pending", description="Not answered yet: until it is, answers count rows, not patients"
    )
    confirmation: str = Field(
        ...,
        description=(
            "Send back with the answer to POST /api/datasets/{upload_id}/one-row-per-patient. "
            "Bound to this dataset version and row count: refused after a re-upload"
        ),
    )


class DatasetUploadResponse(BaseModel):
    """Response after a successful dataset upload."""

    model_config = ConfigDict(extra="forbid")

    upload_id: str = Field(..., description="Created upload identifier")
    dataset_name: str = Field(..., description="Display name for the dataset")
    status: Literal["ready", "needs_review", "failed"] = Field(
        ...,
        description=(
            "ready: stored and every error-level data check passed. needs_review: stored and "
            "queryable, but an error-level check failed (named in failed_checks and message)"
        ),
    )
    message: str = Field(
        "", description="Human-readable status message; for needs_review, the failed check and what to do"
    )
    failed_checks: list[str] = Field(
        default_factory=list, description="Ids of the error-level data checks that failed (needs_review)"
    )
    quality: QualityReport | QualityNotRun | None = Field(
        None, description="This upload's data-quality report, or status 'not_run' with a reason"
    )
    facts: list[str] = Field(
        default_factory=list,
        description=(
            "What the file holds, from its data contract's semantics, in clinical words: what one "
            "line is and how many patients, dates read, values that mean not recorded, open questions"
        ),
    )
    type_notes: list[str] = Field(
        default_factory=list,
        description="Columns kept as text or partly unreadable, and units rows, each with the reason",
    )
    cohort_notes: list[str] = Field(
        default_factory=list,
        description="Multi-table uploads: tables, rows and features left out of the cohort, and why",
    )
    population_estimates: bool = Field(
        default=False,
        description="The uploader declared the file public population estimates with no patient-level data",
    )
    population_estimates_note: str | None = Field(
        default=None,
        description="Declared, but the rows don't look like estimates: the full patient rules still apply",
    )
    sheet_note: str = Field(
        default="",
        description="Excel uploads: the sheet that was read and the other sheets with data that were not",
    )
    unit_note: str = Field(
        default="",
        description=(
            "Set when no column identifies a patient: what tells the rows apart, that answers count "
            "rows, not patients, and how to get patient counts. Empty when patient_id names patients, "
            "or the user confirmed that each row is one patient"
        ),
    )
    one_row_per_patient: OneRowPerPatientQuestion | None = Field(
        None,
        description=(
            "Set when the user should be asked whether each row is one patient: a column unique on "
            "every row (or nothing) tells rows apart, none is named as a patient ID, and no ID-like "
            "column repeats. Null when not asked (IDs name patients, IDs repeat) or already answered yes"
        ),
    )


def population_estimates_mark(value: str | None) -> bool:
    """Whether the upload form declares public population estimates (ADR014 §7).

    Only the exact acknowledgement value counts. A generic truthy value ("true", "on", "1") is
    rejected rather than read as yes: a checkbox default or a script flag must never switch a
    patient table to estimate answers by accident. Absent or empty means not declared.

    Raises:
        HTTPException: 400 for any other value.
    """
    if value is None or value == "":
        return False
    if value == POPULATION_ESTIMATES_FORM_VALUE:
        return True
    raise HTTPException(
        status_code=400,
        detail=(
            "population_estimates must be absent, or exactly "
            f"'{POPULATION_ESTIMATES_FORM_VALUE}' to declare public population estimates."
        ),
    )


def _declare_population_estimates(storage: UserDatasetStorage, upload_id: str) -> str | None:
    """Store the uploader's declaration that ``upload_id`` holds public population estimates, for
    the version just stored (ADR014 §7). Returns a note when the stored rows don't look like
    estimates (the declaration is kept, but nothing is relaxed for such a table).

    Raises:
        HTTPException 500 when it can't be saved: an upload whose declaration was lost would be
        answered under the wrong rules.
    """
    from clinical_analytics.contracts.config import POPULATION_ESTIMATES_NOT_DETECTED, POPULATION_ESTIMATES_STATEMENT
    from clinical_analytics.contracts.declarations import FileDeclarationStore, PopulationEstimatesDeclaration
    from clinical_analytics.contracts.errors import ContractStoreError
    from clinical_analytics.core.data_home import uploads_dir
    from clinical_analytics.core.estimates import detect_frame
    from clinical_analytics.ui.storage.user_datasets import active_version_id

    metadata = storage.get_upload_metadata(upload_id)
    version = active_version_id(metadata)
    if metadata is None or version is None:
        raise HTTPException(status_code=500, detail=POPULATION_ESTIMATES_NOT_SAVED)
    declaration = PopulationEstimatesDeclaration(statement=POPULATION_ESTIMATES_STATEMENT, dataset_version=version)
    try:
        FileDeclarationStore(uploads_dir().parent).update(
            upload_id, lambda current: current.with_population_estimates(declaration)
        )
    except ContractStoreError as e:
        logger.error("population_estimates_not_saved", upload_id=upload_id, error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=POPULATION_ESTIMATES_NOT_SAVED) from e
    logger.info("population_estimates_declared", upload_id=upload_id, dataset_version=version)
    frame = storage.stored_dataset(upload_id, metadata).scan_cohort().collect()
    if detect_frame(frame) is None:
        logger.warning("population_estimates_not_detected", upload_id=upload_id)
        return POPULATION_ESTIMATES_NOT_DETECTED
    return None


def _upload_type_notes(storage: UserDatasetStorage, upload_id: str) -> list[str]:
    """How staging typed this upload's columns, for the user, with each count as it is."""
    from clinical_analytics.storage.staging import describe_column_types
    from clinical_analytics.ui.storage.user_datasets import COLUMN_TYPES_KEY, COLUMN_UNITS_KEY

    metadata = storage.get_upload_metadata(upload_id) or {}
    return describe_column_types(metadata.get(COLUMN_TYPES_KEY) or [], metadata.get(COLUMN_UNITS_KEY) or {})


def _upload_declarations(storage: UserDatasetStorage, upload_id: str) -> Declarations:
    """The upload's stored declarations; none when they can't be read (logged): the question is
    then asked, and answers count rows, as the question path reads unreadable declarations."""
    try:
        return FileDeclarationStore(storage.upload_dir.parent).load(upload_id)
    except (ContractStoreError, InvalidDeclarationKeyError) as e:
        logger.error("dataset_upload_declarations_unreadable", upload_id=upload_id, **error_fields(e))
        return Declarations()


def _contract_semantics(
    storage: UserDatasetStorage, upload_id: str, *, declared: bool = False
) -> DatasetSemantics | None:
    """The active version's semantics (``contracts.service.version_semantics``: from its contract,
    or for a version stored before ADR017 the same pass over its stored frame). As inferred
    unless ``declared``; None only when the dataset has no active version.

    Raises:
        HTTPException: 500 when the contract or declarations can't be read (never a silent fallback:
            the unit note and "Is each row one patient?" depend on it).
    """
    from clinical_analytics.contracts.declarations import MemoryDeclarationStore
    from clinical_analytics.contracts.service import version_semantics

    metadata = storage.get_upload_metadata(upload_id)
    version = active_version_id(metadata)
    if metadata is None or version is None:
        return None
    try:
        return version_semantics(
            storage.upload_dir.parent,
            upload_id,
            version,
            metadata=metadata,
            stored_frame=lambda: storage.stored_dataset(upload_id, metadata).scan_cohort().collect(),
            declarations=None if declared else MemoryDeclarationStore(),
        )
    except ContractError as e:
        logger.error("dataset_semantics_unreadable", upload_id=upload_id, **error_fields(e))
        raise HTTPException(status_code=500, detail="This cohort's data contract can't be read.") from e


def _facts(semantics: DatasetSemantics | None, declarations: Declarations) -> list[str]:
    """The file's key facts in clinical words, with the user's confirmations applied."""
    from clinical_analytics.contracts.semantics import describe, with_declarations

    return [] if semantics is None else describe(with_declarations(semantics, declarations))


def _upload_unit_note(
    storage: UserDatasetStorage, upload_id: str, semantics: DatasetSemantics | None, declarations: Declarations
) -> str:
    """Rows, not patients: said at upload when the contract names no patient key ("" otherwise,
    and "" once the user confirmed this version's rows are one per patient)."""
    from clinical_analytics.core.identifier_rules import row_key_note
    from clinical_analytics.core.patient_rows import confirmed_patient_rows

    if semantics is None:
        return ""
    metadata = storage.get_upload_metadata(upload_id) or {}
    return "" if confirmed_patient_rows(metadata, semantics, declarations) else row_key_note(semantics)


def _upload_patient_rows_question(
    storage: UserDatasetStorage, upload_id: str, semantics: DatasetSemantics | None, declarations: Declarations
) -> OneRowPerPatientQuestion | None:
    """ "Is each row one patient?" with its token: only when the contract's semantics hold that
    open question, and it is not answered yes yet."""
    from clinical_analytics.core.patient_rows import pending_token

    token = pending_token(storage.get_upload_metadata(upload_id) or {}, semantics, declarations)
    return None if token is None else OneRowPerPatientQuestion(status="pending", confirmation=token)


def _upload_sheet_note(storage: UserDatasetStorage, upload_id: str) -> str:
    """Which sheet of a workbook was read and which were not, for the user ("" when one sheet)."""
    from clinical_analytics.storage.staging import describe_sheets
    from clinical_analytics.ui.storage.user_datasets import EXCEL_SHEETS_KEY

    return describe_sheets((storage.get_upload_metadata(upload_id) or {}).get(EXCEL_SHEETS_KEY))


def _upload_cohort_notes(storage: UserDatasetStorage, upload_id: str) -> list[str]:
    """What the cohort left out, for the user: ZIP files kept as documentation instead of tables,
    then what a multi-table cohort dropped (tables, keyless rows, features)."""
    from clinical_analytics.ui.storage.user_datasets import COHORT_BUILD_KEY, ZIP_NOTES_KEY, describe_cohort_build

    metadata = storage.get_upload_metadata(upload_id) or {}
    return [*metadata.get(ZIP_NOTES_KEY, []), *describe_cohort_build(metadata.get(COHORT_BUILD_KEY))]


class DatasetDeleteResponse(BaseModel):
    """What a dataset delete removed."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="Deleted dataset identifier")
    removed_paths: int = Field(..., description="Files and folders removed from the data home")
    dropped_tables: int = Field(..., description="DuckDB tables dropped")
    deleted_sessions: int = Field(..., description="Sessions (with their messages) deleted")


# .zip: tables plus their documentation (ADR004), the multi-table path; .xpt: SAS transport (NHANES);
# .tsv and .txt: tab-separated text (CDC WONDER exports, with their notes block as documentation).
ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".sav", ".xpt", ".zip"}

# What to do next when an error-level check fails, by rule id suffix (contracts.builder._id_rules).
_NEXT_STEP_BY_RULE_SUFFIX: dict[str, str] = {
    "_no_duplicates": (
        "Questions that count patients will first ask you to confirm that rows are visits. "
        "If each row should be one patient, fix the repeated IDs and upload the file again."
    ),
    "_not_null": ("Rows without an ID are not counted as patients. Fill in the missing IDs and upload the file again."),
}
_NEXT_STEP_DEFAULT = "Open the cohort details to see the check, then fix the file and upload it again."


def _failed_error_checks(quality: QualityReport | QualityNotRun) -> list[QualityCheckResult]:
    """Error-level checks that failed. A hidden count ("review") is at least 1, so on an
    error-level check it is a failure too (the count itself stays hidden)."""
    if not isinstance(quality, QualityReport):
        return []
    return [c for c in quality.checks if c.severity == "error" and c.status in ("failed", "review")]


def _review_message(failed: list[QualityCheckResult]) -> str:
    """'Uploaded, but a data check failed: patient_id: 127 repeated values. <next step>'."""
    found = "; ".join(c.message for c in failed)
    lead = (
        "Uploaded, but a data check failed" if len(failed) == 1 else f"Uploaded, but {len(failed)} data checks failed"
    )
    steps: list[str] = []
    for check in failed:
        step = next((v for k, v in _NEXT_STEP_BY_RULE_SUFFIX.items() if check.id.endswith(k)), _NEXT_STEP_DEFAULT)
        if step not in steps:
            steps.append(step)
    return f"{lead}: {found}. {' '.join(steps)}"


def _upload_response(
    storage: UserDatasetStorage, upload_id: str, dataset_name: str, message: str, quality: QualityReport | QualityNotRun
) -> DatasetUploadResponse:
    """ "ready" only when no error-level check failed; else "needs_review", naming the checks."""
    failed = _failed_error_checks(quality)
    declarations = _upload_declarations(storage, upload_id)
    semantics = _contract_semantics(storage, upload_id)
    facts = _facts(semantics, declarations)
    return DatasetUploadResponse(
        upload_id=upload_id,
        dataset_name=dataset_name,
        status="needs_review" if failed else "ready",
        message=_review_message(failed) if failed else message,
        failed_checks=[c.id for c in failed],
        quality=quality.model_copy(update={"facts": facts}),
        facts=facts,
        type_notes=_upload_type_notes(storage, upload_id),
        cohort_notes=_upload_cohort_notes(storage, upload_id),
        sheet_note=_upload_sheet_note(storage, upload_id),
        unit_note=_upload_unit_note(storage, upload_id, semantics, declarations),
        one_row_per_patient=_upload_patient_rows_question(storage, upload_id, semantics, declarations),
    )


class OneRowPerPatientAnswer(BaseModel):
    """The user's answer to "Is each row one patient?"."""

    model_config = ConfigDict(extra="forbid")

    confirmation: str = Field(
        ...,
        min_length=CONFIRMATION_TOKEN_CHARS,
        max_length=CONFIRMATION_TOKEN_CHARS,
        pattern=r"^[0-9a-f]+$",
        description="The upload response's one_row_per_patient.confirmation",
    )
    one_row_per_patient: StrictBool = Field(
        ..., description="true: each row is one patient. false: rows are visits, tests or other records"
    )


class OneRowPerPatientResult(BaseModel):
    """What answers on this dataset version count after the answer."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    counts: Literal["patients", "rows"]


@router.post("/datasets/{dataset_id}/one-row-per-patient", response_model=OneRowPerPatientResult)
def answer_one_row_per_patient(dataset_id: str, answer: OneRowPerPatientAnswer) -> OneRowPerPatientResult:
    """Record the answer to "Is each row one patient?" for the dataset's current version.

    Stored as that version's one-row-per-patient grain (``core.patient_rows``), which questions
    and governed SQL both read. Held under the dataset's files lock, so a delete cannot purge
    the declarations while the answer is written.

    Raises:
        HTTPException: 400 for a bad id, 404 when the dataset does not exist, 409 when the
            question is not asked on this version or the token is not this version's (nothing is
            stored), or the dataset is being deleted; 500 when the answer cannot be saved.
    """
    from clinical_analytics.core.patient_rows import PatientRowsError, record_answer

    try:
        validate_dataset_id(dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    log = logger.bind(dataset_id=dataset_id)
    storage = UserDatasetStorage()
    with dataset_files_lock(dataset_id):
        reject_if_deleting(dataset_id)
        metadata = storage.get_upload_metadata(dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        try:
            counts = record_answer(
                dataset_id,
                metadata,
                _contract_semantics(storage, dataset_id),
                FileDeclarationStore(storage.upload_dir.parent),
                confirmation=answer.confirmation,
                one_row_per_patient=answer.one_row_per_patient,
            )
        except PatientRowsError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except (ContractStoreError, InvalidDeclarationKeyError) as e:
            log.error("patient_rows_answer_not_saved", **error_fields(e))
            raise HTTPException(status_code=500, detail="The answer couldn’t be stored. Try again.") from e
    return OneRowPerPatientResult(dataset_id=dataset_id, counts=counts)


class MissingCodeAnswer(BaseModel):
    """The answer to "Does 99 in los_days mean not recorded?" (an open question of the contract)."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(..., min_length=1, max_length=512)
    value: StrictInt | StrictFloat | str = Field(..., description="The value the question asks about, as given")
    not_recorded: StrictBool = Field(..., description="true: the value means not recorded; false: a real value")


class MissingCodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    facts: list[str]


@router.post("/datasets/{dataset_id}/missing-codes", response_model=MissingCodeResult)
def answer_missing_code(dataset_id: str, answer: MissingCodeAnswer) -> MissingCodeResult:
    """Record whether a value the contract asked about means "not recorded", for this version.

    A "yes" makes every answer read the value as missing (the stored data keeps it: answers apply
    the declaration). Only a value the version's contract asks about is accepted.

    Raises:
        HTTPException: 400 bad id, 404 unknown dataset, 409 not asked on this version, 500 when
            the answer can't be stored.
    """
    from clinical_analytics.contracts.declarations import MissingCodeDeclaration
    from clinical_analytics.contracts.semantics import describe

    try:
        validate_dataset_id(dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    storage = UserDatasetStorage()
    with dataset_files_lock(dataset_id):
        reject_if_deleting(dataset_id)
        semantics = _contract_semantics(storage, dataset_id)
        if semantics is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        asked = next(
            (
                q
                for q in semantics.open_questions
                if q.kind == "missing_code" and q.column == answer.column and str(q.value) == str(answer.value)
            ),
            None,
        )
        if asked is None or asked.value is None:
            raise HTTPException(status_code=409, detail="This cohort is not asked about that value.")
        declaration = MissingCodeDeclaration(
            column=answer.column,
            value=asked.value,
            missing=answer.not_recorded,
            dataset_version=semantics.dataset_version,
        )
        try:
            FileDeclarationStore(storage.upload_dir.parent).update(
                dataset_id, lambda current: current.with_missing_code(declaration)
            )
        except (ContractStoreError, InvalidDeclarationKeyError) as e:
            logger.error("missing_code_answer_not_saved", dataset_id=dataset_id, **error_fields(e))
            raise HTTPException(status_code=500, detail="The answer couldn’t be stored. Try again.") from e
    invalidate_semantic_layer_cache(dataset_id)
    forget_query_service(dataset_id)
    declared = _contract_semantics(storage, dataset_id, declared=True)
    return MissingCodeResult(dataset_id=dataset_id, facts=describe(declared) if declared else [])


@router.post("/datasets/sample", response_model=DatasetUploadResponse)
def load_sample_dataset() -> DatasetUploadResponse:
    """Load the bundled sample cohort into upload storage (stranger-ready demo)."""
    from clinical_analytics.datasets.demo import build_demo_metadata, demo_display_name, get_demo_csv_path

    log = logger.bind()
    log.info("dataset_sample_load_requested")

    path = get_demo_csv_path()
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Sample cohort missing: {path}")

    storage = UserDatasetStorage()
    display_name = demo_display_name(storage)
    success, message, upload_id = storage.save_upload(
        file_bytes=path.read_bytes(),
        original_filename=path.name,
        metadata=build_demo_metadata(display_name),
        overwrite=True,
    )
    if not success or upload_id is None:
        log.error("dataset_sample_load_failed", message=message)
        raise HTTPException(status_code=500, detail=message)

    # Reloading the sample overwrites the same upload_id; drop the cached layer like /upload does.
    invalidate_semantic_layer_cache(upload_id)
    log.info("dataset_sample_load_completed", upload_id=upload_id)
    return _upload_response(
        storage, upload_id, display_name, message or "Sample cohort loaded", _upload_quality(storage, upload_id)
    )


@router.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV, Excel, SPSS, SAS transport or ZIP file to upload"),
    dataset_name: str | None = Form(None, description="Optional display name"),
    population_estimates: str | None = Form(
        None,
        description="Public population estimates – no patient-level data: send exactly "
        f"'{POPULATION_ESTIMATES_FORM_VALUE}' to declare that every row is a published estimate "
        "for a group, never a patient (ADR014 §7). For this version only, the table is shown as "
        "published (owner's decision, 2026-11-19: trust the source); cells the source suppressed stay "
        "hidden. Ignored for a file with patient-level structure. Any other value is rejected.",
    ),
) -> DatasetUploadResponse:
    """Upload a dataset file (CSV, Excel .xlsx/.xls, SPSS .sav, SAS transport .xpt, or ZIP).

    A ZIP holds tables (any of those formats, .tsv/.psv, gzipped text) and their documentation
    (.pdf, .md, .txt; ADR004); several tables become one cohort (the multi-table path).
    Validates the file extension and size, then delegates to UserDatasetStorage.

    Args:
        file: Multipart file upload
        dataset_name: Optional human-readable name (defaults to filename stem)
    """
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    # A file name can name a patient: log its keyed hash, length and extension, never the text.
    log = logger.bind(
        **text_fingerprint(filename, prefix="filename"),
        filename_ext=ext if ext in ALLOWED_UPLOAD_EXTENSIONS else "other",
    )
    log.info("dataset_upload_requested")

    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{ext}' not allowed. Accepted: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
        )

    file_bytes = await file.read()

    valid, msg = UploadSecurityValidator.validate_file_size(file_bytes)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    declared = population_estimates_mark(population_estimates)
    display_name = dataset_name or Path(filename).stem

    metadata: dict[str, Any] = {
        "dataset_name": display_name,
        "original_filename": filename,
    }
    if declared:
        metadata["population_estimates"] = True

    storage = UserDatasetStorage()
    # A name already in use is a conflict the user resolves (another name, or delete the old
    # dataset), not a server failure. Checked before parsing, and again on the save's answer
    # (another upload may take the name meanwhile). There is no overwrite option on this route.
    if any(meta.get("dataset_name") == display_name for meta in storage.list_uploads()):
        log.info("dataset_upload_name_taken")
        raise HTTPException(status_code=409, detail=duplicate_name_message(display_name))
    # Parsing, DuckDB and Parquet writes block: keep them off the event loop.
    try:
        success, message, upload_id = await asyncio.to_thread(
            storage.save_upload,
            file_bytes=file_bytes,
            original_filename=filename,
            metadata=metadata,
            raise_user_errors=True,
        )
    except USER_FILE_ERRORS as e:
        # The file cannot be used as sent (unreadable, no patient ID, a damaged ZIP): the user's
        # to fix, not a server fault. These messages are written for the user and quote no cells.
        log.warning("dataset_upload_refused", error_type=type(e).__name__)
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not success or upload_id is None:
        if message == duplicate_name_message(display_name):
            log.info("dataset_upload_name_taken")
            raise HTTPException(status_code=409, detail=message)
        # Never "ready" unless the stored version answered the upload-time query-view probe.
        log.error("dataset_upload_failed", message=message)
        raise HTTPException(status_code=500, detail=message)

    # A save can reuse an existing upload_id (e.g. overwrite); drop any cached layer so the
    # next query reads the new data instead of the previous upload's DuckDB view.
    invalidate_semantic_layer_cache(upload_id)
    estimates_note = await asyncio.to_thread(_declare_population_estimates, storage, upload_id) if declared else None
    quality = await asyncio.to_thread(_upload_quality, storage, upload_id)
    response = _upload_response(storage, upload_id, display_name, message, quality).model_copy(
        update={"population_estimates": declared, "population_estimates_note": estimates_note}
    )
    log.info(
        "dataset_upload_completed",
        upload_id=upload_id,
        status=response.status,
        failed_checks=len(response.failed_checks),
        population_estimates=declared,
    )
    return response


@router.delete("/datasets/{dataset_id}", response_model=DatasetDeleteResponse)
def delete_dataset(dataset_id: str, db: Annotated[Session, Depends(get_db)]) -> DatasetDeleteResponse:
    """Delete a dataset: raw file, metadata, DuckDB tables, Parquet exports, enrichment overlays,
    its sessions and messages, and in-memory query results.

    Order, all under the store lock: DuckDB tables, files, then sessions and messages (their own
    transaction), then metadata last. A busy store (409) changes nothing; any later failure
    leaves the metadata, so the dataset stays listed and a retry finishes the delete. A retry
    after metadata is gone (a crash between the two) still removes leftover sessions.
    """
    try:
        validate_dataset_id(dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    log = logger.bind(dataset_id=dataset_id)
    storage = UserDatasetStorage()
    has_metadata = (storage.metadata_dir / f"{dataset_id}.json").exists()
    has_sessions = db.query(db_models.Session).filter(db_models.Session.dataset_id == dataset_id).count() > 0
    if not has_metadata and not has_sessions:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    log.info("dataset_delete_requested")

    # While the guard is held, queries, previews and semantic-layer builds for this id get 409,
    # so nothing can reload the rows into memory between the cache drop and the file purge.
    with dataset_deletion_guard(dataset_id):
        forget_query_service(dataset_id)

        deleted_sessions = 0

        def delete_sessions() -> None:
            # Before the metadata goes: if this commit fails, the dataset stays listed for a retry.
            nonlocal deleted_sessions
            sessions = db.query(db_models.Session).filter(db_models.Session.dataset_id == dataset_id).all()
            for db_session in sessions:
                db.delete(db_session)  # cascades to messages
            db.commit()
            deleted_sessions = len(sessions)

        try:
            # Waits for an in-flight quality backfill, so its files are purged too.
            with dataset_files_lock(dataset_id):
                report = purge_dataset_files(dataset_id, storage.upload_dir, before_metadata=delete_sessions)
        except DatasetStoreBusyError as e:
            log.warning("dataset_delete_store_busy")
            raise HTTPException(status_code=409, detail=str(e)) from e
        except DatasetDeletionError as e:
            log.error("dataset_delete_failed", error_type=type(e).__name__)
            # Messages name files, never absolute paths (see storage.dataset_deletion).
            raise HTTPException(status_code=500, detail=f"Dataset only partly deleted, try again: {e}") from e
        except SQLAlchemyError as e:
            db.rollback()
            log.error("dataset_delete_sessions_failed", error_type=type(e).__name__)
            raise HTTPException(
                status_code=500, detail="Dataset files deleted but its chats were not; try the delete again"
            ) from e
        finally:
            # Anything rebuilt despite the guard (a request already past it) is dropped again.
            invalidate_semantic_layer_cache(dataset_id)
            forget_query_service(dataset_id)

    log.info(
        "dataset_delete_completed",
        removed_paths=len(report.removed_paths),
        dropped_tables=len(report.dropped_tables),
        deleted_sessions=deleted_sessions,
    )
    return DatasetDeleteResponse(
        dataset_id=dataset_id,
        removed_paths=len(report.removed_paths),
        dropped_tables=len(report.dropped_tables),
        deleted_sessions=deleted_sessions,
    )

"""Dataset API routes for Electron UI.

Endpoints:
- GET  /api/datasets                      - List available datasets
- GET  /api/datasets/{dataset_id}         - Get dataset metadata
- GET  /api/datasets/{dataset_id}/preview - Get sample rows
- POST /api/datasets/upload               - Upload a dataset file
- DELETE /api/datasets/{dataset_id}       - Delete a dataset and everything derived from it
"""

import asyncio
from pathlib import Path
from typing import Annotated, Any, Literal

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from clinical_analytics.api.db.database import get_db
from clinical_analytics.api.dependencies import (
    dataset_deletion_guard,
    invalidate_semantic_layer_cache,
    reject_if_deleting,
)
from clinical_analytics.api.models import database as db_models
from clinical_analytics.api.routes.queries import forget_query_service
from clinical_analytics.core.log_redaction import error_fields
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
from clinical_analytics.storage.dataset_deletion import (
    DatasetDeletionError,
    DatasetStoreBusyError,
    InvalidDatasetIdError,
    purge_dataset_files,
    validate_dataset_id,
)
from clinical_analytics.ui.storage.user_datasets import (
    UploadSecurityValidator,
    UserDatasetStorage,
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

        # Build schema from columns
        columns = info.get("columns", [])
        schema = {col: "unknown" for col in columns}  # Type inference would go here

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

        import polars as pl

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


# ============================================================================
# Upload (Phase 8)
# ============================================================================


class DatasetUploadResponse(BaseModel):
    """Response after a successful dataset upload."""

    model_config = ConfigDict(extra="forbid")

    upload_id: str = Field(..., description="Created upload identifier")
    dataset_name: str = Field(..., description="Display name for the dataset")
    status: Literal["ready", "failed"] = Field(..., description="Upload processing status")
    message: str = Field("", description="Human-readable status message")


class DatasetDeleteResponse(BaseModel):
    """What a dataset delete removed."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="Deleted dataset identifier")
    removed_paths: int = Field(..., description="Files and folders removed from the data home")
    dropped_tables: int = Field(..., description="DuckDB tables dropped")
    deleted_sessions: int = Field(..., description="Sessions (with their messages) deleted")


ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls", ".sav"}


@router.post("/datasets/sample", response_model=DatasetUploadResponse)
def load_sample_dataset() -> DatasetUploadResponse:
    """Load the bundled sample cohort into upload storage (stranger-ready demo)."""
    from clinical_analytics.datasets.demo import DEMO_DATASET_NAME, build_demo_metadata, get_demo_csv_path

    log = logger.bind()
    log.info("dataset_sample_load_requested")

    path = get_demo_csv_path()
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Sample cohort missing: {path}")

    storage = UserDatasetStorage()
    success, message, upload_id = storage.save_upload(
        file_bytes=path.read_bytes(),
        original_filename=path.name,
        metadata=build_demo_metadata(),
        overwrite=True,
    )
    if not success or upload_id is None:
        log.error("dataset_sample_load_failed", message=message)
        raise HTTPException(status_code=500, detail=message)

    # Reloading the sample overwrites the same upload_id; drop the cached layer like /upload does.
    invalidate_semantic_layer_cache(upload_id)
    log.info("dataset_sample_load_completed", upload_id=upload_id)
    return DatasetUploadResponse(
        upload_id=upload_id,
        dataset_name=DEMO_DATASET_NAME,
        status="ready",
        message=message or "Sample cohort loaded",
    )


@router.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV or Excel file to upload"),
    dataset_name: str | None = Form(None, description="Optional display name"),
) -> DatasetUploadResponse:
    """Upload a dataset file (CSV, Excel, or SPSS).

    Validates the file extension and size, then delegates to UserDatasetStorage.

    Args:
        file: Multipart file upload
        dataset_name: Optional human-readable name (defaults to filename stem)
    """
    log = logger.bind(filename=file.filename)
    log.info("dataset_upload_requested")

    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{ext}' not allowed. Accepted: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
        )

    file_bytes = await file.read()

    valid, msg = UploadSecurityValidator.validate_file_size(file_bytes)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    display_name = dataset_name or Path(filename).stem

    metadata: dict[str, Any] = {
        "dataset_name": display_name,
        "original_filename": filename,
    }

    storage = UserDatasetStorage()
    # Parsing, DuckDB and Parquet writes block: keep them off the event loop.
    success, message, upload_id = await asyncio.to_thread(
        storage.save_upload,
        file_bytes=file_bytes,
        original_filename=filename,
        metadata=metadata,
    )

    if not success or upload_id is None:
        log.error("dataset_upload_failed", message=message)
        raise HTTPException(status_code=500, detail=message)

    # A save can reuse an existing upload_id (e.g. overwrite); drop any cached layer so the
    # next query reads the new data instead of the previous upload's DuckDB view.
    invalidate_semantic_layer_cache(upload_id)
    log.info("dataset_upload_completed", upload_id=upload_id)
    return DatasetUploadResponse(
        upload_id=upload_id,
        dataset_name=display_name,
        status="ready",
        message=message,
    )


@router.delete("/datasets/{dataset_id}", response_model=DatasetDeleteResponse)
def delete_dataset(dataset_id: str, db: Annotated[Session, Depends(get_db)]) -> DatasetDeleteResponse:
    """Delete a dataset: raw file, metadata, DuckDB tables, Parquet exports, enrichment overlays,
    its sessions and messages, and in-memory query results.

    Order: in-memory state, then sessions, then files with metadata last, so a failure leaves the
    dataset listed and the delete can simply be retried.
    """
    try:
        validate_dataset_id(dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    log = logger.bind(dataset_id=dataset_id)
    storage = UserDatasetStorage()
    if not (storage.metadata_dir / f"{dataset_id}.json").exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    log.info("dataset_delete_requested")

    # While the guard is held, queries, previews and semantic-layer builds for this id get 409,
    # so nothing can reload the rows into memory between the cache drop and the file purge.
    with dataset_deletion_guard(dataset_id):
        forget_query_service(dataset_id)

        sessions = db.query(db_models.Session).filter(db_models.Session.dataset_id == dataset_id).all()
        for db_session in sessions:
            db.delete(db_session)  # cascades to messages
        db.commit()

        try:
            report = purge_dataset_files(dataset_id, storage.upload_dir)
        except DatasetStoreBusyError as e:
            log.warning("dataset_delete_store_busy", deleted_sessions=len(sessions))
            raise HTTPException(status_code=409, detail=str(e)) from e
        except DatasetDeletionError as e:
            log.error("dataset_delete_failed", deleted_sessions=len(sessions), error_type=type(e).__name__)
            # Messages name files, never absolute paths (see storage.dataset_deletion).
            raise HTTPException(status_code=500, detail=f"Dataset only partly deleted, try again: {e}") from e
        finally:
            # Anything rebuilt despite the guard (a request already past it) is dropped again.
            invalidate_semantic_layer_cache(dataset_id)
            forget_query_service(dataset_id)

    log.info(
        "dataset_delete_completed",
        removed_paths=len(report.removed_paths),
        dropped_tables=len(report.dropped_tables),
        deleted_sessions=len(sessions),
    )
    return DatasetDeleteResponse(
        dataset_id=dataset_id,
        removed_paths=len(report.removed_paths),
        dropped_tables=len(report.dropped_tables),
        deleted_sessions=len(sessions),
    )

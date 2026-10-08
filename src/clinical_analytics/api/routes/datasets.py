"""Dataset API routes for Electron UI.

Endpoints:
- GET  /api/datasets                      - List available datasets
- GET  /api/datasets/{dataset_id}         - Get dataset metadata
- GET  /api/datasets/{dataset_id}/preview - Get sample rows
- POST /api/datasets/upload               - Upload a dataset file
"""

from pathlib import Path
from typing import Any, Literal

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
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


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets() -> DatasetListResponse:
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
        log.warning("datasets_list_uploads_failed", error=str(e))

    # Future: Add built-in datasets from registry (deferred to Phase 1.1)

    log.info("datasets_list_completed", count=len(datasets))
    return DatasetListResponse(datasets=datasets, total=len(datasets))


@router.get("/datasets/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: str) -> DatasetDetail:
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

        log.info("dataset_get_completed", name=info.get("name"))
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
        log.warning("dataset_get_not_found", error=str(e))
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}") from e

    except Exception as e:
        log.error("dataset_get_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}") from e


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(
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

    try:
        # Load dataset
        storage = UserDatasetStorage()
        metadata = storage.get_upload_metadata(dataset_id)

        if not metadata:
            raise ValueError(f"Upload {dataset_id} not found")

        # Get unified cohort CSV
        csv_path = storage.upload_dir / f"{dataset_id}_unified_cohort.csv"
        if not csv_path.exists():
            raise ValueError(f"Unified cohort not found for {dataset_id}")

        # Load with Polars and get preview
        import polars as pl

        df = pl.read_csv(csv_path)
        total_rows = df.height
        preview_df = df.head(limit)

        rows = preview_df.to_dicts()
        columns = df.columns

        log.info("dataset_preview_completed", row_count=len(rows), total=total_rows)
        return DatasetPreview(
            dataset_id=dataset_id,
            rows=rows,
            columns=columns,
            total_rows=total_rows,
        )

    except ValueError as e:
        log.warning("dataset_preview_not_found", error=str(e))
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}") from e

    except Exception as e:
        log.error("dataset_preview_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to preview dataset: {str(e)}") from e


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


ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls", ".sav"}


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
    success, message, upload_id = storage.save_upload(
        file_bytes=file_bytes,
        original_filename=filename,
        metadata=metadata,
    )

    if not success or upload_id is None:
        log.error("dataset_upload_failed", message=message)
        raise HTTPException(status_code=500, detail=message)

    log.info("dataset_upload_completed", upload_id=upload_id)
    return DatasetUploadResponse(
        upload_id=upload_id,
        dataset_name=display_name,
        status="ready",
        message=message,
    )

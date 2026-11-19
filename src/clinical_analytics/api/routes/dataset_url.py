# SPDX-License-Identifier: MIT
"""Load a public dataset from a link.

Endpoint:
- POST /api/datasets/from-url - download a public data file over HTTPS and upload it

The download (``storage/url_fetch.py``) is the only new step: its bytes go through the file
upload route unchanged, so formats, size limits, typing, quality checks and the ready probe are
the same as for a chosen file. The link, the time, the size and the SHA-256 are then saved with
the dataset (metadata key ``source``) for the user and the methods section.
"""

import asyncio
import io
from typing import Final

import structlog
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.api.routes.datasets import (
    ALLOWED_UPLOAD_EXTENSIONS,
    DatasetUploadResponse,
    upload_dataset,
)
from clinical_analytics.core.log_redaction import text_fingerprint
from clinical_analytics.storage.url_fetch import (
    MAX_URL_LENGTH,
    FetchFailureKind,
    SourceProvenance,
    UrlFetchError,
    fetch_public_file,
)
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage, active_version_id

logger = structlog.get_logger()

router = APIRouter()

SOURCE_METADATA_KEY: Final[str] = "source"

# The user's link was the problem (4xx), or the website was (502/504).
FETCH_FAILURE_STATUS: Final[dict[FetchFailureKind, int]] = {
    "invalid_link": 400,
    "blocked_address": 400,
    "insecure": 400,
    "not_data": 415,
    "too_large": 413,
    "missing": 502,
    "refused": 502,
    "site_error": 502,
    "unreachable": 502,
    "too_many_redirects": 502,
    "timeout": 504,
}


class DatasetFromUrlRequest(BaseModel):
    """A link to a public data file, and an optional display name."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., min_length=1, max_length=MAX_URL_LENGTH, description="HTTPS link to a public data file")
    dataset_name: str | None = Field(None, max_length=200, description="Optional display name")


class SourceRecord(SourceProvenance):
    """Provenance as returned to the app, with the sentence for a methods section."""

    methods_text: str = Field(..., description="One sentence for a methods section")


class DatasetFromUrlResponse(DatasetUploadResponse):
    """The upload response, plus where the data came from."""

    source: SourceRecord = Field(..., description="Link, download time, size and SHA-256")


def source_record(provenance: SourceProvenance) -> SourceRecord:
    return SourceRecord(**provenance.model_dump(), methods_text=provenance.methods_sentence())


def _record_source(upload_id: str, provenance: SourceProvenance) -> bool:
    """Save provenance on the dataset's metadata, tagged with the version it describes."""
    storage = UserDatasetStorage()
    version = active_version_id(storage.get_upload_metadata(upload_id))
    record = {**provenance.model_dump(mode="json"), "dataset_version": version}
    ok, _ = storage.update_metadata(upload_id, {SOURCE_METADATA_KEY: record})
    return ok


@router.post("/datasets/from-url", response_model=DatasetFromUrlResponse)
async def load_dataset_from_url(request: DatasetFromUrlRequest) -> DatasetFromUrlResponse:
    """Download a public data file from an HTTPS link and load it like an uploaded file.

    Raises:
        HTTPException: 400 bad or non-public link, 413 too large, 415 not a data file,
            502 the website failed or refused, 504 it was too slow; the detail is written
            for the user. Upload failures come back exactly as the upload route reports them.
    """
    # A link can carry an access token in its query: log a keyed hash of it, never the text.
    log = logger.bind(**text_fingerprint(request.url, prefix="url"))
    log.info("dataset_url_requested")
    try:
        fetched = await asyncio.to_thread(
            fetch_public_file, request.url, accepted_extensions=frozenset(ALLOWED_UPLOAD_EXTENSIONS)
        )
    except UrlFetchError as e:
        log.warning("dataset_url_fetch_failed", kind=e.kind)
        raise HTTPException(status_code=FETCH_FAILURE_STATUS[e.kind], detail=e.message) from e

    upload = UploadFile(file=io.BytesIO(fetched.content), filename=fetched.filename, size=len(fetched.content))
    uploaded = await upload_dataset(file=upload, dataset_name=request.dataset_name)

    message = uploaded.message
    if not await asyncio.to_thread(_record_source, uploaded.upload_id, fetched.provenance):
        # The data is loaded; say plainly that its source was not kept, and show it once here.
        log.error("dataset_url_source_not_recorded", upload_id=uploaded.upload_id)
        message = (
            f"{message} The link and download date couldn't be saved with this cohort; "
            "copy them from below for your records."
        ).strip()
    log.info("dataset_url_completed", upload_id=uploaded.upload_id)
    fields = {**dict(uploaded), "message": message}
    return DatasetFromUrlResponse(**fields, source=source_record(fetched.provenance))

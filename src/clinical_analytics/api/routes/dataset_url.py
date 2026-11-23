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
import json
import re
import zipfile
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

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
    population_estimates: str | None = Field(
        None,
        description="Declare public population estimates exactly as the upload form does (ADR014 §7); "
        "or later with POST /api/datasets/{id}/population-estimates",
    )


class SourceRecord(SourceProvenance):
    """Provenance as returned to the app, with the sentence for a methods section."""

    methods_text: str = Field(..., description="One sentence for a methods section")


class DatasetFromUrlResponse(DatasetUploadResponse):
    """The upload response, plus where the data came from."""

    source: SourceRecord = Field(..., description="Link, download time, size and SHA-256")
    source_recorded: bool = Field(..., description="False if the source could not be saved with the dataset")


SOCRATA_HOST: Final[str] = "data.cdc.gov"
# A Socrata dataset id: two groups of four lower-case letters or digits ("bi63-dtpu").
_SOCRATA_ID: Final = re.compile(r"(?<![a-z0-9])([a-z0-9]{4}-[a-z0-9]{4})(?![a-z0-9])")
_SOCRATA_ID_FULL: Final = re.compile(r"[a-z0-9]{4}-[a-z0-9]{4}")
MAX_DICTIONARY_COLUMNS: Final[int] = 500


def socrata_id(url: str) -> str | None:
    """The dataset id of a data.cdc.gov link (``/api/views/<id>/rows.csv``, ``/resource/<id>.csv``,
    ``/<category>/<name>/<id>``), or None for any other host or a link without one."""
    parts = urlsplit(url)
    if (parts.hostname or "").lower() != SOCRATA_HOST or parts.scheme != "https":
        return None
    try:
        port = parts.port
    except ValueError:  # a port that is not a number
        return None
    if port not in (None, 443) or parts.username is not None or parts.password is not None:
        return None
    found = _SOCRATA_ID.findall(parts.path.lower())
    return found[-1] if found else None


def socrata_dictionary(url: str) -> str | None:
    """The column names and descriptions of a data.cdc.gov dataset, as dictionary text, or None
    (another host, no id, or the metadata could not be read: logged, and the file loads without it).

    Only ``https://data.cdc.gov/api/views/<id>.json`` is fetched, through the same checked public
    download as the data, with an id of the exact Socrata form.
    """
    dataset = socrata_id(url)
    if dataset is None or not _SOCRATA_ID_FULL.fullmatch(dataset):
        return None
    meta_url = f"https://{SOCRATA_HOST}/api/views/{dataset}.json"
    try:
        fetched = fetch_public_file(meta_url, accepted_extensions=frozenset({".json"}))
        document = json.loads(fetched.content)
    except (UrlFetchError, ValueError) as e:
        logger.warning("dataset_url_socrata_metadata_unavailable", error_type=type(e).__name__)
        return None
    if not isinstance(document, dict) or not isinstance(document.get("columns"), list):
        logger.warning("dataset_url_socrata_metadata_unexpected")
        return None
    lines = [f"DATA DICTIONARY FROM {SOCRATA_HOST} (dataset {dataset})"]
    about = document.get("description")
    if isinstance(about, str) and about.strip():
        lines.extend(line.strip() for line in about.strip().splitlines() if line.strip())
    lines.append("")
    for column in document["columns"][:MAX_DICTIONARY_COLUMNS]:
        if not isinstance(column, dict):
            continue
        description = str(column.get("description") or "").strip().replace("\n", " ")
        for name in dict.fromkeys(str(column.get(k) or "").strip() for k in ("name", "fieldName")):
            if name and description and " " not in name:
                lines.append(f"{name}  {description}")
            elif name and description:
                lines.append(f"| {name} | {description} |")
    logger.info("dataset_url_socrata_dictionary", columns=len(document["columns"]))
    return "\n".join(lines) + "\n"


def _with_dictionary(content: bytes, filename: str, dictionary: str) -> tuple[bytes, str]:
    """The data file and its dictionary as one ZIP, the upload's table-plus-documentation path."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, content)
        archive.writestr("dictionary.txt", dictionary)
    return buffer.getvalue(), f"{Path(filename).stem}.zip"


def source_record(provenance: SourceProvenance) -> SourceRecord:
    return SourceRecord(**provenance.model_dump(), methods_text=provenance.methods_sentence())


def _record_source(upload_id: str, provenance: SourceProvenance) -> bool:
    """Save provenance on the dataset's metadata, tagged with the version it describes."""
    storage = UserDatasetStorage()
    try:
        version = active_version_id(storage.get_upload_metadata(upload_id))
    except (OSError, ValueError) as e:
        logger.error("dataset_url_source_metadata_unreadable", upload_id=upload_id, error_type=type(e).__name__)
        return False
    record = {**provenance.model_dump(mode="json"), "dataset_version": version}
    ok, _ = storage.update_metadata(upload_id, {SOURCE_METADATA_KEY: record})
    return ok


def _refuse_taken_name(name: str) -> None:
    """409 before uploading when a cohort already has this name (portal exports share names)."""
    taken = any(meta.get("dataset_name") == name for meta in UserDatasetStorage().list_uploads())
    if taken:
        raise HTTPException(
            status_code=409,
            detail=(
                f"A cohort named “{name}” already exists. "
                "Type a different name in Cohort name, then load the link again."
            ),
        )


@router.post("/datasets/from-url", response_model=DatasetFromUrlResponse)
async def load_dataset_from_url(request: DatasetFromUrlRequest) -> DatasetFromUrlResponse:
    """Download a public data file from an HTTPS link and load it like an uploaded file.

    Raises:
        HTTPException: 400 bad or non-public link, 413 too large, 415 not a data file,
            409 a cohort with that name exists, 502 the website failed or refused, 504 it
            was too slow; the detail is written
            for the user. Upload failures come back exactly as the upload route reports them.
    """
    # A link can carry an access token in its query: log a keyed hash of it, never the text.
    log = logger.bind(**text_fingerprint(request.url, prefix="url"))
    log.info("dataset_url_requested")
    chosen_name = (request.dataset_name or "").strip() or None
    if chosen_name:
        await asyncio.to_thread(_refuse_taken_name, chosen_name)
    try:
        fetched = await asyncio.to_thread(
            fetch_public_file, request.url, accepted_extensions=frozenset(ALLOWED_UPLOAD_EXTENSIONS)
        )
    except UrlFetchError as e:
        log.warning("dataset_url_fetch_failed", kind=e.kind)
        raise HTTPException(status_code=FETCH_FAILURE_STATUS[e.kind], detail=e.message) from e

    if chosen_name is None:
        await asyncio.to_thread(_refuse_taken_name, Path(fetched.filename).stem)
    # A data.cdc.gov (Socrata) link: its column names and descriptions are the data dictionary the
    # contract inference reads, as for an uploaded ZIP of a table and its dictionary.
    dictionary = await asyncio.to_thread(socrata_dictionary, fetched.provenance.final_url)
    content, filename = fetched.content, fetched.filename
    if dictionary is not None:
        content, filename = _with_dictionary(fetched.content, fetched.filename, dictionary)
    upload = UploadFile(file=io.BytesIO(content), filename=filename, size=len(content))
    # Called directly, not through FastAPI, so every Form() parameter must be passed explicitly:
    # a default left out is a Form object, not None. The upload runs the same contract inference
    # (aggregate tables included); estimates can also be declared after loading.
    uploaded = await upload_dataset(
        file=upload,
        dataset_name=chosen_name or Path(fetched.filename).stem,
        population_estimates=request.population_estimates,
    )

    message = uploaded.message
    recorded = await asyncio.to_thread(_record_source, uploaded.upload_id, fetched.provenance)
    if not recorded:
        # The data is loaded; say plainly that its source was not kept, and show it once here.
        log.error("dataset_url_source_not_recorded", upload_id=uploaded.upload_id)
        message = (
            f"{message} The link and download date couldn't be saved with this cohort; "
            "keep the source sentence for your records."
        ).strip()
    log.info("dataset_url_completed", upload_id=uploaded.upload_id)
    fields = {**dict(uploaded), "message": message}
    return DatasetFromUrlResponse(**fields, source=source_record(fetched.provenance), source_recorded=recorded)

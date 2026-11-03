"""Upload-time orchestration: build (or reuse) the version's contract, check the data, store both."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from clinical_analytics.contracts.builder import ContractInputs, build_contract
from clinical_analytics.contracts.errors import ContractError
from clinical_analytics.contracts.quality import QualityReport, run_quality
from clinical_analytics.contracts.store import contract_path, read_contract, write_contract, write_report
from clinical_analytics.core.log_redaction import error_fields
from clinical_analytics.core.metadata_patch import (
    ExclusionPatternPatch,
    MetadataPatch,
    PatchStatus,
    RelationshipPatch,
    ResolvedDatasetMetadata,
)

logger = structlog.get_logger()

Patch = MetadataPatch | ExclusionPatternPatch | RelationshipPatch


def id_column_from_metadata(metadata: dict[str, Any]) -> str | None:
    """The column schema inference (or the upload wizard) mapped to ``patient_id``."""
    mapping = (metadata.get("inferred_schema") or {}).get("column_mapping") or {}
    return next((col for col, role in mapping.items() if role == "patient_id"), None)


def accepted_patch_ids(patches: list[Patch]) -> tuple[str, ...]:
    """Ids whose latest log row is ACCEPTED (same rule as ``metadata_resolver``), sorted."""
    latest: dict[str, Patch] = {p.patch_id: p for p in patches}
    return tuple(sorted(pid for pid, p in latest.items() if p.status == PatchStatus.ACCEPTED))


def _resolve(df: pl.DataFrame, patches: list[Patch]) -> ResolvedDatasetMetadata:
    from clinical_analytics.core.metadata_resolver import resolve_metadata
    from clinical_analytics.core.schema_inference import SchemaInferenceEngine

    return resolve_metadata(SchemaInferenceEngine().infer_schema(df), patches)


def check_upload(
    data_dir: Path,
    df: pl.DataFrame,
    *,
    upload_id: str,
    dataset_version: str,
    metadata: dict[str, Any],
    overlay_dir: Path | None = None,
    run_at: datetime | None = None,
) -> QualityReport:
    """Write the version's contract if it has none, check ``df`` against it, store the report.

    Running twice is safe: the contract is reused (immutable per version) and each run adds a
    report file.

    Raises:
        ContractError: building, storing or compiling the contract failed.
    """
    log = logger.bind(upload_id=upload_id, dataset_version=dataset_version)
    path = contract_path(data_dir, upload_id, dataset_version)
    if path.exists():
        contract = read_contract(path)
    else:
        patches: list[Patch] = []
        if overlay_dir is not None:
            from clinical_analytics.core.overlay_store import OverlayStore

            patches = OverlayStore(base_dir=overlay_dir).load_patches(upload_id, dataset_version)
        accepted = accepted_patch_ids(patches)
        inputs = ContractInputs(
            upload_id=upload_id,
            dataset_version=dataset_version,
            dataset_name=str(metadata.get("dataset_name") or upload_id),
            id_column=id_column_from_metadata(metadata),
            source_file_sha256=metadata.get("source_file_sha256"),
            resolved=_resolve(df, patches) if accepted else None,
            accepted_patch_ids=accepted,
        )
        contract = build_contract(df, inputs)
        if not write_contract(data_dir, upload_id, dataset_version, contract):
            contract = read_contract(path)  # another writer won; its contract is the version's
    report = run_quality(df.lazy(), contract, run_at=run_at)
    write_report(data_dir, report)
    log.info("upload_checked", failed=report.failed, warnings=report.warnings)
    return report


def check_upload_or_log(
    data_dir: Path,
    df: pl.DataFrame,
    *,
    upload_id: str,
    dataset_version: str,
    metadata: dict[str, Any],
    overlay_dir: Path | None = None,
) -> QualityReport | None:
    """``check_upload`` for the upload path: a failure is logged as an error, not raised.

    The upload's data is already stored when this runs; a contract bug must not lose it. The
    quality endpoint regenerates a missing contract and report on the next request.
    """
    try:
        return check_upload(
            data_dir,
            df,
            upload_id=upload_id,
            dataset_version=dataset_version,
            metadata=metadata,
            overlay_dir=overlay_dir,
        )
    except (ContractError, OSError, ValueError, pl.exceptions.PolarsError) as e:
        logger.error("upload_check_failed", upload_id=upload_id, dataset_version=dataset_version, **error_fields(e))
        return None

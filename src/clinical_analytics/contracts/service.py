# SPDX-License-Identifier: MIT
"""Upload-time orchestration: build (or reuse) the version's contract, check the data, store both."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog
from open_data_contract_standard.model import OpenDataContractStandard

from clinical_analytics.contracts.builder import ContractInputs, build_contract
from clinical_analytics.contracts.errors import ContractDamagedError, ContractError, ContractValidationError
from clinical_analytics.contracts.quality import QualityNotRun, QualityReport, run_quality
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

DAMAGED_CONTRACT_REASON = "the saved data contract for this version is damaged"

Patch = MetadataPatch | ExclusionPatternPatch | RelationshipPatch


def id_column_from_metadata(metadata: dict[str, Any]) -> str | None:
    """The column schema inference (or the upload wizard) mapped to ``patient_id``."""
    mapping = (metadata.get("inferred_schema") or {}).get("column_mapping") or {}
    return next((col for col, role in mapping.items() if role == "patient_id"), None)


def _flatten_columns(value: Any) -> list[str]:
    """Column names in a mapping value: a name, a list of names, or a dict of them
    (``time_variables: {"time_zero": "admit_date"}``), nested to any depth."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [column for item in value.values() for column in _flatten_columns(item)]
    if isinstance(value, list | tuple):
        return [column for item in value for column in _flatten_columns(item)]
    return []


def analysis_columns_from_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Columns the upload names as analysis variables (outcomes, predictors, time and event).

    Only these get the 5% missingness warning at upload; other columns' missingness is listed, and
    an answer notes it for any column the question uses (``core.answer_checks.report_notes``).
    """
    found: list[str] = []
    mapping = metadata.get("variable_mapping") or {}
    for role, value in mapping.items():
        if role == "patient_id":
            continue
        for column in _flatten_columns(value):
            if column not in found:
                found.append(column)
    outcomes = (metadata.get("inferred_schema") or {}).get("outcomes") or {}
    for column in outcomes if isinstance(outcomes, dict) else []:
        if isinstance(column, str) and column not in found:
            found.append(column)
    return tuple(found)


def accepted_patch_ids(patches: list[Patch]) -> tuple[str, ...]:
    """Ids whose latest log row is ACCEPTED (same rule as ``metadata_resolver``), sorted."""
    latest: dict[str, Patch] = {p.patch_id: p for p in patches}
    return tuple(sorted(pid for pid, p in latest.items() if p.status == PatchStatus.ACCEPTED))


def _resolve(df: pl.DataFrame, patches: list[Patch]) -> ResolvedDatasetMetadata:
    from clinical_analytics.core.metadata_resolver import resolve_metadata
    from clinical_analytics.core.schema_inference import SchemaInferenceEngine

    return resolve_metadata(SchemaInferenceEngine().infer_schema(df), patches)


def read_saved_contract(path: Path) -> OpenDataContractStandard:
    """The version's saved contract, which must hold a schema object to check against.

    Raises:
        ContractDamagedError: empty, truncated, not ODCS, or without a schema object.
        ContractStoreError: the file cannot be read.
    """
    try:
        contract = read_contract(path)
    except ContractValidationError as e:
        raise ContractDamagedError(f"Saved contract {path.name} is damaged") from e
    if not contract.schema_:
        raise ContractDamagedError(f"Saved contract {path.name} has no schema object")
    return contract


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
        contract = read_saved_contract(path)
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
            analysis_columns=analysis_columns_from_metadata(metadata),
            source_file_sha256=metadata.get("source_file_sha256"),
            resolved=_resolve(df, patches) if accepted else None,
            accepted_patch_ids=accepted,
        )
        contract = build_contract(df, inputs)
        if not write_contract(data_dir, upload_id, dataset_version, contract):
            contract = read_saved_contract(path)  # another writer won; its contract is the version's
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
) -> QualityReport | QualityNotRun:
    """``check_upload`` for the upload path: a failure is logged and returned as not run, not raised.

    The upload's data is already stored when this runs; a contract bug must not lose it, but the
    user must see that the check did not run. The quality endpoint retries on the next request.
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
    except ContractDamagedError as e:
        logger.error(
            "upload_check_contract_damaged", upload_id=upload_id, dataset_version=dataset_version, **error_fields(e)
        )
        return QualityNotRun(reason=DAMAGED_CONTRACT_REASON)
    except (ContractError, OSError, ValueError, pl.exceptions.PolarsError) as e:
        logger.error("upload_check_failed", upload_id=upload_id, dataset_version=dataset_version, **error_fields(e))
        return QualityNotRun(reason=f"the check failed ({type(e).__name__})")

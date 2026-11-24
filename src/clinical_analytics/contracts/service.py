# SPDX-License-Identifier: MIT
"""Upload-time orchestration: build (or reuse) the version's contract, check the data, store both."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog
from open_data_contract_standard.model import OpenDataContractStandard

from clinical_analytics.contracts import config
from clinical_analytics.contracts.builder import ContractInputs, build_contract
from clinical_analytics.contracts.errors import ContractDamagedError, ContractError, ContractValidationError
from clinical_analytics.contracts.quality import QualityNotRun, QualityReport, run_quality
from clinical_analytics.contracts.semantics import DatasetSemantics
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


def stored_semantics(df: pl.DataFrame, metadata: dict[str, Any]) -> DatasetSemantics:
    """Semantics for a stored version whose contract has none (written before ADR017, or never
    built): the one inference pass (``contracts.inference``) over the stored frame without the
    columns the app added, with the upload's documentation, and the app columns recorded from
    the upload's ``synthetic_id_metadata``. Never stored over an existing contract."""
    from clinical_analytics.contracts.inference import UploadFacts, infer_semantics
    from clinical_analytics.contracts.semantics import AppColumn
    from clinical_analytics.core.identifier_rules import SOURCE_FORMAT_KEY, added_id_columns_from_metadata

    added = [c for c in added_id_columns_from_metadata(metadata) if c in df.columns]
    synthetic = metadata.get("synthetic_id_metadata")
    info = synthetic.get("patient_id") if isinstance(synthetic, dict) else None
    info = info if isinstance(info, dict) else {}
    semantics = infer_semantics(
        df.drop(added),
        UploadFacts(
            doc_context=metadata.get("doc_context"),
            source_format=str(metadata.get(SOURCE_FORMAT_KEY) or ""),
            population_estimates=metadata.get("population_estimates") is True
            or info.get("population_estimates") is True,
        ),
    )
    source = info.get("patient_id_source")
    made_from = tuple(str(c) for c in info.get("patient_id_columns") or [])
    apps = tuple(
        AppColumn(
            column=c,
            kind="row_number" if source == "row_index" else ("hash" if source == "composite" else "copy"),
            made_from=made_from,
        )
        for c in added
    )
    return semantics.model_copy(update={"app_columns": apps})


def version_semantics(
    data_dir: Path,
    dataset_id: str,
    version: str,
    *,
    metadata: dict[str, Any],
    stored_frame: Callable[[], pl.DataFrame],
    declarations: Any = None,
) -> DatasetSemantics:
    """``semantics_for``, or for a version stored before its contract held semantics, the same
    inference pass over its stored frame (``stored_semantics``), read-only. Every reader (the
    semantic layer, the upload response, ``GET /semantics``, aggregate) uses this, so they agree.

    ``declarations``: a ``DeclarationStore``; default the file store in ``data_dir``.

    Raises:
        ContractError: the contract or the declarations can't be read (never hidden).
    """
    return read_version_semantics(
        data_dir, dataset_id, version, metadata=metadata, stored_frame=stored_frame, declarations=declarations
    )()


def read_version_semantics(
    data_dir: Path,
    dataset_id: str,
    version: str,
    *,
    metadata: dict[str, Any],
    stored_frame: Callable[[], pl.DataFrame],
    declarations: Any = None,
) -> Callable[[], DatasetSemantics]:
    """``version_semantics`` in two steps. This call does every file read (the contract, the
    declarations and, for a version stored before ADR017, the stored frame); the step it returns
    reads no file. That step is the inference pass over the frame, which is slow on a large one
    (4 to 6 s for a 1.8M-row, 13-column cohort), so a caller that holds a lock around the reads
    runs it after releasing the lock.

    Raises:
        ContractError: the contract or the declarations can't be read (never hidden).
    """
    from clinical_analytics.contracts.declarations import FileDeclarationStore
    from clinical_analytics.contracts.semantics import SemanticsMissingError, semantics_for, with_declarations

    store = declarations if declarations is not None else FileDeclarationStore(data_dir)
    try:
        semantics = semantics_for(data_dir, dataset_id, version, declarations=store)
    except SemanticsMissingError:
        frame = stored_frame()
        declared = store.load(dataset_id)

        def infer() -> DatasetSemantics:
            logger.info("semantics_inferred_for_stored_version", dataset_id=dataset_id, dataset_version=version)
            inferred = stored_semantics(frame, metadata).model_copy(update={"dataset_version": version})
            return with_declarations(inferred, declared)

        return infer
    return lambda: semantics


def accepted_patch_ids(patches: list[Patch]) -> tuple[str, ...]:
    """Ids whose latest log row is ACCEPTED (same rule as ``metadata_resolver``), sorted."""
    latest: dict[str, Patch] = {p.patch_id: p for p in patches}
    return tuple(sorted(pid for pid, p in latest.items() if p.status == PatchStatus.ACCEPTED))


def _resolve(df: pl.DataFrame, patches: list[Patch], semantics: DatasetSemantics) -> ResolvedDatasetMetadata:
    from clinical_analytics.core.metadata_resolver import resolve_metadata
    from clinical_analytics.core.schema_inference import SchemaInferenceEngine

    return resolve_metadata(SchemaInferenceEngine().infer_schema(df, semantics=semantics), patches)


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


def previous_version_id(metadata: dict[str, Any], dataset_version: str) -> str | None:
    """The version before ``dataset_version`` in the upload's ``version_history`` (an overwrite
    keeps the upload id and appends the new version), or None for a first upload."""
    history = [str(v.get("version")) for v in metadata.get("version_history") or [] if v.get("version")]
    if dataset_version not in history:
        return None
    before = history[: len(history) - 1 - history[::-1].index(dataset_version)]
    return next((version for version in reversed(before) if version != dataset_version), None)


def _previous_contract(
    data_dir: Path, upload_id: str, metadata: dict[str, Any], dataset_version: str
) -> OpenDataContractStandard | None:
    """The previous version's saved contract, or None (first version, a legacy version without a
    contract, or a damaged one: the type check is then reported as not run, never as passed)."""
    previous = previous_version_id(metadata, dataset_version)
    if previous is None:
        return None
    path = contract_path(data_dir, upload_id, previous)
    if not path.exists():
        return None
    try:
        return read_saved_contract(path)
    except ContractError as e:
        logger.warning(
            "previous_contract_unreadable", upload_id=upload_id, previous_version=previous, **error_fields(e)
        )
        return None


def population_estimates_properties(metadata: dict[str, Any]) -> dict[str, Any]:
    """Contract properties for an upload declared public population estimates (ADR014): the
    privacy class and the statement the uploader agreed to. Empty for any other upload."""
    if metadata.get("population_estimates") is not True:
        return {}
    return {
        "privacyClass": config.POPULATION_ESTIMATES_PRIVACY_CLASS,
        "populationEstimates": config.POPULATION_ESTIMATES_STATEMENT,
    }


def write_version_contract(
    data_dir: Path,
    df: pl.DataFrame,
    *,
    upload_id: str,
    dataset_version: str,
    metadata: dict[str, Any],
    semantics: DatasetSemantics | None = None,
    overlay_dir: Path | None = None,
) -> tuple[OpenDataContractStandard, bool]:
    """The version's contract: the saved one, or one built now (with ``semantics``, else the
    stored-version inference) and written. Returns (contract, written by this call).

    Raises:
        ContractError: building, storing or reading the contract failed.
    """
    path = contract_path(data_dir, upload_id, dataset_version)
    if path.exists():
        return read_saved_contract(path), False
    patches: list[Patch] = []
    if overlay_dir is not None:
        from clinical_analytics.core.overlay_store import OverlayStore

        patches = OverlayStore(base_dir=overlay_dir).load_patches(upload_id, dataset_version)
    accepted = accepted_patch_ids(patches)
    version_semantics = (semantics or stored_semantics(df, metadata)).model_copy(
        update={"dataset_version": dataset_version}
    )
    inputs = ContractInputs.from_semantics(
        version_semantics,
        upload_id=upload_id,
        dataset_version=dataset_version,
        dataset_name=str(metadata.get("dataset_name") or upload_id),
        source_file_sha256=metadata.get("source_file_sha256"),
        resolved=_resolve(df, patches, version_semantics) if accepted else None,
        accepted_patch_ids=accepted,
        extra_custom_properties=population_estimates_properties(metadata),
    )
    contract = build_contract(df, inputs)
    if not write_contract(data_dir, upload_id, dataset_version, contract):
        return read_saved_contract(path), False  # another writer won; its contract is the version's
    return contract, True


def check_upload(
    data_dir: Path,
    df: pl.DataFrame,
    *,
    upload_id: str,
    dataset_version: str,
    metadata: dict[str, Any],
    overlay_dir: Path | None = None,
    run_at: datetime | None = None,
    semantics: DatasetSemantics | None = None,
) -> QualityReport:
    """Write the version's contract if it has none, check ``df`` against it, store the report.

    Column types are checked against the previous version's contract (from ``version_history``
    in ``metadata``), so a column that changed type between versions fails the check.

    Running twice is safe: the contract is reused (immutable per version) and each run adds a
    report file.

    Raises:
        ContractError: building, storing or compiling the contract failed.
    """
    log = logger.bind(upload_id=upload_id, dataset_version=dataset_version)
    contract, _ = write_version_contract(
        data_dir,
        df,
        upload_id=upload_id,
        dataset_version=dataset_version,
        metadata=metadata,
        semantics=semantics,
        overlay_dir=overlay_dir,
    )
    previous = _previous_contract(data_dir, upload_id, metadata, dataset_version)
    report = run_quality(df.lazy(), contract, run_at=run_at, data_version=dataset_version, previous_contract=previous)
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
    semantics: DatasetSemantics | None = None,
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
            semantics=semantics,
        )
    except ContractDamagedError as e:
        logger.error(
            "upload_check_contract_damaged", upload_id=upload_id, dataset_version=dataset_version, **error_fields(e)
        )
        return QualityNotRun(reason=DAMAGED_CONTRACT_REASON)
    except (ContractError, OSError, ValueError, pl.exceptions.PolarsError) as e:
        logger.error("upload_check_failed", upload_id=upload_id, dataset_version=dataset_version, **error_fields(e))
        return QualityNotRun(reason=f"the check failed ({type(e).__name__})")

# SPDX-License-Identifier: MIT
"""Questions across several uploaded public aggregate tables (ADR014 §10).

POST /api/aggregate/questions
    {"dataset_ids": ["gbd_hcv", "wonder_hcv"], "question": "...", "choices": {...}}

Each dataset is read at its active version. Every one must be declared public population
estimates for that version, and detected as an estimate table, or the question is refused. The
answer is synchronous: the tables are small public extracts, read once each.

A held answer lists ``options``; the client re-asks with ``choices`` set to the confirmed values
({dataset_id: {column: level}}). Confirmed choices on an answered question are saved per dataset
version, so the same question is not held twice, except a year or season pairing, which holds for
that request only.
"""

from __future__ import annotations

from typing import Any, Literal

import polars as pl
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.contracts.declarations import DimensionChoice, FileDeclarationStore
from clinical_analytics.contracts.errors import ContractStoreError, InvalidDeclarationKeyError
from clinical_analytics.contracts.holds import Hold, held_message, open_question_holds
from clinical_analytics.contracts.readings import apply_declared, declared_reads
from clinical_analytics.contracts.semantics import DatasetSemantics, with_declarations
from clinical_analytics.core import estimate_join, estimates
from clinical_analytics.core.data_home import uploads_dir
from clinical_analytics.storage.dataset_deletion import InvalidDatasetIdError, validate_dataset_id
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage, active_version_id

logger = structlog.get_logger()
router = APIRouter()

MAX_TABLES = 6
MAX_LEVELS_LISTED = 60

ChoiceValue = str | int | float
CHOICES_NOT_SAVED = "Your choices for {datasets} were used but couldn't be saved; you may be asked again."


class AggregateQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_ids: list[str] = Field(min_length=1, max_length=MAX_TABLES, description="Uploaded tables to join")
    question: str = Field(min_length=1, max_length=2000)
    choices: dict[str, dict[str, ChoiceValue]] = Field(
        default_factory=dict, description="Confirmed hold options: {dataset_id: {column: level}}"
    )


class HoldOptionModel(BaseModel):
    dataset_id: str
    dataset_name: str
    column: str
    default: ChoiceValue | None
    levels: list[ChoiceValue | None]


class AggregateAnswerResponse(BaseModel):
    status: Literal["answered", "held", "refused"]
    message: str
    table: dict[str, Any] | None = None
    method: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    options: list[HoldOptionModel] = Field(default_factory=list)


def _declaration_store() -> FileDeclarationStore:
    return FileDeclarationStore(uploads_dir().parent)


def _version_semantics(
    storage: UserDatasetStorage, dataset_id: str, version: str | None, metadata: dict[str, Any], frame: pl.DataFrame
) -> DatasetSemantics | None:
    """The version's semantics as inferred (``version_semantics``: the contract's, or the same pass
    over the stored frame for a version stored before ADR017); None without a version.

    Raises:
        HTTPException: 500 when the contract can't be read: a patient file must never be read as
            estimates because its contract could not be checked.
    """
    from clinical_analytics.contracts.declarations import MemoryDeclarationStore
    from clinical_analytics.contracts.errors import ContractError
    from clinical_analytics.contracts.service import version_semantics

    if version is None:
        return None
    try:
        return version_semantics(
            storage.upload_dir.parent,
            dataset_id,
            version,
            metadata=metadata,
            stored_frame=lambda: frame,
            declarations=MemoryDeclarationStore(),
        )
    except ContractError as e:
        logger.error("aggregate_semantics_unreadable", dataset_id=dataset_id, error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="This dataset's data contract can't be read.") from e


def load_source(
    storage: UserDatasetStorage, store: FileDeclarationStore, dataset_id: str
) -> tuple[estimate_join.SourceInput, list[Hold]]:
    """One uploaded table at its active version, with its declaration and saved choices, and the
    open questions any answer from it waits on (read with the same declarations as its frame).

    Raises:
        HTTPException: 400 for a malformed id, 404 for an unknown dataset, 500 when its data or
            declarations can't be read.
    """
    try:
        validate_dataset_id(dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    metadata = storage.get_upload_metadata(dataset_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    version = active_version_id(metadata)
    try:
        declarations = store.load(dataset_id)
        # The stored data as every reader sees it: confirmed "not recorded" values and date
        # readings applied (contracts.readings), never the raw Parquet values.
        raw = storage.stored_dataset(dataset_id, metadata).scan_cohort().collect()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} has no stored data") from e
    except (ContractStoreError, InvalidDeclarationKeyError) as e:
        logger.error("aggregate_declarations_unreadable", dataset_id=dataset_id, error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="This dataset's declarations can't be read.") from e
    declared = declarations.population_estimates_for(version) is not None
    inferred = _version_semantics(storage, dataset_id, version, metadata, raw)
    settled = with_declarations(inferred, declarations) if inferred is not None else None
    frame = apply_declared(raw, declared_reads(declarations, version, settled))
    # A file whose contract names a patient (ADR017) holds patient rows: never read as estimates,
    # whatever was declared. Detection runs only on declared tables without a patient key.
    names_patient = declared and inferred is not None and inferred.patient_key is not None
    table = estimates.detect_frame(frame) if declared and not names_patient else None
    name = str(metadata.get("dataset_name") or dataset_id)
    holds = open_question_holds(settled, columns=None, counts_lines=False)
    return estimate_join.SourceInput(
        dataset_id=dataset_id,
        name=name,
        frame=frame,
        table=table,
        declared=declared,
        saved_choices=declarations.choices_for(version),
        version=version,
    ), holds


def _check_choices(sources: list[estimate_join.SourceInput], choices: dict[str, dict[str, ChoiceValue]]) -> None:
    """Reject a choice that names no dataset of the question, no column, or no published level.

    Raises:
        HTTPException: 400 naming the bad choice.
    """
    by_id = {s.dataset_id: s for s in sources}
    for dataset_id, own in choices.items():
        source = by_id.get(dataset_id)
        if source is None:
            raise HTTPException(status_code=400, detail=f"Choice for {dataset_id}, which this question doesn't use")
        problem = estimate_join.choice_problem(source, own)
        if problem is not None:
            raise HTTPException(status_code=400, detail=problem)


def _save_choices(
    store: FileDeclarationStore, versions: dict[str, str | None], choices: dict[str, dict[str, ChoiceValue]]
) -> list[str]:
    """Save confirmed choices per dataset version; returns the datasets whose choices weren't
    saved. The answer still stands (it applied the choices explicitly); the caller says so."""
    failed: list[str] = []
    for dataset_id, own in choices.items():
        version = versions.get(dataset_id)  # the version answered, not whatever is active now

        def change(declarations: Any, own: dict[str, ChoiceValue] = own, version: str | None = version) -> Any:
            for column, value in own.items():
                declarations = declarations.with_choice(
                    DimensionChoice(column=column, value=value, dataset_version=version)
                )
            return declarations

        try:
            store.update(dataset_id, change)
        except ContractStoreError as e:
            logger.error("aggregate_choice_not_saved", dataset_id=dataset_id, error_type=type(e).__name__)
            failed.append(dataset_id)
            continue
        logger.info("aggregate_choice_saved", dataset_id=dataset_id, columns=sorted(own))
    return failed


@router.post("/aggregate/questions", response_model=AggregateAnswerResponse)
def ask_aggregate(request: AggregateQuestionRequest) -> AggregateAnswerResponse:
    """Answer, hold or refuse a question across declared public aggregate tables."""
    if len(set(request.dataset_ids)) != len(request.dataset_ids):
        raise HTTPException(status_code=400, detail="A dataset is listed twice.")
    log = logger.bind(tables=len(request.dataset_ids), question_length=len(request.question))
    storage = UserDatasetStorage()
    store = _declaration_store()
    loaded = [load_source(storage, store, dataset_id) for dataset_id in request.dataset_ids]
    sources = [source for source, _ in loaded]
    # Only tables that can be answered from wait on their questions; an undeclared table is refused.
    holds = [hold for source, found in loaded if source.declared for hold in found]
    if holds:
        # An open question of any table holds the answer until it is answered (contracts.holds).
        log.info("aggregate_question_held_for_questions", questions=len(holds))
        return AggregateAnswerResponse(status="held", message=held_message(holds))
    _check_choices(sources, request.choices)
    result = estimate_join.answer(sources, request.question, request.choices)
    savable = {
        s.dataset_id: kept
        for s in sources
        if (kept := estimate_join.savable_choices(s, request.choices.get(s.dataset_id, {})))
    }
    if result.status == "answered" and savable:
        unsaved = _save_choices(store, {s.dataset_id: s.version for s in sources}, savable)
        if unsaved:
            result.notes.append(CHOICES_NOT_SAVED.format(datasets=", ".join(unsaved)))
    log.info("aggregate_question_done", status=result.status, choices=len(request.choices))
    return AggregateAnswerResponse(
        status=result.status,
        message=result.message,
        table=result.table,
        method=result.method,
        notes=result.notes,
        options=[
            HoldOptionModel(
                dataset_id=o.dataset_id,
                dataset_name=o.dataset_name,
                column=o.column,
                default=o.default,
                levels=list(o.levels[:MAX_LEVELS_LISTED]),
            )
            for o in result.options
        ],
    )

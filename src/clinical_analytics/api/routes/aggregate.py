# SPDX-License-Identifier: MIT
"""Questions across several uploaded public aggregate tables (ADR014 §10).

POST /api/aggregate/questions
    {"dataset_ids": ["gbd_hcv", "wonder_hcv"], "question": "...", "choices": {...}}

Each dataset is read at its active version. Every one must be declared public population
estimates for that version, and detected as an estimate table, or the question is refused. The
answer is synchronous: the tables are small public extracts, read once each.

A held answer lists ``options``; the client re-asks with ``choices`` set to the confirmed values
({dataset_id: {column: level}}). Confirmed choices on an answered question are saved per dataset
version, so the same question is not held twice.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.contracts.declarations import DimensionChoice, FileDeclarationStore
from clinical_analytics.contracts.errors import ContractStoreError, InvalidDeclarationKeyError
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


def load_source(storage: UserDatasetStorage, store: FileDeclarationStore, dataset_id: str) -> estimate_join.SourceInput:
    """One uploaded table at its active version, with its declaration and saved choices.

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
        frame = storage.stored_dataset(dataset_id, metadata).scan_cohort().collect()
        declarations = store.load(dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} has no stored data") from e
    except (ContractStoreError, InvalidDeclarationKeyError) as e:
        logger.error("aggregate_declarations_unreadable", dataset_id=dataset_id, error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="This dataset's declarations can't be read.") from e
    declared = declarations.population_estimates_for(version) is not None
    # Detection runs only on declared tables: an undeclared table is refused before its rows matter.
    table = estimates.detect_frame(frame) if declared else None
    name = str(metadata.get("dataset_name") or dataset_id)
    return estimate_join.SourceInput(
        dataset_id=dataset_id,
        name=name,
        frame=frame,
        table=table,
        declared=declared,
        saved_choices=declarations.choices_for(version),
    )


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
    store: FileDeclarationStore, storage: UserDatasetStorage, choices: dict[str, dict[str, ChoiceValue]]
) -> list[str]:
    """Save confirmed choices per dataset version; returns the datasets whose choices weren't
    saved. The answer still stands (it applied the choices explicitly); the caller says so."""
    failed: list[str] = []
    for dataset_id, own in choices.items():
        version = active_version_id(storage.get_upload_metadata(dataset_id))

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
    sources = [load_source(storage, store, dataset_id) for dataset_id in request.dataset_ids]
    _check_choices(sources, request.choices)
    result = estimate_join.answer(sources, request.question, request.choices)
    if result.status == "answered" and request.choices:
        unsaved = _save_choices(store, storage, request.choices)
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

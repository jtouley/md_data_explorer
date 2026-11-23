# SPDX-License-Identifier: MIT
"""The open questions about an uploaded file, as the app's "Questions about your file" panel shows
them (ADR017): each in clinical words, with its evidence as counts only (never a row value), and
one plain answer per option. Each answer names the existing route that records it and the body to
send, so the panel stays generic over every question kind; what each answer changes is said in one
short note.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.contracts.semantics import DatasetSemantics, OpenQuestion

ANSWER_ROUTES = ("/answers", "/missing-codes", "/date-readings", "/one-row-per-patient")


class FileQuestionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    route: str = Field(description="Posted to /api/datasets/{id}<route>: one of ANSWER_ROUTES")
    body: dict[str, Any]
    changes: str = Field(description="What this answer changes, in one short sentence")


class FileQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    column: str | None
    question: str
    evidence: str
    answers: list[FileQuestionAnswer]


def _lines(n: int) -> str:
    return f"{n:,} line" + ("" if n == 1 else "s")


def _equal_count(frame: pl.LazyFrame, column: str, value: object) -> int:
    """Lines whose ``column`` holds ``value`` (compared as numbers when both are numbers)."""
    dtype = frame.collect_schema().get(column)
    if dtype is None:
        return 0
    try:
        number = float(str(value))
    except ValueError:
        number = None
    target = pl.col(column)
    matched = (
        target.cast(pl.Float64, strict=False) == number
        if number is not None and dtype.is_numeric()
        else target.cast(pl.String) == str(value)
    )
    return int(frame.select(matched.sum()).collect().item() or 0)


def _filled(frame: pl.LazyFrame, column: str) -> int:
    if column not in frame.collect_schema():
        return 0
    return int(frame.select(pl.col(column).is_not_null().sum()).collect().item() or 0)


def file_questions(
    semantics: DatasetSemantics, frame: pl.LazyFrame, patient_rows_token: Callable[[], str | None]
) -> list[FileQuestion]:
    """Every unanswered question of ``semantics``, ready for the panel. ``frame`` is the stored
    data (only counted); ``patient_rows_token`` gives the confirmation token "Is each line one
    patient?" is answered with (None: it can't be answered on this version)."""
    found = []
    for q in semantics.open_questions:
        if q.answered:
            continue
        view = _view(q, semantics, frame, patient_rows_token)
        if view is not None:
            found.append(view)
    return found


def _view(
    q: OpenQuestion, semantics: DatasetSemantics, frame: pl.LazyFrame, token: Callable[[], str | None]
) -> FileQuestion | None:
    column = q.column or ""

    def generic(changes: tuple[str, ...]) -> list[FileQuestionAnswer]:
        return [
            FileQuestionAnswer(label=o, route="/answers", body={"question_id": q.id, "answer": o}, changes=c)
            for o, c in zip(q.options, changes, strict=False)
        ]

    if q.kind == "missing_code":
        evidence = f"{_lines(_equal_count(frame, column, q.value))} of {column} hold {q.value}."
        answers = [
            FileQuestionAnswer(
                label=o,
                route="/missing-codes",
                body={"column": column, "value": q.value, "not_recorded": i == 0},
                changes=(
                    f"Answers leave {q.value} out of {column}, as not recorded (the file keeps it)."
                    if i == 0
                    else f"Answers count {q.value} in {column} as a real value."
                ),
            )
            for i, o in enumerate(q.options)
        ]
    elif q.kind in ("date_order", "date_century"):
        evidence = f"{_lines(_filled(frame, column))} of {column} hold a date written this way."
        codes = ("day_first", "month_first") if q.kind == "date_order" else ("1900s", "2000s")
        answers = [
            FileQuestionAnswer(
                label=o,
                route="/date-readings",
                body={"column": column, "answer": code},
                changes=f"Every reader reads {column} as dates this way; until then it stays text.",
            )
            for o, code in zip(q.options, codes, strict=False)
        ]
    elif q.kind in ("rows_are_patients", "aggregate_unsure"):
        confirmation = token()
        if confirmation is None:
            return None
        evidence = f"The file has {_lines(semantics.rows)} and no column named as a patient ID."
        if q.kind == "rows_are_patients":
            answers = [
                FileQuestionAnswer(
                    label=q.options[0],
                    route="/one-row-per-patient",
                    body={"confirmation": confirmation, "one_row_per_patient": True},
                    changes="Counts count patients, one per line.",
                ),
                FileQuestionAnswer(
                    label=q.options[1],
                    route="/one-row-per-patient",
                    body={"confirmation": confirmation, "one_row_per_patient": False},
                    changes="Counts count lines (visits, tests or records), never patients.",
                ),
            ]
        else:
            answers = [
                FileQuestionAnswer(
                    label=q.options[0],
                    route="/one-row-per-patient",
                    body={"confirmation": confirmation, "one_row_per_patient": False, "counts_cases_in": column},
                    changes=f"Counts add up {column}; lines are never counted as patients.",
                ),
                FileQuestionAnswer(
                    label=q.options[1],
                    route="/one-row-per-patient",
                    body={"confirmation": confirmation, "one_row_per_patient": True},
                    changes="Each line is one patient; counts count lines.",
                ),
            ]
    elif q.kind == "total_row":
        evidence = f"{_lines(_equal_count(frame, column, q.value))} of {column} say {q.value}."
        answers = generic(
            (f"{q.value} lines are left out of every sum.", f"{q.value} lines are added like every other group.")
        )
    elif q.kind == "repeated_strata":
        repeated = semantics.aggregate.repeated_strata_lines if semantics.aggregate else 0
        evidence = f"{_lines(repeated)} repeat another line's groups."
        answers = generic(("Their counts are added together.", "Answers stay on hold; the file needs fixing first."))
    elif q.kind == "censored_value":
        evidence = f"{_lines(_equal_count(frame, column, q.value))} of {column} hold {q.value}."
        answers = generic((f"Answers use {q.value} as given.", f"Answers leave the {q.value} lines of {column} out."))
    elif q.kind == "person_role":
        evidence = f"The file has {_lines(semantics.rows)}; {column} names a baby on each."
        answers = generic(
            ("Counts count babies, one per line.", "Answers stay on hold: counts by mother aren't supported.")
        )
    else:
        raise ValueError(f"No panel view for question kind {q.kind!r}")
    return FileQuestion(id=q.id, kind=q.kind, column=q.column, question=q.question, evidence=evidence, answers=answers)

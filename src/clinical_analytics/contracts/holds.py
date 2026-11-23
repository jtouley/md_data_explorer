# SPDX-License-Identifier: MIT
"""One hold rule for every open question (ADR017).

A fact that would change a number and that only the user can settle is an open question of the
contract's semantics. Until it is answered, any answer that reads what it concerns is held: the
question path, the aggregate path and governed SQL all call ``open_question_holds`` on the
semantics their data was read with (the leased layer's own, never a later store read), and show
the question instead of a number. A plan confirmation never releases it; only an answer does.

What a question concerns:

* a column (``missing_code``, ``date_order``, ``date_century``, ``censored_value``): answers that
  read the column;
* the whole table (``aggregate_unsure``, ``total_row``, ``repeated_strata``): every answer, since
  any number read from a table of groups depends on which lines are groups and which totals;
* what a line is (``rows_are_patients``, ``person_role``): answers that count lines.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Final

import structlog

from clinical_analytics.contracts.semantics import DatasetSemantics, OpenQuestion

logger = structlog.get_logger()

COLUMN_KINDS: Final = frozenset(
    {"missing_code", "date_order", "date_century", "censored_value", "unit_pair", "unit_missing"}
)
TABLE_KINDS: Final = frozenset({"aggregate_unsure", "total_row", "repeated_strata"})
COUNT_KINDS: Final = frozenset({"rows_are_patients", "person_role"})

HELD_MESSAGE: Final = "This answer waits for you to answer {count} about this file first: {questions}"


@dataclass(frozen=True)
class Hold:
    """An open question an answer waits on."""

    question_id: str
    kind: str
    column: str | None
    question: str


def open_question_holds(
    semantics: DatasetSemantics | None, *, columns: Collection[str] | None, counts_lines: bool
) -> list[Hold]:
    """The unanswered questions an answer reading ``columns`` (None: every column) waits on.

    ``counts_lines``: the answer counts lines (a COUNT, a share); "Is each line one patient?"
    then holds it. A reader with its own gate for that question (governed SQL's one-row-per-
    patient confirmation) passes False.
    """
    if semantics is None:
        return []
    found = [
        Hold(q.id, q.kind, q.column, q.question) for q in semantics.open_questions if _holds(q, columns, counts_lines)
    ]
    if found:
        logger.info("answer_held_for_questions", kinds=sorted({h.kind for h in found}), questions=len(found))
    return found


def _holds(question: OpenQuestion, columns: Collection[str] | None, counts_lines: bool) -> bool:
    if question.answered:
        return False
    if question.kind in TABLE_KINDS:
        return True
    if question.kind in COUNT_KINDS:
        return counts_lines
    if question.kind in COLUMN_KINDS:
        return columns is None or question.column in columns
    raise ValueError(f"No hold rule for question kind {question.kind!r}")


def held_message(holds: Collection[Hold]) -> str:
    """The sentence shown in place of a held answer, naming every question it waits on."""
    count = "1 question" if len(holds) == 1 else f"{len(holds)} questions"
    return HELD_MESSAGE.format(count=count, questions=" ".join(h.question for h in holds))

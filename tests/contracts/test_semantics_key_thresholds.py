# SPDX-License-Identifier: MIT
"""Sensitivity of "Which column identifies the patient?" to its three shares (ADR017).

``KEY_MIN_SHARE``, ``KEY_RECORD_SHARE`` and ``KEY_SECOND_PERSON_SHARE`` only choose which option
is pre-selected (the tentative key). Moving each by 0.1 either way must leave the options, the
hold on answers that count patients, and every applied fact (missing codes, dates, labels,
ranges, units) unchanged. Synthetic, invented files: kennel-cough clinic visits, plaster-room
casts with a carer, and a sling register.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest

from clinical_analytics.contracts import config
from clinical_analytics.contracts.holds import open_question_holds
from clinical_analytics.contracts.inference import UploadFacts, infer_semantics

DOC = "temp_c  Temperature (30-45). 999 = not recorded\nsling  Sling type: 1 = broad arm, 2 = collar and cuff\n"


def _extras(n: int) -> dict[str, Any]:
    return {
        "temp_c": [999 if i % 17 == 0 else 36 + i % 4 for i in range(n)],
        "sling": [1 + i % 2 for i in range(n)],
        "seen": [date(2022, 1, 1) + timedelta(days=3 * i) for i in range(n)],
    }


def _few_lines_each() -> pl.DataFrame:
    """30 values over 120 lines: 0.25 values per line, between 0.15 and 0.35."""
    return pl.DataFrame({"kennel_ref": [f"K{i % 30:03d}" for i in range(120)], **_extras(120)})


def _second_person() -> pl.DataFrame:
    """A line-unique cast ID nested in a carer ID; 55 % of carers on one line."""
    carers = [f"C{i:03d}" for i in range(55)] + [f"C{100 + i // 2:03d}" for i in range(90)]
    n = len(carers)
    return pl.DataFrame({"cast_ref": [f"X{i:04d}" for i in range(n)], "carer_ref": carers, **_extras(n)})


def _nearly_records() -> pl.DataFrame:
    """85 % of the register's values sit on one line only."""
    values = [f"S{i:03d}" for i in range(85)] + [f"S{900 + i // 2:03d}" for i in range(30)]
    return pl.DataFrame({"sling_ref": values, **_extras(len(values))})


FILES = {"few_lines_each": _few_lines_each, "second_person": _second_person, "nearly_records": _nearly_records}


def _applied(semantics: Any) -> dict[str, Any]:
    """Everything that may change a number, plus the question's options and the count hold."""
    key_q = semantics.question("patient_key")
    return {
        "columns": {
            name: (
                sorted(str(m.value) for m in col.missing_codes),
                col.date is not None,
                dict(col.labels),
                col.stated_range,
                col.unit,
                sorted(str(c.value) for c in col.nonresponse_codes),
            )
            for name, col in semantics.columns.items()
        },
        "aggregate": semantics.aggregate,
        "questions": sorted(q.id for q in semantics.open_questions),
        "options": dict(zip(key_q.options, key_q.option_counts, strict=False)) if key_q else None,
        "counts_held": bool(open_question_holds(semantics, columns=None, counts_lines=True)),
    }


@pytest.mark.parametrize("file", sorted(FILES))
@pytest.mark.parametrize("share", ["KEY_MIN_SHARE", "KEY_RECORD_SHARE", "KEY_SECOND_PERSON_SHARE"])
@pytest.mark.parametrize("delta", [-0.1, 0.1])
def test_keyShares_movedByATenth_onlyThePreselectedOptionChanges(
    file: str, share: str, delta: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    frame = FILES[file]()
    base = infer_semantics(frame, UploadFacts(doc_context=DOC))

    # Act
    monkeypatch.setattr(config, share, getattr(config, share) + delta)
    moved = infer_semantics(frame, UploadFacts(doc_context=DOC))

    # Assert: the same options, holds and applied facts; any key is only pre-selected (low)
    assert _applied(moved) == _applied(base)
    assert _applied(base)["counts_held"] is True
    for semantics in (base, moved):
        assert semantics.patient_key is None or semantics.patient_key.confidence == "low"


@pytest.mark.parametrize(
    ("file", "share", "low", "high"),
    [
        ("few_lines_each", "KEY_MIN_SHARE", "kennel_ref", None),
        ("second_person", "KEY_SECOND_PERSON_SHARE", "cast_ref", "carer_ref"),
        ("nearly_records", "KEY_RECORD_SHARE", None, "sling_ref"),
    ],
)
def test_keyShares_eachOnItsEdge_choosesThePreselectionOnly(
    file: str, share: str, low: str | None, high: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: each file sits between share - 0.1 and share + 0.1 of the share it tests
    frame = FILES[file]()
    found: list[str | None] = []

    # Act
    for delta in (-0.1, 0.1):
        monkeypatch.setattr(config, share, getattr(config, share) + delta)
        found.append(infer_semantics(frame, UploadFacts(doc_context=DOC)).patient_key_column)
        monkeypatch.undo()

    # Assert: the share decides the pre-selection, and it is the only thing it decides
    assert found == [low, high]

# SPDX-License-Identifier: MIT
"""Edges of ``contracts.readings``, found by mutation testing: a confirmed code on a text column,
totals on two columns at once, and which tables have total rows to leave out. Every reader (the
raw frame, the semantic layer and governed SQL) is checked. Invented hand-clinic tables only.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import polars as pl
import pytest

from clinical_analytics.contracts.declarations import Declarations, QuestionAnswer
from clinical_analytics.contracts.readings import DeclaredReads, declared_reads, missing_expr, total_rows, unit_factors
from clinical_analytics.contracts.semantics import AggregateInfo, DatasetSemantics, Grain, OpenQuestion, TotalLevel

READERS = ("polars", "layer", "sql")


def test_everyReader_textColumnConfirmedCode_blanked(
    read_by_every_reader: Callable[..., dict[str, pl.DataFrame]],
) -> None:
    # Arrange: "NR" in a text column means not recorded
    frame = pl.DataFrame({"hand_ref": ["H1", "H2", "H3", "H4"], "dominance": ["left", "NR", "right", None]})
    reads = DeclaredReads(missing={"dominance": ("NR",)})

    # Act
    read = read_by_every_reader(frame, reads, "hand_ref")

    # Assert
    for name in READERS:
        assert read[name].get_column("dominance").to_list() == ["left", None, "right", None], name


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param(("99",), [1.5, None, 3.0], id="text_code_on_a_number_column"),
        pytest.param((99, "unknown"), [1.5, None, 3.0], id="non_number_code_ignored"),
        pytest.param(("1.5",), [None, 99.0, 3.0], id="decimal_code"),
    ],
)
def test_missingExpr_numberColumn_codesComparedAsNumbers(values: tuple[object, ...], expected: list[float]) -> None:
    # Arrange
    frame = pl.DataFrame({"grip": [1.5, 99.0, 3.0]})

    # Act
    read = frame.select(missing_expr("grip", frame.schema["grip"], values))

    # Assert
    assert read.get_column("grip").to_list() == expected


def test_everyReader_totalsOnTwoColumns_everyTotalLineLeftOut(
    read_by_every_reader: Callable[..., dict[str, pl.DataFrame]],
) -> None:
    # Arrange: splint fittings by clinic and quarter, totals on both (and on a column the file lacks,
    # named first), one line of no clinic
    frame = pl.DataFrame(
        {
            "line_no": [1, 2, 3, 4, 5, 6],
            "clinic": ["East", "West", "All clinics", "East", "All clinics", None],
            "quarter": ["Q1", "Q1", "Q1", "All quarters", "All quarters", "Q2"],
            "fittings": [10, 20, 30, 10, 30, 4],
        }
    )
    reads = DeclaredReads(totals={"absent": ("All",), "clinic": ("All clinics",), "quarter": ("All quarters",)})

    # Act
    read = read_by_every_reader(frame, reads, "line_no")

    # Assert: only the group lines (and the line of no clinic) are read
    for name in READERS:
        assert read[name].get_column("line_no").to_list() == [1, 2, 6], name


def _semantics(kind: str, levels: tuple[TotalLevel, ...]) -> DatasetSemantics:
    grain = Grain(kind="aggregate", noun="group", rows=6, one_row_per_patient=False, source="structure")
    aggregate = AggregateInfo(kind=kind, count_column="fittings", strata=("clinic",), total_levels=levels,
                              source="structure")  # fmt: skip
    return DatasetSemantics(rows=6, grain=grain, aggregate=aggregate)


@pytest.mark.parametrize(
    ("kind", "levels", "expected"),
    [
        pytest.param(
            "frequency",
            (TotalLevel(column="clinic", values=("All clinics",)), TotalLevel(column="quarter", values=())),
            {"clinic": ("All clinics",)},
            id="frequency_levels_with_values",
        ),
        pytest.param("estimate", (TotalLevel(column="clinic", values=("All clinics",)),), {}, id="estimate_keeps"),
    ],
)
def test_totalRows_tableKind_onlyFrequencyTotalsLeftOut(
    kind: str, levels: tuple[TotalLevel, ...], expected: dict[str, tuple[str, ...]]
) -> None:
    # Act / Assert
    assert total_rows(_semantics(kind, levels)) == expected


def test_totalRows_noSemantics_nothingLeftOut() -> None:
    # Act / Assert
    assert total_rows(None) == {}


def test_unitFactors_targetUnit_spellingsReadInIt() -> None:
    # Act
    factors = unit_factors("months")

    # Assert: a year is twelve months, a month one, a week and a day fractions of one
    assert factors["years"] == pytest.approx(12.0)
    assert factors["month"] == pytest.approx(1.0)
    assert 0 < factors["weeks"] < factors["months"]
    assert 0 < factors["days"] < factors["weeks"]


def test_everyReader_skippedAndAbsentColumns_otherReadsStillApplied(
    read_by_every_reader: Callable[..., dict[str, pl.DataFrame]],
) -> None:
    # Arrange: reads naming columns the file lacks (or no values) come first; the grip code after
    frame = pl.DataFrame({"hand_ref": ["H1", "H2", "H3"], "grip": [12, 99, 30], "wait": [2, 1, 3]})
    reads = DeclaredReads(
        missing={"pinch": (9,), "wait": (), "grip": (99,)},
        units={"wait": ("wait_unit", "months")},
    )

    # Act
    read = read_by_every_reader(frame, reads, "hand_ref")

    # Assert
    for name in READERS:
        assert read[name].get_column("grip").to_list() == [12, None, 30], name
        assert read[name].get_column("wait").to_list() == [2, 1, 3], name


@pytest.mark.parametrize(
    ("values", "reading", "expected"),
    [
        pytest.param(["03/04/2021", "25/12/2020"], ("%d/%m/%Y", None), [date(2021, 4, 3), date(2020, 12, 25)],
                     id="day_first"),
        pytest.param(["03/04/21", "25/12/98"], ("%d/%m/%y", "1900s"), [date(1921, 4, 3), date(1998, 12, 25)],
                     id="two_digit_years_in_the_1900s"),
        pytest.param(["03/04/21", "25/12/98"], ("%d/%m/%y", "2000s"), [date(2021, 4, 3), date(2098, 12, 25)],
                     id="two_digit_years_in_the_2000s"),
    ],
)  # fmt: skip
def test_everyReader_confirmedDateReading_appliedByAllThree(
    values: list[str],
    reading: tuple[str, str | None],
    expected: list[date],
    read_by_every_reader: Callable[..., dict[str, pl.DataFrame]],
) -> None:
    # Arrange: splint fitting dates, read as the user confirmed (a column the file lacks is skipped)
    frame = pl.DataFrame({"hand_ref": ["H1", "H2"], "fitted_on": values})
    reads = DeclaredReads(dates={"removed_on": ("%Y-%m-%d", None), "fitted_on": reading})

    # Act
    read = read_by_every_reader(frame, reads, "hand_ref")

    # Assert
    for name in READERS:
        assert read[name].get_column("fitted_on").cast(pl.Date).to_list() == expected, name


def test_everyReader_mixedTimeUnits_readInTheOneUnit(
    read_by_every_reader: Callable[..., dict[str, pl.DataFrame]],
) -> None:
    # Arrange: time to fitting in mixed units, read in months; a line with no known unit is missing
    frame = pl.DataFrame(
        {
            "hand_ref": ["H1", "H2", "H3", "H4"],
            "wait": [2, 1, 3, 5],
            "wait_unit": [" Years", "months", "fortnights", None],
        }
    )
    reads = DeclaredReads(units={"wait": ("wait_unit", "months")})

    # Act
    read = read_by_every_reader(frame, reads, "hand_ref")

    # Assert
    for name in READERS:
        assert read[name].get_column("wait").to_list() == pytest.approx([24.0, 1.0, None, None], nan_ok=True), name


def _censored(answered_on: str | None, answer: str) -> tuple[Declarations, DatasetSemantics]:
    question = OpenQuestion(
        id="censored_value:pinch:30",
        kind="censored_value",
        question="In pinch, 30 means 30 kg or more. Is that right for your questions?",
        column="pinch",
        value=30,
        options=("yes, use 30 as given", "no, leave those lines out"),
    )
    grain = Grain(kind="record", noun="visit", rows=3, one_row_per_patient=False, source="default")
    semantics = DatasetSemantics(rows=3, grain=grain, open_questions=(question,), dataset_version="v1")
    answers = () if answered_on is None else (QuestionAnswer(question_id=question.id, answer=answer,
                                                              dataset_version=answered_on),)  # fmt: skip
    return Declarations(answers=answers), semantics


@pytest.mark.parametrize(
    ("answered_on", "answer", "missing"),
    [
        pytest.param("v1", "no, leave those lines out", {"pinch": (30,)}, id="left_out"),
        pytest.param("v1", "yes, use 30 as given", {}, id="used_as_given"),
        pytest.param("v2", "no, leave those lines out", {}, id="answered_on_another_version"),
        pytest.param(None, "", {}, id="unanswered"),
    ],
)
def test_declaredReads_censoredValue_leftOutOnlyWhenTheUserSaysSo(
    answered_on: str | None, answer: str, missing: dict[str, tuple[int, ...]]
) -> None:
    # Arrange
    declarations, semantics = _censored(answered_on, answer)

    # Act
    reads = declared_reads(declarations, "v1", semantics)

    # Assert
    assert reads.missing == missing

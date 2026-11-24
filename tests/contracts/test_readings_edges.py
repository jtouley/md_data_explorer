# SPDX-License-Identifier: MIT
"""Edges of ``contracts.readings``, found by mutation testing: a confirmed code on a text column,
totals on two columns at once, and which tables have total rows to leave out. Every reader (the
raw frame, the semantic layer and governed SQL) is checked. Invented hand-clinic tables only.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from clinical_analytics.contracts import config
from clinical_analytics.contracts.declarations import DateReadingDeclaration, Declarations, QuestionAnswer
from clinical_analytics.contracts.readings import (
    DeclaredReads,
    century_for,
    declared_reads,
    missing_expr,
    total_rows,
    unit_factors,
)
from clinical_analytics.contracts.semantics import (
    AggregateInfo,
    ColumnSemantics,
    DatasetSemantics,
    Grain,
    OpenQuestion,
    TotalLevel,
    Unit,
)

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
        pytest.param(["03/04/00", "25/12/00"], ("%d/%m/%y", "1900s"), [date(1900, 4, 3), date(1900, 12, 25)],
                     id="year_00_in_the_1900s"),
        pytest.param(["03/04/00", "25/12/00"], ("%d/%m/%y", "2000s"), [date(2000, 4, 3), date(2000, 12, 25)],
                     id="year_00_in_the_2000s"),
        pytest.param(["03/04/69", "25/12/70"], ("%d/%m/%y", None), [date(2069, 4, 3), date(1970, 12, 25)],
                     id="two_digit_years_no_century_stated_default"),
        pytest.param(["03/04/69", "25/12/70"], ("%d/%m/%y", "pivot"), [date(2069, 4, 3), date(1970, 12, 25)],
                     id="two_digit_years_stated_default"),
        pytest.param(["03/04/69", "25/12/70"], ("%d/%m/%y", "1900s"), [date(1969, 4, 3), date(1970, 12, 25)],
                     id="year_69_in_the_1900s"),
        pytest.param(["03/04/2021", "  "], ("%d/%m/%Y", None), [date(2021, 4, 3), None], id="blank_is_missing"),
        pytest.param(["03/04/2021", "not known"], ("%d/%m/%Y", None), [date(2021, 4, 3), None],
                     id="unreadable_is_missing"),
    ],
)  # fmt: skip
def test_everyReader_confirmedDateReading_appliedByAllThree(
    values: list[str],
    reading: tuple[str, str | None],
    expected: list[date | None],
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


def _in_century(two_digits: int, century: str | None) -> int:
    if century == "1900s":
        return 1900 + two_digits
    if century == "2000s":
        return 2000 + two_digits
    return (2000 if two_digits < config.TWO_DIGIT_YEAR_PIVOT else 1900) + two_digits


@pytest.mark.parametrize("century", [None, "pivot", "1900s", "2000s"])
@pytest.mark.parametrize("choice_format", ["%d/%m/%y", "%y-%m-%d"])
def test_everyReader_everyTwoDigitYear_allThreeReadTheSameCentury(
    century: str | None, choice_format: str, read_by_every_reader: Callable[..., dict[str, pl.DataFrame]]
) -> None:
    # Arrange: one cast-check date for each two-digit year (polars reads 69 as 2069, DuckDB as 1969)
    years = list(range(100))
    written = [date(2001, 1 + y % 12, 1 + y % 28).replace(year=2000 + y).strftime(choice_format) for y in years]
    frame = pl.DataFrame({"hand_ref": [f"H{y:02d}" for y in years], "checked_on": written})
    reads = DeclaredReads(dates={"checked_on": (choice_format, century)})

    # Act
    read = read_by_every_reader(frame, reads, "hand_ref")

    # Assert: no reader is left to its engine's own century
    expected = [date(_in_century(y, century), 1 + y % 12, 1 + y % 28) for y in years]
    for name in READERS:
        assert read[name].get_column("checked_on").cast(pl.Date).to_list() == expected, name


@pytest.mark.parametrize(
    ("century", "expected_years"),
    [
        pytest.param(None, [1950, 1969, 1970, 1999, 2000, 2069, 2070, 2085], id="no_century_years_kept"),
        pytest.param("pivot", [2050, 2069, 1970, 1999, 2000, 2069, 1970, 1985], id="stated_default"),
        pytest.param("1900s", [1950, 1969, 1970, 1999, 1900, 1969, 1970, 1985], id="in_the_1900s"),
        pytest.param("2000s", [2050, 2069, 2070, 2099, 2000, 2069, 2070, 2085], id="in_the_2000s"),
    ],
)
def test_everyReader_centuryOnFourDigitYears_eachReaderSetsTheSameCentury(
    century: str | None, expected_years: list[int], read_by_every_reader: Callable[..., dict[str, pl.DataFrame]]
) -> None:
    # Arrange: years written in full, so the engine's own two-digit pivot plays no part and only
    # the century rule moves a year
    years = [1950, 1969, 1970, 1999, 2000, 2069, 2070, 2085]
    frame = pl.DataFrame({"hand_ref": [f"H{i}" for i in range(len(years))], "cast_on": [f"{y}-06-15" for y in years]})
    reads = DeclaredReads(dates={"cast_on": ("%Y-%m-%d", century)})

    # Act
    read = read_by_every_reader(frame, reads, "hand_ref")

    # Assert
    expected = [date(y, 6, 15) for y in expected_years]
    for name in READERS:
        assert read[name].get_column("cast_on").cast(pl.Date).to_list() == expected, name


@pytest.mark.parametrize(
    ("choice_format", "century", "expected"),
    [
        pytest.param("%d/%m/%y", None, config.TWO_DIGIT_YEAR_DEFAULT, id="two_digit_year_gets_the_default"),
        pytest.param("%d/%m/%y", "1900s", "1900s", id="answered_century_kept"),
        pytest.param("%d/%m/%Y", None, None, id="four_digit_year_needs_none"),
    ],
)
def test_dateReadingDeclaration_twoDigitYear_alwaysCarriesACentury(
    choice_format: str, century: str | None, expected: str | None
) -> None:
    # Act
    declared = DateReadingDeclaration(column="cast_on", format=choice_format, century=century)

    # Assert: the declaration and the readers' own fallback agree
    assert declared.century == expected
    assert century_for(choice_format, century) == expected


def test_dateReadingDeclaration_storedWithoutCentury_loadsWithTheDefault() -> None:
    # Arrange: a reading saved before a two-digit year always carried a century
    stored = '{"column": "cast_on", "format": "%m/%d/%y", "century": null, "dataset_version": "v1"}'

    # Act
    loaded = Declarations.model_validate_json(f'{{"date_readings": [{stored}]}}').date_readings_for("v1")

    # Assert
    assert [d.century for d in loaded] == [config.TWO_DIGIT_YEAR_DEFAULT]


def test_readByEveryReader_repeatedCalls_leaveNoFileOrOpenConnection(
    read_by_every_reader: Callable[..., dict[str, pl.DataFrame]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: count the layer's DuckDB connections opened and closed
    import ibis

    opened, closed = [], []
    connect = ibis.duckdb.connect

    def _counted() -> Any:
        backend = connect()
        opened.append(backend)
        disconnect = backend.disconnect
        monkeypatch.setattr(backend, "disconnect", lambda: (closed.append(backend), disconnect())[1])
        return backend

    monkeypatch.setattr(ibis.duckdb, "connect", _counted)
    frame = pl.DataFrame({"hand_ref": ["H1", "H2"], "dominance": ["left", "NR"]})

    # Act
    for _ in range(3):
        read_by_every_reader(frame, DeclaredReads(missing={"dominance": ("NR",)}), "hand_ref")

    # Assert
    assert len(opened) == 3 and closed == opened
    assert list(tmp_path.glob("stored_*.parquet")) == []


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


def test_declaredReads_unitColumns_onlyATimeUnitWithItsUnitColumnIsRead() -> None:
    # Arrange: a wait with its unit column (months), a wait with no unit column, a dose whose unit
    # column holds a unit that is no time
    grain = Grain(kind="record", noun="visit", rows=3, one_row_per_patient=False, source="default")
    columns = {
        "wait": ColumnSemantics(logical_type="integer", unit=Unit(value="months", source="structure"),
                                unit_column="wait_unit"),
        "rest": ColumnSemantics(logical_type="integer", unit=Unit(value="days", source="dictionary")),
        "dose": ColumnSemantics(logical_type="integer", unit=Unit(value="mg", source="dictionary"),
                                unit_column="dose_unit"),
    }  # fmt: skip
    semantics = DatasetSemantics(rows=3, grain=grain, columns=columns)

    # Act
    reads = declared_reads(Declarations(), None, semantics)

    # Assert
    assert reads.units == {"wait": ("wait_unit", "months")}

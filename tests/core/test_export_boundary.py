"""The export boundary (ADR015): counts of 1-10 become "<11" only when an answer leaves the machine."""

from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from clinical_analytics.core.export_boundary import (
    EXPORT_MARKER,
    EXPORT_NOTE,
    export_answer,
    export_text,
    is_small_count,
)
from clinical_analytics.core.result_presentation import present_analysis


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, False), (1, True), (10, True), (11, False), (-3, False), (True, False), (5.0, False), (None, False)],
)
def test_isSmallCount_boundaryValues_onlyWholeNumbersOneToTen(value: Any, expected: bool) -> None:
    # Arrange: the value (parametrized: zero, edges, bool, float, None)

    # Act
    result = is_small_count(value)

    # Assert
    assert result is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Total count: 7 patients.", f"Total count: {EXPORT_MARKER} patients."),
        ("n=10; 1 excluded", f"n={EXPORT_MARKER}; {EXPORT_MARKER} excluded"),
        ("Most common ward: ICU (3.0 patients).", f"Most common ward: ICU ({EXPORT_MARKER} patients)."),
        ("n=11 patients", "n=11 patients"),
        ("n=0", "n=0"),
        ("p=0.004; r=0.71", "p=0.004; r=0.71"),
        ("47.1% of patients", "47.1% of patients"),
        ("5% missing", "5% missing"),
        ("A1c and 2024-01-05", "A1c and 2024-01-05"),
        ("1,234 rows", "1,234 rows"),
        ("", ""),
    ],
)
def test_exportText_numbersInASentence_onlyStandaloneOneToTenMasked(text: str, expected: str) -> None:
    # Arrange: the sentence (parametrized)

    # Act
    exported = export_text(text)

    # Assert
    assert exported == expected


def test_exportAnswer_groupedCount_smallCellsMaskedOthersKept() -> None:
    # Arrange: what the app shows on screen, small groups included
    analysis = {
        "type": "count",
        "grouped_by": "ward",
        "total_count": 303,
        "group_counts": {"A": 200, "B": 100, "ICU": 3},
    }
    shown = present_analysis(analysis)
    before = copy.deepcopy(shown)

    # Act
    exported = export_answer(shown)

    # Assert
    assert shown == before  # the on-screen answer is never changed
    assert shown["table"] is not None and {"ward": "ICU", "count": 3} in shown["table"]["rows"]
    assert exported["table"] is not None
    assert {"ward": "ICU", "count": EXPORT_MARKER} in exported["table"]["rows"]
    assert {"ward": "A", "count": 200} in exported["table"]["rows"]
    # The claim's "across 3 ward groups" is masked too: prose numbers can't be told from counts.
    assert exported["claim"] == f"303 total across {EXPORT_MARKER} ward groups."
    assert exported["masked"] == 2
    assert exported["note"] == EXPORT_NOTE


@pytest.mark.parametrize("presentation", [None, {"claim": None, "table": None}, {"claim": "n=12", "table": None}])
def test_exportAnswer_nothingSmall_noNoteAndNothingMasked(presentation: dict[str, Any] | None) -> None:
    # Arrange: nothing to mask (parametrized: none, empty, a large count)

    # Act
    exported = export_answer(presentation)

    # Assert
    assert exported["masked"] == 0
    assert exported["note"] is None


def test_exportAnswer_floatsAndBooleansInTable_leftAsTheyAre() -> None:
    # Arrange: a rate and a flag are not counts
    presentation = {
        "claim": None,
        "table": {"columns": ["g", "rate", "flag"], "rows": [{"g": "x", "rate": 0.5, "flag": True}]},
    }

    # Act
    exported = export_answer(presentation)

    # Assert
    assert exported["table"] == presentation["table"]
    assert exported["masked"] == 0


def test_exportAnswer_crosstabLabelsAndNestedCells_maskedConsistently() -> None:
    # Arrange: a header that is a level label with a digit, and a nested cell
    presentation = {
        "claim": "Type 2 by sex",
        "table": {"columns": ["level", "Type 2"], "rows": [{"level": "x", "Type 2": 4, "extra": {"k": [3, 40]}}]},
    }

    # Act
    exported = export_answer(presentation)

    # Assert: the header and the row key change together, so the table still lines up
    table = exported["table"]
    assert table is not None
    marked = f"Type {EXPORT_MARKER}"
    assert table["columns"] == ["level", marked]
    assert table["rows"][0][marked] == EXPORT_MARKER
    assert table["rows"][0]["extra"] == {"k": [EXPORT_MARKER, 40]}
    assert exported["claim"] == f"{marked} by sex"


@pytest.mark.parametrize("table", ["not a table", {"columns": "x", "rows": []}, {"columns": [], "rows": ["x"]}])
def test_exportAnswer_unrecognisedTable_droppedAndSaidSo(table: Any) -> None:
    # Arrange: a shape the boundary cannot read (parametrized)

    # Act
    exported = export_answer({"claim": "n=3", "table": table})

    # Assert
    assert exported["table"] is None
    assert exported["claim"] == f"n={EXPORT_MARKER}"
    assert exported["note"] is not None and "left out" in exported["note"]


@pytest.mark.parametrize(
    ("table", "column"),
    [
        pytest.param({"columns": ["g", "n"], "rows": [{"g": "a", "n": np.int64(4)}]}, "n", id="numpy_integer"),
        pytest.param({"columns": ["g", "count"], "rows": [{"g": "a", "count": 4.0}]}, "count", id="float_count_column"),
        pytest.param(
            {"columns": ["statistic", "age"], "rows": [{"statistic": "count", "age": 5.0}]},
            "age",
            id="describe_count_row",
        ),
    ],
)
def test_exportAnswer_countsNotPlainInts_stillMasked(table: dict[str, Any], column: str) -> None:
    # Arrange: counts as a numpy integer or a whole-number float (sharp-edges review)

    # Act
    exported = export_answer({"claim": None, "table": table})

    # Assert
    assert exported["table"] is not None
    assert exported["table"]["rows"][0][column] == EXPORT_MARKER


def test_exportAnswer_wholeFloatInStatisticColumn_notMasked() -> None:
    # Arrange: a mean of 5.0 is not a count
    table = {"columns": ["variable", "mean"], "rows": [{"variable": "x", "mean": 5.0}]}

    # Act
    exported = export_answer({"claim": None, "table": table})

    # Assert
    assert exported["table"] == table


def test_exportText_fullWidthDigit_masked() -> None:
    # Arrange / Act
    exported = export_text("n=５ patients")

    # Assert
    assert exported == f"n={EXPORT_MARKER} patients"


@pytest.mark.parametrize("presentation", [["claim"], "Total count: 3.", 7])
def test_exportAnswer_notAMapping_raisesTypeError(presentation: Any) -> None:
    # Arrange: the wrong kind of input (parametrized)

    # Act / Assert
    with pytest.raises(TypeError, match="presentation mapping"):
        export_answer(presentation)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(Decimal("4"), EXPORT_MARKER, id="whole_decimal"),
        pytest.param(Decimal("10.00"), EXPORT_MARKER, id="whole_decimal_with_scale"),
        pytest.param(Decimal("1"), EXPORT_MARKER, id="lower_edge"),
        pytest.param(Decimal("11"), Decimal("11"), id="above_threshold"),
        pytest.param(Decimal("0"), Decimal("0"), id="zero"),
        pytest.param(Decimal("4.5"), Decimal("4.5"), id="fraction"),
        pytest.param(Decimal("-3"), Decimal("-3"), id="negative"),
        pytest.param(Decimal("NaN"), Decimal("NaN"), id="nan"),
        pytest.param(Decimal("Infinity"), Decimal("Infinity"), id="infinity"),
    ],
)
def test_exportAnswer_decimalCell_wholeSmallCountMasked(value: Decimal, expected: Any) -> None:
    # Arrange: DuckDB returns SUMs (and some counts) as Decimal
    table = {"columns": ["group", "total"], "rows": [{"group": "a", "total": value}]}

    # Act
    exported = export_answer({"claim": None, "table": table})

    # Assert
    assert exported["table"] is not None
    cell = exported["table"]["rows"][0]["total"]
    assert str(cell) == str(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("05", EXPORT_MARKER),
        ("007", EXPORT_MARKER),
        ("n=05 patients", f"n={EXPORT_MARKER} patients"),
        ("010", EXPORT_MARKER),
        ("(03)", f"({EXPORT_MARKER})"),
        ("00", "00"),
        ("011", "011"),
        ("0100", "0100"),
        ("2024-01-05", "2024-01-05"),
        ("05/2024", f"{EXPORT_MARKER}/2024"),
        ("v05", "v05"),
    ],
)
def test_exportText_zeroPaddedNumber_maskedUnlessDatePart(text: str, expected: str) -> None:
    # Arrange: a zero-padded count, as a cell string or in a sentence (parametrized)

    # Act
    exported = export_text(text)

    # Assert
    assert exported == expected


def test_exportAnswer_zeroPaddedStringCell_masked() -> None:
    # Arrange: a count column read as text keeps its leading zeros
    table = {"columns": ["ward", "n"], "rows": [{"ward": "ICU", "n": "04"}, {"ward": "ED", "n": "012"}]}

    # Act
    exported = export_answer({"claim": None, "table": table})

    # Assert
    assert exported["table"] is not None
    assert [row["n"] for row in exported["table"]["rows"]] == [EXPORT_MARKER, "012"]
    assert exported["masked"] == 1

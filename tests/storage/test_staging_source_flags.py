# SPDX-License-Identifier: MIT
"""A source's own suppression markers ("Suppressed", "(Unreliable)") and "lo to hi" intervals at
staging (ADR014 §10): numbers stay numbers, a suppressed cell is never zero, and other text
columns are untouched."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.storage.staging import read_csv_typed


def _stage(text: str, public: bool = False) -> pl.DataFrame:
    return read_csv_typed(text.encode(), public=public).data


class TestSourceFlags:
    def test_readCsvTyped_wonderSuppressedAndUnreliable_numbersWithFlags(self):
        # Arrange
        csv = (
            "State,Year,Deaths,Crude Rate\n"
            "Michigan,2010,Suppressed,Suppressed\n"
            "Michigan,2011,25,0.5 (Unreliable)\n"
            "Ohio,2010,120,2.1\n"
            "Ohio,2011,130,Not Applicable\n"
        )

        # Act: declared public (or a WONDER-shaped file): the source's words are read as markers
        frame = _stage(csv, public=True)

        # Assert
        expected = pl.DataFrame(
            {
                "State": ["Michigan", "Michigan", "Ohio", "Ohio"],
                "Year": [2010, 2011, 2010, 2011],
                "Deaths": [None, 25, 120, 130],
                "Deaths suppressed": [True, False, False, False],
                "Crude Rate": [None, 0.5, 2.1, None],
                "Crude Rate suppressed": [True, False, False, False],
                "Crude Rate unreliable": [False, True, False, False],
            },
            schema_overrides={"Deaths": pl.Int64, "Crude Rate": pl.Float64},
        )
        assert_frame_equal(frame, expected)

    def test_readCsvTyped_undeclaredNonWonderSuppressed_columnKeptAsText(self):
        # Arrange: the same table, undeclared and not WONDER-shaped (no "Notes" first column):
        # fail closed, a file is never reinterpreted, exactly as on main (ADR014 §7)
        csv = (
            "State,Year,Deaths,Crude Rate\n"
            "Michigan,2010,Suppressed,Suppressed\n"
            "Michigan,2011,25,0.5 (Unreliable)\n"
            "Ohio,2010,120,2.1\n"
            "Ohio,2011,130,Not Applicable\n"
        )

        # Act
        frame = _stage(csv)

        # Assert
        assert frame.columns == ["State", "Year", "Deaths", "Crude Rate"]
        assert frame.schema["Deaths"] == pl.String
        assert frame["Deaths"].to_list() == ["Suppressed", "25", "120", "130"]
        assert frame["Crude Rate"].to_list() == ["Suppressed", "0.5 (Unreliable)", "2.1", "Not Applicable"]

    def test_readCsvTyped_wonderShapedUndeclared_wordsReadAsMarkers(self):
        # Arrange: CDC WONDER's own layout (first column "Notes"), not declared
        csv = "Notes,State,Deaths\n,Michigan,Suppressed\n,Ohio,120\n,Indiana,95\n"

        # Act
        frame = _stage(csv)

        # Assert
        assert frame["Deaths"].to_list() == [None, 120, 95]
        assert frame["Deaths suppressed"].to_list() == [True, False, False]

    @pytest.mark.parametrize(
        "cells",
        [
            pytest.param(("Suppressed", "mild", "severe"), id="other_words"),
            pytest.param(("none", "none", "none"), id="no_marker"),
            pytest.param(("Suppressed", "Suppressed", ""), id="only_markers"),
        ],
    )
    def test_readCsvTyped_markerBesideWords_columnLeftAsText(self, cells):
        # Arrange: a text column (a category that happens to say "Suppressed") is never split
        csv = "id,grade\n" + "\n".join(f"{i},{c}" for i, c in enumerate(cells)) + "\n"

        # Act
        frame = _stage(csv)

        # Assert
        assert frame.columns == ["id", "grade"]
        assert frame.schema["grade"] == pl.String


class TestIntervalRanges:
    def test_readCsvTyped_ciColumnLoToHi_splitIntoBounds(self):
        # Arrange: FluVaxView writes the 95% CI in one cell
        csv = "Geography,Estimate (%),95% CI (%)\nMichigan,45.1,43.0 to 47.2\nOhio,50.0,48.1 to 51.9\n"

        # Act
        frame = _stage(csv)

        # Assert
        assert frame.columns == ["Geography", "Estimate (%)", "95% CI (%) lower", "95% CI (%) upper"]
        assert frame.get_column("95% CI (%) lower").to_list() == [43.0, 48.1]
        assert frame.get_column("95% CI (%) upper").to_list() == [47.2, 51.9]

    def test_readCsvTyped_rangeUnderOtherHeader_leftAsText(self):
        # Arrange: "18 to 64" under an age header is a label, not an interval
        csv = "Geography,Age group,Estimate\nMichigan,18 to 64,45.1\nOhio,65 to 74,50.0\n"

        # Act
        frame = _stage(csv)

        # Assert
        assert frame.columns == ["Geography", "Age group", "Estimate"]
        assert frame.get_column("Age group").to_list() == ["18 to 64", "65 to 74"]


@pytest.mark.parametrize(
    "marker",
    ["Suppressed", "<10", "< 5", "<=10", "≤10", "(D)", "D", "*", "**", "(S)", "1-9", "1–10", "Data suppressed"],
)
def test_readCsvTyped_sourceSuppressionMarker_flaggedNeverANumber(marker):
    # Arrange: the source's own marker in an otherwise numeric column
    rows = "\n".join(f"x{i},{marker if i % 5 == 0 else 20 + i}" for i in range(30))

    # Act: a table declared public estimates
    frame = _stage(f"key,deaths\n{rows}\n", public=True)

    # Assert
    assert frame.schema["deaths"] == pl.Int64
    suppressed = frame.filter(pl.col("deaths suppressed"))
    assert suppressed.height == 6
    assert suppressed.get_column("deaths").null_count() == 6


@pytest.mark.parametrize("marker", ["<5", "<10", "*", "(D)", "1-9"])
def test_readCsvTyped_markerInUndeclaredFile_notSuppression(marker):
    # Arrange: in a patient file "<5" is a censored lab value, never a suppressed cell
    rows = "\n".join(f"x{i},{marker if i % 5 == 0 else 20 + i}" for i in range(30))

    # Act
    frame = _stage(f"key,value\n{rows}\n")

    # Assert
    assert "value suppressed" not in frame.columns
    assert frame.schema["value"] == pl.String

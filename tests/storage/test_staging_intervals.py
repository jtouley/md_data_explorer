"""Staging splits "value (lower, upper)" cells and turns year columns into rows (ADR014).

Synthetic text frames only; every number is made up.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.storage.staging import stage_text_frame, unpack_intervals


def _text(data: dict[str, list[str | None]]) -> pl.DataFrame:
    return pl.DataFrame(data, schema=dict.fromkeys(data, pl.String))


class TestUnpackIntervals:
    def test_unpack_packedColumn_threeNumericColumns(self):
        # Arrange
        frame = _text({"Location": ["A", "B"], "Rate": ["401.63 (390.33, 412.62)", "1,020.5 (990.0, 1,050.0)"]})

        # Act
        staged = stage_text_frame(frame).data

        # Assert
        expected = pl.DataFrame(
            {
                "Location": ["A", "B"],
                "Rate": [401.63, 1020.5],
                "Rate lower": [390.33, 990.0],
                "Rate upper": [412.62, 1050.0],
            }
        )
        assert_frame_equal(staged, expected)

    @pytest.mark.parametrize(
        "cells",
        [
            pytest.param(["401.63 (390.33, 412.62)", "about 400"], id="one_cell_not_packed"),
            pytest.param(["401.63", "380.1"], id="plain_numbers"),
            pytest.param(["Heart (cardiac)", "Lung (other)"], id="text_with_brackets"),
            pytest.param([None, None], id="empty"),
        ],
    )
    def test_unpack_notEveryCellPacked_columnKept(self, cells):
        # Arrange
        frame = _text({"Rate": cells})

        # Act
        unpacked = unpack_intervals(frame)

        # Assert
        assert unpacked.columns == ["Rate"]

    def test_unpack_yearColumns_unpivotedToRows(self):
        # Arrange
        frame = _text(
            {
                "Location": ["A", "B"],
                "Mortality Rate, 1980*": ["10 (9, 11)", "20 (18, 22)"],
                "Mortality Rate, 1985*": ["12 (11, 13)", "21 (19, 23)"],
                "Mortality Rate, 1990*": ["14 (13, 15)", "22 (20, 24)"],
            }
        )

        # Act
        staged = stage_text_frame(frame).data.sort("Location", "year")

        # Assert
        assert staged.columns == ["Location", "year", "Mortality Rate", "Mortality Rate lower", "Mortality Rate upper"]
        assert staged.height == 6
        assert staged.filter((pl.col("Location") == "B") & (pl.col("year") == 1985)).row(0)[2:] == (21.0, 19.0, 23.0)

    def test_unpack_twoYearColumns_leftWide(self):
        # Arrange: fewer than three year columns is not a table laid out by year
        frame = _text({"Location": ["A"], "Rate, 1980": ["10 (9, 11)"], "Rate, 1985": ["12 (11, 13)"]})

        # Act
        unpacked = unpack_intervals(frame)

        # Assert
        assert "year" not in unpacked.columns and unpacked.width == 7

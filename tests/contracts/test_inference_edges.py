# SPDX-License-Identifier: MIT
"""Edges of ``contracts.inference``, found by mutation testing: when a code list sits on a measure,
which of its codes are applied, asked or kept, and the bounds of a column's stated range.
Invented tremor-clinic columns only.
"""

from __future__ import annotations

import polars as pl
import pytest

from clinical_analytics.contracts import config
from clinical_analytics.contracts.inference import _far_outside, _largest_unit, _measure_nonresponse

CODES = {"-9": "Refused", "-8": "unknown"}


def _tremor(distinct: int) -> pl.Series:
    """A tremor amplitude with ``distinct`` values besides its codes, and both codes."""
    return pl.Series("tremor_amp", [*range(1, distinct + 1), -9, -8, -9])


def test_measureNonresponse_fewerValuesThanAMeasure_codeListKeptAsItIs() -> None:
    # Act
    result = _measure_nonresponse(_tremor(config.MEASURE_MIN_VALUES - 1), CODES)

    # Assert: a coded item; its codes are its categories
    assert result == (CODES, {}, {})


def test_measureNonresponse_asManyValuesAsAMeasure_explicitAppliedOtherAsked() -> None:
    # Act
    result = _measure_nonresponse(_tremor(config.MEASURE_MIN_VALUES), CODES)

    # Assert
    assert result == ({}, {"-9": "Refused"}, {"-8": "-8 = unknown"})


@pytest.mark.parametrize(
    ("ordinary", "kept"),
    [
        pytest.param({"50": "largest seen"}, {}, id="one_label_is_no_list"),
        pytest.param(
            {"50": "largest seen", "60": "faint"}, {"50": "largest seen", "60": "faint"}, id="two_labels_kept"
        ),
    ],
)
def test_measureNonresponse_ordinaryLabels_keptOnlyAsAList(ordinary: dict[str, str], kept: dict[str, str]) -> None:
    # Act
    labels, applied, asked = _measure_nonresponse(_tremor(config.MEASURE_MIN_VALUES), {"-9": "Refused", **ordinary})

    # Assert
    assert (labels, applied, asked) == (kept, {"-9": "Refused"}, {})


@pytest.mark.parametrize(
    ("code", "applied"),
    [
        pytest.param("0", False, id="low_edge_inside"),
        pytest.param("40", False, id="high_edge_inside"),
        pytest.param("41", True, id="above_the_range"),
        pytest.param("-1", True, id="below_the_range"),
    ],
)
def test_measureNonresponse_explicitCodeAgainstStatedRange_appliedOnlyOutsideIt(code: str, applied: bool) -> None:
    # Act
    _, found, asked = _measure_nonresponse(_tremor(config.MEASURE_MIN_VALUES), {code: "Refused"}, (0.0, 40.0))

    # Assert: inside the range (edges included) it is asked, never blanked
    assert (code in found, code in asked) == (applied, not applied)


@pytest.mark.parametrize(
    ("name", "values", "code", "stated", "far"),
    [
        pytest.param("tremor_amp", [999, 999], 999, None, False, id="no_other_value"),
        pytest.param("tremor_amp", [999, 999, 5], 999, None, True, id="one_other_value"),
        pytest.param("tremor_amp", [0, 0, 0, 5, 10], 0, None, False, id="zero_is_no_negative_code"),
        pytest.param("tremor_amp", [-1, -1, 0, 5, 10], -1, None, True, id="negative_among_values_from_zero"),
        pytest.param("tremor_amp", [-1, -1, -2, 5, 10], -1, None, False, id="negative_among_negatives"),
        pytest.param("tremor_change", [-1, -1, 0, 5, 10], -1, None, False, id="negative_on_a_change"),
        pytest.param("tremor_amp", [50, 50, 5, 100], 50, None, False, id="below_the_largest_value"),
        pytest.param("tremor_amp", [30, 30, 5, 20], 30, None, False, id="exactly_the_gap"),
        pytest.param("tremor_amp", [99, 99, 5, 30], 99, None, True, id="beyond_the_gap"),
        pytest.param("tremor_amp", [1, 1, 1, 0, 0, 0], 1, None, True, id="above_values_all_zero"),
        pytest.param("tremor_count", [99, 99, 5, 30], 99, None, False, id="count_below_999"),
        pytest.param("tremor_count", [999, 999, 5, 50], 999, None, True, id="count_999_far_beyond"),
        pytest.param("tremor_count", [999, 999, 5, 300], 999, None, False, id="count_999_within_its_gap"),
        pytest.param("tremor_amp", [120, 120, 5, 90], 120, (0.0, 100.0), True, id="above_range_values_inside"),
        pytest.param("tremor_amp", [100, 100, 5, 60], 100, (0.0, 100.0), False, id="at_the_range_top"),
        pytest.param("tremor_amp", [200, 200, 5, 150], 200, (0.0, 100.0), True, id="values_at_the_range_gap"),
        pytest.param("tremor_amp", [200, 200, 5, 160], 200, (0.0, 100.0), False, id="values_beyond_the_range"),
    ],
)
def test_farOutside_codeAgainstTheOtherValues_onlyFarOutsideIsACode(
    name: str, values: list[int], code: int, stated: tuple[float, float] | None, far: bool
) -> None:
    # Act / Assert: codes reach the rule as floats, as candidates from the column's values do
    assert _far_outside(pl.Series(name, values), float(code), stated) is far


@pytest.mark.parametrize(
    ("units", "largest"),
    [
        pytest.param(["weeks", "Months ", None], "months", id="months_over_weeks"),
        pytest.param(["days", "hours", "fortnights"], "days", id="unknown_spelling_ignored"),
    ],
)
def test_largestUnit_unitsUsed_largestByLength(units: list[str | None], largest: str) -> None:
    # Act / Assert
    assert _largest_unit(pl.DataFrame({"wait_unit": units}), "wait_unit") == largest

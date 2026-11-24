# SPDX-License-Identifier: MIT
"""Edges of ``contracts.inference``, found by mutation testing: when a code list sits on a measure,
which of its codes are applied, asked or kept, and the bounds of a column's stated range.
Invented tremor-clinic columns only.
"""

from __future__ import annotations

import polars as pl
import pytest

from clinical_analytics.contracts import config
from clinical_analytics.contracts.inference import _measure_nonresponse

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

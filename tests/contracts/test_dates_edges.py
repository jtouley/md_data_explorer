# SPDX-License-Identifier: MIT
"""Edges of ``contracts.dates``, found by mutation testing: the plausible-year bounds a text
column's dates must fall within to be read as dates. Invented plaster-room columns only.
"""

from __future__ import annotations

import polars as pl
import pytest

from clinical_analytics.contracts import config
from clinical_analytics.contracts.dates import choose_text_format


@pytest.mark.parametrize(
    ("values", "read"),
    [
        pytest.param(["2000-06-30"], True, id="one_date"),
        pytest.param([f"{config.DATE_YEAR_MIN}-01-01", "2000-06-30"], True, id="earliest_plausible_year"),
        pytest.param(["2000-06-30", f"{config.DATE_YEAR_MAX}-12-31"], True, id="latest_plausible_year"),
        pytest.param([f"{config.DATE_YEAR_MIN - 1}-12-31", "2000-06-30"], False, id="year_before_the_range"),
        pytest.param(["2000-06-30", f"{config.DATE_YEAR_MAX + 1}-01-01"], False, id="year_after_the_range"),
    ],
)
def test_chooseTextFormat_yearsAtThePlausibleBounds_readOnlyInside(values: list[str], read: bool) -> None:
    # Act
    found = choose_text_format(pl.Series("cast_on", values))

    # Assert
    assert (found is not None and getattr(found, "format", None) == "%Y-%m-%d") is read

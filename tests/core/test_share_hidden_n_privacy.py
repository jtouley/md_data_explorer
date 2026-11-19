"""A share, rate or proportion is never shown while its n is hidden.

A hidden n (1-10 patients excluded for missing values, or 1-10 removed by the question's
filters) lies in a window of ten values a reader knows from other answers: the cohort minus 10
to the cohort minus 1. A share shown to four decimals (0.2538) is given by one (n, events) pair
of that window only (n=197, 50 events), which gives back the 1-10 excluded patients. The proofs
here run every (n, events) pair of the window through the presenter and require one output for
all of them, so no pair can be told apart. Synthetic payloads only.
"""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD
from clinical_analytics.core.result_presentation import N_USED, SECONDARY_SUPPRESSED, _share_withheld, present_analysis

WINDOW = range(1, SMALL_CELL_THRESHOLD)  # 1-10 patients excluded or filtered out


def _proportion(n_used: int, events: int, *, excluded: int = 0, filtered: int = 0) -> dict[str, Any]:
    return {
        "type": "proportion",
        "column": "diabetes",
        "value": "yes",
        "proportion": events / n_used if n_used else None,
        "events": events,
        "limited": bool(filtered),
        "n_used": n_used,
        "n_excluded": excluded,
        "n_filtered_out": filtered,
        "unit": "patients",
    }


def _rate(groups: list[tuple[int, int]], *, excluded: int, filtered: int = 0) -> dict[str, Any]:
    """A rate by site; ``groups`` is (n, events) per site."""
    return {
        "type": "rate",
        "column": "diabetes",
        "value": "yes",
        "group_col": "site",
        "groups": [
            {"group": f"S{i}", "n": n, "events": e, "rate": e / n if n else None} for i, (n, e) in enumerate(groups)
        ],
        "n_used": sum(n for n, _ in groups),
        "n_excluded": excluded,
        "n_filtered_out": filtered,
        "unit": "patients",
    }


def test_proportion_threeOfCohort200Excluded_shareNotShown() -> None:
    """The reported leak: 50 of 197 recorded showed 0.2538, which only n=197 gives."""
    # Arrange
    analysis = _proportion(197, 50, excluded=3)
    pinned = [(n, e) for n in range(200 - 10, 200) for e in range(n + 1) if round(e / n, 4) == round(50 / 197, 4)]

    # Act
    shown = present_analysis(analysis)

    # Assert
    assert pinned == [(197, 50)]  # the leak the old output had
    assert shown["table"] == {
        "columns": ["proportion", "n"],
        "rows": [{"proportion": SECONDARY_SUPPRESSED, "n": SECONDARY_SUPPRESSED}],
    }
    assert shown["claim"] == "The share with diabetes = yes is not shown because n is hidden (n hidden)."
    assert "%" not in shown["claim"] and "25." not in shown["claim"]


@pytest.mark.parametrize("cohort", [13, 25, 200, 1000])
@pytest.mark.parametrize("missing", ["excluded", "filtered"])
def test_proportion_everyPairOfTheHiddenWindow_sameOutput(cohort: int, missing: str) -> None:
    # Arrange: every n the reader cannot rule out, with every events count it could have
    outputs = set()
    for hidden in WINDOW:
        n = cohort - hidden
        for events in range(n + 1):
            kwargs = {"excluded": hidden} if missing == "excluded" else {"filtered": hidden}
            analysis = _proportion(n, events, **kwargs)

            # Act
            shown = present_analysis(analysis)
            outputs.add(repr(shown))

    # Assert: one output for the whole window, so no (n, events) pair is told apart
    assert len(outputs) == 1
    assert SECONDARY_SUPPRESSED in outputs.pop()


@pytest.mark.parametrize(
    ("excluded", "share_shown"),
    [
        pytest.param(0, True, id="none_excluded_shown"),
        pytest.param(1, False, id="1_excluded_withheld"),
        pytest.param(10, False, id="10_excluded_withheld"),
        pytest.param(11, True, id="11_excluded_shown"),
    ],
)
def test_proportion_windowBoundaries_shownOnlyWithNShown(excluded: int, share_shown: bool) -> None:
    # Arrange
    analysis = _proportion(200 - excluded, 50, excluded=excluded)

    # Act
    row = present_analysis(analysis)["table"]["rows"][0]

    # Assert
    assert (row["proportion"] != SECONDARY_SUPPRESSED) is share_shown
    assert (row["n"] != SECONDARY_SUPPRESSED) is share_shown


@pytest.mark.parametrize(
    ("events", "n_used"),
    [
        pytest.param(None, 197, id="events_missing"),
        pytest.param(50, None, id="n_missing"),
        pytest.param(50, "197", id="n_not_a_count"),
        pytest.param(True, 197, id="events_bool"),
        pytest.param(-5, 197, id="events_negative"),
        pytest.param(250, 197, id="events_above_n"),
    ],
)
def test_proportion_countsNotRecorded_shareWithheld(events: Any, n_used: Any) -> None:
    # Arrange: fail closed when a count cannot be checked
    analysis = {**_proportion(197, 50), "events": events, "n_used": n_used, "n_excluded": 0}

    # Act
    row = present_analysis(analysis)["table"]["rows"][0]

    # Assert
    assert row["proportion"] == SECONDARY_SUPPRESSED


@pytest.mark.parametrize("cohort", [26, 200])
def test_rate_everyPairOfTheHiddenWindow_sameOutput(cohort: int) -> None:
    # Arrange: two sites; the 1-10 excluded patients may be in either, events anywhere feasible
    outputs = set()
    for hidden in WINDOW:
        n = cohort - hidden
        for first in range(SMALL_CELL_THRESHOLD, n - SMALL_CELL_THRESHOLD + 1, max(1, n // 7)):
            second = n - first
            for e1 in range(0, first + 1, max(1, first // 5)):
                analysis = _rate([(first, e1), (second, second // 2)], excluded=hidden)

                # Act
                outputs.add(repr(present_analysis(analysis)))

    # Assert
    assert len(outputs) == 1
    assert "%" not in outputs.pop()


@pytest.mark.parametrize(
    ("excluded", "rates_shown"),
    [
        pytest.param(1, False, id="1_excluded_withheld"),
        pytest.param(10, False, id="10_excluded_withheld"),
        pytest.param(11, True, id="11_excluded_shown"),
    ],
)
def test_rate_windowBoundaries_shownOnlyWithNShown(excluded: int, rates_shown: bool) -> None:
    # Arrange
    analysis = _rate([(100, 30), (100 - excluded, 40)], excluded=excluded)

    # Act
    shown = present_analysis(analysis)

    # Assert
    rates = [row["rate"] for row in shown["table"]["rows"]]
    assert all((rate != SECONDARY_SUPPRESSED) is rates_shown for rate in rates)
    assert ("%" in shown["claim"]) is rates_shown


def test_shareWithheld_overallNHiddenGroupKeyNot_withheld() -> None:
    # Arrange: a payload whose hidden set marks only the overall n (a caller that forgot to hide
    # the group sizes with it) still shows no rate: the share rule checks the overall n too
    hidden = {N_USED}

    # Act
    withheld = _share_withheld(30, 100, "North", hidden)

    # Assert
    assert withheld is True

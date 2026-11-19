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

from clinical_analytics.analysis.grouping import HIDDEN_GROUPS_LABEL
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


def _rate(groups: list[tuple[int, int, int]], *, excluded: int, filtered: int = 0) -> dict[str, Any]:
    """A rate by site; ``groups`` is (n, events, rows without a value) per site."""
    return {
        "type": "rate",
        "column": "diabetes",
        "value": "yes",
        "group_col": "site",
        "groups": [
            {"group": f"S{i}", "n": n, "events": e, "rate": e / n if n else None, "n_rows": n + x}
            for i, (n, e, x) in enumerate(groups)
        ],
        "n_used": sum(group[0] for group in groups),
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
                analysis = _rate([(first, e1, hidden), (second, second // 2, 0)], excluded=hidden)

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
    analysis = _rate([(100, 30, 0), (100 - excluded, 40, excluded)], excluded=excluded)

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


def test_rate_fewExcludedInOneGroup_thatGroupNeverShown() -> None:
    """n is shown (16 excluded), but North has 5 of them: its n beside its size from a COUNT by
    site (100) would give back 5. The group's n and value are hidden with its exclusions."""
    # Arrange
    groups = [("North", 95, 30, 5), ("South", 89, 40, 11)]
    analysis: dict[str, Any] = {
        "type": "rate",
        "column": "diabetes",
        "value": "yes",
        "group_col": "site",
        "groups": [{"group": g, "n": n, "events": e, "rate": e / n, "n_rows": n + x} for g, n, e, x in groups],
        "n_used": 184,
        "n_excluded": 16,
        "n_filtered_out": 0,
    }

    # Act
    shown = present_analysis(analysis)

    # Assert
    north = next(row for row in shown["table"]["rows"] if row["site"] in ("North", HIDDEN_GROUPS_LABEL))
    assert north["n"] == SECONDARY_SUPPRESSED
    assert 95 not in [row["n"] for row in shown["table"]["rows"]]
    assert "n=184" in shown["claim"]


@pytest.mark.parametrize(
    ("n_rows", "n_excluded", "values_shown"),
    [
        pytest.param(None, 0, True, id="unrecorded_nobody_excluded_shown"),
        pytest.param(None, 22, False, id="unrecorded_some_excluded_withheld"),
        pytest.param(90, 22, False, id="n_rows_below_n_withheld"),
    ],
)
def test_rate_groupSizeUnrecordedOrInconsistent_failsClosed(n_rows: Any, n_excluded: int, values_shown: bool) -> None:
    # Arrange: a payload from before n_rows was recorded, or one that disagrees with itself
    analysis = _rate([(100, 30, 11), (100, 40, 11)], excluded=n_excluded)
    for group in analysis["groups"]:
        group["n_rows"] = n_rows

    # Act
    rows = present_analysis(analysis)["table"]["rows"]

    # Assert
    assert all((row["rate"] != SECONDARY_SUPPRESSED) is values_shown for row in rows)


@pytest.mark.parametrize("n_excluded", [None, -3, "3", True])
def test_proportion_exclusionCountUnchecked_nAndShareWithheld(n_excluded: Any) -> None:
    # Arrange: with the row count visible, n beside an unchecked exclusion count gives it back
    analysis = {**_proportion(197, 50), "n_excluded": n_excluded}

    # Act
    row = present_analysis(analysis)["table"]["rows"][0]

    # Assert
    assert row == {"proportion": SECONDARY_SUPPRESSED, "n": SECONDARY_SUPPRESSED}


@pytest.mark.parametrize(
    ("events", "reason"),
    [
        pytest.param(5, "one of its counts is fewer than 11 patients", id="small"),
        pytest.param(250, "its counts are missing or disagree", id="above_n"),
        pytest.param(None, "its counts are missing or disagree", id="missing"),
    ],
)
def test_proportion_withheldWithNShown_reasonMatchesTheCause(events: Any, reason: str) -> None:
    # Arrange
    analysis = {**_proportion(200, 50), "events": events}

    # Act
    claim = present_analysis(analysis)["claim"]

    # Assert
    assert reason in claim
    assert "n=200" in claim


@pytest.mark.parametrize("cohort", [25, 200])
def test_descriptive_meanOfFlagOverHiddenN_sameOutputForTheWindow(cohort: int) -> None:
    """The mean of a 0/1 flag is a share: over a hidden n it is withheld like a proportion."""
    # Arrange: every n of the window, every events count
    outputs = set()
    for hidden in WINDOW:
        n = cohort - hidden
        for events in range(n + 1):
            p = events / n
            analysis = {
                "type": "descriptive",
                "focused_variable": "died",
                "is_numeric": True,
                "mean": p,
                "median": 0.0,
                "std": (p * (1 - p) * n / (n - 1)) ** 0.5,
                "min": 0.0,
                "max": 1.0,
                "non_null_count": n,
                "null_count": hidden,
            }

            # Act
            outputs.add(repr(present_analysis(analysis)))

    # Assert
    assert len(outputs) == 1
    assert "The mean and SD of died are not shown because n is hidden (n hidden)." in outputs.pop()

"""The exact solvability test (``analysis/cell_solvability.py``) behind statistics of columns that
behave like categories.

#144 and #146 withheld every mean, SD, median, minimum and maximum of such a column (and every
t-test or ANOVA of it) once a (value, group) cell held 1-10 patients. The reasoning, that with a
count by value shown a mean adds one equation which can recover a hidden cell, holds only when
the equations actually pin one. These tests show the solver pins the review repros, leaves free
the Table-1 statistics that pin nothing, agrees with brute force, and fails closed.

All data is synthetic.
"""

from __future__ import annotations

import itertools
import statistics
import time
from collections import Counter, OrderedDict
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from clinical_analytics.analysis import cell_solvability as cs
from clinical_analytics.analysis.cell_solvability import (
    LEVELS,
    MAX_TABLE_CELLS,
    NO_HIDDEN_CELL,
    NOT_PINNED,
    SOLVE_BUDGET_S,
    hidden_cell_ranges,
    statistics_level,
)
from clinical_analytics.analysis.grouping import (
    NO_COUNT_TABLE,
    SMALL_CELL_TEST_REFUSAL,
    UNRECORDED_CELLS_TEST_REFUSAL,
    value_statistic_refusal,
)

Counts = dict[tuple[Any, Any], int]


@pytest.fixture(autouse=True)
def _fresh_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test decides its tables afresh (``statistics_level`` keeps decided tables)."""
    monkeypatch.setattr(cs, "_DECIDED", OrderedDict())


# Ages 79-84 of the synthetic cohort (tests/fixtures/synthetic/cohort.csv): "average age in the
# oldest decile", withheld by #146.
OLDEST_DECILE: Mapping[int, int] = {79: 6, 80: 5, 81: 7, 82: 11, 83: 8, 84: 7}
# Ages 51-69 of the same cohort: "average age of patients older than 50 and younger than 70".
AGES_51_TO_69: Mapping[int, int] = {
    51: 10, 52: 2, 53: 7, 54: 5, 55: 13, 56: 6, 57: 16, 58: 6, 59: 5, 60: 9,
    61: 6, 62: 13, 63: 10, 64: 8, 65: 4, 66: 6, 67: 12, 68: 10, 69: 4,
}  # fmt: skip


def _one_group(counts: Mapping[Any, int]) -> Counts:
    return {(value, "all"): n for value, n in counts.items()}


def _cells(values: list[Any], groups: list[Any]) -> Counts:
    return dict(Counter(zip(values, groups, strict=True)))


def length_of_stay_by_arm() -> Counts:
    """Length of stay 1-15 days for 300 patients, 150 per arm (negative binomial, seed 32): the
    long stays (9-15 days) hold 1-10 patients per arm."""
    rng = np.random.default_rng(32)
    stay = np.clip(1 + rng.negative_binomial(2, 0.3, 300), 1, 15)
    return _cells([int(v) for v in stay], ["A"] * 150 + ["B"] * 150)


def pain_by_group() -> Counts:
    """Pain score 0-10 for 120 patients in three groups (rounded normal, seed 69): the extremes
    of each group hold 1-10 patients."""
    rng = np.random.default_rng(1069)
    groups = rng.choice(["control", "drug A", "drug B"], 120)
    centre = {"control": 6.0, "drug A": 5.0, "drug B": 4.5}
    pain = [int(np.clip(np.rint(rng.normal(centre[g], 2.5)), 0, 10)) for g in groups]
    return _cells(pain, [str(g) for g in groups])


class TestReviewReprosStayPinned:
    def test_hiddenCellRanges_zeroOneOutcomeWithOneHalf_pinsSeven(self) -> None:
        """#144 H2: A has 30 ones of 100; B 7 ones, one 0.5, 92 zeros. B's mean 0.075 pinned 7."""
        # Arrange
        counts = _cells(
            [1.0] * 30 + [0.0] * 70 + [1.0] * 7 + [0.5] + [0.0] * 92,
            ["A"] * 100 + ["B"] * 100,
        )

        # Act
        result = hidden_cell_ranges(counts)

        # Assert: both hidden cells of B have one solution each
        assert result.level == "none"
        assert result.verdict == "pinned"
        assert {(c.value, c.group): (c.low, c.high) for c in result.ranges} == {
            (Fraction(1, 2), "B"): (1, 1),
            (Fraction(1), "B"): (7, 7),
        }
        assert statistics_level(counts).level == "none"

    def test_hiddenCellRanges_doseModelWithStrayHalf_pinsFiveEvents(self) -> None:
        """#144 H1: dose 0 (500 patients, 100 events) or 1 (500, 5 events) and one patient at 0.5.

        A one-predictor logistic model's estimates are those of its sufficient statistics: the
        events and the sum of dose over the patients with the event, i.e. the mean dose by
        outcome. So the model is a mean of dose by outcome group, and its table is pinned. (The
        model itself stays on the table rule, ``predictor_model_refusal``, and is withheld.)
        """
        # Arrange
        dose = [0.0] * 500 + [1.0] * 500 + [0.5]
        event = [int(i < 100) for i in range(500)] + [int(i < 5) for i in range(500)] + [0]

        # Act
        result = hidden_cell_ranges(_cells(dose, event))

        # Assert
        assert result.verdict == "pinned"
        assert {(c.value, c.group): (c.low, c.high) for c in result.ranges} == {
            (Fraction(1, 2), 0): (1, 1),
            (Fraction(1), 1): (5, 5),
        }

    def test_hiddenCellRanges_averageDiedBySex_pinsSixWomen(self) -> None:
        """#146: died 0/1 by sex showed F 0.06 of 100: 6 women who died."""
        # Arrange
        counts = _cells([1] * 6 + [0] * 94 + [1] * 40 + [0] * 60, ["F"] * 100 + ["M"] * 100)

        # Act
        result = hidden_cell_ranges(counts)

        # Assert
        assert result.verdict == "pinned"
        assert [(c.value, c.group, c.low, c.high) for c in result.ranges] == [(Fraction(1), "F", 6, 6)]
        assert statistics_level(counts).level == "none"


class TestTableOneStatisticsAnswered:
    def test_hiddenCellRanges_oldestDecile_rangesMatchBruteForce(self) -> None:
        """Mean age over ages 79-84 (44 patients, 5 ages held by 5-8): every hidden age's count
        ranges over 3 or more values given n, the mean, SD and median."""
        # Arrange
        counts = _one_group(OLDEST_DECILE)

        # Act
        result = hidden_cell_ranges(counts, "median")

        # Assert: the solver's ranges, and the same from every table of counts a reader could hold
        ranges = [(int(c.value), c.low, c.high) for c in result.ranges]
        assert ranges == [(79, 3, 6), (80, 4, 10), (81, 5, 9), (83, 3, 10), (84, 6, 10)]
        assert ranges == _brute_force_ranges(OLDEST_DECILE)
        assert result.level == "median"
        assert statistics_level(counts).level == "median"

    def test_statisticsLevel_ages51To69_everyStatisticShown(self) -> None:
        # Arrange: 152 patients over 19 ages, 15 of them held by 2-10 patients
        counts = _one_group(AGES_51_TO_69)

        # Act
        result = statistics_level(counts)

        # Assert
        assert (result.level, result.verdict, result.hidden) == ("median", NOT_PINNED, 15)

    @pytest.mark.parametrize(
        ("table", "level"),
        [
            pytest.param(length_of_stay_by_arm, "median", id="length_of_stay_by_arm_300"),
            pytest.param(pain_by_group, "median", id="pain_0_10_by_three_groups_120"),
        ],
    )
    def test_statisticsLevel_realisticFewValuedMeasures_answeredWhenNothingPinned(self, table: Any, level: str) -> None:
        # Arrange
        counts = table()

        # Act
        result = statistics_level(counts)

        # Assert: some cells are hidden, and nothing they and the statistics leave is pinned
        assert result.hidden >= 8
        assert (result.level, result.verdict) == (level, NOT_PINNED)

    def test_statisticsLevel_noHiddenCell_everyLevel(self) -> None:
        # Arrange
        counts = _one_group({1: 20, 2: 30, 3: 11})

        # Act
        result = statistics_level(counts)

        # Assert
        assert (result.level, result.verdict, result.hidden) == (LEVELS[-1], NO_HIDDEN_CELL, 0)


def _brute_force_ranges(counts: Mapping[int, int]) -> list[tuple[int, int, int]]:
    """Each hidden value's lowest and highest count over every assignment of 1-10 patients to the
    hidden values with the same n, sum, sum of squares and median as ``counts`` (the reader's
    view, computed from patients' values directly)."""
    hidden = [v for v, n in counts.items() if 0 < n < 11]

    def facts(table: Mapping[int, int]) -> tuple[Any, ...]:
        ages = [v for v, n in table.items() for _ in range(n)]
        return len(ages), sum(ages), sum(a * a for a in ages), statistics.median(ages)

    target = facts(counts)
    low, high = dict.fromkeys(hidden, 99), dict.fromkeys(hidden, 0)
    for combo in itertools.product(range(1, 11), repeat=len(hidden)):
        table = {**counts, **dict(zip(hidden, combo, strict=True))}
        if sum(table.values()) == target[0] and facts(table) == target:
            for value, count in zip(hidden, combo, strict=True):
                low[value], high[value] = min(low[value], count), max(high[value], count)
    return [(v, low[v], high[v]) for v in hidden]


class TestNarrowAndUnions:
    def test_statisticsLevel_cellKnownToWithinOne_sdWithheld(self) -> None:
        # Arrange: four hidden ages of 5 each; with n, the sum and the sum of squares the only
        # tables are (5,5,5,5) + t(1,-3,3,-1), t in {-1,0,1}: age 66 is 4, 5 or 6
        counts = _one_group({60: 40, 66: 5, 67: 5, 68: 5, 69: 5})

        # Act
        result = statistics_level(counts)
        sd_ranges = {int(c.value): (c.low, c.high) for c in hidden_cell_ranges(counts, "sd").ranges}

        # Assert: the mean alone pins nothing; with the SD age 66 is known to +-1
        assert (result.level, result.verdict) == ("mean", "narrow")
        assert sd_ranges[66] == (4, 6)

    def test_statisticsLevel_unionPinnedThoughEachCellFree_sdWithheld(self) -> None:
        # Arrange: (5,7,4,5) + t(1,-3,3,-1), t in -1..2: every age's count spans 3 or more,
        # but ages 66 and 69 together always hold 10 patients
        counts = _one_group({60: 40, 66: 5, 67: 7, 68: 4, 69: 5})

        # Act
        result = statistics_level(counts)
        sd_ranges = [c.high - c.low for c in hidden_cell_ranges(counts, "sd").ranges]

        # Assert
        assert min(sd_ranges) >= 3
        assert (result.level, result.verdict) == ("mean", "union_pinned")

    def test_statisticsLevel_singleHiddenCellPerGroup_pinnedByCountsAlone(self) -> None:
        # Arrange: one hidden value; n minus the shown cells gives it back with no statistic
        counts = _one_group({0: 60, 1: 60, 2: 7})

        # Act
        result = statistics_level(counts)

        # Assert
        assert (result.level, result.verdict) == ("none", "pinned")

    def test_statisticsLevel_totalsPinWithStatisticOnly_withheld(self) -> None:
        # Arrange: each arm alone leaves its hidden cells free, but the value totals (a COUNT by
        # length of stay) and the means together pin a cell (seed 1 of the length-of-stay model)
        rng = np.random.default_rng(1)
        stay = np.clip(1 + rng.negative_binomial(2, 0.3, 300), 1, 15)
        counts = _cells([int(v) for v in stay], ["A"] * 150 + ["B"] * 150)

        # Act
        result = statistics_level(counts)

        # Assert
        assert result.level == "none"
        assert result.verdict in ("pinned_with_totals", "union_pinned")


class TestDecisionsAndValues:
    def test_statisticsLevel_floatsOnDecimalGrid_sameAsDecimals(self) -> None:
        """Creatinine-like values in 0.1 steps: as floats (binary 0.8000000000000000444) they
        are read as written, so they lie on the same exact grid as their decimals."""
        # Arrange
        counts = {0.8: 40, 0.9: 35, 1.0: 30, 1.1: 6, 1.2: 5, 1.3: 7, 1.4: 4, 1.5: 6, 1.6: 3}
        as_float = {(v, "all"): n for v, n in counts.items()}
        as_decimal = {(Decimal(str(v)), "all"): n for v, n in counts.items()}

        # Act
        floats, decimals = statistics_level(as_float), statistics_level(as_decimal)

        # Assert
        assert floats.verdict != "unsolved"
        assert (floats.level, floats.verdict) == (decimals.level, decimals.verdict)

    def test_statisticsLevel_sameTableTwice_decidedOnceSameWay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        counts = _one_group(OLDEST_DECILE)
        first = statistics_level(counts)
        monkeypatch.setattr(cs, "_decide", lambda *args: pytest.fail("decided again"))

        # Act
        second = statistics_level(counts)

        # Assert
        assert second == first

    def test_statisticsLevel_timeout_notKept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: no time at all, then the normal budget
        counts = _one_group(OLDEST_DECILE)
        monkeypatch.setattr(cs, "SOLVE_BUDGET_S", 0.0)
        timed_out = statistics_level(counts)
        monkeypatch.setattr(cs, "SOLVE_BUDGET_S", SOLVE_BUDGET_S)

        # Act
        retried = statistics_level(counts)

        # Assert
        assert timed_out.verdict == "unsolved"
        assert retried.level == "median"


class TestFailClosed:
    def test_statisticsLevel_tableOverBound_withheldWithoutSolving(self) -> None:
        # Arrange: 20 values x 10 groups, every cell 5
        counts = {(v, g): 5 for v in range(20) for g in range(10)}
        assert len(counts) > MAX_TABLE_CELLS

        # Act
        started = time.monotonic()
        result = statistics_level(counts)

        # Assert
        assert (result.level, result.verdict) == ("none", "too_large")
        assert time.monotonic() - started < 1.0

    @pytest.mark.parametrize(
        ("status", "x"),
        [
            pytest.param(1, None, id="time_limit"),
            pytest.param(2, None, id="infeasible_claimed"),
            pytest.param(0, "off_by_one", id="solution_not_checking_out"),
        ],
    )
    def test_statisticsLevel_solverFailure_withheld(self, monkeypatch: pytest.MonkeyPatch, status: int, x: Any) -> None:
        # Arrange: a table that is answered when the solver works
        counts = _one_group(OLDEST_DECILE)

        class Result:
            def __init__(self, width: int) -> None:
                self.status = status
                self.x = None if x is None else np.full(width, 10.0)

        def broken(solver: Any, objective: np.ndarray, time_limit: float) -> Result:
            return Result(len(objective))

        monkeypatch.setattr(cs._Solver, "_milp", broken)

        # Act
        result = statistics_level(counts)

        # Assert
        assert (result.level, result.verdict) == ("none", "unsolved")

    def test_statisticsLevel_budgetSpent_withheld(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setattr(cs, "SOLVE_BUDGET_S", 0.0)

        # Act
        result = statistics_level(_one_group(OLDEST_DECILE))

        # Assert
        assert (result.level, result.verdict) == ("none", "unsolved")

    @pytest.mark.parametrize(
        "counts",
        [
            pytest.param({(float("nan"), "a"): 5}, id="nan_value"),
            pytest.param({(float("inf"), "a"): 5}, id="inf_value"),
            pytest.param({("high", "a"): 5}, id="text_value"),
            pytest.param({(1.0, "a"): -1}, id="negative_count"),
            pytest.param({(1.0, "a"): 2.5}, id="fractional_count"),
            pytest.param({(1.0, "a"): True}, id="bool_count"),
            pytest.param({}, id="empty_table"),
            pytest.param({(1.0, "a"): 0}, id="no_patient"),
        ],
    )
    def test_statisticsLevel_malformedCells_raise(self, counts: Counts) -> None:
        # Act / Assert
        with pytest.raises(ValueError):
            statistics_level(counts)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(Decimal("0.5"), id="decimal"),
            pytest.param(np.float64(0.5), id="numpy_float"),
            pytest.param(Fraction(1, 2), id="fraction"),
        ],
    )
    def test_statisticsLevel_numericValueTypes_sameAsFloat(self, value: Any) -> None:
        # Arrange: the #144 H2 table with its 0.5 in another numeric type
        counts = _cells([1.0] * 30 + [0.0] * 70 + [1.0] * 7 + [0.0] * 92, ["A"] * 100 + ["B"] * 99)
        counts[(value, "B")] = 1

        # Act
        result = statistics_level(counts)

        # Assert
        assert (result.level, result.verdict) == ("none", "pinned")


@pytest.mark.slow
def test_statisticsLevel_twentyValuesByFiveGroupsAllHidden_decidedWithinBudget() -> None:
    """The cost bound: 20 values x 5 groups with every cell hidden (3-9 patients) and nothing
    pinned is the slowest table within MAX_TABLE_CELLS (every level runs to the end): 3-5 s on
    one core. Past SOLVE_BUDGET_S it is withheld, never shown unchecked."""
    # Arrange
    rng = np.random.default_rng(0)
    counts = {(v, g): int(rng.integers(3, 10)) for v in range(1, 21) for g in "abcde"}

    # Act
    started = time.monotonic()
    result = statistics_level(counts)
    elapsed = time.monotonic() - started

    # Assert
    assert result.cells == 100
    assert elapsed < SOLVE_BUDGET_S + 1.0
    assert result.level == "median" or result.verdict == "unsolved"


class TestSharedDecision:
    @pytest.mark.parametrize(
        ("min_cell", "level", "statistic", "expected"),
        [
            pytest.param(NO_COUNT_TABLE, None, "mean", None, id="measurement_shown"),
            pytest.param(11, None, "std", None, id="no_hidden_cell_shown"),
            pytest.param(4, None, "mean", SMALL_CELL_TEST_REFUSAL, id="hidden_unchecked_withheld"),
            pytest.param(4, "none", "mean", SMALL_CELL_TEST_REFUSAL, id="level_none_withheld"),
            pytest.param(4, "mean", "mean", None, id="mean_level_mean_shown"),
            pytest.param(4, "mean", "max", None, id="mean_level_max_shown"),
            pytest.param(4, "mean", "std", SMALL_CELL_TEST_REFUSAL, id="mean_level_sd_withheld"),
            pytest.param(4, "sd", "median", SMALL_CELL_TEST_REFUSAL, id="sd_level_median_withheld"),
            pytest.param(4, "median", "median", None, id="median_level_median_shown"),
            pytest.param(4, "median", "sum", SMALL_CELL_TEST_REFUSAL, id="unknown_statistic_withheld"),
            pytest.param(4, "MEDIAN", "mean", SMALL_CELL_TEST_REFUSAL, id="unknown_level_withheld"),
            pytest.param(4, ["median"], "mean", SMALL_CELL_TEST_REFUSAL, id="level_not_text_withheld"),
            pytest.param("4", "median", "mean", SMALL_CELL_TEST_REFUSAL, id="cell_text_withheld"),
            pytest.param(True, "median", "mean", SMALL_CELL_TEST_REFUSAL, id="cell_bool_withheld"),
            pytest.param(0, "median", "mean", SMALL_CELL_TEST_REFUSAL, id="cell_zero_withheld"),
            pytest.param(None, "median", "mean", UNRECORDED_CELLS_TEST_REFUSAL, id="cell_missing_withheld"),
        ],
    )
    def test_valueStatisticRefusal_cellAndLevel_decideStatistic(
        self, min_cell: Any, level: Any, statistic: str, expected: str | None
    ) -> None:
        # Act
        refusal = value_statistic_refusal(min_cell, level, statistic)

        # Assert
        assert refusal == expected

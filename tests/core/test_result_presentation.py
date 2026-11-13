"""Tests for claim-first presentation of analysis results (core/result_presentation.py)."""

import json
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from clinical_analytics.core import result_presentation
from clinical_analytics.core.result_presentation import (
    SECONDARY_SUPPRESSED,
    _single_value,
    _suppress_lines,
    present_analysis,
    suppress_count_matrix,
    to_json_safe,
)

T_TEST = {
    "type": "comparison",
    "test_type": "t_test",
    "outcome_col": "ldl",
    "group_col": "treatment",
    "groups": ["drug", "placebo"],
    "p_value": 0.0004,
    "mean_diff": -16.14,
    "ci_lower": -18.83,
    "ci_upper": -13.44,
    "group_means": {"drug": 113.96, "placebo": 130.09},
    "group_sizes": {"drug": 160, "placebo": 140},
    "n_used": 300,
    "n_excluded": 0,
}


class TestPresentAnalysisClaims:
    @pytest.mark.parametrize(
        ("analysis", "expected_claim"),
        [
            (
                T_TEST,
                "Mean ldl was 16.14 higher in placebo than drug (95% CI 13.44 to 18.83; t-test p<0.001; n=300). "
                "Unadjusted comparison by treatment.",
            ),
            (
                {**T_TEST, "mean_diff": 2.5, "p_value": 0.2, "ci_lower": None, "ci_upper": None, "group_sizes": None},
                "Mean ldl was 2.50 higher in drug than placebo, not statistically significant (t-test p=0.200; n=300). "
                "Unadjusted comparison by treatment.",
            ),
            (
                {
                    "type": "comparison",
                    "test_type": "anova",
                    "outcome_col": "ldl",
                    "group_col": "site",
                    "n_groups": 3,
                    "p_value": 0.03,
                    "group_means": {"a": 1.0, "b": 2.0, "c": 3.0},
                },
                "Mean ldl differed across 3 site groups (ANOVA p=0.030). Unadjusted.",
            ),
            (
                {
                    "type": "comparison",
                    "test_type": "chi_square",
                    "outcome_col": "died",
                    "group_col": "arm",
                    "p_value": 0.4,
                    "contingency": [{"died": "yes", "a": 3, "b": 4}],
                },
                "The distribution of died did not differ significantly by arm (chi-square p=0.400). Unadjusted.",
            ),
            (
                {
                    "type": "predictor",
                    "outcome_col": "outcome",
                    "summary": {
                        "Intercept": {"Odds Ratio": 0.1, "CI Lower": 0.0, "CI Upper": 0.5, "P-Value": 0.01},
                        "age": {"Odds Ratio": 1.05, "CI Lower": 1.02, "CI Upper": 1.08, "P-Value": 0.001},
                    },
                    "significant_predictors": [
                        {"variable": "age", "odds_ratio": 1.05, "ci_lower": 1.02, "ci_upper": 1.08, "p_value": 0.001}
                    ],
                },
                "age had the strongest evidence of association with outcome (OR 1.05; 95% CI 1.02 to 1.08; p=0.001). "
                "Logistic regression of outcome on age; unadjusted.",
            ),
            (
                {
                    "type": "predictor",
                    "outcome_col": "outcome",
                    "summary": {"hdl": {"Odds Ratio": 1.0, "CI Lower": 0.9, "CI Upper": 1.1, "P-Value": 0.8}},
                    "significant_predictors": [],
                },
                "No predictor reached p<0.05 for outcome. Logistic regression of outcome on hdl; unadjusted.",
            ),
            (
                {
                    "type": "relationship",
                    "correlations": [
                        {"var1": "age", "var2": "age", "correlation": 1.0},
                        {"var1": "age", "var2": "ldl", "correlation": -0.42},
                        {"var1": "age", "var2": "hdl", "correlation": 0.1},
                    ],
                    "n_observations": 250,
                },
                "The strongest correlation was moderate: age and ldl (r=-0.42; n=250).",
            ),
            (
                {"type": "survival", "time_col": "days", "event_col": "death", "median_survival": 412.0, "n_used": 120},
                "Median time to death was 412.0 (days) (n=120).",
            ),
            (
                {"type": "survival", "time_col": "days", "event_col": "death", "median_survival": None, "n_used": 120},
                "Median time to death was not reached during follow-up (days) (n=120).",
            ),
            (
                {
                    "type": "descriptive",
                    "focused_variable": "ldl",
                    "is_numeric": True,
                    "mean": 121.49,
                    "median": 121.5,
                    "std": 14.4,
                    "min": 83.0,
                    "max": 159.0,
                    "non_null_count": 300,
                },
                "Mean ldl was 121.49 (SD 14.40; median 121.50; n=300).",
            ),
            (
                {
                    "type": "count",
                    "total_count": 300,
                    "grouped_by": "sex",
                    "group_counts": [{"sex": "F", "count": 178}],
                },
                "300 total across 1 sex group.",
            ),
            ({"type": "count", "total_count": 0}, "Total count: 0."),
        ],
        ids=[
            "t_test_full",
            "t_test_missing_ci_and_sizes",
            "anova",
            "chi_square_not_significant",
            "predictor_significant",
            "predictor_none_significant",
            "relationship_strongest_off_diagonal",
            "survival_median",
            "survival_not_reached",
            "descriptive_numeric",
            "count_grouped",
            "count_total_zero",
        ],
    )
    def test_present_analysis_byType_leadsWithClaim(self, analysis, expected_claim):
        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation["claim"] == expected_claim

    @pytest.mark.parametrize(
        "analysis",
        [None, {}, {"type": "comparison", "error": "Not enough data"}, {"type": "mystery"}],
        ids=["none", "empty", "error_result", "unknown_type"],
    )
    def test_present_analysis_noSupportedResult_returnsNoClaim(self, analysis):
        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation == {"claim": None, "table": None}

    @pytest.mark.parametrize(
        ("mean_diff", "ci", "expected_ci"),
        [(-1.25, (-8.30, 5.80), "95% CI -5.80 to 8.30"), (1.25, (-5.80, 8.30), "95% CI -5.80 to 8.30")],
        ids=["second_group_higher_ci_flipped", "first_group_higher_ci_kept"],
    )
    def test_present_analysis_tTestDirection_ciCentredOnStatedDifference(self, mean_diff, ci, expected_ci):
        """The CI is for the difference the claim states, so its centre equals the stated effect."""
        # Arrange
        analysis = {**T_TEST, "mean_diff": mean_diff, "ci_lower": ci[0], "ci_upper": ci[1]}

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert "1.25 higher in" in claim
        assert expected_ci in claim


class TestPresentAnalysisTables:
    def test_present_analysis_tTest_tableHasGroupMeansAndSizes(self):
        # Act
        table = present_analysis(T_TEST)["table"]

        # Assert
        assert table == {
            "columns": ["treatment", "mean ldl", "n"],
            "rows": [
                {"treatment": "drug", "mean ldl": 113.96, "n": 160},
                {"treatment": "placebo", "mean ldl": 130.09, "n": 140},
            ],
        }

    def test_present_analysis_predictor_tableExcludesIntercept(self):
        # Arrange
        analysis = {
            "type": "predictor",
            "outcome_col": "y",
            "summary": {
                "Intercept": {"Odds Ratio": 0.1, "CI Lower": 0.0, "CI Upper": 0.5, "P-Value": 0.01},
                "age": {"Odds Ratio": 1.05, "CI Lower": 1.02, "CI Upper": 1.08, "P-Value": 0.001},
            },
            "significant_predictors": [],
        }

        # Act
        table = present_analysis(analysis)["table"]

        # Assert
        assert [row["variable"] for row in table["rows"]] == ["age"]

    def test_present_analysis_dummyCodedTerm_usesReadableLabel(self):
        # Arrange
        stats = {"Odds Ratio": 1.96, "CI Lower": 1.11, "CI Upper": 3.47, "P-Value": 0.021}
        analysis = {
            "type": "predictor",
            "outcome_col": "outcome",
            "summary": {"treatment_placebo[T.True]": stats},
            "significant_predictors": [
                {
                    "variable": "treatment_placebo[T.True]",
                    "odds_ratio": 1.96,
                    "ci_lower": 1.11,
                    "ci_upper": 3.47,
                    "p_value": 0.021,
                }
            ],
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation["claim"].startswith("treatment_placebo had the strongest evidence of association")
        assert presentation["table"]["rows"][0]["variable"] == "treatment_placebo"

    def test_present_analysis_tableFloats_roundedToFourPlaces(self):
        # Arrange
        analysis = {**T_TEST, "group_means": {"drug": 114.68387096774194, "placebo": 130.97931034482758}}

        # Act
        rows = present_analysis(analysis)["table"]["rows"]

        # Assert
        assert [row["mean ldl"] for row in rows] == [114.6839, 130.9793]


class TestToJsonSafe:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.float64(1.5), 1.5),
            (np.int64(3), 3),
            (float("nan"), None),
            (math.inf, None),
            ({"a": [np.float32(2.0), None]}, {"a": [2.0, None]}),
            ((1, "x"), [1, "x"]),
        ],
        ids=["numpy_float", "numpy_int", "nan", "inf", "nested", "tuple"],
    )
    def test_to_json_safe_values_convertedForJson(self, value, expected):
        # Act / Assert
        assert to_json_safe(value) == expected


class TestPresentAnalysisStatisticalReporting:
    """Claims name the test actually run, the analysed n, exclusions, and suppress small cells."""

    @pytest.mark.parametrize(
        ("analysis", "expected_claim"),
        [
            (
                {**T_TEST, "test_type": "welch_t_test", "n_used": 280, "n_excluded": 20},
                "Mean ldl was 16.14 higher in placebo than drug (95% CI 13.44 to 18.83; Welch t-test p<0.001; "
                "n=280; 20 excluded for missing data). Unadjusted comparison by treatment.",
            ),
            (
                {**T_TEST, "test_type": "welch_t_test", "n_used": 280, "n_excluded": 4},
                "Mean ldl was 16.14 higher in placebo than drug (95% CI 13.44 to 18.83; Welch t-test p<0.001; "
                "n hidden). Unadjusted comparison by treatment.",
            ),
            (
                {**T_TEST, "test_type": "welch_t_test", "n_used": None, "n_excluded": None},
                "Mean ldl was 16.14 higher in placebo than drug (95% CI 13.44 to 18.83; Welch t-test p<0.001). "
                "Unadjusted comparison by treatment.",
            ),
        ],
        ids=["exclusions_reported", "small_exclusion_suppressed", "n_missing_omitted"],
    )
    def test_present_analysis_welchComparison_reportsTestAndN(self, analysis, expected_claim):
        # Arrange: analysis dict from parametrize

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert claim == expected_claim

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(0, "0"), (1, "<11"), (10, "<11"), (11, "11"), (300, "300")],
        ids=["zero", "one", "just_below", "threshold", "large"],
    )
    def test_present_analysis_countClaim_suppressesSmallCells(self, count, expected):
        # Arrange
        analysis = {"type": "count", "total_count": count}

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert claim == f"Total count: {expected}."

    def test_present_analysis_groupedCountTable_suppressesSmallCells(self):
        # Arrange
        analysis = {
            "type": "count",
            "total_count": 120,
            "grouped_by": "sex",
            "group_counts": [{"sex": "F", "count": 115}, {"sex": "M", "count": 5}],
        }

        # Act
        table = present_analysis(analysis)["table"]

        # Assert: the small cell is hidden, and so is the other one, or it is total minus 115
        assert table["rows"] == [{"sex": "F", "count": "suppressed"}, {"sex": "M", "count": "suppressed"}]


class TestSmallCellSuppression:
    """A hidden count must not be recoverable from totals, other cells or percentages."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([115, 5], ["suppressed", "suppressed"]),
            ([100, 40, 5], [100, "suppressed", "suppressed"]),
            ([100, 3, 9], [100, "suppressed", "suppressed"]),
            ([100, 1, 1], ["suppressed", "suppressed", "suppressed"]),
            ([100, 40, 0], [100, 40, 0]),
            ([50, 60], [50, 60]),
            ([None, 5, 30], [None, "suppressed", "suppressed"]),
        ],
        ids=[
            "two_groups",
            "hide_next_smallest",
            "two_small_sum_reaches_threshold",
            "two_small_sum_below_threshold",
            "zero_shown",
            "none_small",
            "null",
        ],  # fmt: skip
    )
    def test_suppress_counts_withVisibleTotal_hidesComplement(self, values, expected):
        # Arrange
        from clinical_analytics.core.result_presentation import suppress_counts

        # Act
        result = suppress_counts(values)

        # Assert
        assert result == expected

    def test_present_comparison_smallCell_withholdsPercentagesAndComplements(self):
        # Arrange: 5 of 105 in A have the event, 0 of 95 in B
        analysis = {
            "type": "comparison",
            "test_type": "chi_square",
            "outcome_col": "y",
            "group_col": "arm",
            "groups": ["A", "B"],
            "group_sizes": {"A": 105, "B": 95},
            "contingency": [{"y": "0", "A": 100, "B": 95}, {"y": "1", "A": 5, "B": 0}],
            "event_level": "1",
            "group_proportions": {"A": 5 / 105, "B": 0.0},
            "p_value": 0.03,
            "n_used": 200,
            "n_excluded": 0,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert: no percentage in the claim; the 5 and its complement 100 are both hidden
        assert "%" not in presentation["claim"].split("withheld")[0]
        assert "withheld" in presentation["claim"]
        cells = {(row["y"], key): value for row in presentation["table"]["rows"] for key, value in row.items()}
        assert cells[("y=1", "A")] == "suppressed"
        assert cells[("y=0", "A")] == "suppressed"

    def test_present_survival_ungrouped_showsSummaryNotEventTimes(self):
        # Arrange
        analysis = {
            "type": "survival",
            "time_col": "days",
            "event_col": "Status 1: Dead 2: Alive",
            "event_level": "1",
            "median_survival": 400.0,
            "n_used": 120,
            "n_events": 115,
            "summary": [{"timeline": 3.0, "KM_estimate": 0.99}],
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert: the event is named; censored=5 is hidden along with its complement
        assert presentation["claim"].startswith("Median time to Status = Dead was 400.0 (days)")
        assert presentation["table"]["rows"] == [
            {"n": 120, "events": "suppressed", "censored": "suppressed", "median days": 400.0}
        ]

    @pytest.mark.parametrize(
        ("p_value", "fragment"),
        [
            (0.2, "higher in drug than placebo, not statistically significant ("),
            (0.01, "higher in drug than placebo ("),
        ],
        ids=["not_significant", "significant"],
    )
    def test_present_tTest_pValue_saysWhetherSignificant(self, p_value, fragment):
        # Arrange
        analysis = {**T_TEST, "mean_diff": 2.5, "p_value": p_value}

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert fragment in claim

    def test_present_survival_tinyGroup_hidesItsMedian(self):
        # Arrange: a median of 3 patients is one patient's time
        analysis = {
            "type": "survival",
            "time_col": "days",
            "event_col": "event",
            "event_level": "1",
            "group_col": "arm",
            "group_medians": {"A": 300.0, "B": 20.9},
            "group_sizes": {"A": 57, "B": 3},
            "logrank_p_value": 0.2,
            "n_used": 60,
            "n_excluded": 0,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert: A's size is hidden as B's complement, so A's median is hidden too; a shown
        # median would mark A as the large group and give B away as n - A
        assert "20.9" not in presentation["claim"]
        assert "300.0" not in presentation["claim"]
        assert "A not shown, B not shown (log-rank p=0.200; n=60)" in presentation["claim"]
        rows = {row["arm"]: row for row in presentation["table"]["rows"]}
        assert rows["B"] == {"arm": "B", "n": "suppressed", "median days": "suppressed"}
        assert rows["A"] == {"arm": "A", "n": "suppressed", "median days": "suppressed"}


# --- Brute-force recovery audit -------------------------------------------------------------
#
# An attacker sees one answer (claim + table) and the dataset row count, and knows how the
# presenter works. For a synthetic analysis built from integer cells, enumerate every other
# cell assignment with the same visible total, keep those whose answer is identical, and
# collect the feasible values of each quantity. A count from 1 to 10 whose feasible set is a
# single value has been leaked.

_SMALL = range(1, 11)


def _compositions(total: int, minimums: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Every tuple of len(minimums) integers, each >= its minimum, summing to total."""
    if len(minimums) == 1:
        if total >= minimums[0]:
            yield (total,)
        return
    for first in range(minimums[0], total - sum(minimums[1:]) + 1):
        for rest in _compositions(total - first, minimums[1:]):
            yield (first, *rest)


@dataclass(frozen=True)
class _Scenario:
    build: Callable[[tuple[int, ...]], dict[str, Any]]
    quantities: Callable[[tuple[int, ...]], dict[str, int]]
    candidates: Callable[[tuple[int, ...]], Iterable[tuple[int, ...]]]


def _answer(analysis: dict[str, Any]) -> str:
    return json.dumps(present_analysis(analysis), sort_keys=True, default=str)


def _recovered(scenario: _Scenario, cells: tuple[int, ...]) -> dict[str, int]:
    """Small quantities (1-10) whose value is the only one consistent with the answer."""
    shown = _answer(scenario.build(cells))
    feasible: dict[str, set[int]] = {}
    for candidate in scenario.candidates(cells):
        if _answer(scenario.build(candidate)) == shown:
            for name, value in scenario.quantities(candidate).items():
                feasible.setdefault(name, set()).add(value)
    truth = scenario.quantities(cells)
    return {name: value for name, value in truth.items() if value in _SMALL and feasible.get(name) == {value}}


def _same_total(minimums: tuple[int, ...]) -> Callable[[tuple[int, ...]], Iterable[tuple[int, ...]]]:
    """Candidates sharing the visible dataset row count (the sum of all cells)."""
    return lambda cells: _compositions(sum(cells), minimums)


def _two_group_comparison(cells: tuple[int, ...]) -> dict[str, Any]:
    drug, placebo, excluded = cells
    return {
        **T_TEST,
        "group_sizes": {"drug": drug, "placebo": placebo},
        "n_used": drug + placebo,
        "n_excluded": excluded,
    }


def _chi_square(cells: tuple[int, ...]) -> dict[str, Any]:
    a0, b0, a1, b1, excluded = cells
    return {
        "type": "comparison",
        "test_type": "chi_square",
        "outcome_col": "y",
        "group_col": "arm",
        "groups": ["A", "B"],
        "group_sizes": {"A": a0 + a1, "B": b0 + b1},
        "contingency": [{"y": "0", "A": a0, "B": b0}, {"y": "1", "A": a1, "B": b1}],
        "event_level": "1",
        "group_proportions": {"A": a1 / (a0 + a1) if a0 + a1 else 0.0, "B": b1 / (b0 + b1) if b0 + b1 else 0.0},
        "p_value": 0.03,
        "n_used": a0 + b0 + a1 + b1,
        "n_excluded": excluded,
    }


def _chi_square_candidates(cells: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    # every group has at least one patient
    return (c for c in _compositions(sum(cells), (0, 0, 0, 0, 0)) if c[0] + c[2] > 0 and c[1] + c[3] > 0)


def _predictor(cells: tuple[int, ...]) -> dict[str, Any]:
    n_used, excluded = cells
    return {
        "type": "predictor",
        "outcome_col": "outcome",
        "summary": {"hdl": {"Odds Ratio": 1.0, "CI Lower": 0.9, "CI Upper": 1.1, "P-Value": 0.8}},
        "significant_predictors": [],
        "n_used": n_used,
        "n_excluded": excluded,
    }


def _survival_grouped(cells: tuple[int, ...]) -> dict[str, Any]:
    a, b, excluded = cells
    return {
        "type": "survival",
        "time_col": "days",
        "event_col": "event",
        "event_level": "1",
        "group_col": "arm",
        "group_medians": {"A": 300.0, "B": 20.9},
        "group_sizes": {"A": a, "B": b},
        "logrank_p_value": 0.2,
        "n_used": a + b,
        "n_excluded": excluded,
    }


def _survival_summary(cells: tuple[int, ...]) -> dict[str, Any]:
    events, censored, excluded = cells
    return {
        "type": "survival",
        "time_col": "days",
        "event_col": "event",
        "event_level": "1",
        "median_survival": 400.0,
        "n_used": events + censored,
        "n_events": events,
        "n_excluded": excluded,
    }


def _survival_summary_candidates(cells: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    return (c for c in _compositions(sum(cells), (0, 0, 0)) if c[0] + c[1] > 0)


def _value_counts(cells: tuple[int, ...]) -> dict[str, Any]:
    # nulls are a value_counts row too, so the counts sum to the dataset row count
    return {
        "type": "descriptive",
        "focused_variable": "smoker",
        "is_numeric": False,
        "value_counts": [{"smoker": f"level{i}", "count": c} for i, c in enumerate(cells)],
    }


def _value_counts_candidates(cells: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    # value_counts is sorted by count, descending
    minimums = tuple(1 for _ in cells)
    return (c for c in _compositions(sum(cells), minimums) if list(c) == sorted(c, reverse=True))


def _grouped_count(cells: tuple[int, ...]) -> dict[str, Any]:
    return {
        "type": "count",
        "total_count": sum(cells),
        "grouped_by": "site",
        "group_counts": [{"site": f"s{i}", "count": c} for i, c in enumerate(cells)],
    }


def _grouped_count_candidates(cells: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    # The total is not tied to the dataset row count (filters), so vary it and the group count.
    for total in range(1, 14):
        for k in range(1, min(total, 10) + 1):
            yield from _compositions(total, tuple(1 for _ in range(k)))
    if sum(cells) >= 14:
        yield from _compositions(sum(cells), tuple(1 for _ in cells))


def _numeric_descriptive(cells: tuple[int, ...]) -> dict[str, Any]:
    n, missing = cells
    return {
        "type": "descriptive",
        "focused_variable": "ldl",
        "is_numeric": True,
        "mean": 121.49,
        "median": 121.5,
        "std": 14.4,
        "min": 83.0,
        "max": 159.0,
        "non_null_count": n,
    }


def _named(*names: str) -> Callable[[tuple[int, ...]], dict[str, int]]:
    return lambda cells: dict(zip(names, cells, strict=True))


def _with_n_used(*names: str, parts: int) -> Callable[[tuple[int, ...]], dict[str, int]]:
    return lambda cells: {**dict(zip(names, cells, strict=True)), "n_used": sum(cells[:parts])}


_SCENARIOS = {
    "t_test": _Scenario(
        _two_group_comparison, _with_n_used("drug", "placebo", "excluded", parts=2), _same_total((1, 1, 0))
    ),
    "chi_square": _Scenario(
        _chi_square, _with_n_used("A0", "B0", "A1", "B1", "excluded", parts=4), _chi_square_candidates
    ),
    "predictor": _Scenario(_predictor, _named("n_used", "excluded"), _same_total((1, 0))),
    "survival_grouped": _Scenario(
        _survival_grouped, _with_n_used("A", "B", "excluded", parts=2), _same_total((1, 1, 0))
    ),
    "survival_summary": _Scenario(
        _survival_summary, _with_n_used("events", "censored", "excluded", parts=2), _survival_summary_candidates
    ),
    "value_counts": _Scenario(
        _value_counts, lambda c: {f"level{i}": v for i, v in enumerate(c)}, _value_counts_candidates
    ),
    "grouped_count": _Scenario(
        # a site missing from a candidate has no patients in it
        _grouped_count,
        lambda c: {**{f"s{i}": c[i] if i < len(c) else 0 for i in range(10)}, "total": sum(c)},
        _grouped_count_candidates,
    ),
    "numeric_descriptive": _Scenario(_numeric_descriptive, _named("n", "missing"), _same_total((0, 0))),
}

_AUDIT_CASES = [
    # (scenario, cells) at the boundaries: counts 1, 10, 11; totals 11, 12, 21, 22; n_excluded 0
    ("t_test", (11, 10, 1)),
    ("t_test", (10, 11, 0)),
    ("t_test", (11, 11, 0)),
    ("t_test", (6, 5, 1)),
    ("t_test", (1, 10, 10)),
    ("t_test", (11, 10, 0)),
    ("chi_square", (5, 5, 1, 0, 0)),
    ("chi_square", (10, 0, 1, 0, 1)),
    ("chi_square", (5, 6, 0, 0, 10)),
    ("predictor", (11, 1)),
    ("predictor", (1, 10)),
    ("predictor", (10, 11)),
    ("predictor", (11, 10)),
    ("predictor", (22, 0)),
    ("predictor", (397, 3)),
    ("predictor", (10, 390)),
    ("survival_grouped", (11, 1, 0)),
    ("survival_grouped", (11, 10, 1)),
    ("survival_grouped", (10, 1, 11)),
    ("survival_summary", (11, 1, 0)),
    ("survival_summary", (10, 11, 1)),
    ("survival_summary", (1, 10, 11)),
    ("survival_summary", (5, 5, 1)),
    ("value_counts", (10, 10)),
    ("value_counts", (11, 10)),
    ("value_counts", (11, 1)),
    ("value_counts", (10, 1, 1)),
    ("value_counts", (11, 11)),
    ("grouped_count", (1,) * 10),
    ("grouped_count", (10, 1)),
    ("grouped_count", (11, 1)),
    ("grouped_count", (5, 5)),
    ("numeric_descriptive", (10, 1)),
    ("numeric_descriptive", (1, 10)),
]


class TestSmallCellRecovery:
    """No count from 1 to 10 can be worked out from one answer plus the dataset row count."""

    @pytest.mark.parametrize(
        ("scenario", "cells"),
        _AUDIT_CASES,
        ids=[f"{name}-{'_'.join(map(str, cells))}" for name, cells in _AUDIT_CASES],
    )
    def test_present_analysis_bruteForceAttack_recoversNoSmallCount(self, scenario, cells):
        # Arrange
        audit = _SCENARIOS[scenario]

        # Act
        recovered = _recovered(audit, cells)

        # Assert
        assert recovered == {}, f"answer {_answer(audit.build(cells))} reveals {recovered}"

    @pytest.mark.parametrize(
        ("scenario", "cells"),
        _AUDIT_CASES,
        ids=[f"{name}-{'_'.join(map(str, cells))}" for name, cells in _AUDIT_CASES],
    )
    def test_present_analysis_hiddenCounts_useOneMarkerPerAnswer(self, scenario, cells):
        # Arrange
        analysis = _SCENARIOS[scenario].build(cells)

        # Act
        shown = _answer(analysis)

        # Assert: "<11" next to "suppressed" tells primary (1-10) from secondary (>=11) cells
        assert not ("<11" in shown and SECONDARY_SUPPRESSED in shown), shown

    @pytest.mark.parametrize(
        ("n_used", "n_excluded", "expected"),
        [
            (397, 3, "n hidden"),
            (10, 390, "n hidden"),
            (389, 11, "n=389; 11 excluded for missing data"),
            (400, 0, "n=400"),
            (12, 0, "n=12"),
            (11, 0, "n hidden"),
        ],
        ids=[
            "small_excluded",
            "small_n",
            "both_large",
            "none_excluded",
            "floor_plus_two_none_excluded",
            "floor_plus_one_always_hidden",
        ],
    )
    def test_present_predictor_nClause_hidesBothWhenEitherSmall(self, n_used, n_excluded, expected):
        # Arrange: n_used + n_excluded is the dataset row count, which the app shows; logistic
        # regression needs 10 rows, so of 11 rows n is 10 or 11 and is hidden either way
        analysis = _predictor((n_used, n_excluded))

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert f"(n={n_used}" in claim if expected.startswith("n=") else "(n hidden" in claim
        assert claim.endswith(f"Logistic regression of outcome on hdl ({expected}); unadjusted.")


class TestSmallGroupStatistics:
    """A group statistic of a handful of patients is close to their values; it hides with n."""

    @pytest.mark.parametrize(
        ("sizes", "expected_means"),
        [
            ((1, 1, 1), ["suppressed", "suppressed", "suppressed"]),
            ((10, 50, 60), ["suppressed", "suppressed", 30.0]),
            ((11, 11, 11), [10.0, 20.0, 30.0]),
            ((0, 50, 60), [10.0, 20.0, 30.0]),
        ],
        ids=["one_patient_each", "small_group_and_complement", "threshold", "empty_group"],
    )
    def test_present_anova_smallGroup_hidesItsMeanKeepsTest(self, sizes, expected_means):
        # Arrange: one patient per group makes each mean that patient's value
        analysis = {
            "type": "comparison",
            "test_type": "anova",
            "outcome_col": "ldl",
            "group_col": "site",
            "groups": ["a", "b", "c"],
            "n_groups": 3,
            "p_value": 0.03,
            "group_means": {"a": 10.0, "b": 20.0, "c": 30.0},
            "group_sizes": dict(zip(["a", "b", "c"], sizes, strict=True)),
            "n_used": sum(sizes),
            "n_excluded": 0,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert: the overall test is still reported; each hidden size hides its mean
        assert presentation["claim"].startswith("Mean ldl differed across 3 site groups (ANOVA p=0.030")
        rows = presentation["table"]["rows"]
        assert [row["mean ldl"] for row in rows] == expected_means
        assert all((row["n"] == SECONDARY_SUPPRESSED) == (row["mean ldl"] == SECONDARY_SUPPRESSED) for row in rows)


class TestHiddenNStatistics:
    """With n hidden, no statistic that can be one patient's value is shown."""

    @pytest.mark.parametrize(
        ("n", "shown"),
        [(1, False), (10, False), (11, True), (300, True)],
        ids=["one_patient", "just_below", "threshold", "large"],
    )
    def test_present_descriptive_hiddenN_hidesEveryStatistic(self, n, shown):
        # Arrange: min/max of a handful of patients are their values; a mean of one is too
        analysis = {
            "type": "descriptive",
            "focused_variable": "ldl",
            "is_numeric": True,
            "mean": 121.49,
            "median": 121.5,
            "std": 14.4,
            "min": 83.0,
            "max": 159.0,
            "non_null_count": n,
        }

        # Act
        answer = json.dumps(present_analysis(analysis))

        # Assert
        values = ("121.49", "121.5", "14.4", "83.0", "159.0")
        assert all((value in answer) == shown for value in values), answer
        if not shown:
            assert "Statistics for ldl are not shown because n is hidden (n hidden)." in answer

    @pytest.mark.parametrize(
        ("missing", "shown"),
        [(1, False), (2, True), (10, True)],
        ids=["cohort_of_twelve", "cohort_of_thirteen", "cohort_of_twenty_one"],
    )
    def test_present_descriptive_filteredCohortKnown_statisticsDoNotPinN(self, missing, shown):
        # Arrange: n=11 with a small missing count, 20 rows filtered out. A COUNT with the same
        # filter shows the cohort (n + missing); a shown mean says n >= 11, so with a cohort of
        # 12 it would pin n=11 and 1 missing
        analysis = {
            "type": "descriptive",
            "focused_variable": "ldl",
            "is_numeric": True,
            "mean": 121.49,
            "median": 121.5,
            "std": 14.4,
            "min": 83.0,
            "max": 159.0,
            "non_null_count": 11,
            "null_count": missing,
            "n_filtered_out": 20,
        }

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert ("121.49" in claim) == shown, claim
        assert "n hidden" in claim

    @pytest.mark.parametrize(
        ("n_used", "n_events", "n_excluded", "shown"),
        [
            (1, 1, 0, False),
            (10, 5, 0, False),
            (11, 1, 1, False),
            (11, 5, 0, False),
            (12, 5, 0, True),
            (60, 20, 11, True),
        ],
        ids=["one_patient", "just_below", "exclusion_hides_n", "floor_plus_one_rows", "floor_plus_two_rows", "large"],
    )
    def test_present_survival_ungroupedHiddenN_hidesMedian(self, n_used, n_events, n_excluded, shown):
        # Arrange: a median of one patient is that patient's time
        analysis = _survival_summary((n_events, n_used - n_events, n_excluded))

        # Act
        presentation = present_analysis(analysis)

        # Assert: with n=11 and 1 excluded (row count 12) a shown median would mark n as >= 11
        assert ("400.0" in json.dumps(presentation)) == shown
        assert presentation["table"]["rows"][0]["median days"] == (400.0 if shown else SECONDARY_SUPPRESSED)


# --- Algorithm-aware recovery audit -----------------------------------------------------------
#
# The reader sees one answer and the dataset row count R, knows how the presenter suppresses
# cells, and knows the analysis floors (a t-test needs two patients per group, Kaplan-Meier and
# logistic regression ten complete rows, a correlation three). Every candidate in a box of cell
# values is presented; candidates with the same R and the same answer cannot be told apart. A
# count from 1 to 10 whose value is the same in every candidate sharing its answer is leaked.
# Only R up to the box size is judged, where every composition of R is in the box.


def _attack(
    build: Callable[[tuple[int, ...]], dict[str, Any]],
    candidates: Iterable[tuple[int, ...]],
    names: tuple[str, ...],
    r_max: int,
    judged: Callable[[str], bool] = lambda answer: True,
) -> list[tuple[tuple[int, ...], str]]:
    book: dict[tuple[int, str], dict[str, set[int]]] = {}
    shown: dict[tuple[int, ...], tuple[int, str]] = {}
    for cells in candidates:
        if sum(cells) > r_max:
            continue  # answers are compared within one row count, so these never matter
        key = (sum(cells), _answer(build(cells)))
        shown[cells] = key
        for name, value in zip(names, cells, strict=True):
            book.setdefault(key, {}).setdefault(name, set()).add(value)
    return [
        (cells, name)
        for cells, key in shown.items()
        if 11 <= key[0] <= r_max and judged(key[1])
        for name, value in zip(names, cells, strict=True)
        if value in _SMALL and book[key][name] == {value}
    ]


def _box(
    sizes: tuple[range | tuple[int, ...], ...], keep: Callable[[tuple[int, ...]], bool] = lambda c: True
) -> Callable[[], Iterator[tuple[int, ...]]]:
    """Candidates in a box of cell values, built when the test runs (not at collection)."""
    from itertools import product

    return lambda: (c for c in product(*sizes) if keep(c))


def _ttest_filtered(cells: tuple[int, ...]) -> dict[str, Any]:
    drug, placebo, excluded, filtered = cells
    return {**_two_group_comparison((drug, placebo, excluded)), "n_filtered_out": filtered}


def _anova(cells: tuple[int, ...]) -> dict[str, Any]:
    a, b, c, excluded = cells
    return {
        "type": "comparison",
        "test_type": "anova",
        "outcome_col": "ldl",
        "group_col": "site",
        "groups": ["a", "b", "c"],
        "n_groups": 3,
        "p_value": 0.03,
        "group_means": {"a": 10.0, "b": 20.0, "c": 30.0},
        "group_sizes": {"a": a, "b": b, "c": c},
        "n_used": a + b + c,
        "n_excluded": excluded,
    }


def _predictor_filtered(cells: tuple[int, ...]) -> dict[str, Any]:
    n_used, excluded, filtered = cells
    return {**_predictor((n_used, excluded)), "n_filtered_out": filtered}


def _count_filtered(cells: tuple[int, ...]) -> dict[str, Any]:
    total, filtered = cells
    return {"type": "count", "total_count": total, "n_filtered_out": filtered}


def _grouped_count3(cells: tuple[int, ...]) -> dict[str, Any]:
    return _grouped_count(cells)


def _numeric_with_nulls(cells: tuple[int, ...]) -> dict[str, Any]:
    n, nulls = cells
    return {**_numeric_descriptive((n, nulls)), "row_count": n + nulls, "null_count": nulls}


def _relationship(cells: tuple[int, ...]) -> dict[str, Any]:
    n, excluded = cells
    return {
        "type": "relationship",
        "correlations": [{"var1": "age", "var2": "ldl", "correlation": -0.42}],
        "n_observations": n,
        "n_excluded": excluded,
    }


def _value_counts_cut(cells: tuple[int, ...]) -> dict[str, Any]:
    # ten levels are listed (value_counts().head(10)); the last cell is the cut remainder
    *listed, rest = cells
    analysis = _value_counts(tuple(listed))
    return {**analysis, "row_count": sum(cells)}


_ALGORITHM_AUDITS = {
    "t_test": (_two_group_comparison, ("A", "B", "excluded"), _box((range(2, 25), range(2, 25), range(25))), 24),
    "t_test_filtered": (
        _ttest_filtered,
        ("A", "B", "excluded", "filtered"),
        _box((range(2, 16), range(2, 16), range(16), range(16))),
        15,
    ),
    "anova": (_anova, ("a", "b", "c", "excluded"), _box((range(2, 24), range(2, 24), range(2, 24), range(24))), 23),
    "survival_summary": (
        _survival_summary,
        ("events", "censored", "excluded"),
        _box((range(25), range(25), range(25)), lambda c: c[0] + c[1] >= 10),
        24,
    ),
    "predictor": (_predictor, ("n_used", "excluded"), _box((range(10, 40), range(40))), 39),
    "predictor_filtered": (
        _predictor_filtered,
        ("n_used", "excluded", "filtered"),
        _box((range(10, 30), range(30), range(30))),
        29,
    ),
    "chi_square": (
        _chi_square,
        ("A0", "B0", "A1", "B1", "excluded"),
        lambda: (
            c
            for total in range(2, 21)
            for c in _compositions(total, (0,) * 5)
            if c[0] + c[2] > 0 and c[1] + c[3] > 0 and sum(c[:4]) >= 2
        ),
        20,
    ),
    "grouped_count": (_grouped_count3, ("A", "B", "C"), _box((range(1, 33),) * 3), 32),
    "count_filtered": (_count_filtered, ("total", "filtered"), _box((range(40), range(40))), 39),
    "numeric_nulls": (_numeric_with_nulls, ("n", "nulls"), _box((range(40), range(40))), 39),
    "relationship": (_relationship, ("n", "excluded"), _box((range(3, 40), range(40))), 39),
    "value_counts_cut": (
        _value_counts_cut,
        tuple(f"level{i}" for i in range(10)) + ("rest",),
        # eight listed levels fixed at 30; the last two listed levels and the cut remainder vary
        # (the remainder is any number of levels no larger than the tenth)
        _box(
            ((30,),) * 8 + (range(1, 31), range(1, 31), range(31)), lambda c: c[8] >= c[9] and (c[10] == 0 or c[9] >= 1)
        ),
        240 + 30,
    ),
}

_JUDGED: dict[str, Callable[[str], bool]] = {"value_counts_cut": lambda answer: answer.count('"count": 30') == 8}


class TestAlgorithmAwareRecovery:
    """No count from 1 to 10 is pinned by one answer for a reader who knows the algorithm."""

    @pytest.mark.parametrize("scenario", sorted(_ALGORITHM_AUDITS))
    def test_present_analysis_algorithmAwareAttack_pinsNoSmallCount(self, scenario):
        # Arrange
        build, names, candidates, r_max = _ALGORITHM_AUDITS[scenario]
        # A box that fixes some cells is exact only for answers that show those cells.
        judged = _JUDGED.get(scenario, lambda answer: True)

        # Act
        leaks = _attack(build, candidates(), names, r_max, judged)

        # Assert
        examples = [(cells, name, _answer(build(cells))) for cells, name in leaks[:3]]
        assert leaks == [], f"{len(leaks)} pinned cells, e.g. {examples}"


# --- Review follow-ups (PR #79) ---------------------------------------------------------------


class TestMissingCountsFailClosed:
    """A statistic is shown only when the count that guards it is present and is a count."""

    @pytest.mark.parametrize(
        ("counts", "reason"),
        [
            ({"n_used": 5}, "n is hidden (n hidden)"),
            ({"n_used": 5, "n_events": None}, "n is hidden (n hidden)"),
            ({"n_used": np.int64(5), "n_events": 3}, "n is hidden (n hidden)"),
            ({"n_events": 3}, "n is missing"),
            ({"n_used": None, "n_events": 3}, "n is missing"),
            ({"n_used": "120", "n_events": 3}, "n is missing"),
            ({"n_used": 5.0, "n_events": 3}, "n is missing"),
            ({"n_used": True, "n_events": 1}, "n is missing"),
        ],
        ids=[
            "small_n_events_absent",
            "small_n_events_none",
            "small_n_numpy",
            "n_absent",
            "n_none",
            "n_text",
            "n_float",
            "n_bool",
        ],
    )
    def test_present_survival_ungroupedMissingCount_hidesMedian(self, counts, reason):
        # Arrange: a median of five patients can be one patient's time
        analysis = {
            "type": "survival",
            "time_col": "days",
            "event_col": "event",
            "event_level": "1",
            "median_survival": 400.0,
            **counts,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert "400" not in json.dumps(presentation)
        assert presentation["claim"] == f"Median time to event=1 (days) is not shown because {reason}."

    @pytest.mark.parametrize("n", [None, "300", 300.0, True], ids=["none", "text", "float", "bool"])
    def test_present_descriptive_missingN_hidesEveryStatistic(self, n):
        # Arrange
        analysis = {
            "type": "descriptive",
            "focused_variable": "ldl",
            "is_numeric": True,
            "mean": 121.49,
            "median": 121.5,
            "std": 14.4,
            "min": 83.0,
            "max": 159.0,
            "non_null_count": n,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        answer = json.dumps(presentation)
        assert not any(value in answer for value in ("121.49", "121.5", "14.4", "83.0", "159.0")), answer
        assert presentation["claim"] == "Statistics for ldl are not shown because n is missing."

    @pytest.mark.parametrize(
        "sizes",
        [None, {"a": 40, "b": None, "c": 50}, {"a": 40, "b": "3", "c": 50}, {"a": 40, "b": 3.0, "c": 50}],
        ids=["no_sizes", "size_none", "size_text", "size_float"],
    )
    def test_present_anova_missingGroupSize_hidesThatGroupMean(self, sizes):
        # Arrange: group b's size is not a count, so its mean may be one patient's value
        analysis = {
            "type": "comparison",
            "test_type": "anova",
            "outcome_col": "ldl",
            "group_col": "site",
            "groups": ["a", "b", "c"],
            "n_groups": 3,
            "p_value": 0.03,
            "group_means": {"a": 10.0, "b": 20.0, "c": 30.0},
            "group_sizes": sizes,
        }

        # Act
        rows = present_analysis(analysis)["table"]["rows"]

        # Assert
        assert rows[1]["mean ldl"] == SECONDARY_SUPPRESSED
        assert all((row["mean ldl"] == SECONDARY_SUPPRESSED) == (not isinstance(row["n"], int)) for row in rows)

    @pytest.mark.parametrize(
        "sizes",
        [None, {"A": 57, "B": None}, {"A": 57, "B": "3"}],
        ids=["no_sizes", "size_none", "size_text"],
    )
    def test_present_survival_missingGroupSize_hidesThatGroupMedian(self, sizes):
        # Arrange
        analysis = {**_survival_grouped((57, 40, 0)), "group_sizes": sizes}

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert "20.9" not in json.dumps(presentation)
        assert "B not shown" in presentation["claim"]

    @pytest.mark.parametrize("cell", [None, "3", 3.0], ids=["none", "text", "float"])
    def test_present_chiSquare_cellNotACount_withholdsPercentages(self, cell):
        # Arrange: one cell of the table is not a count, so the shares cannot be checked
        analysis = _chi_square((40, 50, 60, 70, 0))
        analysis["contingency"][0]["A"] = cell

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert "%" not in claim
        assert claim.startswith("The distribution of y differed by arm")


class TestFilteredOutFailsClosed:
    @pytest.mark.parametrize("filtered", [-3, -1, 2.5, "unknown"], ids=["negative", "minus_one", "float", "text"])
    def test_present_count_unreadableFilteredOut_hidesTotal(self, filtered):
        # Arrange: a filtered-out value that is not a non-negative count cannot be checked
        analysis = {"type": "count", "total_count": 400, "n_filtered_out": filtered}

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert claim == "Total count hidden."


class TestNClauseHiddenN:
    @pytest.mark.parametrize(
        ("cells", "filtered"),
        [((2, 3, 50), 20), ((2, 3, 50), 0), ((12, 13, 50), 4), ((2, 3, 11), 300)],
        ids=["small_n_filtered", "small_n_unfiltered", "n_complement_of_filtered_out", "exclusions_at_threshold"],
    )
    def test_present_tTest_hiddenN_doesNotSayWhetherRowsWereExcluded(self, cells, filtered):
        # Arrange: n is hidden while the exclusion count (50 or 11) is large enough to show
        analysis = _two_group_comparison(cells)
        analysis["n_filtered_out"] = filtered

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert "; n hidden). Unadjusted" in claim
        assert "excluded" not in claim

    @pytest.mark.parametrize(
        ("cells", "filtered"),
        [((200, 177, 3), 20), ((200, 177, 1), 11), ((200, 177, 10), 500)],
        ids=["three_excluded", "one_excluded", "ten_excluded"],
    )
    def test_present_tTest_smallExclusionsWithFilter_hidesN(self, cells, filtered):
        # Arrange: a COUNT with the same filter shows n + excluded (380 for the first case), so
        # "n=377; some excluded" would give the exclusions away
        analysis = {**_two_group_comparison(cells), "n_filtered_out": filtered}

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert "; n hidden). Unadjusted" in claim
        assert "n=377" not in claim
        assert "excluded" not in claim


def _hidden_cells(rows: list[list[int]], columns_first: bool) -> set[tuple[int, int]]:
    """Hidden cells of a count table, with its lines checked rows first or columns first."""
    cells = {(i, j): value for i, row in enumerate(rows) for j, value in enumerate(row)}
    by_row = [[(i, j) for j in range(len(row))] for i, row in enumerate(rows)]
    by_column = [[(i, j) for i in range(len(rows))] for j in range(len(rows[0]))]
    return _suppress_lines(cells, by_column + by_row if columns_first else by_row + by_column)


def _matrix_view(rows: list[list[int]], columns_first: bool) -> tuple[Any, ...]:
    """What a reader sees of a count table: the visible cells, the hidden mask, every line total."""
    hidden = _hidden_cells(rows, columns_first)
    shown = tuple(
        tuple(None if (i, j) in hidden else value for j, value in enumerate(row)) for i, row in enumerate(rows)
    )
    return shown, tuple(sum(row) for row in rows), tuple(sum(column) for column in zip(*rows, strict=True))


def _same_totals(rows: list[list[int]], hidden_columns: list[int]) -> Iterator[list[list[int]]]:
    """Every two-row table with the same visible cells and the same row and column totals."""
    top, bottom = rows
    for values in _compositions(sum(top[j] for j in hidden_columns), (0,) * len(hidden_columns)):
        first, second = list(top), list(bottom)
        for j, value in zip(hidden_columns, values, strict=True):
            first[j], second[j] = value, top[j] + bottom[j] - value
        if min(second) >= 0:
            yield [first, second]


class TestCellsHiddenThroughAnotherLine:
    """A cell hidden through its column can be larger than every visible cell of its row."""

    @pytest.mark.parametrize(
        ("rows", "columns_first", "hidden_columns"),
        [
            ([[5, 50, 20, 30], [40, 8, 60, 70]], True, [0, 1]),
            ([[5, 50, 20, 30], [40, 8, 60, 70]], False, [0, 1, 2]),
            ([[10, 50, 20, 30], [40, 10, 60, 70]], True, [0, 1]),
            ([[10, 50, 20, 30], [40, 10, 60, 70]], False, [0, 1, 2]),
            ([[1, 50, 20, 30], [40, 1, 60, 70]], True, [0, 1]),
        ],
        ids=[
            "reviewer_example",
            "reviewer_example_rows_first",
            "boundary_ten",
            "boundary_ten_rows_first",
            "boundary_one",
        ],
    )
    def test_suppress_lines_largeCellHiddenByColumn_keepsRowVisibleAndPinsNothing(
        self, rows, columns_first, hidden_columns
    ):
        # Arrange: 50 is hidden as the complement of the small cell in its column, above the
        # row's 20 and 30. Checked columns first, the row then has 5 and 50 hidden and 20 and 30
        # visible; rows first, the row has already hidden 20 as its own complement. The reader
        # knows every row and column total.
        shown = _matrix_view(rows, columns_first)

        # Act
        hidden = _hidden_cells(rows, columns_first)
        same_answer = [m for m in _same_totals(rows, hidden_columns) if _matrix_view(m, columns_first) == shown]

        # Assert: 30 and 70 stay visible, and no small count has a single feasible value
        assert hidden == {(i, j) for i in range(2) for j in hidden_columns}
        for i, j in sorted(hidden):
            feasible = {m[i][j] for m in same_answer}
            assert rows[i][j] not in _SMALL or len(feasible) > 1, f"cell {(i, j)} pinned to {feasible}"

    def test_present_chiSquare_largeCellHiddenByColumn_showsTheLargestGroup(self):
        # Arrange: outcome levels as rows, four groups as columns (the reviewer's example).
        # Rows are checked first, so the first row hides 20 as the complement of 5.
        groups = ["A", "B", "C", "D"]
        counts = [[5, 50, 20, 30], [40, 8, 60, 70]]
        analysis = {
            "type": "comparison",
            "test_type": "chi_square",
            "outcome_col": "y",
            "group_col": "arm",
            "groups": groups,
            "group_sizes": {g: counts[0][j] + counts[1][j] for j, g in enumerate(groups)},
            "contingency": [{"y": str(i), **dict(zip(groups, row, strict=True))} for i, row in enumerate(counts)],
            "p_value": 0.03,
            "n_used": 283,
            "n_excluded": 0,
        }

        # Act
        rows = present_analysis(analysis)["table"]["rows"]

        # Assert
        assert rows == [
            {"y": "y=0", "A": "suppressed", "B": "suppressed", "C": "suppressed", "D": 30},
            {"y": "y=1", "A": "suppressed", "B": "suppressed", "C": "suppressed", "D": 70},
        ]

    @pytest.mark.parametrize("scale", [1, 1_000, 1_000_000], ids=["reviewer_example", "thousands", "millions"])
    def test_suppress_count_matrix_largeHiddenCell_workDoesNotGrowWithCounts(self, scale, monkeypatch):
        # Arrange: count the interval checks instead of timing them; the closed form does the
        # same work whatever the size of the counts
        calls: list[tuple[int, ...]] = []
        original = result_presentation._feasible

        def counting(*args: int) -> int:
            calls.append(args)
            return original(*args)

        monkeypatch.setattr(result_presentation, "_feasible", counting)
        rows = [[5, 50 * scale, 20 * scale, 30 * scale], [40 * scale, 8, 60 * scale, 70 * scale]]
        reference = [[5, 50, 20, 30], [40, 8, 60, 70]]

        # Act
        hidden = suppress_count_matrix(rows)
        scaled_calls = len(calls)
        calls.clear()
        suppress_count_matrix(reference)

        # Assert
        assert hidden == {(i, j) for i in range(2) for j in range(3)}
        assert scaled_calls == len(calls)

    @pytest.mark.parametrize("size", [10, 30, 60], ids=["10x10", "30x30", "60x60"])
    def test_suppress_count_matrix_largeTable_checksEachLineOncePerNewHiddenCell(self, size, monkeypatch):
        # Arrange: one small cell per row on the diagonal, so every row and column needs a
        # complement; count the line checks instead of timing them
        checks: list[int] = []
        original = result_presentation._Lines.exposed

        def counting(self: Any, i: int) -> bool:
            checks.append(i)
            return original(self, i)

        monkeypatch.setattr(result_presentation._Lines, "exposed", counting)
        rows = [[3 if i == j else 20 + i + j for j in range(size)] for i in range(size)]

        # Act
        hidden = suppress_count_matrix(rows)

        # Assert: a line is checked again only after it gains a hidden cell, so the checks grow
        # with lines times hidden cells, not with passes over the whole table
        lines = 2 * size
        assert {(i, i) for i in range(size)} <= hidden
        assert len(checks) <= lines + 2 * 2 * len(hidden)


def _single_value_by_enumeration(
    values: tuple[int, ...], floors: tuple[int, ...], caps: tuple[int, ...], limit: int, one_small: bool
) -> bool:
    """Reference for the closed form: list every assignment of the hidden cells a reader allows."""
    from itertools import product

    feasible = [
        cells
        for cells in product(*(range(floor, cap + 1) for floor, cap in zip(floors, caps, strict=True)))
        if sum(cells) == sum(values) and (not one_small or min(cells) < limit)
    ]
    return any(len({cells[i] for cells in feasible}) < 2 for i in range(len(values)))


class TestSingleValueClosedForm:
    @pytest.mark.parametrize(
        ("values", "floors", "caps", "one_small", "expected"),
        [
            ((10, 10), (1, 1), (10, 10), True, True),
            ((11, 10, 10), (1, 1, 1), (31, 31, 31), True, False),
            ((10, 1), (10, 1), (11, 11), True, True),
            ((5, 50), (1, 1), (19, 19), True, True),
            ((5, 50), (1, 1), (19, 60), False, False),
            ((10, 50), (1, 1), (19, 50), False, False),
            ((3, 20), (1, 1), (23, 23), True, False),
        ],
        ids=[
            "tens_below_eleven",
            "nothing_visible",
            "floor_pins_n",
            "cell_above_every_cap",
            "cell_capped_by_its_column",
            "column_cap_equals_cell",
            "small_and_complement",
        ],
    )
    def test_single_value_knownLines_matchesReaderModel(self, values, floors, caps, one_small, expected):
        # Arrange
        limit = 11

        # Act
        single = _single_value(list(values), list(floors), list(caps), limit, one_small)

        # Assert
        assert single is expected
        assert _single_value_by_enumeration(values, floors, caps, limit, one_small) is expected

    @pytest.mark.parametrize(
        ("cells", "one_small"), [(2, True), (2, False), (3, True), (3, False)], ids=["two", "two_any", "three", "any"]
    )
    def test_single_value_everySmallLine_matchesEnumeration(self, cells, one_small):
        # Arrange: every line of 2 or 3 hidden cells up to 7, threshold 4, floors 1-2, caps 3-8
        from itertools import product

        limit = 4
        lines = [
            (values, floors, caps)
            for values in product(range(1, 8), repeat=cells)
            for floors in product((1, 2), repeat=cells)
            for caps in product((3, 5, 8), repeat=cells)
            if all(f <= v for f, v in zip(floors, values, strict=True))
        ]

        # Act
        closed = [_single_value(list(v), list(f), list(c), limit, one_small) for v, f, c in lines]
        listed = [_single_value_by_enumeration(v, f, c, limit, one_small) for v, f, c in lines]

        # Assert
        assert closed == listed

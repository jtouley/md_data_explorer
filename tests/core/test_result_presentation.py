"""Tests for claim-first presentation of analysis results (core/result_presentation.py)."""

import math

import numpy as np
import pytest

from clinical_analytics.core.result_presentation import present_analysis, to_json_safe

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
                {"type": "survival", "time_col": "days", "event_col": "death", "median_survival": 412.0},
                "Median time to death was 412.0 (days).",
            ),
            (
                {"type": "survival", "time_col": "days", "event_col": "death", "median_survival": None},
                "Median time to death was not reached during follow-up (days).",
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
                "n=280; <11 excluded for missing data). Unadjusted comparison by treatment.",
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

        # Assert
        assert "20.9" not in presentation["claim"]
        assert "B not shown (fewer than 11 patients)" in presentation["claim"]
        rows = {row["arm"]: row for row in presentation["table"]["rows"]}
        assert rows["B"]["median days"] == "suppressed"
        assert rows["A"]["median days"] == 300.0

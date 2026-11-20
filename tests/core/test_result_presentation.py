"""Tests for claim-first presentation of analysis results (core/result_presentation.py)."""

import math

import numpy as np
import pytest

from clinical_analytics.core.result_presentation import counted_note, present_analysis, text_note, to_json_safe

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
                # Without group sizes the comparison is still shown (ADR015).
                "Mean ldl was 2.50 higher in drug than placebo, not statistically significant (t-test p=0.200; "
                "n=300). Unadjusted comparison by treatment.",
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
                    "contingency": [{"died": "yes", "a": 30, "b": 40}],
                    "table_min_cell": 30,
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
                    "term_sizes": {"age": None},
                    "cell_sizes": {"age": None},
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
                    "term_sizes": {"hdl": None},
                    "cell_sizes": {"hdl": None},
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
            "term_sizes": {"age": None},
            "cell_sizes": {"age": None},
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
            "term_sizes": {"treatment_placebo[T.True]": 40},
            "cell_sizes": {"treatment_placebo[T.True]": 40},
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
    """Claims name the test actually run, the analysed n and exclusions, small counts included."""

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
                "n=280; 4 excluded for missing data). Unadjusted comparison by treatment.",
            ),
            (
                {**T_TEST, "test_type": "welch_t_test", "n_used": None, "n_excluded": None},
                "Mean ldl was 16.14 higher in placebo than drug (95% CI 13.44 to 18.83; Welch t-test p<0.001). "
                "Unadjusted comparison by treatment.",
            ),
        ],
        ids=["exclusions_reported", "small_exclusion_shown", "n_missing_omitted"],
    )
    def test_present_analysis_welchComparison_reportsTestAndN(self, analysis, expected_claim):
        # Arrange: analysis dict from parametrize

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert claim == expected_claim

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(0, "0"), (1, "1"), (10, "10"), (11, "11"), (300, "300")],
        ids=["zero", "one", "ten", "eleven", "large"],
    )
    def test_present_analysis_countClaim_showsTheTrueCount(self, count, expected):
        # Arrange
        analysis = {"type": "count", "total_count": count}

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert claim == f"Total count: {expected}."

    def test_present_analysis_groupedCountTable_showsSmallGroupsUnderTheirLabels(self):
        # Arrange
        analysis = {
            "type": "count",
            "total_count": 120,
            "grouped_by": "sex",
            "group_counts": [{"sex": "F", "count": 115}, {"sex": "M", "count": 5}],
        }

        # Act
        table = present_analysis(analysis)["table"]

        # Assert: every group under its own label with its true count (ADR015)
        assert table["rows"] == [{"sex": "F", "count": 115}, {"sex": "M", "count": 5}]

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

        # Assert: the event is named; censored=5 is shown as it is
        assert presentation["claim"].startswith("Median time to Status = Dead was 400.0 (days)")
        assert presentation["table"]["rows"] == [{"n": 120, "events": 115, "censored": 5, "median days": 400.0}]

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


class TestSmallCountsShownOnScreen:
    """ADR015: every answer shows its true numbers on screen, counts of 1-10 included."""

    def test_present_comparison_chiSquareWithSmallCell_showsTestSharesAndCells(self):
        # Arrange: a 2x2 with a cell of 2 patients
        analysis = {
            "type": "comparison",
            "test_type": "fisher_exact",
            "outcome_col": "y",
            "group_col": "arm",
            "groups": ["A", "B"],
            "p_value": 0.03,
            "contingency": [{"y": "0", "A": 40, "B": 2}, {"y": "1", "A": 10, "B": 48}],
            "group_proportions": {"A": 0.2, "B": 0.96},
            "event_level": "1",
            "group_sizes": {"A": 50, "B": 50},
            "n_used": 100,
            "n_excluded": 0,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert "Fisher's exact test p=0.030" in presentation["claim"]
        assert "20% in A and 96% in B" in presentation["claim"]
        assert presentation["table"]["rows"] == [{"y": "y=0", "A": 40, "B": 2}, {"y": "y=1", "A": 10, "B": 48}]

    @pytest.mark.parametrize("small", [1, 3, 10], ids=["one", "three", "ten"])
    def test_present_tTest_smallGroup_showsItsMeanSizeAndTheComparison(self, small):
        # Arrange
        analysis = {**T_TEST, "group_sizes": {"drug": small, "placebo": 140}, "n_used": 140 + small}

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation["claim"].startswith("Mean ldl was 16.14 higher in placebo than drug")
        assert {"treatment": "drug", "mean ldl": 113.96, "n": small} in presentation["table"]["rows"]

    def test_present_predictor_smallLevel_showsTheModel(self):
        # Arrange: a level held by 3 patients (no sizes are needed any more)
        analysis = {
            "type": "predictor",
            "outcome_col": "died",
            "event_level": "1",
            "n_used": 100,
            "n_excluded": 0,
            "summary": {
                "Intercept": {},
                "site_West": {"Odds Ratio": 2.0, "CI Lower": 0.5, "CI Upper": 8.0, "P-Value": 0.3},
            },
            "significant_predictors": [],
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation["claim"].startswith("No predictor reached p<0.05")
        assert presentation["table"]["rows"][0]["odds ratio"] == 2.0

    def test_present_survival_smallGroup_showsMedianAndLogRank(self):
        # Arrange
        analysis = {
            "type": "survival",
            "time_col": "days",
            "event_col": "died",
            "group_col": "arm",
            "group_medians": {"A": 300.0, "B": 120.0},
            "group_sizes": {"A": 57, "B": 3},
            "logrank_p_value": 0.04,
            "n_used": 60,
            "n_excluded": 0,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert "A 300.0, B 120.0" in presentation["claim"]
        assert "log-rank p=0.040" in presentation["claim"]
        assert {"arm": "B", "n": 3, "median days": 120.0} in presentation["table"]["rows"]

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_present_descriptive_smallN_showsEveryStatistic(self, n):
        # Arrange
        analysis = {
            "type": "descriptive",
            "focused_variable": "ldl",
            "is_numeric": True,
            "mean": 120.0,
            "median": 119.0,
            "std": 3.0,
            "min": 115.0,
            "max": 125.0,
            "non_null_count": n,
            "null_count": 2,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert (
            presentation["claim"]
            == f"Mean ldl was 120.00 (SD 3.00; median 119.00; n={n}; 2 excluded for missing data)."
        )
        assert presentation["table"]["rows"][0]["n"] == n

    def test_present_frequency_smallLevel_shownUnderItsLabel(self):
        # Arrange
        analysis = {
            "type": "descriptive",
            "focused_variable": "ward",
            "is_numeric": False,
            "value_counts": [{"ward": "A", "count": 200}, {"ward": "ICU", "count": 3}],
            "row_count": 203,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation["claim"] == "Most common ward: A (200 patients)."
        assert {"ward": "ICU", "count": 3} in presentation["table"]["rows"]

    @pytest.mark.parametrize(("events", "n"), [(2, 100), (98, 100), (1, 3)], ids=["few_with", "few_without", "small_n"])
    def test_present_proportion_smallCounts_showsTheShare(self, events, n):
        # Arrange
        analysis = {
            "type": "proportion",
            "column": "died",
            "value": 1,
            "proportion": events / n,
            "events": events,
            "n_used": n,
            "n_excluded": 0,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation["claim"].startswith(f"{100 * events / n:.1f}% of patients had died=1 (n={n})")
        assert presentation["table"]["rows"] == [{"proportion": round(events / n, 4), "n": n}]

    def test_present_rate_smallGroup_showsItsRate(self):
        # Arrange
        analysis = {
            "type": "rate",
            "column": "died",
            "value": 1,
            "group_col": "site",
            "groups": [
                {"group": "N", "rate": 0.25, "events": 25, "n": 100},
                {"group": "S", "rate": 0.5, "events": 2, "n": 4},
            ],
            "n_used": 104,
            "n_excluded": 0,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert "N 25.0%, S 50.0%" in presentation["claim"]
        assert {"site": "S", "rate": 0.5, "n": 4} in presentation["table"]["rows"]

    def test_present_correlation_codedColumns_showsR(self):
        # Arrange
        analysis = {
            "type": "relationship",
            "correlations": [{"var1": "a", "var2": "b", "correlation": 0.6}],
            "n_observations": 30,
            "n_excluded": 0,
        }

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation["claim"] == "The strongest correlation was strong: a and b (r=0.60; n=30)."


class TestDataNotes:
    @pytest.mark.parametrize(
        ("note", "expected"),
        [
            (counted_note("Data check: x is missing on {amount} rows.", 3, 100), "Data check: x is missing on 3 rows."),
            (
                counted_note("Data check: x has {amount} of its values missing.", 7, 100, "percent"),
                "Data check: x has 7.0% of its values missing.",
            ),
            (text_note("Visit 2 date is recorded."), "Visit 2 date is recorded."),
            ("Plain note with 4 values.", "Plain note with 4 values."),
            ({"text": "", "counted": False}, None),
            ({"text": "{amount} and {amount}", "count": 1, "total": None, "form": "number"}, None),
        ],
        ids=["count", "percent", "text", "plain_string", "empty_dropped", "malformed_dropped"],
    )
    def test_present_analysis_dataNote_writtenWithItsTrueCount(self, note, expected):
        # Arrange
        analysis = {"type": "count", "total_count": 100, "data_notes": [note]}

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert claim == "Total count: 100." + (f" {expected}" if expected else "")

    @pytest.mark.parametrize(
        "make",
        [
            lambda: counted_note("no placeholder", 1, None),
            lambda: counted_note("{amount}", 1, None, "percent"),
            lambda: text_note("has {amount}"),
        ],
        ids=["no_placeholder", "percent_without_total", "text_with_placeholder"],
    )
    def test_note_constructors_malformed_raiseValueError(self, make):
        # Arrange: a malformed note (parametrized)

        # Act / Assert
        with pytest.raises(ValueError):
            make()

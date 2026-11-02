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
}


class TestPresentAnalysisClaims:
    @pytest.mark.parametrize(
        ("analysis", "expected_claim"),
        [
            (
                T_TEST,
                "Mean ldl was 16.14 higher in placebo than drug (95% CI -18.83 to -13.44; t-test p<0.001; n=300). "
                "Unadjusted comparison by treatment.",
            ),
            (
                {**T_TEST, "mean_diff": 2.5, "p_value": 0.2, "ci_lower": None, "ci_upper": None, "group_sizes": None},
                "Mean ldl was 2.50 higher in drug than placebo (t-test p=0.200). "
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
                "age was the strongest significant predictor of outcome (OR 1.05; 95% CI 1.02 to 1.08; p=0.001). "
                "Logistic regression on outcome with age.",
            ),
            (
                {
                    "type": "predictor",
                    "outcome_col": "outcome",
                    "summary": {"hdl": {"Odds Ratio": 1.0, "CI Lower": 0.9, "CI Upper": 1.1, "P-Value": 0.8}},
                    "significant_predictors": [],
                },
                "No predictor reached p<0.05 for outcome. Logistic regression on outcome with hdl.",
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
        assert presentation["claim"].startswith("treatment_placebo was the strongest significant predictor")
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

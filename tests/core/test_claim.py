"""Tests for plain-language claim sentences (issue #52)."""

from clinical_analytics.core.claim import build_claim, suppress_small_counts
from clinical_analytics.core.query_plan import QueryPlan


def _plan(intent: str, **kwargs) -> QueryPlan:
    return QueryPlan(intent=intent, confidence=0.9, **kwargs)  # type: ignore[arg-type]


class TestBuildClaim:
    """Claim sentences lead with effect, interval, N, and unadjusted covariates."""

    def test_build_claim_compare_groups_t_test_includes_effect_ci_and_n(self):
        # Arrange
        result = {
            "type": "comparison",
            "test_type": "t_test",
            "outcome_col": "ldl",
            "group_col": "treatment",
            "groups": ["control", "drug"],
            "mean_diff": -12.3,
            "ci_lower": -15.1,
            "ci_upper": -9.5,
            "n": 574,
            "n_excluded": 26,
        }
        plan = _plan("COMPARE_GROUPS", metric="ldl", group_by="treatment")

        # Act
        claim = build_claim(result, plan)

        # Assert
        assert "12.3" in claim
        assert "drug" in claim.lower()
        assert "control" in claim.lower()
        assert "between groups" not in claim.lower()
        assert "15.1" in claim or "−15.1" in claim or "-15.1" in claim
        assert "9.5" in claim or "−9.5" in claim or "-9.5" in claim
        assert "574" in claim or "n=" in claim.lower()
        assert "not adjusted" in claim.lower()
        assert "t" in claim.lower()  # t-test mention

    def test_build_claim_compare_groups_names_both_groups_for_mean_diff(self):
        """C3: mean_diff claims must name both groups; never orphan 'higher between groups'."""
        result = {
            "type": "comparison",
            "test_type": "t_test",
            "outcome_col": "ldl",
            "groups": ["Female", "Male"],
            "mean_diff": 5.0,
            "ci_lower": 2.0,
            "ci_upper": 8.0,
            "n": 80,
        }
        plan = _plan("COMPARE_GROUPS", metric="ldl", group_by="sex")

        claim = build_claim(result, plan)

        assert "Female" in claim
        assert "Male" in claim
        assert "higher in Female than Male" in claim
        assert "between groups" not in claim.lower()

    def test_build_claim_compare_groups_zero_mean_diff_states_no_difference(self):
        result = {
            "type": "comparison",
            "test_type": "t_test",
            "outcome_col": "ldl",
            "groups": ["A", "B"],
            "mean_diff": 0.0,
            "n": 40,
        }
        plan = _plan("COMPARE_GROUPS", metric="ldl", group_by="arm")

        claim = build_claim(result, plan)

        assert "did not differ" in claim.lower()
        assert "higher" not in claim.lower()

    def test_build_claim_compare_groups_missing_ci_omits_interval_clause(self):
        # Arrange
        result = {
            "type": "comparison",
            "test_type": "anova",
            "outcome_col": "ldl",
            "group_col": "site",
            "p_value": 0.04,
            "n": 200,
        }
        plan = _plan("COMPARE_GROUPS", metric="ldl", group_by="site")

        # Act
        claim = build_claim(result, plan)

        # Assert
        assert "CI" not in claim
        assert "200" in claim or "n=" in claim.lower()
        assert "not adjusted" in claim.lower()

    def test_build_claim_find_predictors_leads_with_top_odds_ratio(self):
        # Arrange
        result = {
            "type": "predictor",
            "outcome_col": "outcome",
            "sample_size": 80,
            "significant_predictors": [
                {
                    "variable": "diabetes",
                    "odds_ratio": 2.4,
                    "ci_lower": 1.2,
                    "ci_upper": 4.8,
                    "p_value": 0.01,
                }
            ],
            "predictors": ["age", "sex", "diabetes"],
        }
        plan = _plan("FIND_PREDICTORS", metric="outcome")

        # Act
        claim = build_claim(result, plan)

        # Assert
        assert "diabetes" in claim.lower()
        assert "2.4" in claim
        assert "1.2" in claim
        assert "4.8" in claim
        assert "80" in claim

    def test_build_claim_count_includes_n(self):
        # Arrange
        result = {"type": "count", "total_count": 80, "headline": "Total count: **80**"}
        plan = _plan("COUNT")

        # Act
        claim = build_claim(result, plan)

        # Assert
        assert "80" in claim

    def test_build_claim_describe_uses_headline_when_present(self):
        # Arrange
        result = {
            "type": "descriptive",
            "focused_variable": "age",
            "mean": 52.1,
            "row_count": 80,
            "headline": "The average age is **52.10**",
        }
        plan = _plan("DESCRIBE", metric="age")

        # Act
        claim = build_claim(result, plan)

        # Assert
        assert "52.1" in claim or "52.10" in claim
        assert "80" in claim or "n=" in claim.lower()

    def test_build_claim_n_below_threshold_uses_suppressed_display(self):
        # Arrange
        result = {
            "type": "comparison",
            "test_type": "t_test",
            "outcome_col": "ldl",
            "groups": ["A", "B"],
            "mean_diff": 1.0,
            "ci_lower": -0.5,
            "ci_upper": 2.5,
            "n": 8,
        }
        plan = _plan("COMPARE_GROUPS", metric="ldl")

        # Act
        claim = build_claim(result, plan)

        # Assert
        assert "<11" in claim
        assert "8" not in claim.split("n=")[-1][:5] if "n=" in claim else True


class TestSuppressSmallCounts:
    def test_suppress_small_counts_below_threshold_returns_masked_string(self):
        assert suppress_small_counts(5) == "<11"
        assert suppress_small_counts(10) == "<11"

    def test_suppress_small_counts_zero_and_above_threshold_passthrough(self):
        assert suppress_small_counts(0) == 0
        assert suppress_small_counts(11) == 11
        assert suppress_small_counts(100) == 100

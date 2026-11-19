"""A test statistic of a table of counts never gives back a cell of 1-10 patients.

The comparison of a binary or categorical outcome across groups showed its chi-square or Fisher
p-value even when a cell of the table held 1-10 patients (the table itself hid that cell). With
the margins other answers show (patients per group, patients with the event) and the direction a
claim states ("more common in women"), the p-value of a 2x2 is monotone in |ad - bc|, so it pins
the table and the hidden cell. The same holds for the log-rank test of groups by event and for a
correlation of two coded columns (phi). Each is now withheld when any non-empty cell of its table
holds 1-10 patients, by ``table_statistic_refusal``: compute does not record the statistic, and
the presenter withholds one that a payload carries anyway.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy import stats

from clinical_analytics.analysis.compute import (
    _table_min_cell,
    compute_analysis_by_type,
)
from clinical_analytics.analysis.grouping import (
    NO_COUNT_TABLE,
    SMALL_CELL_TEST_REFUSAL,
    SMALL_VALUE_MODEL_REFUSAL,
    UNRECORDED_CELLS_TEST_REFUSAL,
    table_statistic_refusal,
)
from clinical_analytics.core.analysis_types import AnalysisContext
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED, present_analysis

CohortFactory = Callable[..., pl.DataFrame]
_GROUP_N = 100  # patients per group: large, so only a cell can be small
_TEST_WORDS = ("p=", "p<", "chi-square", "Fisher", "log-rank", "significant", "expected count")


def _two_by_two(make_cohort_with_categorical: CohortFactory, women_events: int, men_events: int) -> pl.DataFrame:
    """200 patients, 100 women and 100 men (``treatment``), ``status`` Yes for the first
    ``women_events`` women and ``men_events`` men."""
    sex = ["Female"] * _GROUP_N + ["Male"] * _GROUP_N
    events = [i < women_events for i in range(_GROUP_N)] + [i < men_events for i in range(_GROUP_N)]
    return make_cohort_with_categorical(
        patient_ids=[f"P{i:03d}" for i in range(2 * _GROUP_N)],
        treatment=sex,
        status=["Yes" if flag else "No" for flag in events],
        ages=[30 + i % 50 for i in range(2 * _GROUP_N)],
    )


@pytest.fixture
def table_context(sample_context_compare: AnalysisContext) -> AnalysisContext:
    sample_context_compare.primary_variable = "status"
    sample_context_compare.grouping_variable = "treatment"
    return sample_context_compare


def _shows_a_test(claim: str | None) -> bool:
    return claim is not None and any(word in claim for word in _TEST_WORDS)


def _p_value(table: np.ndarray, test: str) -> float:
    if test == "fisher":
        return float(stats.fisher_exact(table)[1])
    return float(stats.chi2_contingency(table)[1])


class TestTableTestLeakRepro:
    @pytest.mark.parametrize("test", ["chi_square", "fisher"])
    def test_pValue_marginsAndDirection_pinSmallCell(self, test: str) -> None:
        """The attack this change closes: p-value (as shown, 3 decimals) + margins + direction
        leaves exactly one table, so the men's 7 events are given back."""
        # Arrange: 100 women (18 events), 100 men (7 events); the reader knows 100, 100, 25
        women_events, men_events, total_events = 18, 7, 25
        shown = f"{_p_value(np.array([[18, 82], [7, 93]]), test):.3f}"

        # Act: every table with those margins where the event is more common in women
        candidates = [
            a
            for a in range(total_events + 1)
            if a > total_events - a
            and f"{_p_value(np.array([[a, 100 - a], [total_events - a, 100 - total_events + a]]), test):.3f}" == shown
        ]

        # Assert
        assert candidates == [women_events]
        assert total_events - candidates[0] == men_events

    def test_comparison_cellOfSevenPatients_withholdsTestAndPValue(
        self, make_cohort_with_categorical: CohortFactory, table_context: AnalysisContext
    ) -> None:
        """Regression: main showed 'chi-square p=0.033' for this table."""
        # Arrange
        frame = _two_by_two(make_cohort_with_categorical, women_events=18, men_events=7)

        # Act
        result = compute_analysis_by_type(frame, table_context)
        presented = present_analysis(result)

        # Assert
        for key in ("p_value", "chi2", "dof", "test_type", "odds_ratio", "warning", "min_expected_count"):
            assert key not in result
        assert result["table_min_cell"] == 7
        assert presented["claim"] == SMALL_CELL_TEST_REFUSAL
        assert presented["table"] is not None  # the suppressed table is still shown

    def test_comparison_fisherCellOfThree_withholdsTestAndOddsRatio(
        self, make_cohort_with_categorical: CohortFactory, table_context: AnalysisContext
    ) -> None:
        # Arrange: 3 + 0 events, so an expected count is below 5 (the Fisher path)
        frame = _two_by_two(make_cohort_with_categorical, women_events=3, men_events=0)

        # Act
        result = compute_analysis_by_type(frame, table_context)

        # Assert
        assert "p_value" not in result
        assert "odds_ratio" not in result
        assert not _shows_a_test(present_analysis(result)["claim"])


class TestComparisonCellBoundaries:
    @pytest.mark.parametrize(
        ("men_events", "shown"),
        [
            pytest.param(1, False, id="one"),
            pytest.param(SMALL_CELL_THRESHOLD - 1, False, id="ten_withheld"),
            pytest.param(SMALL_CELL_THRESHOLD, True, id="eleven_shown"),
            pytest.param(_GROUP_N - SMALL_CELL_THRESHOLD + 1, False, id="non_event_cell_ten"),
            pytest.param(_GROUP_N - SMALL_CELL_THRESHOLD, True, id="non_event_cell_eleven"),
        ],
    )
    def test_comparison_smallestCell_testShownOnlyFromThreshold(
        self,
        make_cohort_with_categorical: CohortFactory,
        table_context: AnalysisContext,
        men_events: int,
        shown: bool,
    ) -> None:
        # Arrange
        frame = _two_by_two(make_cohort_with_categorical, women_events=50, men_events=men_events)

        # Act
        result = compute_analysis_by_type(frame, table_context)
        claim = present_analysis(result)["claim"]

        # Assert
        assert ("p_value" in result) is shown
        assert _shows_a_test(claim) is shown
        assert (claim == SMALL_CELL_TEST_REFUSAL) is not shown

    @pytest.mark.parametrize(
        ("women_events", "men_events", "shown"),
        [
            pytest.param(30, 0, True, id="zero_cell_rest_large"),
            pytest.param(_GROUP_N, 50, True, id="zero_non_event_cell"),
            pytest.param(5, 0, False, id="zero_cell_beside_five"),
        ],
    )
    def test_comparison_zeroCell_isNotSmall(
        self,
        make_cohort_with_categorical: CohortFactory,
        table_context: AnalysisContext,
        women_events: int,
        men_events: int,
        shown: bool,
    ) -> None:
        # Arrange
        frame = _two_by_two(make_cohort_with_categorical, women_events, men_events)

        # Act
        result = compute_analysis_by_type(frame, table_context)

        # Assert
        assert ("p_value" in result) is shown
        assert _shows_a_test(present_analysis(result)["claim"]) is shown

    def test_comparison_threeByThreeTableWithSmallCell_withholdsChiSquare(
        self, make_cohort_with_categorical: CohortFactory, table_context: AnalysisContext
    ) -> None:
        # Arrange: three sites by three statuses, one site x status cell of 4 patients
        sites = ["East"] * 60 + ["North"] * 60 + ["West"] * 60
        statuses = (["A"] * 20 + ["B"] * 20 + ["C"] * 20) * 2 + ["A"] * 4 + ["B"] * 28 + ["C"] * 28
        frame = make_cohort_with_categorical(
            patient_ids=[f"P{i:03d}" for i in range(180)], treatment=sites, status=statuses, ages=[40] * 180
        )

        # Act
        result = compute_analysis_by_type(frame, table_context)

        # Assert
        assert result["table_min_cell"] == 4
        assert "chi2" not in result
        assert present_analysis(result)["claim"] == SMALL_CELL_TEST_REFUSAL


class TestComparisonVisitGrain:
    def _visits(self, make_cohort_with_categorical: CohortFactory, small_cell_keys: list[str | None]) -> pl.DataFrame:
        """Visits: 200 large-cell visits of 200 patients, plus the Male x Yes cell, one visit per
        entry of ``small_cell_keys`` (its patient key, or None)."""
        base = _two_by_two(make_cohort_with_categorical, women_events=40, men_events=0)
        extra = make_cohort_with_categorical(
            patient_ids=small_cell_keys,
            treatment=["Male"] * len(small_cell_keys),
            status=["Yes"] * len(small_cell_keys),
            ages=[50] * len(small_cell_keys),
        )
        return pl.concat([base, extra.with_columns(pl.col("patient_id").cast(pl.Utf8))])

    @pytest.mark.parametrize(
        ("small_cell_keys", "shown"),
        [
            pytest.param([f"V{i % 3}" for i in range(30)], False, id="thirty_visits_of_three_patients"),
            pytest.param([None] * 12, False, id="twelve_visits_all_null_key"),
            pytest.param([f"V{i}" for i in range(5)] + [None] * 10, False, id="five_patients_and_null_keys"),
            pytest.param([f"V{i % 11}" for i in range(22)], True, id="twentytwo_visits_of_eleven_patients"),
        ],
    )
    def test_comparison_visitRows_cellCountsPatients(
        self,
        make_cohort_with_categorical: CohortFactory,
        table_context: AnalysisContext,
        small_cell_keys: list[str | None],
        shown: bool,
    ) -> None:
        # Arrange: the cell's rows are at least 11 in every case; only its patients differ
        frame = self._visits(make_cohort_with_categorical, small_cell_keys)

        # Act
        result = compute_analysis_by_type(frame, table_context, entity_key="patient_id")

        # Assert
        assert ("p_value" in result) is shown
        assert _shows_a_test(present_analysis(result)["claim"]) is shown

    def test_comparison_entityKeyNotRead_refusesWholeAnswer(
        self, make_cohort_with_categorical: CohortFactory, table_context: AnalysisContext
    ) -> None:
        # Arrange
        frame = _two_by_two(make_cohort_with_categorical, 50, 50).drop("patient_id")

        # Act
        result = compute_analysis_by_type(frame, table_context, entity_key="patient_id")

        # Assert
        assert "error" in result
        assert "p_value" not in result


class TestPresenterWithholdsRecordedTest:
    def _payload(self, **overrides: Any) -> dict[str, Any]:
        """A comparison payload compute would show (every cell 50), before ``overrides``."""
        payload: dict[str, Any] = {
            "type": "comparison",
            "outcome_col": "status",
            "group_col": "treatment",
            "outcome_kind": "binary",
            "groups": ["Female", "Male"],
            "n_groups": 2,
            "group_sizes": {"Female": 100, "Male": 100},
            "n_used": 200,
            "n_excluded": 0,
            "contingency": [
                {"status": "No", "Female": 50, "Male": 50},
                {"status": "Yes", "Female": 50, "Male": 50},
            ],
            "contingency_index_col": "status",
            "table_min_cell": 50,
            "test_type": "chi_square",
            "chi2": 0.0,
            "p_value": 1.0,
            "dof": 1,
            "warning": "Some expected counts are below 5; the chi-square p-value may be unreliable.",
        }
        payload.update(overrides)
        return payload

    def test_presentComparison_largeCells_showsTest(self) -> None:
        # Act
        claim = present_analysis(self._payload())["claim"]

        # Assert
        assert claim is not None and "chi-square p=1.000" in claim

    @pytest.mark.parametrize(
        ("min_cell", "refusal"),
        [
            pytest.param(9, SMALL_CELL_TEST_REFUSAL, id="small"),
            pytest.param(SMALL_CELL_THRESHOLD - 1, SMALL_CELL_TEST_REFUSAL, id="ten"),
            pytest.param(0, SMALL_CELL_TEST_REFUSAL, id="zero_is_malformed"),
            pytest.param(-3, SMALL_CELL_TEST_REFUSAL, id="negative"),
            pytest.param(11.0, SMALL_CELL_TEST_REFUSAL, id="float"),
            pytest.param(True, SMALL_CELL_TEST_REFUSAL, id="bool"),
            pytest.param("11", SMALL_CELL_TEST_REFUSAL, id="text"),
            pytest.param(None, UNRECORDED_CELLS_TEST_REFUSAL, id="unrecorded"),
            pytest.param(NO_COUNT_TABLE, UNRECORDED_CELLS_TEST_REFUSAL, id="no_table_marker_on_a_table"),
        ],
    )
    def test_presentComparison_recordedCellNotSafe_withholdsTest(self, min_cell: Any, refusal: str) -> None:
        # Arrange: a payload that skipped compute's check still carries a p-value and a warning
        payload = self._payload(table_min_cell=min_cell)

        # Act
        claim = present_analysis(payload)["claim"]

        # Assert
        assert claim == refusal
        assert not _shows_a_test(claim)

    def test_presentComparison_storedPayloadWithoutCells_withheldAsUnrecorded(self) -> None:
        # Arrange
        payload = self._payload()
        del payload["table_min_cell"]

        # Act
        claim = present_analysis(payload)["claim"]

        # Assert
        assert claim == UNRECORDED_CELLS_TEST_REFUSAL


class TestSurvivalLogRank:
    def _survival(
        self, make_cohort_with_categorical: CohortFactory, women_events: int, men_events: int
    ) -> pl.DataFrame:
        """100 women and 100 men (``treatment``); ``status`` Yes is the event, at times
        (``age``) 1-25; everyone else is censored at 30."""
        events = [(sex, i < n) for sex, n in (("Female", women_events), ("Male", men_events)) for i in range(_GROUP_N)]
        return make_cohort_with_categorical(
            patient_ids=[f"P{i:03d}" for i in range(2 * _GROUP_N)],
            treatment=[sex for sex, _ in events],
            status=["Yes" if flag else "No" for _, flag in events],
            ages=[(i % 25) + 1 if flag else 30 for i, (_, flag) in enumerate(events)],
        )

    @pytest.fixture
    def survival_context(self, sample_context_survival: AnalysisContext) -> AnalysisContext:
        sample_context_survival.time_variable = "age"
        sample_context_survival.event_variable = "status"
        sample_context_survival.grouping_variable = "treatment"
        return sample_context_survival

    @pytest.mark.parametrize(
        ("men_events", "shown"),
        [
            pytest.param(7, False, id="seven_events"),
            pytest.param(SMALL_CELL_THRESHOLD - 1, False, id="ten_events"),
            pytest.param(SMALL_CELL_THRESHOLD, True, id="eleven_events"),
            pytest.param(_GROUP_N - 5, False, id="five_censored"),
        ],
    )
    def test_survival_groupEventCell_logRankShownOnlyFromThreshold(
        self,
        make_cohort_with_categorical: CohortFactory,
        survival_context: AnalysisContext,
        men_events: int,
        shown: bool,
    ) -> None:
        # Arrange
        frame = self._survival(make_cohort_with_categorical, women_events=40, men_events=men_events)

        # Act
        result = compute_analysis_by_type(frame, survival_context)
        claim = present_analysis(result)["claim"]

        # Assert
        assert ("logrank_p_value" in result) is shown
        assert claim is not None
        assert ("log-rank" in claim) is shown
        assert (SMALL_CELL_TEST_REFUSAL in claim) is not shown

    def test_survival_storedLogRankWithoutCells_withheld(
        self, make_cohort_with_categorical: CohortFactory, survival_context: AnalysisContext
    ) -> None:
        # Arrange
        result = compute_analysis_by_type(self._survival(make_cohort_with_categorical, 40, 40), survival_context)
        del result["table_min_cell"]

        # Act
        claim = present_analysis(result)["claim"]

        # Assert
        assert claim is not None
        assert "log-rank" not in claim
        assert UNRECORDED_CELLS_TEST_REFUSAL in claim


class TestRelationshipPhi:
    def _binary_pair(self, make_cohort_with_categorical: CohortFactory, both_yes: int) -> pl.DataFrame:
        """200 patients: 0/1 columns a (``treatment`` A) and b (``status`` Yes); a = 1 for the
        first 100, b = 1 for ``both_yes`` of them and 50 others; ages vary."""
        cohort = make_cohort_with_categorical(
            patient_ids=[f"P{i:03d}" for i in range(200)],
            treatment=["A"] * 100 + ["B"] * 100,
            status=["Yes" if i < both_yes else "No" for i in range(100)]
            + ["Yes" if i < 50 else "No" for i in range(100)],
            ages=[20 + i % 60 for i in range(200)],
        )
        return cohort.with_columns(
            (pl.col("treatment") == "A").cast(pl.Int64).alias("a"),
            (pl.col("status") == "Yes").cast(pl.Int64).alias("b"),
        )

    @pytest.fixture
    def pair_context(self, sample_context_relationship: AnalysisContext) -> AnalysisContext:
        sample_context_relationship.predictor_variables = ["a", "b", "age"]
        return sample_context_relationship

    def test_phi_marginsAndSign_pinSmallCell(self) -> None:
        """The attack: phi (r of two 0/1 columns, shown to 2 decimals) + margins + sign pins a."""

        # Arrange: a=1 for 100, b=1 for 57 (7 of them with a=1); the reader knows 100, 57, 200
        def phi(x: int) -> float:
            return float(
                np.corrcoef([1] * 100 + [0] * 100, [1] * x + [0] * (100 - x) + [1] * (57 - x) + [0] * (43 + x))[0, 1]
            )

        shown = f"{phi(7):.2f}"

        # Act
        candidates = [x for x in range(58) if x < 57 - x and f"{phi(x):.2f}" == shown]

        # Assert
        assert candidates == [7]

    @pytest.mark.parametrize(
        ("both_yes", "shown"),
        [
            pytest.param(7, False, id="seven"),
            pytest.param(SMALL_CELL_THRESHOLD - 1, False, id="ten"),
            pytest.param(SMALL_CELL_THRESHOLD, True, id="eleven"),
        ],
    )
    def test_relationship_codedPairSmallCell_withholdsR(
        self,
        make_cohort_with_categorical: CohortFactory,
        pair_context: AnalysisContext,
        both_yes: int,
        shown: bool,
    ) -> None:
        # Arrange
        frame = self._binary_pair(make_cohort_with_categorical, both_yes)

        # Act
        result = compute_analysis_by_type(frame, pair_context)
        presented = present_analysis(result)

        # Assert
        pair = next(c for c in result["correlations"] if (c["var1"], c["var2"]) == ("a", "b"))
        assert (pair["correlation"] is not None) is shown
        row = next(r for r in presented["table"]["rows"] if (r["variable 1"], r["variable 2"]) == ("a", "b"))
        assert (row["r"] != SECONDARY_SUPPRESSED) is shown
        assert presented["claim"] is not None
        assert (SMALL_CELL_TEST_REFUSAL in presented["claim"]) is not shown

    def test_relationship_measurementPair_shownAndSortedAboveWithheld(
        self, make_cohort_with_categorical: CohortFactory, pair_context: AnalysisContext
    ) -> None:
        # Arrange
        frame = self._binary_pair(make_cohort_with_categorical, 7)

        # Act
        result = compute_analysis_by_type(frame, pair_context)
        rows = present_analysis(result)["table"]["rows"]

        # Assert: pairs with age rest on no table and are shown; the withheld pair sorts last
        assert all(isinstance(r["r"], float) for r in rows if "age" in (r["variable 1"], r["variable 2"]))
        assert rows[-1]["r"] == SECONDARY_SUPPRESSED

    def test_relationship_storedPairsWithoutCells_withheld(
        self, make_cohort_with_categorical: CohortFactory, pair_context: AnalysisContext
    ) -> None:
        # Arrange
        result = compute_analysis_by_type(self._binary_pair(make_cohort_with_categorical, 40), pair_context)
        for pair in result["correlations"]:
            del pair["table_min_cell"]

        # Act
        presented = present_analysis(result)

        # Assert
        assert presented["claim"] == UNRECORDED_CELLS_TEST_REFUSAL
        assert all(r["r"] == SECONDARY_SUPPRESSED for r in presented["table"]["rows"])


class TestSharedRule:
    @pytest.mark.parametrize(
        ("min_cell", "may_rest_on_no_table", "expected"),
        [
            pytest.param(SMALL_CELL_THRESHOLD, False, None, id="threshold_shown"),
            pytest.param(np.int64(SMALL_CELL_THRESHOLD), False, None, id="numpy_count_shown"),
            pytest.param(SMALL_CELL_THRESHOLD - 1, False, SMALL_CELL_TEST_REFUSAL, id="ten_withheld"),
            pytest.param(1, False, SMALL_CELL_TEST_REFUSAL, id="one_withheld"),
            pytest.param(0, False, SMALL_CELL_TEST_REFUSAL, id="zero_malformed"),
            pytest.param(None, False, UNRECORDED_CELLS_TEST_REFUSAL, id="none_unrecorded"),
            pytest.param(NO_COUNT_TABLE, False, UNRECORDED_CELLS_TEST_REFUSAL, id="marker_rejected_for_table"),
            pytest.param(NO_COUNT_TABLE, True, None, id="marker_accepted_for_correlation"),
            pytest.param("other", True, SMALL_CELL_TEST_REFUSAL, id="other_text"),
        ],
    )
    def test_tableStatisticRefusal_values_failClosed(
        self, min_cell: object, may_rest_on_no_table: bool, expected: str | None
    ) -> None:
        # Act
        refusal = table_statistic_refusal(min_cell, may_rest_on_no_table=may_rest_on_no_table)

        # Assert
        assert refusal == expected

    def test_refusals_nameNoColumnOrLevel(self) -> None:
        # Assert
        for text in (SMALL_CELL_TEST_REFUSAL, UNRECORDED_CELLS_TEST_REFUSAL):
            assert not any(word in text for word in ("treatment", "status", "Female", "Male", "Yes"))

    def test_tableMinCell_valuesWithSeparators_neverJoinCells(self) -> None:
        # Arrange: joined as text, "a\x1f" + "b" and "a" + "\x1fb" would be one cell of 2
        rows = pl.Series(["a\x1f", "a"] + ["z"] * 20)
        columns = pl.Series(["b", "\x1fb"] + ["y"] * 20)

        # Act
        smallest = _table_min_cell(rows, columns)

        # Assert
        assert smallest == 1

    @pytest.mark.parametrize(
        ("rows", "columns"),
        [
            pytest.param([None, "a"], ["x", "y"], id="null_row"),
            pytest.param(["a", "b"], ["x", None], id="null_column"),
            pytest.param([], [], id="empty"),
        ],
    )
    def test_tableMinCell_nullsOrEmpty_raise(self, rows: list[Any], columns: list[Any]) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="table"):
            _table_min_cell(pl.Series(rows, dtype=pl.Utf8), pl.Series(columns, dtype=pl.Utf8))


class TestReviewFollowUps:
    def test_relationship_binaryAgainstUncodedIntegerCodes_withholdsR(
        self, make_cohort_with_categorical: CohortFactory, sample_context_relationship: AnalysisContext
    ) -> None:
        # Arrange: a 0/1 column against codes {1, 2, 4} (classified as a measurement); the cell
        # (a=1, code=4) holds 2 patients
        cohort = make_cohort_with_categorical(
            patient_ids=[f"P{i:03d}" for i in range(200)],
            treatment=["A"] * 100 + ["B"] * 100,
            ages=[4 if i < 2 else (1 if i < 50 else 2) for i in range(100)] + [4 if i < 40 else 1 for i in range(100)],
        )
        frame = cohort.with_columns((pl.col("treatment") == "A").cast(pl.Int64).alias("a"))
        sample_context_relationship.predictor_variables = ["a", "age"]

        # Act
        result = compute_analysis_by_type(frame, sample_context_relationship)

        # Assert
        pair = next(c for c in result["correlations"] if c["var1"] != c["var2"])
        assert pair["table_min_cell"] == 2
        assert pair["correlation"] is None
        assert present_analysis(result)["claim"] == SMALL_CELL_TEST_REFUSAL

    @pytest.mark.parametrize(
        ("men_at_five", "shown"),
        [
            pytest.param(3, False, id="three"),
            pytest.param(SMALL_CELL_THRESHOLD - 1, False, id="ten"),
            pytest.param(SMALL_CELL_THRESHOLD, True, id="eleven"),
        ],
    )
    def test_comparison_twoMeasuredValuesOutcome_meansFollowTableRule(
        self,
        make_cohort_with_categorical: CohortFactory,
        table_context: AnalysisContext,
        men_at_five: int,
        shown: bool,
    ) -> None:
        # Arrange: a dose of 0 or 5 (two measured values, a Welch t-test); a men's mean of 0.15
        # with n=100 shown would give back that 3 men had 5
        frame = _two_by_two(make_cohort_with_categorical, women_events=50, men_events=men_at_five).with_columns(
            pl.when(pl.col("status") == "Yes").then(5.0).otherwise(0.0).alias("status")
        )

        # Act
        result = compute_analysis_by_type(frame, table_context)
        presented = present_analysis(result)

        # Assert
        assert result["table_min_cell"] == men_at_five
        assert ("group_means" in result and "p_value" in result) is shown
        assert (result.get("error") == SMALL_CELL_TEST_REFUSAL) is not shown
        assert (presented["claim"] is not None) is shown  # an error is presented as no claim

    def test_presentComparison_storedTwoValuedMeansWithSmallCell_withheld(self) -> None:
        # Arrange: a t-test payload that skipped compute's check
        payload = {
            "type": "comparison",
            "test_type": "welch_t_test",
            "outcome_col": "dose",
            "group_col": "sex",
            "groups": ["F", "M"],
            "group_means": {"F": 2.5, "M": 0.15},
            "group_sizes": {"F": 100, "M": 100},
            "mean_diff": 2.35,
            "p_value": 0.001,
            "table_min_cell": 3,
            "n_used": 200,
            "n_excluded": 0,
        }

        # Act
        presented = present_analysis(payload)

        # Assert
        assert presented == {"claim": SMALL_CELL_TEST_REFUSAL, "table": None}

    def test_survival_ambiguousEventAtVisitGrain_countsPatientsNotVisits(
        self, make_cohort_with_categorical: CohortFactory, sample_context_survival: AnalysisContext
    ) -> None:
        # Arrange: event codes 2/3 (no stated event); code 2 is 5 patients over 15 visits
        cohort = make_cohort_with_categorical(
            patient_ids=[f"V{i % 5}" for i in range(15)] + [f"P{i:03d}" for i in range(60)],
            status=["2"] * 15 + ["3"] * 60,
            ages=[10 + i % 30 for i in range(75)],
        )
        frame = cohort.with_columns(pl.col("status").cast(pl.Int64))
        sample_context_survival.time_variable = "age"
        sample_context_survival.event_variable = "status"

        # Act
        result = compute_analysis_by_type(frame, sample_context_survival, entity_key="patient_id")

        # Assert: the refusal names no value: 5 patients hold code 2
        assert result["error"] == SMALL_VALUE_MODEL_REFUSAL

    def test_tableMinCell_negativeAndPositiveZero_areOneLevel(self) -> None:
        # Arrange: -0.0 and 0.0 are one level of the contingency table
        rows = pl.Series([-0.0] * 6 + [0.0] * 6 + [1.0] * 12)
        columns = pl.Series(["x"] * 24)

        # Act
        smallest = _table_min_cell(rows, columns)

        # Assert
        assert smallest == 12

"""Acceptance scorer: grades the presented answer, flags shown small cells as LEAK.

Synthetic answers only; the numbers below are made up.
"""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.api.services.query_service import UNEXPECTED_FAILURE_MESSAGE
from clinical_analytics.eval.corpus.run import _server_note, parse_tiers
from clinical_analytics.eval.corpus.score import ApiOutcome, OracleAnswer, find_leaks, score

SEX = {"id": "q-sex", "kind": "count_by", "group_by": "sex", "expect": "answer"}
SEX_COUNTS = OracleAnswer({"female": 900, "male": 240}, "patients", {("female",): 900, ("male",): 240})
SMALL_GROUP = OracleAnswer({"a": 500, "b": 7}, "patients", {("a",): 500, ("b",): 7})
GROUP_AB = {"id": "q-ab", "kind": "count_by", "group_by": "grp", "expect": "answer"}
RATES = {"id": "q-rate", "kind": "rates", "group_by": "sex", "expect": "answer"}
RATE_TRUTH = OracleAnswer(
    {"female": 14 / 900, "male": 1 / 240},
    "proportion",
    {
        ("female",): 900,
        ("female", "events"): 14,
        ("female", "non_events"): 886,
        ("male",): 240,
        ("male", "events"): 1,
        ("male", "non_events"): 239,
    },
)
ODDS = {"id": "q-or", "kind": "odds_ratio", "expect": "answer"}
ODDS_TRUTH = OracleAnswer({"or": 3.7947, "ci_low": 1.3398, "ci_high": 10.7477}, "odds ratio")
CHI = {"id": "q-chi", "kind": "p_value", "test": "chi-square", "expect": "answer"}
CHI_TRUTH = OracleAnswer({"pearson": 0.0346, "yates": 0.0512}, "p")
TOTAL = {"id": "q-total", "kind": "count", "expect": "answer"}
LISTING = {"id": "q-list", "kind": "listing", "expect": "held"}


def _done(presentation: dict[str, Any]) -> ApiOutcome:
    return ApiOutcome("completed", presentation=presentation)


def _or_table(odds: float, low: float, high: float) -> dict[str, Any]:
    columns = ["variable", "odds ratio", "95% CI low", "95% CI high", "p"]
    row = dict(zip(columns, ["comorbid", odds, low, high, 0.01], strict=True))
    return {"claim": "x", "table": {"columns": columns, "rows": [row]}}


class TestScoreCounts:
    @pytest.mark.parametrize(
        ("claim", "truth", "expected_grade"),
        [
            pytest.param("Total count: 1141 patients.", 1141, "RIGHT", id="exact"),
            pytest.param("Total count: 1140 patients.", 1141, "WRONG", id="off_by_one"),
            pytest.param("Total count: 1141 visits.", 1141, "WRONG", id="wrong_grain"),
            pytest.param("Total count: <11.", 7, "RIGHT", id="small_total_hidden"),
            pytest.param("Total count: <11.", 500, "HELD", id="large_total_withheld"),
            pytest.param("No answer.", 1141, "WRONG", id="nothing_shown"),
        ],
    )
    def test_score_totalCount_gradesPresentedClaim(self, claim, truth, expected_grade):
        # Arrange
        oracle = OracleAnswer(truth, "patients", {("total",): truth})

        # Act
        scored = score(TOTAL, oracle, _done({"claim": claim, "table": None}))

        # Assert
        assert scored.grade == expected_grade
        assert scored.leak is False

    def test_score_smallTotalShown_wrongAndLeak(self):
        # Arrange
        oracle = OracleAnswer(4, "patients", {("total",): 4})

        # Act
        scored = score(TOTAL, oracle, _done({"claim": "Total count: 4 patients.", "table": None}))

        # Assert
        assert (scored.grade, scored.leak) == ("WRONG", True)

    @pytest.mark.parametrize(
        ("rows", "expected_grade", "expected_leak"),
        [
            pytest.param([("female", 900), ("male", 240)], "RIGHT", False, id="all_match"),
            pytest.param([("female", 900), ("male", 241)], "WRONG", False, id="one_differs"),
            pytest.param([("female", 900)], "WRONG", False, id="large_row_missing"),
            pytest.param([("female", 900), ("male", 240), ("unknown", 3)], "WRONG", True, id="extra_small_row"),
            pytest.param([("female", "suppressed"), ("male", 240)], "RIGHT", False, id="secondary_hidden_allowed"),
            pytest.param([("female", "suppressed"), ("male", "suppressed")], "HELD", False, id="all_hidden"),
            pytest.param([("Other groups (hidden)", "suppressed")], "HELD", False, id="all_folded_no_size"),
            pytest.param([("female", 900), ("Other groups (hidden)", "suppressed")], "RIGHT", False, id="large_folded"),
            pytest.param(
                [("female", 900), ("Other (groups under 11)", "suppressed")], "WRONG", False, id="large_as_small"
            ),
        ],
    )
    def test_score_groupedCounts_gradesEveryRow(self, make_count_table, rows, expected_grade, expected_leak):
        # Arrange
        presentation = make_count_table("sex", rows)

        # Act
        scored = score(SEX, SEX_COUNTS, _done(presentation))

        # Assert
        assert scored.grade == expected_grade
        assert scored.leak is expected_leak

    @pytest.mark.parametrize(
        ("rows", "expected_grade", "expected_leak"),
        [
            pytest.param([("a", 500), ("Other (groups under 11)", "suppressed")], "RIGHT", False, id="small_folded"),
            pytest.param([("a", 500), ("b", "<11")], "WRONG", True, id="small_label_shown"),
            pytest.param([("a", 500), ("b", 7)], "WRONG", True, id="small_count_shown"),
        ],
    )
    def test_score_smallGroup_mustBeFoldedNotLabelled(self, make_count_table, rows, expected_grade, expected_leak):
        # Arrange
        presentation = make_count_table("grp", rows)

        # Act
        scored = score(GROUP_AB, SMALL_GROUP, _done(presentation))

        # Assert
        assert scored.grade == expected_grade
        assert scored.leak is expected_leak

    def test_score_countAskedRateShown_wrongUnit(self, make_count_table):
        # Arrange
        presentation = make_count_table("sex", [("female", 0.79), ("male", 0.21)], value_column="rate")

        # Act
        scored = score(SEX, SEX_COUNTS, _done(presentation))

        # Assert
        assert scored.grade == "WRONG"
        assert "unit" in scored.reason


class TestScoreRates:
    @pytest.mark.parametrize(
        ("female_rate", "male_rate", "expected_grade", "expected_leak"),
        [
            pytest.param(0.0156, "suppressed", "RIGHT", False, id="small_events_hidden"),
            pytest.param(0.0156, 0.0042, "WRONG", True, id="small_events_rate_shown"),
            pytest.param(0.0200, "suppressed", "WRONG", False, id="rate_differs"),
        ],
    )
    def test_score_rateWithSmallEvents_mustHideRate(
        self, make_count_table, female_rate, male_rate, expected_grade, expected_leak
    ):
        # Arrange
        presentation = make_count_table("sex", [("female", female_rate), ("male", male_rate)], value_column="rate")

        # Act
        scored = score(RATES, RATE_TRUTH, _done(presentation))

        # Assert
        assert scored.grade == expected_grade
        assert scored.leak is expected_leak


class TestScoreStatistics:
    @pytest.mark.parametrize(
        ("presentation", "expected_grade"),
        [
            pytest.param(_or_table(3.7947, 1.3398, 10.7477), "RIGHT", id="or_and_ci_match"),
            pytest.param(_or_table(3.7947, 1.20, 10.7477), "WRONG", id="ci_differs"),
            pytest.param({"claim": "The model is not shown.", "table": None}, "HELD", id="model_withheld"),
        ],
    )
    def test_score_oddsRatio_comparesOrAndCi(self, presentation, expected_grade):
        # Arrange / Act
        scored = score(ODDS, ODDS_TRUTH, _done(presentation))

        # Assert
        assert scored.grade == expected_grade

    @pytest.mark.parametrize(
        ("claim", "expected_grade"),
        [
            pytest.param("Confirmed was 6% in female; chi-square p=0.035. Unadjusted.", "RIGHT", id="pearson"),
            pytest.param("Confirmed differed by sex (chi-square p=0.051).", "RIGHT", id="yates"),
            pytest.param("Confirmed differed by sex (chi-square p=0.200).", "WRONG", id="differs"),
            pytest.param("Confirmed differed by sex (Fisher exact p=0.035).", "WRONG", id="other_test"),
            pytest.param("Confirmed differed by sex (chi-square).", "HELD", id="no_p"),
        ],
    )
    def test_score_pValue_matchesShownPrecision(self, claim, expected_grade):
        # Arrange / Act
        scored = score(CHI, CHI_TRUTH, _done({"claim": claim, "table": None}))

        # Assert
        assert scored.grade == expected_grade


class TestScoreStatuses:
    @pytest.mark.parametrize(
        ("question", "outcome", "expected_grade"),
        [
            pytest.param(TOTAL, ApiOutcome("timeout", error="no answer in 600s"), "ERROR", id="timeout"),
            pytest.param(TOTAL, ApiOutcome("error", error="POST -> 500"), "ERROR", id="http_error"),
            pytest.param(TOTAL, ApiOutcome("failed", error=UNEXPECTED_FAILURE_MESSAGE), "ERROR", id="exception"),
            pytest.param(TOTAL, ApiOutcome("awaiting_confirmation", failure_reason="say which"), "HELD", id="held"),
            pytest.param(TOTAL, ApiOutcome("failed", failure_reason="cannot do that"), "HELD", id="refused"),
            pytest.param(LISTING, ApiOutcome("awaiting_confirmation", failure_reason="no"), "RIGHT", id="held_ok"),
            pytest.param(
                LISTING, _done({"claim": "Total count: 400 patients.", "table": None}), "WRONG", id="listing_answered"
            ),
            pytest.param(
                TOTAL,
                ApiOutcome(
                    "completed",
                    presentation={"claim": "Total count: 1141 patients.", "table": None},
                    silent_fallback="fell back (llm_parse_exception_fallback_to_stub)",
                ),
                "ERROR",
                id="silent_fallback",
            ),
        ],
    )
    def test_score_apiStatus_mapsToGrade(self, question, outcome, expected_grade):
        # Arrange
        oracle = OracleAnswer(1141, "patients", {("total",): 1141})

        # Act
        scored = score(question, oracle, outcome)

        # Assert
        assert scored.grade == expected_grade

    @pytest.mark.parametrize("presentation", [None, {}, {"claim": None, "table": None}], ids=["none", "empty", "nulls"])
    def test_findLeaks_nothingPresented_noLeaks(self, presentation):
        # Arrange / Act
        leaks = find_leaks(SEX, SEX_COUNTS, presentation)

        # Assert
        assert leaks == []


class TestParseTiers:
    @pytest.mark.parametrize(
        ("log", "model", "expected"),
        [
            pytest.param(
                "t [info     ] query_parse_success  intent=COUNT tier=pattern_match\n",
                "llama3.2:3b",
                ("pattern_match", False, "not called"),
                id="pattern_tier",
            ),
            pytest.param(
                "t [info     ] parse_outcome  llm_called=True tier=tier3\n"
                "t [warning  ] llm_parse_exception_fallback_to_stub error_type=TypeError query_len=9\n"
                "t [info     ] query_parse_success  tier=llm_fallback\n",
                "llama3.2:3b",
                ("llm_fallback", True, "fell back (llm_parse_exception_fallback_to_stub: TypeError)"),
                id="model_crashed",
            ),
            pytest.param(
                "t [info     ] parse_outcome  llm_called=True tier=tier3\n"
                "t [warning  ] ollama_not_available_fallback_to_stub\n",
                "none",
                (None, True, "disabled"),
                id="no_model",
            ),
            pytest.param(
                "t [info     ] parse_outcome  llm_called=True tier=tier3\n"
                "t [info     ] query_parse_success  tier=llm_fallback\n",
                "llama3.2:3b",
                ("llm_fallback", True, "ran"),
                id="model_ran",
            ),
            pytest.param("", "none", (None, False, "not called"), id="empty_log"),
        ],
    )
    def test_parseTiers_serverLog_reportsTierAndModelStatus(self, log, model, expected):
        # Arrange / Act
        parsed = parse_tiers(log, model)

        # Assert
        assert parsed == expected


class TestServerNote:
    @pytest.mark.parametrize(
        ("code", "expected_flagged"),
        [
            pytest.param(None, False, id="still_running"),
            pytest.param(-15, False, id="stopped_by_harness"),
            pytest.param(-9, True, id="killed"),
            pytest.param(1, True, id="crashed"),
        ],
    )
    def test_serverNote_exitCode_flagsOnlyUnexpectedExit(self, code, expected_flagged):
        # Arrange / Act
        note = _server_note(code)

        # Assert
        assert ("exited with code" in note) is expected_flagged


MEAN = {"id": "q-mean", "kind": "mean_by", "group_by": "grp", "expect": "answer"}
MEAN_TRUTH = OracleAnswer({"a": 2011.5, "b": None}, "mean metric", {("a",): 400, ("b",): 0})
ODDS_UNDEFINED = OracleAnswer(None, "odds ratio", undefined="a cell of the 2x2 table is zero")
CHI_UNDEFINED = OracleAnswer(None, "p", undefined="a row or column of the 2x2 table is empty")


class TestScoreUndefined:
    @pytest.mark.parametrize(
        ("rows", "expected_grade"),
        [
            pytest.param([("a", 2011.5)], "RIGHT", id="no_value_group_left_out"),
            pytest.param([("a", 2011.5), ("b", None)], "RIGHT", id="no_value_group_shown_empty"),
            pytest.param([("a", 2011.5), ("b", 2000.0)], "WRONG", id="number_for_no_value_group"),
        ],
    )
    def test_score_groupWithNoValues_mustNotShowNumber(self, make_count_table, rows, expected_grade):
        # Arrange
        presentation = make_count_table("grp", rows, value_column="mean metric")

        # Act
        scored = score(MEAN, MEAN_TRUTH, _done(presentation))

        # Assert
        assert scored.grade == expected_grade

    @pytest.mark.parametrize(
        ("question", "truth", "presentation", "expected_grade"),
        [
            pytest.param(ODDS, ODDS_UNDEFINED, _or_table(2.0, 1.0, 4.0), "WRONG", id="or_shown"),
            pytest.param(ODDS, ODDS_UNDEFINED, {"claim": "Not estimable.", "table": None}, "HELD", id="or_withheld"),
            pytest.param(CHI, CHI_UNDEFINED, {"claim": "chi-square p=0.04.", "table": None}, "WRONG", id="p_shown"),
            pytest.param(CHI, CHI_UNDEFINED, {"claim": "chi-square: no test.", "table": None}, "HELD", id="p_withheld"),
        ],
    )
    def test_score_undefinedStatistic_numberWrongWithheldHeld(self, question, truth, presentation, expected_grade):
        # Arrange / Act
        scored = score(question, truth, _done(presentation))

        # Assert
        assert scored.grade == expected_grade

    def test_score_oracleError_gradedErrorWhateverShown(self):
        # Arrange
        truth = OracleAnswer(None, "mean metric", oracle_error="metric: 1 value(s) are neither numbers nor markers")

        # Act
        scored = score(MEAN, truth, _done({"claim": "x", "table": None}))

        # Assert
        assert scored.grade == "ERROR"
        assert "oracle" in scored.reason


class TestTotalCountReading:
    @pytest.mark.parametrize(
        ("claim", "expected_leak"),
        [
            pytest.param("Total count: 5 patients.", True, id="small_total"),
            pytest.param("Total count: 5.5 patients.", False, id="decimal_not_read_as_5"),
            pytest.param("Total count: 1141 patients. Unadjusted.", False, id="large_total"),
        ],
    )
    def test_findLeaks_totalClaim_readLikePresentedAnswer(self, claim, expected_leak):
        # Arrange
        presentation = {"claim": claim, "table": None}

        # Act
        leaks = find_leaks(TOTAL, OracleAnswer(1141, "patients"), presentation)

        # Assert
        assert bool(leaks) is expected_leak

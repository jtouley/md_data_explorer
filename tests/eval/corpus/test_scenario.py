"""Scenario roll-up: which abstract blanks a fellow gets filled, from unchanged question grades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from clinical_analytics.eval.corpus import run as run_module
from clinical_analytics.eval.corpus.run import AcceptanceError, QuestionResult, RunReport, markdown
from clinical_analytics.eval.corpus.scenario import (
    Scenario,
    ScenarioError,
    blank_results,
    blanks_line,
    parse_scenario,
    split,
)

SCENARIO: dict[str, Any] = {
    "case_study": 9,
    "title": "A test abstract",
    "abstract": "Cases [rose/fell], linked to [OR x], ranked [rank].",
    "blanks": [
        {"id": "trend", "text": "[rose/fell]"},
        {"id": "or", "text": "[OR x]"},
        {"id": "rank", "text": "[rank]"},
    ],
}


def _q(qid: str, *blanks: str, **extra: Any) -> dict[str, Any]:
    return {"id": qid, "query": f"question {qid}", "kind": "count", "blanks": list(blanks), **extra}


QUESTIONS = [
    _q("d-trend-01", "trend"),
    _q("d-positivity-02", "trend"),
    _q("d-or-03", "or"),
    _q("d-rank-04", "rank", needs="aggregate_mode"),
    {"id": "d-listing-05", "query": "list them", "kind": "listing", "expect": "held"},
]


@dataclass
class _Graded:
    id: str
    grade: str
    leak: bool = False


class TestParseScenario:
    def test_parseScenario_consistentFile_returnsScenario(self):
        # Arrange / Act
        scenario = parse_scenario({"scenario": SCENARIO}, QUESTIONS)

        # Assert
        assert scenario is not None
        assert [blank.id for blank in scenario.blanks] == ["trend", "or", "rank"]

    def test_parseScenario_noBlockNoBlanks_none(self):
        # Arrange / Act / Assert
        assert parse_scenario({}, [{"id": "a", "query": "q"}]) is None

    @pytest.mark.parametrize(
        ("raw", "questions", "message"),
        [
            pytest.param({}, QUESTIONS, "no scenario block", id="blanks_without_scenario"),
            pytest.param({"scenario": SCENARIO}, [*QUESTIONS, _q("x", "nope")], "unknown blank", id="unknown_blank"),
            pytest.param({"scenario": SCENARIO}, QUESTIONS[:3], r"\['rank'\] have no question", id="blank_unfilled"),
            pytest.param(
                {"scenario": SCENARIO},
                [*QUESTIONS, _q("h", "or", expect="held")],
                "must be held",
                id="held_fills_blank",
            ),
            pytest.param({"scenario": SCENARIO}, [*QUESTIONS, _q("n", needs="gbd")], "needs is 'gbd'", id="bad_needs"),
            pytest.param({"scenario": {**SCENARIO, "blanks": []}}, QUESTIONS, "scenario block", id="no_blanks"),
            pytest.param({"scenario": SCENARIO}, [*QUESTIONS, {"id": "s", "blanks": "or"}], "list", id="blanks_string"),
        ],
    )
    def test_parseScenario_inconsistent_raisesNamingTheProblem(self, raw, questions, message):
        # Arrange / Act / Assert
        with pytest.raises(ScenarioError, match=message):
            parse_scenario(raw, questions)


class TestSplit:
    def test_split_needsCapability_skippedNotAsked(self):
        # Arrange / Act
        asked, skipped = split(QUESTIONS)

        # Assert
        assert [q["id"] for q in asked] == ["d-trend-01", "d-positivity-02", "d-or-03", "d-listing-05"]
        assert [(s.id, s.needs) for s in skipped] == [("d-rank-04", "aggregate_mode")]

    def test_split_noNeeds_allAsked(self):
        # Arrange
        questions = [{"id": "a", "query": "q"}]

        # Act
        asked, skipped = split(questions)

        # Assert
        assert (asked, skipped) == (questions, [])


class TestBlankResults:
    @pytest.mark.parametrize(
        ("grades", "expected"),
        [
            pytest.param([("RIGHT", False), ("RIGHT", False)], "RIGHT", id="all_right"),
            pytest.param([("RIGHT", False), ("HELD", False)], "HELD", id="one_held"),
            pytest.param([("HELD", False), ("WRONG", False)], "WRONG", id="wrong_beats_held"),
            pytest.param([("ERROR", False), ("HELD", False)], "ERROR", id="error_beats_held"),
            pytest.param([("WRONG", True), ("WRONG", False)], "LEAK", id="leak_beats_wrong"),
        ],
    )
    def test_blankResults_trendBlank_worstQuestionDecides(self, grades, expected):
        # Arrange
        scenario = Scenario.model_validate(SCENARIO)
        results = [
            _Graded("d-trend-01", *grades[0]),
            _Graded("d-positivity-02", *grades[1]),
            _Graded("d-or-03", "RIGHT"),
        ]

        # Act
        rolled = {r.id: r for r in blank_results(scenario, QUESTIONS, results)}

        # Assert
        assert rolled["trend"].status == expected
        assert rolled["or"].status == "RIGHT"
        assert rolled["rank"].status == "WAITS"

    def test_blankResults_questionNotGraded_errorNamesIt(self):
        # Arrange
        scenario = Scenario.model_validate(SCENARIO)

        # Act
        rolled = {r.id: r for r in blank_results(scenario, QUESTIONS, [_Graded("d-trend-01", "RIGHT")])}

        # Assert
        assert rolled["trend"].status == "ERROR"
        assert "d-positivity-02" in rolled["trend"].note

    def test_blanksLine_countsAskableSeparatelyFromWaiting(self):
        # Arrange
        scenario = Scenario.model_validate(SCENARIO)
        results = [_Graded("d-trend-01", "RIGHT"), _Graded("d-positivity-02", "RIGHT"), _Graded("d-or-03", "WRONG")]

        # Act
        line = blanks_line(blank_results(scenario, QUESTIONS, results))

        # Assert
        assert line == (
            "Blanks filled right 1 of 2 askable · filled wrong or leaked 1 · waiting on aggregate mode 1 (of 3 blanks)"
        )


def _result(qid: str, grade: str) -> QuestionResult:
    return QuestionResult(qid, "f", "q", grade, False, "r", "completed", None, False, "not called", 0.1, False)


class TestRunnerScenario:
    def test_markdown_scenarioReport_addsBlanksTableAndSkipped(self):
        # Arrange
        scenario = Scenario.model_validate(SCENARIO)
        asked, skipped = split(QUESTIONS)
        results = [_result(q["id"], "RIGHT") for q in asked]
        report = RunReport("d", "none", "t", "abc", "2026-01-01T00:00:00+00:00", {}, results, skipped)
        report.scenario = scenario.model_dump()
        report.blanks = blank_results(scenario, QUESTIONS, results)

        # Act
        text = markdown(report)

        # Assert
        assert "#### Case study 9: A test abstract" in text
        assert "| [OR x] | RIGHT | or-03 RIGHT |" in text
        assert "Not asked (the app cannot do this yet): d-rank-04 (aggregate_mode)" in text

    def test_markdown_noScenario_noBlanksSection(self):
        # Arrange
        report = RunReport("d", "none", "t", "abc", "2026-01-01T00:00:00+00:00", {}, [_result("a", "RIGHT")])

        # Act
        text = markdown(report)

        # Assert
        assert "Blanks filled" not in text

    def test_loadQuestions_blankWithoutQuestion_acceptanceErrorBeforeAnyRun(self, tmp_path, monkeypatch):
        # Arrange
        (tmp_path / "d.yaml").write_text(
            "dataset_id: d\nscenario:\n  case_study: 1\n  title: t\n  abstract: a\n"
            "  blanks: [{id: x, text: '[x]'}]\nquestions:\n  - {id: q1, query: how many, kind: count}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(run_module, "QUESTIONS_DIR", tmp_path)

        # Act / Assert
        with pytest.raises(AcceptanceError, match="have no question"):
            run_module.load_questions("d")

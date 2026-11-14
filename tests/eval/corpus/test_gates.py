"""CI gates on the acceptance run (--fail-on, --baseline) and the corpus load check (--load-only).

Counts and load results are synthetic; no server is started and no dataset is fetched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from clinical_analytics.eval.corpus.fetch import load_corpus
from clinical_analytics.eval.corpus.run import (
    AcceptanceError,
    Baseline,
    LoadResult,
    QuestionResult,
    Rates,
    RunReport,
    check_load,
    gate_failures,
    gate_markdown,
    load_baseline,
    main,
    parse_fail_on,
)

BASELINES = Path(__file__).parent / "baselines"
FAIL_ON = frozenset({"WRONG", "LEAK", "ERROR"})


def _counts(**overrides: int) -> dict[str, int]:
    return {"RIGHT": 2, "WRONG": 0, "HELD": 13, "ERROR": 0, "LEAK": 0} | overrides


def _rates(answered_right: int = 1) -> Rates:
    return Rates(
        answered_right=answered_right,
        right_after_clarification=0,
        expect_answer=14,
        correctly_held=1,
        expect_held=1,
    )


def _baseline(answered_right: int = 1) -> Baseline:
    return Baseline(dataset="covid_ms", model="none", answered_right=answered_right)


def _result(grade: str, expect: str = "answer", confirmed: bool = False) -> QuestionResult:
    return QuestionResult(
        id=f"q-{grade}-{expect}-{confirmed}",
        family="count",
        query="synthetic",
        grade=grade,
        leak=False,
        reason="",
        status="completed",
        tier="pattern_match",
        llm_called=False,
        llm="not called",
        latency_s=0.1,
        confirmed=confirmed,
        expect=expect,  # type: ignore[arg-type]
    )


class TestGateFailures:
    def test_gate_failures_cleanRunAtBaseline_passes(self) -> None:
        # Arrange
        counts = _counts()

        # Act
        failures = gate_failures(counts, _rates(), FAIL_ON, _baseline())

        # Assert
        assert failures == []

    @pytest.mark.parametrize("grade", ["WRONG", "LEAK", "ERROR"])
    def test_gate_failures_oneForbiddenGrade_failsNamingIt(self, grade: str) -> None:
        # Arrange
        counts = _counts(**{grade: 1})

        # Act
        failures = gate_failures(counts, _rates(), FAIL_ON, None)

        # Assert
        assert failures == [
            f"1 question graded {grade}; the gate allows none. The results table gives the reason for each."
        ]

    @pytest.mark.parametrize(("answered", "fails"), [(0, True), (1, False), (2, False)])
    def test_gate_failures_answeredRightAgainstBaseline_failsOnlyBelow(self, answered: int, fails: bool) -> None:
        # Arrange
        rates = _rates(answered_right=answered)

        # Act
        failures = gate_failures(_counts(), rates, frozenset(), _baseline(1))

        # Assert
        assert bool(failures) is fails

    def test_gate_failures_rightHeldSteadyButAnswerLost_fails(self) -> None:
        # Arrange: RIGHT is unchanged (a new correct hold), but an answer became a hold.
        counts = _counts(RIGHT=2)
        rates = _rates(answered_right=0)

        # Act
        failures = gate_failures(counts, rates, FAIL_ON, _baseline(1))

        # Assert
        assert len(failures) == 1 and failures[0].startswith("Answered right fell to 0")

    def test_gate_failures_heldNotInFailOn_passes(self) -> None:
        # Arrange
        counts = _counts(HELD=15, RIGHT=0)

        # Act
        failures = gate_failures(counts, _rates(0), FAIL_ON, None)

        # Assert
        assert failures == []

    def test_gate_failures_belowBaseline_namesFileAndNextStep(self) -> None:
        # Arrange
        path = Path("tests/eval/corpus/baselines/covid_ms.none.json")

        # Act
        failures = gate_failures(_counts(), _rates(0), frozenset(), _baseline(1), path)

        # Assert
        assert failures == [
            "Answered right fell to 0, below the committed baseline of 1. "
            f"Fix the regression, or lower answered_right in {path} and say why in the pull request."
        ]

    @pytest.mark.parametrize(("model_runs", "fails"), [(None, False), (0, True), (1, False)])
    def test_gate_failures_modelRuns_failsOnlyWhenModelNeverRan(self, model_runs: int | None, fails: bool) -> None:
        # Act
        failures = gate_failures(_counts(), _rates(), FAIL_ON, None, None, model_runs)

        # Assert
        assert bool(failures) is fails

    def test_gate_markdown_answeredRightAboveBaseline_asksToRaiseIt(self) -> None:
        # Act
        text = gate_markdown([], _rates(3), _baseline(1))

        # Assert
        assert "Gate passed" in text
        assert "Raise answered_right in the baseline file" in text


class TestRates:
    def test_rates_mixedResults_keepsAnsweredRightAndCorrectlyHeldApart(self) -> None:
        # Arrange
        report = RunReport("covid_ms", "none", "test", "abc", "now", {})
        report.results = [
            _result("RIGHT"),
            _result("HELD"),
            _result("RIGHT", confirmed=True),
            _result("RIGHT", expect="held"),
            _result("WRONG", expect="held"),
        ]

        # Act
        rates = report.rates()

        # Assert
        assert rates == Rates(
            answered_right=1, right_after_clarification=1, expect_answer=3, correctly_held=1, expect_held=2
        )

    def test_rates_noResults_isAllZero(self) -> None:
        # Arrange
        report = RunReport("covid_ms", "none", "test", "abc", "now", {})

        # Act
        rates = report.rates()

        # Assert
        assert rates == Rates(0, 0, 0, 0, 0)


class TestParseFailOn:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("WRONG,LEAK,ERROR", {"WRONG", "LEAK", "ERROR"}),
            ("wrong, leak", {"WRONG", "LEAK"}),
            ("", set()),
        ],
    )
    def test_parse_fail_on_validNames_returnsGrades(self, text: str, expected: set[str]) -> None:
        # Act
        grades = parse_fail_on(text)

        # Assert
        assert grades == expected

    def test_parse_fail_on_typo_raises(self) -> None:
        # Act / Assert
        with pytest.raises(argparse.ArgumentTypeError, match="WRNG"):
            parse_fail_on("WRNG,LEAK")


class TestLoadBaseline:
    def test_load_baseline_committedCovidMsFile_isOneAggregateCount(self) -> None:
        # Arrange
        path = BASELINES / "covid_ms.none.json"

        # Act
        baseline = load_baseline(path, "covid_ms", "none")

        # Assert
        assert set(json.loads(path.read_text())) == {"dataset", "model", "answered_right"}
        assert baseline.answered_right >= 0

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"dataset": "covid_ms", "model": "llama3.2:3b", "answered_right": 1}, "not covid_ms with MODEL=none"),
            ({"dataset": "covid_ms", "model": "none"}, "could not be read"),
            ({"dataset": "covid_ms", "model": "none", "answered_right": -1}, "could not be read"),
            ({"dataset": "covid_ms", "model": "none", "answered_right": "1"}, "could not be read"),
            ({"dataset": "covid_ms", "model": "none", "answered_right": 1, "rows": [1]}, "could not be read"),
        ],
    )
    def test_load_baseline_wrongOrExtraContent_raises(self, tmp_path: Path, payload: dict, match: str) -> None:
        # Arrange
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(payload))

        # Act / Assert
        with pytest.raises(AcceptanceError, match=match):
            load_baseline(path, "covid_ms", "none")

    def test_load_baseline_missingFile_raises(self, tmp_path: Path) -> None:
        # Act / Assert
        with pytest.raises(AcceptanceError, match="could not be read"):
            load_baseline(tmp_path / "absent.json", "covid_ms", "none")


class TestMainFlags:
    @pytest.mark.parametrize("extra", [["--fail-on", "WRONG"], ["--baseline", "x.json"]])
    def test_main_loadOnlyWithQuestionGate_refusesToStart(self, extra: list[str]) -> None:
        # Act / Assert
        with pytest.raises(SystemExit) as raised:
            main(["--load-only", *extra])

        # Assert
        assert raised.value.code == 2


class TestLoadResult:
    def test_failures_matchingLoad_isEmpty(self) -> None:
        # Arrange
        result = LoadResult("covid_ms", "table", 1141, 47, status="ready", rows=1141, columns=47)

        # Act / Assert
        assert result.failures() == []

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"status": "failed"}, "status is failed"),
            ({"rows": 1140}, "loaded 1140 rows"),
            ({"columns": 0}, "loaded 0 columns"),
            ({"rows": None}, "loaded None rows"),
            ({"error": "POST /api/datasets/upload -> 400"}, "-> 400"),
        ],
    )
    def test_failures_mismatch_namesDatasetAndProblem(self, changes: dict, fragment: str) -> None:
        # Arrange
        result = LoadResult("covid_ms", "table", 1141, 47, status="ready", rows=1141, columns=47)
        for name, value in changes.items():
            setattr(result, name, value)

        # Act
        failures = result.failures()

        # Assert
        assert failures and all(f.startswith("covid_ms: ") for f in failures)
        assert any(fragment in f for f in failures)


class TestCheckLoad:
    def test_check_load_unknownDataset_isErrorWithoutServer(self) -> None:
        # Act
        result = check_load("nope", load_corpus(), None, [])

        # Assert
        assert result.error is not None and "not in corpus.yaml" in result.error
        assert result.failures() == [f"nope: {result.error}"]

    def test_check_load_noExpectations_isErrorWithoutServer(self, make_corpus_dataset) -> None:
        # Arrange
        dataset, _ = make_corpus_dataset()

        # Act
        result = check_load(dataset.name, {dataset.name: dataset}, None, [])

        # Assert
        assert result.error == "corpus.yaml has no rows or columns for it. Add both to check its load."

    def test_corpus_everyDataset_hasLoadExpectations(self) -> None:
        # Act
        corpus = load_corpus()

        # Assert
        assert all(d.rows is not None and d.columns is not None for d in corpus.values())

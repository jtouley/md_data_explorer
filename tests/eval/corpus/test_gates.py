"""CI gates on the acceptance run (--fail-on, --baseline) and the corpus load check (--load-only).

Counts and load results are synthetic; no server is started and no dataset is fetched.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from clinical_analytics.eval.corpus import run as run_module
from clinical_analytics.eval.corpus.fetch import CorpusConfigError, CorpusDataset, load_corpus
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
    positive_seconds,
    table_shape,
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
        ],
    )
    def test_parse_fail_on_validNames_returnsGrades(self, text: str, expected: set[str]) -> None:
        # Act
        grades = parse_fail_on(text)

        # Assert
        assert grades == expected

    @pytest.mark.parametrize(
        ("text", "match"),
        [("WRNG,LEAK", "WRNG"), ("", "no grade given"), (" , ", "no grade given")],
    )
    def test_parse_fail_on_typoOrEmpty_raises(self, text: str, match: str) -> None:
        # Act / Assert
        with pytest.raises(argparse.ArgumentTypeError, match=match):
            parse_fail_on(text)

    def test_main_emptyFailOn_refusesToStart(self) -> None:
        # Act
        with pytest.raises(SystemExit) as raised:
            main(["--fail-on", ""])

        # Assert
        assert raised.value.code == 2


class TestPositiveSeconds:
    def test_positive_seconds_positiveNumber_returnsFloat(self) -> None:
        # Act / Assert
        assert positive_seconds("120") == 120.0

    @pytest.mark.parametrize("text", ["0", "-1", "nan", "inf", "", "two"])
    def test_positive_seconds_zeroNegativeOrNotFinite_raises(self, text: str) -> None:
        # Act / Assert
        with pytest.raises(argparse.ArgumentTypeError):
            positive_seconds(text)


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
    def test_load_baseline_wrongOrExtraContent_raises(
        self, tmp_path: Path, payload: dict[str, Any], match: str
    ) -> None:
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
    def test_failures_mismatch_namesDatasetAndProblem(self, changes: dict[str, Any], fragment: str) -> None:
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

    def test_check_load_noExpectations_isErrorWithoutServer(
        self, make_corpus_dataset: Callable[..., tuple[CorpusDataset, dict[str, bytes]]]
    ) -> None:
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


class TestTableShape:
    def test_table_shape_oneTable_returnsRowsAndColumnCount(self) -> None:
        # Arrange
        detail = {"tables": [{"row_count": 1141, "columns": [{"name": "a"}, {"name": "b"}]}]}

        # Act
        shape = table_shape(detail)

        # Assert
        assert shape == (1141, 2)

    @pytest.mark.parametrize(
        "detail",
        [
            {},
            {"tables": None},
            {"tables": []},
            {"tables": [{"row_count": 1, "columns": []}] * 2},
            {"tables": ["not a table"]},
            {"tables": [{"row_count": None, "columns": []}]},
            {"tables": [{"row_count": True, "columns": []}]},
            {"tables": [{"row_count": 1, "columns": None}]},
        ],
    )
    def test_table_shape_malformed_raisesAcceptanceError(self, detail: dict[str, Any]) -> None:
        # Act / Assert
        with pytest.raises(AcceptanceError, match="GET /api/datasets"):
            table_shape(detail)


class TestMainLoadOnly:
    def test_main_loadOnlyMalformedCorpus_writesNamedErrorToSummary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        def broken() -> dict[str, CorpusDataset]:
            raise CorpusConfigError("corpus.yaml could not be loaded: bad")

        monkeypatch.setattr(run_module, "load_corpus", broken)
        summary = tmp_path / "summary.md"

        # Act
        code = main(["--load-only", "--summary", str(summary)])

        # Assert
        assert code == run_module.EXIT_NOT_RUN
        assert "upload check not run: corpus.yaml could not be loaded" in summary.read_text()


class _PendingApi:
    """Stands in for Api: the pending query never finishes, and each status check is counted."""

    def __init__(self) -> None:
        self.pending: str | None = "q-hung"
        self.checks = 0

    def request(self, method: str, path: str) -> dict[str, Any]:
        self.checks += 1
        return {"status": "running"}


class TestPendingQuery:
    def test_asked_pendingQueryStillRunning_isErrorWithoutWaiting(self) -> None:
        # Arrange
        api = _PendingApi()
        started = time.monotonic()

        # Act
        asked = run_module._asked(api, "ds", {"id": "q2", "query": "x"}, {}, 600.0)  # type: ignore[arg-type]

        # Assert
        assert time.monotonic() - started < 5
        assert api.checks == 1
        assert asked.outcome.status == "error"
        assert asked.outcome.error is not None and "q-hung" in asked.outcome.error
        assert api.pending == "q-hung"


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class _Opener:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def open(self, _request: object, timeout: float) -> _Response:
        return _Response(self.body)


class TestApiRequest:
    @pytest.mark.parametrize(
        ("body", "match"),
        [
            (b"<html>proxy error</html>", "not JSON"),
            (b'{"tables": [', "not JSON"),
            (b"\xff\xfe", "not JSON"),
            (b"[1, 2]", "JSON list, expected an object"),
        ],
    )
    def test_request_okStatusBadBody_raisesAcceptanceError(self, body: bytes, match: str) -> None:
        # Arrange
        api = run_module.Api.__new__(run_module.Api)
        api.base, api.token = "http://127.0.0.1:9", "t"
        api._opener = _Opener(body)  # type: ignore[assignment]

        # Act / Assert
        with pytest.raises(AcceptanceError, match=match):
            api.request("GET", "/api/datasets/x")

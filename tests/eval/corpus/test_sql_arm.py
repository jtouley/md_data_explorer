"""SQL comparison arms (raw and governed): their plumbing, with fake models only (no model server).

The arm's answers go through the same ``score`` as the app's; these tests check that each
answer shape reaches the grader as the app's would, that every failure is an ERROR, that the SQL
tool cannot touch files, and that the arm never runs in CI on pull requests.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from clinical_analytics.eval.corpus.run import AcceptanceError
from clinical_analytics.eval.corpus.score import OracleAnswer, score
from clinical_analytics.eval.corpus.sql_arm import (
    ArmError,
    CountAnswer,
    DeclineAnswer,
    GovernedSqlTool,
    ModelCallError,
    OddsRatioAnswer,
    PValueAnswer,
    RawSqlTool,
    TableAnswer,
    answer_question,
    comparison_markdown,
    main,
    ollama_model,
    parse_modes,
    presentation_of,
    run_arm,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTRACT_HEADER = "patient_id,visit_id,sex,flag"
EXTRACT_ROWS = ["P1,V1,female,yes", "P1,V2,female,no", "P2,V3,male,yes", "P3,V4,female,no"]


def _scripted(*replies: Any) -> tuple[Callable[[str, str], str], list[str]]:
    """A fake model that gives ``replies`` in order (dicts as JSON, exceptions raised)."""
    prompts: list[str] = []
    queue = list(replies)

    def model(system: str, prompt: str) -> str:
        prompts.append(prompt)
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    return model, prompts


@pytest.fixture
def sql_tool(write_extract) -> Iterator[RawSqlTool]:
    tool = RawSqlTool(write_extract(EXTRACT_HEADER, EXTRACT_ROWS))
    yield tool
    tool.close()


class TestPresentation:
    @pytest.mark.parametrize(
        ("answer", "question", "oracle"),
        [
            pytest.param(
                CountAnswer(type="count", value=42, unit="patients"),
                {"kind": "count"},
                OracleAnswer(42, "patients", {("total",): 42}),
                id="count",
            ),
            pytest.param(
                TableAnswer(
                    type="table", group_column="sex", value_column="count", rows=[("female", 20), ("male", 30)]
                ),
                {"kind": "count_by", "group_by": "sex"},
                OracleAnswer({"female": 20, "male": 30}, "patients", {("female",): 20, ("male",): 30}),
                id="count_by",
            ),
            pytest.param(
                TableAnswer(type="table", group_column="test_year", value_column="rate", rows=[(2021, 0.25)]),
                {"kind": "rates", "group_by": "test_year"},
                OracleAnswer(
                    {"2021": 0.25},
                    "proportion",
                    {("2021",): 40, ("2021", "events"): 10 + 10, ("2021", "non_events"): 20},
                ),
                id="rates_numeric_labels_shown_as_app",
            ),
            pytest.param(
                OddsRatioAnswer(type="odds_ratio", odds_ratio=2.5, ci_low=1.5, ci_high=4.0),
                {"kind": "odds_ratio"},
                OracleAnswer({"or": 2.5, "ci_low": 1.5, "ci_high": 4.0}, "odds ratio"),
                id="odds_ratio",
            ),
            pytest.param(
                PValueAnswer(type="p_value", test="chi-square", p=0.0312),
                {"kind": "p_value", "test": "chi-square"},
                OracleAnswer({"pearson": 0.03121, "yates": 0.05}, "p"),
                id="p_value",
            ),
        ],
    )
    def test_presentationOf_correctAnswer_gradedRight(self, answer, question, oracle):
        # Arrange
        outcome = presentation_of(answer)

        # Act
        scored = score({"id": "q", **question}, oracle, outcome)

        # Assert
        assert (scored.grade, scored.leak) == ("RIGHT", False), scored.reason

    def test_presentationOf_smallGroupShown_leak(self):
        # Arrange: the arm has no suppression; a group of 4 patients is shown.
        answer = TableAnswer(type="table", group_column="g", value_column="count", rows=[("a", 4), ("b", 40)])
        oracle = OracleAnswer({"a": 4, "b": 40}, "patients", {("a",): 4, ("b",): 40})

        # Act
        scored = score({"id": "q", "kind": "count_by", "group_by": "g"}, oracle, presentation_of(answer))

        # Assert
        assert (scored.grade, scored.leak) == ("WRONG", True)

    def test_presentationOf_decline_gradedLikeAppRefusal(self):
        # Arrange
        outcome = presentation_of(DeclineAnswer(type="decline", reason="patient listing"))

        # Act
        held = score({"id": "q", "kind": "listing", "expect": "held"}, OracleAnswer(None, "listing"), outcome)

        # Assert
        assert held.grade == "RIGHT"


class TestAnswerQuestion:
    def test_answerQuestion_sqlThenAnswer_resultReachesModelAndAnswerCompletes(self, sql_tool):
        # Arrange
        model, prompts = _scripted(
            {"sql": "SELECT count(DISTINCT patient_id) AS n FROM unified"},
            {"answer": {"type": "count", "value": 3, "unit": "patients"}},
        )

        # Act
        answer = answer_question(model, sql_tool, "How many patients?")

        # Assert
        assert answer.outcome.status == "completed"
        assert answer.outcome.presentation == {"claim": "Total count: 3 patients.", "table": None}
        assert "n\n3" in prompts[1]
        assert answer.steps == 2

    @pytest.mark.parametrize(
        ("replies", "message"),
        [
            pytest.param([ModelCallError("refused")], "refused", id="model_call_fails"),
            pytest.param(["not json"], "not JSON", id="not_json"),
            pytest.param([{"thought": "hm"}], "neither 'sql' nor 'answer'", id="no_protocol_key"),
            pytest.param([{"answer": {"type": "count", "value": "many"}}], "schema", id="answer_off_schema"),
            pytest.param([{"sql": "SELECT 1"}] * 3, "no final answer in 3 turns", id="step_limit"),
        ],
    )
    def test_answerQuestion_failure_errorOutcomeNamesWhy(self, sql_tool, replies, message):
        # Arrange
        model, _ = _scripted(*replies)

        # Act
        answer = answer_question(model, sql_tool, "How many?", max_steps=3)

        # Assert
        assert answer.outcome.status == "error"
        assert message in (answer.outcome.error or "")

    def test_answerQuestion_badSql_errorTextGoesBackToModel(self, sql_tool):
        # Arrange
        model, prompts = _scripted({"sql": "SELECT nope FROM unified"}, {"answer": {"type": "decline", "reason": "x"}})

        # Act
        answer = answer_question(model, sql_tool, "How many?")

        # Assert
        assert "SQL error" in prompts[1]
        assert answer.outcome.status == "failed"


class TestSqlTool:
    @pytest.mark.parametrize(
        "sql",
        [
            pytest.param("SELECT * FROM read_csv('/etc/hostname')", id="read_file"),
            pytest.param("COPY unified TO 'leak.csv'", id="write_file"),
            pytest.param("SET enable_external_access = true", id="unlock"),
        ],
    )
    def test_run_fileAccess_refused(self, sql_tool, sql):
        # Arrange / Act
        text = sql_tool.run(sql)

        # Assert
        assert text.startswith("SQL error")

    def test_run_manyRows_cappedWithNote(self, sql_tool):
        # Arrange / Act
        text = sql_tool.run("SELECT * FROM range(200)")

        # Assert
        assert text.count("\n") == 51
        assert text.endswith("(only the first 50 rows are shown)")

    def test_schema_listsColumns(self, sql_tool):
        # Arrange / Act / Assert
        assert sql_tool.schema().splitlines()[0] == "patient_id VARCHAR"


class _FakeOllama(BaseHTTPRequestHandler):
    bodies: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        reply = json.dumps({"model": body["model"], "response": '{"sql": "SELECT 1"}', "done": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args: Any) -> None:
        pass


class TestOllamaModel:
    def test_ollamaModel_loopbackServer_jsonModeNoStreamTemperatureZero(self):
        # Arrange
        _FakeOllama.bodies = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        model = ollama_model(f"http://127.0.0.1:{server.server_port}", "some-model", 5.0)

        # Act
        try:
            reply = model("system text", "prompt text")
        finally:
            server.shutdown()
            server.server_close()

        # Assert
        assert reply == '{"sql": "SELECT 1"}'
        (body,) = _FakeOllama.bodies
        assert body["format"] == "json"
        assert body["stream"] is False
        assert body["options"] == {"temperature": 0}
        assert (body["system"], body["prompt"], body["model"]) == ("system text", "prompt text", "some-model")

    @pytest.mark.parametrize(
        "url",
        [
            pytest.param("https://api.example.com", id="remote_host"),
            pytest.param("http://10.0.0.5:11434", id="lan_address"),
            pytest.param("file:///tmp/x", id="not_http"),
        ],
    )
    def test_ollamaModel_nonLoopbackUrl_refused(self, url):
        # Arrange / Act / Assert
        with pytest.raises(ArmError, match="not on this machine"):
            ollama_model(url, "m", 5.0)

    def test_ollamaModel_nothingListening_modelCallError(self):
        # Arrange
        model = ollama_model("http://127.0.0.1:9", "m", 2.0)

        # Act / Assert
        with pytest.raises(ModelCallError, match="model call failed"):
            model("s", "p")


class TestRunArm:
    def test_runArm_fellowHepb_scriptedModel_gradedWithBlanksAndListingRedacted(self, corpus_cache):
        # Arrange: one SQL turn and a final answer per question; the listing is answered (badly).
        # First matching phrase wins, so the by-region questions come first.
        answers = {
            "by region of birth": {"type": "decline", "reason": "not sure"},
            "How many adult patients": {"type": "count", "value": 8402, "unit": "patients"},
            "ever been screened": {"type": "count", "value": 3000, "unit": "rows"},
            "new hepatitis B diagnosis": {"type": "decline", "reason": "not sure"},
            "reactive surface antigen": {
                "type": "table",
                "group_column": "patient_id",
                "value_column": "count",
                "rows": [["LH000001", 1], ["LH000002", 1]],
            },
        }

        def model(system: str, prompt: str) -> str:
            question = prompt.split("Question: ", 1)[1].splitlines()[0]
            if "Result:" not in prompt:
                return json.dumps({"sql": "SELECT count(*) FROM unified"})
            return json.dumps({"answer": next(a for key, a in answers.items() if key in question)})

        # Act
        report = run_arm("fellow_hepb", "raw", model, "fake", "test")

        # Assert
        grades = {r.id.split("-", 1)[1]: (r.grade, r.leak) for r in report.results}
        assert grades["adults-seen-01"] == ("RIGHT", False)
        assert grades["ever-screened-02"] == ("WRONG", False)
        assert grades["new-diagnoses-03"] == ("HELD", False)
        assert grades["reactive-listing-06"] == ("WRONG", True)
        listing = next(r for r in report.results if r.id.endswith("listing-06"))
        assert "LH000001" not in listing.reason
        assert listing.table is None
        assert [s.id for s in report.skipped] == ["fellow_hepb-state-comparison-07"]
        assert {b.id: b.status for b in report.blanks}["screened_pct"] == "WRONG"
        assert (corpus_cache / "fellow_hepb" / "files" / "fellow_hepb.csv").is_file()

    def test_main_nonLoopbackModelUrl_exitsBeforeAnyQuestion(self, capsys):
        # Arrange / Act
        with pytest.raises(SystemExit) as exit_info:
            main(["--dataset", "fellow_cdiff", "--model", "m", "--ollama-base-url", "https://api.example.com"])

        # Assert
        assert exit_info.value.code == 2
        assert "not on this machine" in capsys.readouterr().err


class TestNeverInPullRequestCi:
    def test_workflows_neverRunTheArm(self):
        # Arrange
        workflows = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))

        # Act
        running = [p.name for p in workflows if "sql_arm" in p.read_text() or "accept-sql-arms" in p.read_text()]

        # Assert: the arm needs a model server and is on demand only.
        assert workflows
        assert running == []


class _FakeGovernedApi:
    """Stands in for the app's API: answers POST /api/queries/sql with a fixed body (or error)."""

    def __init__(self, reply: dict[str, Any] | Exception) -> None:
        self.reply, self.sent, self.stopped = reply, [], False

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.sent.append((path, payload))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def stop(self) -> None:
        self.stopped = True


class TestGovernedSqlTool:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            pytest.param(
                {"status": "answered", "columns": ["sex", "n"], "rows": [{"sex": "female", "n": 40}], "notes": []},
                "sex | n\nfemale | 40",
                id="answered",
            ),
            pytest.param(
                {"status": "answered", "columns": ["n"], "rows": [{"n": None}], "notes": ["Small counts hidden."]},
                "n\nNone\nNotes: Small counts hidden.",
                id="answered_with_note",
            ),
            pytest.param(
                {"status": "refused", "reason": "Rows are visits.", "needs_confirmation": None},
                "Refused: Rows are visits.",
                id="refused",
            ),
            pytest.param(AcceptanceError("POST /api/queries/sql -> 503: busy"), "SQL error: POST", id="http_error"),
        ],
    )
    def test_run_appReply_textForModel(self, reply, expected):
        # Arrange
        api = _FakeGovernedApi(reply)
        tool = GovernedSqlTool(api, "ds1", ["sex"])  # type: ignore[arg-type]

        # Act
        text = tool.run("SELECT sex, count(*) AS n FROM unified GROUP BY sex")

        # Assert
        assert text.startswith(expected)
        assert api.sent[0] == ("/api/queries/sql", {"dataset_id": "ds1", "sql": api.sent[0][1]["sql"]})

    def test_run_needsPatientConfirmation_neverConfirmsForTheUser(self):
        # Arrange
        reply = {
            "status": "refused",
            "reason": "Confirm one row per patient.",
            "needs_confirmation": "x",
            "confirmation": "t",
        }
        api = _FakeGovernedApi(reply)
        tool = GovernedSqlTool(api, "ds1", [])  # type: ignore[arg-type]

        # Act
        tool.run("SELECT count(*) FROM unified")
        tool.run("SELECT count(*) FROM unified")

        # Assert
        assert all("confirm_one_row_per_patient" not in payload for _, payload in api.sent)


class TestModesAndComparison:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("raw", ("raw",), id="one"),
            pytest.param("raw, app", ("app", "raw"), id="canonical_order"),
            pytest.param("app,governed,raw", ("app", "governed", "raw"), id="all"),
        ],
    )
    def test_parseModes_valid_canonicalOrder(self, text, expected):
        # Arrange / Act / Assert
        assert parse_modes(text) == expected

    @pytest.mark.parametrize("text", [pytest.param("", id="empty"), pytest.param("raw,mcp", id="unknown")])
    def test_parseModes_invalid_refused(self, text):
        # Arrange / Act / Assert
        with pytest.raises(argparse.ArgumentTypeError, match="use one or more of"):
            parse_modes(text)

    def test_comparisonMarkdown_threeColumns_separateRatesAndNotRun(self, corpus_cache):
        # Arrange: the raw arm declines everything; the governed column did not run.
        model, _ = _scripted(*[{"answer": {"type": "decline", "reason": "no"}}] * 10)
        raw = run_arm("fellow_measles", "raw", model, "fake", "test")

        # Act
        text = comparison_markdown("fellow_measles", {"app": None, "governed": None, "raw": raw})

        # Assert
        assert "| | app as-is | model + governed SQL | model + raw SQL |" in text
        assert "| blanks filled right | not run | not run | 0 of 2 |" in text
        assert "| answered right | not run | not run | 0 of 4 |" in text
        assert "| correctly held | not run | not run | 1 of 1 |" in text
        assert "waiting on aggregate mode (not asked in any column): 2 (of 4 blanks)" in text


@pytest.mark.slow
@pytest.mark.integration
class TestGovernedThroughTheApp:
    def test_runArm_governedOnVisitRows_everyAnswerHeldNoLeak(self, corpus_cache):
        # Arrange: a model that writes sensible SQL, then answers from whatever the tool returned.
        def model(system: str, prompt: str) -> str:
            if "Result:" not in prompt:
                return json.dumps({"sql": "SELECT count(DISTINCT patient_id) AS n FROM unified"})
            result = prompt.rsplit("Result:\n", 1)[1]
            if result.startswith(("Refused", "SQL error")):
                return json.dumps({"answer": {"type": "decline", "reason": result.splitlines()[0]}})
            return json.dumps({"answer": {"type": "count", "value": int(result.splitlines()[1]), "unit": "patients"}})

        # Act
        report = run_arm("fellow_cdiff", "governed", model, "fake", "test", app_src=REPO_ROOT / "src")

        # Assert: the governed path refuses visit-level rows today, so nothing is shown.
        counts = report.counts()
        assert (counts["WRONG"], counts["LEAK"], counts["ERROR"]) == (0, 0, 0)
        assert counts["HELD"] == report.rates().expect_answer
        assert all("visits" in r.reason for r in report.results if r.grade == "HELD")

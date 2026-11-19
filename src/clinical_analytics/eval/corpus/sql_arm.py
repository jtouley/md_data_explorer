# SPDX-License-Identifier: MIT
"""Comparison arms: the same questions, answered by a language model that writes SQL.

Two modes, graded by the same oracles and the same ``score.score`` as the app's own answers:

- ``raw``: the model has a plain SQL tool over the same file, in DuckDB, with no governance: no
  grain check, no small-cell suppression, no unit check, no hold. It stands in for "a general
  assistant plus a database connector", the bar the app must beat.
- ``governed``: the same model and prompt, but every statement goes through the app's governed SQL
  path (``POST /api/queries/sql`` on a fresh app with ``ENABLE_SQL_QUERIES`` on: checked against
  the data contract, run in the sandbox, small cells suppressed, refusals in plain language).

With ``app`` too (the default), one command gives three columns per scenario: the app as it
answers questions today, the model plus governed SQL, and the model plus raw SQL. The question
it answers: how many held answers does the governed path turn into right ones, at what WRONG and
LEAK cost compared with raw SQL.

Never run in CI on pull requests: it needs a model server and is slow. Run it on demand::

    make accept-sql-arms ARM_MODEL=<ollama model> [ARM_MODES=app,governed,raw] [ARM_DATASETS="fellow_cdiff ..."]

The model is reached through Ollama's ``POST /api/generate`` on a loopback address only, so any
local server that speaks that endpoint works, including a local shim to a hosted model; send it
synthetic or public data only. The tool protocol is plain JSON in the reply (one SQL statement or
one final answer per turn), so no model-specific tool-calling API is needed.

How a final answer becomes what the grader reads (the same shapes the app's API returns):

- ``count``      -> claim "Total count: N patients." (or visits)
- ``table``      -> a table [group column, count|rate|mean|median], labels shown as the app shows them
- ``odds_ratio`` -> a one-row table with "odds ratio", "95% CI low", "95% CI high"
- ``p_value``    -> claim "<test> test: p=<p>"
- ``decline``    -> a refusal (graded as the app's refusals are)
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Protocol, TypeAlias
from urllib.parse import urlsplit

import duckdb
import polars as pl
import structlog
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from clinical_analytics.analysis.variable_types import display_level
from clinical_analytics.eval.corpus.fetch import (
    CorpusFetchError,
    default_cache_dir,
    fetch_dataset,
    load_corpus,
    table_path,
)
from clinical_analytics.eval.corpus.run import (
    EXIT_GATE_FAILED,
    EXIT_NOT_RUN,
    NO_MODEL,
    REPO_ROOT,
    UNREACHABLE_OLLAMA,
    AcceptanceError,
    Api,
    QuestionResult,
    RunReport,
    _expect,
    _git_head,
    gate_failures,
    gate_markdown,
    load_oracles,
    load_questions,
    load_scenario,
    markdown,
    parse_fail_on,
    positive_seconds,
    public_reason,
    run,
    server_env,
    write_results,
)
from clinical_analytics.eval.corpus.scenario import blank_results, blanks_line, split
from clinical_analytics.eval.corpus.score import ApiOutcome, OracleAnswer, score
from clinical_analytics.sql.config import TABLE_NAME

logger = structlog.get_logger()

Mode: TypeAlias = Literal["app", "governed", "raw"]
MODES: Final[tuple[Mode, ...]] = ("app", "governed", "raw")
MODE_TITLES: Final[dict[Mode, str]] = {
    "app": "app as-is",
    "governed": "model + governed SQL",
    "raw": "model + raw SQL",
}
MAX_RESULT_ROWS: Final = 50
MAX_CELL_CHARS: Final = 80
DEFAULT_MAX_STEPS: Final = 8
LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "localhost", "::1"})

# (system prompt, prompt) -> the model's reply text.
ModelFn: TypeAlias = Callable[[str, str], str]


class ArmError(RuntimeError):
    """The arm could not run (bad model URL, model unreachable before any question)."""


class ModelCallError(RuntimeError):
    """One model call failed (refused, timed out, not JSON); that question is an ERROR."""


# ---------------------------------------------------------------------------- answer schema


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CountAnswer(_Strict):
    type: Literal["count"]
    value: int
    unit: Literal["patients", "visits", "rows"]


class TableAnswer(_Strict):
    type: Literal["table"]
    group_column: str
    value_column: Literal["count", "rate", "mean", "median"]
    rows: list[tuple[str | int | float | bool | None, int | float | None]]


class OddsRatioAnswer(_Strict):
    type: Literal["odds_ratio"]
    odds_ratio: float
    ci_low: float
    ci_high: float


class PValueAnswer(_Strict):
    type: Literal["p_value"]
    test: str
    p: float


class DeclineAnswer(_Strict):
    type: Literal["decline"]
    reason: str


FinalAnswer: TypeAlias = Annotated[
    CountAnswer | TableAnswer | OddsRatioAnswer | PValueAnswer | DeclineAnswer, Field(discriminator="type")
]
_ANSWER: Final[TypeAdapter[FinalAnswer]] = TypeAdapter(FinalAnswer)

SYSTEM_PROMPT: Final = f"""\
You are a data analyst helping a physician answer a question from a hospital data extract.
You have one tool: read-only SQL (DuckDB dialect) over one table named {TABLE_NAME}.
The tool may refuse a query or hide some values; it then says why.

Reply with exactly one JSON object per turn, and nothing else:
- To run SQL: {{"sql": "<one SELECT statement>"}}. You will get back up to {MAX_RESULT_ROWS} rows.
- To finish: {{"answer": <answer>}}, where <answer> is one of:
  {{"type": "count", "value": <integer>, "unit": "patients" | "visits" | "rows"}}
  {{"type": "table", "group_column": "<column you grouped by>", "value_column": "count" | "rate" | "mean" | "median",
    "rows": [["<group value>", <number>], ...]}}  (a rate is a proportion between 0 and 1)
  {{"type": "odds_ratio", "odds_ratio": <number>, "ci_low": <number>, "ci_high": <number>}}  (95% CI)
  {{"type": "p_value", "test": "<test name>", "p": <number>}}
  {{"type": "decline", "reason": "<why you will not answer>"}}
Group values in a table are the values exactly as they appear in the column.
"""


# ---------------------------------------------------------------------------- the SQL tool


class SqlTool(Protocol):
    """What the model's SQL runs on: the table's columns, and one statement's result as text."""

    def schema(self) -> str: ...

    def run(self, sql: str) -> str: ...

    def close(self) -> None: ...


class RawSqlTool:
    """Raw mode: the table in an in-memory DuckDB, with file and network access switched off
    once loaded. Every result is shown as computed."""

    def __init__(self, csv_path: Path) -> None:
        self.con = duckdb.connect(":memory:")
        self.con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM read_csv(?)", [str(csv_path)])
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    def schema(self) -> str:
        rows = self.con.execute(f"DESCRIBE {TABLE_NAME}").fetchall()
        return "\n".join(f"{name} {kind}" for name, kind, *_ in rows)

    def run(self, sql: str) -> str:
        """The first rows of the query's result as text, or the error message (for the model)."""
        try:
            cursor = self.con.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
        except duckdb.Error as error:
            return f"SQL error: {error}"
        return _rows_text(columns, [list(r) for r in rows[:MAX_RESULT_ROWS]], len(rows) - MAX_RESULT_ROWS)

    def close(self) -> None:
        self.con.close()


def _rows_text(columns: list[str], rows: list[list[Any]], extra: int = 0) -> str:
    lines = [" | ".join(columns)]
    lines += [" | ".join(str(v)[:MAX_CELL_CHARS] for v in row) for row in rows]
    if extra:
        lines.append(f"(only the first {MAX_RESULT_ROWS} rows are shown)")
    return "\n".join(lines)


class GovernedSqlTool:
    """Governed mode: each statement goes to ``POST /api/queries/sql`` on a fresh app (its own
    data home, ``ENABLE_SQL_QUERIES`` on, no model) holding the uploaded table, so the model gets
    exactly what a user typing that SQL would: the checked answer with small cells suppressed, or
    the plain-language refusal. One-row-per-patient confirmations are never sent: that is the
    data owner's statement to make, not the model's.

    Raises:
        AcceptanceError: the app does not start, or the upload or its contract fails.
    """

    def __init__(self, api: Api, dataset_id: str, columns: list[str], work: Path | None = None) -> None:
        self.api, self.dataset_id, self._columns, self.work = api, dataset_id, columns, work

    @classmethod
    def start(cls, csv_path: Path, dataset: str, app_src: Path | None = None) -> GovernedSqlTool:
        """A fresh app holding ``csv_path``, with its data contract written."""
        work = Path(tempfile.mkdtemp(prefix="mdde-governed-"))
        env = server_env(work, secrets.token_urlsafe(32), NO_MODEL, UNREACHABLE_OLLAMA, None, app_src)
        api = Api(work, env | {"ENABLE_SQL_QUERIES": "true"})
        try:
            api.wait_healthy()
            dataset_id = api.upload(csv_path, dataset)
            detail = api.request("GET", f"/api/datasets/{dataset_id}")
            # The quality report writes the data contract the governed path checks SQL against.
            api.request("GET", f"/api/datasets/{dataset_id}/quality")
        except AcceptanceError:
            api.stop()
            shutil.rmtree(work, ignore_errors=True)
            raise
        tables = detail.get("tables") or [{}]
        columns = [c if isinstance(c, str) else str(c.get("name")) for c in tables[0].get("columns") or []]
        return cls(api, dataset_id, columns, work)

    def schema(self) -> str:
        return "\n".join(self._columns)

    def run(self, sql: str) -> str:
        try:
            body = self.api.post_json("/api/queries/sql", {"dataset_id": self.dataset_id, "sql": sql})
        except AcceptanceError as error:
            return f"SQL error: {error}"
        if body.get("status") != "answered":
            return f"Refused: {body.get('reason') or 'no reason given'}"
        columns = [str(c) for c in body.get("columns") or []]
        rows = [[row.get(c) for c in columns] for row in body.get("rows") or []]
        text = _rows_text(columns, rows[:MAX_RESULT_ROWS], max(len(rows) - MAX_RESULT_ROWS, 0))
        notes = [str(n) for n in body.get("notes") or []]
        return text + ("\nNotes: " + " ".join(notes) if notes else "")

    def close(self) -> None:
        self.api.stop()
        if self.work is not None:
            shutil.rmtree(self.work, ignore_errors=True)


# ---------------------------------------------------------------------------- the model


def ollama_model(base_url: str, model: str, timeout_s: float) -> ModelFn:
    """A ModelFn over Ollama's /api/generate (JSON mode, temperature 0) on a loopback address.

    Raises:
        ArmError: ``base_url`` is not http(s) on a loopback host.
    """
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or parts.hostname not in LOOPBACK_HOSTS:
        raise ArmError(f"model URL {base_url!r} is not on this machine; use a loopback address such as 127.0.0.1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = base_url.rstrip("/") + "/api/generate"

    def call(system: str, prompt: str) -> str:
        body = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"}
        )
        try:
            with opener.open(request, timeout=timeout_s) as response:
                reply = json.loads(response.read())
        except (OSError, ValueError) as error:  # URLError, timeouts, resets; a body that is not JSON
            raise ModelCallError(f"model call failed: {type(error).__name__}: {error}") from error
        text = reply.get("response") if isinstance(reply, dict) else None
        if not isinstance(text, str):
            raise ModelCallError("model reply has no 'response' text")
        return text

    return call


# ---------------------------------------------------------------------------- one question


@dataclass
class ArmAnswer:
    outcome: ApiOutcome
    steps: int
    transcript: str
    latency_s: float


def _parse_turn(text: str) -> dict[str, Any]:
    try:
        turn = json.loads(text)
    except ValueError as error:
        raise ModelCallError(f"reply is not JSON: {text[:120]!r}") from error
    if not isinstance(turn, dict) or not ({"sql", "answer"} & set(turn)):
        raise ModelCallError(f"reply has neither 'sql' nor 'answer': {text[:120]!r}")
    return turn


def presentation_of(answer: FinalAnswer) -> ApiOutcome:
    """The final answer in the shape the app's API returns, so ``score`` reads both alike."""
    if isinstance(answer, DeclineAnswer):
        return ApiOutcome("failed", failure_reason=f"declined: {answer.reason}")
    if isinstance(answer, CountAnswer):
        grain = f" {answer.unit}" if answer.unit in ("patients", "visits") else ""
        return ApiOutcome("completed", {"claim": f"Total count: {answer.value}{grain}.", "table": None})
    if isinstance(answer, OddsRatioAnswer):
        columns = ["term", "odds ratio", "95% CI low", "95% CI high"]
        row = {"term": "exposure", "odds ratio": answer.odds_ratio, "95% CI low": answer.ci_low}
        row["95% CI high"] = answer.ci_high
        return ApiOutcome(
            "completed", {"claim": "Odds ratio with 95% CI.", "table": {"columns": columns, "rows": [row]}}
        )
    if isinstance(answer, PValueAnswer):
        return ApiOutcome("completed", {"claim": f"{answer.test} test: p={answer.p:.4f}", "table": None})
    group, value = answer.group_column, answer.value_column
    rows = [{group: display_level(group, label), value: number} for label, number in answer.rows]
    return ApiOutcome(
        "completed", {"claim": f"{value} by {group}.", "table": {"columns": [group, value], "rows": rows}}
    )


def answer_question(
    model: ModelFn, tool: SqlTool, question: str, context: str = "", max_steps: int = DEFAULT_MAX_STEPS
) -> ArmAnswer:
    """Let the model run SQL until it gives a final answer, at most ``max_steps`` turns.

    Every failure is an ERROR outcome: a model call that fails, a reply that is not the protocol,
    an answer that does not match the schema, or no answer within ``max_steps``."""
    started = time.monotonic()
    system = f"{SYSTEM_PROMPT}\nColumns of {TABLE_NAME}:\n{tool.schema()}\n"
    transcript = f"{context}Question: {question}\n"
    for step in range(1, max_steps + 1):
        try:
            turn = _parse_turn(model(system, transcript))
        except ModelCallError as error:
            return ArmAnswer(ApiOutcome("error", error=str(error)), step, transcript, time.monotonic() - started)
        if "answer" in turn:
            try:
                final = _ANSWER.validate_python(turn["answer"])
            except ValidationError as error:
                reason = f"final answer does not match the schema: {error.errors()[0]['msg']}"
                return ArmAnswer(ApiOutcome("error", error=reason), step, transcript, time.monotonic() - started)
            transcript += f"Answer: {json.dumps(turn['answer'])}\n"
            return ArmAnswer(presentation_of(final), step, transcript, time.monotonic() - started)
        sql = str(turn["sql"])
        transcript += f"SQL: {sql}\nResult:\n{tool.run(sql)}\n"
    reason = f"no final answer in {max_steps} turns"
    return ArmAnswer(ApiOutcome("error", error=reason), max_steps, transcript, time.monotonic() - started)


def ask_all_sql(
    model: ModelFn,
    tool: SqlTool,
    mode: Mode,
    questions: list[dict[str, Any]],
    truth: dict[str, OracleAnswer],
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[QuestionResult]:
    """Ask and grade every question in order with one SQL arm. A follow-up sees its parent's
    question and answer, as a chat would."""
    transcripts: dict[str, str] = {}
    results: list[QuestionResult] = []
    for question in questions:
        parent = question.get("follow_up_of")
        context = transcripts.get(parent, "") if parent else ""
        asked = answer_question(model, tool, question["query"], context, max_steps)
        transcripts[question["id"]] = asked.transcript
        scored = score(question, truth[question["id"]], asked.outcome)
        presentation = asked.outcome.presentation or {}
        reason = scored.reason
        if _expect(question) == "held" and asked.outcome.status == "completed":
            # It answered a question that must be held (a patient listing): the grader's reason
            # names the rows shown, which can be patient keys. Keep the grade, not the rows.
            rows = len((presentation.get("table") or {}).get("rows") or [])
            reason = f"answered a question that must be held: {rows} row(s) shown{'; LEAK' if scored.leak else ''}"
            presentation = {"claim": None, "table": None}
        result = QuestionResult(
            id=question["id"],
            family=question.get("family", ""),
            query=question["query"],
            grade=scored.grade,
            leak=scored.leak,
            reason=reason,
            status=asked.outcome.status,
            tier=f"{mode}_sql",
            llm_called=True,
            llm="error" if asked.outcome.status == "error" else "ran",
            latency_s=round(asked.latency_s, 2),
            confirmed=False,
            expect=_expect(question),
            claim=presentation.get("claim"),
            table=presentation.get("table"),
        )
        results.append(result)
        logger.info("arm_question", id=result.id, grade=result.grade, leak=result.leak, steps=asked.steps)
    return results


# ---------------------------------------------------------------------------- run


def run_arm(
    dataset: str,
    mode: Literal["governed", "raw"],
    model: ModelFn,
    model_name: str,
    label: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    cache_dir: Path | None = None,
    seed_dirs: list[Path] | None = None,
    app_src: Path | None = None,
) -> RunReport:
    """One SQL arm over one dataset's questions: same questions, file, oracles and grader as ``run``.

    Raises:
        AcceptanceError: the questions, oracles or data could not be loaded, or (governed) the
            app could not start or take the upload.
    """
    questions = load_questions(dataset)
    scenario = load_scenario(dataset)
    asked, skipped = split(questions)
    corpus = load_corpus()
    fetch_dataset(dataset, cache_dir, seed_dirs or [], corpus=corpus)
    csv_path = table_path(corpus[dataset], cache_dir)
    try:
        truth = load_oracles(dataset).answers(csv_path)
    except pl.exceptions.PolarsError as error:
        raise AcceptanceError(f"oracles for {dataset} do not run on {csv_path.name}: {error}") from error
    if missing := [q["id"] for q in asked if q["id"] not in truth]:
        raise AcceptanceError(f"no oracle answer for {missing}")
    upload = (
        "the table, to a fresh app; SQL through its governed path"
        if mode == "governed"
        else "none: SQL over the same CSV in DuckDB, no governance"
    )
    report = RunReport(
        dataset=dataset,
        model=f"{mode}-sql:{model_name}",
        label=label,
        git_head=_git_head(app_src.parent if app_src else REPO_ROOT),
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        settings={"upload": upload, "max_steps": max_steps, "mode": mode},
        skipped=skipped,
        scenario=scenario.model_dump() if scenario else None,
    )
    tool: SqlTool
    if mode == "governed":
        tool = GovernedSqlTool.start(csv_path, dataset, app_src)
    else:
        try:
            tool = RawSqlTool(csv_path)
        except duckdb.Error as error:
            raise AcceptanceError(f"{csv_path.name} could not be loaded into DuckDB: {error}") from error
    try:
        report.results = ask_all_sql(model, tool, mode, asked, truth, max_steps)
    finally:
        tool.close()
    if scenario is not None:
        report.blanks = blank_results(scenario, questions, report.results)
    return report


# ---------------------------------------------------------------------------- comparison


def _column(report: RunReport | None) -> list[str]:
    """One comparison column: blanks filled right, then the question grades."""
    if report is None:
        return ["not run"] * 7
    rates, counts = report.rates(), report.counts()
    askable = [b for b in report.blanks if b.status != "WAITS"]
    blanks = f"{sum(b.status == 'RIGHT' for b in askable)} of {len(askable)}" if report.blanks else "-"
    return [
        blanks,
        f"{rates.answered_right} of {rates.expect_answer}",
        f"{rates.right_after_clarification}",
        f"{counts['HELD']}",
        f"{counts['WRONG']}",
        f"{counts['LEAK']}",
        f"{rates.correctly_held} of {rates.expect_held}",
    ]


def comparison_markdown(dataset: str, reports: dict[Mode, RunReport | None]) -> str:
    """Three columns for one dataset: the app as-is, the model + governed SQL, the model + raw SQL.

    Rows: blanks filled right (of those askable today), answered right, right after one
    clarification, held, WRONG, LEAK, and correctly held (questions that must be held). Answered
    right and correctly held stay separate numbers, as in every acceptance table."""
    modes = [mode for mode in MODES if mode in reports]
    names = [
        "blanks filled right",
        "answered right",
        "right after one clarification",
        "HELD",
        "WRONG",
        "LEAK",
        "correctly held",
    ]
    columns = {mode: _column(reports[mode]) for mode in modes}
    lines = [
        f"### {dataset}: {' vs '.join(MODE_TITLES[m] for m in modes)}",
        "",
        "| | " + " | ".join(MODE_TITLES[m] for m in modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    lines += [f"| {name} | " + " | ".join(columns[m][i] for m in modes) + " |" for i, name in enumerate(names)]
    scenario_line = next((blanks_line(r.blanks) for r in reports.values() if r is not None and r.blanks), None)
    if scenario_line is not None:
        waiting = scenario_line.rsplit("waiting on aggregate mode ", 1)[-1]
        lines += ["", f"Blanks waiting on aggregate mode (not asked in any column): {waiting}"]
    return "\n".join(lines) + "\n"


def parse_modes(text: str) -> tuple[Mode, ...]:
    """``"app,raw"`` -> ("app", "raw"); argparse type, so a typo stops the run."""
    names = [part.strip().lower() for part in text.split(",") if part.strip()]
    unknown = sorted(set(names) - set(MODES))
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"use one or more of {', '.join(MODES)} (got {text!r})")
    return tuple(mode for mode in MODES if mode in names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True, help="corpus dataset; repeat for several")
    parser.add_argument("--model", required=True, help="model name the server at --ollama-base-url serves")
    parser.add_argument("--mode", type=parse_modes, default=MODES, help="any of app,governed,raw (default: all)")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434", help="loopback only")
    parser.add_argument("--timeout", type=positive_seconds, default=180.0, help="per model call (and app question), s")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="model turns per question")
    parser.add_argument("--label", default="sql-arms")
    parser.add_argument("--seed", type=Path, action="append", default=[], help="directory of already-fetched files")
    parser.add_argument("--results-dir", type=Path, default=None, help="default: <cache>/results")
    parser.add_argument("--app-src", type=Path, default=None, help="serve the app from another checkout's src/")
    parser.add_argument("--fail-on", type=parse_fail_on, default=frozenset(), help="exit 1 on these grades, any mode")
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    results_dir = args.results_dir or default_cache_dir() / "results"
    if results_dir.resolve().is_relative_to(REPO_ROOT):
        parser.error("--results-dir must be outside the repository")
    try:
        model = ollama_model(args.ollama_base_url, args.model, args.timeout)
    except ArmError as error:
        parser.error(str(error))
    app_src = args.app_src.resolve() if args.app_src else None
    status = 0
    for dataset in args.dataset:
        reports: dict[Mode, RunReport | None] = {}
        for mode in args.mode:
            try:
                if mode == "app":
                    report = run(
                        dataset,
                        args.model,
                        args.ollama_base_url,
                        args.timeout,
                        None,
                        args.label,
                        args.seed,
                        app_src=app_src,
                    )
                else:
                    report = run_arm(
                        dataset,
                        mode,
                        model,
                        args.model,
                        args.label,
                        args.max_steps,
                        seed_dirs=args.seed,
                        app_src=app_src,
                    )
            except (AcceptanceError, CorpusFetchError) as error:
                print(f"{MODE_TITLES[mode]} on {dataset} failed: {public_reason(str(error))}", file=sys.stderr)
                reports[mode] = None
                status = max(status, EXIT_NOT_RUN)
                continue
            reports[mode] = report
            out = write_results(report, results_dir)
            failures = gate_failures(report.counts(), report.rates(), args.fail_on, None)
            text = f"## {MODE_TITLES[mode]}\n\n" + markdown(report)
            if args.fail_on:
                text += "\n" + gate_markdown(failures, report.rates(), None)
            print(text)
            print(f"results: {out}\n")
            if failures:
                status = max(status, EXIT_GATE_FAILED)
        print(comparison_markdown(dataset, reports))
    return status


if __name__ == "__main__":
    sys.exit(main())

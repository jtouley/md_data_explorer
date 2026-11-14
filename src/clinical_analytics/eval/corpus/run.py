# SPDX-License-Identifier: MIT
"""Acceptance run: one public dataset, uploaded and asked through the real API.

Each run fetches the checked upload ZIP, starts a fresh API process (``python -m
clinical_analytics.api.serve``, the installed app's entry point) from an unrelated directory
with a new temporary ``CLINICAL_ANALYTICS_DATA_HOME`` and HOME, uploads the ZIP through
``POST /api/datasets/upload``, and asks every question through ``/api/queries``: each in a new
session, a follow-up in its parent's session. A held question is confirmed once only when its
YAML says ``confirm: true``. The presented answer is graded by ``eval.corpus.score``.

Results (JSON and a markdown table) go to a directory outside the repo; the table is printed.
Only aggregates are written: the claim and table the app presents, never rows.

Gates (CI): ``--fail-on WRONG,LEAK,ERROR`` exits 1 when any question has one of those grades
(and, with a model, when the model ran on no question). ``--baseline FILE`` exits 1 when
answered right falls below the committed count in FILE (one aggregate count, never row
values). The table reports answered right and correctly held apart (ADR013 §11.3).
``--load-only`` asks nothing: it uploads each ``--dataset`` into its own fresh app and checks
the load (status, rows, columns) against ``corpus.yaml``.

    make accept DATASET=covid_ms MODEL=llama3.2:3b OLLAMA_BASE_URL=http://127.0.0.1:11435
    make accept-load ACCEPT_LOAD_DATASETS="covid_ms"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeAlias

import polars as pl
import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from clinical_analytics.eval.corpus.fetch import (
    CorpusDataset,
    CorpusFetchError,
    default_cache_dir,
    fetch_dataset,
    load_corpus,
)
from clinical_analytics.eval.corpus.score import ApiOutcome, OracleAnswer, Scored, score

logger = structlog.get_logger()

REPO_ROOT = Path(__file__).resolve().parents[4]
QUESTIONS_DIR = REPO_ROOT / "tests" / "eval" / "corpus" / "questions"
ORACLES_DIR = REPO_ROOT / "tests" / "eval" / "corpus" / "oracles"
NO_MODEL = "none"
# Nothing listens here: with MODEL=none the language-model tier is configured but unreachable.
UNREACHABLE_OLLAMA = "http://127.0.0.1:9"
HEALTH_TIMEOUT_S = 300.0
POLL_S = 0.5
TERMINAL = {"completed", "failed", "awaiting_confirmation"}
# Log events (core/nl_query_engine.py) that mean the model tier was asked and did not answer.
LLM_FAILURE_EVENTS = (
    "llm_parse_exception_fallback_to_stub",
    "llm_parse_extraction_failed_fallback_to_stub",
    "ollama_generate_failed_fallback_to_stub",
    "ollama_not_available_fallback_to_stub",
    "llm_response_parse_failed",
)
_EVENT = re.compile(r"\]\s+(\S+)")
_TIER = re.compile(r"\btier=(\S+)")
_ERROR_TYPE = re.compile(r"\berror_type=(\w+)")
Grade: TypeAlias = Literal["RIGHT", "WRONG", "HELD", "ERROR", "LEAK"]
GRADES: tuple[Grade, ...] = ("RIGHT", "WRONG", "HELD", "ERROR", "LEAK")
# Exit codes: the run could not start or upload; a gate (--fail-on, --baseline, --load-only) failed.
EXIT_NOT_RUN = 2
EXIT_GATE_FAILED = 1


class AcceptanceError(RuntimeError):
    """The acceptance run could not start or upload (not a graded question failure)."""


Expect: TypeAlias = Literal["answer", "held"]


@dataclass(frozen=True)
class Rates:
    """Answered right (usefulness) and correctly held (safety), each over its own questions.

    ``answered_right`` excludes answers that needed a confirmation; those are
    ``right_after_clarification``. RIGHT alone mixes the two rates, so the gate never uses it."""

    answered_right: int
    right_after_clarification: int
    expect_answer: int
    correctly_held: int
    expect_held: int

    def line(self) -> str:
        return (
            f"Answered right {self.answered_right} of {self.expect_answer} · "
            f"right after one clarification {self.right_after_clarification} of {self.expect_answer} · "
            f"correctly held {self.correctly_held} of {self.expect_held}"
        )


@dataclass
class QuestionResult:
    id: str
    family: str
    query: str
    grade: str
    leak: bool
    reason: str
    status: str
    tier: str | None
    llm_called: bool
    llm: str
    latency_s: float
    confirmed: bool
    expect: Expect = "answer"
    claim: str | None = None
    table: dict[str, Any] | None = None


@dataclass
class RunReport:
    dataset: str
    model: str
    label: str
    git_head: str
    started_at: str
    settings: dict[str, Any]
    results: list[QuestionResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        tally = {grade: 0 for grade in ("RIGHT", "WRONG", "HELD", "ERROR")}
        for result in self.results:
            tally[result.grade] += 1
        tally["LEAK"] = sum(result.leak for result in self.results)
        return tally

    def rates(self) -> Rates:
        """The plan's separate rates (ADR013 §11.3): never add answered right and correctly held."""
        answer = [result for result in self.results if result.expect == "answer"]
        held = [result for result in self.results if result.expect == "held"]
        return Rates(
            answered_right=sum(r.grade == "RIGHT" and not r.confirmed for r in answer),
            right_after_clarification=sum(r.grade == "RIGHT" and r.confirmed for r in answer),
            expect_answer=len(answer),
            correctly_held=sum(r.grade == "RIGHT" for r in held),
            expect_held=len(held),
        )


# ---------------------------------------------------------------------------- questions/oracles


def load_questions(dataset: str) -> list[dict[str, Any]]:
    raw = yaml.safe_load((QUESTIONS_DIR / f"{dataset}.yaml").read_text(encoding="utf-8"))
    questions = [{"dataset_id": raw["dataset_id"], **item} for item in raw["questions"]]
    seen: set[str] = set()
    for question in questions:
        parent = question.get("follow_up_of")
        if parent is not None and parent not in seen:
            raise AcceptanceError(f"{question['id']} follows up {parent}, which is not asked before it")
        _expect(question)
        seen.add(question["id"])
    return questions


def load_oracles(dataset: str) -> ModuleType:
    path = ORACLES_DIR / f"{dataset}.py"
    spec = importlib.util.spec_from_file_location(f"corpus_oracles_{dataset}", path)
    if spec is None or spec.loader is None:
        raise AcceptanceError(f"no oracle module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------- API process


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_env(
    work: Path,
    token: str,
    model: str,
    ollama_base_url: str,
    llm_timeout_s: float | None,
    app_src: Path | None = None,
) -> dict[str, str]:
    """A clean environment: a new data home and HOME, the model settings, nothing inherited
    that could carry caches, overlays or declarations from another run. ``app_src`` serves the
    app from another checkout's ``src`` (same dependencies), to baseline another branch."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(work / "home"),
        "XDG_DATA_HOME": str(work / "xdg"),
        "CLINICAL_ANALYTICS_DATA_HOME": str(work / "data-home"),
        "CLINICAL_ANALYTICS_API_TOKEN": token,
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LANG": "C.UTF-8",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "OLLAMA_BASE_URL": UNREACHABLE_OLLAMA if model == NO_MODEL else ollama_base_url,
    }
    if app_src is not None:
        env["PYTHONPATH"] = str(app_src)
    if model != NO_MODEL:
        env |= {"OLLAMA_DEFAULT_MODEL": model, "OLLAMA_FALLBACK_MODEL": model}
    if llm_timeout_s is not None:
        seconds = str(llm_timeout_s)
        env |= {
            "OLLAMA_TIMEOUT_SECONDS": seconds,
            "LLM_TIMEOUT_PARSE_S": seconds,
            "LLM_TIMEOUT_FILTER_EXTRACTION_S": seconds,
            "LLM_TIMEOUT_MAX_S": seconds,
        }
    for name in ("HOME", "XDG_DATA_HOME", "CLINICAL_ANALYTICS_DATA_HOME"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    return env


class Api:
    """A fresh API server process and a small authenticated client for it."""

    def __init__(self, work: Path, env: dict[str, str]) -> None:
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.token = env["CLINICAL_ANALYTICS_API_TOKEN"]
        self.log_path = work / "server.log"
        cwd = work / "unrelated-cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(  # noqa: S603 - this interpreter, our own module
            [sys.executable, "-m", "clinical_analytics.api.serve", "--port", str(self.port)],
            cwd=cwd,
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        # A query still running after its question timed out (the API has no cancel). Until it
        # ends, its log lines would be read as the next question's.
        self.pending: str | None = None

    def wait_healthy(self) -> float:
        started = time.monotonic()
        while True:
            if self.proc.poll() is not None:
                raise AcceptanceError(f"server exited with code {self.proc.returncode} before /health")
            try:
                with self._opener.open(f"{self.base}/health", timeout=2) as response:
                    if response.status == 200:
                        return time.monotonic() - started
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            if time.monotonic() - started > HEALTH_TIMEOUT_S:
                raise AcceptanceError("timed out waiting for /health")
            time.sleep(POLL_S)

    def request(
        self, method: str, path: str, body: bytes | None = None, ctype: str | None = None, timeout: float = 120
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if ctype:
            headers["Content-Type"] = ctype
        request = urllib.request.Request(f"{self.base}{path}", data=body, method=method, headers=headers)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                return dict(json.loads(response.read() or b"{}"))
        except urllib.error.HTTPError as error:
            detail = error.read()[:300].decode("utf-8", "replace")
            raise AcceptanceError(f"{method} {path} -> {error.code}: {detail}") from error
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            code = self.proc.poll()
            state = f"server exited with code {code}" if code is not None else "server running"
            raise AcceptanceError(f"{method} {path}: {type(error).__name__} ({state})") from error

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", path, json.dumps(payload).encode(), "application/json")

    def upload(self, path: Path, name: str) -> str:
        response = self.upload_response(path, name)
        return required_field(response, "upload_id", "POST /api/datasets/upload")

    def upload_response(self, path: Path, name: str) -> dict[str, Any]:
        ctype = "application/zip" if path.suffix == ".zip" else "text/csv"
        boundary = uuid.uuid4().hex
        body = b"".join(
            [
                f'--{boundary}\r\nContent-Disposition: form-data; name="dataset_name"\r\n\r\n{name}\r\n'.encode(),
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n".encode(),
                path.read_bytes(),
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        return self.request("POST", "/api/datasets/upload", body, f"multipart/form-data; boundary={boundary}", 900)

    def log_offset(self) -> int:
        self._log.flush()
        return self.log_path.stat().st_size

    def log_since(self, offset: int) -> str:
        self._log.flush()
        with self.log_path.open("rb") as handle:
            handle.seek(offset)
            return handle.read().decode("utf-8", "replace")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self._log.close()


def required_field(response: dict[str, Any], name: str, request: str) -> str:
    """``response[name]`` as a non-empty string; AcceptanceError (graded, never a traceback) when
    the API answered 200 with a body that lacks it."""
    value = response.get(name)
    if not isinstance(value, str) or not value:
        raise AcceptanceError(f"{request} returned no {name} (keys: {sorted(response)})")
    return value


# ---------------------------------------------------------------------------- asking


@dataclass
class Asked:
    outcome: ApiOutcome
    latency_s: float
    confirmed: bool
    log: str


def _poll(api: Api, query_id: str, deadline: float) -> dict[str, Any] | None:
    while time.monotonic() < deadline:
        result = api.request("GET", f"/api/queries/{query_id}")
        if result.get("status") in TERMINAL:
            return result
        time.sleep(POLL_S)
    return None


def _submit(api: Api, payload: dict[str, Any]) -> str:
    return required_field(api.post_json("/api/queries", payload), "query_id", "POST /api/queries")


def ask(api: Api, dataset_id: str, session_id: str, question: dict[str, Any], timeout_s: float) -> Asked:
    """Submit one question (and, when held and ``confirm: true``, its confirmation).

    On a timeout the query is waited for up to another ``timeout_s`` (the API cannot cancel
    it), so its log lines stay with this question; if it is still running, ``api.pending``
    holds it and the next question waits for it first."""
    offset = api.log_offset()
    started = time.monotonic()
    deadline = started + timeout_s
    payload: dict[str, Any] = {"session_id": session_id, "dataset_id": dataset_id, "query_text": question["query"]}
    confirmed = False
    try:
        query_id = _submit(api, payload)
        result = _poll(api, query_id, deadline)
        if result is not None and result["status"] == "awaiting_confirmation" and question.get("confirm"):
            confirmed = True
            payload |= {
                "confirmed": True,
                "confirm_plan_hash": result.get("plan_hash"),
                "confirm_plan_line": result.get("plan_line"),
            }
            query_id = _submit(api, payload)
            result = _poll(api, query_id, deadline)
        latency = time.monotonic() - started
        if result is None:
            error = f"no answer in {timeout_s:.0f}s"
            if _poll(api, query_id, time.monotonic() + timeout_s) is None:
                api.pending = query_id
                error += f"; still running {timeout_s:.0f}s later"
            return Asked(ApiOutcome("timeout", error=error), latency, confirmed, api.log_since(offset))
    except AcceptanceError as error:
        outcome = ApiOutcome("error", error=str(error))
        return Asked(outcome, time.monotonic() - started, confirmed, api.log_since(offset))
    data = result.get("result_data") or {}
    outcome = ApiOutcome(
        status=result["status"],
        presentation={"claim": data.get("claim") or result.get("claim"), "table": data.get("table")},
        failure_reason=result.get("failure_reason"),
        error=result.get("error"),
    )
    return Asked(outcome, latency, confirmed, api.log_since(offset))


def parse_tiers(log: str, model: str) -> tuple[str | None, bool, str]:
    """(tier that produced the parse, whether the model tier was called, model-tier status)."""
    tier: str | None = None
    llm_called = False
    failure: str | None = None
    for line in log.splitlines():
        match = _EVENT.search(line)
        if not match:
            continue
        event = match.group(1)
        if event == "query_parse_success":
            found = _TIER.search(line)
            tier = found.group(1) if found else tier
        elif event == "query_parse_partial_match":
            tier = "below_threshold"
        elif event == "parse_outcome" and "llm_called=True" in line:
            llm_called = True
        elif event in LLM_FAILURE_EVENTS and failure is None:
            error_type = _ERROR_TYPE.search(line)
            failure = f"{event}: {error_type.group(1)}" if error_type else event
    if not llm_called:
        return tier, False, "not called"
    if failure is not None:
        return tier, True, "disabled" if model == NO_MODEL else f"fell back ({failure})"
    return tier, True, "ran"


# ---------------------------------------------------------------------------- run and report


def _git_head(checkout: Path) -> str:
    try:
        out = subprocess.run(  # noqa: S603, S607 - fixed git command
            ["git", "rev-parse", "--short", "HEAD"], cwd=checkout, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _asked(api: Api, dataset_id: str, question: dict[str, Any], sessions: dict[str, str], timeout_s: float) -> Asked:
    """Ask one question in its session; every failure to get that far is an ERROR outcome."""

    def failed(error: str) -> Asked:
        return Asked(ApiOutcome("error", error=error), 0.0, False, "")

    parent = question.get("follow_up_of")
    if parent is not None and parent not in sessions:
        return failed(f"session: parent question {parent} has no session (its own session could not be created)")
    try:
        if api.pending is not None:
            if _poll(api, api.pending, time.monotonic() + timeout_s) is None:
                return failed(
                    f"server still running timed-out query {api.pending}; its log would be read as this one's"
                )
            api.pending = None
        session_id = (
            sessions[parent]
            if parent is not None
            else required_field(
                api.post_json("/api/sessions", {"dataset_id": dataset_id}), "session_id", "POST /api/sessions"
            )
        )
    except AcceptanceError as error:
        return failed(f"session: {error}")
    sessions[question["id"]] = session_id
    return ask(api, dataset_id, session_id, question, timeout_s)


def _expect(question: dict[str, Any]) -> Expect:
    expect = question.get("expect", "answer")
    if expect == "answer":
        return "answer"
    if expect == "held":
        return "held"
    raise AcceptanceError(f"{question['id']}: expect is {expect!r}; use answer or held")


def ask_all(
    api: Api,
    dataset_id: str,
    questions: list[dict[str, Any]],
    truth: dict[str, OracleAnswer],
    model: str,
    timeout_s: float,
) -> list[QuestionResult]:
    """Ask and grade every question in order; one question's failure never stops the rest."""
    sessions: dict[str, str] = {}
    results: list[QuestionResult] = []
    for question in questions:
        asked = _asked(api, dataset_id, question, sessions, timeout_s)
        tier, llm_called, llm = parse_tiers(asked.log, model)
        outcome = asked.outcome
        if llm.startswith("fell back") and outcome.status == "completed":
            outcome = ApiOutcome(**{**asdict(outcome), "silent_fallback": llm})
        scored: Scored = score(question, truth[question["id"]], outcome)
        presentation = outcome.presentation or {}
        result = QuestionResult(
            id=question["id"],
            family=question.get("family", ""),
            query=question["query"],
            grade=scored.grade,
            leak=scored.leak,
            reason=scored.reason,
            status=outcome.status,
            tier=tier,
            llm_called=llm_called,
            llm=llm,
            latency_s=round(asked.latency_s, 2),
            confirmed=asked.confirmed,
            expect=_expect(question),
            claim=presentation.get("claim"),
            table=presentation.get("table"),
        )
        results.append(result)
        logger.info("accept_question", id=result.id, grade=result.grade, leak=result.leak, latency_s=result.latency_s)
    return results


def run(
    dataset: str,
    model: str,
    ollama_base_url: str,
    timeout_s: float,
    llm_timeout_s: float | None,
    label: str,
    seed_dirs: list[Path],
    cache_dir: Path | None = None,
    app_src: Path | None = None,
    keep_work: bool = False,
) -> RunReport:
    questions = load_questions(dataset)
    corpus = load_corpus()
    zip_path = fetch_dataset(dataset, cache_dir, seed_dirs, corpus=corpus)
    csv_name = next(item.path for item in corpus[dataset].files if item.kind == "table")
    csv_path = zip_path.parent / "files" / csv_name
    try:
        truth: dict[str, OracleAnswer] = load_oracles(dataset).answers(csv_path)
    except pl.exceptions.PolarsError as error:
        raise AcceptanceError(f"oracles for {dataset} do not run on {csv_name}: {error}") from error
    missing = [q["id"] for q in questions if q["id"] not in truth]
    if missing:
        raise AcceptanceError(f"no oracle answer for {missing}")

    report = RunReport(
        dataset=dataset,
        model=model,
        label=label,
        git_head=_git_head(app_src.parent if app_src else REPO_ROOT),
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        settings={"timeout_s": timeout_s, "llm_timeout_s": llm_timeout_s, "ollama_base_url": ollama_base_url},
    )
    work = Path(tempfile.mkdtemp(prefix="mdde-accept-"))
    api = Api(work, server_env(work, secrets.token_urlsafe(32), model, ollama_base_url, llm_timeout_s, app_src))
    try:
        report.settings["startup_s"] = round(api.wait_healthy(), 1)
        upload_started = time.monotonic()
        try:
            dataset_id = api.upload(zip_path, dataset)
            report.settings["upload"] = "zip (table + dictionary)"
        except AcceptanceError as error:
            if "-> 400" not in str(error):
                raise
            # Recorded, never silent: the API refused the ZIP, so the table goes up alone and
            # the app answers without the dictionary.
            report.settings["upload"] = f"csv only; ZIP rejected: {str(error)[:200]}"
            dataset_id = api.upload(csv_path, dataset)
        report.settings["upload_s"] = round(time.monotonic() - upload_started, 1)
        report.results = ask_all(api, dataset_id, questions, truth, model, timeout_s)
    finally:
        api.stop()
        report.settings["server_exit_code"] = api.proc.returncode
        if keep_work:
            report.settings["work_dir"] = str(work)
        else:
            shutil.rmtree(work, ignore_errors=True)
    return report


def _server_note(code: int | None) -> str:
    """Empty when the server ran until the harness stopped it (SIGTERM), else its exit code."""
    return "" if code in (None, -15) else f" · **API process exited with code {code} during the run**"


def markdown(report: RunReport) -> str:
    counts = report.counts()
    latencies = [result.latency_s for result in report.results]
    median = statistics.median(latencies) if latencies else 0.0
    llm_ran = sum(result.llm == "ran" for result in report.results)
    lines = [
        f"### {report.dataset} · MODEL={report.model} · {report.label} @ {report.git_head}",
        "",
        f"Upload: {report.settings.get('upload')} · "
        f"model-call timeout: {report.settings.get('llm_timeout_s') or 'app default'}"
        + _server_note(report.settings.get("server_exit_code")),
        "",
        f"**{report.rates().line()}**",
        "",
        " · ".join(f"{name} {n}" for name, n in counts.items())
        + f" · median latency {median:.1f}s · LLM tier ran on {llm_ran}/{len(report.results)}",
        "",
        "| id | grade | leak | tier | LLM | latency s | reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in report.results:
        reason = result.reason.replace("|", "\\|").replace("\n", " ")[:180]
        lines.append(
            f"| {result.id} | {result.grade} | {'LEAK' if result.leak else ''} | {result.tier or '-'} | "
            f"{result.llm} | {result.latency_s:.1f} | {reason} |"
        )
    return "\n".join(lines) + "\n"


def write_results(report: RunReport, results_dir: Path) -> Path:
    stamp = report.started_at.replace(":", "").replace("-", "")
    model = re.sub(r"[^\w.-]", "_", report.model)
    out = results_dir / f"{stamp}-{report.dataset}-{model}-{report.label}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(report), "counts": report.counts()}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out / "results.md").write_text(markdown(report), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------- gates


class Baseline(BaseModel):
    """A committed floor for one dataset and model: the answered-right count only, never rows.

    The gate reads answered right, not RIGHT: RIGHT also counts a correctly held question, so a
    lost answer could hide behind a new hold (ADR013 §11.3)."""

    model_config = ConfigDict(extra="forbid", strict=True)
    dataset: str
    model: str
    answered_right: int = Field(ge=0)


def load_baseline(path: Path, dataset: str, model: str) -> Baseline:
    """The baseline in ``path``, checked to be for this dataset and model.

    Raises:
        AcceptanceError: the file is missing or malformed, or is for another dataset or model.
    """
    try:
        baseline = Baseline.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise AcceptanceError(f"baseline {path} could not be read: {error}") from error
    if (baseline.dataset, baseline.model) != (dataset, model):
        raise AcceptanceError(
            f"baseline {path} is for {baseline.dataset} with MODEL={baseline.model}, not {dataset} with MODEL={model}"
        )
    return baseline


def _questions(n: int) -> str:
    return f"{n} question" if n == 1 else f"{n} questions"


def gate_failures(
    counts: dict[str, int],
    rates: Rates,
    fail_on: Collection[str],
    baseline: Baseline | None,
    baseline_path: Path | None = None,
    model_runs: int | None = None,
) -> list[str]:
    """Why this run fails its gate, each with the next step; empty when it passes.

    ``model_runs`` is how many questions the language model answered, for a run with a model:
    0 means every question went without it, which would pass a WRONG/LEAK gate unexamined."""
    failures = [
        f"{_questions(counts.get(grade, 0))} graded {grade}; the gate allows none. "
        "The results table gives the reason for each."
        for grade in GRADES
        if grade in fail_on and counts.get(grade, 0) > 0
    ]
    if model_runs == 0:
        failures.append(
            "The model ran on none of the questions, so this run did not test it. "
            "The LLM column in the results table says why for each question."
        )
    if baseline is not None and rates.answered_right < baseline.answered_right:
        where = baseline_path or "the baseline file"
        failures.append(
            f"Answered right fell to {rates.answered_right}, below the committed baseline of "
            f"{baseline.answered_right}. Fix the regression, or lower answered_right in {where} "
            "and say why in the pull request."
        )
    return failures


def parse_fail_on(text: str) -> frozenset[Grade]:
    """``"WRONG,LEAK"`` -> {"WRONG", "LEAK"}; argparse type, so a typo stops the run."""
    names = {part.strip().upper() for part in text.split(",") if part.strip()}
    unknown = sorted(names - set(GRADES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown grade(s) {', '.join(unknown)}; use {', '.join(GRADES)}")
    return frozenset(name for name in GRADES if name in names)


def gate_markdown(failures: list[str], rates: Rates, baseline: Baseline | None) -> str:
    if failures:
        return "**Gate failed.**\n\n" + "".join(f"- {failure}\n" for failure in failures)
    note = "**Gate passed.**"
    if baseline is not None and rates.answered_right > baseline.answered_right:
        note += (
            f" Answered right rose to {rates.answered_right} from the baseline of {baseline.answered_right}."
            " Raise answered_right in the baseline file to keep the gain."
        )
    return note + "\n"


# ---------------------------------------------------------------------------- upload check


@dataclass
class LoadResult:
    dataset: str
    upload: str
    expected_rows: int | None
    expected_columns: int | None
    status: str | None = None
    rows: int | None = None
    columns: int | None = None
    upload_s: float = 0.0
    error: str | None = None

    def failures(self) -> list[str]:
        if self.error is not None:
            return [f"{self.dataset}: {self.error}"]
        problems = []
        if self.status != "ready":
            problems.append(f"{self.dataset}: upload status is {self.status}, expected ready")
        for what, got, expected in (
            ("rows", self.rows, self.expected_rows),
            ("columns", self.columns, self.expected_columns),
        ):
            if got != expected:
                problems.append(f"{self.dataset}: loaded {got} {what}, corpus.yaml expects {expected}")
        return problems


def _upload_path(dataset: CorpusDataset, zip_path: Path) -> Path:
    if dataset.load_upload == "zip":
        return zip_path
    table = next((item.path for item in dataset.files if item.kind == "table"), None)
    if table is None:
        raise AcceptanceError(f"{dataset.name} has load_upload: table but no file of kind table")
    return zip_path.parent / "files" / table


def check_load(
    name: str, corpus: dict[str, CorpusDataset], cache_dir: Path | None, seed_dirs: list[Path]
) -> LoadResult:
    """Upload one dataset into a fresh app (new data home, new process) and read back its load.

    Never raises for an app failure: an upload or read that fails is the result's ``error``."""
    if name not in corpus:
        known = ", ".join(sorted(corpus))
        error = f"not in corpus.yaml (known: {known}). Fix the name in ACCEPT_LOAD_DATASETS."
        return LoadResult(name, "-", None, None, error=error)
    dataset = corpus[name]
    result = LoadResult(name, dataset.load_upload, dataset.rows, dataset.columns)
    if dataset.rows is None or dataset.columns is None:
        result.error = "corpus.yaml has no rows or columns for it. Add both to check its load."
        return result
    try:
        path = _upload_path(dataset, fetch_dataset(name, cache_dir, seed_dirs, corpus=corpus))
    except (AcceptanceError, CorpusFetchError) as error:
        result.error = f"fetch: {error}"
        return result
    work = Path(tempfile.mkdtemp(prefix="mdde-load-"))
    api: Api | None = None
    try:
        api = Api(work, server_env(work, secrets.token_urlsafe(32), NO_MODEL, UNREACHABLE_OLLAMA, None))
        api.wait_healthy()
        started = time.monotonic()
        response = api.upload_response(path, name)
        result.upload_s = round(time.monotonic() - started, 1)
        result.status = str(response.get("status"))
        upload_id = required_field(response, "upload_id", "POST /api/datasets/upload")
        detail = api.request("GET", f"/api/datasets/{upload_id}")
        tables = detail.get("tables") or []
        if len(tables) != 1:
            result.error = f"GET /api/datasets/{{id}} returned {len(tables)} tables, expected 1"
        else:
            result.rows = tables[0].get("row_count")
            result.columns = len(tables[0].get("columns") or [])
    except (AcceptanceError, OSError) as error:
        result.error = str(error)
    finally:
        if api is not None:
            api.stop()
        shutil.rmtree(work, ignore_errors=True)
    logger.info("accept_load", dataset=name, ok=not result.failures(), upload_s=result.upload_s)
    return result


def _or_dash(value: int | None) -> str:
    return "-" if value is None else str(value)


def load_markdown(results: list[LoadResult]) -> str:
    lines = [
        "### Corpus upload check",
        "",
        "| dataset | file uploaded | status | rows (loaded / expected) | columns (loaded / expected) "
        "| upload s | result |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        problems = "; ".join(result.failures()).replace("|", "\\|")[:200]
        lines.append(
            f"| {result.dataset} | {result.upload} | {result.status or '-'} | "
            f"{_or_dash(result.rows)} / {_or_dash(result.expected_rows)} | "
            f"{_or_dash(result.columns)} / {_or_dash(result.expected_columns)} | "
            f"{result.upload_s:.1f} | {problems or 'OK'} |"
        )
    return "\n".join(lines) + "\n"


def _append_summary(path: Path | None, text: str) -> None:
    """Append to a job summary (``$GITHUB_STEP_SUMMARY``) when one is given."""
    if path is not None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", action="append", default=None, help="corpus dataset (default covid_ms); repeat with --load-only"
    )
    parser.add_argument("--model", default=NO_MODEL, help="'none' (no language model) or an Ollama model name")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-question timeout, seconds")
    parser.add_argument("--llm-timeout", type=float, default=None, help="app's model-call timeouts (default: app's)")
    parser.add_argument("--label", default="local", help="short name for this checkout, e.g. main or pr101")
    parser.add_argument("--seed", type=Path, action="append", default=[], help="directory of already-fetched files")
    parser.add_argument("--results-dir", type=Path, default=None, help="default: <cache>/results")
    parser.add_argument("--app-src", type=Path, default=None, help="serve the app from another checkout's src/")
    parser.add_argument("--keep-work", action="store_true", help="keep the temp data home and server log")
    parser.add_argument(
        "--fail-on", type=parse_fail_on, default=frozenset(), help="exit 1 if any question has these grades"
    )
    parser.add_argument(
        "--baseline", type=Path, default=None, help="exit 1 if answered right falls below this file's count"
    )
    parser.add_argument("--load-only", action="store_true", help="upload each --dataset and check the load; ask none")
    parser.add_argument("--summary", type=Path, default=None, help="append the markdown to this file too")
    args = parser.parse_args(argv)
    datasets: list[str] = args.dataset or ["covid_ms"]
    if args.load_only and (args.fail_on or args.baseline is not None):
        parser.error("--load-only asks no questions, so --fail-on and --baseline would check nothing")
    if args.load_only:
        loads = [check_load(name, load_corpus(), None, args.seed) for name in datasets]
        text = load_markdown(loads)
        print(text)
        _append_summary(args.summary, text)
        failures = [problem for result in loads for problem in result.failures()]
        for problem in failures:
            print(f"upload check failed: {problem}", file=sys.stderr)
        if failures:
            print(
                "If the app changed, fix the upload; if the dataset changed, update its pins in corpus.yaml.",
                file=sys.stderr,
            )
        return EXIT_GATE_FAILED if failures else 0
    if len(datasets) != 1:
        parser.error("ask one --dataset at a time; several are allowed only with --load-only")
    results_dir = args.results_dir or default_cache_dir() / "results"
    if results_dir.resolve().is_relative_to(REPO_ROOT):
        parser.error("--results-dir must be outside the repository")
    try:
        baseline = load_baseline(args.baseline, datasets[0], args.model) if args.baseline else None
        report = run(
            datasets[0],
            args.model,
            args.ollama_base_url,
            args.timeout,
            args.llm_timeout,
            args.label,
            args.seed,
            app_src=args.app_src.resolve() if args.app_src else None,
            keep_work=args.keep_work,
        )
    except (AcceptanceError, CorpusFetchError) as error:
        message = f"acceptance run failed: {error}"
        print(message, file=sys.stderr)
        _append_summary(args.summary, f"**{message}**")
        return EXIT_NOT_RUN
    out = write_results(report, results_dir)
    rates = report.rates()
    # Only a gated run (--fail-on) checks that the model ran; a plain local run reports it.
    gated_model = args.fail_on and args.model != NO_MODEL
    model_runs = sum(result.llm == "ran" for result in report.results) if gated_model else None
    failures = gate_failures(report.counts(), rates, args.fail_on, baseline, args.baseline, model_runs)
    text = markdown(report)
    if args.fail_on or baseline is not None:
        text += "\n" + gate_markdown(failures, rates, baseline)
    print(text)
    print(f"results: {out}")
    _append_summary(args.summary, text)
    for failure in failures:
        print(f"acceptance gate failed: {failure}", file=sys.stderr)
    return EXIT_GATE_FAILED if failures else 0


if __name__ == "__main__":
    sys.exit(main())

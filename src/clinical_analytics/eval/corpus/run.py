# SPDX-License-Identifier: MIT
"""Acceptance run: one public dataset, uploaded and asked through the real API.

Each run fetches the checked upload ZIP, starts a fresh API process (``python -m
clinical_analytics.api.serve``, the installed app's entry point) from an unrelated directory
with a new temporary ``CLINICAL_ANALYTICS_DATA_HOME`` and HOME, uploads the ZIP through
``POST /api/datasets/upload``, and asks every question through ``/api/queries``: each in a new
session, a follow-up in its parent's session. A held question is confirmed once only when its
YAML says ``confirm: true``. The presented answer is graded by ``eval.corpus.score``.

A question marked ``held_out: true`` is reported apart from the rest: nobody tunes the app on it,
so its grades say whether a change generalises beyond the questions it was made for.

Results (JSON and a markdown table) go to a directory outside the repo; the table is printed.
Only aggregates are written: the claim and table the app presents, never rows.

    make accept DATASET=covid_ms MODEL=llama3.2:3b OLLAMA_BASE_URL=http://127.0.0.1:11435
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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl
import structlog
import yaml

from clinical_analytics.eval.corpus.fetch import CorpusFetchError, default_cache_dir, fetch_dataset, load_corpus
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


class AcceptanceError(RuntimeError):
    """The acceptance run could not start or upload (not a graded question failure)."""


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
    held_out: bool = False
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

    def counts(self, held_out: bool | None = None) -> dict[str, int]:
        """Grades over every question, or only the held-out (True) or tuned-on (False) ones."""
        results = [r for r in self.results if held_out is None or r.held_out is held_out]
        tally = {grade: 0 for grade in ("RIGHT", "WRONG", "HELD", "ERROR")}
        for result in results:
            tally[result.grade] += 1
        tally["LEAK"] = sum(result.leak for result in results)
        return tally


# ---------------------------------------------------------------------------- questions/oracles


def load_questions(dataset: str) -> list[dict[str, Any]]:
    raw = yaml.safe_load((QUESTIONS_DIR / f"{dataset}.yaml").read_text(encoding="utf-8"))
    shared = {"dataset_id": raw["dataset_id"], **({"grain": raw["grain"]} if "grain" in raw else {})}
    questions = [{**shared, **item} for item in raw["questions"]]
    seen: set[str] = set()
    for question in questions:
        parent = question.get("follow_up_of")
        if parent is not None and parent not in seen:
            raise AcceptanceError(f"{question['id']} follows up {parent}, which is not asked before it")
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
        response = self.request("POST", "/api/datasets/upload", body, f"multipart/form-data; boundary={boundary}", 900)
        return required_field(response, "upload_id", "POST /api/datasets/upload")

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
            held_out=bool(question.get("held_out", False)),
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
    files_dir = zip_path.parent / "files"
    tables = [item.path for item in corpus[dataset].files if item.kind in ("table", "archive")]
    # One table: the oracles read that file. Several, or an archive of tables: the files' directory.
    single = len(tables) == 1 and tables[0].endswith(".csv")
    oracle_input = files_dir / tables[0] if single else files_dir
    try:
        truth: dict[str, OracleAnswer] = load_oracles(dataset).answers(oracle_input)
    except pl.exceptions.PolarsError as error:
        raise AcceptanceError(f"oracles for {dataset} do not run on {oracle_input.name}: {error}") from error
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
            if "-> 400" not in str(error) or oracle_input.is_dir():
                raise
            # Recorded, never silent: the API refused the ZIP, so the table goes up alone and
            # the app answers without the dictionary. (A multi-table set has no such fallback.)
            report.settings["upload"] = f"csv only; ZIP rejected: {str(error)[:200]}"
            dataset_id = api.upload(oracle_input, dataset)
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
    held_out = report.counts(held_out=True)
    tuned = report.counts(held_out=False)
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
        " · ".join(f"{name} {n}" for name, n in counts.items())
        + f" · median latency {median:.1f}s · LLM tier ran on {llm_ran}/{len(report.results)}",
        "",
        "Tuned-on: " + " · ".join(f"{name} {n}" for name, n in tuned.items()),
        "Held out: " + " · ".join(f"{name} {n}" for name, n in held_out.items()),
        "",
        "| id | split | grade | leak | tier | LLM | latency s | reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in report.results:
        reason = result.reason.replace("|", "\\|").replace("\n", " ")[:180]
        split = "held out" if result.held_out else "tuned"
        lines.append(
            f"| {result.id} | {split} | {result.grade} | {'LEAK' if result.leak else ''} | {result.tier or '-'} | "
            f"{result.llm} | {result.latency_s:.1f} | {reason} |"
        )
    return "\n".join(lines) + "\n"


def write_results(report: RunReport, results_dir: Path) -> Path:
    stamp = report.started_at.replace(":", "").replace("-", "")
    model = re.sub(r"[^\w.-]", "_", report.model)
    out = results_dir / f"{stamp}-{report.dataset}-{model}-{report.label}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(report),
        "counts": report.counts(),
        "counts_tuned": report.counts(held_out=False),
        "counts_held_out": report.counts(held_out=True),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out / "results.md").write_text(markdown(report), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="covid_ms")
    parser.add_argument("--model", default=NO_MODEL, help="'none' (no language model) or an Ollama model name")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-question timeout, seconds")
    parser.add_argument("--llm-timeout", type=float, default=None, help="app's model-call timeouts (default: app's)")
    parser.add_argument("--label", default="local", help="short name for this checkout, e.g. main or pr101")
    parser.add_argument("--seed", type=Path, action="append", default=[], help="directory of already-fetched files")
    parser.add_argument("--results-dir", type=Path, default=None, help="default: <cache>/results")
    parser.add_argument("--app-src", type=Path, default=None, help="serve the app from another checkout's src/")
    parser.add_argument("--keep-work", action="store_true", help="keep the temp data home and server log")
    args = parser.parse_args(argv)
    results_dir = args.results_dir or default_cache_dir() / "results"
    if results_dir.resolve().is_relative_to(REPO_ROOT):
        parser.error("--results-dir must be outside the repository")
    try:
        report = run(
            args.dataset,
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
        print(f"acceptance run failed: {error}", file=sys.stderr)
        return 2
    out = write_results(report, results_dir)
    print(markdown(report))
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

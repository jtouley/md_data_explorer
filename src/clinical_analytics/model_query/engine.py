# SPDX-License-Identifier: MIT
"""Answer one question with a plan the local model fills in.

The model fills the plan's fixed slots (held to the table's columns and values by the reply
schema); the app applies what the question states explicitly (``checks.apply_stated``), compiles
the plan, checks that the compiled plan computes what the question says (``checks.problems``),
and asks back what the question leaves open and the contract says changes the number (which visit
counts, patients or records, an open question of the contract). A meaning problem is fed back
once; if it stays, the app holds and says why. Every answer carries the SQL that ran and a reading
card in clinical words (``card``). When the app holds or asks, the answer it would have shown is
logged as numbers only (``model_query_held_answer``), so hold quality can be measured.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import polars as pl
import structlog

from clinical_analytics.model_query import card, checks, present
from clinical_analytics.model_query.context import DatasetContext
from clinical_analytics.model_query.examples import Example, similar_examples
from clinical_analytics.model_query.llm import ModelFn, ModelTimedOut
from clinical_analytics.model_query.plan import (
    LISTING_REASON,
    Compiled,
    CountColumnUnknownError,
    PlanError,
    PlanRefusedError,
    compile_plan,
    parse_plan,
    reply_schema,
    varies_sql,
)
from clinical_analytics.model_query.prompt import system_prompt, user_prompt
from clinical_analytics.sql import config as sql_config
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.sandbox import SandboxConnection, SqlRunError

logger = structlog.get_logger()

MODEL_UNAVAILABLE_REASON: Final = (
    "The local language model did not answer, so the question was not answered. "
    "Check that Ollama is running, then ask again."
)
MODEL_TIMEOUT_REASON: Final = (
    "The local model took too long to answer (more than {minutes} minutes), so the question was not answered. "
    "The computer may be busy: ask again in a moment. The wait can be raised with model_query_timeout_s."
)
COUNT_COLUMN_QUESTION: Final = (
    "Each line of this table counts a group, so counting lines would count groups, not cases. "
    "Which column holds the number of cases on each line?"
)
NO_QUERY_REASON: Final = "I could not turn this question into a query that runs on your data."
MAX_TEST_ROWS: Final = 2_000_000
Status = Literal["answered", "refused", "unavailable", "clarify"]
_LISTING: Final = re.compile(
    r"\b(?:list|show me|give me|name|identify)\b.{0,40}\b(?:patients?|people|ids?|mrns?|names|charts?|records)\b"
    r"(?!.{0,30}\b(?:how many|number|count|share|percent|average|mean|median)\b)",
    re.I,
)


@dataclass(frozen=True)
class ModelAnswer:
    """The outcome of the model path for one question."""

    status: Status
    analysis: dict[str, Any] | None = None
    sql: str | None = None
    reading: str | None = None
    reason: str | None = None
    reply: dict[str, Any] | None = None  # the plan that was used (kept for follow-ups)
    # What changes the number and the question leaves open: (label, the question to ask instead).
    choices: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    attempts: int = 0
    problems: tuple[str, ...] = field(default_factory=tuple)  # what each failed attempt got wrong


def asks_for_listing(question: str) -> bool:
    return bool(_LISTING.search(question))


def _run(compiled: Compiled, open_sandbox: Callable[[], SandboxConnection]) -> tuple[pl.DataFrame, int | None]:
    with open_sandbox() as sandbox:
        frame = sandbox.run_model_sql(compiled.sql)
        total = int(sandbox.run_model_sql(compiled.total_sql).item() or 0) if compiled.total_sql else None
    return frame, total


def _analysis(
    compiled: Compiled, frame: pl.DataFrame, total: int | None, context: DatasetContext, question: str
) -> dict[str, Any]:
    if compiled.kind == "test":
        if frame.height > MAX_TEST_ROWS:
            raise PlanError("The test would read too many records; add a filter.")
        analysis = present.run_test(compiled, frame, question, context.patient_key)
        if analysis.get("error"):
            raise PlanError(str(analysis["error"]))
        return analysis
    if compiled.group_column is not None and frame.height > sql_config.MAX_GROUPS:
        raise PlanError(
            f"group_by gives {frame.height} groups, too many to show: group by a column with fewer values, or give "
            "cut_points or a period."
        )
    return present.analysis_of(compiled, frame, context, total)


def _numbers(analysis: dict[str, Any]) -> list[float]:
    """The numbers an analysis would show, without any group's label. A minimum or a maximum is
    one record's value, so it is never logged."""
    if analysis.get("statistic") in ("min", "max"):
        return []
    keys = ("total_count", "proportion", "events", "n_used", "mean", "median", "sum", "non_null_count")
    found = [float(analysis[k]) for k in keys if isinstance(analysis.get(k), int | float)]
    for row in analysis.get("group_counts") or analysis.get("groups") or []:
        found += [float(v) for k, v in row.items() if k in ("count", "rate", "value", "n", "events") and v is not None]
    return found


def _log_would_have(
    why: str, compiled: Compiled, context: DatasetContext, open_sandbox: Callable[[], SandboxConnection], question: str
) -> None:
    """Log the answer the app would have shown when it holds or asks (numbers only)."""
    try:
        frame, total = _run(compiled, open_sandbox)
        numbers = _numbers(_analysis(compiled, frame, total, context, question))
    except (SqlRunError, SqlRefusedError, PlanError) as e:
        logger.info(
            "model_query_held_answer", why=why, plan_answer=compiled.plan.answer, ran=False, error_type=type(e).__name__
        )
        return
    logger.info(
        "model_query_held_answer",
        why=why,
        plan_answer=compiled.plan.answer,
        which_visit=compiled.plan.which_visit,
        count_of=compiled.unit,
        ran=True,
        numbers=numbers[:40],
    )


def _ask(
    ask: checks.Ask, question: str, compiled: Compiled, context: DatasetContext, reply: dict[str, Any] | None,
    attempt: int,
) -> ModelAnswer:  # fmt: skip
    logger.info("model_query_asked", kind=ask.kind, attempt=attempt)
    return ModelAnswer(
        "clarify" if ask.choices else "refused",
        reason=ask.reason,
        choices=tuple((label, f"{question.rstrip()} {words}") for label, words in ask.choices),
        sql=compiled.sql,
        reading=card.reading(card.reading_lines(compiled, context)),
        reply=reply,
        attempts=attempt,
    )


def _count_column_ask(
    question: str, context: DatasetContext, reply: dict[str, Any] | None, attempt: int
) -> ModelAnswer:
    """A count on a summary table whose count column the contract does not name: asked, never a
    count of its lines."""
    logger.info("model_query_asked", kind="count_column", attempt=attempt)
    numbers = [c for c in context.columns if c.numeric]
    return ModelAnswer(
        "clarify" if numbers else "refused",
        reason=COUNT_COLUMN_QUESTION,
        choices=tuple((f"Total of {c.display}", f"{question.rstrip()} (the total of {c.display})") for c in numbers),
        reply=reply,
        attempts=attempt,
    )


def _held(
    problem: checks.Problem, compiled: Compiled, context: DatasetContext, reply: dict[str, Any] | None, attempt: int,
    problems: list[str],
) -> ModelAnswer:  # fmt: skip
    logger.info("model_query_held", kind=problem.kind, attempt=attempt)
    lines = card.reading_lines(compiled, context)
    return ModelAnswer(
        "refused",
        reason=f"{problem.reason} My reading was: {' '.join(lines)} Ask again in other words if that is not it.",
        sql=compiled.sql,
        reading=card.reading(lines),
        reply=reply,
        attempts=attempt,
        problems=tuple(problems),
    )


def answer_question(
    question: str,
    context: DatasetContext,
    *,
    model: ModelFn,
    open_sandbox: Callable[[], SandboxConnection],
    max_attempts: int = 3,
    previous: Sequence[tuple[str, dict[str, Any]]] = (),
    examples: Sequence[Example] | None = None,
) -> ModelAnswer:
    """Ask the model for a plan, check it, run it; feed a failure back for a bounded number of
    retries.

    ``model``: the local model call (``llm.ollama_model`` in the app; a recorded reply in tests).
    ``open_sandbox``: a new sandbox over the dataset's stored rows.
    ``previous``: earlier (question, reply) turns of the conversation, for follow-up questions.
    """
    if asks_for_listing(question):
        logger.info("model_query_listing_refused", attempt=0)
        return ModelAnswer("refused", reason=LISTING_REASON)
    shots = list(examples) if examples is not None else similar_examples(question)
    system, schema = system_prompt(context), reply_schema(context)
    # A follow-up ("now only the women") keeps what the earlier questions said.
    asked = " ".join([*(q for q, _reply in previous), question])
    problems: list[str] = []
    problem: str | None = None
    failed_reply: dict[str, Any] | None = None
    held: tuple[checks.Problem, Compiled] | None = None
    for attempt in range(1, max_attempts + 1):
        payload = model(
            system, user_prompt(question, shots, previous=previous, failed_reply=failed_reply, problem=problem), schema
        )
        if isinstance(payload, ModelTimedOut):
            logger.warning("model_query_timed_out", attempt=attempt, timeout_s=payload.timeout_s)
            reason = MODEL_TIMEOUT_REASON.format(minutes=max(1, round(payload.timeout_s / 60)))
            return ModelAnswer("unavailable", reason=reason, attempts=attempt)
        if payload is None:
            logger.warning("model_query_unavailable", attempt=attempt)
            return ModelAnswer("unavailable", reason=MODEL_UNAVAILABLE_REASON, attempts=attempt)
        failed_reply = payload if isinstance(payload, dict) else None
        try:
            plan = parse_plan(payload)
            if plan.answer == "list_patients":
                logger.info("model_query_listing_refused", attempt=attempt)
                return ModelAnswer("refused", reason=LISTING_REASON, reply=failed_reply, attempts=attempt)
            plan, changed = checks.apply_stated(plan, asked, context)
            if changed:
                logger.info("model_query_stated_applied", attempt=attempt, changes=changed)
            compiled = compile_plan(plan, context)
        except PlanRefusedError as e:
            logger.info("model_query_refused", attempt=attempt, plan_answer=plan.answer)
            return ModelAnswer("refused", reason=str(e), reply=failed_reply, attempts=attempt)
        except CountColumnUnknownError:
            return _count_column_ask(question, context, failed_reply, attempt)
        except PlanError as e:
            problem = str(e)
            problems.append(problem)
            logger.info("model_query_attempt_failed", attempt=attempt, kind="plan")
            continue
        reply = plan.model_dump(mode="json")
        found = checks.problems(compiled, asked, context)
        if found:
            if held is None and attempt < max_attempts:
                held = (found[0], compiled)
                problem = " ".join(dict.fromkeys(p.feedback for p in found))
                problems.append(f"meaning: {problem}")
                logger.info("model_query_meaning_fed_back", attempt=attempt, kinds=[p.kind for p in found])
                continue
            _log_would_have(found[0].kind, compiled, context, open_sandbox, asked)
            return _held(found[0], compiled, context, reply, attempt, problems)
        ask = checks.ask_open(compiled, context) or checks.ask_unit(compiled, asked, context)
        try:
            if ask is None and checks.needs_visit_choice(compiled, asked, context):
                names = checks.visit_columns(compiled)
                with open_sandbox() as sandbox:
                    counts = sandbox.run_model_sql(varies_sql(context, names)).row(0)
                varying = int(counts[-1] or 0)
                if varying:
                    ask = checks.visit_ask(
                        compiled, context, varying, [n for n, c in zip(names, counts, strict=False) if c]
                    )
            if ask is not None:
                _log_would_have(ask.kind, compiled, context, open_sandbox, asked)
                return _ask(ask, question, compiled, context, reply, attempt)
            frame, total = _run(compiled, open_sandbox)
            analysis = _analysis(compiled, frame, total, context, question)
        except PlanError as e:
            problem = str(e)
        except SqlRunError as e:
            # The engine's message goes to the local model only; the user sees the plain reason.
            problem = e.engine_message.splitlines()[0][:400] if e.engine_message else e.reason
        except SqlRefusedError as e:
            problem = e.reason
        else:
            lines = card.reading_lines(compiled, context, analysis, total)
            reading = card.reading(lines)
            logger.info("model_query_answered", attempt=attempt, plan_answer=plan.answer, kind=analysis.get("type"))
            return ModelAnswer(
                "answered",
                analysis={**analysis, "sql": compiled.sql, "reading": reading},
                sql=compiled.sql,
                reading=reading,
                reply=reply,
                attempts=attempt,
                problems=tuple(problems),
            )
        problems.append(problem)
        logger.info("model_query_attempt_failed", attempt=attempt, kind="run")
    if held is not None:
        _log_would_have(held[0].kind, held[1], context, open_sandbox, asked)
        return _held(held[0], held[1], context, failed_reply, max_attempts, problems)
    return ModelAnswer(
        "refused",
        reason=NO_QUERY_REASON,
        reply=failed_reply,
        attempts=max_attempts,
        problems=tuple(problems),
    )

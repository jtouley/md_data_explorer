# SPDX-License-Identifier: MIT
"""Answer one question with a query the local model writes.

The loop every working text-to-SQL tool uses: the model writes the query from the table's
description and similar worked examples; the app checks and runs it in the SQL sandbox; a failure
goes back to the model as feedback for a bounded number of retries. No word of the question has
to match a column for the question to be answered: there is no keyword gate here. The answer
always carries the SQL that ran and a one-sentence plain reading of it.

The question is refused only when it asks to list individual patients, when the model cannot be
reached, or when no attempt produced a query that runs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import polars as pl
import structlog

from clinical_analytics.model_query import present
from clinical_analytics.model_query.context import DatasetContext, column_lines, column_search_texts
from clinical_analytics.model_query.examples import Example, related_columns, similar_examples
from clinical_analytics.model_query.grounding import (
    MISSING_FAILS_PHRASE,
    MISSING_MEETS_PHRASE,
    MISSING_OUT_PHRASE,
    Doubt,
    dropped_number,
    find_doubts,
    user_choices,
    without_columns,
)
from clinical_analytics.model_query.llm import ModelFn
from clinical_analytics.model_query.plan import (
    LISTING,
    LISTING_REASON,
    SQL,
    STAT_TESTS,
    TABULATIONS,
    Compiled,
    ModelReply,
    PlanError,
    ReplyFormat,
    TestCall,
    asks_for_listing,
    asks_for_summary,
    compile_plan,
    compile_sql,
    compile_test,
    parse_reply,
    reply_schema,
    unused_mentions,
)
from clinical_analytics.model_query.prompt import system_prompt, user_prompt
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.sandbox import SandboxConnection, SqlRunError

logger = structlog.get_logger()

MODEL_UNAVAILABLE_REASON: Final = (
    "The local language model did not answer, so the question was not answered. "
    "Check that Ollama is running, then ask again."
)
NO_QUERY_REASON: Final = "I could not write a query for this question that runs on your data."
MAX_TEST_ROWS: Final = 2_000_000
Status = Literal["answered", "refused", "unavailable", "clarify"]
ROLLUP_QUESTION: Final = (
    "For {n} {units} the answer to '{event}' differs between their rows (visits). "
    "Should one count when it was ever true, or only when it was always true?"
)
MISSING_QUESTION: Final = (
    "For {n} {units}, '{event}' cannot be decided because a value it reads is missing. How should they count?"
)
MISSING_CHOICES: Final = (
    ("As meeting it", MISSING_MEETS_PHRASE),
    ("As not meeting it", MISSING_FAILS_PHRASE),
    ("Leave them out", MISSING_OUT_PHRASE),
)
ROLLUP_CHOICES: Final = (
    ("Ever: true on at least one row", "{question} Count one when it was ever true (at least once)."),
    ("Always: true on every row that records it", "{question} Count one only when it was always true."),
)


@dataclass(frozen=True)
class ModelAnswer:
    """The outcome of the model path for one question."""

    status: Status
    analysis: dict[str, Any] | None = None
    sql: str | None = None
    reading: str | None = None
    reason: str | None = None
    reply: dict[str, Any] | None = None  # the model's reply that was used (kept for follow-ups)
    # A real ambiguity that changes the number: (label, the question to ask instead) per choice.
    choices: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    attempts: int = 0
    problems: tuple[str, ...] = field(default_factory=tuple)  # what each failed attempt got wrong


def _run(
    compiled: Compiled | TestCall, open_sandbox: Callable[[], SandboxConnection]
) -> tuple[pl.DataFrame, int | None, int, int]:
    """(the result rows, a grouped count's ungrouped total, units whose event varies over their
    rows, units whose share condition is unknown because a value is missing)."""
    with open_sandbox() as sandbox:
        frame = sandbox.run_model_sql(compiled.sql)
        total: int | None = None
        varying = missing = 0
        if isinstance(compiled, Compiled) and compiled.missing_sql is not None:
            missing = int(sandbox.run_model_sql(compiled.missing_sql).item() or 0)
        if isinstance(compiled, Compiled) and compiled.total_sql is not None:
            total = int(sandbox.run_model_sql(compiled.total_sql).item() or 0)
        if isinstance(compiled, Compiled) and compiled.varies_sql is not None:
            varying = int(sandbox.run_model_sql(compiled.varies_sql).item() or 0)
    return frame, total, varying, missing


def _honour_choices(reply: ModelReply, question: str) -> ModelReply:
    """The reply with what a choice the user made says (no limit on a column, one overall number)
    applied, whatever the model wrote: the choice is the user's, not the model's."""
    excluded, one_number = user_choices(question)
    update: dict[str, Any] = {}
    if excluded:
        update["where"] = without_columns(reply.where, excluded)
    if one_number and reply.answer in TABULATIONS + (SQL,):
        update["group_by"] = None
    return reply.model_copy(update=update) if update else reply


def _doubts(
    question: str, compiled: Compiled | TestCall, context: DatasetContext, kept_group: str | None = None
) -> list[Doubt]:
    _excluded, one_number = user_choices(question)
    dropped = dropped_number(question, compiled.sql)
    if isinstance(compiled, TestCall):
        found = find_doubts(question, context, conditions=compiled.conditions, group_column=None, tabulation=False)
        return found + ([dropped] if dropped is not None else [])
    found = find_doubts(
        question,
        context,
        conditions=compiled.conditions,
        group_column=None if one_number else compiled.group_column,
        group_expression=compiled.group_expression and not one_number,
        tabulation=not one_number,
        kept_group=kept_group,
    )
    return found + ([dropped] if dropped is not None else [])


def _asked_back(
    doubt: Doubt, question: str, compiled: Compiled | TestCall, reply: dict[str, Any] | None, attempt: int
) -> ModelAnswer:
    """The user is asked: the model kept a reading the question does not back."""
    logger.info("model_query_doubt_asked", attempt=attempt, kind=doubt.kind)
    if not doubt.choices:
        return ModelAnswer("refused", reason=doubt.reason, reply=reply, attempts=attempt)
    return ModelAnswer(
        "clarify",
        reason=doubt.reason,
        choices=tuple((label, f"{question.rstrip()} {words}") for label, words in doubt.choices),
        sql=compiled.sql,
        reading=compiled.reading,
        reply=reply,
        attempts=attempt,
    )


def _compile(reply: ModelReply, question: str, context: DatasetContext, fmt: ReplyFormat) -> Compiled | TestCall:
    if reply.answer in STAT_TESTS:
        return compile_test(reply, context)
    if reply.answer == SQL and fmt == "sql":
        return compile_sql(reply, question, context)
    if reply.answer in TABULATIONS and fmt == "plan":
        return compile_plan(reply, question, context)
    raise PlanError(f"answer {reply.answer!r} is not one of the allowed answers.")


def answer_question(
    question: str,
    context: DatasetContext,
    *,
    model: ModelFn,
    open_sandbox: Callable[[], SandboxConnection],
    fmt: ReplyFormat = "plan",
    max_attempts: int = 3,
    previous: Sequence[tuple[str, dict[str, Any]]] = (),
    examples: Sequence[Example] | None = None,
) -> ModelAnswer:
    """Ask the model for a query, run it, and retry with the failure as feedback.

    ``model``: the local model call (``llm.ollama_model`` in the app; a recorded reply in tests).
    ``open_sandbox``: a new sandbox over the dataset's stored rows for each attempt.
    ``previous``: earlier (question, reply) turns of the conversation, for follow-up questions.
    """
    if asks_for_listing(question):
        logger.info("model_query_listing_refused", attempt=0)
        return ModelAnswer("refused", reason=LISTING_REASON)
    shots = list(examples) if examples is not None else similar_examples(question)
    lines = column_lines(context)
    related = [lines[name] for name in related_columns(question, column_search_texts(context))]
    system = system_prompt(context, fmt)
    schema = reply_schema(fmt)
    problems: list[str] = []
    failed_reply: dict[str, Any] | None = None
    problem: str | None = None
    # An answer that ran but drew a hint (a value the question names, on a column the query does
    # not read): the model gets one more look, and this answer stands if it does no better.
    standing: ModelAnswer | None = None
    # The first reply whose reading the question does not back (``grounding``): the model gets one
    # more look with the doubt as feedback; if it keeps that reading, or does no better, the user
    # is asked which reading is meant.
    doubted: tuple[Doubt, Compiled | TestCall, dict[str, Any] | None] | None = None
    # A follow-up ("now only the women") keeps what the earlier questions asked for.
    asked_so_far = " ".join([*(asked for asked, _reply in previous), question])
    # A refinement ("now only the women": no counting words of its own) keeps the last breakdown.
    last_group = previous[-1][1].get("group_by") if previous and not asks_for_summary(question) else None
    kept_group = last_group if isinstance(last_group, str) else None
    for attempt in range(1, max_attempts + 1):
        user = user_prompt(
            question, shots, fmt, previous=previous, failed_reply=failed_reply, problem=problem, related=related
        )
        payload = model(system, user, schema)
        if payload is None:
            logger.warning("model_query_unavailable", attempt=attempt)
            if standing is not None:
                return dataclasses.replace(standing, attempts=attempt)
            return ModelAnswer("unavailable", reason=MODEL_UNAVAILABLE_REASON, attempts=attempt)
        failed_reply = payload if isinstance(payload, dict) else None
        try:
            reply = parse_reply(payload)
            if reply.answer == LISTING and asks_for_summary(asked_so_far):
                # "How many ... broken down by X" is a count: the words ask for a number.
                raise PlanError(
                    "The question asks for a number (how many, a share or a summary), not a list of patients. "
                    "Reply with count, share or a summary; group_by breaks it down."
                )
            if reply.answer == LISTING:
                logger.info("model_query_listing_refused", attempt=attempt)
                return ModelAnswer("refused", reason=LISTING_REASON, reply=failed_reply, attempts=attempt)
            compiled = _compile(_honour_choices(reply, question), question, context, fmt)
            doubts = _doubts(asked_so_far, compiled, context, kept_group)
            if doubts and (doubted is not None or attempt == max_attempts):
                return _asked_back(doubts[0], question, compiled, failed_reply, attempt)
            if doubts:
                doubted = (doubts[0], compiled, failed_reply)
                problem = doubts[0].feedback
                problems.append(f"doubt: {problem}")
                logger.info("model_query_doubted", attempt=attempt, kind=doubts[0].kind)
                continue
            frame, total, varying, missing = _run(compiled, open_sandbox)
            if missing and isinstance(compiled, Compiled):
                logger.info("model_query_missing_asked", attempt=attempt)
                return ModelAnswer(
                    "clarify",
                    reason=MISSING_QUESTION.format(n=missing, units=compiled.unit.plural, event=compiled.event_text),
                    choices=tuple((label, f"{question.rstrip()} {words}") for label, words in MISSING_CHOICES),
                    sql=compiled.sql,
                    reading=compiled.reading,
                    reply=failed_reply,
                    attempts=attempt,
                )
            if varying and isinstance(compiled, Compiled):
                # A share of patients whose event differs between a patient's visits: "ever" and
                # "always" give different numbers, and only the asker knows which is meant.
                logger.info("model_query_rollup_asked", attempt=attempt)
                return ModelAnswer(
                    "clarify",
                    reason=ROLLUP_QUESTION.format(n=varying, units=compiled.unit.plural, event=compiled.event_text),
                    choices=tuple((label, text.format(question=question.rstrip())) for label, text in ROLLUP_CHOICES),
                    sql=compiled.sql,
                    reading=compiled.reading,
                    reply=failed_reply,
                    attempts=attempt,
                )
            if isinstance(compiled, TestCall):
                if frame.height > MAX_TEST_ROWS:
                    raise PlanError("The test would read too many rows; add a where condition.")
                analysis = present.run_test(compiled, frame, question)
                if analysis.get("error"):
                    raise PlanError(str(analysis["error"]))
            else:
                analysis = present.analysis_of(compiled, frame, total)
                if not present.has_value(analysis) and attempt == 1 and max_attempts > 1:
                    raise PlanError(
                        "No rows match the conditions. Check each value against the column's listed values, "
                        "and drop any condition the question does not state."
                    )
        except PlanError as e:
            problem = str(e)
            if LISTING_REASON in problem:
                return ModelAnswer("refused", reason=LISTING_REASON, reply=failed_reply, attempts=attempt)
        except SqlRunError as e:
            # The engine's message goes to the local model only; the user sees the plain reason.
            problem = e.engine_message.splitlines()[0][:400] if e.engine_message else e.reason
            if standing is not None:
                return dataclasses.replace(standing, attempts=attempt)
            problems.append(e.reason)
            logger.info("model_query_attempt_failed", attempt=attempt, kind="engine")
            continue
        except SqlRefusedError as e:
            problem = e.reason
        else:
            answered = ModelAnswer(
                "answered",
                analysis={**analysis, "sql": compiled.sql, "reading": compiled.reading},
                sql=compiled.sql,
                reading=compiled.reading,
                reply=failed_reply,
                attempts=attempt,
                problems=tuple(problems),
            )
            unused = unused_mentions(question, compiled.columns, context) if standing is None else []
            if not unused or attempt == max_attempts:
                logger.info("model_query_answered", attempt=attempt, answer=reply.answer, kind=analysis.get("type"))
                return answered
            standing = answered
            named = "; ".join(f"'{said}' (column {column})" for said, column in unused)
            problem = (
                f"The question mentions {named}, but your reply does not use "
                f"{'that column' if len(unused) == 1 else 'those columns'}. If the question limits or groups by it, "
                "add it; if not, send the same reply again."
            )
            problems.append(f"hint: {problem}")
            logger.info("model_query_hinted", attempt=attempt, mentions=len(unused))
            continue
        if standing is not None:
            # The second look did not run: the first answer stands.
            return dataclasses.replace(standing, attempts=attempt)
        problems.append(problem)
        logger.info("model_query_attempt_failed", attempt=attempt, kind="plan")
    if standing is not None:
        return dataclasses.replace(standing, attempts=max_attempts)
    if doubted is not None:
        return _asked_back(doubted[0], question, doubted[1], doubted[2], max_attempts)
    last = problems[-1] if problems else ""
    return ModelAnswer(
        "refused",
        reason=f"{NO_QUERY_REASON} {last}".strip(),
        reply=failed_reply,
        attempts=max_attempts,
        problems=tuple(problems),
    )

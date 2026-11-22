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
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import polars as pl
import structlog
from sqlglot import exp

from clinical_analytics.model_query import coverage, present
from clinical_analytics.model_query.context import DatasetContext, column_lines, column_search_texts
from clinical_analytics.model_query.examples import Example, related_columns, similar_examples
from clinical_analytics.model_query.grounding import (
    MISSING_FAILS_PHRASE,
    MISSING_MEETS_PHRASE,
    MISSING_OUT_PHRASE,
    NO_LIMIT_PHRASE,
    Doubt,
    asks_breakdown,
    columns_with_stated_values,
    dropped_number,
    find_doubts,
    user_choices,
    without_columns,
)
from clinical_analytics.model_query.llm import ModelFn
from clinical_analytics.model_query.plan import (
    GROUP_EVER_PHRASE,
    GROUP_FIRST_PHRASE,
    GROUP_LAST_PHRASE,
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
    yes_no_of,
)
from clinical_analytics.model_query.prompt import system_prompt, user_prompt
from clinical_analytics.model_query.units import SHARE_OF_PHRASE, UnitError, plural, rate_unstated
from clinical_analytics.sql import config as sql_config
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.render import render
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
PER_PATIENT_QUESTION: Final = (
    "{n} patients have more than one {value} over their visits, so a summary over visits and one over "
    "patients differ. Which do you mean?"
)
PER_PATIENT_CHOICES: Final = (
    ("Every visit counts", "Count every visit."),
    ("One value per patient (their average)", "Use one value per patient (their average)."),
)
RATE_QUESTION: Final = (
    "The question asks for a rate without saying what it is out of. Patients have several {rows} here, so a "
    "share of patients and a share of {rows} differ. Which do you mean?"
)
GROUP_VARIES_QUESTION: Final = (
    "{n} patients have more than one {column} over their visits, so they would be counted in more than one "
    "group. Which {column} counts?"
)
GROUP_VARIES_CHOICES: Final = (
    ("Every {column} a patient had", GROUP_EVER_PHRASE),
    ("Each patient's first {column}", GROUP_FIRST_PHRASE),
    ("Each patient's last {column}", GROUP_LAST_PHRASE),
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


def _coverage_doubts(
    question: str, compiled: Compiled | TestCall, context: DatasetContext, reply: ModelReply
) -> list[Doubt]:
    """Doubts from the coverage and link checks (``coverage``): a content word of the question the
    plan does not reach, or a plan column nothing in the question reaches."""
    values = [
        literal.name for term in compiled.conditions for literal in term.find_all(exp.Literal) if literal.is_string
    ]
    if isinstance(compiled, Compiled) and compiled.change:
        values.append(coverage.CHANGE_WORDS)  # the difference is the change the question names
    found = coverage.check(
        question,
        context,
        used=compiled.columns - getattr(compiled, "helper_columns", frozenset()),
        values=values,
        links=coverage.parse_mapping(reply.mapping),
        stated=columns_with_stated_values(question, list(compiled.conditions), context),
    )
    doubts: list[Doubt] = []
    for name in found.unlinked:
        doubts.append(
            Doubt(
                "unlinked",
                feedback=(
                    f"The reply uses {name}, but nothing in the question refers to it. Remove it unless the question "
                    "asks for it; never use a column in place of one the table does not have."
                ),
                reason=(
                    f"Nothing in the question refers to {name}, so I did not use it to answer. "
                    "Ask again saying what you mean, or choose:"
                ),
                choices=(
                    (f"Use {name}", f"By '{name}' I mean {name}."),
                    (f"Without {name}", NO_LIMIT_PHRASE.format(column=name)),
                ),
            )
        )
    if found.uncovered:
        words = found.uncovered
        said = ", ".join(f"'{w}'" for w in words)
        candidates = coverage.closest_columns(words[0], context)
        missing = " (mapped as having no column)" if found.ignored_missing else ""
        doubts.append(
            Doubt(
                "uncovered",
                feedback=(
                    f"The question says {said}, but no part of the reply uses it{missing}. Add it as a filter, a group "
                    f"or a measure on the column it means"
                    + (f" (perhaps {', '.join(candidates)})" if candidates else "")
                    + ", and list it in mapping. If the table has no column for it, keep mapping it as ignored."
                ),
                reason=(
                    f"I could not find {said} in this data, so I did not answer. "
                    + (
                        "Did you mean one of these columns?"
                        if candidates
                        else "Ask again in other words, or leave it out."
                    )
                ),
                choices=(
                    *(
                        (f"'{words[0]}' means {c}", coverage.MEANS_PHRASE.format(word=words[0], column=c))
                        for c in candidates
                    ),
                    (f"Leave out '{words[0]}'", coverage.LEAVE_OUT_PHRASE.format(word=words[0])),
                ),
            )
        )
    return doubts


_FIRST_OP: Final = re.compile(
    r"\b(?:baseline|initial|first|at (?:entry|enrol(?:l)?ment|intake|diagnosis|admission))\b", re.IGNORECASE
)
_LAST_OP: Final = re.compile(r"\b(?:last|final|latest|most recent|at (?:discharge|exit|the end))\b", re.IGNORECASE)
_FIRST_PARTS: Final = frozenset(
    {"baseline", "bl", "base", "first", "initial", "entry", "enrol", "enroll", "intake", "pre"}
)
_LAST_PARTS: Final = frozenset({"last", "final", "latest", "end", "exit", "discharge", "recent", "post"})


def _time_op_doubt(question: str, compiled: Compiled | TestCall, context: DatasetContext) -> Doubt | None:
    """ "Baseline", "first", "last", "most recent" on visit rows: the plan must apply that operation
    (each patient's first or last value, a column for that time, the first visit's number, or the
    latest period); otherwise the word was dropped and the answer would be over every visit."""
    if not isinstance(compiled, Compiled) or context.rows_are_patients or context.patient_key is None:
        return None
    for pattern, parts, op in ((_FIRST_OP, _FIRST_PARTS, "first"), (_LAST_OP, _LAST_PARTS, "last")):
        said = pattern.search(question)
        if said is None:
            continue
        names = {p for name in compiled.columns for p in re.findall(r"[a-z]+", name.casefold())}
        sequence_used = any(c.is_sequence for name in compiled.columns if (c := context.column(name)))
        latest = op == "last" and "DATE_TRUNC" in compiled.sql.upper()
        if compiled.time_op in (op, "both") or names & parts or (op == "first" and sequence_used) or latest:
            continue
        word = said.group(0)
        return Doubt(
            "time_op",
            feedback=(
                f"The question says '{word}', but the reply reads every visit. For each patient's {op} value set "
                f"per_patient to {op} with number = the measure; or use the column for that time if the table has one."
            ),
            reason=(
                f"The question says '{word}', but I read it over every visit, so I did not answer. Ask again "
                f"saying 'each patient's {op} visit' or naming the column."
            ),
            choices=(),
        )
    return None


def _measure_doubt(question: str, compiled: Compiled | TestCall, context: DatasetContext) -> Doubt | None:
    """A measure whose column has siblings (the same measure at another time) the question does
    not choose between: asked at once, never picked."""
    if not isinstance(compiled, Compiled) or compiled.kind != "statistic":
        return None
    for name in sorted(compiled.columns):
        column = context.column(name)
        if column is None or not column.numeric:
            continue
        candidates = coverage.measure_choice_needed(question, column, context)
        # A value computed from every sibling (a change from the first to the last) chose them all.
        if candidates and not set(candidates) <= compiled.columns:
            return Doubt(
                "ambiguous",
                feedback="",
                reason=f"More than one column holds this measure ({', '.join(candidates)}). Which one?",
                choices=tuple((name_, f"Use {name_}.") for name_ in candidates),
            )
    return None


_SHARE_WORDS: Final = re.compile(r"\b(?:share|proportion|percent|percentage|fraction|rate)\b|%", re.IGNORECASE)
_COUNTS_CHOSEN: Final = re.compile(r"\bgive counts, not a share\b", re.IGNORECASE)


def _share_as_count(question: str, compiled: Compiled | TestCall, context: DatasetContext) -> Doubt | None:
    """A question that asks for a share answered as a count whose where tests a yes/no column:
    the yes/no condition was most likely the share's numerator."""
    if not isinstance(compiled, Compiled) or compiled.kind != "count" or _COUNTS_CHOSEN.search(question):
        return None
    if not _SHARE_WORDS.search(question):
        return None
    for term in compiled.conditions:
        names = {c.name for c in term.find_all(exp.Column)}
        flags = [c for name in names if (c := context.column(name)) is not None and yes_no_of(c)]
        if flags:
            plain = " ".join(render(term).replace('"', "").split())
            return Doubt(
                "share_as_count",
                feedback=(
                    f"The question asks for a share, but the reply counts rows where {plain}. If the question asks "
                    f"what share has {plain}, answer share with numerator {plain} and keep only the population "
                    "in where."
                ),
                reason=f"The question asks for a share, but I read it as counts where {plain}. Which did you mean?",
                choices=(
                    (f"The share with {plain}", f"Count {plain} as the share's numerator."),
                    ("Counts, not a share", "Give counts, not a share."),
                ),
            )
    return None


def _rival_doubt(question: str, compiled: Compiled | TestCall, context: DatasetContext) -> Doubt | None:
    """A condition whose question words name another column just as well: asked, never picked."""
    used = [c.name for term in compiled.conditions for c in term.find_all(exp.Column)]
    rivals = coverage.rival_columns(question, context, used)
    if not rivals:
        return None
    first, second, word = rivals[0]
    return Doubt(
        "ambiguous",
        feedback="",
        reason=f"'{word}' could mean {first} or {second}, which do not always agree. Which one?",
        choices=(
            (first, coverage.MEANS_PHRASE.format(word=word, column=first)),
            (second, coverage.MEANS_PHRASE.format(word=word, column=second)),
        ),
    )


def _doubts(
    question: str,
    compiled: Compiled | TestCall,
    context: DatasetContext,
    reply: ModelReply,
    kept_group: str | None = None,
) -> list[Doubt]:
    _excluded, one_number = user_choices(question)
    dropped = dropped_number(question, compiled.sql)
    share_count = _share_as_count(question, compiled, context)
    time_op = _time_op_doubt(question, compiled, context)
    extra = [d for d in (dropped, share_count, time_op) if d is not None]

    covered = _coverage_doubts(question, compiled, context, reply)
    if isinstance(compiled, TestCall):
        found = find_doubts(question, context, conditions=compiled.conditions, group_column=None, tabulation=False)
        return found + extra + covered
    found = find_doubts(
        question,
        context,
        conditions=compiled.conditions,
        group_column=None if one_number else compiled.group_column,
        group_expression=compiled.group_expression and not one_number,
        tabulation=not one_number,
        kept_group=kept_group,
    )
    return found + extra + covered


# What the user is asked about first when several doubts remain: a concept the plan does not
# reach, then a stated-value limit the question does not state, then a column it does not reach.
_DOUBT_ORDER: Final = ("uncovered", "limit", "unlinked", "dropped", "missing_group", "no_group")


def _first(doubts: list[Doubt]) -> Doubt:
    return min(doubts, key=lambda d: _DOUBT_ORDER.index(d.kind) if d.kind in _DOUBT_ORDER else len(_DOUBT_ORDER))


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
        return compile_test(reply, context, question)
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
    refinement = previous and not asks_for_summary(question) and not asks_breakdown(question)
    last_group = previous[-1][1].get("group_by") if refinement else None
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
            measure = _measure_doubt(asked_so_far, compiled, context) or _rival_doubt(asked_so_far, compiled, context)
            if measure is not None:
                return _asked_back(measure, question, compiled, failed_reply, attempt)
            doubts = _doubts(asked_so_far, compiled, context, reply, kept_group)
            if doubts and (doubted is not None or attempt == max_attempts):
                return _asked_back(_first(doubts), question, compiled, failed_reply, attempt)
            if doubts:
                doubted = (_first(doubts), compiled, failed_reply)
                # Every doubt goes back at once: one more look should fix them all.
                problem = " ".join(dict.fromkeys(d.feedback for d in doubts if d.feedback))
                problems.append(f"doubt: {problem}")
                logger.info("model_query_doubted", attempt=attempt, kind=doubts[0].kind)
                continue
            frame, total, varying, missing = _run(compiled, open_sandbox)
            if isinstance(compiled, Compiled) and compiled.per_patient_varies_sql is not None:
                with open_sandbox() as sandbox:
                    per_patient_varies = int(sandbox.run_model_sql(compiled.per_patient_varies_sql).item() or 0)
                if per_patient_varies:
                    logger.info("model_query_per_patient_asked", attempt=attempt)
                    return ModelAnswer(
                        "clarify",
                        reason=PER_PATIENT_QUESTION.format(n=per_patient_varies, value=compiled.value_text),
                        choices=tuple((label, f"{question.rstrip()} {words}") for label, words in PER_PATIENT_CHOICES),
                        sql=compiled.sql,
                        reading=compiled.reading,
                        reply=failed_reply,
                        attempts=attempt,
                    )
            if isinstance(compiled, Compiled) and compiled.kind == "share" and rate_unstated(question, context):
                # "The readmission rate": out of patients or out of visits is the asker's call.
                logger.info("model_query_rate_asked", attempt=attempt)
                rows = plural(min(context.row_nouns, key=len)) if context.row_nouns else "visits"
                return ModelAnswer(
                    "clarify",
                    reason=RATE_QUESTION.format(rows=rows),
                    choices=(
                        ("Out of patients", f"{question.rstrip()} {SHARE_OF_PHRASE.format(noun='patients')}"),
                        (f"Out of {rows}", f"{question.rstrip()} {SHARE_OF_PHRASE.format(noun=rows)}"),
                    ),
                    sql=compiled.sql,
                    reading=compiled.reading,
                    reply=failed_reply,
                    attempts=attempt,
                )
            if isinstance(compiled, Compiled) and compiled.group_varies_sql is not None and compiled.group_column:
                with open_sandbox() as sandbox:
                    group_varies = int(sandbox.run_model_sql(compiled.group_varies_sql).item() or 0)
                if group_varies:
                    # A patient whose group changes between visits would be counted in each group
                    # they had: whether that, or their first or last value, is meant is the asker's.
                    logger.info("model_query_group_varies_asked", attempt=attempt)
                    column = compiled.group_column
                    return ModelAnswer(
                        "clarify",
                        reason=GROUP_VARIES_QUESTION.format(n=group_varies, column=column),
                        choices=tuple(
                            (label.format(column=column), f"{question.rstrip()} {words.format(column=column)}")
                            for label, words in GROUP_VARIES_CHOICES
                        ),
                        sql=compiled.sql,
                        reading=compiled.reading,
                        reply=failed_reply,
                        attempts=attempt,
                    )
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
                if compiled.group is not None and frame.height > sql_config.MAX_GROUPS:
                    raise PlanError(
                        f"group_by gives {frame.height} groups, too many to show. Group by a column with fewer "
                        "values, or by the year or month of a date."
                    )
                analysis = present.analysis_of(compiled, frame, total)
                if not present.has_value(analysis) and attempt == 1 and max_attempts > 1:
                    raise PlanError(
                        "No rows match the conditions. Check each value against the column's listed values, "
                        "and drop any condition the question does not state."
                    )
        except UnitError as e:
            # What the question counts is not countable on this table as asked: the model cannot
            # change that, so the user is told at once.
            logger.info("model_query_unit_refused", attempt=attempt)
            return ModelAnswer("refused", reason=str(e), reply=failed_reply, attempts=attempt)
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

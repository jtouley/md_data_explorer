# SPDX-License-Identifier: MIT
# ruff: noqa: E501  (the instructions are prose: one sentence per line reads better than wrapped)
"""The prompt: instructions and the table (the same for every question on a dataset, so the model
server reuses its work on them), then the similar examples, the conversation and the question."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

from clinical_analytics.model_query.context import DatasetContext, context_text
from clinical_analytics.model_query.examples import Example
from clinical_analytics.model_query.plan import ReplyFormat

_TESTS: Final = """\
  "compare_groups" = a statistical test (chi-square, t-test, ANOVA) of "outcome" between the groups of the column "group_by". The app picks the test and computes it.
  "odds_ratio" = an odds ratio with 95% CI of the yes/no "outcome" for the exposure column "group_by".
  "correlation" = the correlation between the number columns "outcome" and "group_by".
  "survival" = time to event: "number" is the time column, "outcome" the event column, "group_by" an optional group.
  "listing" = the question asks to list, show or identify individual patients or their IDs."""

_RULES: Final = """\
Rules:
- Use only the columns listed below, spelled exactly, and values exactly as listed (text in single quotes).
- Put every limit the question states into "where": age limits, sex, a diagnosis, a result, a year. Add no limit the question does not state.
- A word that only names what the whole table is about is not a limit.
- When the question names several values of one column ("A, B or C", "X versus not X", "each type"), set "group_by" to that column instead of filtering on one value.
- "never", "always", "on every visit" are about each patient's visits: write the condition for one visit in "where"; the app checks it over all of a patient's visits.
- Never compute a number yourself."""

_MAPPING_FIELD: Final = """\
- "mapping": at most 6 entries: each phrase of the question that limits, groups, measures or names what is counted, as {"phrase": the question's own words, "role": "filter" | "group" | "measure" | "unit" | "ignored", "column": the column it became}. Words that only name what the whole table is about, or politeness, are "ignored" with column null. If a phrase has no column in the table, map it as "ignored" with column null: never use another column instead."""

_COMMON_FIELDS: Final = """\
- "where": an SQL condition keeping only the rows the question is about, for example "age_years >= 65 AND sex = 'female'"; null when it is about everyone.
- "group_by": the column to break the answer down by; null for one overall answer.
- "numerator": only for "share": the SQL condition that is counted, for example "infection = 'yes'"; otherwise null.
- "number": only for "mean", "median", "min", "max", "sum": the number column, or an SQL expression over columns (for example to convert units); otherwise null.
- "outcome": only for the tests: the outcome column, or a yes/no SQL condition such as "stage = 'IV'"; otherwise null.
- "per_patient": only when rows are visits and the question asks for a value per patient, summarised with "mean", "median", "min" or "max": "count" = the number of each patient's rows (meeting "where"), "first" / "last" = each patient's first or last value of "number" (baseline = first), "span_days" = days from each patient's first to last date ("number" = the date column), "gap_days" = days between a patient's consecutive dates ("number" = the date column). Otherwise null.
- "order_by": only with per_patient first or last when the table has several date columns: the column that orders a patient's visits; otherwise null."""

PLAN_INSTRUCTIONS: Final = (
    """\
You help a physician query one data table. Reply with ONE JSON object that says what to compute. The app writes and runs the SQL from it.

Fields:
- "thought": one short sentence naming the columns and values the question means.
- "answer": one of
  "count" = how many (patients, admissions, tests) meet "where". With "group_by": how many in each group; also for "what share of patients is in each category of a column" (the app adds the percentages).
  "share" = the share (percentage, proportion, rate) for which the condition "numerator" is true, among the rows kept by "where". With "group_by": that share inside each group.
  "mean", "median", "min", "max", "sum" = that summary of "number". With "group_by": one per group ("in men compared with women" means group_by the sex column).
"""
    + _TESTS
    + "\n"
    + _MAPPING_FIELD
    + "\n"
    + _COMMON_FIELDS
    + "\n\n"
    + _RULES
    + """
- Do not write COUNT, AVG or other aggregates: the app counts the right thing from the grain."""
)

SQL_INSTRUCTIONS: Final = (
    """\
You help a physician query one data table. Reply with ONE JSON object.

Fields:
- "thought": one short sentence saying in plain words what the query computes.
- "answer": one of
  "sql" = counts, shares, means, medians and breakdowns: write the query in "sql".
"""
    + _TESTS
    + """
- "sql": only for "sql": one DuckDB SELECT over the table. Put the group column first and name the result column count, rate (a proportion from 0 to 1), mean or median. Otherwise null.
- "where", "group_by", "number", "outcome": only for the tests ("where" is an SQL condition, the others column names); otherwise null. "numerator" is always null.

"""
    + _RULES
    + """
- Count as the Grain line below says."""
)


def system_prompt(context: DatasetContext, fmt: ReplyFormat) -> str:
    instructions = PLAN_INSTRUCTIONS if fmt == "plan" else SQL_INSTRUCTIONS
    return f"{instructions}\n\n{context_text(context, writes_sql=fmt == 'sql')}"


def user_prompt(
    question: str,
    examples: Sequence[Example],
    fmt: ReplyFormat,
    *,
    previous: Sequence[tuple[str, dict[str, Any]]] = (),
    failed_reply: dict[str, Any] | None = None,
    problem: str | None = None,
    related: Sequence[str] = (),
) -> str:
    """Examples, then the conversation so far, the question, and a failed attempt's feedback.

    ``previous``: earlier (question, reply) turns of this conversation, oldest first.
    ``related``: the description lines of the columns closest to the question (wide tables).
    """
    parts: list[str] = []
    if examples:
        parts.append("Examples on other tables (their columns are not in your table):")
        parts += [example.text(fmt) for example in examples]
    if previous:
        parts.append(
            "Earlier in this conversation, on your table. If the new question refines the last one "
            '("now only the women", "and by year?"), repeat the last reply with that change.'
        )
        parts += [f"Question: {asked}\nReply: {json.dumps(reply)}" for asked, reply in previous]
    if related:
        parts.append("Columns of your table that look closest to the question (the rest are listed above):")
        parts.append("\n".join(related))
    parts.append(f"Now your table.\nQuestion: {question}")
    if problem:
        if failed_reply is not None:
            parts.append(f"Your last reply was: {json.dumps(failed_reply)}")
        parts.append(f"It could not be used: {problem}\nReply again with the corrected JSON object.")
    else:
        parts.append("Reply:")
    return "\n\n".join(parts)

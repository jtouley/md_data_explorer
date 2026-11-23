# SPDX-License-Identifier: MIT
# ruff: noqa: E501  (the instructions are prose: one sentence per line reads better than wrapped)
"""The prompt: instructions and the table (the same for every question on a dataset, so the model
server reuses its work on them), then similar worked examples, the conversation and the question.

The reply schema (``plan.reply_schema``) holds the reply to the table's columns and values; the
instructions say what each slot means, and the examples show every slot filled in."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

from clinical_analytics.model_query.context import DatasetContext, context_text
from clinical_analytics.model_query.examples import Example

INSTRUCTIONS: Final = """\
You help a physician query one data table. Fill in ONE JSON plan; the app writes and runs the query from it.

Fields:
- "answer": "count" = how many. "share" = a share, percent, proportion or rate. "mean", "median", "min", "max", "sum" = that summary of "measure". "count_distinct" = how many different values "measure" takes. "top" = which group of "group_by" has the most.
  "compare_groups" = a test of whether "measure" differs between the groups of "group_by". "odds_ratio" = the odds ratio of the yes/no "numerator" for "group_by". "correlation" = between the numbers "measure" and "group_by". "survival" = time to event: "measure" is the time column, "numerator" the event, "group_by" optional.
  "list_patients" = the question asks to list or identify individual patients.
  "not_in_table" = the question needs something no column holds (deaths, a population for a rate): set "missing" to the question's words for it. Never answer a different figure instead.
- "count_of": what a count or a share counts: patients, or the table's records.
- "filters": every limit the question states, each {"column", "op", "value", "values"}. A category: op "is" or "is not" with "values" from its list. A number: op "<", "<=", ">", ">=" with "value", or "is" with "values". A date: "before" or "on or after" with "value" YYYY-MM-DD. "is missing" / "is recorded" = not recorded / recorded. Add no limit the question does not state.
- "numerator": for "share", "odds_ratio" and "survival" only: the condition counted, among the records "filters" keep. Otherwise [].
- "measure": the number column a summary or a test reads; otherwise null.
- "group_by": [the column] to break the answer down by ("by sex", "for each arm", "A versus B"); otherwise []. A number column needs "cut_points" ([65] = under 65 and 65 or more). A date column needs "period" "year" or "month".
- "which_visit": when patients have several records: "any" = at any of their records (ever), "every" = at every record (always), "none" = at none of their records (never: write the condition they never had, positively, in "numerator"), "first" = at their first record (baseline), "last" = at their latest record, "change" = latest minus first (summaries only). Use "any" unless the question says which.

- "totals_by": for a summary of totals per group ("the average number per year"): the column whose groups are totalled; the answer (mean, median, min, max) is then taken over those totals. Otherwise null.
- "missing": only for "not_in_table"; otherwise null.

Rules:
- "Compare A and B" asks for the figures side by side (a count, share or mean with "group_by"); use a test only when the question asks for one (significant, p-value, odds ratio, correlation, survival).
- Use only the columns and values listed for your table.
- A word that only names what the whole table is about is not a limit.
- Never compute a number yourself."""


def system_prompt(context: DatasetContext, keep: set[str] | None = None) -> str:
    return f"{INSTRUCTIONS}\n\n{context_text(context, keep)}"


def user_prompt(
    question: str,
    examples: Sequence[Example],
    *,
    previous: Sequence[tuple[str, dict[str, Any]]] = (),
    failed_reply: dict[str, Any] | None = None,
    problem: str | None = None,
) -> str:
    """Examples, then the conversation so far, the question, and a failed attempt's feedback.

    ``previous``: earlier (question, reply) turns of this conversation, oldest first.
    """
    parts: list[str] = []
    if examples:
        parts.append("Examples on other tables (their columns are not in your table):")
        parts += [example.text() for example in examples]
    if previous:
        parts.append(
            "Earlier in this conversation, on your table. If the new question refines the last one "
            '("now only the women", "and by year?"), repeat the last reply with that change.'
        )
        parts += [f"Question: {asked}\nReply: {json.dumps(reply)}" for asked, reply in previous]
    parts.append(f"Now your table.\nQuestion: {question}")
    if problem:
        if failed_reply is not None:
            parts.append(f"Your last reply was: {json.dumps(failed_reply)}")
        parts.append(f"It could not be used: {problem}\nReply again with the corrected JSON object.")
    else:
        parts.append("Reply:")
    return "\n\n".join(parts)

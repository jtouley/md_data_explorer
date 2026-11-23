# SPDX-License-Identifier: MIT
"""The reading card: how an answer was computed, in clinical words, from templates.

Built from the compiled plan, the contract's facts (each column's description, labels, unit, the
records' noun) and the computed numbers, never from the model's words. Every card names the unit
counted and the timepoint; counts read "N of M (x%)". No SQL, column names, "null", "grain",
"distinct" or "row" (``card_problems`` is the linter the tests run over every card).
"""

from __future__ import annotations

import re
from typing import Any, Final

from clinical_analytics.model_query.context import Column, DatasetContext
from clinical_analytics.model_query.plan import Compiled, Condition

MAX_LINE_WORDS: Final = 25
_BANNED: Final = re.compile(
    r"\b(?:sql|select|from unified|where|null|nulls|grain|distinct|rows?|group by|bool_\w+|count_if|_rank)\b|"
    r"[=<>]|\bcount\(",
    re.I,
)
_STATISTIC_WORDS: Final = {"mean": "average", "median": "median", "min": "lowest", "max": "highest", "sum": "total"}
_TEST_WORDS: Final = {
    "compare_groups": "a comparison of {outcome} between the groups of {group}",
    "odds_ratio": "the odds ratio of {outcome} for {group}, with its 95% confidence interval",
    "correlation": "the correlation between {outcome} and {group}",
    "survival": "time to {outcome} ({time})",
}
_OPS: Final = {"<": "under", "<=": "at most", ">": "over", ">=": "at least", "before": "before", "on or after": "from"}


def _number(value: Any, column: Column) -> str:
    text = f"{int(value):,}" if isinstance(value, int | float) and float(value).is_integer() else f"{value:g}"
    return f"{text} {column.unit}" if column.unit else text


def condition_words(cond: Condition, context: DatasetContext) -> str:
    column = context.column(cond.column)
    if column is None:
        return "an unknown limit"
    name = column.display
    if cond.op == "is missing":
        return f"{name} not recorded"
    if cond.op == "is recorded":
        return f"{name} recorded"
    if cond.op in _OPS:
        value = cond.value if column.kind != "date" else str(cond.value)
        return f"{name} {_OPS[cond.op]} {_number(value, column) if column.numeric else value}"
    said = [v for v in (cond.values or (cond.value,)) if v is not None]
    if column.numeric:
        words = " or ".join(_number(v, column) for v in said)
    else:
        words = " or ".join(column.label(column.stored(str(v))) for v in said)
    return f"{name} {'not ' if cond.op == 'is not' else ''}{words}"


def _split(head: str, parts: list[str]) -> list[str]:
    """``head`` + parts joined by "; ", over as many lines as keep each at most 25 words."""
    lines: list[str] = []
    current: list[str] = []
    for part in parts:
        trial = f"{head} {'; '.join([*current, part])}."
        if current and len(trial.split()) > MAX_LINE_WORDS:
            lines.append(f"{head} {'; '.join(current)}.")
            current = []
        current.append(part)
    if current:
        lines.append(f"{head} {'; '.join(current)}.")
    return lines


def _counted_line(compiled: Compiled, context: DatasetContext) -> str:
    noun, nouns = context.row_noun, context.row_nouns
    which = compiled.plan.which_visit
    if compiled.plan.answer == "count_distinct":
        return f"Counted: the {compiled.unit}."
    if context.aggregate:
        column = context.column(compiled.measure or "")
        what = column.display if column is not None else "the count"
        return f"Counted: the total of the case counts ({what}); each record counts {context.case_noun} for one group."
    if context.semantics.one_row_per_patient:
        return "Counted: patients, one record each."
    if context.patient_key is None:
        return f"Counted: {nouns}; no column names a patient, so each {noun} counts on its own."
    if not compiled.per_patient:
        return f"Counted: {nouns}, every {noun} on its own; a patient with several {nouns} counts once per {noun}."
    if which == "first":
        return f"Counted: {compiled.unit}, each once, read at their first {noun} with it recorded."
    if which == "last":
        return f"Counted: {compiled.unit}, each once, read at their latest {noun} with it recorded."
    if which == "every":
        return f"Counted: {compiled.unit}, each once, when every one of their {nouns} with it recorded met the limits."
    return f"Counted: {compiled.unit}, each once, when any one of their {nouns} met the limits."


def _measured_line(compiled: Compiled, context: DatasetContext) -> str:
    if compiled.lookup:
        column = context.column(compiled.measure or "")
        what = column.display if column is not None else "the estimate"
        return f"Read: the published {what} of the group named; estimates are never added up."
    if compiled.plan.totals_by is not None:
        by = context.column(compiled.plan.totals_by)
        if context.aggregate:
            total = "total of the case counts"
        elif context.visit_level and compiled.plan.count_of != context.row_nouns:
            total = f"number of {compiled.plan.count_of}"
        else:
            total = f"number of {context.row_nouns}"
        word = _STATISTIC_WORDS[compiled.plan.answer]
        return f"Measured: the {word} of the {total} per {by.display if by else 'group'}."
    column = context.column(compiled.measure or "")
    what = column.display if column else "the measure"
    unit = f" ({column.unit})" if column is not None and column.unit else ""
    word = _STATISTIC_WORDS[compiled.plan.answer]
    noun = context.row_noun
    which = compiled.plan.which_visit if compiled.per_patient else "any"
    if context.semantics.one_row_per_patient:
        timing = "one value per patient"
    elif compiled.constant:
        timing = f"each patient's value once (the same at every {noun})"
    elif which == "first":
        timing = "each patient's first recorded value"
    elif which == "last":
        timing = "each patient's latest recorded value"
    elif which == "change":
        timing = "each patient's change from their first to their latest recorded value"
    else:
        timing = f"every {noun} with a value"
    return f"Measured: the {word} of {what}{unit}, using {timing}."


def _pct(part: int, whole: int, unit: str) -> str:
    share = f" ({100.0 * part / whole:.1f}%)" if whole else ""
    return f"{part:,} of {whole:,} {unit}{share}"


def _result_lines(
    compiled: Compiled, analysis: dict[str, Any], total: int | None, context: DatasetContext
) -> list[str]:
    unit = compiled.unit
    kind = analysis.get("type")
    if kind == "count" and not analysis.get("grouped_by"):
        n = int(analysis.get("total_count") or 0)
        return [f"Result: {_pct(n, total, unit)}." if total is not None else f"Result: {n:,} {unit}."]
    if kind == "count" and compiled.plan.answer == "top":
        lines = [f"Result: the group with the most {unit}; tied groups are shown together."]
        if analysis.get("missing_group"):
            lines.append(f"Not in any group: {analysis['missing_group']:,} {unit} with none recorded.")
        return lines
    if kind == "count":
        overlap = " A patient can be in more than one group." if analysis.get("overlapping") else ""
        kept = " who met the limits" if compiled.filters else " in the file"
        base = f" out of {total:,} {unit}{kept}" if total is not None else ""
        return [f"Result: each group shows its number of {unit}{base}.{overlap}"]
    if kind == "proportion":
        events, n = int(analysis.get("events") or 0), int(analysis.get("n_used") or 0)
        return [f"Result: {_pct(events, n, unit)}."]
    if kind == "rate":
        return [f"Result: each group shows how many of its {unit} met it, as N of M (x%)."]
    if kind == "descriptive":
        value, n = analysis.get(analysis.get("statistic") or "mean"), int(analysis.get("non_null_count") or 0)
        column = context.column(compiled.measure or "")
        shown = f"{value:.4g}" if isinstance(value, float) else str(value)
        unit_words = f" {column.unit}" if column is not None and column.unit else ""
        return [f"Result: {shown}{unit_words}, from {n:,} {unit}."]
    if kind == "grouped_statistic":
        return [f"Result: one value per group, each with the number of {unit} behind it."]
    return []


def reading_lines(
    compiled: Compiled, context: DatasetContext, analysis: dict[str, Any] | None = None, total: int | None = None
) -> list[str]:
    """The card's lines (each at most 25 words)."""
    lines: list[str] = []
    if compiled.kind == "test":
        names = {
            k: (c.display if (c := context.column(v or "")) else "the outcome")
            for k, v in (
                ("outcome", compiled.test_outcome),
                ("group", compiled.test_group),
                ("time", compiled.test_time),
            )
        }
        lines.append(f"Test: {_TEST_WORDS[compiled.plan.answer].format(**names)}.")
        lines.append(f"Computed by the app's statistics code over {context.row_nouns}.")
    elif compiled.kind == "statistic":
        lines.append(_measured_line(compiled, context))
    else:
        lines.append(_counted_line(compiled, context))
    conditions = [condition_words(c, context) for c in compiled.filters]
    lines += _split("Included:", conditions) if conditions else ["Included: everyone in the file."]
    if compiled.numerator and compiled.kind == "share":
        lines += _split("Counted as yes:", [condition_words(c, context) for c in compiled.numerator])
        lines.append("Out of: those with it recorded.")
    if compiled.group_column:
        column = context.column(compiled.group_column)
        by = column.display if column else "group"
        if column is not None and column.kind == "date":
            by = f"{'month' if compiled.plan.period == 'month' else 'year'} of {by}"
        if compiled.kind == "statistic" and compiled.per_patient:
            at = "latest" if compiled.plan.which_visit == "last" else "first"
            by = f"{by}, at each patient's {at} {context.row_noun}"
        lines.append(f"Broken down by: {by}.")
    if compiled.left_out:
        lines.append(f"Left out of the base: answers {', '.join(repr(w) for w in compiled.left_out)}.")
    if compiled.zero_filled:
        lines.append("Groups with none count as zero.")
    if analysis is not None:
        lines += _result_lines(compiled, analysis, total, context)
    if context.semantics.survey_sample:
        lines.append("Counts are of this file's respondents, not weighted national estimates.")
    return lines


def reading(lines: list[str]) -> str:
    return "\n".join(lines)


def card_problems(lines: list[str], context: DatasetContext) -> list[str]:
    """What breaks the card's rules: a line over 25 words, a technical term, or a raw column
    name (one written with "_" or ".", the identifiers a doctor should never have to read)."""
    found: list[str] = []
    raw = [c.name for c in context.columns if re.search(r"[_.]", c.name)]
    for line in lines:
        if len(line.split()) > MAX_LINE_WORDS:
            found.append(f"over {MAX_LINE_WORDS} words: {line}")
        if (match := _BANNED.search(line)) is not None:
            found.append(f"technical term '{match.group(0)}': {line}")
        for name in raw:
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", line):
                found.append(f"column name '{name}': {line}")
    return found

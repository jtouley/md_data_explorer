# SPDX-License-Identifier: MIT
"""Data checks on the columns a question uses, run before the answer is computed.

Each check ends in one of three ways, and never in a silent pass:

* refuse: the answer would be wrong whatever the doctor says (a mean over values recorded as
  "<5"; a column with no values; a group count where a patient has two groups);
* hold: the answer depends on something only the doctor can say (are rows visits? are "Male"
  and "M" one level? are ages of 400 entry errors?), and runs once they confirm;
* note: a sentence added to the answer (implausible values the doctor chose to include, a
  failed check in the dataset's latest quality report).

Counts in sentences follow the small-cell rule (``line_count_text``): a count from 1 to 10, or
one whose neighbour in its line is, reads "some". Sentences name columns, ranges and category
labels, never a patient's value.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from clinical_analytics.analysis.grouping import level_variants
from clinical_analytics.analysis.numeric_text import AmbiguousNumberFormatError, parse_numeric_text
from clinical_analytics.analysis.variable_types import is_numeric_dtype, short_name
from clinical_analytics.contracts.builder import find_plausibility_range
from clinical_analytics.contracts.declarations import Declarations, GrainDeclaration, LevelMapping
from clinical_analytics.contracts.quality import RANGE_METRIC, QualityReport
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.result_presentation import line_count_text
from clinical_analytics.core.schema_inference import patient_key_by_name

_EQUALITY_OPERATORS = frozenset({"==", "!=", "IN", "NOT_IN"})
_STATISTIC_INTENTS = frozenset({"DESCRIBE", "COMPARE_GROUPS"})
# Report checks an answer already handles itself (hold or disclosure), so not repeated as notes.
_HANDLED_METRICS = frozenset({RANGE_METRIC, "duplicateValues", "censoredValues", "levelVariants"})


@dataclass(frozen=True)
class Grain:
    """The patient key, rows, and distinct non-null keys of the dataset (None: no key)."""

    key: str | None
    rows: int
    patients: int | None

    @property
    def repeated(self) -> bool:
        return self.patients is not None and self.patients < self.rows

    def declaration(self) -> GrainDeclaration | None:
        if self.key is None or self.patients is None:
            return None
        return GrainDeclaration(key=self.key, rows=self.rows, patients=self.patients)


@dataclass
class Findings:
    """What the checks found for one plan."""

    refusal: str | None = None
    holds: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    grain_to_declare: GrainDeclaration | None = None
    levels_to_declare: list[LevelMapping] = field(default_factory=list)


def patient_key(layer: Any) -> str | None:
    """The base view's patient key: the column mapped to ``patient_id``, else one named for it."""
    columns = list(layer.get_base_view().columns)
    mapping = (getattr(layer, "config", None) or {}).get("column_mapping") or {}
    if "patient_id" in mapping.values() and "patient_id" in columns:
        return "patient_id"
    return patient_key_by_name(columns)


def grain_of(layer: Any) -> Grain:
    """Rows and distinct patient keys, counted exactly (``n_unique``, never a sample)."""
    view = layer.get_base_view()
    key = patient_key(layer)
    rows = int(view.count().execute())
    if key is None:
        return Grain(key=None, rows=rows, patients=None)
    patients = int(view[key].nunique().execute())  # nulls are not a patient
    return Grain(key=key, rows=rows, patients=patients)


def declared_visits(grain: Grain, declarations: Declarations) -> bool:
    """Whether the stored grain declaration matches the data as it is now."""
    stored = declarations.grain
    return stored is not None and grain.declaration() == stored


def grain_hold_reason(key: str) -> str:
    return (
        f"Some IDs in {key} appear on more than one row, so a row may be a visit rather than a patient. "
        f"Confirm that rows are visits to count each patient once."
    )


def plan_columns(plan: QueryPlan) -> list[str]:
    """Every column the plan reads, in plan order, once each."""
    found: list[str] = []
    for column in (
        plan.metric,
        plan.group_by,
        plan.proportion_column,
        plan.rate_column,
        *(f.column for f in plan.filters),
        *(s.column for s in plan.cohort_steps),
    ):
        if column and column not in found:
            found.append(str(column))
    return found


def _plan_values(plan: QueryPlan) -> Iterator[tuple[str, Any]]:
    """(column, value) for every equality limit of the plan."""
    for spec in plan.filters:
        if spec.operator in _EQUALITY_OPERATORS:
            for value in spec.value if isinstance(spec.value, list) else [spec.value]:
                yield spec.column, value
    for step in plan.cohort_steps:
        if step.op == "eq":
            yield step.column, step.value
    if plan.proportion_column:
        yield plan.proportion_column, plan.proportion_value
    if plan.rate_column:
        yield plan.rate_column, plan.rate_value


def metric_refusal(plan: QueryPlan, frame: pl.DataFrame) -> str | None:
    """Why a mean or comparison of the plan's metric would be wrong, or None."""
    metric = plan.metric
    if plan.intent not in _STATISTIC_INTENTS or not metric or metric not in frame.columns:
        return None
    series = frame.get_column(metric)
    what = "describe" if plan.intent == "DESCRIBE" else "compare"
    if series.null_count() == series.len():
        return f"Every value of {short_name(metric)} is missing, so there is nothing to {what}."
    if series.dtype != pl.String:
        return None
    try:
        parsed = parse_numeric_text(series)
    except AmbiguousNumberFormatError as exc:
        return str(exc)
    if parsed is None:
        return None
    if parsed.values.null_count() == series.len():
        return f"Every value of {short_name(metric)} is missing, so there is nothing to {what}."
    if parsed.n_censored:
        present = series.len() - series.null_count() - parsed.n_missing_tokens
        amount = line_count_text(parsed.n_censored, [present - parsed.n_censored])
        return (
            f"{short_name(metric)} has {amount} values recorded as a limit, such as '<5', so a statistic of it "
            f"would be wrong. Replace the limits with measured values in the file, or ask for a count by "
            f"{short_name(metric)}."
        )
    return None


def implausible_counts(frame: pl.DataFrame, column: str) -> tuple[int, int, str] | None:
    """(values outside the range, values present, range text) for a numeric column with a range."""
    bounds = find_plausibility_range(column)
    if bounds is None or column not in frame.columns:
        return None
    series = frame.get_column(column)
    if not is_numeric_dtype(series.dtype):
        return None
    values = series.drop_nulls()
    if values.dtype.is_float():
        values = values.filter(values.is_finite())
    outside = values < bounds.minimum
    if bounds.maximum is not None:
        outside = outside | (values > bounds.maximum)
    count = int(outside.sum())
    if count == 0:
        return None
    text = (
        f"{bounds.minimum:g}–{bounds.maximum:g} {bounds.unit}"
        if bounds.maximum is not None
        else f"below {bounds.minimum:g} {bounds.unit}"
    )
    return count, int(values.len()), text


def implausible_sentences(plan: QueryPlan, frame: pl.DataFrame) -> list[tuple[str, str, str]]:
    """(column, hold sentence, note sentence) for each plan column with implausible values."""
    found = []
    for column in plan_columns(plan):
        counts = implausible_counts(frame, column)
        if counts is None:
            continue
        count, present, bounds = counts
        amount = line_count_text(count, [present - count])
        name = short_name(column)
        hold = (
            f"{name} has {amount} values outside {bounds}, which are likely entry errors. "
            f"Correct them in the file, or confirm to include them in the answer."
        )
        note = f"{name} has {amount} values outside {bounds} (likely entry errors), included as recorded."
        found.append((column, hold, note))
    return found


def variant_findings(
    plan: QueryPlan, frame: pl.DataFrame, declarations: Declarations
) -> list[tuple[str, LevelMapping, str]]:
    """(column, proposed mapping, hold sentence) for a grouped or limited column whose levels
    include spellings of one category the doctor has not confirmed as one."""
    confirmed = {(m.column, m.spellings) for m in declarations.levels}
    grouped = {plan.group_by} if plan.group_by else set()
    limited: dict[str, set[str]] = {}
    for column, value in _plan_values(plan):
        if isinstance(value, str) and column not in grouped:
            limited.setdefault(column, set()).add(value)
    asked: dict[str, set[str] | None] = {**limited, **dict.fromkeys(grouped)}
    found = []
    for column, values in asked.items():
        if column not in frame.columns or frame.get_column(column).dtype != pl.String:
            continue
        counts = frame.get_column(column).drop_nulls().value_counts(sort=True)
        frequency = dict(zip(counts.get_column(column).to_list(), counts.get_column("count").to_list(), strict=True))
        for group in level_variants(list(frequency)):
            if (column, group) in confirmed or (values is not None and not values & set(group)):
                continue
            canonical = max(group, key=lambda spelling: (frequency[spelling], spelling))
            shown = ", ".join(group)
            sentence = (
                f"{short_name(column)} has values that may be one category written differently ({shown}). "
                f"Confirm to count them together as {canonical}."
            )
            found.append((column, LevelMapping(column=column, spellings=group, canonical=canonical), sentence))
    return found


def grouped_count_refusal(frame: pl.DataFrame, key: str, group: str) -> str | None:
    """At visit grain, patients are counted once per group only if each has one group value."""
    if key not in frame.columns or group not in frame.columns:
        return None
    pairs = frame.select(key, group).drop_nulls(key).unique()
    if pairs.height == pairs.get_column(key).n_unique():
        return None
    return (
        f"{short_name(group)} differs between rows of the same patient, so patients cannot be counted once per "
        f"{short_name(group)} group. Ask for the number of visits by {short_name(group)}, or count patients without "
        f"grouping."
    )


def report_notes(report: QualityReport | None, columns: list[str]) -> list[str]:
    """Failed or flagged checks of the dataset's latest quality report on the plan's columns,
    and failed table-level checks, as sentences (messages are already small-cell suppressed)."""
    if report is None:
        return []
    names = {short_name(column) for column in columns} | set(columns)
    notes = []
    for check in report.checks:
        if check.status not in ("failed", "warning", "review") or check.metric in _HANDLED_METRICS:
            continue
        if (check.column is None and check.status == "failed") or check.column in names:
            notes.append(f"Data check: {check.message}.")
    return notes

# SPDX-License-Identifier: MIT
"""Data checks on the columns a question uses, run before the answer is computed.

Each check ends in one of three ways, and never in a silent pass:

* refuse: the answer would be wrong whatever the doctor says (a mean over values recorded as
  "<5"; a column with no values; a group count or share where a patient has two values);
* hold: the answer depends on something only the doctor can say (are rows visits? are "Male"
  and "M" one level? are ages of 400 entry errors? are glucose values below 30 mmol/L?), and
  runs once they confirm;
* note: a sentence added to the answer (implausible values the doctor chose to include, a
  check of the dataset's latest quality report on a column the question uses).

Every check is a DuckDB aggregate over the layer's base view (counts, a median, value counts of
one column): no column is copied into Python per question. Row and patient counts reuse the
layer's cached base counts.

Counts in sentences follow the small-cell rule (``line_count_text``): a count from 1 to 10, or
one whose neighbour in its line is, reads "some". Sentences name columns, ranges and category
labels held by at least ``MIN_CELL_SIZE`` rows, never a patient's value.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from clinical_analytics.analysis.grouping import level_variants
from clinical_analytics.analysis.numeric_text import all_missing_message, statistic_refusal
from clinical_analytics.analysis.variable_types import short_name
from clinical_analytics.contracts.builder import Implausible, find_plausibility_range, implausible_values, range_for
from clinical_analytics.contracts.builder import range_text as plausibility_text
from clinical_analytics.contracts.config import MIN_CELL_SIZE, MISSINGNESS_WARN_PERCENT
from clinical_analytics.contracts.declarations import Declarations, GrainDeclaration, LevelMapping, UnitRule
from clinical_analytics.contracts.quality import RANGE_METRIC, UNIT_MIX_METRIC, QualityReport
from clinical_analytics.contracts.unit_mix import UnitMix, detect_unit_mix, unit_pair_for
from clinical_analytics.core.plan_values import plan_values
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.result_presentation import line_count_text
from clinical_analytics.core.schema_inference import patient_key_by_name

_STATISTIC_INTENTS = frozenset({"DESCRIBE", "COMPARE_GROUPS"})
# Report checks an answer already handles itself (hold or disclosure), so not repeated as notes.
_HANDLED_METRICS = frozenset({RANGE_METRIC, "duplicateValues", "censoredValues", "levelVariants", UNIT_MIX_METRIC})
# Plan filters that keep a range of a column's values ("glucose above 30": one unit's cluster).
_RANGE_OPERATORS = {">": "gt", ">=": "ge", "<": "lt", "<=": "le"}


@dataclass(frozen=True)
class Grain:
    """The patient key, rows, distinct non-null keys, and rows with no key (None: no key)."""

    key: str | None
    rows: int
    patients: int | None
    missing_keys: int = 0

    @property
    def repeated(self) -> bool:
        """Some ID is on more than one row. Rows with no ID repeat no ID, so they are left out:
        100 rows with 98 IDs and 2 empty ones are one row per patient."""
        return self.patients is not None and self.patients < self.rows - self.missing_keys

    @property
    def rows_are_patients(self) -> bool:
        """Every row has an ID and no ID repeats: counting rows counts patients."""
        return self.key is not None and not self.repeated and self.missing_keys == 0

    def declaration(self, dataset_version: str | None) -> GrainDeclaration | None:
        if self.key is None or self.patients is None:
            return None
        return GrainDeclaration(key=self.key, rows=self.rows, patients=self.patients, dataset_version=dataset_version)


@dataclass
class Findings:
    """What the checks found for one plan."""

    refusal: str | None = None
    holds: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    grain_to_declare: GrainDeclaration | None = None
    levels_to_declare: list[LevelMapping] = field(default_factory=list)
    units_to_declare: list[UnitRule] = field(default_factory=list)
    keyed: bool = False  # the dataset has a patient key, so every n can say what it counts
    visits: bool = False  # rows are declared visits (on this dataset version)
    key: str | None = None  # the patient key
    rows_are_patients: bool = False  # every row has an ID and none repeats


def patient_key(layer: Any) -> str | None:
    """The base view's patient key: the column mapped to ``patient_id``, else the best-ranked
    column named for it that holds more than one value."""
    columns = list(layer.get_base_view().columns)
    mapping = (getattr(layer, "config", None) or {}).get("column_mapping") or {}
    if "patient_id" in mapping.values() and "patient_id" in columns:
        return "patient_id"
    rows = layer._base_count(None)
    return patient_key_by_name(columns, distinct=lambda column: layer._base_count(column) if rows > 1 else 2)


def grain_of(layer: Any) -> Grain:
    """Rows and distinct patient keys, counted exactly with the layer's cached base counts (the
    same counts a COUNT answer reports, so the hold and the answer agree)."""
    key = patient_key(layer)
    rows = int(layer._base_count(None))
    if key is None:
        return Grain(key=None, rows=rows, patients=None)
    return Grain(
        key=key, rows=rows, patients=int(layer._base_count(key)), missing_keys=int(layer.base_missing_count(key))
    )


def missing_key_note(grain: Grain, visits: bool) -> str | None:
    """The note for rows with no patient ID, or None. The count follows the small-cell rule
    (its line: the rows that have an ID). At visit grain those rows are in no patient count."""
    if grain.key is None or grain.missing_keys == 0:
        return None
    amount = line_count_text(grain.missing_keys, [grain.rows - grain.missing_keys])
    counted = "; those rows are not counted as patients" if visits else ""
    return f"Data check: {grain.key} is missing on {amount} rows{counted}."


def declared_visits(grain: Grain, declarations: Declarations, dataset_version: str | None) -> bool:
    """Whether the stored grain declaration was made on this dataset version and still matches
    its counts (a corrected re-upload of the same shape is a new version: asked again)."""
    stored = declarations.grain
    return stored is not None and stored == grain.declaration(dataset_version)


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


def _equality_values(plan: QueryPlan) -> Iterator[tuple[str, Any]]:
    """(column, value) for every equality limit and share value of the plan."""
    for column, value, compared in plan_values(plan):
        if not compared:
            yield column, value


def value_counts(layer: Any, column: str) -> tuple[pl.Series, pl.Series]:
    """(distinct values of ``column``, nulls included; rows per value), counted in DuckDB once
    per column for the layer (``SemanticLayer.base_value_counts``): only the distinct values
    cross into Polars, and the checks of one question share them."""
    values, counts = layer.base_value_counts(column)
    return values, counts


def _is_string(layer: Any, column: str) -> bool:
    return bool(layer.get_base_view()[column].type().is_string())


def metric_refusal(plan: QueryPlan, layer: Any) -> str | None:
    """Why a statistic of the plan's metric would be wrong, or None (one wording for every path:
    ``analysis.numeric_text.statistic_refusal``)."""
    metric = plan.metric
    view = layer.get_base_view()
    if plan.intent not in _STATISTIC_INTENTS or not metric or metric not in view.columns:
        return None
    what = "describe" if plan.intent == "DESCRIBE" else "compare"
    rows = int(layer._base_count(None))
    if not _is_string(layer, metric):
        present = int(view[metric].count().execute())
        return all_missing_message(metric, what) if present == 0 and rows else None
    values, counts = value_counts(layer, metric)
    return statistic_refusal(values, what, counts, rows)[0]


def _implausible_numeric(layer: Any, column: str) -> Implausible | None:
    """``implausible_values`` for a numeric column, as two DuckDB aggregates (median, then the
    count outside the range the median selects)."""
    view = layer.get_base_view()
    values = view.filter(view[column].notnull())
    if values[column].type().is_floating():
        values = values.filter(~values[column].isnan() & ~values[column].isinf())
    stats = values.aggregate(median=values[column].median(), present=values[column].count()).to_polars().row(0)
    median, present = stats
    bounds = range_for(column, float(median) if median is not None else None)
    if bounds is None or not present:
        return None
    outside = values[column] < bounds.minimum
    if bounds.maximum is not None:
        outside = outside | (values[column] > bounds.maximum)
    count = int(values.aggregate(k=outside.cast("int64").sum()).to_polars().item() or 0)
    if count == 0:
        return None
    return Implausible(column=column, count=count, present=int(present), bounds=plausibility_text(bounds))


def implausible_in_layer(layer: Any, column: str) -> Implausible | None:
    """Values of ``column`` outside its plausibility range, numbers stored as text included."""
    view = layer.get_base_view()
    if find_plausibility_range(column) is None or column not in view.columns:
        return None
    dtype = view[column].type()
    if dtype.is_string():
        values, counts = value_counts(layer, column)
        return implausible_values(values, counts)
    if dtype.is_numeric() and not dtype.is_boolean():
        return _implausible_numeric(layer, column)
    return None


def implausible_sentences(plan: QueryPlan, layer: Any) -> list[tuple[str, str, str]]:
    """(column, hold sentence, note sentence) for each plan column with implausible values."""
    found = []
    for column in plan_columns(plan):
        result = implausible_in_layer(layer, column)
        if result is None:
            continue
        amount = line_count_text(result.count, [result.present - result.count])
        name = short_name(column)
        hold = (
            f"{name} has {amount} values outside {result.bounds}, which are likely entry errors. "
            f"Correct them in the file, or confirm to include them in the answer."
        )
        note = f"{name} has {amount} values outside {result.bounds} (likely entry errors), included as recorded."
        found.append((column, hold, note))
    return found


def variant_sentence(column: str, group: tuple[str, ...], frequency: dict[str, int]) -> tuple[str, str]:
    """(canonical spelling, hold sentence) for spellings that may be one category.

    Only spellings held by at least ``MIN_CELL_SIZE`` rows are named, and the canonical one is
    chosen from those: a label of a level with 1 to 10 rows is never shown. With fewer than two
    such spellings the sentence names none.
    """
    large = [s for s in group if frequency.get(s, 0) >= MIN_CELL_SIZE]
    pool = large or list(group)
    canonical = max(pool, key=lambda spelling: (frequency.get(spelling, 0), spelling))
    name = short_name(column)
    if len(large) >= 2:
        rare = ", and some rare spellings" if len(large) < len(group) else ""
        shown = f"({', '.join(large)}{rare})"
    else:
        shown = "(including some rare spellings)"
    target = canonical if large else "one category"
    return canonical, (
        f"{name} has values that may be one category written differently {shown}. "
        f"Confirm to count them together as {target}."
    )


def variant_findings(
    plan: QueryPlan, layer: Any, declarations: Declarations, dataset_version: str | None
) -> list[tuple[str, LevelMapping, str]]:
    """(column, proposed mapping, hold sentence) for a grouped or limited text column whose levels
    include spellings of one category the doctor has not confirmed as one (on this version)."""
    confirmed = {(m.column, m.spellings) for m in declarations.levels if m.dataset_version == dataset_version}
    grouped = {plan.group_by} if plan.group_by else set()
    limited: dict[str, set[str]] = {}
    for column, value in _equality_values(plan):
        if isinstance(value, str) and column not in grouped:
            limited.setdefault(column, set()).add(value)
    asked: dict[str, set[str] | None] = {**limited, **dict.fromkeys(grouped)}
    columns = set(layer.get_base_view().columns)
    found = []
    for column, values in asked.items():
        if column not in columns or not _is_string(layer, column):
            continue
        levels, counts = value_counts(layer, column)
        frequency = {v: int(n) for v, n in zip(levels.to_list(), counts.to_list(), strict=True) if v is not None}
        for group in level_variants(list(frequency)):
            if (column, group) in confirmed or (values is not None and not values & set(group)):
                continue
            canonical, sentence = variant_sentence(column, group, frequency)
            named = tuple(spelling for spelling in group if frequency.get(spelling, 0) >= MIN_CELL_SIZE)
            mapping = LevelMapping(
                column=column, spellings=group, canonical=canonical, named=named, dataset_version=dataset_version
            )
            found.append((column, mapping, sentence))
    return found


def varies_within_patient(layer: Any, key: str, column: str, nulls_are_values: bool) -> bool:
    """Whether some patient's rows hold more than one value of ``column`` (a DuckDB aggregate).
    ``nulls_are_values``: a missing value at one visit counts as a different value."""
    view = layer.get_base_view()
    keyed = view.filter(view[key].notnull())
    distinct = keyed[column].nunique()
    if nulls_are_values:
        distinct = distinct + keyed[column].isnull().cast("int64").max()
    per_patient = keyed.group_by(key).aggregate(d=distinct)
    return int(per_patient.filter(per_patient.d > 1).count().execute()) > 0


def varies_refusal(column: str, counted: str) -> str:
    """The one sentence for a patient count that a value varying across visits would make wrong."""
    name = short_name(column)
    return (
        f"{name} differs between rows of the same patient, so patients cannot be counted once per "
        f"{name} {counted}. Ask for the number of visits by {name}, or count patients without it."
    )


def grouped_count_refusal(layer: Any, key: str, group: str) -> str | None:
    """At visit grain, patients are counted once per group only if each has one group value."""
    if group not in layer.get_base_view().columns:
        return None
    return varies_refusal(group, "group") if varies_within_patient(layer, key, group, nulls_are_values=True) else None


def share_refusal(layer: Any, key: str, plan: QueryPlan) -> str | None:
    """At visit grain a share counts patients (distinct key per level), which is right only when
    the outcome (and a rate's group) is one value per patient: died is a patient-level event, so
    "25% of visits died" is never an answer."""
    columns = set(layer.get_base_view().columns)
    outcome = plan.proportion_column or plan.rate_column
    if outcome and outcome in columns and varies_within_patient(layer, key, outcome, nulls_are_values=False):
        return varies_refusal(outcome, "value")
    if plan.rate_column and plan.group_by and plan.group_by in columns:
        return grouped_count_refusal(layer, key, plan.group_by)
    return None


def missingness_notes(layer: Any, columns: list[str], already: set[str]) -> list[str]:
    """A note for each plan column whose share of missing values is above
    ``MISSINGNESS_WARN_PERCENT``, measured now, so any column a question
    uses is flagged, whatever the upload named as analysis variables (a COUNT per plan column,
    in DuckDB). The share is suppressed
    with the rest of the line: a column with 1 to 10 missing (or present) rows reads "some"."""
    view = layer.get_base_view()
    known = [c for c in columns if c in view.columns and short_name(c) not in already]
    rows = int(layer._base_count(None))
    if not known or rows == 0:
        return []
    notes = []
    for column in known:
        missing = rows - int(view[column].count().execute() or 0)
        if missing * 100 <= MISSINGNESS_WARN_PERCENT * rows:
            continue
        amount = line_count_text(missing, [rows - missing])
        share = f"{100 * missing / rows:.1f}%" if amount != "some" else "some"
        notes.append(f"Data check: {short_name(column)} has {share} of its values missing.")
    return notes


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


def flagged_columns(report: QualityReport | None) -> set[str]:
    """Columns whose missingness the report already flags (so it is not noted twice)."""
    if report is None:
        return set()
    return {
        str(c.column)
        for c in report.checks
        if c.metric == "nullValues" and c.status in ("failed", "warning", "review") and c.column
    }


# --- Mixed units (DQ M3) ----------------------------------------------------------------------


def _range_limited(plan: QueryPlan, column: str, values: pl.Series, counts: pl.Series) -> tuple[pl.Series, pl.Series]:
    """The distinct values (and their counts) that the plan's own range limits on ``column``
    keep: "average glucose above 30" reads one unit's cluster only."""
    frame = pl.DataFrame({"v": values, "n": counts})
    for spec in plan.filters:
        method = _RANGE_OPERATORS.get(spec.operator)
        if spec.column != column or method is None or isinstance(spec.value, bool | str | list):
            continue
        frame = frame.filter(getattr(pl.col("v"), method)(spec.value))
    return frame.get_column("v").alias(column), frame.get_column("n")


def unit_mix_in_layer(layer: Any, column: str, plan: QueryPlan) -> UnitMix | None:
    """``column`` read as a mix of its two units, or None. The statistic's own column is read as
    the plan limits it, so a question about one unit ("glucose above 30") is answered; any
    other use (a limit, a group, a share) reads the whole column. Distinct values only (the
    layer's shared value counts), never the column."""
    view = layer.get_base_view()
    if unit_pair_for(column) is None or column not in view.columns:
        return None
    dtype = view[column].type()
    if not (dtype.is_string() or (dtype.is_numeric() and not dtype.is_boolean())):
        return None
    values, counts = value_counts(layer, column)
    if column == plan.metric and plan.intent in _STATISTIC_INTENTS and not dtype.is_string():
        values, counts = _range_limited(plan, column, values, counts)
    return detect_unit_mix(column, values, counts)


def _side_text(side: str, cutoff: float) -> str:
    return f"below {cutoff:g}" if side == "below" else f"of {cutoff:g} and above"


def unit_mix_sentence(mix: UnitMix, rule: UnitRule | None) -> str:
    """The one sentence for a column that mixes two units. With a cut point that separates the
    clusters (``rule``) confirming converts the other unit's values; without one it cannot,
    and the doctor corrects the file. No value and no count is named."""
    name = short_name(mix.column)
    mixed = f"{name} looks like it mixes {mix.target} and {mix.other}."
    if rule is None:
        return f"{mixed} Correct the file so each value is in one unit, or ask about one unit."
    side = _side_text(rule.side, rule.cutoff)
    convert = f"Confirm that values {side} are {rule.source_unit} to convert them to {rule.target_unit}"
    return f"{mixed} {convert}, or ask about one unit."


def unit_mix_findings(
    plan: QueryPlan, layer: Any, dataset_version: str | None, predictors: list[str]
) -> list[tuple[UnitMix, UnitRule | None, str]]:
    """(mix, rule the doctor may confirm, sentence) for each plan column (and predictor) that
    mixes two units. A rule is offered only on a numeric column whose clusters do not overlap."""
    found = []
    for column in dict.fromkeys([*plan_columns(plan), *predictors]):
        mix = unit_mix_in_layer(layer, column, plan)
        if mix is None:
            continue
        numeric = not _is_string(layer, column)
        rule = mix.rule(dataset_version) if numeric else None
        found.append((mix, rule, unit_mix_sentence(mix, rule)))
    return found


def unit_rule_text(rule: UnitRule) -> str:
    """'glucose values below 30 read as mmol/L and converted to mg/dL (1 mmol/L = 18.016 mg/dL)'."""
    side = _side_text(rule.side, rule.cutoff)
    return (
        f"{short_name(rule.column)} values {side} read as {rule.source_unit} and converted to "
        f"{rule.target_unit} ({rule.relation})"
    )


def applied_unit_rules(plan: QueryPlan, rules: tuple[UnitRule, ...], predictors: list[str]) -> list[UnitRule]:
    """The confirmed unit rules on the columns this plan reads."""
    columns = {*plan_columns(plan), *predictors}
    return [rule for rule in rules if rule.column in columns]


def with_unit_rules(plan_line: str, rules: list[UnitRule]) -> str:
    """The plan line with each applied unit conversion named, so the line says what ran."""
    if not rules:
        return plan_line
    return f"{plan_line} · units {'; '.join(unit_rule_text(rule) for rule in rules)}"

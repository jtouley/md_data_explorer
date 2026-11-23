# SPDX-License-Identifier: MIT
"""Data checks on the columns a question uses, run before the answer is computed.

Each check ends in one of three ways, and never in a silent pass:

* refuse: the answer would be wrong whatever the doctor says (a mean over values recorded as
  "<5"; a column with no values; a group count or share where a patient has two values);
* hold: the answer depends on something only the doctor can say (are rows visits? are "Male"
  and "M" one level? are ages of 400 entry errors?), and runs once they confirm;
* note: a sentence added to the answer (implausible values the doctor chose to include, a
  check of the dataset's latest quality report on a column the question uses).

Every check is a DuckDB aggregate over the layer's base view (counts, a median, value counts of
one column): no column is copied into Python per question. Row and patient counts reuse the
layer's cached base counts.

A note added to an answer carries its count apart from its text
(``result_presentation.counted_note``), which writes it as a number or a percent. Every count
is shown as it is, small ones included (ADR015).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import structlog

from clinical_analytics.analysis.grouping import level_variants, without_listed_codes
from clinical_analytics.analysis.numeric_text import all_missing_message, statistic_refusal
from clinical_analytics.analysis.variable_types import short_name
from clinical_analytics.contracts.builder import Implausible, find_plausibility_range, implausible_values, range_for
from clinical_analytics.contracts.builder import range_text as plausibility_text
from clinical_analytics.contracts.config import MISSINGNESS_WARN_PERCENT
from clinical_analytics.contracts.declarations import Declarations, GrainDeclaration, LevelMapping
from clinical_analytics.contracts.quality import RANGE_METRIC, QualityCheckResult, QualityReport
from clinical_analytics.contracts.semantics import layer_semantics
from clinical_analytics.core.counting_unit import rollup_choice
from clinical_analytics.core.identifier_rules import PATIENT_ROWS_ASKABLE_CONFIG_FLAG, ROW_KEY_CONFIG_FLAG
from clinical_analytics.core.plan_values import plan_values
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.result_presentation import AMOUNT, DataNote, counted_note, text_note

logger = structlog.get_logger(__name__)

_STATISTIC_INTENTS = frozenset({"DESCRIBE", "COMPARE_GROUPS"})
# Report checks an answer already handles itself (hold or disclosure), so not repeated as notes.
_HANDLED_METRICS = frozenset({RANGE_METRIC, "duplicateValues", "censoredValues", "levelVariants"})


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
    notes: list[DataNote] = field(default_factory=list)
    grain_to_declare: GrainDeclaration | None = None
    levels_to_declare: list[LevelMapping] = field(default_factory=list)
    keyed: bool = False  # the dataset has a patient key, so every n can say what it counts
    visits: bool = False  # rows are declared visits (on this dataset version)
    key: str | None = None  # the patient key
    rows_are_patients: bool = False  # every row has an ID and none repeats, or the user confirmed it
    encounter_key: str | None = None  # the encounter ID the question counts (admissions, tests)
    unit_noun: str | None = None  # one counted encounter as the question says it ("admission")


def patient_key(layer: Any) -> str | None:
    """The base view's patient key: the contract semantics' patient key (ADR017) for an uploaded
    dataset; for a built-in dataset configured by hand, the column its config maps to
    ``patient_id``. Never chosen by name: no key means n counts rows."""
    config = getattr(layer, "config", None) or {}
    if config.get(ROW_KEY_CONFIG_FLAG):
        return None  # no column names a patient: n counts rows
    columns = list(layer.get_base_view().columns)
    semantics = layer_semantics(layer)
    if semantics is not None:
        key = semantics.patient_key_column
        return key if key in columns else None
    mapping = config.get("column_mapping") or {}
    if "patient_id" in mapping.values() and "patient_id" in columns:
        return "patient_id"
    return None


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


def missing_key_note(grain: Grain, visits: bool, encounter: bool = False) -> DataNote | None:
    """The note for rows with no patient ID, or None. Its count's line is the dataset's rows
    (the rows with an ID are the rest). At visit grain those rows are in no patient count; in an
    encounter count their encounters are counted."""
    if grain.key is None or grain.missing_keys == 0:
        return None
    if encounter:
        counted = "; their encounters are counted"
    else:
        counted = "; those rows are not counted as patients" if visits else ""
    key = grain.key.replace(AMOUNT, "")
    return counted_note(
        f"Data check: {key} is missing on {AMOUNT} rows{counted}.",
        grain.missing_keys,
        grain.rows,
    )


def declared_visits(grain: Grain, declarations: Declarations, dataset_version: str | None) -> bool:
    """Whether the stored grain declaration was made on this dataset version and still matches
    its counts (a corrected re-upload of the same shape is a new version: asked again)."""
    stored = declarations.grain_for(dataset_version)
    return stored is not None and stored == grain.declaration(dataset_version)


def declared_patient_rows(layer: Any, grain: Grain, declarations: Declarations, dataset_version: str | None) -> bool:
    """Whether the user confirmed one row per patient on this dataset version, where no column
    identifies a patient (the upload's "Is each row one patient?", or governed SQL's
    confirmation; ``core.patient_rows``). Only for the row count it was confirmed on, and only on
    a dataset the upload may ask about (``PATIENT_ROWS_ASKABLE_CONFIG_FLAG``): where an ID-like
    column repeats, no confirmation makes rows patients."""
    config = getattr(layer, "config", None) or {}
    stored = declarations.grain_for(dataset_version)
    return (
        config.get(PATIENT_ROWS_ASKABLE_CONFIG_FLAG) is True
        and grain.key is None
        and stored is not None
        and stored.grain == "patient"
        and stored.key is None
        and stored.rows == grain.rows
    )


def grain_hold_reason(key: str, encounter: str | None = None, nouns: tuple[str, ...] = ()) -> str:
    """The hold for repeated patient IDs; with an encounter ID it also offers the encounter count,
    naming only the nouns that column can count (``counting_unit.nouns_for``)."""
    asked = " or ".join(nouns)
    choice = f", or ask for {asked} to count each {encounter} once" if encounter is not None and nouns else ""
    return (
        f"Some IDs in {key} appear on more than one row, so a row may be a visit rather than a patient. "
        f"Confirm that rows are visits to count each patient once{choice}."
    )


def missing_encounter_note(layer: Any, encounter: str) -> DataNote | None:
    """The note for rows with no encounter ID when encounters are counted: they are not counted.
    Its count's line is the dataset's rows."""
    missing = int(layer.base_missing_count(encounter))
    if missing == 0:
        return None
    rows = int(layer._base_count(None))
    name = encounter.replace(AMOUNT, "")
    return counted_note(
        f"Data check: {name} is missing on {AMOUNT} rows; those rows are not counted.",
        missing,
        rows,
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
    return Implausible(
        column=column,
        count=count,
        present=int(present),
        bounds=plausibility_text(bounds),
        minimum=float(bounds.minimum),
        maximum=None if bounds.maximum is None else float(bounds.maximum),
    )


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


def implausible_sentences(plan: QueryPlan, layer: Any) -> list[tuple[str, str, DataNote]]:
    """(column, hold sentence, note) for each plan column with implausible values; the hold and
    the note name how many values are outside the range."""
    found: list[tuple[str, str, DataNote]] = []
    for column in plan_columns(plan):
        result = implausible_in_layer(layer, column)
        if result is None:
            continue
        name = short_name(column)
        hold = (
            f"{name} has {result.count} values outside {result.bounds}, which are likely entry errors. "
            f"Correct them in the file, or confirm to include them in the answer."
        )
        note = counted_note(
            f"{name.replace(AMOUNT, '')} has {AMOUNT} values outside {result.bounds.replace(AMOUNT, '')} "
            f"(likely entry errors), included as recorded.",
            result.count,
            result.present,
        )
        found.append((column, hold, note))
    return found


def variant_sentence(column: str, group: tuple[str, ...], frequency: dict[str, int]) -> tuple[str, str]:
    """(canonical spelling, hold sentence) for spellings that may be one category: every spelling
    is named, and the most frequent one is the canonical (ties: the later in text order)."""
    canonical = max(group, key=lambda spelling: (frequency.get(spelling, 0), spelling))
    name = short_name(column)
    return canonical, (
        f"{name} has values that may be one category written differently ({', '.join(group)}). "
        f"Confirm to count them together as {canonical}."
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
        semantics = layer_semantics(layer)
        column_facts = semantics.column(column) if semantics is not None else None
        listed = column_facts.labels if column_facts is not None else {}
        for group in without_listed_codes(level_variants(list(frequency)), listed):
            if (column, group) in confirmed or (values is not None and not values & set(group)):
                continue
            canonical, sentence = variant_sentence(column, group, frequency)
            mapping = LevelMapping(
                column=column, spellings=group, canonical=canonical, named=group, dataset_version=dataset_version
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


def limits_varying(layer: Any, key: str, plan: QueryPlan) -> list[str]:
    """The plan's limit columns (filters and cohort steps) whose value differs between rows of
    one unit (distinct ``key``), in plan order: a limit on them picks rows, not units."""
    columns = set(layer.get_base_view().columns)
    limited: list[str] = []
    for column in [*(f.column for f in plan.filters), *(s.column for s in plan.cohort_steps)]:
        if column in columns and column != key and column not in limited:
            limited.append(column)
    return [c for c in limited if varies_within_patient(layer, key, c, nulls_are_values=True)]


def share_refusal(layer: Any, key: str, plan: QueryPlan, unit: str = "patient") -> str | None:
    """At visit grain a share counts units (distinct ``key``: patients, or the encounters the
    question asked for), each rolled up over its rows. When the outcome differs between a unit's
    rows, the roll-up must come from the question ("ever", "always": ``plan.rollup``); without
    one the question is refused with that plain choice, never guessed: "25% of visits died" and
    "25% of patients ever screened" are different answers.

    A grouping that differs between a unit's rows is answered: the unit is counted in each
    group it has rows in, and the presentation says so (``result_presentation._overlap_sentence``).
    """
    columns = set(layer.get_base_view().columns)
    outcome = plan.proportion_column or plan.rate_column
    if (
        plan.rollup is None
        and outcome
        and outcome in columns
        and varies_within_patient(layer, key, outcome, nulls_are_values=False)
    ):
        return rollup_choice(outcome, unit)
    return None


def missingness_notes(layer: Any, columns: list[str], already: set[str]) -> list[DataNote]:
    """A note for each plan column whose share of missing values is above
    ``MISSINGNESS_WARN_PERCENT``, measured now, so any column a question
    uses is flagged, whatever the upload named as analysis variables (a COUNT per plan column,
    in DuckDB), written as a percent of the rows."""
    view = layer.get_base_view()
    known = [c for c in columns if c in view.columns and short_name(c) not in already]
    rows = int(layer._base_count(None))
    if not known or rows == 0:
        return []
    notes: list[DataNote] = []
    for column in known:
        missing = rows - int(view[column].count().execute() or 0)
        if missing * 100 <= MISSINGNESS_WARN_PERCENT * rows:
            continue
        name = short_name(column).replace(AMOUNT, "")
        notes.append(
            counted_note(
                f"Data check: {name} has {AMOUNT} of its values missing.",
                missing,
                rows,
                "percent",
            )
        )
    return notes


def report_notes(report: QualityReport | None, columns: list[str]) -> list[DataNote]:
    """Failed or flagged checks of the dataset's latest quality report on the plan's columns,
    and failed table-level checks, as notes (``_report_note``)."""
    if report is None:
        return []
    names = {short_name(column) for column in columns} | set(columns)
    notes: list[DataNote] = []
    for check in report.checks:
        if check.status not in ("failed", "warning", "review") or check.metric in _HANDLED_METRICS:
            continue
        if (check.column is None and check.status == "failed") or check.column in names:
            note = _report_note(check, report.row_count)
            if note is not None:
                notes.append(note)
    return notes


def _report_note(check: QualityCheckResult, rows: int | str) -> DataNote | None:
    """One report check as a note, its count carried apart from the text. A check with no count
    is shown as written. A row-count check names no count (the app shows the row count). A count
    whose place in the message cannot be found is not shown at all (logged)."""
    if check.metric == "rowCount":
        return text_note("Data check: the number of rows does not meet the dataset's contract.")
    if not isinstance(check.count, int) or isinstance(check.count, bool) or check.count == 0:
        return text_note(f"Data check: {check.message}.".replace(AMOUNT, ""))  # no count
    total = rows if isinstance(rows, int) and rows > 0 else None
    head = f"{check.column or 'Table'}: "
    if not check.message.startswith(head):
        return None
    body = check.message[len(head) :]
    head = head.replace(AMOUNT, "")
    if check.percent is not None and body.startswith(f"{check.percent:.1f}% ") and total is not None:
        rest = body[len(f"{check.percent:.1f}%") :].replace(AMOUNT, "")
        return counted_note(f"Data check: {head}{AMOUNT}{rest}.", check.count, total, "percent")
    if body.startswith(f"{check.count} "):
        rest = body[len(str(check.count)) :].replace(AMOUNT, "")
        return counted_note(f"Data check: {head}{AMOUNT}{rest}.", check.count, total)
    logger.warning("report_note_count_not_found", check_id=check.id, metric=check.metric)
    return None


def flagged_columns(report: QualityReport | None) -> set[str]:
    """Columns whose missingness the report already flags (so it is not noted twice)."""
    if report is None:
        return set()
    return {
        str(c.column)
        for c in report.checks
        if c.metric == "nullValues" and c.status in ("failed", "warning", "review") and c.column
    }

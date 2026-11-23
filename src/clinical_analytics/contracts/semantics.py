# SPDX-License-Identifier: MIT
"""What an uploaded file means: the contract's semantics section, and the one way to read it.

The semantics section (ADR017) is written once per dataset version, inside the version's data
contract (``contracts/<upload id>/<version>.odcs.yaml``, top-level custom property
``semantics``). It says what one row is, who the patient is, which columns number a patient's
rows or group them, and for each column its type, date format, missing-value codes, stated range,
unit, code labels and timepoint. Every fact names its source: inferred from the file's
structure, read from the data dictionary, declared by the user, or a default.

Every part of the app that needs one of these facts reads it here, with ``semantics_for`` (or
``layer_semantics`` for a semantic layer built from it), and from nothing else. What the user
confirms later (``contracts.declarations``) overrides what was inferred, without rewriting the
contract: ``semantics_for`` applies the version's declarations on read.

Stable API (also read by governed SQL and the model-written query path):

* ``semantics_for(data_dir, dataset_id, version) -> DatasetSemantics``
* ``semantics_from_contract(contract) -> DatasetSemantics | None``
* ``with_declarations(semantics, declarations) -> DatasetSemantics``
* ``DatasetSemantics.patient_key_column``, ``.one_row_per_patient``, ``.column(name)``,
  ``.question(kind)``, ``.missing_codes()``
* ``describe(semantics) -> list[str]``: the facts in clinical words, for the screen.

Holds no patient values: column names, counts, codes and labels from the dictionary, and date
examples built from a fixed date (3 April 2021), never from a row.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias

import pydantic
import structlog
from open_data_contract_standard.model import OpenDataContractStandard
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.contracts import config
from clinical_analytics.contracts.errors import ContractError, ContractStoreError, ContractValidationError

logger = structlog.get_logger()

FactSource: TypeAlias = Literal["structure", "dictionary", "declared", "default"]
Confidence: TypeAlias = Literal["high", "medium", "low"]
QuestionKind: TypeAlias = Literal[
    "rows_are_patients", "person_role", "date_order", "date_century", "missing_code", "repeated_strata",
    "aggregate_unsure", "total_row", "censored_value",
]  # fmt: skip
AggregateKind: TypeAlias = Literal["frequency", "estimate"]
LogicalType: TypeAlias = Literal[
    "string", "integer", "number", "boolean", "date", "timestamp", "time", "array", "object"
]
ColumnRole: TypeAlias = Literal["patient_key", "row_key", "sequence", "grouping", "app_added", "date", "value"]
AppColumnKind: TypeAlias = Literal["copy", "hash", "row_number"]
RowKeyKind: TypeAlias = Literal["column", "composite", "none"]
Direction: TypeAlias = Literal["lower_is_better", "higher_is_better"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Grain(_Model):
    """What one row is, as a clinical noun (``kind`` from ``GRAIN_KINDS``; ``noun`` as said)."""

    kind: str = Field(min_length=1)
    noun: str = Field(min_length=1)
    one_row_per_patient: bool
    rows: int = Field(ge=0)
    patients: int | None = Field(default=None, ge=0)
    source: FactSource
    evidence: str = ""
    # A whole sentence when a noun can't say it ("Each line counts cases for one region, year and
    # category."), for a summary table.
    sentence: str | None = None


class PatientKey(_Model):
    """The column that names the patient (the file's own column, never one the app added)."""

    column: str = Field(min_length=1)
    role: str = "patient"  # patient, infant, mother: who the key names
    unique: bool  # one row per key value
    confidence: Confidence
    source: FactSource
    evidence: str = ""


class RowKey(_Model):
    """What tells rows apart: one column, a combination, or nothing (rows are numbered)."""

    kind: RowKeyKind
    columns: tuple[str, ...] = ()
    source: FactSource = "structure"
    evidence: str = ""


class SequenceColumn(_Model):
    """A column that numbers each patient's rows (visit 1, 2, 3): order within patient, never an ID."""

    column: str
    within: str
    start: int = 1
    noun: str = "row"  # what it numbers: visit, cycle, scan, episode
    # "order": numbers each patient's lines 1..k; "time_since_index": the time since a start
    # (weeks since enrolment), different on each line of a patient.
    kind: Literal["order", "time_since_index"] = "order"
    source: FactSource
    evidence: str = ""


class GroupingColumn(_Model):
    """A column for a different role than the patient (a mother, a site, a surgeon, a ward)."""

    column: str
    role: str
    # An ID that repeats (site_id, mat_id): lines may share someone, so "Is each line one
    # patient?" is not asked. False for a staff or place attribute (a ward name).
    is_id: bool = True
    source: FactSource
    evidence: str = ""


class EpisodeKey(_Model):
    """An ID of an episode within one patient (an admission within an MRN): episodes can be counted."""

    column: str
    within: str
    noun: str = "episode"
    source: FactSource
    evidence: str = ""


class TotalLevel(_Model):
    """Values of a stratum column that mean "all" ("All ages", "United States", "Total"): a line
    holding one is a total or subtotal of the other lines, never one more group."""

    column: str
    values: tuple[str, ...]


class AggregateInfo(_Model):
    """A table whose lines are groups, not patients (ADR017 with ADR014).

    * ``frequency``: each line counts cases for one stratum; "how many" is the SUM of
      ``count_column`` over the lines that are not totals.
    * ``estimate``: rates, percentages, or counts with total rows; never summed: a question reads
      the matching line (the total row for "all").
    """

    kind: AggregateKind
    count_column: str | None = None
    value_columns: tuple[str, ...] = ()
    strata: tuple[str, ...] = ()
    total_levels: tuple[TotalLevel, ...] = ()
    repeated_strata_lines: int = 0
    source: FactSource
    evidence: str = ""

    def total_values(self, column: str) -> tuple[str, ...]:
        return next((t.values for t in self.total_levels if t.column == column), ())


class AppColumn(_Model):
    """A column the app added to the file: never a patient identifier of its own."""

    column: str
    kind: AppColumnKind
    made_from: tuple[str, ...] = ()


class DateFormat(_Model):
    """The one format a column's dates are read with, and a worked example."""

    format: str
    written: str  # the format as a person writes it ("DD/MM/YY")
    example_in: str
    example_out: str
    has_time: bool = False
    source: FactSource
    evidence: str = ""


class MissingCode(_Model):
    """A value that means "not recorded": stored as missing wherever answers read the data."""

    value: int | float | str
    label: str = "not recorded"
    source: FactSource
    evidence: str = ""


class CensoredCode(_Model):
    """A labelled value that stands for a range, not one value ("80 = 80 years or older", "666 =
    less than 1 year"): a real value, never missing; an answer that reads its column asks first."""

    value: int | float | str
    meaning: str
    source: FactSource


class StatedRange(_Model):
    minimum: float
    maximum: float
    source: FactSource


class Unit(_Model):
    value: str
    source: FactSource


class Timepoint(_Model):
    """One measure taken at several times (``weight_preop_kg`` / ``weight_12m_kg``)."""

    measure: str
    timepoint: str
    order: int
    paired_with: tuple[str, ...]
    source: FactSource


class ColumnSemantics(_Model):
    logical_type: str
    role: ColumnRole = "value"
    date: DateFormat | None = None
    missing_codes: tuple[MissingCode, ...] = ()
    stated_range: StatedRange | None = None
    unit: Unit | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    labels_source: FactSource | None = None
    timepoint: Timepoint | None = None
    direction: Direction | None = None
    direction_source: FactSource | None = None
    description: str | None = None
    # Non-response codes of a coded item ("7 = Refused, 9 = Don't know"): kept as categories in
    # the data, left out of denominators and averages by every reader.
    nonresponse_codes: tuple[MissingCode, ...] = ()
    censored_codes: tuple[CensoredCode, ...] = ()
    # A column holding this column's unit ("dur" with "dur_unit" = months/years).
    unit_column: str | None = None


class OpenQuestion(_Model):
    """Something only a clinician can settle, asked in clinical words."""

    id: str
    kind: QuestionKind
    question: str
    column: str | None = None
    options: tuple[str, ...] = ()
    answered: bool = False
    # The value asked about (a suspected missing-value code), when there is one.
    value: int | float | str | None = None


class DatasetSemantics(_Model):
    schema_version: int = config.SEMANTICS_SCHEMA_VERSION
    dataset_version: str | None = None
    rows: int = Field(ge=0)
    grain: Grain
    patient_key: PatientKey | None = None
    row_key: RowKey = RowKey(kind="none")
    sequence_columns: tuple[SequenceColumn, ...] = ()
    grouping_columns: tuple[GroupingColumn, ...] = ()
    episode_keys: tuple[EpisodeKey, ...] = ()
    # A summary table's count column (the cases each line counts), or None.
    case_count_column: str | None = None
    # Lines are groups (a frequency or estimate table), with its count column and total rows.
    aggregate: AggregateInfo | None = None
    app_columns: tuple[AppColumn, ...] = ()
    columns: dict[str, ColumnSemantics] = Field(default_factory=dict)
    open_questions: tuple[OpenQuestion, ...] = ()
    # Upload facts that decide what may be asked: a declared table of public estimates, or a
    # published export (CDC WONDER) has rows of groups, never patients.
    population_estimates: bool = False
    source_format: str = ""
    # A survey sample: counts are of this file's respondents, never weighted national estimates.
    survey_sample: bool = False
    survey_weight_columns: tuple[str, ...] = ()

    @property
    def patient_key_column(self) -> str | None:
        return self.patient_key.column if self.patient_key else None

    @property
    def one_row_per_patient(self) -> bool:
        return self.grain.one_row_per_patient

    def column(self, name: str) -> ColumnSemantics | None:
        return self.columns.get(name)

    def question(self, kind: QuestionKind) -> OpenQuestion | None:
        return next((q for q in self.open_questions if q.kind == kind), None)

    def missing_codes(self) -> dict[str, tuple[int | float | str, ...]]:
        """column -> the values stored as missing in that column."""
        return {
            name: tuple(m.value for m in col.missing_codes) for name, col in self.columns.items() if col.missing_codes
        }

    def app_column(self, name: str) -> AppColumn | None:
        return next((a for a in self.app_columns if a.column == name), None)


class SemanticsMissingError(ContractError):
    """The version's contract was written before semantics existed (or has none)."""


def semantics_from_contract(contract: OpenDataContractStandard) -> DatasetSemantics | None:
    """The contract's semantics section, validated, or None when the contract has none.

    Raises:
        ContractValidationError: the section is there but is not valid semantics.
    """
    for prop in contract.customProperties or []:
        if prop.property == config.SEMANTICS_PROPERTY:
            try:
                return DatasetSemantics.model_validate(prop.value)
            except pydantic.ValidationError as e:
                raise ContractValidationError(f"Contract semantics are invalid ({e.error_count()} errors)") from e
    return None


def semantics_property(semantics: DatasetSemantics) -> dict[str, Any]:
    """The contract custom property that stores ``semantics``."""
    return {"property": config.SEMANTICS_PROPERTY, "value": semantics.model_dump(mode="json")}


def with_declarations(semantics: DatasetSemantics, declarations: Any) -> DatasetSemantics:
    """``semantics`` with what the user confirmed on its version (``Declarations``) applied.

    * A one-row-per-patient grain on the version's row count: rows are patients; the row key
      column (if one) becomes the patient key; "Is each line one patient?" is answered.
    * A visit grain naming a key: that key is the patient key, rows are visits of it.
    """
    if declarations is None:
        return semantics
    semantics = _with_missing_answers(semantics, declarations.missing_codes_for(semantics.dataset_version))
    semantics = _with_table_kind(semantics, declarations)
    semantics = _with_date_readings(semantics, declarations.date_readings_for(semantics.dataset_version))
    semantics = _with_answers(semantics, declarations.answers_for(semantics.dataset_version))
    grain = declarations.grain_for(semantics.dataset_version)
    if grain is None:
        return semantics
    rows = semantics.rows
    # A grain answers both "Is each line one patient?" and "Does each line count cases?".
    questions = tuple(
        q.model_copy(update={"answered": True}) if q.kind in _TABLE_QUESTIONS else q for q in semantics.open_questions
    )
    if grain.grain == "patient" and grain.rows == rows:
        key = semantics.patient_key
        if key is None and semantics.row_key.kind == "column" and len(semantics.row_key.columns) == 1:
            key = PatientKey(
                column=semantics.row_key.columns[0],
                unique=True,
                confidence="high",
                source="declared",
                evidence="you confirmed that each line is one patient",
            )
        return semantics.model_copy(
            update={
                "patient_key": key,
                "grain": semantics.grain.model_copy(
                    update={
                        "kind": "patient",
                        "noun": "patient",
                        "one_row_per_patient": True,
                        "patients": rows,
                        "source": "declared",
                        "evidence": "you confirmed that each line is one patient",
                    }
                ),
                "open_questions": questions,
            }
        )
    if grain.grain == "visit" and grain.key and grain.key in semantics.columns:
        current = semantics.patient_key
        key = (
            current
            if current is not None and current.column == grain.key
            else PatientKey(column=grain.key, unique=False, confidence="high", source="declared")
        )
        key = key.model_copy(update={"source": "declared", "evidence": "you confirmed that lines are visits"})
        return semantics.model_copy(
            update={
                "patient_key": key,
                "grain": semantics.grain.model_copy(
                    update={"one_row_per_patient": False, "patients": grain.patients, "source": "declared"}
                ),
                "open_questions": questions,
            }
        )
    return semantics


def _with_answers(semantics: DatasetSemantics, answers: Any) -> DatasetSemantics:
    """Answers to questions no other declaration holds (``QuestionAnswer``), each one of its
    question's options; the first option is "yes".

    * "Is each line one patient?" (either answer): settled. "No" also says the lines are records,
      not groups: the table reading the structure suggested is dropped.
    * "Is Statewide a total?": yes adds it to the table's total rows.
    * "Should repeated lines be added together?": yes settles it; no leaves the table held.
    """
    if not answers:
        return semantics
    given = {a.question_id: a.answer for a in answers}
    aggregate = semantics.aggregate
    records = given.get(config.ROWS_QUESTION_ID) == config.ROWS_ANSWER_RECORDS
    questions = []
    for q in semantics.open_questions:
        answer = given.get(q.id)
        settled = answer is not None and answer in q.options
        if q.kind == "aggregate_unsure" and records:
            settled = True
        if q.kind == "repeated_strata" and settled and answer != q.options[0]:
            settled = False  # lines that replace each other: no sum can be given
        if q.kind == "total_row" and settled and answer == q.options[0] and aggregate is not None and q.column:
            aggregate = _with_total(aggregate, q.column, str(q.value))
        questions.append(q.model_copy(update={"answered": True}) if settled else q)
    update: dict[str, Any] = {"open_questions": tuple(questions), "aggregate": aggregate}
    if records and aggregate is not None and aggregate.source != "declared":
        update.update({"aggregate": None, "case_count_column": None})
    return semantics.model_copy(update=update)


def _with_total(aggregate: AggregateInfo, column: str, value: str) -> AggregateInfo:
    levels = {t.column: t.values for t in aggregate.total_levels}
    levels[column] = tuple(sorted({*levels.get(column, ()), value}))
    totals = tuple(TotalLevel(column=c, values=v) for c, v in levels.items())
    return aggregate.model_copy(update={"total_levels": totals})


def _same_value(value: object) -> str:
    """999, 999.0 and "999" are one value."""
    try:
        return f"{float(str(value)):g}"
    except ValueError:
        return str(value)


def _with_missing_answers(semantics: DatasetSemantics, answers: Any) -> DatasetSemantics:
    """Confirmed "not recorded" values become declared missing codes; every answered question is
    marked answered (a "no" keeps the value as a real value)."""
    if not answers:
        return semantics
    by_key = {(a.column, _same_value(a.value)): a for a in answers}
    columns = dict(semantics.columns)
    for answer in answers:
        column = columns.get(answer.column)
        if column is None or not answer.missing:
            continue
        if all(_same_value(m.value) != _same_value(answer.value) for m in column.missing_codes):
            code = MissingCode(value=answer.value, source="declared", evidence="you confirmed it means not recorded")
            columns[answer.column] = column.model_copy(update={"missing_codes": (*column.missing_codes, code)})
    questions = tuple(
        q.model_copy(update={"answered": True})
        if q.kind == "missing_code" and (q.column, _same_value(q.value)) in by_key
        else q
        for q in semantics.open_questions
    )
    return semantics.model_copy(update={"columns": columns, "open_questions": questions})


def _with_date_readings(semantics: DatasetSemantics, readings: Any) -> DatasetSemantics:
    """A date column the user settled ("day first", "the 1900s") is a date column, read with that
    format by every reader; its question is answered."""
    if not readings:
        return semantics
    columns = dict(semantics.columns)
    settled = set()
    for reading in readings:
        column = columns.get(reading.column)
        if column is None:
            continue
        settled.add(reading.column)
        century = f", years in the {reading.century}" if reading.century else ""
        fmt = DateFormat(
            format=reading.format, written=reading.format, example_in="", example_out="", source="declared",
            evidence=f"you said how to read it{century}",
        )  # fmt: skip
        columns[reading.column] = column.model_copy(update={"date": fmt, "logical_type": "date", "role": "date"})
    questions = tuple(
        q.model_copy(update={"answered": True})
        if q.kind in ("date_order", "date_century") and q.column in settled
        else q
        for q in semantics.open_questions
    )
    return semantics.model_copy(update={"columns": columns, "open_questions": questions})


_TABLE_QUESTIONS = frozenset({"rows_are_patients", "aggregate_unsure"})


def _with_table_kind(semantics: DatasetSemantics, declarations: Any) -> DatasetSemantics:
    """A declared table of groups: "Each row counts cases for a group (in column …)", or public
    population estimates (ADR014). Lines are then groups, never patients."""
    version = semantics.dataset_version
    declared = declarations.aggregate_for(version)
    estimates = declarations.population_estimates_for(version) is not None
    if declared is None and not estimates:
        return semantics
    answered = tuple(
        q.model_copy(update={"answered": True}) if q.kind in _TABLE_QUESTIONS else q for q in semantics.open_questions
    )
    if estimates:
        return semantics.model_copy(
            update={
                "population_estimates": True,
                "patient_key": None,
                "grain": semantics.grain.model_copy(
                    update={"kind": "estimate", "noun": "published estimate", "one_row_per_patient": False,
                            "patients": None, "source": "declared", "sentence": None,
                            "evidence": "you declared public population estimates"}
                ),
                "open_questions": answered,
            }
        )  # fmt: skip
    assert declared is not None
    current = semantics.aggregate
    aggregate = AggregateInfo(
        kind=declared.kind,
        count_column=declared.count_column,
        value_columns=current.value_columns if current else (),
        strata=current.strata if current else tuple(c for c in semantics.columns if c != declared.count_column),
        total_levels=current.total_levels if current else (),
        repeated_strata_lines=current.repeated_strata_lines if current else 0,
        source="declared",
        evidence=f"you said each line counts cases for a group, in {declared.count_column}",
    )
    return semantics.model_copy(
        update={
            "patient_key": None,
            "aggregate": aggregate,
            "case_count_column": declared.count_column,
            "grain": semantics.grain.model_copy(
                update={"kind": "stratum", "noun": "count of cases", "one_row_per_patient": False, "patients": None,
                        "source": "declared",
                        "sentence": f"Each line counts cases for a group, in {declared.count_column}.",
                        "evidence": aggregate.evidence}
            ),
            "open_questions": answered,
        }
    )  # fmt: skip


def semantics_for(data_dir: Path, dataset_id: str, version: str, *, declarations: Any = None) -> DatasetSemantics:
    """The semantics of ``dataset_id`` at ``version``, with the user's declarations applied.

    ``declarations``: a ``DeclarationStore`` (default: the file store in ``data_dir``).

    Raises:
        SemanticsMissingError: the version has no contract, or its contract has no semantics.
        ContractDamagedError / ContractValidationError: the contract can't be read as written.
        ContractStoreError: the contract or the declarations can't be read.
    """
    from clinical_analytics.contracts.declarations import FileDeclarationStore
    from clinical_analytics.contracts.service import read_saved_contract
    from clinical_analytics.contracts.store import contract_path

    path = contract_path(data_dir, dataset_id, version)
    if not path.is_file():
        raise SemanticsMissingError(f"No contract for version {version}")
    semantics = semantics_from_contract(read_saved_contract(path))
    if semantics is None:
        raise SemanticsMissingError(f"The contract for version {version} has no semantics")
    store = declarations if declarations is not None else FileDeclarationStore(data_dir)
    try:
        declared = store.load(dataset_id)
    except ContractStoreError:
        logger.error("semantics_declarations_unreadable", dataset_id=dataset_id, dataset_version=version)
        raise
    return with_declarations(semantics.model_copy(update={"dataset_version": version}), declared)


# Semantic-layer config key that carries the version's semantics (``DatasetSemantics`` as JSON).
LAYER_CONFIG_KEY = "semantics"
# The same semantics with the declarations the layer was built with: what every hold reads, so an
# answer stored while a question runs never releases a hold on a layer built before it.
LAYER_DECLARED_KEY = "declared_semantics"


def layer_semantics(layer: Any, *, declared: bool = False) -> DatasetSemantics | None:
    """The semantics a semantic layer was built with (from ``semantics_for``), or None for a layer
    not built from an uploaded dataset's contract (a built-in dataset). ``declared``: with the
    declarations the layer was built with (``LAYER_DECLARED_KEY``)."""
    config_dict = getattr(layer, "config", None)
    if not isinstance(config_dict, Mapping):
        return None  # a stand-in layer (a test double) or no config: no semantics
    raw = config_dict.get(LAYER_DECLARED_KEY if declared else LAYER_CONFIG_KEY)
    if raw is None:
        return None
    if isinstance(raw, DatasetSemantics):
        return raw
    try:
        return DatasetSemantics.model_validate(raw)
    except pydantic.ValidationError as e:
        raise ContractValidationError(f"Layer semantics are invalid ({e.error_count()} errors)") from e


# ----------------------------------------------------------------------------- on screen
def _count(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def _plural(noun: str) -> str:
    """'visit' -> 'visits', 'pregnancy' -> 'pregnancies'."""
    if noun.endswith("y") and noun[-2:-1] not in ("a", "e", "i", "o", "u"):
        return noun[:-1] + "ies"
    return noun if noun.endswith("s") else noun + "s"


def _list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def describe(semantics: DatasetSemantics, *, max_lines: int = 12) -> list[str]:
    """The key facts in clinical words ("Each line is one ED encounter; 700 patients")."""
    lines: list[str] = []
    grain, key = semantics.grain, semantics.patient_key
    if key is not None and grain.patients is not None:
        if grain.one_row_per_patient:
            who = key.role if key.role in ("infant", "mother") else "patient"
            lines.append(
                f"Each line is one {grain.noun}; {_count(grain.patients, who)}, each named by {key.column}."
                if grain.kind != "patient"
                else f"Each line is one patient; {_count(grain.patients, 'patient')}, each named by {key.column}."
            )
        else:
            lines.append(
                f"Each line is one {grain.noun}; {_count(grain.patients, 'patient')} "
                f"({key.column} names the patient) over {_count(grain.rows, 'line')}."
            )
    elif grain.sentence:
        lines.append(f"{grain.sentence[:-1]} ({_count(grain.rows, 'line')}).")
    elif semantics.population_estimates or grain.kind == "estimate":
        lines.append(f"Each line is one published estimate for a group; {_count(grain.rows, 'line')}.")
    else:
        rows_q = semantics.question("rows_are_patients")
        told = (
            f" Lines are told apart by {_list(list(semantics.row_key.columns))}." if semantics.row_key.columns else ""
        )
        lines.append(
            f"No column names a patient, so counts are of lines ({_count(grain.rows, 'line')}).{told}"
            + (" You can confirm that each line is one patient." if rows_q is not None and not rows_q.answered else "")
        )
    for seq in semantics.sequence_columns:
        if seq.kind == "time_since_index":
            lines.append(f"{seq.column} is the time of each line ({seq.noun}), different on each line of a patient.")
            continue
        start = "1" if seq.start == 1 else str(seq.start)
        lines.append(f"{seq.column} numbers each patient's {_plural(seq.noun)} in order ({start} = first).")
    for episode in semantics.episode_keys:
        if episode.column not in semantics.row_key.columns:
            lines.append(
                f"{episode.column} names each {episode.noun} of a patient, so {_plural(episode.noun)} can be counted."
            )
    for group in semantics.grouping_columns:
        if group.is_id:
            lines.append(f"{group.column} groups lines by {group.role}; it does not name the patient.")
    for name, col in semantics.columns.items():
        if col.date is not None and col.date.format not in (
            "native",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ):
            lines.append(f"Dates like {col.date.example_in} in {name} read as {col.date.example_out}.")
    for name, col in semantics.columns.items():
        for code in col.missing_codes:
            why = "" if code.label in ("not recorded", "") else f" ({code.label})"
            shown = f"{code.value:g}" if isinstance(code.value, float) else str(code.value)
            lines.append(f"{shown} in {name} means not recorded{why}.")
    paired: dict[str, list[str]] = {}
    for name, col in semantics.columns.items():
        if col.timepoint is not None:
            paired.setdefault(col.timepoint.measure, []).append(name)
    for measure, names in paired.items():
        lines.append(f"{_list(names)} are one measure ({measure}) at {len(names)} times.")
    for name, col in semantics.columns.items():
        if col.nonresponse_codes:
            codes = ", ".join(f"{c.value} ({c.label})" for c in col.nonresponse_codes)
            lines.append(f"In {name}, {codes} are kept as answers but left out of percentages and averages.")
        if col.unit_column:
            lines.append(f"{col.unit_column} gives the unit of {name}.")
    if semantics.survey_sample:
        lines.append(config.SURVEY_NOTE)
    for q in semantics.open_questions:
        if not q.answered:
            lines.append(f"Question for you: {q.question}")
    if len(lines) > max_lines:
        lines = [*lines[: max_lines - 1], f"…and {len(lines) - max_lines + 1} more"]
    return lines

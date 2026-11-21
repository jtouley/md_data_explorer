# SPDX-License-Identifier: MIT
"""What the model is told about the table: columns, descriptions, values and the grain.

Read once per dataset version from the SQL sandbox (the same stored Parquet and contract governed
SQL reads) and from the upload's data dictionary text. Everything here stays on the machine: it is
sent to the local model only.

The grain is what a count counts: rows are patients when the patient ID never repeats (or the
user confirmed one row per patient); otherwise rows are visits and patients are counted with
``COUNT(DISTINCT <patient ID>)``, admissions or tests by their own encounter ID.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

import structlog
from sqlglot import exp

from clinical_analytics.core import counting_unit
from clinical_analytics.sql import config as sql_config
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.render import render
from clinical_analytics.sql.sandbox import SandboxConnection

logger = structlog.get_logger()

# A column with at most this many different values has them all listed for the model.
MAX_LISTED_VALUES: Final = 12
MAX_VALUE_CHARS: Final = 40
MAX_DESCRIPTION_CHARS: Final = 220
# A dictionary this short is given whole (it usually carries counting rules worth every word).
WHOLE_DICTIONARY_CHARS: Final = 2500
MAX_NOTES_CHARS: Final = 900
_ENTRY = re.compile(r"^\s{0,3}[\"'“]?([A-Za-z_][\w.]*)[\"'”]?\s*(?::|-|–|\s{2,}|\t)\s*(\S.*)?$")
_CODE_LINE = re.compile(r"^\s*[\"'“]?[\w.<>=+-]{1,24}[\"'”]?\s*[:=]\s*\S")


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    sql_type: str  # DuckDB's type name (VARCHAR, BIGINT, DOUBLE, DATE, ...)
    logical_type: str  # the contract's (string, integer, number, boolean, date, ...)
    distinct: int
    nulls: int
    values: tuple[Any, ...] | None = None  # every stored value, when there are few
    low: Any = None  # smallest and largest value of a number or date column
    high: Any = None
    description: str | None = None
    is_patient_id: bool = False
    is_encounter_id: bool = False

    @property
    def numeric(self) -> bool:
        return self.logical_type in ("integer", "number")

    @property
    def text(self) -> bool:
        return self.logical_type == "string"


@dataclass(frozen=True)
class DatasetContext:
    """The table as the model and the compiler see it."""

    table: str
    columns: tuple[ColumnInfo, ...]
    rows: int
    distinct_rows: int
    patient_key: str | None
    patients: int | None  # distinct patient IDs (None: no patient ID column)
    confirmed_patient_rows: bool = False  # the user confirmed one row per patient
    notes: str = ""  # the data dictionary's general notes (or the whole short dictionary)
    encounter_keys: tuple[str, ...] = field(default_factory=tuple)

    def column(self, name: str) -> ColumnInfo | None:
        folded = name.casefold()
        return next((c for c in self.columns if c.name.casefold() == folded), None)

    @property
    def duplicate_rows(self) -> int:
        return self.rows - self.distinct_rows

    @property
    def rows_are_patients(self) -> bool:
        """Counting rows counts patients: every row has its own patient ID, or the user said so."""
        if self.patient_key is None:
            return self.confirmed_patient_rows
        key = self.column(self.patient_key)
        return key is not None and self.patients == self.rows and key.nulls == 0

    @property
    def row_unit(self) -> str:
        """What one row is called in an answer: patients, visits or rows."""
        if self.rows_are_patients:
            return "patients"
        return "visits" if self.patient_key is not None else "rows"


def _literal(value: Any) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def dictionary_descriptions(doc: str | None, columns: list[str]) -> tuple[dict[str, str], str]:
    """({column: its description}, general notes) read from a data dictionary's plain text.

    An entry starts on a line that begins with a column's name followed by a colon, a dash or a
    gap; the lines after it (wrapped text, code lists such as ``1: yes``) belong to it until the
    next entry or a new paragraph that is not a code line. Everything else is general notes.
    """
    if not doc:
        return {}, ""
    known = {c.casefold(): c for c in columns}
    described: dict[str, list[str]] = {}
    notes: list[str] = []
    current: str | None = None
    blank_before = False
    for raw in doc.splitlines():
        line = raw.rstrip()
        if not line.strip():
            blank_before = True
            continue
        match = _ENTRY.match(line)
        name = known.get(match.group(1).casefold()) if match else None
        if name is not None and name not in described:
            current = name
            described[name] = [match.group(2) or ""] if match else []
        elif current is not None and (not blank_before or line[:1].isspace() or _CODE_LINE.match(line)):
            described[current].append(line.strip())
        else:
            current = None
            notes.append(line.strip())
        blank_before = False
    descriptions = {name: _clip(" ".join(parts), MAX_DESCRIPTION_CHARS) for name, parts in described.items()}
    return {k: v for k, v in descriptions.items() if v}, _clip(" ".join(notes), MAX_NOTES_CHARS)


def build_context(
    sandbox: SandboxConnection,
    catalog: SqlCatalog,
    *,
    doc_context: str | None = None,
    confirmed_patient_rows: bool = False,
) -> DatasetContext:
    """Profile the table in the sandbox (three aggregate statements) and attach the dictionary."""
    table = sql_config.TABLE_NAME
    types = {str(r["column_name"]): str(r["column_type"]) for r in sandbox.run_generated(f"DESCRIBE {table}").to_dicts()}
    select: list[exp.Expr] = [exp.alias_(exp.Count(this=exp.Star()), "n")]
    for i, col in enumerate(catalog.columns):
        ref = exp.column(col.name, quoted=True)
        select.append(exp.alias_(exp.Count(this=exp.Distinct(expressions=[ref.copy()])), f"d{i}"))
        select.append(exp.alias_(exp.Count(this=ref.copy()), f"k{i}"))
        if col.logical_type in ("integer", "number", "date", "timestamp"):
            select.append(exp.alias_(exp.Min(this=ref.copy()), f"lo{i}"))
            select.append(exp.alias_(exp.Max(this=ref.copy()), f"hi{i}"))
    profile = sandbox.run_generated(render(exp.select(*select).from_(table))).row(0, named=True)
    rows = int(profile["n"])
    distinct_rows = int(sandbox.run_generated(f"SELECT COUNT(*) AS n FROM (SELECT DISTINCT * FROM {table})").item())
    key = catalog.id_column.name if catalog.id_column is not None else None
    names = [c.name for c in catalog.columns]
    encounter_keys = tuple(counting_unit.encounter_key_candidates(names, key))
    listed = [
        (i, col)
        for i, col in enumerate(catalog.columns)
        if 0 < int(profile[f"d{i}"]) <= MAX_LISTED_VALUES and not col.is_id and col.name not in encounter_keys
    ]
    values: dict[str, list[Any]] = {}
    for _i, col in listed:
        ref = exp.column(col.name, quoted=True)
        query = exp.select(exp.alias_(ref, "v")).from_(table).where(ref.copy().is_(exp.null()).not_()).distinct()
        found = sandbox.run_generated(render(query.order_by("v"))).get_column("v").to_list()
        values[col.name] = found
    whole = bool(doc_context) and len(doc_context or "") <= WHOLE_DICTIONARY_CHARS
    descriptions, notes = ({}, (doc_context or "").strip()) if whole else dictionary_descriptions(doc_context, names)
    columns = []
    for i, col in enumerate(catalog.columns):
        columns.append(
            ColumnInfo(
                name=col.name,
                sql_type=types.get(col.name, "VARCHAR"),
                logical_type=col.logical_type,
                distinct=int(profile[f"d{i}"]),
                nulls=rows - int(profile[f"k{i}"]),
                values=tuple(values[col.name]) if col.name in values else None,
                low=profile.get(f"lo{i}"),
                high=profile.get(f"hi{i}"),
                description=descriptions.get(col.name),
                is_patient_id=col.name == key,
                is_encounter_id=col.name in encounter_keys,
            )
        )
    patients = next((c.distinct for c in columns if c.is_patient_id), None)
    logger.info("model_query_context_built", columns=len(columns), rows=rows, duplicate_rows=rows - distinct_rows)
    return DatasetContext(
        table=table,
        columns=tuple(columns),
        rows=rows,
        distinct_rows=distinct_rows,
        patient_key=key,
        patients=patients,
        confirmed_patient_rows=confirmed_patient_rows,
        notes=notes,
        encounter_keys=encounter_keys,
    )


def _column_line(col: ColumnInfo) -> str:
    parts = [f"- {col.name} ({col.sql_type})"]
    if col.is_patient_id:
        parts.append("patient ID")
    elif col.is_encounter_id:
        nouns = counting_unit.nouns_for(col.name)
        parts.append(f"ID of one {counting_unit.singular(nouns[0]) if nouns else 'encounter'}")
    if col.values is not None:
        shown = ", ".join(_clip(_literal(v), MAX_VALUE_CHARS) for v in col.values)
        parts.append(f"values: {shown}")
    elif col.low is not None and col.high is not None and not col.is_patient_id:
        parts.append(f"range {col.low} to {col.high}")
    if col.nulls:
        parts.append("some rows empty (NULL)")
    line = "; ".join(parts)
    return f"{line}. {col.description}" if col.description else line


def grain_text(context: DatasetContext, *, writes_sql: bool = True) -> str:
    """What a row is and how it is counted, in the model's words. ``writes_sql``: the model
    writes the COUNT itself (the sql format); otherwise the app does and the model is only told."""
    key = context.patient_key
    if context.rows_are_patients:
        if not writes_sql:
            return f"Each row is one patient ({context.rows} patients)."
        how = f'COUNT(*) or COUNT(DISTINCT "{key}")' if key else "COUNT(*)"
        return f"Each row is one patient ({context.rows} patients). Count patients with {how}."
    if key is None:
        return f"There is no patient ID column: {context.rows} rows. Counts are counts of rows."
    lines = [
        f"Rows are visits, not patients: {context.rows} rows for {context.patients} patients, "
        "so a patient can be on several rows."
    ]
    if writes_sql:
        lines.append(f'Count patients with COUNT(DISTINCT "{key}"), never COUNT(*).')
    else:
        lines.append(f"The app counts each patient once ({key}), whatever the number of their rows.")
    for column in context.encounter_keys:
        nouns = counting_unit.nouns_for(column) or ("encounters",)
        if writes_sql:
            lines.append(f'Count {" or ".join(nouns)} with COUNT(DISTINCT "{column}").')
        else:
            lines.append(f"A question about {' or '.join(nouns)} counts each {column} once.")
    if context.duplicate_rows:
        lines.append(f"{context.duplicate_rows} rows are exact duplicates of another row.")
    return " ".join(lines)


def context_text(context: DatasetContext, *, writes_sql: bool = True) -> str:
    """The table description block of the prompt. The same for every question on the dataset, so
    the model server can reuse its work on it from one question to the next."""
    lines = [f"Table {context.table} ({context.rows} rows). Columns:"]
    lines += [_column_line(col) for col in context.columns]
    lines += ["", f"Grain: {grain_text(context, writes_sql=writes_sql)}"]
    if context.notes:
        lines += ["", "Notes from the data dictionary:", context.notes]
    return "\n".join(lines)

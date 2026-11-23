# SPDX-License-Identifier: MIT
"""What the model and the compiler know about the table: the contract's facts, nothing guessed.

Every fact about meaning (what one record is, who the patient is, which column orders a patient's
records, dates, code labels, units, stated ranges, descriptions, open questions) comes from the
dataset version's semantics (``contracts.semantics``, ADR017) with the user's declarations
applied. The sandbox adds only what the stored values are: the category values a column holds
(so the reply schema can list them) and a number column's range. Nothing here reads the question.

Everything stays on the machine: it is sent to the local model only.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import structlog

from clinical_analytics.contracts.semantics import DatasetSemantics, OpenQuestion
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.sandbox import SandboxConnection

logger = structlog.get_logger()

# A column with at most this many values is a category: its values are listed and the reply
# schema holds them as an enum. The prompt shows at most MAX_PROMPT_VALUES of them.
MAX_CATEGORY_VALUES: Final = 40
MAX_PROMPT_VALUES: Final = 12
MAX_VALUE_CHARS: Final = 60
MAX_DESCRIPTION_CHARS: Final = 140
MAX_DISPLAY_WORDS: Final = 8

Kind = Literal["key", "category", "number", "date", "text", "flag"]

# Stored yes/no spellings (structure, never the question): a column holding only these is a flag.
_BIRTH: Final = re.compile(r"\b(?:birth|dob)\b|_dob\b|^dob", re.I)
_YES: Final = frozenset({"y", "yes", "true", "1", "t"})
_NO: Final = frozenset({"n", "no", "false", "0", "f"})


@dataclass(frozen=True)
class Column:
    """One column as the contract describes it, with the values the table holds."""

    name: str
    kind: Kind
    logical_type: str
    values: tuple[Any, ...] = ()  # the stored values of a category or flag
    labels: dict[str, str] = field(default_factory=dict)  # stored code (as text) -> its label
    low: Any = None
    high: Any = None
    description: str = ""
    unit: str | None = None
    stated_range: tuple[float, float] | None = None
    timepoint: tuple[str, str, int] | None = None  # (measure, timepoint, order)
    is_sequence: bool = False
    phi: bool = False
    distinct: int = 0
    # Non-response codes (stored value -> label, "9" -> "Don't know"): kept as categories, left out
    # of denominators and averages (the contract's ``nonresponse_codes``).
    nonresponse: dict[str, str] = field(default_factory=dict)
    unit_column: str | None = None  # a column holding this column's unit (months/years)

    def label(self, value: Any) -> str:
        """What a stored value means in words: its dictionary label, yes/no for a flag, or itself."""
        text = _text(value)
        if text in self.labels:
            return self.labels[text]
        if self.kind == "flag":
            return "yes" if text.casefold() in _YES else "no"
        return text

    def choices(self) -> tuple[str, ...]:
        """The words the reply may use for this column's values: labels where the dictionary
        gives them, else the stored values."""
        return tuple(dict.fromkeys(self.label(v) for v in self.values))

    def stored(self, said: str) -> Any:
        """The stored value a reply's word names, or None (never a near spelling)."""
        folded = said.strip().casefold()
        found = [v for v in self.values if self.label(v).casefold() == folded or _text(v).casefold() == folded]
        return found[0] if len(found) == 1 or (found and len({_text(v) for v in found}) == 1) else None

    def negative(self, value: Any) -> bool:
        """A stored value that says no (a flag's no, a label written as a negation)."""
        word = self.label(value).casefold()
        return word in _NO or word.startswith(("no ", "not ", "non", "never", "none", "negative"))

    @property
    def numeric(self) -> bool:
        return self.kind == "number"

    def words(self) -> set[str]:
        """The words that name this column in a question: its name's parts, its description's
        first clause and its value labels (lower case, three letters or more)."""
        clause = re.split(r"[:;(]|,\s|\.\s", self.description, maxsplit=1)[0] if self.description else ""
        parts = re.split(r"[_.\s]+|(?<=[a-z])(?=[A-Z])", self.name)
        labels = " ".join(self.choices()) if self.kind in ("category", "flag") else ""
        found = {w.casefold() for w in re.findall(r"[A-Za-z][A-Za-z0-9-]+", " ".join([*parts, clause, labels]))}
        return {w for w in found if len(w) >= 3 and not w.isdigit()}

    @property
    def display(self) -> str:
        """The column in clinical words, for the reading card (never its raw name when the
        dictionary describes it)."""
        if self.description:
            text = re.split(r"[:;(]|,\s|\.\s|\s[-–]\s", self.description, maxsplit=1)[0].strip(" .,")
            label_words = {w for lab in self.choices() for w in re.findall(r"\w+", lab.casefold())}
            if text and set(re.findall(r"\w+", text.casefold())) <= label_words:
                text = ""  # a description that only lists the labels ("female / male") names nothing
            if text:
                words = text.split()[:MAX_DISPLAY_WORDS]
                first = words[0]
                if not (len(first) > 1 and first[:2].isupper()):
                    words[0] = first[0].lower() + first[1:]
                return " ".join(words)
        parts = [p for p in re.split(r"[_.\s]+|(?<=[a-z])(?=[A-Z])", self.name) if p]
        unit_words = set(re.split(r"\W+", (self.unit or "").casefold()))
        kept = [p for p in parts if p.casefold() not in unit_words] or parts
        text = " ".join(p if p.isupper() and len(p) > 1 else p.lower() for p in kept)
        return f"{text} ({self.unit})" if self.unit and len(kept) < len(parts) else text


@dataclass(frozen=True)
class DatasetContext:
    """The table as the contract describes it (``semantics``) plus the values it holds."""

    table: str
    title: str
    semantics: DatasetSemantics
    columns: tuple[Column, ...]
    rows: int
    patients: int | None

    def column(self, name: str) -> Column | None:
        folded = name.casefold()
        return next((c for c in self.columns if c.name.casefold() == folded), None)

    @functools.cached_property
    def common_words(self) -> frozenset[str]:
        """Words too common in this table to name one column: in three or more columns' words, or
        in the contract's own nouns for a patient and a record."""
        counts: dict[str, int] = {}
        for column in self.columns:
            for w in column.words():
                counts[w] = counts.get(w, 0) + 1
        nouns = {w.casefold() for w in re.findall(r"[A-Za-z]+", f"{self.row_noun} {self.row_nouns} patient patients")}
        return frozenset({w for w, n in counts.items() if n >= 3} | nouns)

    @property
    def patient_key(self) -> str | None:
        """The file's own patient key (never a column the app added)."""
        key = self.semantics.patient_key_column
        return key if key is not None and self.column(key) is not None else None

    @property
    def visit_level(self) -> bool:
        """Patients have several records: counting patients and counting records differ."""
        return self.patient_key is not None and not self.semantics.one_row_per_patient

    @property
    def row_noun(self) -> str:
        """What one record is, as the contract names it ("clinic contact", "visit")."""
        if self.semantics.one_row_per_patient:
            return "patient"
        noun = self.semantics.grain.noun.strip()
        return "record" if noun in ("", "row", "line") else noun

    @property
    def row_nouns(self) -> str:
        return plural(self.row_noun)

    @property
    def aggregate_kind(self) -> Literal["frequency", "estimate"] | None:
        """What the contract says each line of a table of groups is: counts of cases to add up
        (frequency), or published estimates and rates never added up (estimate). Read from the
        contract, never inferred here."""
        sem = self.semantics
        if sem.aggregate is not None:
            return sem.aggregate.kind
        if sem.population_estimates or sem.grain.kind == "estimate":
            return "estimate"
        if sem.case_count_column is not None or sem.grain.kind == "stratum":
            return "frequency"
        return None

    @property
    def aggregate(self) -> bool:
        """Each line is a group (counts or estimates), never one person."""
        return self.aggregate_kind is not None

    @property
    def count_column(self) -> str | None:
        """The contract's case-count column of a summary table, when the table has it."""
        agg = self.semantics.aggregate
        name = (agg.count_column if agg is not None else None) or self.semantics.case_count_column
        column = self.column(name) if name else None
        return column.name if column is not None and column.numeric else None

    @property
    def total_levels(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """(column, values) the contract says mean "all" in a table of groups: a line holding one
        is a total of other lines, never one more group."""
        agg = self.semantics.aggregate
        if agg is None:
            return ()
        return tuple((t.column, t.values) for t in agg.total_levels if self.column(t.column) is not None)

    @property
    def case_noun(self) -> str:
        """What a summary table's count column counts ("deaths"), from the contract's grain noun."""
        noun = self.semantics.grain.noun
        counted = noun.removeprefix("count of ").strip() if noun.startswith("count of ") else ""
        return counted or "cases"

    @property
    def episode(self) -> tuple[str, str] | None:
        """(column, plural noun) of the contract's episode key within the patient (an admission
        within a person), when it is not simply the records' own key."""
        for ep in self.semantics.episode_keys:
            if self.column(ep.column) is not None and ep.column not in self.semantics.row_key.columns:
                return ep.column, plural(ep.noun)
        return None

    @property
    def units(self) -> tuple[str, ...]:
        """What a count can count here: patients, records (or both on a visit-level table)."""
        if self.aggregate:
            return (self.case_noun,)
        if self.semantics.one_row_per_patient:
            return ("patients",)
        if self.patient_key is None:
            return (self.row_nouns,)
        episode = (self.episode[1],) if self.episode and self.episode[1] != self.row_nouns else ()
        return ("patients", *episode, self.row_nouns)

    @property
    def order_column(self) -> str | None:
        """The column that orders a patient's records: the contract's sequence column, else its
        one date column (first or last needs one)."""
        for seq in self.semantics.sequence_columns:
            if self.column(seq.column) is not None:
                return seq.column
        # A birth date is a fact about the person, never the order of their records.
        dates = [c.name for c in self.columns if c.kind == "date" and not _BIRTH.search(f"{c.name} {c.description}")]
        return dates[0] if len(dates) == 1 else None

    def sequence_start(self, name: str) -> int | None:
        seq = next((s for s in self.semantics.sequence_columns if s.column == name), None)
        return seq.start if seq is not None else None

    def open_questions(self) -> tuple[OpenQuestion, ...]:
        return tuple(q for q in self.semantics.open_questions if not q.answered)


def plural(noun: str) -> str:
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _kind(name: str, logical: str, distinct: int, values: tuple[Any, ...], semantics: DatasetSemantics) -> Kind:
    col = semantics.column(name)
    if name == semantics.patient_key_column or (col is not None and col.role in ("row_key", "app_added")):
        return "key"
    if (col is not None and col.date is not None) or logical in ("date", "timestamp"):
        return "date"
    if logical == "boolean" or (values and {_text(v).casefold() for v in values} <= _YES | _NO and len(values) == 2):
        return "flag"
    if col is not None and col.labels and values:
        return "category"
    if logical in ("integer", "number"):
        return "number"
    return "category" if values else "text"


def build_context(
    sandbox: SandboxConnection, catalog: SqlCatalog, semantics: DatasetSemantics, *, title: str = ""
) -> DatasetContext:
    """The table's context from its contract semantics and the stored values (counts only)."""
    app = {a.column for a in semantics.app_columns}
    columns: list[Column] = []
    for spec in catalog.columns:
        if spec.name in app:
            continue  # the app's own ID copy or row number: never a fact of the file
        ident = '"' + spec.name.replace('"', '""') + '"'
        stats = sandbox.run_model_sql(
            f"SELECT COUNT(DISTINCT {ident}) AS d, MIN({ident}) AS lo, MAX({ident}) AS hi FROM unified"
        ).row(0)
        distinct, low, high = int(stats[0] or 0), stats[1], stats[2]
        col = semantics.column(spec.name)
        values: tuple[Any, ...] = ()
        is_listable = spec.logical_type not in ("date", "timestamp") and not spec.is_id
        if (
            is_listable
            and 0 < distinct <= MAX_CATEGORY_VALUES
            and (spec.logical_type not in ("integer", "number") or (col is not None and col.labels) or distinct <= 2)
        ):
            listed = sandbox.run_model_sql(
                f"SELECT DISTINCT {ident} AS v FROM unified WHERE {ident} IS NOT NULL ORDER BY 1"
            ).get_column("v")
            values = tuple(v for v in listed.to_list() if len(_text(v)) <= MAX_VALUE_CHARS)
        kind = _kind(spec.name, spec.logical_type, distinct, values, semantics)
        if spec.phi and kind in ("category", "flag"):
            kind, values = "text", ()  # a restricted column's values never reach the prompt or schema
        columns.append(
            Column(
                name=spec.name,
                kind=kind,
                logical_type=spec.logical_type,
                values=values if kind in ("category", "flag") else (),
                labels=dict(col.labels) if col is not None else {},
                low=low if kind in ("number", "date") else None,
                high=high if kind in ("number", "date") else None,
                description=" ".join((col.description or "").split())[:MAX_DESCRIPTION_CHARS] if col else "",
                unit=col.unit.value if col is not None and col.unit is not None else None,
                stated_range=(col.stated_range.minimum, col.stated_range.maximum)
                if col is not None and col.stated_range is not None
                else None,
                timepoint=(col.timepoint.measure, col.timepoint.timepoint, col.timepoint.order)
                if col is not None and col.timepoint is not None
                else None,
                is_sequence=any(s.column == spec.name for s in semantics.sequence_columns),
                phi=spec.phi,
                distinct=distinct,
                nonresponse={_text(c.value): c.label for c in col.nonresponse_codes} if col is not None else {},
                unit_column=col.unit_column if col is not None else None,
            )
        )
    counts = sandbox.run_model_sql(_counts_sql(semantics, {c.name for c in columns})).row(0)
    logger.info(
        "model_query_context_built",
        columns=len(columns),
        visit_level=semantics.patient_key_column is not None and not semantics.one_row_per_patient,
        open_questions=sum(not q.answered for q in semantics.open_questions),
    )
    return DatasetContext(
        table="unified",
        title=title,
        semantics=semantics,
        columns=tuple(columns),
        rows=int(counts[0] or 0),
        patients=int(counts[1]) if counts[1] is not None else None,
    )


def _counts_sql(semantics: DatasetSemantics, names: set[str]) -> str:
    key = semantics.patient_key_column
    patients = f'COUNT(DISTINCT "{key}")' if key is not None and key in names else "NULL"
    return f"SELECT COUNT(*) AS n, {patients} AS p FROM unified"


def _value_list(column: Column) -> str:
    shown = column.choices()[:MAX_PROMPT_VALUES]
    more = len(column.choices()) - len(shown)
    return ", ".join(f"'{v}'" for v in shown) + (f" (+{more} more)" if more > 0 else "")


def column_line(column: Column, context: DatasetContext) -> str:
    """One line of the table description the model reads."""
    if column.name == context.patient_key:
        head = f"{column.name}: the patient ID (counts patients; never a filter)"
    elif column.kind == "key":
        head = f"{column.name}: the ID of one {context.row_noun}"
    elif column.kind in ("category", "flag"):
        head = f"{column.name}: values {_value_list(column)}"
    elif column.kind == "number":
        span = f" {_short(column.low)} to {_short(column.high)}" if column.low is not None else ""
        head = f"{column.name}: number{span}" + (f" {column.unit}" if column.unit else "")
    elif column.kind == "date":
        head = f"{column.name}: date" + (f" {_short(column.low)} to {_short(column.high)}" if column.low else "")
    else:
        head = f"{column.name}: text"
    if column.is_sequence:
        head += f" (numbers each patient's {plural(context.row_noun)} in order)"
    if column.nonresponse:
        head += f" ({', '.join(column.nonresponse.values())}: non-response, left out of shares and averages)"
    if column.unit_column:
        head += f" (its unit is in {column.unit_column})"
    return head + (f". {column.description}" if column.description else "")


def _short(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())[:10]
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def context_text(context: DatasetContext, keep: set[str] | None = None) -> str:
    """The table as the model reads it: what a record is, who the patient is, the columns."""
    sem = context.semantics
    lines = [f"Table: {context.title}" if context.title else "Table:"]
    if context.aggregate_kind == "estimate":
        lines.append(
            f"{sem.grain.sentence or 'Each record is one published estimate for a group.'} Estimates are never added "
            "up: read the record of the group the question names."
        )
    elif context.aggregate:
        counted = (
            f"{context.count_column} holds how many {context.case_noun} each record counts: a count adds it up."
            if context.count_column
            else "No column is known to hold each record's count."
        )
        lines.append(f"{sem.grain.sentence or 'Each record counts a group.'} {counted}")
    elif sem.one_row_per_patient:
        lines.append(f"Each record is one patient ({context.rows} patients). Count patients.")
    elif context.patient_key is not None:
        lines.append(
            f"Each record is one {context.row_noun}; {context.patients} patients over {context.rows} records "
            f"({context.patient_key} names the patient, who can have several {context.row_nouns})."
        )
        order = context.order_column
        lines.append(
            f"A patient's {context.row_nouns} are in the order of {order}."
            if order
            else f"Nothing orders a patient's {context.row_nouns}: first or last cannot be used."
        )
    else:
        lines.append(f"Each record is one {context.row_noun} ({context.rows} records); no column names a patient.")
    if sem.survey_sample:
        lines.append("A survey sample: counts are of this file's respondents, never weighted national estimates.")
    lines.append("Columns:")
    lines += [f"- {column_line(c, context)}" for c in context.columns if keep is None or c.name in keep]
    return "\n".join(lines)

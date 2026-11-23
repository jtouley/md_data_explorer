# SPDX-License-Identifier: MIT
"""The one inference pass at upload: fill in the contract's semantics section (ADR017).

Inputs are the staged table (typed by ``storage.staging``, which reads text dates with the same
one-format rule, ``contracts.dates``), the upload's documentation text, and facts about the
upload (a workbook, a published export, declared estimates). Every fact records its source:

* **dictionary**: the data dictionary says it ("The patient identifier is owner_ref",
  "-5 = sheet not returned", "(0-40)", "MM/DD/YYYY HH:MM", "One row per clinic attendance").
* **structure**: the values show it. A column whose values repeat while another column numbers
  1..k within each value is a patient key and a sequence; a column named by a broad identifier
  convention (``PATIENT_KEY_NAMES``: ``patient_id``, ``MRN``, ...) is the key; an ID-like column
  repeating across many patients groups rows. Any other ID is only a tentative key, and asked.
* **default**: nothing says more ("Each line is one record").

No fact comes from a name beyond those broad conventions, a unit grammar (``contracts.units``)
and date words: what neither the structure nor the dictionary settles is an open question in
clinical words ("Which column identifies the patient?"), never a guess. Then ``apply_semantics``
writes missing-value codes as missing and reads dates with their one format, at the point where
the typed Parquet is written, and ``add_id_columns`` adds the app's own ID column (``patient_id``,
a copy of the key; or a row number), recorded as an app column and never as a patient identifier.

Deterministic: the same table and documentation always give the same semantics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final, Literal

import polars as pl
import structlog

from clinical_analytics.contracts import config
from clinical_analytics.contracts.dates import (
    EXCEL_SERIAL,
    NATIVE,
    YYYYMMDD,
    DateAmbiguity,
    DateChoice,
    choose_number_format,
    choose_text_format,
    describe_format,
    spoken,
)
from clinical_analytics.contracts.dictionary import Dictionary, DictionaryEntry, parse_dictionary
from clinical_analytics.contracts.semantics import (
    AggregateInfo,
    AppColumn,
    CensoredCode,
    ColumnSemantics,
    DatasetSemantics,
    DateFormat,
    EpisodeKey,
    FactSource,
    Grain,
    GroupingColumn,
    MissingCode,
    OpenQuestion,
    PatientKey,
    RowKey,
    SequenceColumn,
    StatedRange,
    Timepoint,
    TotalLevel,
    Unit,
)
from clinical_analytics.contracts.units import unit_from_tokens

logger = structlog.get_logger()

PATIENT_ID_COLUMN: Final = "patient_id"
ROW_NUMBER_COLUMN: Final = "row_number"

# Name tokens, after splitting on anything but letters and digits and at a case change.
_SPLIT: Final = re.compile(r"[^a-z0-9]+")
_CAMEL: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
ID_TOKENS: Final = frozenset({"id", "ids", "identifier", "mrn", "uuid", "guid", "ref", "key"})
# Number words name an ID only as the name's last word ("unit_no", "Bay Nr"); first, they name a
# count ("num_doses", "number_of_children").
ID_NUMBER_TOKENS: Final = frozenset({"no", "nr", "num", "number"})
# Words that make a name a count or an attribute of someone, never their ID ("owner_status",
# "owner_zip", "num_doses"), unless the name ends in an ID word.
ATTRIBUTE_TOKENS: Final = frozenset(
    {"status", "zip", "zipcode", "postcode", "postal", "name", "names", "type", "sex", "gender", "age", "dob",
     "race", "ethnicity", "state", "city", "address", "phone", "email", "weight", "height", "score", "group",
     "class", "category", "count", "total", "outcome", "flag", "initials", "language", "insurance"}
)  # fmt: skip
COUNT_TOKENS: Final = frozenset(
    {"meds", "medications", "drugs", "children", "visits", "admissions", "events", "episodes", "times", "days",
     "count", "cases", "deaths", "births", "pregnancies", "falls", "doses", "lesions", "nodes", "procedures"}
)  # fmt: skip
# The broad conventions for a patient identifier, as a whole name with its separators removed
# ("Patient_ID", "patientId", "MRN"). In this order when a file has several. Any other key comes
# from the data dictionary, a user's answer or the structure (a 1..k sequence within it); a name
# alone never makes one.
PATIENT_KEY_NAMES: Final[tuple[str, ...]] = (
    "patientid", "subjectid", "personid", "mrn", "studyid", "memberid", "caseid"
)  # fmt: skip
# Whole column names that the app and staging keep as text (an ID of digits is still an ID).
KEY_COLUMN_NAMES: Final = frozenset({"id", *PATIENT_KEY_NAMES})
# Nouns that make a stated grain ("one row per participant") one line per patient.
PATIENT_NOUNS: Final = frozenset({"patient", "person", "participant", "subject", "individual", "respondent"})
# Timepoint words in a measure's name, ordered: their rank orders the times of one measure. Any
# other word is a timepoint only when the column's dictionary entry says it is the baseline.
_TIMEPOINT_RANK: Final[Mapping[str, int]] = {
    "pre": -1,
    "before": -1,
    "baseline": 0,
    "start": 0,
    "followup": 500000,
    "post": 900000,
    "after": 900000,
    "end": 990000,
    "final": 990000,
}
_BASELINE_TEXT: Final = re.compile(r"\bbaseline\b", re.IGNORECASE)
_TIME_NUMBER: Final = re.compile(
    r"^(?:(?P<pre>m|mo|month|w|wk|week|d|day|y|yr|year|t|v|visit)(?P<n1>\d{1,3})|(?P<n2>\d{1,3})(?P<post>m|mo|mos|w|wk|wks|d|y|yr|h|hr))$"
)
# Hours per unit of a numbered timepoint ("m6" = 6 months, "24h" = 24 hours), so all times order.
_TIME_DAYS: Final[Mapping[str, int]] = {
    "m": 720,
    "mo": 720,
    "mos": 720,
    "month": 720,
    "w": 168,
    "wk": 168,
    "wks": 168,
    "week": 168,
    "d": 24,
    "day": 24,
    "y": 8760,
    "yr": 8760,
    "year": 8760,
    "t": 24,
    "v": 24,
    "visit": 24,
    "h": 1,
    "hr": 1,
}
_EXAMPLE_DAY: Final = datetime(2021, 4, 3, 14, 5)
_MAX_KEY_CANDIDATES: Final = 6


def name_tokens(column: str) -> list[str]:
    """'OwnerRef' -> ['owner', 'ref']; 'grip_left_kg' -> ['grip', 'left', 'kg']."""
    return [t for t in _SPLIT.split(_CAMEL.sub("_", column).lower()) if t]


def has_id_word(column: str) -> bool:
    """An ID word anywhere ("owner_id", "UnitRef", "operator_code"), or a number word last ("bay_no")."""
    tokens = name_tokens(column)
    return bool(tokens) and (any(t in ID_TOKENS for t in tokens) or tokens[-1] in ID_NUMBER_TOKENS)


def _attribute_named(tokens: list[str]) -> bool:
    """A count or an attribute of someone ("owner_status", "num_doses"), unless it ends in an ID word."""
    if tokens and (tokens[-1] in ID_TOKENS or tokens[-1] in ID_NUMBER_TOKENS):
        return False
    return any(t in ATTRIBUTE_TOKENS or t in COUNT_TOKENS for t in tokens) or (
        bool(tokens) and tokens[0] in ID_NUMBER_TOKENS
    )


def key_name_rank(column: str) -> int | None:
    """The column's place in ``PATIENT_KEY_NAMES`` (its whole name, separators removed), or None."""
    joined = "".join(name_tokens(column))
    return PATIENT_KEY_NAMES.index(joined) if joined in PATIENT_KEY_NAMES else None


def key_column_named(column: str) -> bool:
    """A whole name staging keeps as text: ``id`` or a patient-identifier convention."""
    return "".join(name_tokens(column)) in KEY_COLUMN_NAMES


def names_patient(column: str) -> bool:
    """A column named by a broad patient-identifier convention (``patient_id``, ``MRN``, ``subject_id``)."""
    return key_name_rank(column) is not None


def id_named(column: str) -> bool:
    """Named as an ID of someone or something: an ID word (not a count or attribute), or a key convention."""
    tokens = name_tokens(column)
    return (has_id_word(column) and not _attribute_named(tokens)) or names_patient(column)


def grain_kind(noun: str) -> str:
    """'patient' when a stated grain names a person ("one row per participant"), 'estimate' for
    published estimates, else the noun's last word ("unit stay" -> "stay")."""
    words = [w for w in re.split(r"[^a-z]+", noun.lower()) if w]
    if not words:
        return "record"
    last = words[-1]
    singular = last[:-1] if last.endswith("s") and len(last) > 3 and not last.endswith("ss") else last
    if singular in PATIENT_NOUNS:
        return "patient"
    return singular


# ----------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class UploadFacts:
    """What the upload path knows beyond the table: its documentation and where it came from."""

    doc_context: str | None = None
    from_excel: bool = False
    source_format: str = ""
    population_estimates: bool = False
    # Formats staging read text dates with (``StagedTable.dates``), by column.
    staged_dates: Mapping[str, DateChoice] | None = None
    # A key a multi-table cohort was built on (its grain key).
    cohort_key: str | None = None


def _logical_type(dtype: pl.DataType) -> str:
    from clinical_analytics.contracts.builder import logical_type

    return logical_type(dtype)


def _synthetic_example(choice_format: str, has_time: bool) -> tuple[str, str]:
    """A worked example on a fixed day (3 April 2021), never a row's value."""
    day = _EXAMPLE_DAY if has_time else _EXAMPLE_DAY.date()
    if choice_format == EXCEL_SERIAL:
        serial = (_EXAMPLE_DAY.date() - date(1899, 12, 30)).days
        return str(serial), spoken(day)
    if choice_format == NATIVE:
        return day.isoformat(), spoken(day)
    pattern = choice_format.replace("%.f", ".%f").replace("%B", "%b")
    text = day.strftime(pattern.replace("Z", "") if "T" in pattern else pattern)
    return text, spoken(day)


def _date_format(choice: DateChoice, source: FactSource) -> DateFormat:
    ex_in, ex_out = _synthetic_example(choice.format, choice.has_time)
    return DateFormat(
        format=choice.format,
        written=describe_format(choice.format),
        example_in=ex_in,
        example_out=ex_out,
        has_time=choice.has_time,
        source=source,
        evidence=choice.evidence,
    )


# ----------------------------------------------------------------------------- per column
def _typed_code(code: str, dtype: pl.DataType) -> int | float | str | None:
    """A dictionary code as a value of the column's type (99 for Int64, 99.0 for Float64, "99"
    for text), or None when the column can't hold it."""
    if dtype == pl.String:
        return code
    if dtype.is_integer() and re.fullmatch(r"-?\d+", code):
        return int(code)
    if dtype.is_float() and re.fullmatch(r"-?\d+(?:\.\d+)?", code):
        return float(code)
    return None


def _holds_years(series: pl.Series, unit: str) -> bool:
    """A "years" name unit on a column of calendar years ("birth_yr" = 1957) is not a unit."""
    if unit != "years" or not series.dtype.is_numeric():
        return False
    values = series.drop_nulls()
    return values.len() > 0 and bool(values.is_between(config.DATE_YEAR_MIN, config.DATE_YEAR_MAX).all())


def _date_question(name: str, found: DateAmbiguity) -> OpenQuestion:
    """The clinician's question for a date column that stays text, with a worked example written
    in the column's own style on a fixed day (never a row's value)."""
    if found.kind == "century":
        example = datetime(1945, 3, 15).strftime(found.format.replace("%B", "%b"))
        return OpenQuestion(
            id=f"date_century:{name}",
            kind="date_century",
            column=name,
            value=found.format,
            question=f"In {name}, does {example} mean 15 March 1945 or 15 March 2045?",
            options=("1900s (15 March 1945)", "2000s (15 March 2045)"),
        )
    example = _EXAMPLE_DAY.strftime(found.format.replace("%B", "%b"))
    return OpenQuestion(
        id=f"date_order:{name}",
        kind="date_order",
        column=name,
        value=found.format,
        question=f"In {name}, does {example} mean 3 April 2021 or March 4, 2021?",
        options=("day first (3 April)", "month first (March 4)"),
    )


def _undocumented_missing(series: pl.Series, stated: tuple[float, float] | None) -> list[MissingCode]:
    """Repeated values from ``MISSING_CODE_CANDIDATES`` far outside the column's other values:
    the largest value, and every negative code in a column whose other values are 0 or more
    ("pack_years" with -9 and -8). Counts, changes and lengths of stay hold real -1 and 99 values:
    asked there only when far beyond everything else (a stay of 999 days)."""
    if not series.dtype.is_numeric() or series.dtype == pl.Boolean:
        return []
    values = series.drop_nulls()
    if values.len() < 3:
        return []
    floats = values.cast(pl.Float64)
    distinct = [_float(v) for v in floats.unique().to_list()]
    negatives = [v for v in distinct if v < 0 and v.is_integer() and int(v) in config.MISSING_CODE_CANDIDATES]
    top = _float(floats.max())
    candidates = sorted(negatives)
    if top > 0 and top.is_integer() and int(top) in config.MISSING_CODE_CANDIDATES:
        candidates.append(top)
    found: list[MissingCode] = []
    for code in candidates:
        hits = int((floats == code).sum())
        if hits < config.MISSING_CODE_MIN_ROWS or hits < config.MISSING_CODE_MIN_SHARE * values.len():
            continue
        if _far_outside(series, code, stated, ignore=negatives):
            side = "below every other value" if code < 0 else "far above every other value"
            found.append(_suspect(series, code, hits, side))
    return found


def _far_outside(
    series: pl.Series, code: float, stated: tuple[float, float] | None, ignore: Sequence[float] = ()
) -> bool:
    """``code`` lies far outside the column's other values (``ignore``: other codes left out), so it
    cannot be a real measurement: a negative code in a column of values 0 or more, or a code far
    above every other value (and above the stated range). On a count, change or stay the -1 and 99
    rules do not apply; 999 and above far beyond the rest still counts (a stay of 999 days)."""
    exempt = bool(set(name_tokens(series.name)) & config.MISSING_CODE_EXEMPT_TOKENS)
    floats = series.drop_nulls().cast(pl.Float64)
    rest = floats.filter(~floats.is_in([code, *ignore]))
    if rest.len() == 0:
        return False
    if code < 0:
        return not exempt and _float(rest.min()) >= 0
    if exempt and code < 999:
        return False
    if code < _float(rest.max()):
        return False
    if stated is not None and not exempt:
        return code > stated[1] and _float(rest.max()) <= stated[1] * config.MISSING_CODE_GAP
    gap = config.MISSING_CODE_EXEMPT_GAP if exempt else config.MISSING_CODE_GAP
    return code > max(_float(rest.max()), 0.0) * gap


_UNIT_WORDS: Final = frozenset({"unit", "units", "u", "uom", "scale"})
_RATE_UNIT_WORDS: Final = frozenset({"per", "freq", "frequency", "rate", "times", "schedule", "interval", "every"})


def _unit_columns(df: pl.DataFrame) -> list[tuple[str, str]]:
    """(value column, unit column) candidates: a text column whose filled values are mostly time
    units ("months", "years") beside the number column it names ("dur" and "dur_unit") or, failing
    that, the number column just before it. Never a rate ("dose_per" = day/week beside "dose_mg"):
    a unit column named for a frequency, or a value column with a unit of its own."""
    pairs = []
    order = list(df.columns)
    for i, c in enumerate(order):
        if df.schema[c] != pl.String or set(name_tokens(c)) & _RATE_UNIT_WORDS:
            continue
        filled = [str(v).strip().lower() for v in df.get_column(c).drop_nulls().to_list() if str(v).strip()]
        known = [v for v in filled if v in config.TIME_UNIT_NAMES]
        if not known or len(set(filled)) > 8 or len(known) < 0.8 * len(filled):
            continue
        stem = [t for t in name_tokens(c) if t not in _UNIT_WORDS]
        numbers = [n for n in order if n != c and df.schema[n].is_numeric() and _unit_from_name(n) is None]
        named = [n for n in numbers if stem and name_tokens(n) == stem]
        before = [order[i - 1]] if i > 0 and order[i - 1] in numbers else []
        partners = named or before
        if partners:
            pairs.append((partners[0], c))
    return pairs


def _unit_stated(dictionary: Dictionary, value_column: str, unit_column: str) -> bool:
    """The dictionary says ``unit_column`` gives the unit of ``value_column`` ("Gives the unit of
    dur")."""
    entry = dictionary.entry(unit_column)
    text = f"{entry.description} {entry.text}" if entry is not None else ""
    return bool(re.search(r"\bunits?\b", text, re.I)) and bool(
        re.search(rf"(?<![\w]){re.escape(value_column)}(?![\w])", text, re.I)
    )


def _unit_pair_questions(df: pl.DataFrame, value_column: str, unit_column: str, stated: bool) -> list[OpenQuestion]:
    """ "Does dur_unit give the unit of dur?" (unless the dictionary says so), and "N lines of dur
    have no unit: leave them out?" (blank or unrecognised units: never read as any unit)."""
    found = []
    target = _largest_unit(df, unit_column)
    if not stated:
        found.append(
            OpenQuestion(
                id=f"unit_pair:{value_column}:{unit_column}",
                kind="unit_pair",
                column=value_column,
                value=target,
                question=(
                    f"Does {unit_column} give the unit of {value_column} (days, months, years…)? If so, answers read "
                    f"every line of {value_column} in {target}."
                ),
                options=(f"yes, {unit_column} gives the unit of {value_column}", "no, they are separate"),
            )
        )
    unit = pl.col(unit_column).cast(pl.String).str.strip_chars().str.to_lowercase()
    missing = int(
        df.select(
            (pl.col(value_column).is_not_null() & ~unit.is_in(list(config.TIME_UNIT_NAMES)).fill_null(True)).sum()
        ).item()
    )
    if missing:
        found.append(
            OpenQuestion(
                id=f"unit_missing:{value_column}",
                kind="unit_missing",
                column=value_column,
                value=missing,
                question=(
                    f"{missing:,} lines of {value_column} have no unit in {unit_column}. Leave them out of answers?"
                ),
                options=("yes, leave them out", "no, I'll fix the file"),
            )
        )
    return found


def _largest_unit(df: pl.DataFrame, unit_column: str) -> str:
    used = {
        config.TIME_UNIT_NAMES[v]
        for v in (str(x).strip().lower() for x in df.get_column(unit_column).drop_nulls().unique().to_list())
        if v in config.TIME_UNIT_NAMES
    }
    return max(used, key=lambda unit: config.TIME_UNIT_DAYS[unit])


def _coded(codes: Mapping[str, str], dtype: pl.DataType, present: set[str]) -> tuple[MissingCode, ...]:
    """A codebook's codes (code -> label) the column holds, typed to the column."""
    found = []
    for code, label in codes.items():
        typed = _typed_code(code, dtype)
        if typed is not None and (str(typed) in present or code in present):
            found.append(MissingCode(value=typed, label=label, source="dictionary", evidence="the codebook lists it"))
    return tuple(found)


_TOP_CODE: Final = re.compile(
    r"\b(?:or|and)\s+(?:over|more|older|above|higher)\b|\d\s*\+|\bat least\b|\btop[- ]?coded\b", re.IGNORECASE
)


def _noted_code_question(column: str, code: int | float, note: str) -> OpenQuestion:
    """ "Does 99 in age mean not recorded?", quoting the note that states it. A note that the
    code stands for a range ("90 or over coded as 99") offers that reading: a real value."""
    options = ["yes, not recorded", "no, it is a real value"]
    if _TOP_CODE.search(note):
        options.append("it stands for a value at the top of the range (top-coded): a real value, not missing")
    return OpenQuestion(
        id=f"missing_code:{column}:{code}",
        kind="missing_code",
        column=column,
        value=code,
        question=f'The data dictionary says "{note.strip().rstrip(".")}". Does {code:g} in {column} mean not recorded?',
        options=tuple(options),
    )


def _same_code(a: object, b: object) -> bool:
    try:
        return float(str(a)) == float(str(b))
    except ValueError:
        return str(a) == str(b)


def _suspect(series: pl.Series, code: float, hits: int, side: str) -> MissingCode:
    value: int | float = int(code) if series.dtype.is_integer() else code
    return MissingCode(value=value, source="structure", evidence=f"{_n(hits)} lines hold {value:g}, {side}")


def _is_number(value: object) -> bool:
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def _float(value: object) -> float:
    """A Series aggregate (min/max of a Float64 series) as a float."""
    if not isinstance(value, int | float):
        raise TypeError(f"Expected a number, got {type(value).__name__}")
    return float(value)


def _n(count: int) -> str:
    return f"{count:,}"


_CODED_TOKENS: Final = frozenset({"cat", "category", "categories", "band", "banded", "grp", "class", "flag"})


def _unit_from_name(column: str) -> str | None:
    """The unit a name ends in, by the unit grammar (``contracts.units``): "urea_mmol_l" is
    mmol/L; "vitamin_d" names no unit."""
    return unit_from_tokens([t for t in _SPLIT.split(column.lower()) if t])  # no case split: "cmH2O"


def _timepoint_token(token: str) -> tuple[str, int] | None:
    if token in _TIMEPOINT_RANK:
        return token, _TIMEPOINT_RANK[token]
    m = _TIME_NUMBER.match(token)
    if m is None:
        return None
    unit = m.group("pre") or m.group("post")
    number = int(m.group("n1") or m.group("n2"))
    return token, number * _TIME_DAYS[unit]


def _timepoints(columns: Iterable[str], numeric: set[str], baseline: Collection[str] = ()) -> dict[str, Timepoint]:
    """Measures stored at two or more times: same name but for one timepoint word. A word that is
    no timepoint word ("grip_t0", "grip_ref") is one only when the column's dictionary entry says
    it is the baseline (``baseline``: those columns)."""
    groups: dict[str, list[tuple[str, str, int]]] = {}
    unplaced: list[tuple[str, list[str]]] = []
    for column in columns:
        if column not in numeric:
            continue
        tokens = name_tokens(column)
        for i, token in enumerate(tokens):
            found = _timepoint_token(token)
            if found is None:
                continue
            base = "_".join(tokens[:i] + tokens[i + 1 :])
            if base:
                groups.setdefault(base, []).append((column, found[0], found[1]))
            break
        else:
            if column in baseline:
                unplaced.append((column, tokens))
    for column, tokens in unplaced:
        for i, token in enumerate(tokens):
            base = "_".join(tokens[:i] + tokens[i + 1 :])
            if base in groups:
                groups[base].append((column, token, 0))
                break
    out: dict[str, Timepoint] = {}
    for base, members in groups.items():
        if len({m[1] for m in members}) < 2:
            continue
        ordered = sorted(members, key=lambda m: (m[2], m[0]))
        names = tuple(m[0] for m in ordered)
        for order, (column, word, _rank) in enumerate(ordered):
            out[column] = Timepoint(
                measure=base,
                timepoint=word,
                order=order,
                paired_with=tuple(n for n in names if n != column),
                source="structure",
            )
    return out


# ----------------------------------------------------------------------------- keys
@dataclass(frozen=True)
class _Candidate:
    column: str
    distinct: int
    nulls: int
    unique: bool  # one row per value, no empty value
    convention: int | None  # its place in PATIENT_KEY_NAMES, when named by one
    named: bool  # named as an ID
    shaped: bool = False  # its values are written like identifiers ("AB-0012"), whatever its name

    @property
    def id_like(self) -> bool:
        return self.named or self.shaped


def _candidates(df: pl.DataFrame, excluded: set[str], stated: Collection[str] = ()) -> list[_Candidate]:
    """ID-like columns: named as an ID, text with many values, or one the dictionary states is the
    person's identifier ("Respondent sequence number"), whatever its name."""
    from clinical_analytics.core.identifier_rules import can_identify_patients

    rows = df.height
    eligible = [
        c
        for c in df.columns
        if c not in excluded
        and c.lstrip("_") != ROW_NUMBER_COLUMN
        and (df.schema[c].is_integer() or df.schema[c] == pl.String or df.schema[c].is_float())
        and can_identify_patients(df.get_column(c))
    ]
    if not eligible or rows == 0:
        return []
    stats = df.select(
        *(pl.col(c).n_unique().alias(f"{c}\x00u") for c in eligible),
        *(pl.col(c).null_count().alias(f"{c}\x00n") for c in eligible),
    ).row(0, named=True)
    found: list[_Candidate] = []
    for c in eligible:
        nulls = int(stats[f"{c}\x00n"])
        distinct = int(stats[f"{c}\x00u"]) - (1 if nulls else 0)
        named = id_named(c)
        text_like = df.schema[c] == pl.String and distinct >= max(2, rows // 20)
        if distinct < 2 or not (named or text_like or c in stated):
            continue
        found.append(
            _Candidate(
                column=c,
                distinct=distinct,
                nulls=nulls,
                unique=nulls == 0 and distinct == rows,
                convention=key_name_rank(c),
                named=named,
                shaped=id_shaped(df.get_column(c)),
            )
        )
    return found


_ID_SHAPE: Final = r"^[A-Za-z]{1,4}[-_./]?[0-9][A-Za-z0-9\-_./]{0,15}$"


def id_shaped(series: pl.Series) -> bool:
    """Text written like identifiers: a short prefix of letters, then digits ("P0012", "AB-17",
    "OW001"), on every filled line; never words or numbers with a decimal point."""
    if series.dtype != pl.String:
        return False
    values = series.drop_nulls()
    return (
        values.len() > 0
        and bool(values.str.contains(_ID_SHAPE).all())
        and not bool(values.str.contains(r"^\d+\.\d+$").any())
    )


def _sequence_within(df: pl.DataFrame, key: str, columns: Sequence[str]) -> list[tuple[str, int]]:
    """Integer columns that number the rows of each ``key`` value 1..k (or 0..k-1)."""
    numbered: list[tuple[str, int]] = []
    keyed = df.lazy().filter(pl.col(key).is_not_null())
    for column in columns:
        g = (
            keyed.group_by(key)
            .agg(
                pl.len().alias("n"),
                pl.col(column).n_unique().alias("u"),
                pl.col(column).null_count().alias("z"),
                pl.col(column).min().alias("lo"),
                pl.col(column).max().alias("hi"),
            )
            .filter(pl.col("n") > 1)
            .select(
                pl.len().alias("groups"),
                (
                    (pl.col("u") == pl.col("n"))
                    & (pl.col("z") == 0)
                    & pl.col("lo").is_in([0, 1])
                    & (pl.col("hi") <= pl.col("lo") + 3 * pl.col("n"))
                )
                .sum()
                .alias("ok"),
                pl.col("lo").min().alias("start"),
            )
            .collect()
            .row(0, named=True)
        )
        if g["groups"] and g["ok"] / g["groups"] >= config.SEQUENCE_MIN_SHARE:
            numbered.append((column, int(g["start"] or 1)))
    return numbered


def _sequence_columns(df: pl.DataFrame, excluded: set[str]) -> list[str]:
    """Integer columns that could number rows: whole numbers from 0 or 1, up to 1000, and more
    than 0/1 (a yes/no flag never numbers visits)."""
    ints = [c for c in df.columns if c not in excluded and df.schema[c].is_integer()]
    if not ints:
        return []
    stats = df.select(
        *(pl.col(c).min().alias(f"{c}\x00lo") for c in ints), *(pl.col(c).max().alias(f"{c}\x00hi") for c in ints)
    ).row(0, named=True)
    return [
        c
        for c in ints
        if stats[f"{c}\x00lo"] in (0, 1) and stats[f"{c}\x00hi"] is not None and 2 <= stats[f"{c}\x00hi"] <= 1000
    ]


@dataclass
class _Keys:
    patient: PatientKey | None = None
    row: RowKey = RowKey(kind="none")
    sequences: tuple[SequenceColumn, ...] = ()
    groups: tuple[GroupingColumn, ...] = ()
    question: OpenQuestion | None = None
    episodes: tuple[EpisodeKey, ...] = ()


def _nested(df: pl.DataFrame, fine: str, coarse: str) -> bool:
    """Every value of ``fine`` sits on lines of one ``coarse`` value (a record within a person)."""
    most = (
        df.lazy()
        .filter(pl.col(fine).is_not_null() & pl.col(coarse).is_not_null())
        .group_by(fine)
        .agg(pl.col(coarse).n_unique().alias("k"))
        .select(pl.col("k").max())
        .collect()
        .item()
    )
    return bool(most == 1)


def _cross_classified(df: pl.DataFrame, columns: Sequence[str]) -> bool:
    """Two or more repeating IDs that together tell every line apart: the dimensions of a table of
    groups (a place by a year by a category), not a person and their records."""
    if len(columns) < 2:
        return False
    keyed = df.drop_nulls(list(columns))
    return keyed.height > 0 and not bool(keyed.select(pl.struct(list(columns)).is_duplicated().any()).item())


def _single_line_share(df: pl.DataFrame, column: str) -> float:
    """The share of ``column``'s values that sit on one line only."""
    counts = df.lazy().filter(pl.col(column).is_not_null()).group_by(column).len().collect().get_column("len")
    return float((counts == 1).sum()) / counts.len() if counts.len() else 0.0


def key_question(options: Sequence[_Candidate], why: str) -> OpenQuestion:
    """ "Which column identifies the patient?" over ``options`` (the first is the tentative key, if
    any), with each option's number of different values so an answer can name the patients."""
    names = [c.column for c in options]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" or {names[-1]}"
    return OpenQuestion(
        id=config.KEY_QUESTION_ID,
        kind="patient_key",
        column=names[0],
        question=(
            f"Does {listed} identify the patient? {why}"
            if len(names) == 1
            else f"Which column identifies the patient: {listed}? {why}"
        ),
        options=(*names, config.KEY_ANSWER_NONE),
        option_counts=tuple(c.distinct for c in options),
    )


def _structural_key(
    df: pl.DataFrame, candidates: Sequence[_Candidate], order: Mapping[str, int]
) -> tuple[_Candidate | None, str, OpenQuestion | None]:
    """(tentative key, evidence, question) when only the structure is left to go on: always asked.

    A repeating ID says lines belong to something, not that it is the patient (a ward, a device
    and a clinician repeat too). A repeating ID with a few lines per value (``KEY_MIN_SHARE``
    different values per line or more) and no rival is the tentative key; a line-unique ID nested
    in a rarely repeating one (a second person on each line: a child and a carer, a donor and a
    recipient) makes the unique one tentative; two such IDs, the one with more values. With many
    lines per value there is no tentative key. Nothing in a name settles it.
    """
    rows = df.height
    ids = [c for c in candidates if c.id_like]
    repeating = sorted((c for c in ids if not c.unique), key=lambda c: (-c.distinct, order[c.column]))
    if not repeating:
        return None, "", None
    top, rivals = repeating[0], [c for c in repeating[1:] if c.distinct >= config.KEY_MIN_SHARE * rows]
    share = top.distinct / rows
    if share >= config.KEY_MIN_SHARE and rivals:
        why = "Each repeats across lines with a few lines per value; nothing in the file says which names the patient."
        return top, f"{top.column} has the most different values", key_question([top, *rivals], why)
    if share >= config.KEY_MIN_SHARE:
        single = _single_line_share(df, top.column)
        if single >= config.KEY_RECORD_SHARE:
            # Nearly every value on one line: record IDs with a few repeats (or gaps), not
            # patients with several lines each. Asked, with no tentative key.
            why = (
                f"Nearly all of its {_n(top.distinct)} values sit on one line; nothing in the file says whom it names."
            )
            return None, "", key_question([top], why)
        fine = [c for c in ids if c.unique and _nested(df, c.column, top.column)]
        if fine and single >= config.KEY_SECOND_PERSON_SHARE:
            # Most values of the repeating ID sit on one line: each line may be one person, with
            # the repeating ID naming a second person some lines share.
            why = (
                f"{fine[0].column} is on every line once and {top.column} repeats on a few lines; "
                "each line may be one person, grouped by another."
            )
            return fine[0], f"{fine[0].column} is on every line once", key_question([fine[0], top], why)
        why = (
            f"Its values repeat across lines ({_n(top.distinct)} values over {_n(rows)} lines); "
            "nothing in the file says whom it names."
        )
        return (
            top,
            f"{top.column} is an ID whose values repeat across lines, a few lines each",
            key_question([top], why),
        )
    why = (
        f"{top.column} has {_n(top.distinct)} different values over {_n(rows)} lines; "
        "nothing in the file says whom it names."
    )
    return None, "", key_question(repeating[:_MAX_KEY_CANDIDATES], why)


_ID_TEXT: Final = re.compile(r"\b(?:id|identifier|identifies)\b", re.IGNORECASE)
_ROLE_TEXT: Final = re.compile(r"^\s*(?P<who>[A-Za-z][A-Za-z '\-]{0,40}?)\s+(?:id|identifier|number|code)\b", re.I)


def _group_role(column: str, entry: DictionaryEntry | None) -> str:
    """Who a grouping ID names, in the dictionary's words ("Operating clinician identifier" ->
    "operating clinician"), else in the column's own name without its ID word, else "group"."""
    said = _ROLE_TEXT.match(entry.description) if entry is not None and entry.description else None
    if said is not None:
        return said.group("who").strip().lower()
    words = [t for t in name_tokens(column) if t not in ID_TOKENS and t not in ID_NUMBER_TOKENS and not t.isdigit()]
    return " ".join(words) or "group"


def _grouping(
    df: pl.DataFrame, dictionary: Dictionary, key: str, taken: set[str], excluded: set[str]
) -> list[GroupingColumn]:
    """IDs for another role than the patient, by structure: an ID-like column (named as an ID, or
    one its dictionary entry calls an identifier) with fewer values than half the patients, whose
    values repeat across several patients. Never a measure."""
    from clinical_analytics.core.identifier_rules import can_identify_patients

    patients = int(df.get_column(key).drop_nulls().n_unique())
    found: list[GroupingColumn] = []
    for column in df.columns:
        if column in taken or column in excluded or column == key:
            continue
        dtype = df.schema[column]
        entry = dictionary.entry(column)
        said = entry is not None and bool(_ID_TEXT.search(entry.description))
        if entry is not None and (entry.labels or entry.nonresponse):
            continue  # a coded item (its codes have labels) is a category, never an ID
        if not (id_named(column) or said or id_shaped(df.get_column(column))) or not (
            dtype == pl.String or dtype.is_integer() or dtype.is_float()
        ):
            continue
        series = df.get_column(column)
        distinct = int(series.drop_nulls().n_unique())
        if distinct < 2 or distinct * 2 > patients or not can_identify_patients(series):
            continue
        spans = (
            df.lazy()
            .filter(pl.col(column).is_not_null())
            .group_by(column)
            .agg(pl.col(key).n_unique().alias("p"))
            .select(pl.col("p").max())
            .collect()
            .item()
        )
        if not spans or spans < 2:
            continue
        found.append(
            GroupingColumn(
                column=column,
                role=_group_role(column, entry),
                is_id=True,
                source="dictionary" if said else "structure",
                evidence=f"its {_n(distinct)} values each repeat across several patients",
            )
        )
    return found


def _choose_keys(df: pl.DataFrame, dictionary: Dictionary, facts: UploadFacts, excluded: set[str]) -> _Keys:
    """Patient key, row key, sequences and grouping columns, in this order of evidence:

    1. the dictionary names the patient identifier; 2. the cohort was joined on it; 3. a broad
    identifier convention (``PATIENT_KEY_NAMES``); 4. an ID whose values repeat while another
    column numbers 1..k within each; 5. a unique ID on a file the dictionary says has one line per
    patient; 6. the structure alone (``_structural_key``), which asks whenever it is unsure.
    """
    rows = df.height
    stated_entries = [e.column for e in dictionary.entries.values() if e.names_patient_key]
    stated_names = (*dictionary.key_statements, *stated_entries)
    # A non-unique ID with two values is a flag or a pair of sites, not patients (unless the
    # dictionary or a convention says it is the patient).
    candidates = [
        c
        for c in _candidates(df, excluded, stated_names)
        if (c.unique or c.distinct > 2 or c.convention is not None or c.column in stated_names)
        # A key is on most lines: an ID missing on more than half of them names no one reliably.
        and (c.unique or c.nulls * 2 <= rows)
    ]
    by_name = {c.column: c for c in candidates}
    stated = [c for c in stated_names if c in by_name]
    # Times since an index event are sequences of their own kind (``_time_since``), never order numbers.
    seq_cols = [s for s in _sequence_columns(df, excluded) if not _time_since_named(s)]
    keys = _Keys()
    chosen: _Candidate | None = None
    source: FactSource = "structure"
    confidence: str = "high"
    evidence = ""
    sequences: list[tuple[str, int]] = []
    order = {c: i for i, c in enumerate(df.columns)}

    def nested(c: _Candidate) -> list[tuple[str, int]]:
        return [] if c.unique else _sequence_within(df, c.column, [s for s in seq_cols if s != c.column])

    if stated:
        chosen, source = by_name[stated[0]], "dictionary"
        evidence = "the data dictionary names it as the patient identifier"
    elif facts.cohort_key is not None and facts.cohort_key in by_name:
        chosen, evidence = by_name[facts.cohort_key], "the tables were joined on it, one patient each"
    if chosen is None:
        named = [c for c in candidates if c.convention is not None]
        if any(c.unique for c in named):
            # A repeating column with few values never beats a unique patient ID.
            floor = max(3, config.GENERIC_KEY_MIN_SHARE * rows)
            named = [c for c in named if c.unique or c.distinct >= floor]
        named = sorted(named, key=lambda c: (c.convention, c.unique, order[c.column]))
        if named:
            chosen = named[0]
            evidence = f"{chosen.column} is a usual name for a patient identifier"
    if chosen is None:
        # Beside a count column, a group name numbered 1..k is a crossed table of groups (a region
        # by bands 1..4), never a patient; an ID-named column still can be.
        counted = bool(_count_columns(df, excluded, dictionary))
        unknown = sorted(
            (c for c in candidates if not c.unique and (c.named or not counted)),
            key=lambda c: (not c.named, -c.distinct, order[c.column]),
        )[:_MAX_KEY_CANDIDATES]
        for c in unknown:
            found = nested(c)
            if found:
                chosen, sequences = c, found
                evidence = f"its values repeat while {found[0][0]} numbers 1, 2, 3… within each"
                break
    unique_ids = [c for c in candidates if c.unique and c.id_like]
    says_patient = dictionary.one_per_person or grain_kind(dictionary.grain_noun or "") == "patient"
    if chosen is None and unique_ids and says_patient:
        chosen = sorted(unique_ids, key=lambda c: order[c.column])[0]
        source, evidence = (
            "dictionary",
            "the data dictionary says each line is one patient, and it is on every line once",
        )
    repeating_ids = [c.column for c in candidates if c.id_like and not c.unique]
    if chosen is None and not _cross_classified(df, repeating_ids):
        chosen, evidence, keys.question = _structural_key(df, candidates, order)
        confidence = "low" if keys.question is not None else "medium"
    if chosen is not None:
        if not sequences:
            sequences = nested(chosen)
        keys.patient = PatientKey(
            column=chosen.column,
            unique=chosen.unique,
            confidence=confidence if source == "structure" else "high",  # type: ignore[arg-type]
            source=source,
            evidence=evidence,
        )
        keys.sequences = tuple(
            SequenceColumn(
                column=s,
                within=chosen.column,
                start=start,
                noun="line",
                source="structure",
                evidence=f"it numbers the lines of each {chosen.column} from {start}",
            )
            for s, start in sequences
        )
    seq_names = {s.column for s in keys.sequences}
    row_ids = [c for c in candidates if c.unique and c.column not in seq_names]
    if keys.patient is not None and keys.patient.unique:
        keys.row = RowKey(kind="column", columns=(keys.patient.column,), evidence="it is on every line once")
    elif row_ids:
        best = sorted(row_ids, key=lambda c: (not c.id_like, c.nulls, str(df.get_column(c.column).head(3).to_list())))[
            0
        ]
        keys.row = RowKey(kind="column", columns=(best.column,), evidence="it is on every line once")
    if keys.patient is not None:
        key = keys.patient.column
        taken = seq_names | set(keys.row.columns)
        keys.groups = tuple(_grouping(df, dictionary, key, taken, excluded))
        grouped = taken | {g.column for g in keys.groups} | {key}
        keys.episodes = _episode_keys(df, key, [c for c in candidates if c.id_like and c.column not in grouped])
        keys.sequences = (*keys.sequences, *_time_since(df, key, excluded | grouped))
    return keys


_SINCE_TOKENS: Final = frozenset({"since", "from", "post", "after"})
_TIME_UNIT_TOKENS: Final = frozenset({"day", "days", "week", "weeks", "wk", "wks", "month", "months", "hour", "hours"})


def _time_since_named(column: str) -> bool:
    tokens = set(name_tokens(column))
    return bool(tokens & _SINCE_TOKENS and tokens & _TIME_UNIT_TOKENS)


def _time_since(df: pl.DataFrame, key: str, excluded: set[str]) -> tuple[SequenceColumn, ...]:
    """Integer times since an index event ("weeks_since_start", "day_of_admission") that differ on
    each line of a patient: when the line was taken, in order (a sequence, never a measure)."""
    found: list[SequenceColumn] = []
    for column in df.columns:
        tokens = name_tokens(column)
        if column == key or column in excluded or not df.schema[column].is_integer():
            continue
        if not _time_since_named(column):
            continue
        g = (
            df.lazy()
            .filter(pl.col(key).is_not_null())
            .group_by(key)
            .agg(pl.len().alias("n"), pl.col(column).n_unique().alias("u"), pl.col(column).null_count().alias("z"))
            .filter(pl.col("n") > 1)
            .select(pl.len().alias("groups"), ((pl.col("u") == pl.col("n")) & (pl.col("z") == 0)).sum().alias("ok"))
            .collect()
            .row(0, named=True)
        )
        if g["groups"] and g["ok"] / g["groups"] >= config.TIME_SINCE_MIN_SHARE:
            unit = next(t for t in tokens if t in _TIME_UNIT_TOKENS).rstrip("s")
            found.append(
                SequenceColumn(
                    column=column,
                    within=key,
                    start=0,
                    noun=f"{unit} since the start",
                    kind="time_since_index",
                    source="structure",
                    evidence=f"it differs on each line of a {key}: the time each line was taken",
                )
            )
    return tuple(found)


def _composite_row_key(df: pl.DataFrame, aggregate: AggregateInfo | None) -> RowKey:
    """Columns that together tell every line apart: a table's groups when they do, else the first
    combination found over the columns in an order of their values (most values first), so the
    answer never depends on what the columns are called or where they stand."""
    from clinical_analytics.ui.components.variable_detector import VariableTypeDetector

    evidence = "together they are unique on every line"
    if aggregate is not None and aggregate.strata and aggregate.repeated_strata_lines == 0:
        return RowKey(kind="composite", columns=aggregate.strata, evidence=evidence)
    stats: dict[str, tuple[int, str, str]] = {
        c: (-df.get_column(c).n_unique(), str(df.schema[c]), str(df.get_column(c).head(3).to_list()))
        for c in df.columns
    }
    ordered: list[str] = sorted(df.columns, key=lambda c: stats[c])
    composite = VariableTypeDetector.find_composite_identifier_candidates(df.select(ordered))
    if not composite:
        return RowKey(kind="none")
    return RowKey(kind="composite", columns=tuple(c for c in df.columns if c in composite), evidence=evidence)


def _episode_keys(df: pl.DataFrame, key: str, ids: Sequence[_Candidate]) -> tuple[EpisodeKey, ...]:
    """IDs of records nested within one patient (a stay ID within a patient ID): more values than
    patients, and every value belongs to exactly one patient, so the records can be counted."""
    patients = int(df.get_column(key).drop_nulls().n_unique())
    return tuple(
        EpisodeKey(
            column=c.column, within=key, noun="record", source="structure", evidence=f"each value belongs to one {key}"
        )
        for c in ids
        if c.column != key and c.distinct > patients and _nested(df, c.column, key)
    )


# ----------------------------------------------------------------------------- the pass
def infer_semantics(df: pl.DataFrame, facts: UploadFacts | None = None) -> DatasetSemantics:
    """Every semantic fact about ``df`` with its source (see the module docstring).

    ``df`` is the table as staged, before ``apply_semantics`` and before the app adds an ID.
    """
    facts = facts or UploadFacts()
    dictionary = parse_dictionary(facts.doc_context, df.columns)
    staged = dict(facts.staged_dates or {})
    columns: dict[str, ColumnSemantics] = {}
    questions: list[OpenQuestion] = []
    date_columns: set[str] = set()
    numeric = {c for c, t in df.schema.items() if t.is_numeric() and t != pl.Boolean}
    baseline = {
        c for c in df.columns if (e := dictionary.entry(c)) is not None and _BASELINE_TEXT.search(e.description)
    }
    timepoints = _timepoints(df.columns, numeric, baseline)

    for name in df.columns:
        series = df.get_column(name)
        entry: DictionaryEntry | None = dictionary.entry(name)
        dtype = series.dtype
        logical = _logical_type(dtype)
        date_fmt: DateFormat | None = None
        if dtype.is_temporal():
            choice = staged.get(name)
            if choice is not None:
                date_fmt = _date_format(choice, "structure")
            else:
                date_fmt = _date_format(
                    DateChoice(name, NATIVE, "", "", isinstance(dtype, pl.Datetime), None, "the file stores dates"),
                    "structure",
                )
        elif dtype == pl.String and (entry is not None and entry.date_format or _date_like(name)):
            found = choose_text_format(series, stated=entry.date_format if entry else None)
            if isinstance(found, DateChoice):
                date_fmt = _date_format(
                    found, "dictionary" if found.evidence.startswith("the data dictionary") else "structure"
                )
                logical = "timestamp" if found.has_time else "date"
            elif isinstance(found, DateAmbiguity):
                questions.append(_date_question(name, found))
        elif dtype.is_numeric():
            # The dictionary saying the column is a date ("Date of first contact") counts like a name.
            said_date = entry is not None and bool(re.search(r"\bdates?\b", entry.description, re.IGNORECASE))
            choice = choose_number_format(
                series,
                from_excel=facts.from_excel,
                stated=entry.date_format if entry else None,
                date_hint=said_date,
            )
            if choice is not None:
                date_fmt = _date_format(
                    choice, "dictionary" if entry is not None and entry.date_format else "structure"
                )
                logical = "timestamp" if choice.has_time else "date"
        if date_fmt is not None:
            date_columns.add(name)

        missing: list[MissingCode] = []
        labels: dict[str, str] = {}
        labels_source: FactSource | None = None
        if entry is not None and date_fmt is None:
            for code, label in entry.missing.items():
                typed = _typed_code(code, dtype)
                if typed is not None:
                    missing.append(
                        MissingCode(
                            value=typed, label=label, source="dictionary", evidence="the data dictionary says so"
                        )
                    )
            if entry.labels:
                labels, labels_source = dict(entry.labels), "dictionary"
        if not labels:
            from clinical_analytics.analysis.variable_types import code_labels

            header = code_labels(name)
            if header:
                labels, labels_source = header, "structure"
        if date_fmt is None and dtype.is_numeric() and dtype != pl.Boolean:
            # A code stated in prose ("Coded fields: 99 = unknown", "ages of 90 or over were coded
            # as 99", a portal's description) is asked, never applied: only the column's own
            # code list applies a code (ADR017).
            present = {_float(v) for v in series.drop_nulls().cast(pl.Float64).unique().to_list()}
            notes = {
                **dictionary.shared_missing,
                **dictionary.noted_missing.get(name, {}),
                **(entry.noted_missing if entry else {}),
            }
            own = entry.noted_missing if entry else {}
            for code, note in notes.items():
                typed = _typed_code(code, dtype)
                # A shared note never overrides a category of the column's own code list; the
                # list's own "9 = Unknown primary" is asked about, kept as a category until then.
                listed = code in labels and code not in own
                if not isinstance(typed, int | float) or listed or float(typed) not in present:
                    continue
                if any(_same_code(m.value, typed) for m in missing):
                    continue
                questions.append(_noted_code_question(name, typed, note))
        if not missing and date_fmt is None and not labels:
            # Never stored as missing without the dictionary or the user: asked instead.
            asked = {q.id for q in questions}
            for suspect in _undocumented_missing(series, entry.stated_range if entry else None):
                if f"missing_code:{name}:{suspect.value}" in asked:
                    continue
                questions.append(
                    OpenQuestion(
                        id=f"missing_code:{name}:{suspect.value}",
                        kind="missing_code",
                        column=name,
                        value=suspect.value,
                        question=f"Does {suspect.value:g} in {name} mean not recorded?",
                        options=("yes, not recorded", "no, it is a real value"),
                    )
                )

        present_values = (
            {str(v) for v in series.drop_nulls().cast(pl.String).unique().to_list()} if entry is not None else set()
        )
        unit: Unit | None = None
        # A coded column ("1 = below target, 2 = in range") or a band of a measure ("sbp_target",
        # "bmi_cat") holds codes, not the measure's unit.
        name_parts = set(name_tokens(name))
        coded = bool(labels) or bool(name_parts & _CODED_TOKENS)
        if "target" in name_parts and dtype.is_integer():
            coded = coded or series.n_unique() <= 5  # "a1c_target" 1/2/3: a code, not a percentage
        if coded:
            pass
        elif entry is not None and entry.unit:
            unit = Unit(value=entry.unit, source="dictionary")
        elif date_fmt is None and (from_name := _unit_from_name(name)) and not _holds_years(series, from_name):
            unit = Unit(value=from_name, source="structure")
        columns[name] = ColumnSemantics(
            logical_type=logical,
            role="date" if date_fmt is not None else "value",
            date=date_fmt,
            missing_codes=tuple(missing),
            stated_range=(
                StatedRange(minimum=entry.stated_range[0], maximum=entry.stated_range[1], source="dictionary")
                if entry is not None and entry.stated_range
                else None
            ),
            unit=unit,
            labels=labels,
            labels_source=labels_source,
            timepoint=timepoints.get(name),
            direction=entry.direction if entry else None,
            direction_source="dictionary" if entry is not None and entry.direction else None,
            description=entry.description or None if entry else None,
            nonresponse_codes=_coded(entry.nonresponse if entry and labels else {}, dtype, present_values),
            censored_codes=tuple(
                CensoredCode(value=code.value, meaning=code.label, source="dictionary")
                for code in _coded(entry.censored if entry else {}, dtype, present_values)
            ),
        )
        for censored in columns[name].censored_codes:
            # A value that stands for a range is a real value; an answer reading it asks first.
            questions.append(
                OpenQuestion(
                    id=f"censored_value:{name}:{censored.value}",
                    kind="censored_value",
                    column=name,
                    value=censored.value,
                    question=(
                        f"In {name}, {censored.value} means “{censored.meaning}”. Answers that use {name} would count "
                        f"those lines as {censored.value}. Is that right for your questions?"
                    ),
                    options=(f"yes, use {censored.value} as given", "no, leave those lines out"),
                )
            )

    for value_column, unit_column in _unit_columns(df):
        # Durations in mixed units are read in one unit, the largest the file uses, by every reader
        # (contracts.readings): an average of 6 months and 2 years is never 4. Applied only when the
        # dictionary states the pair; otherwise asked (ADR017 principle A).
        stated = _unit_stated(dictionary, value_column, unit_column)
        questions.extend(_unit_pair_questions(df, value_column, unit_column, stated))
        if stated:
            unit = Unit(value=_largest_unit(df, unit_column), source="dictionary")
            columns[value_column] = columns[value_column].model_copy(update={"unit_column": unit_column, "unit": unit})
    weights = tuple(c for c in df.columns if config.SURVEY_WEIGHT_NAME.search(c.lower()))
    survey = bool(weights) or dictionary.mentions_weights

    published = facts.population_estimates or bool(facts.source_format)
    # Rows of a published table are estimates for groups: nothing in it is a patient key.
    # A coded item (its codes have labels) is a category: never a key, a sequence or a group.
    coded_items = {name for name, col in columns.items() if col.labels}
    keys = _Keys() if published else _choose_keys(df, dictionary, facts, excluded=date_columns | coded_items)
    if keys.question is not None:
        questions.insert(0, keys.question)
    roles: dict[str, str] = {}
    if keys.patient is not None:
        roles[keys.patient.column] = "patient_key"
    for col in keys.row.columns:
        roles.setdefault(col, "row_key")
    for s in keys.sequences:
        roles[s.column] = "sequence"
    for g in keys.groups:
        roles[g.column] = "grouping"
    # A key, a sequence or a grouping column is never asked about "not recorded" values.
    questions = [q for q in questions if not (q.kind == "missing_code" and q.column in roles)]
    for col, role in roles.items():
        current = columns[col]
        # A key or a sequence is never a measure: no missing codes, units or timepoints on it.
        columns[col] = current.model_copy(update={"role": role, "missing_codes": (), "unit": None, "timepoint": None})

    grain = _grain(df, dictionary, keys, facts)
    case_count: str | None = None
    aggregate: AggregateInfo | None = None
    summary: _Summary | str | None = None
    if keys.patient is None and not published:
        summary = _summary_table(df, date_columns, dictionary)
        if summary is not None:
            # Lines that may be groups: which column names a patient is not the question.
            questions = [q for q in questions if q.kind != "patient_key"]
        if isinstance(summary, _Summary):
            grain, aggregate, case_count = summary.grain, summary.aggregate, summary.aggregate.count_column
            questions.extend(summary.questions)
        elif isinstance(summary, str):
            # Looks like a table of groups but not recognised with confidence: asked, never
            # silently read as one line per patient.
            logger.info("summary_table_not_recognised", reason=summary)
            questions.append(_aggregate_question(_count_columns(df, date_columns, dictionary)[0], summary))
    if keys.patient is None and keys.row.kind == "none" and not facts.population_estimates:
        keys.row = _composite_row_key(df, aggregate)
    if (
        keys.patient is None
        and not facts.population_estimates
        and not facts.source_format
        and keys.row.kind in ("column", "none")
        and not any(q.kind == "patient_key" for q in questions)
        and grain.kind != "stratum"
    ):
        questions.insert(
            0,
            OpenQuestion(
                id=config.ROWS_QUESTION_ID,
                kind="rows_are_patients",
                question="Is each line one patient?",
                options=("yes, one patient per line", config.ROWS_ANSWER_RECORDS),
            ),
        )
    semantics = DatasetSemantics(
        rows=df.height,
        grain=grain,
        patient_key=keys.patient,
        row_key=keys.row,
        sequence_columns=keys.sequences,
        grouping_columns=keys.groups,
        episode_keys=keys.episodes,
        case_count_column=case_count,
        aggregate=aggregate,
        columns=columns,
        open_questions=tuple(questions),
        population_estimates=facts.population_estimates,
        source_format=facts.source_format,
        survey_sample=survey,
        survey_weight_columns=weights,
    )
    logger.info(
        "semantics_inferred",
        rows=df.height,
        columns=df.width,
        grain=grain.kind,
        grain_source=grain.source,
        has_patient_key=keys.patient is not None,
        key_source=keys.patient.source if keys.patient else None,
        sequences=len(keys.sequences),
        missing_code_columns=sum(1 for c in columns.values() if c.missing_codes),
        date_columns=len(date_columns),
        open_questions=len(questions),
    )
    return semantics


# Names that say a column counts something ("cases", "Deaths", "n", "number_of_events"): the usual
# published words. A count column without one is found by the structure (``_count_columns``).
COUNT_NAME_TOKENS: Final = frozenset(
    {"count", "counts", "n", "cases", "deaths", "events", "number", "total", "freq", "frequency"}
)  # fmt: skip
_COUNT_TEXT: Final = re.compile(r"\b(?:number|count|counts|total)\s+of\b|\bfrequency\b", re.IGNORECASE)


def _human(column: str) -> str:
    return " ".join(name_tokens(column)) or column


_RATE_TOKENS: Final = frozenset(
    {"rate", "rates", "pct", "percent", "percentage", "ratio", "crude", "adjusted", "prevalence", "incidence",
     "proportion", "share", "mean", "median"}
)  # fmt: skip
_BAND_TOKENS: Final = frozenset({"group", "grp", "band", "cat", "category", "class", "level", "bin"})
# Time columns are strata whatever their number of values (52 weeks, 20+ years).
_TIME_TOKENS: Final = frozenset(
    {"year", "yr", "years", "week", "wk", "weeks", "month", "months", "mo", "quarter", "qtr", "period"}
)  # fmt: skip
_TIME_TEXT: Final = re.compile(r"\b(?:week|month|quarter|year)\b", re.IGNORECASE)
_WHAT_WORDS: Final = ("deaths", "cases", "events")


@dataclass(frozen=True)
class _Summary:
    """A table of groups: its grain, what it holds, and what only the clinician can settle."""

    grain: Grain
    aggregate: AggregateInfo
    questions: tuple[OpenQuestion, ...] = ()


def _whole_nonnegative(series: pl.Series) -> bool:
    dtype = series.dtype
    if not (dtype.is_integer() or dtype.is_float()):
        return False
    values = series.drop_nulls()
    if values.len() == 0 or _float(values.min()) < 0:
        return False
    return dtype.is_integer() or bool((values == values.floor()).all())


def _count_columns(df: pl.DataFrame, dates: set[str], dictionary: Dictionary | None = None) -> list[str]:
    """Count columns of a table of groups: whole, non-negative numbers, named or described as
    counts ("cases", "Number of admissions"), or else found by structure: the only such column that
    the line's groups do not need, because the other columns (text, dates, IDs, years, codes
    with few values) already tell every line apart. A coded group (sex as 1/2) is needed to tell
    lines apart, so it is never the count, however few values it has."""
    whole = [c for c in df.columns if c not in dates and _whole_nonnegative(df.get_column(c))]
    said = [
        c
        for c in whole
        if set(name_tokens(c)) & COUNT_NAME_TOKENS
        or (
            dictionary is not None
            and (entry := dictionary.entry(c)) is not None
            and bool(_COUNT_TEXT.search(entry.description))
        )
    ]
    if said:
        return said
    numeric = [c for c in df.columns if df.schema[c].is_numeric() and df.schema[c] != pl.Boolean and c not in dates]
    found = []
    for c in whole:
        if id_named(c) or _rate_column(df, c, dictionary) or _year_like(df, c, dictionary):
            continue
        others = [
            o
            for o in df.columns
            if o != c and (o not in numeric or id_named(o) or _year_like(df, o, dictionary) or _few_codes(df, o))
        ]
        measures = [o for o in numeric if o != c and o not in others]
        # Groups repeat: a column with a different value on (nearly) every line names records.
        record_ids = [
            o
            for o in others
            if df.get_column(o).n_unique() in (df.height,) or df.get_column(o).n_unique() > max(60, df.height // 2)
        ]
        # The groups of a table cross each other: nearly every combination has its line.
        cells = math.prod(df.get_column(o).n_unique() for o in others) if others else 0
        if (
            others
            and not measures
            and not record_ids
            and cells <= 2 * df.height
            and not bool(df.select(pl.struct(others).is_duplicated().any()).item())
        ):
            found.append(c)
    # A coded group of a crossed table holds each of its values equally often; a count does not.
    found = [c for c in found if not _balanced(df.get_column(c))] if len(found) > 1 else found
    return found if len(found) == 1 else []


def _balanced(series: pl.Series) -> bool:
    """Every value occurs equally often (the levels of a crossed table: 1, 2, 3 on 8 lines each)."""
    counts = series.value_counts().get_column("count")
    return counts.len() > 1 and counts.n_unique() == 1


def _few_codes(df: pl.DataFrame, column: str) -> bool:
    return df.schema[column].is_integer() and df.get_column(column).n_unique() <= config.STRATUM_MAX_CODES


def _year_like(df: pl.DataFrame, column: str, dictionary: Dictionary | None) -> bool:
    """A period column (a year, a week, a month), by its name, its dictionary entry or its values."""
    if not df.schema[column].is_integer():
        return False
    entry = dictionary.entry(column) if dictionary is not None else None
    values = df.get_column(column).drop_nulls()
    return (
        (entry is not None and bool(_TIME_TEXT.search(entry.description or entry.text)))
        or bool(set(name_tokens(column)) & _TIME_TOKENS)
        or (values.len() > 0 and bool(values.is_between(config.DATE_YEAR_MIN, config.DATE_YEAR_MAX).all()))
    )


def _total_levels(
    df: pl.DataFrame, strata: Sequence[str], count: str
) -> tuple[tuple[TotalLevel, ...], tuple[tuple[str, str, str], ...]]:
    """(total rows to leave out of sums, (column, value, why) to ask about) of the strata.

    A value is a total only when two facts agree: its whole label is one of
    ``ESTIMATE_TOTAL_LEVELS`` ("Total", "All ages", "United States"), and its count is the sum of
    its column's other values on every line sharing its other groups. A label alone ("All other
    causes" is not a total label; an "Overall" whose sums disagree) or a sum alone (an unlabelled
    "Statewide", or a coincidence) is asked, never applied (ADR017).
    """
    applied: dict[str, list[str]] = {}
    asked: list[tuple[str, str, str]] = []
    for c in strata:
        if not isinstance(df.schema[c], pl.String | pl.Categorical | pl.Enum):
            continue
        levels = [
            v for v in df.get_column(c).cast(pl.String).drop_nulls().unique().sort().to_list() if isinstance(v, str)
        ]
        labelled = {v for v in levels if " ".join(v.lower().split()) in config.ESTIMATE_TOTAL_LEVELS}
        summed = set(_sum_totals(df, strata, count, c, levels))
        for v in sorted(labelled & summed):
            applied.setdefault(c, []).append(v)
        for v in sorted(labelled - summed):
            asked.append((c, v, "is named like a total, but its counts are not the sum of the other lines"))
        for v in sorted(summed - labelled):
            asked.append((c, v, "holds the sum of the other lines"))
    return tuple(TotalLevel(column=c, values=tuple(vs)) for c, vs in applied.items()), tuple(asked)


def _sum_totals(df: pl.DataFrame, strata: Sequence[str], count: str, column: str, levels: list[str]) -> list[str]:
    """Values of ``column`` whose count is the sum of its other values' counts, for every set of
    the other strata where it appears (at least two such sets, or one with three other values)."""
    if len(levels) < 3 or len(levels) > config.STRATUM_MAX_CODES * 3:
        return []
    others = [c for c in strata if c != column]
    keys = [pl.col(c).cast(pl.String).fill_null("\x00") for c in others]
    cells = (
        df.lazy()
        .with_columns(*keys, pl.lit(1).alias("__all"))
        .group_by(["__all", *others, column])
        .agg(pl.col(count).sum().alias("__n"))
        .collect()
    )
    on = ["__all", *others]
    found = []
    for v in levels:
        total = cells.filter(pl.col(column).cast(pl.String) == v).select(*on, pl.col("__n").alias("__t"))
        rest = (
            cells.filter((pl.col(column).cast(pl.String) != v) | pl.col(column).is_null())
            .group_by(on)
            .agg(pl.col("__n").sum().alias("__r"), pl.len().alias("__k"))
        )
        joined = total.join(rest, on=on, how="inner")
        if joined.height == 0 or _float(joined.get_column("__t").sum()) <= 0:
            continue
        lines = int(joined.get_column("__k").min() or 0)  # type: ignore[arg-type]
        if lines < 2 or (joined.height < 2 and lines < 3):
            continue
        if bool((joined.get_column("__t") == joined.get_column("__r")).all()):
            found.append(v)
    return found


def _rate_column(df: pl.DataFrame, column: str, dictionary: Dictionary | None) -> bool:
    """A numeric column of rates or percentages, by its name or its dictionary entry, whatever its
    type (an integer "rate_per_100k" or a "%" column is a value, never a group)."""
    if not df.schema[column].is_numeric() or df.schema[column] == pl.Boolean:
        return False
    tokens = set(name_tokens(column))
    if tokens & _BAND_TOKENS:
        return False  # "age_group_per_10y" names a band of ages, not a rate
    if tokens & _RATE_TOKENS or config.ESTIMATE_RATE_NAME.search(column.lower()) or "%" in column:
        return not bool(tokens & {"year", "yr"})
    entry = dictionary.entry(column) if dictionary is not None else None
    return entry is not None and bool(config.SUMMARY_RATE_TEXT.search(entry.description or entry.text))


def _stratum(df: pl.DataFrame, column: str, dates: set[str], loose: bool, dictionary: Dictionary | None = None) -> bool:
    """A low-cardinality category, a place, an ID of a group, or a period (a year, a week, a
    month: by its name or its dictionary entry, whatever its number of values): never a measure
    like age or SBP."""
    dtype = df.schema[column]
    series = df.get_column(column)
    if column in dates:
        return True
    if dtype == pl.String:
        distinct = series.n_unique()
        return distinct <= max(60, df.height // 2)
    if not dtype.is_integer():
        return False
    if id_named(column) or _year_like(df, column, dictionary):
        return True
    return series.n_unique() <= (200 if loose else config.STRATUM_MAX_CODES)


def _summary_table(df: pl.DataFrame, dates: set[str], dictionary: Dictionary | None = None) -> _Summary | str | None:
    """A table of groups (region x year x category with a count column, no person ID).

    Returns the summary; or a reason (str) when it looks like one (a count column, no person ID)
    but is not recognised with confidence; or None when it does not look like one (a record-level
    file: a measure such as age or SBP with many values beside the count).

    * frequency: each line counts cases for one stratum (the SUM of the count is "how many");
    * estimate: rates or percentages, or counts with total rows ("All", "United States"): never
      summed, a question reads the matching line.

    Strata repeated on several lines are kept and asked about, never a reason to reject. A
    dictionary that says the lines are counts for groups (``says_aggregate``) loosens the strata.
    """
    if df.height < 2:
        return None
    says_aggregate = dictionary is not None and dictionary.says_aggregate
    counts = _count_columns(df, dates, dictionary)
    if not counts:
        return None
    count = counts[0]
    values = [c for c in df.columns if c not in counts and _rate_column(df, c, dictionary)]
    strata = [
        c
        for c in df.columns
        if c not in counts and c not in values and _stratum(df, c, dates, says_aggregate, dictionary)
    ]
    measures = [c for c in df.columns if c not in counts and c not in values and c not in strata]
    many_valued = [
        c for c in measures if df.schema[c].is_numeric() and df.get_column(c).n_unique() > config.STRATUM_MAX_CODES
    ]
    if many_valued and not says_aggregate:
        return None  # a measure with many values beside the count: one line per record
    if not strata or not any(df.schema[c] == pl.String for c in strata):
        return f"{count} looks like a count, but no column of group names (a place, a year, a category) was found"
    if measures and not says_aggregate:
        return f"{count} looks like a count, but {', '.join(measures[:3])} are neither groups nor counts"
    rates = values
    values = [*values, *measures]
    repeated = int(df.select(pl.struct(strata).is_duplicated().sum()).item())
    if not says_aggregate and repeated > config.SUMMARY_MAX_REPEATED_SHARE * df.height:
        # A table of groups has few repeated groups; most lines repeating one are records (a
        # survey with banded answers and a visit count per respondent).
        return f"{repeated:,} of {df.height:,} lines repeat another line's groups, so they may be records, not groups"
    totals, asked_totals = _total_levels(df, strata, count)
    # Rates or percentages are read, never added; counts with their totals are added up without
    # the total lines.
    kind: Literal["frequency", "estimate"] = "estimate" if rates else "frequency"
    tokens = set(name_tokens(count))
    what = next((t for t in _WHAT_WORDS if t in tokens), "cases")
    names = [_human(c) for c in strata]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
    total_words = "; ".join(f"{t.column} = {', '.join(t.values)}" for t in totals)
    evidence = f"{count} counts {what} for each {listed}" + (f"; total rows: {total_words}" if totals else "")
    questions: list[OpenQuestion] = []
    if repeated:
        questions.append(
            OpenQuestion(
                id="repeated_strata",
                kind="repeated_strata",
                question=(
                    f"{repeated:,} lines repeat a group that another line also has (the same {listed}). "
                    "Should their counts be added together?"
                ),
                options=("yes, add them together", "no, one of them replaces the other"),
            )
        )
        logger.info("summary_table_repeated_strata", lines=repeated, strata=len(strata))
    for column, level, why in asked_totals:
        questions.append(
            OpenQuestion(
                id=f"total_row:{column}:{level}",
                kind="total_row",
                column=column,
                value=level,
                question=f"In {column}, {level} {why}. Is {level} a total of the other {column} lines?",
                options=("yes, it is a total", "no, it is one more group"),
            )
        )
    if asked_totals:
        logger.info("summary_table_totals_asked", totals=len(asked_totals))
    # The structure suggests a table of groups; only the user's answer applies it (ADR017).
    questions.insert(0, _aggregate_question(count, f"each line seems to count {what} for one {listed}"))
    sentence = (
        f"Each line seems to count {what} for one {listed} (to confirm)."
        if kind == "frequency"
        else f"Each line seems to give published {what} for one {listed} (to confirm); lines are read, never added."
    )
    return _Summary(
        grain=Grain(
            kind="stratum",
            noun=f"count of {what}",
            one_row_per_patient=False,
            rows=df.height,
            source="structure",
            evidence=evidence,
            sentence=sentence,
        ),
        aggregate=AggregateInfo(
            kind=kind,
            count_column=count,
            value_columns=tuple(values),
            strata=tuple(strata),
            total_levels=totals,
            repeated_strata_lines=repeated,
            source="dictionary" if says_aggregate else "structure",
            evidence=evidence,
        ),
        questions=tuple(questions),
    )


def _aggregate_question(count_column: str, why: str) -> OpenQuestion:
    return OpenQuestion(
        id="aggregate_unsure",
        kind="aggregate_unsure",
        column=count_column,
        question=(
            f"Does each line count cases for a group (in {count_column}), rather than describe one patient? "
            f"({why[:1].upper()}{why[1:]}.)"
        ),
        options=("yes, each line counts cases for a group", "no, each line is about one patient"),
    )


def _date_like(column: str) -> bool:
    from clinical_analytics.contracts.dates import date_named

    return date_named(column)


def _grain(df: pl.DataFrame, dictionary: Dictionary, keys: _Keys, facts: UploadFacts) -> Grain:
    rows = df.height
    key = keys.patient
    patients = int(df.get_column(key.column).drop_nulls().n_unique()) if key is not None else None
    if facts.population_estimates or facts.source_format:
        return Grain(
            kind="estimate",
            noun="published estimate",
            one_row_per_patient=False,
            rows=rows,
            source="declared" if facts.population_estimates else "structure",
            evidence="rows are estimates for groups",
        )
    # One row per patient: no ID repeats among the lines that have one (a missing ID is reported by
    # the contract's not-null check, not read as a visit).
    present = df.get_column(key.column).drop_nulls() if key is not None else None
    one_per = present is not None and present.len() > 0 and patients == present.len()
    if dictionary.grain_noun:
        noun = dictionary.grain_noun
        kind = grain_kind(noun)
        if kind == "patient":
            noun = "patient"
        return Grain(
            kind=kind,
            noun=noun,
            one_row_per_patient=one_per or (kind == "patient" and key is not None),
            rows=rows,
            patients=patients,
            source="dictionary",
            evidence=f"the data dictionary says “{dictionary.grain_text}”",
        )
    if key is not None and one_per:
        return Grain(
            kind="patient",
            noun="patient",
            one_row_per_patient=True,
            rows=rows,
            patients=patients,
            source="structure",
            evidence=f"{key.column} is on every line once",
        )
    if key is not None:
        # What a line is comes from the dictionary ("One row per clinic attendance"); the
        # structure only says that the key repeats.
        return Grain(
            kind="record",
            noun="record",
            one_row_per_patient=False,
            rows=rows,
            patients=patients,
            source="default",
            evidence=f"{key.column} repeats across lines",
        )
    return Grain(
        kind="record",
        noun="record",
        one_row_per_patient=False,
        rows=rows,
        patients=None,
        source="default",
        evidence="no column names a patient",
    )


# ----------------------------------------------------------------------------- applying it
def apply_semantics(df: pl.DataFrame, semantics: DatasetSemantics) -> pl.DataFrame:
    """``df`` with missing-value codes stored as missing and dates read with their one format.

    Runs where the typed Parquet is written, so every reader sees the same values. Safe to run
    twice: a code already missing stays missing, a column already of date type is left alone.

    Raises:
        polars.exceptions.InvalidOperationError: a date value that the chosen format does not
            read (the format was chosen over these same values, so this is a bug, never data loss).
    """
    exprs: list[pl.Expr] = []
    for name, col in semantics.columns.items():
        if name not in df.columns:
            continue
        dtype = df.schema[name]
        expr: pl.Expr | None = None
        if col.missing_codes and not dtype.is_temporal():
            # Codes as the column's own type: an integer 99 against a Float64 column, or the reverse.
            codes = pl.Series([str(m.value) for m in col.missing_codes]).cast(dtype, strict=False).drop_nulls()
            if dtype.is_numeric() and codes.len() < len(col.missing_codes):
                codes = pl.Series([float(m.value) for m in col.missing_codes if _is_number(m.value)]).cast(dtype)
            if codes.len():
                expr = pl.when(pl.col(name).is_in(codes.implode())).then(None).otherwise(pl.col(name)).alias(name)
        if col.date is not None and col.date.format != NATIVE and not dtype.is_temporal():
            from clinical_analytics.contracts.dates import parse_expr

            if col.date.format in (YYYYMMDD, EXCEL_SERIAL) or dtype == pl.String:
                expr = parse_expr(name, col.date.format, col.date.has_time, dtype)
        if expr is not None:
            exprs.append(expr)
    return df.lazy().with_columns(exprs).collect() if exprs else df


def add_id_columns(
    df: pl.DataFrame, semantics: DatasetSemantics
) -> tuple[pl.DataFrame, dict[str, Any], DatasetSemantics]:
    """Add the app's ID column and record it in the semantics as an app column.

    * a patient key or a unique row key column: copied to ``patient_id`` (the file's column stays);
    * a combination unique on every line: hashed into ``patient_id``;
    * nothing: lines numbered 1..n in ``row_number``.

    A file that already has ``patient_id`` gets nothing added. Returns the frame, the upload's
    ``synthetic_id_metadata.patient_id`` record, and the semantics with ``app_columns`` set.
    """
    from clinical_analytics.core.identifier_rules import REPEATED_ID_COLUMNS_KEY

    repeated = [g.column for g in semantics.grouping_columns]
    if semantics.population_estimates:
        return _numbered(df, semantics, {"population_estimates": True})
    if PATIENT_ID_COLUMN in df.columns:
        return df, {"patient_id_source": "existing", "patient_id_columns": None}, semantics
    source_col = semantics.patient_key_column or (
        semantics.row_key.columns[0] if semantics.row_key.kind == "column" else None
    )
    if source_col is not None:
        added = AppColumn(column=PATIENT_ID_COLUMN, kind="copy", made_from=(source_col,))
        meta: dict[str, Any] = {"patient_id_source": "single_column", "patient_id_columns": [source_col]}
        if semantics.patient_key is None:
            meta[REPEATED_ID_COLUMNS_KEY] = repeated
        return (
            df.with_columns(pl.col(source_col).cast(pl.String).alias(PATIENT_ID_COLUMN)),
            meta,
            _with_app_column(semantics, added),
        )
    if semantics.row_key.kind == "composite":
        from clinical_analytics.ui.components.variable_detector import VariableTypeDetector

        parts = list(semantics.row_key.columns)
        added = AppColumn(column=PATIENT_ID_COLUMN, kind="hash", made_from=tuple(parts))
        return (
            VariableTypeDetector.create_synthetic_patient_id(df, parts),
            {"patient_id_source": "composite", "patient_id_columns": parts},
            _with_app_column(semantics, added),
        )
    return _numbered(df, semantics, {REPEATED_ID_COLUMNS_KEY: repeated})


def _numbered(
    df: pl.DataFrame, semantics: DatasetSemantics, extra: dict[str, Any]
) -> tuple[pl.DataFrame, dict[str, Any], DatasetSemantics]:
    """Lines numbered 1..n in ``row_number`` (``_row_number`` if the file has that name)."""
    row_column = ROW_NUMBER_COLUMN
    while row_column in df.columns:
        row_column = f"_{row_column}"
    added = AppColumn(column=row_column, kind="row_number")
    meta = {"patient_id_source": "row_index", "patient_id_columns": None, "row_number_column": row_column, **extra}
    return (
        df.with_row_index(row_column, offset=1).with_columns(pl.col(row_column).cast(pl.Int64)),
        meta,
        _with_app_column(semantics, added),
    )


def _with_app_column(semantics: DatasetSemantics, added: AppColumn) -> DatasetSemantics:
    """``semantics`` with ``added`` recorded as an app column and as a column of its own."""
    logical = "integer" if added.kind == "row_number" else "string"
    columns = {**semantics.columns, added.column: ColumnSemantics(logical_type=logical, role="app_added")}
    return semantics.model_copy(update={"app_columns": (*semantics.app_columns, added), "columns": columns})

# SPDX-License-Identifier: MIT
"""The one inference pass at upload: fill in the contract's semantics section (ADR017).

Inputs are the staged table (typed by ``storage.staging``, which reads text dates with the same
one-format rule, ``contracts.dates``), the upload's documentation text, and facts about the
upload (a workbook, a published export, declared estimates). Every fact records its source:

* **dictionary**: the data dictionary says it ("subj is the patient identifier", "99 = not done",
  "(0-27)", "MM/DD/YYYY HH:MM", "One row per ED encounter").
* **structure**: the values show it. A column whose values repeat while another column numbers
  1..k within each value is a patient key and a sequence; an ID named for the patient role
  (``pt_id``, ``subj``, ``patient_ref``, ``case_id``) is the key; an all-unique ID named for a
  person (``infant_id``) on a file of one row per person is the row key and names that person;
  an ID that repeats for another role (a mother, a household, a site, a surgeon) groups rows.
* **default**: nothing says more ("Each line is one record").

What only a clinician can settle is an open question in clinical words ("Is each line one baby or
one mother?"), never a guess. Then ``apply_semantics`` writes missing-value codes as missing and
reads dates with their one format, at the point where the typed Parquet is written, and
``add_id_columns`` adds the app's own ID column (``patient_id``, a copy of the key; or a row
number), recorded as an app column and never as a patient identifier.

Deterministic: the same table and documentation always give the same semantics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final

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
    AppColumn,
    ColumnSemantics,
    DatasetSemantics,
    DateFormat,
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
    Unit,
)

logger = structlog.get_logger()

PATIENT_ID_COLUMN: Final = "patient_id"
ROW_NUMBER_COLUMN: Final = "row_number"

# Name tokens, after splitting on anything but letters and digits and at a case change.
_SPLIT: Final = re.compile(r"[^a-z0-9]+")
_CAMEL: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
ID_TOKENS: Final = frozenset(
    {"id", "ids", "identifier", "mrn", "mr", "uuid", "guid", "no", "nr", "num", "number", "ref", "key"}
)
# Who an ID names. Strong patient words name the patient on their own ("subj", "MRN"); weak ones
# only beside an ID word ("pt_id", "case_id", "member_id", "study_no"): "PT" alone is a lab test,
# "case" alone a case-control flag.
PATIENT_ROLE_TOKENS: Final = frozenset(
    {
        "patient",
        "subject",
        "subj",
        "participant",
        "mrn",
        "pid",
        "ptid",
        "patid",
        "patientid",
        "patientno",
        "patientnumber",
        "subjid",
        "subjectid",
        "subjectno",
        "participantid",
        "participantno",
        "medicalrecordnumber",
        "studyid",
        "memberid",
        "caseid",
    }
)
WEAK_PATIENT_TOKENS: Final = frozenset(
    {"pt", "pat", "member", "mbr", "person", "case", "client", "resident", "enrollee", "study"}
)
PERSON_ROLE_TOKENS: Final[Mapping[str, str]] = {
    "infant": "infant",
    "baby": "infant",
    "neonate": "infant",
    "newborn": "infant",
    "nb": "infant",
    "mother": "mother",
    "mat": "mother",
    "maternal": "mother",
    "mom": "mother",
    "mum": "mother",
}
GROUP_ROLE_TOKENS: Final[Mapping[str, str]] = {
    "site": "site",
    "center": "site",
    "centre": "site",
    "hospital": "hospital",
    "hosp": "hospital",
    "clinic": "clinic",
    "ward": "ward",
    "practice": "practice",
    "facility": "facility",
    "fac": "facility",
    "household": "household",
    "hh": "household",
    "family": "family",
    "fam": "family",
    "surgeon": "surgeon",
    "provider": "provider",
    "physician": "physician",
    "doctor": "doctor",
    "dr": "doctor",
    "nurse": "nurse",
    "prescriber": "prescriber",
    "clinician": "clinician",
    "team": "team",
    "cluster": "cluster",
    "village": "village",
    "district": "district",
    "region": "region",
    "school": "school",
    # Dimensions of published tables (GBD, CDC): an ID for a place, a cause or a year names no one.
    "location": "place",
    "loc": "place",
    "state": "place",
    "county": "place",
    "country": "place",
    "cause": "category",
    "measure": "category",
    "metric": "category",
    "sex": "category",
    "year": "category",
    "age": "category",
    "race": "category",
    "indicator": "category",
    "topic": "category",
    "stratification": "category",
    "category": "category",
    "type": "category",
    "batch": "batch",
    "device": "device",
    "twin": "twin pair",
    "pair": "pair",
}
# What a row is, by the word that names it (in an ID, a sequence, a date or the dictionary).
EVENT_TOKENS: Final[Mapping[str, str]] = {
    "enc": "encounter",
    "encounter": "encounter",
    "visit": "visit",
    "vis": "visit",
    "admission": "admission",
    "adm": "admission",
    "admit": "admission",
    "episode": "episode",
    "ep": "episode",
    "session": "session",
    "sess": "session",
    "cycle": "cycle",
    "cyc": "cycle",
    "scan": "scan",
    "test": "test",
    "draw": "lab draw",
    "sample": "sample",
    "specimen": "specimen",
    "order": "order",
    "rx": "prescription",
    "prescription": "prescription",
    "event": "event",
    "record": "record",
    "rec": "record",
    "obs": "observation",
    "observation": "observation",
    "measurement": "measurement",
    "reading": "reading",
    "proc": "procedure",
    "procedure": "procedure",
    "surgery": "operation",
    "op": "operation",
    "stay": "stay",
    "call": "call",
    "appt": "appointment",
    "appointment": "appointment",
    "delivery": "delivery",
    "del": "delivery",
    "birth": "birth",
    "pregnancy": "pregnancy",
    "preg": "pregnancy",
    "infusion": "infusion",
    "dose": "dose",
    "round": "round",
    "contact": "contact",
    "fu": "follow-up visit",
    "followup": "follow-up visit",
}
GRAIN_KIND_WORDS: Final[Mapping[str, str]] = {
    **EVENT_TOKENS,
    "patient": "patient",
    "patients": "patient",
    "person": "patient",
    "participant": "patient",
    "subject": "patient",
    "member": "patient",
    "woman": "patient",
    "man": "patient",
    "child": "patient",
    "mother": "mother",
    "infant": "infant",
    "baby": "infant",
    "newborn": "infant",
    "neonate": "infant",
    "result": "test",
    "draws": "lab draw",
    "estimate": "estimate",
}
SEQUENCE_TOKENS: Final = frozenset({"no", "num", "number", "nr", "seq", "sequence", "order", "n", "index", "idx", "k"})
# Timepoint words in a measure's name, ordered: their rank orders the times of one measure.
_TIMEPOINT_RANK: Final[Mapping[str, int]] = {
    "screen": -2,
    "screening": -2,
    "pre": -1,
    "preop": -1,
    "before": -1,
    "baseline": 0,
    "bl": 0,
    "base": 0,
    "initial": 0,
    "init": 0,
    "first": 0,
    "admit": 0,
    "start": 0,
    "intake": 0,
    "post": 9000,
    "postop": 9000,
    "after": 9000,
    "discharge": 9500,
    "dc": 9500,
    "end": 9900,
    "final": 9900,
    "last": 9900,
    "exit": 9900,
    "followup": 5000,
    "fu": 5000,
}
_TIME_NUMBER: Final = re.compile(
    r"^(?:(?P<pre>m|mo|month|w|wk|week|d|day|y|yr|year|t|v|visit)(?P<n1>\d{1,3})|(?P<n2>\d{1,3})(?P<post>m|mo|mos|w|wk|wks|d|y|yr|h|hr))$"
)
_TIME_DAYS: Final[Mapping[str, int]] = {
    "m": 30,
    "mo": 30,
    "mos": 30,
    "month": 30,
    "w": 7,
    "wk": 7,
    "wks": 7,
    "week": 7,
    "d": 1,
    "day": 1,
    "y": 365,
    "yr": 365,
    "year": 365,
    "t": 1,
    "v": 1,
    "visit": 1,
    "h": 0,
    "hr": 0,
}
# Units written as the last name tokens ("_kg", "_mg_wk", "_cmH2O", "_ng_ml", "_g_cm2", "_pct").
_NAME_UNITS: Final[Mapping[tuple[str, ...], str]] = {
    ("ng", "ml"): "ng/mL",
    ("mg", "dl"): "mg/dL",
    ("g", "dl"): "g/dL",
    ("g", "l"): "g/L",
    ("mmol", "l"): "mmol/L",
    ("umol", "l"): "µmol/L",
    ("mg", "l"): "mg/L",
    ("u", "l"): "U/L",
    ("iu", "l"): "IU/L",
    ("g", "cm2"): "g/cm²",
    ("kg", "m2"): "kg/m²",
    ("mg", "wk"): "mg/week",
    ("mg", "d"): "mg/day",
    ("mg", "day"): "mg/day",
    ("mg", "kg"): "mg/kg",
    ("ml", "min"): "mL/min",
    ("l", "min"): "L/min",
    ("ml", "kg"): "mL/kg",
    ("cmh2o",): "cmH2O",
    ("cm", "h2o"): "cmH2O",
    ("mmhg",): "mmHg",
    ("kg",): "kg",
    ("g",): "g",
    ("mg",): "mg",
    ("mcg",): "mcg",
    ("ug",): "µg",
    ("pct",): "%",
    ("percent",): "%",
    ("ml",): "mL",
    ("l",): "L",
    ("cm",): "cm",
    ("mm",): "mm",
    ("min",): "minutes",
    ("mins",): "minutes",
    ("hr",): "hours",
    ("hrs",): "hours",
    ("h",): "hours",
    ("d",): "days",
    ("days",): "days",
    ("wk",): "weeks",
    ("wks",): "weeks",
    ("weeks",): "weeks",
    ("yrs",): "years",
    ("yr",): "years",
    ("years",): "years",
    ("bpm",): "beats/min",
    ("sec",): "seconds",
    ("copies",): "copies/mL",
    ("lbs",): "lb",
    ("lb",): "lb",
    ("in",): "in",
    ("iu",): "IU",
}
_EXAMPLE_DAY: Final = datetime(2021, 4, 3, 14, 5)
_MAX_KEY_CANDIDATES: Final = 6


def name_tokens(column: str) -> list[str]:
    """'PtID' -> ['pt', 'id']; 'weight_preop_kg' -> ['weight', 'preop', 'kg']."""
    return [t for t in _SPLIT.split(_CAMEL.sub("_", column).lower()) if t]


def person_key_role(column: str) -> str | None:
    """'patient' for a name of the patient-ID role, 'infant'/'mother' for another person, else None."""
    tokens = name_tokens(column)
    joined = "".join(tokens)
    if joined in PATIENT_ROLE_TOKENS or any(t in PATIENT_ROLE_TOKENS for t in tokens):
        return "patient"
    if any(t in WEAK_PATIENT_TOKENS for t in tokens) and any(t in ID_TOKENS for t in tokens):
        return "patient"
    return next((PERSON_ROLE_TOKENS[t] for t in tokens if t in PERSON_ROLE_TOKENS), None)


# Among several columns named for the patient, the most specific name wins ("patient_id" beside
# "subject_id"), then a repeating one (visits of a patient), then column order.
_PATIENT_NAME_RANK: Final[tuple[frozenset[str], ...]] = (
    frozenset({"patient", "patientid", "patientno", "patientnumber", "pt", "ptid", "pat", "patid"}),
    frozenset({"subject", "subj", "subjid", "subjectid", "subjectno"}),
    frozenset({"participant", "participantid", "participantno"}),
    frozenset({"mrn", "medicalrecordnumber"}),
)


def patient_name_rank(column: str) -> int:
    """0 for a "patient" name, then subject, participant, MRN; 4 for any other patient word."""
    tokens = set(name_tokens(column)) | {"".join(name_tokens(column))}
    return next((rank for rank, words in enumerate(_PATIENT_NAME_RANK) if tokens & words), len(_PATIENT_NAME_RANK))


def names_patient(column: str) -> bool:
    """A column named for the patient-ID role (``pt_id``, ``subj``, ``MRN``, ``study_no``)."""
    return person_key_role(column) == "patient"


def _group_role(column: str) -> str | None:
    return next((GROUP_ROLE_TOKENS[t] for t in name_tokens(column) if t in GROUP_ROLE_TOKENS), None)


def _event_word(column: str) -> str | None:
    return next((EVENT_TOKENS[t] for t in name_tokens(column) if t in EVENT_TOKENS), None)


def id_named(column: str) -> bool:
    return any(t in ID_TOKENS for t in name_tokens(column)) or person_key_role(column) is not None


def grain_kind(noun: str) -> str:
    words = [w for w in re.split(r"[^a-z]+", noun.lower()) if w]
    for word in reversed(words):
        if word in GRAIN_KIND_WORDS:
            return GRAIN_KIND_WORDS[word]
        if word.endswith("s") and word[:-1] in GRAIN_KIND_WORDS:
            return GRAIN_KIND_WORDS[word[:-1]]
    return "record"


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
    if dtype == pl.String:
        return code
    if dtype.is_integer() and re.fullmatch(r"-?\d+", code):
        return int(code)
    if dtype.is_float() and re.fullmatch(r"-?\d+(?:\.\d+)?", code):
        value = float(code)
        return int(value) if value.is_integer() else value
    return None


def _undocumented_missing(series: pl.Series, stated: tuple[float, float] | None) -> MissingCode | None:
    """One repeated value from ``MISSING_CODE_CANDIDATES`` far outside the column's other values."""
    if not series.dtype.is_numeric() or series.dtype == pl.Boolean:
        return None
    values = series.drop_nulls()
    if values.len() < 3:
        return None
    floats = values.cast(pl.Float64)
    top, bottom = _float(floats.max()), _float(floats.min())
    for candidate in (top, bottom):
        if not candidate.is_integer() or int(candidate) not in config.MISSING_CODE_CANDIDATES:
            continue
        hits = int((floats == candidate).sum())
        rest = floats.filter(floats != candidate)
        if hits < config.MISSING_CODE_MIN_ROWS or rest.len() == 0:
            continue
        if candidate < 0:
            far = _float(rest.min()) >= 0
        elif stated is not None:
            far = candidate > stated[1] and _float(rest.max()) <= stated[1] * config.MISSING_CODE_GAP
        else:
            far = candidate > max(_float(rest.max()), 0.0) * config.MISSING_CODE_GAP
        if far:
            value: int | float = int(candidate) if series.dtype.is_integer() else candidate
            side = "below every other value" if candidate < 0 else "far above every other value"
            return MissingCode(value=value, source="structure", evidence=f"{_n(hits)} lines hold {value:g}, {side}")
    return None


def _float(value: object) -> float:
    """A Series aggregate (min/max of a Float64 series) as a float."""
    if not isinstance(value, int | float):
        raise TypeError(f"Expected a number, got {type(value).__name__}")
    return float(value)


def _n(count: int) -> str:
    return f"{count:,}"


def _unit_from_name(column: str) -> str | None:
    tokens = [t for t in _SPLIT.split(column.lower()) if t]  # no case split: "cmH2O" is one unit
    if len(tokens) < 2:
        return None
    for width in (2, 1):
        tail = tuple(tokens[-width:])
        if len(tokens) > width and tail in _NAME_UNITS:
            return _NAME_UNITS[tail]
    return None


def _timepoint_token(token: str) -> tuple[str, int] | None:
    if token in _TIMEPOINT_RANK:
        return token, _TIMEPOINT_RANK[token]
    m = _TIME_NUMBER.match(token)
    if m is None:
        return None
    unit = m.group("pre") or m.group("post")
    number = int(m.group("n1") or m.group("n2"))
    return token, number * _TIME_DAYS[unit]


def _timepoints(columns: Iterable[str], numeric: set[str]) -> dict[str, Timepoint]:
    """Measures stored at two or more times: same name but for one timepoint word."""
    groups: dict[str, list[tuple[str, str, int]]] = {}
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
    role: str | None  # patient, infant, mother
    group: str | None
    event: str | None
    named: bool  # named as an ID


def _candidates(df: pl.DataFrame, excluded: set[str]) -> list[_Candidate]:
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
        if distinct < 2 or not (named or text_like):
            continue
        found.append(
            _Candidate(
                column=c,
                distinct=distinct,
                nulls=nulls,
                unique=nulls == 0 and distinct == rows,
                role=person_key_role(c),
                group=_group_role(c),
                event=_event_word(c),
                named=named,
            )
        )
    return found


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
        needed = config.SEQUENCE_MIN_SHARE if _sequence_named(column) else config.SEQUENCE_UNNAMED_MIN_SHARE
        if g["groups"] and g["ok"] / g["groups"] >= needed:
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


def _sequence_named(column: str) -> bool:
    tokens = name_tokens(column)
    return any(t in SEQUENCE_TOKENS or t in EVENT_TOKENS for t in tokens)


@dataclass
class _Keys:
    patient: PatientKey | None = None
    row: RowKey = RowKey(kind="none")
    sequences: tuple[SequenceColumn, ...] = ()
    groups: tuple[GroupingColumn, ...] = ()
    question: OpenQuestion | None = None


def _seq_noun(column: str, fallback: str) -> str:
    word = _event_word(column)
    return word if word is not None else fallback


def _choose_keys(df: pl.DataFrame, dictionary: Dictionary, facts: UploadFacts, excluded: set[str]) -> _Keys:
    """Patient key, row key, sequences and grouping columns, in this order of evidence:

    1. the dictionary names the patient identifier; 2. the cohort was joined on it; 3. an ID named
    for the patient role; 4. an ID whose values repeat while another column numbers 1..k within
    each; 5. an ID named for a person (a baby, a mother); 6. any ID whose values repeat; 7. a
    unique ID on a file the dictionary says has one line per patient. Group, encounter and
    sequence IDs are never the patient.
    """
    rows = df.height
    # A non-unique ID with two values is a flag or a pair of sites, not patients (unless the
    # dictionary or a patient-role name says it is the patient).
    candidates = [
        c
        for c in _candidates(df, excluded)
        if (c.unique or c.distinct > 2 or c.role == "patient" or c.column in dictionary.key_statements)
        # A key is on most lines: an ID missing on more than half of them names no one reliably.
        and (c.unique or c.nulls * 2 <= rows)
    ]
    by_name = {c.column: c for c in candidates}
    stated_names = (*dictionary.key_statements, *(e.column for e in dictionary.entries.values() if e.names_patient_key))
    stated = [c for c in stated_names if c in by_name]
    seq_cols = _sequence_columns(df, excluded)
    keys = _Keys()
    chosen: _Candidate | None = None
    source: FactSource = "structure"
    confidence: str = "high"
    evidence = ""
    sequences: list[tuple[str, int]] = []
    order = {c: i for i, c in enumerate(df.columns)}

    def nested(c: _Candidate) -> list[tuple[str, int]]:
        return [] if c.unique else _sequence_within(df, c.column, [s for s in seq_cols if s != c.column])

    def free(c: _Candidate) -> bool:
        return c.group is None and c.event is None

    if stated:
        chosen, source = by_name[stated[0]], "dictionary"
        evidence = "the data dictionary names it as the patient identifier"
    elif facts.cohort_key is not None and facts.cohort_key in by_name:
        chosen, evidence = by_name[facts.cohort_key], "the tables were joined on it, one patient each"
    if chosen is None:
        named = sorted(
            (c for c in candidates if c.role == "patient" and free(c)),
            key=lambda c: (patient_name_rank(c.column), c.unique, order[c.column]),
        )
        if named:
            chosen = named[0]
            evidence = f"{chosen.column} is named for the patient"
    if chosen is None:
        unknown = sorted(
            (c for c in candidates if c.role is None and free(c) and not c.unique),
            key=lambda c: (not c.named, -c.distinct, order[c.column]),
        )[:_MAX_KEY_CANDIDATES]
        for c in unknown:
            found = nested(c)
            if found:
                chosen, sequences = c, found
                evidence = f"its values repeat while {found[0][0]} numbers 1, 2, 3… within each"
                break
    if chosen is None:
        people = sorted(
            (c for c in candidates if c.role in ("infant", "mother") and c.group is None), key=lambda c: order[c.column]
        )
        infants = [c for c in people if c.role == "infant"]
        mothers = [c for c in people if c.role == "mother"]
        if infants and mothers:
            chosen = next((c for c in infants if c.unique), infants[0])
            confidence = "medium"
            evidence = f"{chosen.column} names a baby on every line; {mothers[0].column} repeats for a mother"
            keys.question = OpenQuestion(
                id="person_role",
                kind="person_role",
                question="Is each line one baby or one mother?",
                column=chosen.column,
                options=("one baby", "one mother"),
            )
        elif people:
            chosen = people[0]
            confidence = "medium"
            evidence = f"{chosen.column} is named for a {chosen.role}"
    if chosen is None:
        # A repeating ID that names no role: a patient only when it has many values (at least
        # GENERIC_KEY_MIN_SHARE of the lines), as patients with a few visits each do.
        generic = [
            c
            for c in candidates
            if c.named
            and free(c)
            and not c.unique
            and c.role is None
            and c.distinct >= config.GENERIC_KEY_MIN_SHARE * rows
        ]
        if generic:
            chosen = max(generic, key=lambda c: (c.distinct, -order[c.column]))
            confidence = "medium"
            evidence = f"{chosen.column} is an ID whose values repeat across lines"
    unique_ids = [c for c in candidates if c.unique and c.named and c.group is None]
    says_patient = dictionary.one_per_person or grain_kind(dictionary.grain_noun or "") == "patient"
    if chosen is None and unique_ids and says_patient:
        chosen = sorted(unique_ids, key=lambda c: order[c.column])[0]
        source, evidence = (
            "dictionary",
            "the data dictionary says each line is one patient, and it is on every line once",
        )
    if chosen is not None:
        if not sequences:
            sequences = nested(chosen)
        keys.patient = PatientKey(
            column=chosen.column,
            role=chosen.role or "patient",
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
                noun=_seq_noun(s, "line"),
                source="structure",
                evidence=f"it numbers the lines of each {chosen.column} from {start}",
            )
            for s, start in sequences
        )
    seq_names = {s.column for s in keys.sequences}
    row_ids = [c for c in candidates if c.unique and c.column not in seq_names and c.group is None]
    if keys.patient is not None and keys.patient.unique:
        keys.row = RowKey(kind="column", columns=(keys.patient.column,), evidence="it is on every line once")
    elif row_ids:
        best = sorted(row_ids, key=lambda c: (c.event is None, not c.named, order[c.column]))[0]
        keys.row = RowKey(kind="column", columns=(best.column,), evidence="it is on every line once")
    groups: list[GroupingColumn] = []
    for c in candidates:
        if (keys.patient is not None and c.column == keys.patient.column) or c.unique or c.column in seq_names:
            continue
        if c.group is None and (not c.named or c.event is not None):
            continue  # an encounter ID on several lines of one encounter groups no people
        if not c.named and c.group in ("category", "place"):
            continue  # a dimension of the table (sex, year, state), not an ID for a role
        role = c.group or (c.role if c.role not in (None, "patient") else None) or "group"
        groups.append(
            GroupingColumn(column=c.column, role=role, source="structure", evidence="its values repeat across lines")
        )
    keys.groups = tuple(groups)
    return keys


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
    timepoints = _timepoints(df.columns, numeric)

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
                ex_in, _ = _synthetic_example("%d/%m/%y", False)
                questions.append(
                    OpenQuestion(
                        id=f"date_order:{name}",
                        kind="date_order",
                        column=name,
                        question=f"In {name}, does {ex_in} mean 3 April 2021 or March 4, 2021?",
                        options=("day first (3 April)", "month first (March 4)"),
                    )
                )
        elif dtype.is_numeric():
            choice = choose_number_format(
                series, from_excel=facts.from_excel, stated=entry.date_format if entry else None
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
        if entry is not None:
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
        if not missing and date_fmt is None and name not in labels and not labels:
            found_code = _undocumented_missing(series, entry.stated_range if entry else None)
            if found_code is not None:
                missing.append(found_code)

        unit: Unit | None = None
        if entry is not None and entry.unit:
            unit = Unit(value=entry.unit, source="dictionary")
        elif date_fmt is None and (from_name := _unit_from_name(name)):
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
        )

    published = facts.population_estimates or bool(facts.source_format)
    # Rows of a published table are estimates for groups: nothing in it is a patient key.
    keys = _Keys() if published else _choose_keys(df, dictionary, facts, excluded=date_columns)
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
    for col, role in roles.items():
        current = columns[col]
        # A key or a sequence is never a measure: no missing codes, units or timepoints on it.
        columns[col] = current.model_copy(update={"role": role, "missing_codes": (), "unit": None, "timepoint": None})

    if keys.patient is None and keys.row.kind == "none" and not facts.population_estimates:
        from clinical_analytics.ui.components.variable_detector import VariableTypeDetector

        composite = VariableTypeDetector.find_composite_identifier_candidates(df)
        if composite:
            keys.row = RowKey(
                kind="composite", columns=tuple(composite), evidence="together they are unique on every line"
            )

    grain = _grain(df, dictionary, keys, facts)
    if (
        keys.patient is None
        and not facts.population_estimates
        and not facts.source_format
        and keys.row.kind in ("column", "none")
        and not keys.groups
    ):
        questions.insert(
            0,
            OpenQuestion(
                id="rows_are_patients",
                kind="rows_are_patients",
                question="Is each line one patient?",
                options=("yes, one patient per line", "no, lines are visits, tests or other records"),
            ),
        )
    semantics = DatasetSemantics(
        rows=df.height,
        grain=grain,
        patient_key=keys.patient,
        row_key=keys.row,
        sequence_columns=keys.sequences,
        grouping_columns=keys.groups,
        columns=columns,
        open_questions=tuple(questions),
        population_estimates=facts.population_estimates,
        source_format=facts.source_format,
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
        noun = key.role if key.role in ("infant", "mother") else "patient"
        kind = noun
        return Grain(
            kind=kind,
            noun="baby" if noun == "infant" else noun,
            one_row_per_patient=True,
            rows=rows,
            patients=patients,
            source="structure",
            evidence=f"{key.column} is on every line once",
        )
    if key is not None:
        word = next((s.noun for s in keys.sequences if s.noun != "row"), None)
        if word is None and keys.row.kind == "column":
            word = _event_word(keys.row.columns[0])
        if word is None:
            word = next(
                (
                    w
                    for c in df.columns
                    if df.schema[c].is_temporal() or c.lower().endswith(("date", "dt"))
                    for w in [_event_word(c)]
                    if w
                ),
                None,
            )
        if word is not None:
            return Grain(
                kind=word,
                noun=word,
                one_row_per_patient=False,
                rows=rows,
                patients=patients,
                source="structure",
                evidence=f"{key.column} repeats; the columns name each line a {word}",
            )
        return Grain(
            kind="record",
            noun="record of a patient",
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
            values = [m.value for m in col.missing_codes]
            expr = pl.when(pl.col(name).is_in(values)).then(None).otherwise(pl.col(name)).alias(name)
        if col.date is not None and col.date.format != NATIVE and not dtype.is_temporal():
            from clinical_analytics.contracts.dates import parse_expr

            if col.date.format in (YYYYMMDD, EXCEL_SERIAL) or dtype == pl.String:
                expr = parse_expr(name, col.date.format, col.date.has_time, dtype)
        if expr is not None:
            exprs.append(expr)
    if not exprs:
        return df
    return df.lazy().with_columns(exprs).collect()


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
        parts = list(semantics.row_key.columns)
        key = pl.concat_str([pl.col(c).cast(pl.String).fill_null("") for c in parts], separator="|")
        added = AppColumn(column=PATIENT_ID_COLUMN, kind="hash", made_from=tuple(parts))
        return (
            df.with_columns(key.hash(seed=0).cast(pl.String).alias(PATIENT_ID_COLUMN)),
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

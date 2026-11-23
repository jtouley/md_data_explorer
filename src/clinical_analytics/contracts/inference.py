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

logger = structlog.get_logger()

PATIENT_ID_COLUMN: Final = "patient_id"
ROW_NUMBER_COLUMN: Final = "row_number"

# Name tokens, after splitting on anything but letters and digits and at a case change.
_SPLIT: Final = re.compile(r"[^a-z0-9]+")
_CAMEL: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
ID_TOKENS: Final = frozenset({"id", "ids", "identifier", "mrn", "uuid", "guid", "ref", "key"})
# Number words name an ID only as the name's last word ("study_no", "Hosp No"); first, they name a
# count ("num_meds", "number_of_children").
ID_NUMBER_TOKENS: Final = frozenset({"no", "nr", "num", "number"})
# Words that make a name a count or an attribute of someone, never their ID ("patient_status",
# "patient_zip", "num_visits"), unless the name ends in an ID word.
ATTRIBUTE_TOKENS: Final = frozenset(
    {"status", "zip", "zipcode", "postcode", "postal", "name", "names", "type", "sex", "gender", "age", "dob",
     "race", "ethnicity", "state", "city", "address", "phone", "email", "weight", "height", "score", "group",
     "class", "category", "count", "total", "outcome", "flag", "initials", "language", "insurance"}
)  # fmt: skip
COUNT_TOKENS: Final = frozenset(
    {"meds", "medications", "drugs", "children", "visits", "admissions", "events", "episodes", "times", "days",
     "count", "cases", "deaths", "births", "pregnancies", "falls", "doses", "lesions", "nodes", "procedures"}
)  # fmt: skip
# Whole column names that the app and staging treat as an ID (kept as text by staging).
KEY_COLUMN_NAMES: Final = frozenset({"patient_id", "patientid", "subject_id", "subjectid", "id", "mrn", "study_id"})
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
    "anaesthetist": "anaesthetist",
    "anesthetist": "anaesthetist",
    "operator": "operator",
    "reviewer": "reviewer",
    "reader": "reader",
    "rater": "rater",
    "room": "room",
    "bed": "bed",
    "theatre": "theatre",
    "theater": "theatre",
    "consultant": "consultant",
    "midwife": "midwife",
    "therapist": "therapist",
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
    "admission": 0,
    "arrival": 0,
    "adm": 0,
    "start": 0,
    "intake": 0,
    "post": 900000,
    "postop": 900000,
    "after": 900000,
    "discharge": 950000,
    "dc": 950000,
    "end": 990000,
    "final": 990000,
    "last": 990000,
    "exit": 990000,
    "followup": 500000,
    "fu": 500000,
}
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
    ("mg",): "mg",
    ("mcg",): "mcg",
    ("ug",): "µg",
    ("pct",): "%",
    ("percent",): "%",
    ("ml",): "mL",
    ("cm",): "cm",
    ("mm",): "mm",
    ("hr",): "hours",
    ("hrs",): "hours",
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
    ("iu",): "IU",
}
_EXAMPLE_DAY: Final = datetime(2021, 4, 3, 14, 5)
_MAX_KEY_CANDIDATES: Final = 6


def name_tokens(column: str) -> list[str]:
    """'PtID' -> ['pt', 'id']; 'weight_preop_kg' -> ['weight', 'preop', 'kg']."""
    return [t for t in _SPLIT.split(_CAMEL.sub("_", column).lower()) if t]


def has_id_word(column: str) -> bool:
    """An ID word anywhere ("pt_id", "PtRef", "operator_code"), or a number word last ("study_no")."""
    tokens = name_tokens(column)
    return bool(tokens) and (any(t in ID_TOKENS for t in tokens) or tokens[-1] in ID_NUMBER_TOKENS)


def _attribute_named(tokens: list[str]) -> bool:
    """A count or an attribute of someone ("patient_status", "num_meds"), unless it ends in an ID word."""
    if tokens and (tokens[-1] in ID_TOKENS or tokens[-1] in ID_NUMBER_TOKENS):
        return False
    return any(t in ATTRIBUTE_TOKENS or t in COUNT_TOKENS for t in tokens) or (
        bool(tokens) and tokens[0] in ID_NUMBER_TOKENS
    )


def person_key_role(column: str) -> str | None:
    """'patient' for a name of the patient-ID role, 'infant'/'mother' for another person, else None.

    A patient word names the key when it is the whole name ("subj", "MRN", "patient") or comes with
    an ID word ("pt_id", "patient_ref", "case_id", "study_no"); never beside an attribute or count
    word ("patient_status", "patient_zip", "num_patients").
    """
    tokens = name_tokens(column)
    joined = "".join(tokens)
    if not tokens or _attribute_named(tokens):
        return None
    with_id = has_id_word(column)
    if joined in PATIENT_ROLE_TOKENS or (
        any(t in PATIENT_ROLE_TOKENS for t in tokens) and (with_id or len(tokens) == 1)
    ):
        return "patient"
    if any(t in WEAK_PATIENT_TOKENS for t in tokens) and with_id:
        return "patient"
    if any(t in PERSON_ROLE_TOKENS for t in tokens) and (with_id or len(tokens) == 1):
        return next(PERSON_ROLE_TOKENS[t] for t in tokens if t in PERSON_ROLE_TOKENS)
    return None


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
    """Named as an ID of someone or something: an ID word (not a count or attribute), or a person role."""
    tokens = name_tokens(column)
    return (has_id_word(column) and not _attribute_named(tokens)) or person_key_role(column) is not None


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


_TIME_UNITS: Final = frozenset(
    {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours", "minute", "minutes",
     "d", "wk", "mo", "yr", "yrs", "h", "hr", "hrs", "min"}
)  # fmt: skip
_UNIT_WORDS: Final = frozenset({"unit", "units", "u", "uom", "scale"})
_TIME_UNIT_NAMES: Final = {
    **dict.fromkeys(("day", "days", "d"), "days"),
    **dict.fromkeys(("week", "weeks", "wk"), "weeks"),
    **dict.fromkeys(("month", "months", "mo"), "months"),
    **dict.fromkeys(("year", "years", "yr", "yrs"), "years"),
    **dict.fromkeys(("hour", "hours", "h", "hr", "hrs"), "hours"),
    **dict.fromkeys(("minute", "minutes", "min"), "minutes"),
}
_TIME_UNIT_DAYS: Final = {
    "minutes": 1 / 1440, "hours": 1 / 24, "days": 1.0, "weeks": 7.0, "months": 365.25 / 12, "years": 365.25,
}  # fmt: skip


def _unit_days(unit_column: str) -> pl.Expr:
    """Days in one unit of each line's ``unit_column`` value (null for a value it doesn't name)."""
    unit = pl.col(unit_column).cast(pl.String).str.strip_chars().str.to_lowercase()
    return unit.replace_strict(
        {k: _TIME_UNIT_DAYS[v] for k, v in _TIME_UNIT_NAMES.items()}, default=None, return_dtype=pl.Float64
    )


def _unit_columns(df: pl.DataFrame) -> list[tuple[str, str]]:
    """(value column, unit column) pairs: a text column whose values are all time units
    ("months", "years") gives the unit of the number column it names ("dur" and "dur_unit") or,
    failing that, of the number column just before it."""
    pairs = []
    order = list(df.columns)
    for i, c in enumerate(order):
        if df.schema[c] != pl.String:
            continue
        values = {str(v).strip().lower() for v in df.get_column(c).drop_nulls().unique().to_list()}
        if not values or len(values) > 6 or not values <= _TIME_UNITS:
            continue
        stem = [t for t in name_tokens(c) if t not in _UNIT_WORDS]
        numbers = [n for n in order if n != c and df.schema[n].is_numeric()]
        named = [n for n in numbers if stem and name_tokens(n) == stem]
        before = [order[i - 1]] if i > 0 and order[i - 1] in numbers else []
        partners = named or before
        if partners:
            pairs.append((partners[0], c))
    return pairs


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
_MASS_STEMS: Final = frozenset({"weight", "wt", "mass", "bw", "birthweight"})
_DAY_STEMS: Final = frozenset({"age", "ga", "gest", "gestation", "gestational", "dur", "duration", "los"})


def _unit_from_name(column: str) -> str | None:
    tokens = [t for t in _SPLIT.split(column.lower()) if t]  # no case split: "cmH2O" is one unit
    # One letter is a unit only after a stem that measures in it: "bw_g", "ga_d" (not "vitamin_d").
    if len(tokens) >= 2 and tokens[-1] == "g" and set(tokens[:-1]) & _MASS_STEMS:
        return "g"
    if len(tokens) >= 2 and tokens[-1] == "d" and set(tokens[:-1]) & _DAY_STEMS:
        return "days"
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
    episodes: tuple[EpisodeKey, ...] = ()


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
    stated_entries = [e.column for e in dictionary.entries.values() if e.names_patient_key]
    candidates = [
        c
        for c in _candidates(df, excluded, (*dictionary.key_statements, *stated_entries))
        if (c.unique or c.distinct > 2 or c.role == "patient" or c.column in dictionary.key_statements)
        # A key is on most lines: an ID missing on more than half of them names no one reliably.
        and (c.unique or c.nulls * 2 <= rows)
    ]
    by_name = {c.column: c for c in candidates}
    stated_names = (*dictionary.key_statements, *stated_entries)
    # A stated key still passes the role filter: a surgeon's or a site's ID is never the patient.
    stated = [c for c in stated_names if c in by_name and by_name[c].group is None and by_name[c].event is None]
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

    # Times since an index event are sequences of their own kind (``_time_since``), never order numbers.
    seq_cols = [s for s in seq_cols if not _time_since_named(s)]

    def free(c: _Candidate) -> bool:
        return c.group is None and c.event is None

    if stated:
        chosen, source = by_name[stated[0]], "dictionary"
        evidence = "the data dictionary names it as the patient identifier"
    elif facts.cohort_key is not None and facts.cohort_key in by_name:
        chosen, evidence = by_name[facts.cohort_key], "the tables were joined on it, one patient each"
    if chosen is None:
        named = [c for c in candidates if c.role == "patient" and free(c)]
        if any(c.unique for c in named):
            # A repeating column with few values never beats a unique patient ID.
            floor = max(3, config.GENERIC_KEY_MIN_SHARE * rows)
            named = [c for c in named if c.unique or c.distinct >= floor]
        named = sorted(named, key=lambda c: (patient_name_rank(c.column), c.unique, order[c.column]))
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
        if infants and not any(c.unique for c in infants) and any(c.distinct < rows - c.nulls for c in mothers):
            # A mother ID repeats beside a baby ID that repeats too: one line per baby, or per
            # mother? Only the clinician can say; the baby is the tentative patient. A baby ID
            # unique on every line settles it: each line is one baby (twins share a mother).
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
        elif infants:
            chosen = next((c for c in infants if c.unique), infants[0])
            evidence = f"{chosen.column} names a baby"
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
            GroupingColumn(
                column=c.column, role=role, is_id=c.named, source="structure", evidence="its values repeat across lines"
            )
        )
    grouped = {g.column for g in groups} | seq_names | set(keys.row.columns)
    if keys.patient is not None:
        grouped.add(keys.patient.column)
    for column in df.columns:
        # Staff and places (an anaesthetist, a ward, a room, a region) group lines; never measures.
        place = _group_role(column)
        if place is None or column in grouped or column in excluded or place == "category":
            continue
        # A yes/no flag named like a role ("hosp": admitted to hospital, Y/N) is not a grouping.
        if (df.schema[column] == pl.String or df.schema[column].is_integer()) and df.get_column(
            column
        ).drop_nulls().n_unique() >= 3:
            groups.append(
                GroupingColumn(column=column, role=place, is_id=False, source="structure", evidence="named for a role")
            )
    keys.groups = tuple(groups)
    if keys.patient is not None:
        keys.episodes = _episode_keys(df, keys.patient.column, [c for c in candidates if c.event is not None])
        keys.sequences = (*keys.sequences, *_time_since(df, keys.patient.column, excluded | grouped))
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
        if g["groups"] and g["ok"] / g["groups"] >= config.SEQUENCE_MIN_SHARE:
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


def _episode_keys(df: pl.DataFrame, key: str, events: Sequence[_Candidate]) -> tuple[EpisodeKey, ...]:
    """IDs of episodes nested within one patient (an admission ID within an MRN): every episode
    value belongs to exactly one patient, so episodes can be counted."""
    found: list[EpisodeKey] = []
    for c in events:
        if c.column == key or not c.named:
            continue
        most = (
            df.lazy()
            .filter(pl.col(c.column).is_not_null())
            .group_by(c.column)
            .agg(pl.col(key).n_unique().alias("p"))
            .select(pl.col("p").max())
            .collect()
            .item()
        )
        if most == 1:
            found.append(
                EpisodeKey(
                    column=c.column,
                    within=key,
                    noun=c.event or "episode",
                    source="structure",
                    evidence=f"each value belongs to one {key}",
                )
            )
    return tuple(found)


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
        # Durations in mixed units are read in one unit, the largest the file uses: an average of
        # 6 months and 2 years is never 4 (``apply_semantics`` converts every line).
        units_used = {
            _TIME_UNIT_NAMES[str(v).strip().lower()] for v in df.get_column(unit_column).drop_nulls().unique()
        }
        target = max(units_used, key=lambda unit: _TIME_UNIT_DAYS[unit])
        unit = Unit(value=target, source="structure")
        columns[value_column] = columns[value_column].model_copy(update={"unit_column": unit_column, "unit": unit})
    weights = tuple(c for c in df.columns if config.SURVEY_WEIGHT_NAME.search(c.lower()))
    survey = bool(weights) or dictionary.mentions_weights

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
    # A key, a sequence or a grouping column is never asked about "not recorded" values.
    questions = [q for q in questions if not (q.kind == "missing_code" and q.column in roles)]
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
    case_count: str | None = None
    aggregate: AggregateInfo | None = None
    if keys.patient is None and not published and not any(g.is_id for g in keys.groups):
        summary = _summary_table(df, date_columns, dictionary)
        if isinstance(summary, _Summary):
            grain, aggregate, case_count = summary.grain, summary.aggregate, summary.aggregate.count_column
            questions.extend(summary.questions)
        elif isinstance(summary, str):
            # Looks like a table of groups but not recognised with confidence: asked, never
            # silently read as one line per patient.
            logger.info("summary_table_not_recognised", reason=summary)
            questions.append(_aggregate_question(_count_columns(df)[0], summary))
    if (
        keys.patient is None
        and not facts.population_estimates
        and not facts.source_format
        and keys.row.kind in ("column", "none")
        and not any(g.is_id for g in keys.groups)
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


_CASE_COUNT_TOKENS: Final = frozenset(
    {"count", "counts", "n", "cases", "deaths", "events", "number", "total", "freq", "frequency", "admissions",
     "births", "patients", "visits", "infections", "hospitalizations", "hospitalisations"}
)  # fmt: skip


def _human(column: str) -> str:
    return " ".join(name_tokens(column)) or column


_RATE_TOKENS: Final = frozenset(
    {"rate", "rates", "pct", "percent", "percentage", "ratio", "crude", "adjusted", "prevalence", "incidence",
     "proportion", "share", "mean", "median"}
)  # fmt: skip
_BAND_TOKENS: Final = frozenset({"group", "grp", "band", "cat", "category", "class", "level", "bin"})
# Time columns are strata whatever their number of values (52 weeks, 20+ years).
_TIME_TOKENS: Final = frozenset(
    {"year", "yr", "years", "week", "wk", "weeks", "epiweek", "mmwr", "month", "months", "mo", "quarter", "qtr",
     "period"}
)  # fmt: skip
_TIME_TEXT: Final = re.compile(r"\b(?:week|month|quarter|year)\b", re.IGNORECASE)
_WHAT_WORDS: Final = ("deaths", "cases", "events", "admissions", "births", "visits", "infections")


@dataclass(frozen=True)
class _Summary:
    """A table of groups: its grain, what it holds, and what only the clinician can settle."""

    grain: Grain
    aggregate: AggregateInfo
    questions: tuple[OpenQuestion, ...] = ()


def _count_columns(df: pl.DataFrame) -> list[str]:
    """Whole, non-negative numbers named as counts ("cases", "Deaths", "n", "number_of_events")."""
    found = []
    for c in df.columns:
        dtype = df.schema[c]
        if not (dtype.is_integer() or dtype.is_float()) or not set(name_tokens(c)) & _CASE_COUNT_TOKENS:
            continue
        values = df.get_column(c).drop_nulls()
        if values.len() == 0 or _float(values.min()) < 0:
            continue
        if dtype.is_float() and not bool((values == values.floor()).all()):
            continue
        found.append(c)
    return found


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
    """A low-cardinality category, a place, or a period (a year, a week, a month: by its name or
    its dictionary entry, whatever its number of values): never a measure like age or SBP."""
    dtype = df.schema[column]
    series = df.get_column(column)
    if column in dates:
        return True
    if dtype == pl.String:
        distinct = series.n_unique()
        return distinct <= max(60, df.height // 2)
    if not dtype.is_integer():
        return False
    distinct = series.n_unique()
    values = series.drop_nulls()
    entry = dictionary.entry(column) if dictionary is not None else None
    described = entry is not None and bool(_TIME_TEXT.search(entry.description or entry.text))
    year_like = (
        described
        or bool(set(name_tokens(column)) & _TIME_TOKENS)
        or (values.len() > 0 and bool(values.is_between(config.DATE_YEAR_MIN, config.DATE_YEAR_MAX).all()))
    )
    return year_like or distinct <= (200 if loose else config.STRATUM_MAX_CODES)


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
    counts = _count_columns(df)
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
        # An order number named for an event ("visit_no", "cycle") says what a line is; a time
        # since a start ("weeks_since_start") or an unnamed number never does.
        word = next((s.noun for s in keys.sequences if s.kind == "order" and s.noun in EVENT_TOKENS.values()), None)
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
    converted = df.lazy().with_columns(exprs).collect() if exprs else df
    return _in_one_unit(converted, semantics)


def _in_one_unit(df: pl.DataFrame, semantics: DatasetSemantics) -> pl.DataFrame:
    """Durations whose unit is given per line (``unit_column``) converted to the column's one
    unit, the unit column rewritten to it: safe to run twice (a converted line has factor 1)."""
    exprs: list[pl.Expr] = []
    for name, col in semantics.columns.items():
        unit_column = col.unit_column
        if unit_column is None or col.unit is None or name not in df.columns or unit_column not in df.columns:
            continue
        target = _TIME_UNIT_DAYS[col.unit.value]
        days = _unit_days(unit_column)
        exprs += [
            pl.when(days.is_not_null())
            .then(pl.col(name).cast(pl.Float64) * days / target)
            .otherwise(pl.col(name))
            .alias(name),
            pl.when(days.is_not_null()).then(pl.lit(col.unit.value)).otherwise(pl.col(unit_column)).alias(unit_column),
        ]
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

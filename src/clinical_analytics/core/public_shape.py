# SPDX-License-Identifier: MIT
"""Is this table shaped like a public estimate table? An allowlist, fail closed (ADR014 §7).

Under the owner's "trust the source" decision (2026-11-19) a declared public table is shown as
published, so this check is what keeps patient-level files under the patient rules. It is an
allowlist: the estimates path is allowed only when every key column is a recognised public
dimension *by its values*, each value matched whole (never by prefix):

* places: US states, territories, census regions and divisions, HHS regions, counties (FIPS in a
  column named for a place, or "Name County, ST"), Detroit and out-Wayne, the nation, countries;
* time: years, months (2023-01), dates (2023-01-07), MMWR weeks (202401), flu seasons (2022-23),
  week or month numbers in a column named for them;
* standard age bands, sex, standard race and ethnicity labels, and total labels;
* label columns named for a cause, indicator, measure, metric, question or type, whose values are
  words (not ID-like, not a person's name, not a code; ICD codes only in a cause or ICD column).
  A table with a label key must also have a place or time key and an estimate with an interval, a
  rate, or a published total level.

A place, time or demographic column may carry a minority of plain labels ("Entire Network",
"COVID-NET" beside states). Every other column (attributes, aliases, one-value columns) is scanned
too, with no lower threshold: SSN, MRN or case-number shapes, ID-like codes, person names, or a
numeric column named as a patient ID refuse the table. Reasons never quote a value.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

from clinical_analytics.contracts import config, geography

ValueClass: TypeAlias = Literal["place", "time", "age", "sex", "race", "total", "placeholder"]

_SSN: Final = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_MRN: Final = re.compile(r"^(mrn|mr|pt|pid|id|acct|enc|hadm)?[-_ #:]?\d{4,}$", re.IGNORECASE)
_CASE_NUMBER: Final = re.compile(r"^(\d{4})[-/](\d{3,})$")
_ID_CODE: Final = re.compile(r"^([A-Za-z]{1,6}[-_ #]?\d+[A-Za-z]?|\d{3,}|[0-9a-fA-F]{6,}(-[0-9a-fA-F]{4,})*)$")
_HEX: Final = re.compile(r"^[0-9a-f]{6,}$", re.IGNORECASE)
# Contact details (an email, phone number or web address) identify a person or an account.
_CONTACT: Final = re.compile(
    r"^([^@\s]+@[^@\s]+\.[a-z]{2,}|\+?1?[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]\d{4}|(https?://|www\.)\S+)$",
    re.IGNORECASE,
)
_ICD: Final = re.compile(r"^\*?[A-Z]\d{2}(\.\d{1,2})?(\s*[-–]\s*[A-Z]?\d{2}(\.\d{1,2})?)?$")
_PERSON_NAME: Final = re.compile(r"^[A-Z][a-z'\-]{1,15}(\s+[A-Z]\.?)?(\s+[A-Z][a-z'\-]{1,20}){1,2}$")
_INITIALS: Final = re.compile(r"^[A-Z]{2}$|^[A-Z]\.\s?[A-Z]\.?$")
_YEAR: Final = re.compile(r"^(18|19|20|21)\d\d$")
_MONTH: Final = re.compile(r"^((18|19|20|21)\d\d)[-/](0[1-9]|1[0-2])$")
_DATE: Final = re.compile(r"^((18|19|20|21)\d\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
_MMWR: Final = re.compile(r"^((19|20)\d\d)(0[1-9]|[1-4]\d|5[0-3])$")
_SEASON: Final = re.compile(r"^((18|19|20|21)\d\d)\s*[-–/]\s*(\d{2}|\d{4})$")
_AGE: Final = re.compile(
    r"^(ages?\s*)?"
    r"(?P<body>(?P<lo>\d{1,3})\s*(-|–|to)\s*<?\s*(?P<hi>\d{1,3})|(?P<op>[<>≥≤]=?|>=|<=)\s*(?P<edge>\d{1,3})"
    r"|(?P<plus>\d{1,3})\s*\+|under\s+(?P<under>\d{1,3})|(?P<older>\d{1,3})\s*(years?\s*)?(and|or)\s*(over|older))"
    r"\s*(years?|yrs?|y|months?|mos?)?(\s*old)?(\s*\([a-z ]+\))?$",
    re.IGNORECASE,
)
_SEX: Final = frozenset(
    {"female", "male", "both", "both sexes", "all sexes", "f", "m", "women", "men", "females", "males", "unknown"}
)
_RACE_TOKENS: Final = frozenset(
    {
        "white", "black", "african", "american", "asian", "hispanic", "latino", "latina", "latinx", "native",
        "hawaiian", "pacific", "islander", "islanders", "indian", "alaska", "alaskan", "multiracial", "multiple",
        "races", "race", "two", "more", "other", "unknown", "non", "nh", "all", "ethnicity", "ethnicities", "and",
        "or", "of", "aian", "nhopi", "api", "aapi", "mixed",
    }
)  # fmt: skip
_RACE_CORE: Final = frozenset(
    {
        "white",
        "black",
        "asian",
        "hispanic",
        "latino",
        "native",
        "hawaiian",
        "pacific",
        "indian",
        "multiracial",
        "aian",
        "nhopi",
        "api",
        "races",
        "race",
        "multiple",
    }  # fmt: skip
)
# A level for what is not known or not listed: recognised in a key, but never a published total
# (a patient file has "Other" and "Unknown" too).
_PLACEHOLDERS: Final = frozenset(
    {"unknown", "other", "not stated", "missing", "none", "not specified", "unspecified", "other race"}
)
_TOTALS: Final = (
    config.ESTIMATE_TOTAL_LEVELS
    | geography.OTHER_PLACE_LABELS
    | {"age-standardized", "age-standardised", "age standardized", "age-adjusted", "all ages", "all causes"}
)
# Words of medical and public-health labels: a two-word capitalised label holding one is not a
# person's name ("Diabetes Mellitus", "Kidney Disease").
_LABEL_VOCABULARY: Final = frozenset(
    {
        "disease", "diseases", "infection", "infections", "cancer", "cancers", "syndrome", "deaths", "death",
        "injury", "injuries", "hepatitis", "syphilis", "tuberculosis", "influenza", "pneumonia", "respiratory",
        "chronic", "acute", "primary", "secondary", "congenital", "causes", "vaccination", "coverage", "smoking",
        "diabetes", "mellitus", "heart", "kidney", "liver", "lung", "stroke", "rate", "rates", "prevalence",
        "incidence", "cases", "number", "percent", "virus", "hiv", "aids", "covid", "flu", "rsv", "measles",
        "pertussis", "legionella", "legionellosis", "salmonella", "campylobacter", "candida", "neoplasms",
        "disorders", "conditions", "condition", "hospitalization", "hospitalizations", "visits", "emergency",
        "care", "intensive", "mortality", "morbidity", "risk", "exposure", "estimate", "estimated", "observed",
        "crude", "adjusted", "weekly", "cumulative", "network", "surveillance", "admitted", "underlying",
        "mechanical", "ventilation", "pneumococcal", "seasonal", "adults", "children", "infants", "total",
        "week", "weeks", "ending", "date", "dates", "year", "years", "month", "season", "type", "data", "value",
        "state", "states", "county", "region", "national", "age", "group", "groups", "category", "overall",
        "all", "other", "unknown", "reported", "confirmed", "probable", "new", "per", "population", "count",
    }
)  # fmt: skip
_LABEL_TOKENS: Final = frozenset(
    {
        "cause", "causes", "indicator", "measure", "metric", "question", "topic", "condition", "disease",
        "pathogen", "etiology", "rei", "risk", "vaccine", "type", "category", "response", "network",
        "outcome", "icd", "ucd", "mcd", "stratification", "stratification1", "stratification2",
        "datavaluetype", "dimension", "class", "event", "events", "data", "breakout", "subtopic",
    }
)  # fmt: skip
_ID_SUFFIX: Final = frozenset({"id", "code", "cd", "no", "num", "nbr", "number", "key"})


def tokens(column: str) -> list[str]:
    """Lowercase word tokens of a column name, camelCase split ("ClientID" -> client, id)."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", column)
    return re.findall(r"[a-z]+|\d+", spaced.casefold())


def _text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _age(value: str) -> bool:
    match = _AGE.match(value)
    if match is None:
        return False
    numbers = [int(n) for n in re.findall(r"\d+", match.group("body"))]
    if any(n > 120 for n in numbers):
        return False
    lo, hi = match.group("lo"), match.group("hi")
    return not (lo is not None and hi is not None and int(lo) >= int(hi))


def _race(value: str) -> bool:
    words = re.findall(r"[a-z]+", value.casefold())
    return bool(words) and all(w in _RACE_TOKENS for w in words) and any(w in _RACE_CORE for w in words)


def classify_value(value: Any, column: str) -> ValueClass | None:
    """The public dimension one value belongs to, matched whole; None when it is none of them."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return "total"  # an empty key level is a total row (CDC WONDER's "Total" rows)
    text = _text(value)
    folded = text.casefold()
    names = set(tokens(column))
    if folded in _PLACEHOLDERS:
        return "placeholder"
    if folded in _TOTALS:
        return "total"
    if _YEAR.match(text) or _MONTH.match(text) or _DATE.match(text) or _MMWR.match(text) or _season(text):
        return "time"
    if text.isdigit() and int(text) <= 53 and names & {"week", "wk", "mmwr"}:
        return "time"
    if text.isdigit() and 1 <= int(text) <= 12 and names & {"month", "mo"}:
        return "time"
    place = geography.classify(value, codes=geography.is_geo_code_column(column))
    if place.kind in ("state", "nation", "territory", "region", "country"):
        return "place"
    if place.kind == "other":
        return "total"
    if place.kind == "substate" and (
        _county_shaped(text) or geography.is_detroit(text) or geography.is_out_wayne(text)
    ):
        return "place"
    if place.kind == "substate" and text.isdigit() and geography.is_geo_code_column(column):
        return "place"
    if folded in _SEX:
        return "sex"
    if _age(text):
        return "age"
    if _race(text):
        return "race"
    return None


def _season(text: str) -> bool:
    match = _SEASON.match(text)
    if match is None:
        return False
    start, end = int(match.group(1)), match.group(3)
    following = int(end) if len(end) == 4 else int(str(start)[:2] + end)
    return following == start + 1


_COUNTY: Final = re.compile(
    r"^[A-Z][A-Za-z .'\-]+ (County|Parish|Borough|Census Area|City and Borough|Municipality)"
    r"(, ?[A-Z]{2}|, [A-Z][a-z]+( [A-Z][a-z]+)?)?$"
)


def _county_shaped(text: str) -> bool:
    return bool(_COUNTY.match(text))


def is_label(value: Any, column: str) -> bool:
    """A plain label: words, not an ID, not a person's name, not a code (ICD codes only in a cause
    or ICD column)."""
    if value is None:
        return True
    text = _text(value)
    if _ICD.match(text):
        return bool(set(tokens(column)) & {"cause", "icd", "ucd", "mcd", "diagnosis", "code"})
    if not re.search(r"[A-Za-z]{3,}", text) and not re.search(r"[A-Za-z]+[-/][A-Za-z]+", text):
        return False
    return not (id_shaped(text) or person_name(text))


def id_shaped(text: str) -> bool:
    """An SSN, MRN, case number, contact detail, hash or ID code shape (whole value)."""
    text = text.strip()
    if _SSN.match(text) or _MRN.match(text) or _CONTACT.match(text) or _HEX.match(text) and re.search(r"\d", text):
        return True
    case = _CASE_NUMBER.match(text)
    if case is not None and not _season(text):
        return True
    return bool(_ID_CODE.match(text)) and not (_YEAR.match(text) or _MMWR.match(text))


def person_name(text: str) -> bool:
    """Shaped like a person's name ("Ana Lopez", "J. Smith") or initials, and holding no word of a
    public-health label."""
    if _INITIALS.match(text):
        return True
    if not _PERSON_NAME.match(text):
        return False
    return not (set(re.findall(r"[a-z]+", text.casefold())) & _LABEL_VOCABULARY)


@dataclass(frozen=True)
class KeyVerdict:
    ok: bool
    kinds: frozenset[str]  # the classes its values fall in
    label: bool  # accepted as a label column


def key_column(column: str, values: Sequence[Any], alias_values: Mapping[str, Sequence[Any]]) -> KeyVerdict:
    """Whether one key column is a recognised public dimension, by its values (or an alias's: a bare
    county name beside its FIPS code)."""
    present = list(values)
    classes = [classify_value(v, column) for v in present]
    strict = [c for c in classes if c is not None]
    kinds = frozenset(strict)
    if present and len(strict) == len(present):
        return KeyVerdict(True, kinds, False)
    for alias, vals in alias_values.items():
        alias_classes = [classify_value(v, alias) for v in vals]
        if vals and all(c is not None for c in alias_classes):
            return KeyVerdict(True, frozenset(c for c in alias_classes if c), False)
    rest = [v for v, c in zip(present, classes, strict=True) if c is None]
    if set(tokens(column)) & _LABEL_TOKENS:
        labelled = len([v for v in rest if is_label(v, column)]) == len(rest) and len(present) >= 2
        return KeyVerdict(labelled, kinds, labelled)
    # A place, time or demographic column may carry a minority of plain labels beside its values.
    mostly = len(strict) * 4 >= len(present) * 3 and kinds - {"total", "placeholder"}
    return KeyVerdict(bool(mostly) and all(is_label(v, column) for v in rest), kinds, False)


def other_column_problem(column: str, values: Sequence[Any], *, is_numeric: bool, id_alias: bool) -> bool:
    """A non-key column (attribute, alias, one-value column, numeric column) that reveals people:
    SSN, MRN or case-number shapes, ID-like codes, person names, or a name that says patient ID.
    ``id_alias``: a code column 1:1 with an accepted key column (GBD location_id), which is fine."""
    name_tokens = set(tokens(column))
    joined = "".join(tokens(column))
    named_id = (
        joined in config.ESTIMATE_PERSON_NAMES
        or _patient_key_name(joined)
        or (bool(name_tokens & config.ESTIMATE_PERSON_TOKENS) and bool(name_tokens & _ID_SUFFIX))
    )
    if is_numeric:
        return named_id  # "patient_id" stored as a float (SPSS), "member_no"; never "Patient Days"
    present = [_text(v) for v in values if v is not None]
    if not present:
        return False
    if all(classify_value(v, column) is not None for v in present):
        return False  # years, FIPS, states, totals: public values whatever the column is called
    if named_id or name_tokens & config.ESTIMATE_PERSON_TOKENS:
        return True
    if any(_SSN.match(v) or _MRN.match(v) or (_CASE_NUMBER.match(v) and not _season(v)) for v in present):
        return not id_alias
    if not id_alias and sum(person_name(v) for v in present) * 10 >= len(present) * 3:
        return True
    if not id_alias and sum(id_shaped(v) for v in present) * 2 >= len(present):
        return True
    return False


def _patient_key_name(joined: str) -> bool:
    from clinical_analytics.core.schema_inference import PATIENT_KEY_NAMES_SPECIFIC

    return joined in PATIENT_KEY_NAMES_SPECIFIC


def id_named(column: str) -> bool:
    """A column name that ends in an ID word (location_id, State Code)."""
    names = tokens(column)
    joined = "".join(t for t in names if not t.isdigit())
    return bool(set(names) & _ID_SUFFIX) or joined.endswith(("id", "code"))  # "topicid", "stratificationid1"

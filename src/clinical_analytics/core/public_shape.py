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
import unicodedata
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
        "vitamin", "serogroup", "serotype", "strain", "hepatitis", "influenza", "dose", "doses", "coli",
        "hospitals", "hospital", "regions", "selected", "census", "hhs", "sites", "site", "clinics",
        "laboratories", "laboratory", "facilities", "jurisdictions", "jurisdiction", "areas", "area",
        "territories", "metro", "metropolitan", "urban", "rural", "sentinel", "providers", "division",
        "divisions", "nation", "country", "countries", "world", "global", "tract", "tracts", "zip",
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
# Words that make a label a person, record or bed ("Patient 7", "Subject 01", "Room A12").
_PERSON_WORDS: Final = frozenset(
    {
        "patient", "pt", "subject", "resident", "participant", "member", "client", "case", "mrn", "record",
        "room", "bed", "person", "enc", "encounter", "pid", "enrollee", "respondent", "inmate", "student",
        "employee", "child", "infant", "baby", "mother", "donor", "chart", "acct", "account",
    }
)  # fmt: skip
# A column named only by these holds people's names ("name", "full_name", "Last Name").
_NAME_COLUMN_TOKENS: Final = frozenset(
    {"name", "names", "first", "last", "full", "given", "family", "surname", "middle"}
)
# A letters-then-digits code inside a label ("A12", "MRN.0007", "Patient7"); public-health codes
# that look like one are named here ("COVID-19", "H1N1", "PM2.5").
# Letters in any script ("É12" is a code too), so [^\W\d_] rather than [A-Za-z].
_CODE_TOKEN: Final = re.compile(
    r"(?<![^\W_.])([^\W\d_]{1,8}[-_.#]?\d+(\.\d+)?[^\W\d_]?"
    r"|(?=\w*\d)(?=\w*[^\W\d_])[^\W_]+)(?![^\W_])"
)
_PUBLIC_CODES: Final = re.compile(
    r"^(covid-?19|sars-cov-?2|h\d{1,2}n\d{1,2}|hiv-?[12]|htlv-?[12]|hpv-?\d{1,2}|pm2\.5|pm10|cd4|hba1c|"
    r"vitamin\s?d3?|ivd\d*|cvd19|covid19|mmr2?|dtap\d?|tdap\d?|pcv\d{2}|ppsv23|hepb\d?|ccc2|q[1-4]|h[1-9]|"
    r"pdm09|o157|o26|o45|o103|o111|o121|o145|\d{1,2}(st|nd|rd|th)|\d{1,2}v)$",
    re.IGNORECASE,
)
# "Smith J", "J. Smith", "Smith, J.": a surname with an initial.
_NAME_INITIAL: Final = re.compile(r"^([A-Z][A-Za-z'’\-]+,?\s+[A-Z]\.?|[A-Z]\.?\s+[A-Z][A-Za-z'’\-]+)$")
# A published ICD code list after a cause: "(A16-A19)", "(*U01-*U02,X85-Y09,Y87.1)", "(R95)".
_ICD_RANGES: Final = re.compile(
    r"\(\s*\*?[A-Z]\d{2}(\.\d+)?(\s*[-–]\s*\*?[A-Z]?\d{2}(\.\d+)?)?"
    r"(\s*,\s*\*?[A-Z]\d{2}(\.\d+)?(\s*[-–]\s*\*?[A-Z]?\d{2}(\.\d+)?)?)*\s*\)"
)
_ICD_LIST_COLUMN_TOKENS: Final = frozenset({"cause", "causes", "icd", "ucd", "mcd", "list", "diagnosis", "condition"})
# Columns about a named clinician: any value that reads as a name refuses the table.
_CLINICIAN_TOKENS: Final = frozenset(
    {
        "provider", "providers", "physician", "doctor", "clinician", "nurse", "attending", "surgeon",
        "practitioner", "prescriber", "pcp", "staff", "therapist", "dentist",
    }
)  # fmt: skip
# Columns of a person's own dates or events: never a public time key ("birth_year", "visit_month").
PERSON_EVENT_TOKENS: Final = frozenset(
    {"birth", "dob", "born", "visit", "admit", "admission", "discharge", "encounter", "death", "deceased"}
)
# Matched inside a column name too ("birthyear", "dobyear", "admityear", "yob", "dod").
_PERSON_EVENT_STEMS: Final = ("birth", "dob", "yob", "dod", "born", "admit", "admission", "visit", "discharge",
                              "death", "deceased", "encounter")  # fmt: skip


def person_event_named(column: str) -> bool:
    """A column named for a person's own dates or events, as a word or inside one."""
    names = tokens(column)
    joined = "".join(names)
    return (
        bool(set(names) & PERSON_EVENT_TOKENS)
        or any(name.startswith(_PERSON_EVENT_STEMS) for name in names)
        or any(stem in joined for stem in ("birth", "dob", "admit", "discharge", "encounter"))
    )


_UNICODE_LAST_FIRST: Final = re.compile(
    r"^[^\W\d_][^\W\d_'’\-]*(\s+[^\W\d_][^\W\d_'’\-]*){0,2},\s*[^\W\d_][^\W\d_'’\-.]*(\s+[^\W\d_][^\W\d_'’\-.]*)?$"
)
_LAST_FIRST: Final = re.compile(r"^[A-Za-z'’\-]+(\s[A-Za-z'’\-]+)?,\s*[A-Za-z'’\-.]+(\s[A-Za-z'’\-.]+)?$")


def tokens(column: str) -> list[str]:
    """Lowercase word tokens of a column name, camelCase split ("ClientID" -> client, id)."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", column)
    return re.findall(r"[a-z]+|\d+", spaced.casefold())


def _text(value: Any) -> str:
    """A value as text, NFKC-normalised (fullwidth "ＭＲＮ０００７" reads as "MRN0007")."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = unicodedata.normalize("NFKC", str(value))
    # Format characters (zero-width space and joiners) never split a word: "Pat\u200bient".
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf").strip()


def mixed_script(text: str) -> bool:
    """Letters from Latin and another script in one value: a lookalike ("Pаtient" with a Cyrillic
    "а"). Accented Latin ("Doña Ana County", "Bayamón") is Latin."""
    scripts = {unicodedata.name(ch, "?").split(" ")[0] for ch in text if ch.isalpha()}
    return "LATIN" in scripts and len(scripts) > 1


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
        # An empty level: recognised (CDC WONDER's "Total" rows are empty), but never evidence of a
        # published total, since one blank cell in a patient grid is empty too.
        return "placeholder"
    text = _text(value)
    if mixed_script(text):
        return None
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
        return "total" if folded in _TOTALS else "placeholder"
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
    return label_shaped(value, column) and not person_name(_text(value)) if value is not None else True


def label_shaped(value: Any, column: str) -> bool:
    """A label by everything but the name test: words, no person or record word, no ID or code."""
    if value is None:
        return True
    text = _text(value)
    # A published ICD code list after a cause ("Tuberculosis (A16-A19)") is part of the label, in a
    # cause or ICD column, after a cause that reads as one ("Smith (B12)" does not).
    if set(tokens(column)) & _ICD_LIST_COLUMN_TOKENS and _ICD_RANGES.search(text):
        rest = _ICD_RANGES.sub("", text).strip()
        rest_words = _words_of(rest)
        if (
            rest.startswith("#")
            or len(rest_words) >= 2
            or set(rest_words) & _NOT_NAME_WORDS
            or any(_MEDICAL_ENDING.search(w) for w in rest_words)
        ):
            text = rest
    if mixed_script(text) or not text.isascii() and not _latin(text):
        return False  # a lookalike letter from another script
    if _PUBLIC_CODES.match(text):
        return True  # "PM2.5", "COVID-19"
    if _ICD.match(text):
        return bool(set(tokens(column)) & {"icd", "ucd", "mcd", "diagnosis", "code"})
    if not re.search(r"[A-Za-z]{3,}", text) and not re.search(r"[A-Za-z]+[-/][A-Za-z]+", text):
        return False
    words = set(re.findall(r"[a-z]+", text.casefold()))
    if words & _PERSON_WORDS:
        return False  # "Patient 7", "Subject 01", "Room A12"
    if any(not _PUBLIC_CODES.match(m.group(0)) for m in _CODE_TOKEN.finditer(text)):
        return False  # "MRN.0007", "A12", "Pt7"
    if any(not _YEAR.match(run) for run in re.findall(r"(?<!\d)\d{4,}(?!\d)", text)):
        return False  # "Admission 004417": a word beside a long number
    if re.search(r"\d{3,}\s*[-_/]\s*[A-Za-z]", text):
        return False  # "12345-ABC"
    return not id_shaped(text)


def _latin(text: str) -> bool:
    return all(not ch.isalpha() or unicodedata.name(ch, "?").startswith("LATIN") for ch in text)


def id_shaped(text: str) -> bool:
    """An SSN, MRN, case number, contact detail, hash or ID code shape (whole value)."""
    text = text.strip()
    if _PUBLIC_CODES.match(text):
        return False  # "COVID-19", "H1N1": a disease, not an ID
    if _SSN.match(text) or _MRN.match(text) or _CONTACT.match(text) or _HEX.match(text) and re.search(r"\d", text):
        return True
    case = _CASE_NUMBER.match(text)
    if case is not None and not _season(text):
        return True
    return bool(_ID_CODE.match(text)) and not (_YEAR.match(text) or _MMWR.match(text))


_TITLES: Final = frozenset({"dr", "mr", "mrs", "ms", "miss", "mx", "prof", "sir", "dame", "rev", "fr", "sr", "jr"})
_PARTICLES: Final = frozenset(
    {
        "de", "la", "las", "los", "del", "della", "van", "von", "der", "den", "bin", "binti", "al", "el", "da",
        "di", "du", "le", "ibn", "ben", "dos", "das", "ter", "ten", "y", "e", "st", "mac", "o",
    }
)  # fmt: skip
# One part of a name in any script: letters, joined by an apostrophe or hyphen ("D'Angelo",
# "al-Rashid", "Nguyễn"); an initial is one letter with an optional dot.
_NAME_PART: Final = re.compile(r"^[^\W\d_]+([’'\-][^\W\d_]+)*$")
_INITIAL_PART: Final = re.compile(r"^[^\W\d_]\.?$")


_NOT_NAME_WORDS: Final = _LABEL_VOCABULARY | {
    "sexes", "both", "atlantic", "central", "north", "south", "east", "west", "new", "mid", "hispanic", "latino",
    "fever", "encephalitis", "spotted", "nile", "valley", "mountain", "louis", "fungal", "bacterial", "viral",
}  # fmt: skip
# Medical word endings: a value holding one is a condition, not a person ("Lyme borreliosis").
_MEDICAL_ENDING: Final = re.compile(r"(itis|osis|iasis|emia|aemia|oma|pathy|plasia|algia|virus|pox|ism|ectomy)$")


def _words_of(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.casefold())


def person_name(text: str) -> bool:
    """Shaped like a person's name in any script and case ("Ana Lopez", "José García", "Nguyễn Văn
    An", "Maria de la Cruz", "Dr. Ana Lopez", "LOPEZ, ANA", "J. Smith") or initials, and holding no
    word of a public-health label. A binomial Latin name ("Shigella sonnei": capitalised genus,
    lower-case species) is not a person's name."""
    text = _text(text)
    if _INITIALS.match(text):
        return True
    words = _words_of(text)
    if set(words) & _NOT_NAME_WORDS or any(_MEDICAL_ENDING.search(w) for w in words) or _race(text):
        return False
    parts = text.replace(",", " ").split()
    titled = bool(parts) and parts[0].rstrip(".").casefold() in _TITLES
    while parts and parts[0].rstrip(".").casefold() in _TITLES:
        parts = parts[1:]
    if titled and 1 <= len(parts) <= 4 and all(_NAME_PART.match(p.rstrip(".")) for p in parts):
        return parts[0][:1].isupper() or parts[0][:1].upper() == parts[0][:1].lower()  # "Dr. Patel", "Mrs Jones"
    if "," in text and _UNICODE_LAST_FIRST.match(text):
        return True  # "Del Rio, Ana", "Van Buren, Martin", "LOPEZ, ANA"
    if not 2 <= len(parts) <= 5 or parts[-1].casefold() in _PARTICLES:
        return False
    if parts[0].casefold() in _PARTICLES and not parts[0][:1].isupper():
        return False  # a lower-case particle never starts a name; "Le Van Anh" and "Del Rio" do
    core = [p for p in parts if not (p.casefold() in _PARTICLES and p.islower())]
    if not all(_NAME_PART.match(p.rstrip(".")) or _INITIAL_PART.match(p) for p in core):
        return False
    if not any(_NAME_PART.match(p) and len(p) >= 2 for p in core) or len(core) < 2:
        return False
    if all(_INITIAL_PART.match(p) for p in core):
        return False
    lettered = [p for p in core if not _INITIAL_PART.match(p)]
    if len(core) == 2 and lettered == core and core[0][:1].isupper() and core[1].islower() and core[0][1:].islower():
        return False  # binomial Latin ("Shigella sonnei", "Escherichia coli")
    # Capitalised parts, one case throughout, or a script without case (Arabic, Devanagari).
    same_case = text.isupper() or text.islower() or all(p[:1].isupper() or p[:1].upper() == p[:1].lower() for p in core)
    return bool(same_case)


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
    name_tokens = set(tokens(column))
    if "time" in kinds and person_event_named(column):
        return KeyVerdict(False, kinds, False)  # "birth_year", "visit_month": a person's own dates
    if present and len(strict) == len(present):
        return KeyVerdict(True, kinds, False)
    for alias, vals in alias_values.items():
        alias_classes = [classify_value(v, alias) for v in vals]
        if vals and all(c is not None for c in alias_classes):
            return KeyVerdict(True, frozenset(c for c in alias_classes if c), False)
    rest = [v for v, c in zip(present, classes, strict=True) if c is None]
    if name_tokens & _LABEL_TOKENS and not name_tokens & _CLINICIAN_TOKENS:
        labelled = len(present) >= 2 and labels_ok(rest, column)
        # A place among labels ("Texas" in a cause column) never makes the column a place key.
        return KeyVerdict(labelled, kinds - {"place"}, labelled)
    # A place, time or demographic column may carry a minority of plain labels beside its values.
    mostly = len(strict) * 4 >= len(present) * 3 and kinds - {"total", "placeholder"}
    return KeyVerdict(bool(mostly) and all(is_label(v, column) for v in rest), kinds, False)


def labels_ok(values: Sequence[Any], column: str) -> bool:
    """Label values: every one label-shaped (no person or record word, ID or code), and fewer than
    three in ten reading as people's names (one odd label such as "Selected Hospitals" is not a
    names column; a column of patients' names is)."""
    present = [v for v in values if v is not None]
    if not all(label_shaped(v, column) for v in present):
        return False
    return sum(person_name(_text(v)) for v in present) < len(present) * config.ESTIMATE_NAME_SHARE_LIMIT


def other_column_problem(column: str, values: Sequence[Any], *, is_numeric: bool, id_alias: bool) -> bool:
    """A non-key column (attribute, alias, one-value column, numeric column) that reveals people:
    SSN, MRN or case-number shapes, ID-like codes, person names, or a name that says patient ID.
    ``id_alias``: a code column 1:1 with an accepted key column (GBD location_id), which is fine."""
    name_tokens = set(tokens(column))
    if person_id_named(column):
        return True  # "subject_id" of year-shaped values is still a subject ID
    if is_numeric:
        return False  # "Patient Days", "Member Months", "Beneficiary Count" are values
    if name_tokens & config.ESTIMATE_PERSON_TOKENS:
        return True  # "Client type", "member": a text column about people
    present = [_text(v) for v in values if v is not None]
    if not present:
        return False
    if name_tokens and name_tokens <= _NAME_COLUMN_TOKENS:
        # "name", "full_name": people's names, unless every value is a place (Census NAME).
        return not all(classify_value(v, column) == "place" for v in present)
    if name_tokens & _CLINICIAN_TOKENS and any(person_name(v) for v in present):
        return True  # "Provider": "José García"
    if person_event_named(column) and any(classify_value(v, column) == "time" for v in present):
        return True  # "dob", "visit_date": a person's own dates
    if any(mixed_script(v) or not _latin(v) for v in present):
        return True  # lookalike letters, or text in a script these tables never use (fail closed)
    if all(classify_value(v, column) is not None for v in present):
        return False  # years, FIPS, states, totals: public values whatever the column is called
    if any(_SSN.match(v) or _MRN.match(v) or (_CASE_NUMBER.match(v) and not _season(v)) for v in present):
        return not id_alias
    if not id_alias and sum(person_name(v) for v in present) >= len(present) * config.ESTIMATE_NAME_SHARE_LIMIT:
        return True
    if not id_alias and sum(id_shaped(v) for v in present) * 2 >= len(present):
        return True
    return False


def year_with_period(columns: Sequence[str], listed_by: Mapping[str, Sequence[Any]]) -> bool:
    """Two time keys that are one time dimension: a year column beside week numbers (FluView,
    NREVSS: YEAR x WEEK) or month numbers 1-12 (Year x Month)."""
    if len(columns) != 2:
        return False
    years = [c for c in columns if all(v is None or _YEAR.match(_text(v)) for v in listed_by[c])]
    if len(years) != 1:
        return False
    (period,) = [c for c in columns if c not in years]
    names = set(tokens(period))
    values = [_text(v) for v in listed_by[period] if v is not None]
    if not values or not all(v.isdigit() for v in values):
        return False
    if names & {"week", "wk", "mmwr"}:
        return all(1 <= int(v) <= 53 for v in values)
    if names & {"month", "mo"}:
        return all(1 <= int(v) <= 12 for v in values)
    return False


def person_id_named(column: str) -> bool:
    """A column named as a person or record ID ("subject_id", "patient_id", "mrn", "MemberNo"),
    which refuses the table whatever its values (ADR014 §7)."""
    names = tokens(column)
    name_tokens = set(names)
    joined = "".join(names)
    person = name_tokens & config.ESTIMATE_PERSON_TOKENS
    return (
        joined in config.ESTIMATE_PERSON_NAMES
        or _patient_key_name(joined)
        or (bool(person) and bool(name_tokens & _ID_SUFFIX))
        or (bool(names) and name_tokens <= config.ESTIMATE_PERSON_TOKENS)  # "mrn", "patient", "Client"
    )


def _patient_key_name(joined: str) -> bool:
    """Named for the patient-ID role (the contract inference's role names, ``names_patient``).
    A privacy guard on tables declared public: it refuses, it never picks a key."""
    from clinical_analytics.contracts.inference import names_patient

    return names_patient(joined)


def id_named(column: str) -> bool:
    """A column name that ends in an ID word (location_id, State Code)."""
    names = tokens(column)
    joined = "".join(t for t in names if not t.isdigit())
    return bool(set(names) & _ID_SUFFIX) or joined.endswith(("id", "code"))  # "topicid", "stratificationid1"

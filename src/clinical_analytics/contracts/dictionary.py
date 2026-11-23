# SPDX-License-Identifier: MIT
"""Read a data dictionary (the upload's documentation text) into per-column facts.

Only columns the data has are read: an entry starts on a line whose first word is a column name
("BRACE_TP    Brace worn after the fall:"), and runs over the indented lines under it. The lines
that belong to no entry are prose, where the file states what a row is ("One row per clinic
attendance", "Unit of analysis: kennel stay") and who the patient is ("'owner_ref' = patient").

Code labels: a label runs from its code's "=" to the next code of the same list, so commas,
slashes, "=" and parentheses inside a label stay in it ("2 = splint (thermoplastic/padded, worn
at night)"). A code list starts after a line start, a comma, a semicolon, a colon, a sentence
end or an opening parenthesis; a list inside parentheses ends at the parenthesis that closes it.
One pair alone ("1 = first fitting") is a note, not a code list, unless its label says the value
was not recorded ("-5 = sheet not returned").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from clinical_analytics.contracts import config
from clinical_analytics.contracts.dates import format_from_text
from clinical_analytics.contracts.units import is_atom

Direction = Literal["lower_is_better", "higher_is_better"]

# "code = label"; a number may also be written "code: label" ("1: Yes 2: No").
_CODE: Final = re.compile(
    r"((?<![\w.])-\d{1,5}|[A-Za-z0-9][A-Za-z0-9.+\-]{0,7})\s*=(?!=)\s*|(?<![\w.:])(\d{1,4})\s*:\s*(?=[A-Za-z])"
)
_LIST_START: Final = frozenset("\n,;:(.*†‡•")
_BOUNDARY_ONLY: Final = frozenset({"blank", "blanks", "empty"})  # "blank = not taken": no code value
_RANGE: Final = re.compile(r"(?<![\w.^])(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)(?![\w.%/^])")
_PAREN: Final = re.compile(r"\([^()]*\)")
_SENTENCE_END: Final = re.compile(r"(?<=[.;!?])\s+(?=[A-Z0-9\-])")
# The patient must be what the identifier identifies: "patient identifier", "(patient)",
# "identifies the patient", "identifier of the patient". "Identifier of the technician who ran
# the pump for the patient" names the technician.
_PATIENT_KEY_TEXT: Final = re.compile(
    r"\(\s*patient\s*\)|\bpatient(?:'s)?\s+(?:id|identifier|number|key|record\s+number)\b|"
    r"\bidentif(?:ies|ier)\s+(?:(?:of|for)\s+)?(?:the|each|a)\s+patients?\b|"
    r"\b(?:id|identifier|number)\s+of\s+(?:the|each|a)\s+patient\b|"
    r"\b(?:respondent|participant|subject|patient|person)\s+(?:sequence|seq\.?|id|identification)\s+(?:number|no\.?)\b",
    re.IGNORECASE,
)
_KEY_STATEMENT: Final = (
    re.compile(
        r"['\"‘’`]?\b(?P<col>[A-Za-z_][\w]*)\b['\"‘’`]?\s*(?:=|is|identifies|names)\s*(?:the\s+|a\s+)?"
        r"(?:patient|subject|participant)\b(?!['’]s)",
        re.IGNORECASE,
    ),
    re.compile(r"\bpatient\s+(?:id|identifier|key)\s+is\s+['\"‘’`]?(?P<col>[A-Za-z_][\w]*)", re.IGNORECASE),
)
_GRAIN_STATEMENT: Final = (
    re.compile(
        r"\b(?:one|each|1)\s+(?:row|line|record|observation)s?\s*(?:per|=|is|represents)\s*"
        r"(?:one\s+|an?\s+|1\s+)?(?P<noun>[^.;:,()\n]+)",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*rows?\s*=\s*(?:one\s+|an?\s+)?(?P<noun>[^.;:,()\n]+)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bunit of (?:observation|analysis)\s*[:=]?\s*(?:the\s+|one\s+)?(?P<noun>[^.;:,()\n]+)", re.IGNORECASE),
)
_NOUN_STOP: Final = re.compile(
    r"\s+(?:that|which|who|where|with|for|in|at|from|and|per|when|to|of|during|after|before|on|by|if|whose)\b.*$",
    re.IGNORECASE,
)
_MAX_NOUN_WORDS: Final = 4
# Markup around a dictionary's words (Markdown emphasis, code, links): never part of a noun.
_MARKUP: Final = re.compile(r"[*_`#>\[\]{}|~]+")
# English past participles: regular "-ed" forms and common irregular ones (a closed class of the
# language, not of any file). Irregular forms that are also everyday nouns ("set", "run", "lead")
# are left out: a noun phrase is never cut at a word that may belong to it.
_PARTICIPLE: Final = re.compile(
    r"[a-z]{3,}ed|given|taken|seen|done|drawn|held|made|kept|sent|written|known|shown|grown|"
    r"brought|bought|caught|taught|thought|found|paid|told"
)
# A word of a noun phrase: letters, with an inner hyphen or apostrophe ("follow-up", "child's").
_WORD: Final = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_SAYS_AGGREGATE: Final = re.compile(
    r"\b(?:counts?|number|numbers|totals?|tally|tallies)\s+of\s+[a-z ]+?\s+by\s+[a-z]|"
    r"\b(?:summary|aggregate[d]?|frequency)\s+(?:table|data|counts)\b|\bno\s+(?:patient|person)[- ]level\b",
    re.IGNORECASE,
)
_ONE_PER_PERSON: Final = re.compile(
    r"\bone\s+(?:row|line|record)?\s*per\s+(?:patient|person|participant|subject)\b", re.IGNORECASE
)
_REPEATS: Final = re.compile(r"\b(?:more than once|repeat(?:s|ed)?|several (?:rows|times|visits))\b", re.IGNORECASE)
_DIRECTION: Final = re.compile(
    r"\b(?P<side>lower|higher)\s+(?:is|=|scores?\s+(?:are|is|indicate|mean)|values?\s+(?:are|indicate|mean))\s+"
    r"(?P<verdict>better|worse)\b",
    re.IGNORECASE,
)
_RATIO_UNIT: Final = re.compile(r"(?<![\w/])((?:x10\^\d+|[A-Za-zµμ%°]+\d?)/(?:[A-Za-z0-9µμ²]+))(?![\w/])")
_WORD_UNIT: Final = re.compile(
    r"(?:^|,\s*|\(\s*|\bin\s+|;\s*)(minutes|hours|days|weeks|months|years|grams|kilograms|kg|mg|g|mL|cm|mm|"
    r"percent|%|seconds|mmHg|cmH2O|bpm|beats per minute|breaths per minute|IU|U)(?![\w/])",
    re.IGNORECASE,
)
# A unit the dictionary states in so many words, whatever it is ("Units: AU", "unit of measure:
# points", "expressed as a ratio (A/B)").
_STATED_UNIT: Final = re.compile(
    r"\b(?:units?(?:\s+of\s+measure)?|measured\s+in|expressed\s+in)\s*[:=]\s*(?P<unit>[^\s,;()]+)|"
    r"\bratio\s*\(\s*(?P<ratio>[A-Za-z][\w.\-]*/[A-Za-z][\w.\-]*)\s*\)",
    re.IGNORECASE,
)
# A ratio is a unit only when one side is a unit atom (``contracts.units``: "mmol/L", "x10^9/L",
# "kg/m2"); "Yes/No" is a label set, never a unit.
_DATE_WORDS: Final = re.compile(r"\b(?:YYYY|YY|MM|DD|HH|Mon)\b")


@dataclass(frozen=True)
class DictionaryEntry:
    column: str
    text: str
    description: str  # the text before the first code list
    labels: dict[str, str] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)
    stated_range: tuple[float, float] | None = None
    unit: str | None = None
    date_format: str | None = None
    direction: Direction | None = None
    names_patient_key: bool = False
    # "Not recorded" codes the entry states in a sentence, not its code list ("Missing values
    # were coded as -9"): asked about, never applied (code -> the sentence).
    noted_missing: dict[str, str] = field(default_factory=dict)
    # From a codebook table: a coded item's non-response codes ("7 = Refused"), and labelled values
    # that stand for a range ("65 = 65 hours or more").
    nonresponse: dict[str, str] = field(default_factory=dict)
    censored: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Dictionary:
    entries: dict[str, DictionaryEntry] = field(default_factory=dict)
    grain_noun: str | None = None
    grain_text: str | None = None
    key_statements: tuple[str, ...] = ()
    one_per_person: bool = False
    repeats_stated: bool = False
    # Codes a prose note states for no column in particular ("For all measurements, 999 = not
    # recorded"), and for columns a sentence names: asked about, never applied (code -> the sentence).
    shared_missing: dict[str, str] = field(default_factory=dict)
    noted_missing: dict[str, dict[str, str]] = field(default_factory=dict)
    # The dictionary says lines are counts for groups ("each row is a count of cases by state").
    says_aggregate: bool = False
    # The dictionary mentions sample or survey weights: the file is a weighted survey sample.
    mentions_weights: bool = False

    def entry(self, column: str) -> DictionaryEntry | None:
        return self.entries.get(column)


def _depths(text: str) -> list[int]:
    depth, out = 0, []
    for ch in text:
        out.append(depth)
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
    return out


def _trim_label(label: str) -> str:
    label = label.strip().rstrip(",;.").strip()
    while label.endswith(")") and label.count(")") > label.count("("):
        label = label[:-1].rstrip(",;. ")
    return label


def parse_pairs(text: str) -> list[tuple[str, str]]:
    """Code/label pairs in order; see the module docstring for where a label ends."""
    depth = _depths(text)
    found: list[tuple[int, int, str, int]] = []
    spaced: list[tuple[int, int, str, int]] = []  # numeric codes after a word ("ECOG 0 = ...", "Smoking 0=never")
    for m in _CODE.finditer(text):
        group = 1 if m.group(1) is not None else 2
        start = m.start(group)
        j = start - 1
        while j >= 0 and text[j] in " \t":
            j -= 1
        prev = text[j] if j >= 0 else "\n"
        listed = prev in _LIST_START and not (prev == "." and j == start - 1)
        # Within a list, a numeric code may follow the previous label after a space ("1: Yes 2: No").
        continues = j < start - 1 and m.group(group).isdigit() and any(f[3] == depth[start] for f in found)
        if not (listed or continues):
            if j < start - 1 and m.group(group).isdigit():
                spaced.append((start, m.end(), m.group(group), depth[start]))
            continue
        found.append((start, m.end(), m.group(group), depth[start]))
    # A numeric code after a word starts a list when there are at least two numeric codes at its
    # depth: "ECOG 0 = fully active; 1 = restricted" keeps 0, "Smoking 0=never 1=former" reads.
    for item in spaced:
        same = [f for f in (*found, *spaced) if f[3] == item[3] and f[2].isdigit()]
        if len(same) >= 2:
            found.append(item)
    found.sort()
    pairs: list[tuple[str, str]] = []
    for i, (_start, end, code, d) in enumerate(found):
        nxt = next((c for c in found[i + 1 :] if c[3] == d), None)
        limit = nxt[0] if nxt is not None else len(text)
        stop = next(
            (k for k in range(end, limit) if depth[k] < d or (text[k] == ")" and depth[k] == d and d > 0)), limit
        )
        label = _trim_label(text[end:stop])
        if code.lower() in _BOUNDARY_ONLY or not label:
            continue
        pairs.append((code, label))
    return pairs


# A "not recorded" code written as a sentence, not "code = label":
# "A value of 999 indicates the measurement was not taken.", "999 is used where no reading was
# available.", "Missing values were coded as -9.", "Coded as 99 when unknown."
_CODE_NUM: Final = r"(?<![\w.\-])(?P<code>-?\d{1,5})(?!\w|\.\d)"
_FREE_MISSING: Final = (
    re.compile(
        _CODE_NUM + r"\s+(?:indicates?|means?|denotes?|signifies|represents?|stands\s+for|"
        r"(?:is|was|are|were)\s+(?:used|recorded|entered|given|coded)\s+(?:where|when|if|for|to\s+(?:mean|indicate))|"
        r"(?:is|was)\s+(?:a\s+)?(?:code|placeholder)\s+for)\s+(?P<label>[^.;\n]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<label>(?:missing|unknown|unrecorded|absent)(?:\s+(?:values?|data|readings?|results?|answers?))?|"
        r"(?:values?|readings?|results?)\s+not\s+(?:recorded|taken|available|measured))\s+"
        r"(?:were|was|are|is|have\s+been|has\s+been)\s+(?:coded|recorded|entered|stored|set|given)\s+(?:as|to)\s+"
        + _CODE_NUM,
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:coded|recorded|entered|set)\s+(?:as|to)\s+" + _CODE_NUM + r"\s+(?:if|when|where)\s+(?P<label>[^.;\n]+)",
        re.IGNORECASE,
    ),
)


def free_missing_pairs(text: str) -> list[tuple[str, str]]:
    """(code, label) of "not recorded" codes stated in sentences (``_FREE_MISSING``), in order."""
    found: list[tuple[int, str, str]] = []
    for pattern in _FREE_MISSING:
        for m in pattern.finditer(text):
            label = m.group("label").strip()
            if says_not_taken(label) and all(code != m.group("code") for _, code, _ in found):
                found.append((m.start(), m.group("code"), label))
    return [(code, label) for _, code, label in sorted(found)]


def is_missing_label(label: str) -> bool:
    """The label says the measurement itself was not recorded ("not recorded", "missing")."""
    return bool(config.MISSING_LABEL_STRICT.search(label)) and not config.MISSING_LABEL_EXCEPT.search(label)


def says_not_taken(label: str) -> bool:
    """A note beside a measurement that says it was not taken ("sheet not returned")."""
    return bool(config.MISSING_LABEL_NOT_TAKEN.search(label)) and not config.MISSING_LABEL_EXCEPT.search(label)


def _outside(code: str, stated: tuple[float, float] | None) -> bool:
    """A numeric code outside the stated range, or (no range) a usual "not recorded" number."""
    try:
        value = float(code)
    except ValueError:
        return False
    if stated is not None:
        return value < stated[0] or value > stated[1]
    return value.is_integer() and int(value) in config.MISSING_CODE_CANDIDATES


def split_missing(
    pairs: list[tuple[str, str]], stated: tuple[float, float] | None
) -> tuple[dict[str, str], dict[str, str]]:
    """(labels, missing codes) of an entry's pairs.

    In a code list, a code is missing only when its label says the value was not recorded and at
    least ``MISSING_CODE_MIN_CATEGORIES`` coded categories remain; every other label is a category
    ("0 = not done, 1 = done"). A lone note on a measurement is a missing code when it says the
    measurement was not taken and its code lies outside the measurement's values.
    """
    if len(pairs) == 1:
        code, label = pairs[0]
        return ({}, {code: label}) if says_not_taken(label) and _outside(code, stated) else ({}, {})
    if all(says_not_taken(label) and _outside(code, stated) for code, label in pairs):
        # Notes on a measurement, every one saying the value was not taken ("999 = not recorded,
        # 888 = refused"): missing-value codes, however many, never a code list of the measure.
        return {}, dict(pairs)
    missing = {code: label for code, label in pairs if is_missing_label(label)}
    labels = {code: label for code, label in pairs if code not in missing}
    if len(labels) < config.MISSING_CODE_MIN_CATEGORIES:
        return dict(pairs), {}
    return labels, missing


def _stated_range(description: str) -> tuple[float, float] | None:
    for text in (description, _PAREN.sub(" ", description)):
        for m in _RANGE.finditer(text):
            low, high = float(m.group(1)), float(m.group(2))
            if low < high:
                return low, high
    return None


def _unit(description: str) -> str | None:
    stated = _STATED_UNIT.search(description)
    if stated:
        return stated.group("unit") or stated.group("ratio")
    for m in _RATIO_UNIT.finditer(description):
        unit = m.group(1)
        parts = {p.lower().replace("µ", "u").replace("μ", "u").rstrip("²") for p in re.split(r"/", unit)}
        if _DATE_WORDS.search(unit) or not (any(is_atom(p) for p in parts if not p.startswith("x10")) or "x10" in unit):
            continue
        return unit
    word = _WORD_UNIT.search(description)
    if word is None:
        return None
    unit = word.group(1)
    # "Days" at a sentence start is "days"; "mmHg", "IU" and "U" keep their case.
    return unit if len(unit) == 1 or any(c.isupper() for c in unit[1:]) else unit.lower()


def _direction(text: str) -> Direction | None:
    m = _DIRECTION.search(text)
    if m is None:
        return None
    lower_better = (m.group("side").lower() == "lower") == (m.group("verdict").lower() == "better")
    return "lower_is_better" if lower_better else "higher_is_better"


def _entry(column: str, text: str) -> DictionaryEntry:
    pairs = parse_pairs(text)
    listed = {code for code, _ in pairs}
    noted = {
        code: sentence
        for sentence in _SENTENCE_END.split(text)
        for code, _label in free_missing_pairs(sentence)
        if code not in listed
    }
    first = re.search(r"-?[A-Za-z0-9][A-Za-z0-9.+\-]{0,7}\s*=(?!=)", text)
    description = text[: first.start()] if first else text
    date_format = format_from_text(text)
    stated = None if date_format else _stated_range(description)
    labels, missing = split_missing(pairs, stated)
    return DictionaryEntry(
        column=column,
        text=text,
        description=description.strip(),
        labels=labels if len(labels) >= 2 else {},
        missing=missing,
        stated_range=stated,
        unit=None if date_format else _unit(description),
        date_format=date_format,
        direction=_direction(text),
        names_patient_key=bool(_PATIENT_KEY_TEXT.search(description)),
        noted_missing=noted,
    )


def _entry_start(line: str, by_lower: dict[str, str]) -> tuple[str, str] | None:
    """(column, rest) when the line starts an entry: a column name, then space, ':' or ' -'."""
    if not line or line[0] in " \t":
        return None
    # A column name with spaces ("Cause Name"), longest first: then two spaces, a tab or ":".
    lowered = line.lower()
    for name in sorted((n for n in by_lower if " " in n), key=len, reverse=True):
        if lowered.startswith(name):
            rest = line[len(name) :]
            if rest == "" or rest.startswith(("  ", "\t", ":")):
                return by_lower[name], rest.lstrip(" \t:")
    m = re.match(r"^['\"`]?([^\s:'\"`]+)['\"`]?\s*(?::|\s-\s|\s–\s|\t|\s)\s*(.*)$", line)
    if m is None:
        return None
    column = by_lower.get(m.group(1).lower())
    return (column, m.group(2)) if column is not None else None


def _noun(text: str) -> str:
    """The noun phrase of a grain statement, whole or not at all: "clinic attendance" from "clinic
    attendance that was booked". A phrase longer than ``_MAX_NOUN_WORDS`` is dropped, never cut
    mid-phrase (the line is then a "record")."""
    plain = _MARKUP.sub(" ", text)
    words = _NOUN_STOP.sub("", plain.strip()).strip(" -").split()
    # A noun phrase ends before a participle ("dose given to the child" -> "dose").
    cut = next((i for i, w in enumerate(words) if i > 0 and _PARTICIPLE.fullmatch(w.lower())), len(words))
    words = [w for w in words[:cut] if _WORD.fullmatch(w)]
    return " ".join(words) if 0 < len(words) <= _MAX_NOUN_WORDS and len(words) == cut else ""


_TABLE_RULE: Final = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _table_rows_as_entries(text: str) -> str:
    """A Markdown table row ``| column | description | codes |`` as an entry line
    ``column  description; codes`` (``<br>`` inside a cell as a new line); other lines unchanged."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _TABLE_RULE.match(stripped):
            continue
        if stripped.startswith("|") and stripped.count("|") >= 3:
            cells = [c.strip().strip("`*") for c in stripped.strip("|").split("|")]
            if cells and cells[0]:
                rest = "; ".join(c for c in cells[1:] if c)
                parts = re.split(r"<br\s*/?>", rest, flags=re.IGNORECASE)
                out.append(f"{cells[0]}  {parts[0].strip()}")
                out.extend(f"    {p.strip()}" for p in parts[1:] if p.strip())
                continue
        out.append(line)
    return "\n".join(out)


# ---- Codebook tables (SAS/SPSS/NCHS style): a variable header, then rows of
# "Code or Value | Value Description | Count | Cumulative", including "1 to 79 | Range of Values".
_CODEBOOK_HEADER: Final = re.compile(
    r"\b(?:code\s+or\s+value|value|code)\b.*\b(?:value\s+)?(?:description|label)\b", re.I
)
_CELL_SPLIT: Final = re.compile(r"\s*\|\s*|\t+|\s{2,}")
_VARIABLE_LINE: Final = re.compile(
    r"^\s*(?:(?:sas\s+)?variable(?:\s+name)?\s*[:=]\s*)?(?P<name>[A-Za-z_][\w.]*)\s*(?:$|[:\-–]\s+(?P<label>.*))", re.I
)
_LABEL_LINE: Final = re.compile(
    r"^\s*(?:sas\s+label|label|english\s+text|question|description)\s*:\s*(?P<text>.+)", re.I
)
_RANGE_ROW: Final = re.compile(r"^(-?\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*(-?\d+(?:\.\d+)?)$")
_NONRESPONSE: Final = re.compile(
    r"\b(?:refused|don['’]?t\s+know|not\s+asked|missing|not\s+ascertained|no\s+answer|unknown|not\s+applicable|"
    r"skipped|legitimate\s+skip)\b",
    re.I,
)
_EXPLICIT_NONRESPONSE: Final = re.compile(
    r"\brefused\b|\bdon['’]?t\s+know\b|\bdo\s+not\s+know\b|\bnot\s+asked\b|^\s*(?:missing|not\s+recorded)\s*\.?\s*$",
    re.I,
)
_RANGE_OF_VALUES: Final = re.compile(r"range\s+of\s+values?", re.I)
# A range row whose label restates its own range in brackets ("Number of hours [1-24]").
_OWN_RANGE: Final = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(-?\d+(?:\.\d+)?)\s*\]")
_CENSORED: Final = re.compile(
    r"\b(?:or\s+(?:older|more|over|above|higher|less|fewer|younger)|and\s+(?:over|older|above)|less\s+than|"
    r"more\s+than|under|at\s+least|at\s+most)\b|\d\s*\+",
    re.I,
)


def is_explicit_nonresponse(label: str) -> bool:
    """ "Refused", "Don't know", "Not asked", or a label that is exactly "Missing"/"Not recorded"."""
    return bool(_EXPLICIT_NONRESPONSE.search(label))


def is_nonresponse(label: str) -> bool:
    """A label that may mean the value is absent ("not applicable", "unknown", "skipped")."""
    return bool(_NONRESPONSE.search(label)) or is_explicit_nonresponse(label)


def _cells(line: str) -> list[str]:
    return [c for c in (part.strip() for part in _CELL_SPLIT.split(line.strip().strip("|"))) if c]


def _codebook_entries(text: str, by_lower: dict[str, str]) -> dict[str, DictionaryEntry]:
    """Per-column entries read from codebook tables (labels, the value range, non-response and
    range codes): structured per column, so they apply like a column's own code list. A variable
    header with a label and no table ("Variable Name: RESPNO / Respondent sequence number") gives
    an entry of its label."""
    found: dict[str, DictionaryEntry] = {}
    labels: dict[str, str] = {}
    lines = text.splitlines()
    current: str | None = None
    pending = ""  # a label line before its variable line ("Label: ..." then "SAS Variable Name: ...")
    i = 0
    while i < len(lines):
        line = lines[i]
        header = _VARIABLE_LINE.match(line)
        if header and header.group("name").lower() in by_lower:
            current = by_lower[header.group("name").lower()]
            labels[current] = (header.group("label") or "").strip() or pending
            pending = ""
        elif (labelled := _LABEL_LINE.match(line)) and current is not None and not labels[current]:
            labels[current] = labelled.group("text").strip()
        elif labelled is not None and current is None:
            pending = labelled.group("text").strip()
        elif current is not None and _CODEBOOK_HEADER.search(line) and len(_cells(line)) >= 2:
            rows: list[tuple[str, str]] = []
            i += 1
            while i < len(lines) and (_TABLE_RULE.match(lines[i]) or len(cells := _cells(lines[i])) >= 2):
                if not _TABLE_RULE.match(lines[i]):
                    rows.append((cells[0], cells[1]))
                i += 1
            found[current] = _codebook_entry(current, labels.pop(current), rows)
            current = None
            continue
        i += 1
    for column, label in labels.items():
        if label and column not in found:
            found[column] = _entry(column, label)
    return found


def _codebook_entry(column: str, label: str, rows: list[tuple[str, str]]) -> DictionaryEntry:
    """A codebook variable's entry. Applied automatically (ADR017 principle A) only what the
    codebook states exactly: its "Range of Values" row, explicit non-response labels ("Refused",
    "Don't know", "Not asked", or exactly "Missing"/"Not recorded"), and on a measure, range codes
    whose label names the edge of that range ("65 = 65 hours or more" beside 0 to 64). Any other
    label that may mean not recorded ("Unknown side", "Not applicable") is a note to ask about; on
    a coded item it stays a category until answered."""
    stated: tuple[float, float] | None = None
    coded: dict[str, str] = {}
    for code, description in rows:
        ranged = _RANGE_ROW.match(code)
        own = _OWN_RANGE.search(description)
        if ranged and (
            _RANGE_OF_VALUES.fullmatch(description.strip()) or (own is not None and own.groups() == ranged.groups())
        ):
            stated = (float(ranged.group(1)), float(ranged.group(2)))
        elif code != "." and re.fullmatch(r"-?\d+(?:\.\d+)?|[A-Za-z]", code):
            coded[code] = description
    explicit = {c: d for c, d in coded.items() if _EXPLICIT_NONRESPONSE.search(d)}
    asked = {c: f"{c} = {d}" for c, d in coded.items() if c not in explicit and _NONRESPONSE.search(d)}
    if stated is not None:
        # A measure ("0 to 79 | Range of Values"): explicit non-response codes are missing values,
        # range codes at its edge are real values; nothing else is a category.
        censored = {c: d for c, d in coded.items() if c not in explicit and _at_range_edge(d, stated)}
        return DictionaryEntry(
            column=column, text=label, description=label, missing=explicit, censored=censored, stated_range=stated,
            unit=_unit(label), noted_missing=asked, direction=_direction(label),
        )  # fmt: skip
    return DictionaryEntry(
        column=column, text=label, description=label, labels=coded if len(coded) >= 2 else {}, nonresponse=explicit,
        noted_missing=asked, unit=None if coded else _unit(label), direction=_direction(label),
    )  # fmt: skip


def _at_range_edge(description: str, stated: tuple[float, float]) -> bool:
    """ "65 hours or more" beside 0 to 64, "less than 1 hour" beside 1 to 64: a boundary word with
    a number at the edge of the measure's own range."""
    if not _CENSORED.search(description):
        return False
    low, high = stated
    edges = {low, high, high + 1, low - 1}
    return any(float(n) in edges for n in re.findall(r"-?\d+(?:\.\d+)?", description))


# The codebook says weights must be applied to the counts ("Weights must be used to produce
# estimates", "apply the sampling weight for representative figures"); a passing mention is not.
_WEIGHTS: Final = re.compile(
    r"\bweights?\s+(?:must|should|need\s+to|have\s+to)\s+be\s+(?:used|applied)\b|"
    r"\b(?:use|apply|applying|using)\s+(?:the\s+|a\s+)?(?:\w+\s+){0,2}weights?\b[^.]*\b(?:estimates?|representative)\b",
    re.IGNORECASE,
)


def parse_dictionary(text: str | None, columns: Sequence[str]) -> Dictionary:
    """The dictionary facts for ``columns`` found in ``text`` (empty when there is no text)."""
    if not text or not text.strip():
        return Dictionary()
    by_lower = {c.lower(): c for c in columns}
    blocks: dict[str, list[str]] = {}
    prose: list[str] = []
    current: str | None = None
    for raw in _table_rows_as_entries(text).splitlines():
        line = raw.rstrip()
        started = _entry_start(line, by_lower)
        if started is not None:
            current = started[0]
            blocks.setdefault(current, []).append(started[1])
        elif current is not None and line[:1] in (" ", "\t") and line.strip():
            blocks[current].append(line.strip())
        else:
            current = None
            if line.strip():
                prose.append(line.strip())
    entries = {column: _entry(column, "\n".join(lines)) for column, lines in blocks.items()}
    for variable, codebook in _codebook_entries(text, by_lower).items():
        if codebook.labels or codebook.stated_range or variable not in entries:
            entries[variable] = codebook
    prose_text = "\n".join(prose)
    grain_noun = grain_text = None
    for pattern in _GRAIN_STATEMENT:
        m = pattern.search(prose_text)
        if m is not None and _noun(m.group("noun")):
            grain_noun, grain_text = _noun(m.group("noun")), m.group(0).strip()
            break
    statements: list[str] = []
    for pattern in _KEY_STATEMENT:
        for m in pattern.finditer(prose_text):
            column = by_lower.get(m.group("col").lower())
            if column is not None and column not in statements:
                statements.append(column)
    shared: dict[str, str] = {}
    noted: dict[str, dict[str, str]] = {}
    for line in prose:
        for sentence in _SENTENCE_END.split(line):
            notes = [*parse_pairs(sentence), *free_missing_pairs(sentence)]
            notes = [(code, label) for code, label in notes if says_not_taken(label) and _outside(code, None)]
            if not notes:
                continue
            named = [
                c
                for lower, c in by_lower.items()
                if len(lower) >= 3 and re.search(rf"(?<![\w]){re.escape(lower)}(?![\w])", sentence.lower())
            ]
            for code, _label in notes:
                # "In sbp, 999 indicates the reading was not taken" is about sbp; a sentence that
                # names no column, about every one. Either way the sentence is quoted and asked.
                for column in named:
                    noted.setdefault(column, {}).setdefault(code, sentence.strip())
                if not named:
                    shared.setdefault(code, sentence.strip())
    return Dictionary(
        shared_missing=shared,
        noted_missing=noted,
        says_aggregate=bool(_SAYS_AGGREGATE.search(prose_text)),
        mentions_weights=bool(_WEIGHTS.search(text)),
        entries=entries,
        grain_noun=grain_noun,
        grain_text=grain_text,
        key_statements=tuple(statements),
        one_per_person=bool(_ONE_PER_PERSON.search(prose_text)),
        repeats_stated=bool(_REPEATS.search(prose_text)),
    )

# SPDX-License-Identifier: MIT
"""Read a data dictionary (the upload's documentation text) into per-column facts.

Only columns the data has are read: an entry starts on a line whose first word is a column name
("SRC_CD    Suspected source of infection:"), and runs over the indented lines under it. The
lines that belong to no entry are prose, where the file states what a row is ("One row per ED
encounter", "Unit of observation: patient") and who the patient is ("'subject' = patient").

Code labels: a label runs from its code's "=" to the next code of the same list, so commas,
slashes, "=" and parentheses inside a label stay in it ("2 = AC-T (doxorubicin/cyclophosphamide
then paclitaxel, q21 days)"). A code list starts after a line start, a comma, a semicolon, a
colon, a sentence end or an opening parenthesis; a list inside parentheses ends at the
parenthesis that closes it. One pair alone ("1 = first cycle") is a note, not a code list,
unless its label says the value was not recorded ("99 = questionnaire not done").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from clinical_analytics.contracts import config
from clinical_analytics.contracts.dates import format_from_text

Direction = Literal["lower_is_better", "higher_is_better"]

# "code = label"; a number may also be written "code: label" ("1: Yes 2: No").
_CODE: Final = re.compile(r"([A-Za-z0-9][A-Za-z0-9.+\-]{0,7})\s*=(?!=)\s*|(?<![\w.:])(\d{1,4})\s*:\s*(?=[A-Za-z])")
_LIST_START: Final = frozenset("\n,;:(.*†‡•")
_BOUNDARY_ONLY: Final = frozenset({"blank", "blanks", "empty"})  # "blank = not drawn": no code value
_RANGE: Final = re.compile(r"(?<![\w.^])(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)(?![\w.%/^])")
_PAREN: Final = re.compile(r"\([^()]*\)")
# The patient must be what the identifier identifies: "patient identifier", "(patient)",
# "identifies the patient", "identifier of the patient". "Identifier of the surgeon who operated
# on the patient" names the surgeon.
_PATIENT_KEY_TEXT: Final = re.compile(
    r"\(\s*patient\s*\)|\bpatient(?:'s)?\s+(?:id|identifier|number|key|record\s+number)\b|"
    r"\bidentif(?:ies|ier)\s+(?:(?:of|for)\s+)?(?:the|each|a)\s+patients?\b|"
    r"\b(?:id|identifier|number)\s+of\s+(?:the|each|a)\s+patient\b",
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
_SAYS_AGGREGATE: Final = re.compile(
    r"\b(?:counts?|number|numbers|totals?|tally|tallies)\s+of\s+[a-z ]+?\s+by\s+[a-z]|"
    r"\b(?:summary|aggregate[d]?|frequency)\s+(?:table|data|counts)\b|\bno\s+(?:patient|person)[- ]level\b",
    re.IGNORECASE,
)
_ONE_PER_PERSON: Final = re.compile(
    r"\beach\s+(?:woman|patient|person|participant|subject|mother|infant|child)\b[^.]*\bone\b|"
    r"\bfirst\s+\w+\s+only\b|\bone\s+per\s+(?:woman|patient|person|participant)\b",
    re.IGNORECASE,
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
# Clinical units that are not made of unit words ("Kt/V" for dialysis adequacy, "INR").
_NAMED_UNIT: Final = re.compile(r"(?<![\w/])(Kt/V|INR|MET-min/week|METs|IU/mL)(?![\w/])")
# A ratio is a unit only when one side is a unit word ("mmol/L", "x10^9/L", "kg/m2"); "Yes/No" is
# a label set, never a unit.
_UNIT_PARTS: Final = frozenset(
    {"mg", "g", "kg", "mcg", "ug", "µg", "ng", "pg", "mmol", "umol", "µmol", "mol", "meq", "l", "dl", "ml", "min",
     "h", "hr", "day", "d", "wk", "week", "m2", "cm2", "m", "iu", "u", "mu", "copies", "cells", "mm3", "ul", "µl",
     "s", "sec", "kpa", "mmhg", "beats", "breaths"}
)  # fmt: skip
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


@dataclass(frozen=True)
class Dictionary:
    entries: dict[str, DictionaryEntry] = field(default_factory=dict)
    grain_noun: str | None = None
    grain_text: str | None = None
    key_statements: tuple[str, ...] = ()
    one_per_person: bool = False
    repeats_stated: bool = False
    # Codes a note documents for every measurement ("For all measurements, 999 = not recorded").
    shared_missing: dict[str, str] = field(default_factory=dict)
    # The dictionary says lines are counts for groups ("each row is a count of cases by state").
    says_aggregate: bool = False

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


def is_missing_label(label: str) -> bool:
    """The label says the measurement itself was not recorded ("not recorded", "missing")."""
    return bool(config.MISSING_LABEL_STRICT.search(label)) and not config.MISSING_LABEL_EXCEPT.search(label)


def says_not_taken(label: str) -> bool:
    """A note beside a measurement that says it was not taken ("questionnaire not done")."""
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
    named = _NAMED_UNIT.search(description)
    if named:
        return named.group(1)
    for m in _RATIO_UNIT.finditer(description):
        unit = m.group(1)
        parts = {p.lower() for p in re.split(r"/", unit)}
        if _DATE_WORDS.search(unit) or not (
            {p for p in parts if not p.startswith("x10")} & _UNIT_PARTS or "x10" in unit
        ):
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
    first = re.search(r"[A-Za-z0-9][A-Za-z0-9.+\-]{0,7}\s*=(?!=)", text)
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
    """The noun phrase of a grain statement, whole or not at all: "ED encounter" from "ED
    encounter that met the alert". A phrase longer than ``_MAX_NOUN_WORDS`` ends at its last word
    that names what a row is ("emergency department patient visit" -> "patient visit"), never
    cut mid-phrase; with no such word it is dropped."""
    from clinical_analytics.contracts.inference import GRAIN_KIND_WORDS

    words = _NOUN_STOP.sub("", text.strip()).strip(" -").split()
    named = [w for w in words if w.lower().rstrip("s") in GRAIN_KIND_WORDS or w.lower() in GRAIN_KIND_WORDS]
    if len(words) <= _MAX_NOUN_WORDS and (not named or words[-1] in named):
        return " ".join(words)
    last = next(
        (
            i
            for i in range(len(words) - 1, -1, -1)
            if words[i].lower().rstrip("s") in GRAIN_KIND_WORDS or words[i].lower() in GRAIN_KIND_WORDS
        ),
        None,
    )
    if last is None:
        return ""
    return " ".join(words[max(0, last - 1) : last + 1])


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
    shared = {
        code: label
        for line in prose
        for code, label in parse_pairs(line)
        if says_not_taken(label) and _outside(code, None)
    }
    return Dictionary(
        shared_missing=shared,
        says_aggregate=bool(_SAYS_AGGREGATE.search(prose_text)),
        entries=entries,
        grain_noun=grain_noun,
        grain_text=grain_text,
        key_statements=tuple(statements),
        one_per_person=bool(_ONE_PER_PERSON.search(prose_text)),
        repeats_stated=bool(_REPEATS.search(prose_text)),
    )

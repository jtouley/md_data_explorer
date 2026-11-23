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
_LIST_START: Final = frozenset("\n,;:(.")
_BOUNDARY_ONLY: Final = frozenset({"blank", "blanks", "empty"})  # "blank = not drawn": no code value
_RANGE: Final = re.compile(r"(?<![\w.^])(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)(?![\w.%/^])")
_PAREN: Final = re.compile(r"\([^()]*\)")
_PATIENT_KEY_TEXT: Final = re.compile(
    r"\(\s*patient\s*\)|\bpatient\s+(?:id|identifier|number|key)\b|"
    r"\bidentif(?:ies|ier)\b[^.;]*\bpatient\b|\bpatient\b[^.;]*\bidentifier\b",
    re.IGNORECASE,
)
_KEY_STATEMENT: Final = (
    re.compile(
        r"['\"‘’`]?\b(?P<col>[A-Za-z_][\w]*)\b['\"‘’`]?\s*(?:=|is|identifies|names)\s*(?:the\s+|a\s+)?"
        r"(?:patient|subject|participant)\b",
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
_NOUN_STOP: Final = re.compile(r"\s+(?:that|which|who|where|with|for|in|at|from|and|per|when)\b.*$", re.IGNORECASE)
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
    r"(?:,\s*|\(\s*|\bin\s+)(minutes|hours|days|weeks|months|years|grams|kilograms|kg|mg|g|mL|cm|mm|"
    r"percent|%|seconds|mmHg|cmH2O|bpm)\b",
    re.IGNORECASE,
)
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
            continue
        found.append((start, m.end(), m.group(group), depth[start]))
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
    return bool(config.MISSING_LABEL_PATTERN.search(label)) and not config.MISSING_LABEL_EXCEPT.search(label)


def _stated_range(description: str) -> tuple[float, float] | None:
    for text in (description, _PAREN.sub(" ", description)):
        for m in _RANGE.finditer(text):
            low, high = float(m.group(1)), float(m.group(2))
            if low < high:
                return low, high
    return None


def _unit(description: str) -> str | None:
    for m in _RATIO_UNIT.finditer(description):
        unit = m.group(1)
        if not _DATE_WORDS.search(unit) and unit.upper() not in ("Y/N", "TRUE/FALSE"):
            return unit
    word = _WORD_UNIT.search(description)
    return word.group(1) if word else None


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
    missing = {code: label for code, label in pairs if is_missing_label(label)}
    labels = {code: label for code, label in pairs if code not in missing}
    date_format = format_from_text(text)
    return DictionaryEntry(
        column=column,
        text=text,
        description=description.strip(),
        labels=labels if len(labels) >= 2 else {},
        missing=missing,
        stated_range=None if date_format else _stated_range(description),
        unit=None if date_format else _unit(description),
        date_format=date_format,
        direction=_direction(text),
        names_patient_key=bool(_PATIENT_KEY_TEXT.search(description)),
    )


def _entry_start(line: str, by_lower: dict[str, str]) -> tuple[str, str] | None:
    """(column, rest) when the line starts an entry: a column name, then space, ':' or ' -'."""
    if not line or line[0] in " \t":
        return None
    m = re.match(r"^['\"`]?([^\s:'\"`]+)['\"`]?\s*(?::|\s-\s|\s–\s|\t|\s)\s*(.*)$", line)
    if m is None:
        return None
    column = by_lower.get(m.group(1).lower())
    return (column, m.group(2)) if column is not None else None


def _noun(text: str) -> str:
    noun = _NOUN_STOP.sub("", text.strip()).strip(" -")
    words = noun.split()
    return " ".join(words[:4])


def parse_dictionary(text: str | None, columns: Sequence[str]) -> Dictionary:
    """The dictionary facts for ``columns`` found in ``text`` (empty when there is no text)."""
    if not text or not text.strip():
        return Dictionary()
    by_lower = {c.lower(): c for c in columns}
    blocks: dict[str, list[str]] = {}
    prose: list[str] = []
    current: str | None = None
    for raw in text.splitlines():
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
    return Dictionary(
        entries=entries,
        grain_noun=grain_noun,
        grain_text=grain_text,
        key_statements=tuple(statements),
        one_per_person=bool(_ONE_PER_PERSON.search(prose_text)),
        repeats_stated=bool(_REPEATS.search(prose_text)),
    )

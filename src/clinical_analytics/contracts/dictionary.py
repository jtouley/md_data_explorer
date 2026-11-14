# SPDX-License-Identifier: MIT
"""What an uploaded data dictionary documents per column: description, unit and codebook.

The text is the upload's ``doc_context`` (``core.doc_parser.extract_context_from_docs``).
Descriptions come from ``SchemaInferenceEngine.parse_dictionary_text``; this module adds what it
does not read:

* codebooks written one code per line under the column's entry ("age_in_cat: ..." then
  "0: if the age range is ...", "“overweight”: if BMI > 30"), and inline ones
  ("hemophilia (0=no, 1=yes)"), next to the one-line form ``extract_codebooks_from_docs`` reads;
* a unit named in parentheses ("weight (kg) at baseline"), from ``config.DICTIONARY_UNITS``.

Only an entry whose name is exactly a column's (case aside) is used: a near match ("duration_
treatment_cat" for column "duration_treatment_cat2") is left out, not guessed.

Published dictionaries often state how many patients have each value ("“Yes” (0.35%)"). That is
an observed count, and of 1,141 patients 0.35% is 4: ``strip_statistics`` removes every
percentage and small count before the text is kept, so the contract (and the model's context,
which is read from it) never carries one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from clinical_analytics.contracts import config

# Same entry form as parse_dictionary_text's first pattern: "name: text" or "name - text".
_ENTRY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:\-]\s*(.+)$")
_QUOTES = "\"'“”‘’"
# A code line under an entry: a number, a quoted label or one word, then ":" or "=", then its label.
_CODE_LINE = re.compile(
    rf"^(?:[{_QUOTES}]([^{_QUOTES}]{{1,60}})[{_QUOTES}]|(-?\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]{{0,40}}))\s*[:=]\s*(.+)$"
)
_INLINE_CODE = re.compile(r"(?<![\w.])(-?\d+)\s*=\s*['\"]?([^,;()'\"=]+?)['\"]?\s*(?=[,;)'\"]|$)")
_PARENTHETICAL = re.compile(r"\(([^()]*)\)")
_PERCENT = re.compile(r"\d\s*%")
_PERCENT_PAREN = re.compile(r"\s*\([^()]*\d\s*%[^()]*\)")
_N_EQUALS = re.compile(r"\bn\s*=\s*\d", re.IGNORECASE)
_SMALL_COUNT = re.compile(
    r"\b(?:[1-9]|10)\s+(?:patients?|people|persons?|subjects?|cases?|individuals?|participants?)\b", re.IGNORECASE
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")
_MIN_DESCRIPTION_CHARS = 10  # parse_dictionary_text's own floor
_MIN_INLINE_CODES = 2
_MIN_NEAR_NAME = 6  # shorter keys ("yes", "male") are codes, not variable names


@dataclass(frozen=True)
class ColumnDoc:
    """What the dictionary says about one column; every text has passed ``strip_statistics``."""

    description: str | None = None
    unit: str | None = None
    codebook: dict[str, str] = field(default_factory=dict)


def strip_statistics(text: str) -> str:
    """``text`` without percentages, "n = ..." or counts of 1–10 people.

    Parentheses holding a percentage are removed ("“Yes” (0.35%) if ..." -> "“Yes” if ..."); any
    sentence still holding one, an "n =" or a small count is dropped.
    """
    text = _PERCENT_PAREN.sub("", text)
    kept = [
        sentence
        for sentence in _SENTENCE_END.split(_SPACES.sub(" ", text).strip())
        if not (_PERCENT.search(sentence) or _N_EQUALS.search(sentence) or _SMALL_COUNT.search(sentence))
    ]
    return _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(kept)).strip()


def _label(text: str) -> str | None:
    label = strip_statistics(text).strip(_QUOTES + " ").rstrip(".").strip()
    return label or None


def inline_codebook(text: str) -> dict[str, str]:
    """Codes written inside a description: "(0=no, 1=yes)", "(1 = failure, 0 = censoring)".

    At least two codes, so "(1=yes)" alone, or a ratio, is not read as a codebook.
    """
    codes: dict[str, str] = {}
    for code, raw in _INLINE_CODE.findall(text):
        label = _label(raw)
        if label:
            codes[code] = label
    return codes if len(codes) >= _MIN_INLINE_CODES else {}


def documented_unit(text: str) -> str | None:
    """The first parenthesised unit in ``text`` that ``config.DICTIONARY_UNITS`` knows ("(kg)")."""
    for match in _PARENTHETICAL.finditer(text):
        unit = match.group(1).strip()
        if unit.lower() in config.DICTIONARY_UNITS:
            return unit
    return None


def _code_line(line: str) -> tuple[str, str] | None:
    match = _CODE_LINE.match(line)
    if match is None:
        return None
    code = (match.group(1) or match.group(2) or "").strip()
    label = _label(match.group(3))
    return (code, label) if code and label else None


def _near_column(key: str, columns: dict[str, str]) -> bool:
    """``key`` is a variable name one character-run off a column ("x_cat" for "x_cat2")."""
    if len(key) < _MIN_NEAR_NAME:
        return False
    return any(len(c) >= _MIN_NEAR_NAME and (c.startswith(key) or key.startswith(c)) for c in columns)


def _line_codebooks(text: str, columns: dict[str, str]) -> dict[str, dict[str, str]]:
    """Codes on the lines after a column's entry, up to the next entry or a line that is neither."""
    codebooks: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        entry = _ENTRY.match(line)
        key = entry.group(1).lower() if entry else None
        if key is not None and key in columns:
            current = columns[key]
            continue
        if key is not None and _near_column(key, columns):
            current = None  # another variable's entry ("duration_treatment_cat"), not a code
            continue
        code = _code_line(line) if current else None
        if current and code:
            codebooks.setdefault(current, {})[code[0]] = code[1]
        else:
            current = None
    return codebooks


def read_dictionary(doc_text: str | None, columns: Sequence[str]) -> dict[str, ColumnDoc]:
    """Per column of ``columns``, what ``doc_text`` documents about it (columns it does not name
    are left out).

    Precedence for a codebook: one-line codes (``extract_codebooks_from_docs``), then inline
    codes in the description, then code lines under the entry (last wins per code).
    """
    from clinical_analytics.core.schema_inference import SchemaInferenceEngine, extract_codebooks_from_docs

    if not doc_text:
        return {}
    by_key = {column.lower(): column for column in columns}
    parsed = SchemaInferenceEngine().parse_dictionary_text(doc_text)
    descriptions = {
        by_key[key]: text for key, text in (parsed.column_descriptions if parsed else {}).items() if key in by_key
    }
    one_line = {by_key[k]: v for k, v in extract_codebooks_from_docs(doc_text).items() if k in by_key}
    lines = _line_codebooks(doc_text, by_key)

    docs: dict[str, ColumnDoc] = {}
    for column in columns:
        raw = descriptions.get(column)
        codebook = {
            **{str(k): v for k, v in one_line.get(column, {}).items()},
            **(inline_codebook(raw) if raw else {}),
            **lines.get(column, {}),
        }
        description = strip_statistics(raw) if raw else None
        if description is not None and len(description) <= _MIN_DESCRIPTION_CHARS:
            description = None
        doc = ColumnDoc(description=description, unit=documented_unit(raw) if raw else None, codebook=codebook)
        if doc.description or doc.unit or doc.codebook:
            docs[column] = doc
    return docs

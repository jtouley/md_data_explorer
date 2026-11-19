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
an observed count, and of 1,141 patients 0.35% is 4: ``strip_statistics`` removes every such
share and small count before the text is kept, so the contract (and the model's context, which
is read from it) never carries one. A percentage that is a threshold ("EF < 40%", "HbA1c below
7%") is part of the definition, not a count of patients, and is kept.

Codes read here are a heuristic: ``contracts.builder`` puts them in the contract only when every
value the column holds is one of them (``builder.documented_enum``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from clinical_analytics.contracts import config

# Same entry form as parse_dictionary_text's first pattern: "name: text" or "name - text".
_ENTRY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:\-]\s*(.+)$")
_QUOTES = "\"'“”‘’"
# A code line under an entry: a number, a quoted label or one word, then ":" or "=", then its label.
_CODE_LINE = re.compile(
    rf"^(?:[{_QUOTES}]([^{_QUOTES}]{{1,60}})[{_QUOTES}]|(-?\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]{{0,40}}))\s*[:=]\s*(.+)$"
)
# A label may hold a comparison ("0 = EF < 40%, 1 = EF >= 40%"): "<", ">", "<=", ">=" are part of it.
_INLINE_CODE = re.compile(r"(?<![\w.<>=])(-?\d+)\s*=\s*['\"]?((?:[<>≤≥]=?|[^,;()'\"=<>≤≥])+?)['\"]?\s*(?=[,;)'\"]|$)")
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
_PARENTHETICAL = re.compile(r"\s*\(([^()]*)\)")
_PERCENT = re.compile(r"(?<![\d.,])\d+(?:[.,]\d+)?\s*%")
# What makes a percentage a threshold: a comparison or bound right before it ("< 40%", "below
# 7%", "5.7–6.4%", "between 5% and"). A share of people ("35% of patients") never is.
_THRESHOLD_BEFORE = re.compile(
    r"(?:[<>≤≥]\s*=?|=\s*[<>]|\b(?:below|above|under|over|than|exceed(?:s|ing)?|at\s+least|at\s+most|up\s+to|between)"
    r"|\d(?:[.,]\d+)?\s*%?\s*(?:[-–—]|\bto|\band))\s*$",
    re.IGNORECASE,
)
_SHARE_AFTER = re.compile(r"^\s*of\b", re.IGNORECASE)
_ALONE_IN_PARENS = re.compile(r"\(\s*(?:[<>≤≥~≈]\s*=?)?\s*$")
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
_MAX_ABBREVIATION_CHARS = 3  # "M", "NA", "UNK": a code written in capitals
_MAX_GLOSSARY_TERM_WORDS = 4
_MIN_GLOSSARY_WORDS = 2
_GLOSSARY_WORD = re.compile(r"[A-Za-z]{2,}")


@dataclass(frozen=True)
class ColumnDoc:
    """What the dictionary says about one column; every text has passed ``strip_statistics``."""

    description: str | None = None
    unit: str | None = None
    codebook: dict[str, str] = field(default_factory=dict)


def _is_threshold(text: str, match: re.Match[str]) -> bool:
    """The percentage at ``match`` bounds a value ("EF < 40%"); it is not a share of people."""
    before, after = text[: match.start()], text[match.end() :]
    if _SHARE_AFTER.match(after):
        return False
    # A percentage alone in parentheses ("(<1%)", "(0.35%)") is a share, whatever its comparison.
    if _ALONE_IN_PARENS.search(before) and after.lstrip().startswith(")"):
        return False
    return _THRESHOLD_BEFORE.search(before[-40:]) is not None


def _has_statistic(text: str) -> bool:
    """``text`` holds a share of people (a percentage that is not a threshold), "n = ..." or a
    count of 1–10 people."""
    shares = (m for m in _PERCENT.finditer(text) if not _is_threshold(text, m))
    return next(shares, None) is not None or bool(_N_EQUALS.search(text) or _SMALL_COUNT.search(text))


def strip_statistics(text: str) -> str:
    """``text`` without shares of people, "n = ..." or counts of 1–10 people.

    Parentheses holding one are removed ("“Yes” (0.35%) if ..." -> "“Yes” if ..."); any sentence
    still holding one is dropped. Thresholds stay: "EF < 40%", "target HbA1c below 7%".
    """
    text = _PARENTHETICAL.sub(lambda m: "" if _has_statistic(m.group(0).strip()) else m.group(0), text)
    kept = [
        sentence for sentence in _SENTENCE_END.split(_SPACES.sub(" ", text).strip()) if not _has_statistic(sentence)
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


CodeShape: TypeAlias = Literal["number", "quoted", "abbreviation", "word"]


def _code_shape(line: str) -> CodeShape | None:
    """How the code on ``line`` is written, or None when ``line`` is not a code line.

    A codebook keeps one shape ("1", "2"; "M", "F"; "never", "ever"); an entry of another shape
    after it ("ethnicity: ..." after "M: male") names another variable.
    """
    match = _CODE_LINE.match(line)
    if match is None:
        return None
    if match.group(1) is not None:
        return "quoted"
    code = match.group(2) or ""
    if _NUMBER.match(code):
        return "number"
    if len(code) <= _MAX_ABBREVIATION_CHARS and code.isupper():
        return "abbreviation"
    return "word"


def _other_variable(key_shape: CodeShape | None, following: str | None, shapes: set[CodeShape], near: bool) -> bool:
    """The entry on a line (not a column's) names another variable, so it ends the current
    codebook instead of being one of its codes.

    It does when it is a near name of a column ("x_cat" for "x_cat2"), when it is written unlike
    the codes so far ("ethnicity: ..." after "1: male" or "M: male"), or when the line after it
    is a number, quoted or capital-letter code written unlike it (it heads a codebook of its own:
    "race: Race" then "1: white").
    """
    if near:
        return True
    if shapes and key_shape not in shapes:
        return True
    # A word line after it is no evidence: it may be one more entry ("smoking: ..." after "F: female").
    following_shape = _code_shape(following) if following is not None else None
    return following_shape not in (None, "word", key_shape)


def _line_codebooks(text: str, columns: dict[str, str]) -> dict[str, dict[str, str]]:
    """Codes on the lines after a column's entry, up to the next variable's entry or a line that
    is neither (``_other_variable``)."""
    codebooks: dict[str, dict[str, str]] = {}
    current: str | None = None
    shapes: set[CodeShape] = set()  # how the current codebook's codes are written
    lines = [line for line in (raw.strip() for raw in text.splitlines()) if line]
    for index, line in enumerate(lines):
        entry = _ENTRY.match(line)
        key = entry.group(1).lower() if entry else None
        if key is not None and key in columns:
            current, shapes = columns[key], set()
            continue
        following = lines[index + 1] if index + 1 < len(lines) else None
        if key is not None and _other_variable(_code_shape(line), following, shapes, _near_column(key, columns)):
            current = None  # another variable's entry ("ethnicity: ..."), not a code of this one
            continue
        code = _code_line(line) if current else None
        shape = _code_shape(line)
        if current and code and shape is not None:
            codebooks.setdefault(current, {})[code[0]] = code[1]
            shapes.add(shape)
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


def _glossary_entry(doc_text: str, term: str, definition: str) -> bool:
    """``term`` is defined on one line of ``doc_text`` ("EDSS: ..." or "EDSS - ..."), in a few
    words, by words: not a hyphenated word ("Patient-level ..."), a link or a version number."""
    if "\n" in term or len(term.split()) > _MAX_GLOSSARY_TERM_WORDS or "://" in definition:
        return False
    if len(_GLOSSARY_WORD.findall(definition)) < _MIN_GLOSSARY_WORDS:
        return False
    return re.search(rf"^[\s*-]*{re.escape(term)}(?::|\s+-\s)", doc_text, re.MULTILINE) is not None


def read_glossary(doc_text: str | None) -> dict[str, str]:
    """Terms ``doc_text`` defines ("EDSS: Expanded Disability Status Scale"), read as the model's
    glossary always was (``core.autocontext._extract_glossary_from_docs``) and kept when
    ``_glossary_entry`` holds, every definition through ``strip_statistics``; a definition left
    empty is dropped."""
    from clinical_analytics.core.autocontext import _extract_glossary_from_docs

    if not doc_text:
        return {}
    glossary: dict[str, str] = {}
    for term, definition in _extract_glossary_from_docs(doc_text).items():
        kept = strip_statistics(definition)
        if kept and _glossary_entry(doc_text, term, kept):
            glossary[term] = kept
    return glossary

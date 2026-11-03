"""Classify a variable by what it measures, not how it is stored.

Test choice depends on this: a t-test needs a continuous outcome; a 0/1 flag or a coded
category ("1: Normal 2: Osteopenia 3: Osteoporosis") needs a test for proportions, even
though both are stored as integers. Storage width (Int8 vs Int64, Float32 vs Float64) must
not change the answer.
"""

from __future__ import annotations

import re
from typing import Literal, TypeAlias

import polars as pl

VariableKind: TypeAlias = Literal["continuous", "binary", "categorical"]

# "1: Normal", "2=no" — a code, then ':' or '=', then its label. Not preceded by a word
# character, '.', or '-' so "IL-6: pg/mL" and "1.5: x" are not codes.
_CODE_FIRST = re.compile(r"(?<![\w.-])(\d+)\s*[:=]\s*")
# "Yes: 1", "No=0" — a label, then its code.
_LABEL_FIRST = re.compile(r"(?<![\w-])([A-Za-z]+)\s*[:=]\s*(\d+)(?![\w.])")
_LABEL_TRIM = " \t,;/|()[]{}"
# Fewer than two code/label pairs is a unit or a time point ("Week 4: weight"), not a code list.
_MIN_PAIRS = 2

# Words that name the event (modelled as 1) or the non-event of a binary variable.
_EVENT_WORDS = frozenset(
    {
        "yes", "y", "true", "positive", "pos", "present", "dead", "died", "death", "deceased", "event",
        "relapse", "relapsed", "recurrence", "recurred", "failure", "failed", "case", "abnormal",
    }
)  # fmt: skip
_NON_EVENT_WORDS = frozenset(
    {
        "no", "n", "false", "negative", "neg", "absent", "alive", "survived", "living", "censored",
        "none", "control", "normal",
    }
)  # fmt: skip


def is_numeric_dtype(dtype: pl.DataType) -> bool:
    """Any integer or float width counts as numeric; booleans do not."""
    return dtype.is_numeric() and dtype != pl.Boolean


def code_labels(name: str) -> dict[str, str]:
    """Map codes to labels from a coded header; empty when the header documents fewer than two.

    'DEXA 1: Normal, 2: Osteopenia' -> {'1': 'Normal', '2': 'Osteopenia'};
    'Smoker (1=yes, 2=no)' -> {'1': 'yes', '2': 'no'}; 'Yes: 1 No: 0' -> {'1': 'Yes', '0': 'No'}.
    """
    matches = list(_CODE_FIRST.finditer(name))
    labels: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(name)
        label = name[match.end() : end].strip(_LABEL_TRIM)
        if label[:1].isalpha():
            labels[match.group(1)] = label
    if len(labels) >= _MIN_PAIRS:
        return labels
    pairs = {code: label for label, code in _LABEL_FIRST.findall(name)}
    return pairs if len(pairs) >= _MIN_PAIRS else {}


def header_has_code_labels(name: str) -> bool:
    """Whether a column name documents categorical codes."""
    return bool(code_labels(name))


def _code_key(level: object) -> str:
    """'2', 2 and 2.0 all name code '2'."""
    if isinstance(level, float) and level.is_integer():
        return str(int(level))
    return str(level)


def classify_variable(series: pl.Series) -> VariableKind:
    """Continuous, binary or categorical, from values and header (never from storage width)."""
    values = series.drop_nulls()
    distinct = values.unique().to_list()
    numeric = is_numeric_dtype(series.dtype)
    coded = header_has_code_labels(series.name)
    if len(distinct) <= 2:
        # Two measured values (e.g. 1234.5 and 2345.6) are still a measurement; only 0/1 and
        # 1/2 codes, coded headers, and two-level text are binary.
        if not numeric or coded or set(distinct) <= {0, 1} or set(distinct) <= {1, 2}:
            return "binary"
        return "continuous"
    if coded or not numeric:
        return "categorical"
    # Three small integer codes (0/1/2, 1/2/3) are levels, not a measurement.
    if len(distinct) == 3 and (set(distinct) <= {0, 1, 2} or set(distinct) <= {1, 2, 3}):
        return "categorical"
    return "continuous"


def binary_event_level(series: pl.Series) -> object | None:
    """The level modelled as the event (1) for a binary variable, or None when it cannot be told.

    - Boolean: True. Numeric 0/1: 1 (also when only one of the two values occurs).
    - Otherwise the level whose label (decoded from the header, or the text value itself) is
      an event word (yes, dead, positive, ...), else the level opposite a non-event word
      (no, alive, ...).
    - Unrecognised text: the last level in sorted order; callers report the chosen level.
    - Numeric codes other than 0/1 with no decodable labels (e.g. bare 1/2): None, because
      SPSS-style data often codes 2 as "No" and guessing would flip the answer.
    """
    if series.dtype == pl.Boolean:
        return True
    levels = sorted(series.drop_nulls().unique().to_list(), key=str)
    numeric = is_numeric_dtype(series.dtype)
    labels = code_labels(series.name)
    if any(word in _NON_EVENT_WORDS | {"survival"} for word in re.findall(r"[a-z]+", short_name(series.name).lower())):
        # "Alive 1: Yes 2: No": the yes-level is the non-event, so word matching would invert it.
        return None
    if numeric and set(levels) <= {0, 1} and not labels:
        return 1.0 if series.dtype.is_float() else 1
    chosen: object | None = None
    names = {lvl: (labels.get(_code_key(lvl)) or (None if numeric else str(lvl))) for lvl in levels}
    events = [lvl for lvl, text in names.items() if text and text.strip().lower() in _EVENT_WORDS]
    if len(events) == 1:
        chosen = events[0]
        return chosen
    non_events = [lvl for lvl, text in names.items() if text and text.strip().lower() in _NON_EVENT_WORDS]
    if len(levels) == 2 and len(non_events) == 1:
        chosen = next(lvl for lvl in levels if lvl != non_events[0])
    elif numeric and set(levels) <= {0, 1}:
        chosen = 1.0 if series.dtype.is_float() else 1  # labels did not name the event; 1 by convention
    elif not numeric and levels:
        chosen = levels[-1]
    return chosen


def display_level(column: str, level: object) -> str:
    """Readable name for a group level: decoded label, or 'column=level' for bare codes."""
    labels = code_labels(column)
    text = _code_key(level)
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    if text in labels:
        return labels[text]
    if isinstance(level, bool) or re.fullmatch(r"-?\d+(\.\d+)?", text):
        return f"{short_name(column)}={text}"
    return text


def short_name(column: str) -> str:
    """Column name without its embedded code list: 'DEXA result 1: Normal 2: ...' -> 'DEXA result'."""
    if not header_has_code_labels(column):
        return column
    starts = [m.start() for m in _CODE_FIRST.finditer(column)] + [m.start() for m in _LABEL_FIRST.finditer(column)]
    head = column[: min(starts)] if starts else column
    return re.sub(r"[\s?:=(\[-]+$", "", head).strip() or column

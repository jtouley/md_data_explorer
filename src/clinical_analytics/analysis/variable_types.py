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
# A label that starts with a unit ("years", "mg/dL") is a measurement note, not a category:
# "Age at visit 1: years, visit 2: years" is two time points, not codes 1 and 2. Single
# letters are left out because "1: M 2: F" is a real codebook.
_UNIT_WORDS = frozenset(
    {
        "year", "years", "yr", "yrs", "month", "months", "mo", "mos", "week", "weeks", "wk", "wks",
        "day", "days", "hour", "hours", "hr", "hrs", "minute", "minutes", "min", "mins", "second",
        "seconds", "sec", "kg", "mg", "mcg", "ug", "ng", "pg", "lb", "lbs", "cm", "mm", "ml", "dl",
        "mmol", "umol", "mmhg", "bpm", "iu", "units", "percent", "pct",
    }
)  # fmt: skip
# Separators inside a label mean the text runs on past the label ("years, visit"), so the
# header is prose with numbers in it rather than a code list.
_LABEL_RUN_ON = re.compile(r"[,;]")

# Integer-like text ("1", " 2", "1.0"): codes stored as text follow the numeric rule.
_INTEGER_TEXT = re.compile(r"\s*-?\d+(\.0+)?\s*")

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
# Column-name tokens that say the 1/yes level is NOT the event ("Alive 1: Yes 2: No", "survived",
# CDISC "CNSR" where 1 = censored). "n" is left out: "n_stage" is not the word "no".
_NON_EVENT_NAME_WORDS = (_NON_EVENT_WORDS - {"n"}) | {"survival", "censor", "censoring", "cnsr"}
# Event words that name the outcome itself. Under a non-event column name these still identify
# the event ("CNSR 0: event 1: censored"); affirmations like "yes" or "positive" do not, because
# "yes" under "Alive" means alive.
_OUTCOME_EVENT_WORDS = frozenset(
    {"dead", "died", "death", "deceased", "event", "relapse", "relapsed", "recurrence", "recurred", "failure"}
)


def is_numeric_dtype(dtype: pl.DataType) -> bool:
    """Any integer or float width counts as numeric; booleans do not."""
    return dtype.is_numeric() and dtype != pl.Boolean


def code_labels(name: str) -> dict[str, str]:
    """Map codes to labels from a coded header; empty when the header documents fewer than two.

    'DEXA 1: Normal, 2: Osteopenia' -> {'1': 'Normal', '2': 'Osteopenia'};
    'Smoker (1=yes, 2=no)' -> {'1': 'yes', '2': 'no'}; 'Yes: 1 No: 0' -> {'1': 'Yes', '0': 'No'}.

    Labels that start with a unit ("1: years") or run on past a comma ("1: years, visit") mean
    the numbers are time points or measurements, so the header is not a code list.
    """
    matches = list(_CODE_FIRST.finditer(name))
    labels: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(name)
        label = name[match.end() : end].strip(_LABEL_TRIM)
        if label[:1].isalpha():
            labels[match.group(1)] = label
    if len(labels) >= _MIN_PAIRS:
        return labels if all(_is_category_label(label) for label in labels.values()) else {}
    pairs = {code: label for label, code in _LABEL_FIRST.findall(name)}
    return pairs if len(pairs) >= _MIN_PAIRS else {}


def _is_category_label(label: str) -> bool:
    """A category name ("Osteopenia", "Treatment A", "n/a"), not a unit or run-on prose."""
    if _LABEL_RUN_ON.search(label):
        return False
    first_word = re.split(r"[\s/]+", label.lower(), maxsplit=1)[0]
    return first_word not in _UNIT_WORDS


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


def _one_level(series: pl.Series, levels: list[object], text_codes: bool) -> object:
    """The level that means 1: the value 1 (1.0 for floats), or the text level that reads as 1."""
    if text_codes:
        return next((lvl for lvl in levels if int(float(str(lvl))) == 1), "1")
    return 1.0 if series.dtype.is_float() else 1


def binary_event_level(series: pl.Series) -> object | None:
    """The level modelled as the event (1) for a binary variable, or None when it cannot be told.

    - Boolean: True. Numeric 0/1: 1 (also when only one of the two values occurs).
    - Integer-like text ("0"/"1", "1"/"2") follows the numeric rule, returning the text level:
      a CSV that stores codes as text means the same thing as one that stores numbers.
    - Otherwise the level whose label (decoded from the header, or the text value itself) is
      an event word (yes, dead, positive, ...), else the level opposite a non-event word
      (no, alive, ...).
    - Column named for the non-event (alive, survived, censored, censor, CNSR): None, unless
      the header labels a level with an outcome word ("CNSR 0: event 1: censored"). In CDISC,
      CNSR=1 means censored, so the usual 0/1 rule would model censoring as the event.
    - Unrecognised text: the last level in sorted order; callers report the chosen level.
    - Codes other than 0/1 with no decodable labels (e.g. bare 1/2, numeric or text): None,
      because SPSS-style data often codes 2 as "No" and guessing would flip the answer.
    """
    if series.dtype == pl.Boolean:
        return True
    levels = sorted(series.drop_nulls().unique().to_list(), key=str)
    numeric = is_numeric_dtype(series.dtype)
    text_codes = not numeric and bool(levels) and all(_INTEGER_TEXT.fullmatch(str(lvl)) for lvl in levels)
    coded = numeric or text_codes
    codes = {lvl: _code_key(int(float(str(lvl))) if text_codes else lvl) for lvl in levels}
    zero_one = coded and set(codes.values()) <= {"0", "1"}
    labels = code_labels(series.name)
    names = {lvl: (labels.get(codes[lvl]) or (None if coded else str(lvl))) for lvl in levels}
    words = {lvl: text.strip().lower() for lvl, text in names.items() if text}
    name_tokens = re.findall(r"[a-z]+", short_name(series.name).lower())
    if any(token in _NON_EVENT_NAME_WORDS for token in name_tokens):
        # "Alive 1: Yes 2: No": the yes-level is the non-event, so word matching would invert it.
        # Only a label that names the outcome itself ("0: event", "1: died") still identifies it.
        stated = [lvl for lvl, word in words.items() if word in _OUTCOME_EVENT_WORDS]
        return stated[0] if len(stated) == 1 else None
    if zero_one and not labels:
        return _one_level(series, levels, text_codes)
    events = [lvl for lvl, word in words.items() if word in _EVENT_WORDS]
    if len(events) == 1:
        return events[0]
    non_events = [lvl for lvl, word in words.items() if word in _NON_EVENT_WORDS]
    if len(levels) == 2 and len(non_events) == 1:
        return next(lvl for lvl in levels if lvl != non_events[0])
    if zero_one:
        return _one_level(series, levels, text_codes)  # labels did not name the event; 1 by convention
    if not coded and levels:
        return levels[-1]
    return None


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

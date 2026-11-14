# SPDX-License-Identifier: MIT
"""Word accounting: every content word of a question must be bound by the plan that runs.

A plan that leaves a word of the question unused answers a different question, whatever its
confidence. This gate runs once, after parsing (any tier) and before execution, and holds
such a plan for confirmation with the unused words named.

A word is used only when:
- it is in the shared reviewed vocabulary of words that carry no limit (question_vocabulary);
- it is an intent or statistic word and the plan has that intent (INTENT_WORDS);
- it names a column the plan references (metric, group, predictors, time, event, filters,
  cohort steps, proportion/rate), by name, alias or a close stem;
- it names a value the plan binds (equality/IN filter or cohort step, or any value of the
  grouping column);
- it is a number bound to a comparison with the same direction on the column it belongs to;
- it is a logic word and the plan holds the matching operator (LOGIC rules below).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, TypeAlias

from clinical_analytics.core.column_parser import parse_column_name
from clinical_analytics.core.execution_gate import label_words_of, limit_text
from clinical_analytics.core.nl_query_config import STEM_MATCH_MIN_PREFIX
from clinical_analytics.core.query_plan import STATISTIC_WORDS, CohortStep, as_executed, grouped_statistic

# Words that carry no limit come from one reviewed vocabulary, shared with the cohort peel
# and the pre-execution gap check.
from clinical_analytics.core.question_vocabulary import (
    FUNCTION_WORDS,
    LINK_WORDS,
    NEGATION_CUES,
    NO_LIMIT_WORDS,
    RANGE_PATTERN,
    SUBJECT_WORDS,
    NegationScope,
    complement_tail,
    negation_scope,
    plain_grouping,
    range_ends,
    range_in_order,
)
from clinical_analytics.core.question_vocabulary import NEGATION_WORDS as BASE_NEGATION_WORDS
from clinical_analytics.core.value_index import ColumnValues, IndexedWords, StemIndex, cached, matches_stored

INTENT_WORDS: dict[str, frozenset[str]] = {
    "COUNT": frozenset({"how", "many", "number", "count", "total"}),
    "STAT": frozenset(
        {"average", "avg", "mean", "median", "minimum", "maximum", "min", "max", "std", "standard", "deviation"}
    ),
    "COMPARE_GROUPS": frozenset({"compare", "compared", "comparison", "versus", "vs", "difference"}),
    "CORRELATIONS": frozenset({"correlation", "correlated", "correlate", "relationship", "association"}),
    "FIND_PREDICTORS": frozenset({"predict", "predicts", "predictor", "predictors"}),
    "SURVIVAL": frozenset({"survival"}),
}
# Logic words: used only when the plan holds the matching operator.
NEGATION_WORDS: frozenset[str] = BASE_NEGATION_WORDS | {"__neg__"}
OR_WORDS: frozenset[str] = frozenset({"or"})
BETWEEN_WORDS: frozenset[str] = frozenset({"between"})
RATE_WORDS: frozenset[str] = frozenset(
    {"proportion", "percent", "percentage", "share", "rate", "fraction", "likely", "__rate__"}
)
BY_WORDS: frozenset[str] = frozenset({"by", "each", "per"})
CUTOFF_WORDS: frozenset[str] = frozenset({"top", "decile", "highest", "oldest"})
# Verbs that only point at the place named after them ("admitted to hospital", "stayed in the ICU").
PLACE_VERBS: frozenset[str] = frozenset({"admitted", "stayed"})
DIRECTION_OPS: dict[str, str] = {
    "over": "gt",
    "above": "gt",
    "exceeding": "gt",
    "__gt__": "gt",
    "__gt_age__": "gt",
    "under": "lt",
    "below": "lt",
    "__lt__": "lt",
    "__lt_age__": "lt",
    "__ge__": "ge",
    "__le__": "le",
    # The two ends of a range, both included; shown as the numbers alone.
    "__rlo__": "ge",
    "__rhi__": "le",
}
_POSTFIX = {"__ge__", "__le__", "__rlo__", "__rhi__"}
_AGE_WORDS = {"aged", "age", "old", "__gt_age__", "__lt_age__"}
LOGIC_WORDS: frozenset[str] = (
    NEGATION_WORDS | OR_WORDS | BETWEEN_WORDS | RATE_WORDS | BY_WORDS | CUTOFF_WORDS | frozenset(DIRECTION_OPS)
)

_NUM = r"\d+(?:\.\d+)?"
_INTENT_VOCABULARY: frozenset[str] = frozenset().union(*INTENT_WORDS.values())
# Multi-word logic phrases, rewritten to one placeholder before tokenising.
_PHRASES: tuple[tuple[str, str], ...] = (
    (r"n't\b", " not"),
    (r"(\d)\s*\+", r"\1 __ge__ "),
    (r"(\d)\s+plus\b", r"\1 __ge__ "),
    (rf"\bat least ({_NUM})\b", r" \1 __ge__ "),
    (rf"\bat most ({_NUM})\b", r" \1 __le__ "),
    # "65 or older", not "under 40 or over 80" (there "or" joins two comparisons).
    (r"\b(?:or|and) (?:more|older|over|above|greater|higher)\b(?! than)(?!\s+\d)", " __ge__ "),
    (r"\bat least\b", " __ge__ "),
    (r"\b(?:or|and) (?:less|fewer|younger|under|below|lower)\b(?! than)(?!\s+\d)", " __le__ "),
    (r"\bat most\b", " __le__ "),
    (r"\b(?:more|less) likely\b", " __rate__ "),
    (r"\bother than\b", " __neg__ "),
    (r"\bolder than\b", " __gt_age__ "),
    (r"\byounger than\b", " __lt_age__ "),
    (r"\b(?:greater|higher|more) than\b", " __gt__ "),
    (r"\b(?:less|fewer|lower) than\b", " __lt__ "),
)
_SHOWN = {
    "__ge__": "or more",
    "__le__": "or less",
    "__gt__": "more than",
    "__lt__": "less than",
    "__gt_age__": "older than",
    "__lt_age__": "younger than",
    "__neg__": "other than",
    "__rate__": "more likely",
    "__rlo__": "",
    "__rhi__": "",
}
_SEX_INFLECTIONS = {
    "f": {"f", "female", "females", "woman", "women"},
    "female": {"female", "females", "woman", "women"},
    "m": {"m", "male", "males", "man", "men"},
    "male": {"male", "males", "man", "men"},
}
# A word starts with a letter and keeps its digits ("hba1c", "cd4"); a number stands alone.
_TOKEN = r"__[a-z_]+__|[a-z][a-z0-9]*|\d+(?:\.\d+)?"


@dataclass(frozen=True)
class Token:
    text: str
    index: int


def _range_markers(match: re.Match[str]) -> str:
    """A range in order becomes "<lo> __rlo__ <hi> __rhi__" (both ends included), read by the same
    pattern as the peel (question_vocabulary.RANGE_PATTERN). A reversed one ("between 65 and 50")
    keeps its numbers with no direction, so neither end is ever marked used and the plan holds."""
    low, high, _lo, _hi = range_ends(match)
    lead = "between " if match.group("lo") is not None else ""
    if not range_in_order(match):
        return f" {lead}{low} and {high} "
    return f" {lead}{low} __rlo__ {high} __rhi__ "


def tokenize(question: str) -> list[str]:
    """Lowercase words, numbers and logic placeholders, in order."""
    text = RANGE_PATTERN.sub(_range_markers, plain_grouping(question.lower()))
    for pattern, replacement in _PHRASES:
        text = re.sub(pattern, replacement, text)
    return re.findall(_TOKEN, text)


def comparison_scope(tokens: list[str], at: int, direction_at: int) -> NegationScope:
    """Negation scope of the comparison of the number at ``at`` with its direction word: read
    from where the comparison starts ("over 65", "65 or older"), by the shared rule."""
    return negation_scope(tokens[: min(at, direction_at)])


def shown(word: str) -> str:
    return _SHOWN.get(word, word)


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def hold_reason(words: Iterable[str]) -> str:
    """One plain sentence naming the words the plan leaves out and what to do about them."""
    quoted = [f"'{shown(word)}'" for word in words]
    if len(quoted) == 1:
        return f"I couldn't use the word {quoted[0]} in the plan; rephrase it with a column or value from your data."
    return (
        f"I couldn't use the words {_listed(quoted)} in the plan; rephrase them with columns or values from your data."
    )


def name_tokens(name: str) -> list[str]:
    return re.findall(_TOKEN, name.lower())


def stem_match(word: str, target: str) -> bool:
    """Same word, a plural, or a shared stem of at least STEM_MATCH_MIN_PREFIX letters covering
    75% of the shorter."""
    if word == target or word.rstrip("s") == target.rstrip("s"):
        return True
    if word.isdigit() or target.isdigit():
        return False
    if len(word) >= 3 and target.startswith(word) and target[len(word) :].isdigit():
        # A name with a number appended to the word ("type" of ms_type2, "covid" of covid19).
        return True
    shorter = min(len(word), len(target))
    common = 0
    for left, right in zip(word, target, strict=False):
        if left != right:
            break
        common += 1
    return common >= STEM_MATCH_MIN_PREFIX and common >= 0.75 * shorter


# Endings an inflection adds to an intent word ("counted", "predicting", "correlates").
_INFLECTION_ENDINGS: tuple[str, ...] = ("ions", "ion", "ing", "ed", "es", "e", "s", "d")


def _inflection_stem(word: str) -> str:
    for ending in _INFLECTION_ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= STEM_MATCH_MIN_PREFIX:
            return word[: -len(ending)]
    return word


def inflection_of(word: str, target: str) -> bool:
    """``word`` is ``target`` with a different inflectional ending ("counted" of "count",
    "associated" of "association"), on a stem of at least STEM_MATCH_MIN_PREFIX letters. Unlike
    ``stem_match`` a longer word sharing the letters is not one: "country" is not "count", and
    "meaning" is not "mean" (a four-letter stem)."""
    stem = _inflection_stem(word)
    return len(stem) >= STEM_MATCH_MIN_PREFIX and stem == _inflection_stem(target)


_AGE_UNITS = {"years", "year", "yrs", "yr", "y", "at", "baseline", "in", "current"}
# Words after an age cutoff that only restate it ("65 years old").
_AGE_UNIT_WORDS = frozenset({"years", "year", "yrs", "yr", "y", "yo", "old"})


def age_column(columns: list[str]) -> str | None:
    """The age column by meaning: "age", "Age (years)", "age_years", "age at baseline"."""
    for column in columns:
        parts = name_tokens(column)
        if "age" in parts and set(parts) - {"age"} <= _AGE_UNITS:
            return column
    return None


def numeric_ranges(columns: list[str], values_for: Callable[[str], list[Any]]) -> dict[str, tuple[float, float]]:
    """Observed (min, max) of each measured column."""
    ranges: dict[str, tuple[float, float]] = {}
    for column in columns:
        values = values_for(column)
        if _is_measure(values):
            ranges[column] = cached(values, column, "range", partial(_range, values))
    return ranges


def _range(values: list[Any]) -> tuple[float, float]:
    numbers = [float(value) for value in values if value is not None]
    return min(numbers), max(numbers)


def number_owner(
    tokens: list[str],
    number_at: int,
    columns: list[str],
    numeric_columns: set[str],
    alias_index: dict[str, str] | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> str | None:
    """Column a compared number belongs to, or None when the question does not say.

    The nearest content word before the comparison names the column ("bmi over 30" is bmi).
    A unit right after the number names the column whose name carries it ("over 100 kg" is
    "Weight (kg)"); when the words before name a different column the number has no owner.
    Age is used only when the words say age ("older than", "aged"), or the number follows a
    subject ("patients over 65") and either no other measure is named before it, or the
    number is outside the observed range of every measure named before it.
    """
    units = unit_columns(tokens, number_at, columns)
    before = _owner_before(tokens, number_at, columns, numeric_columns, alias_index, ranges)
    if not units:
        return before[0]
    if len(units) > 1:
        # "over 2 years": years of age or years on dialysis? The words do not say.
        return before[0] if before[1] and before[0] in units else None
    unit = next(iter(units))
    if before[1] and before[0] != unit:
        return None
    return unit


def unit_columns(tokens: list[str], number_at: int, columns: list[str]) -> set[str]:
    """Columns whose name carries the unit written right after the number (age owns "years")."""
    at = number_at + 1
    if at < len(tokens) and tokens[at] in _POSTFIX:
        at += 1
    if at >= len(tokens) or tokens[at] in LOGIC_WORDS or tokens[at] in NO_LIMIT_WORDS:
        return set()
    word = tokens[at]
    owners = {column for column in columns if word in unit_words(column)}
    age = age_column(columns)
    if age and word in _AGE_UNIT_WORDS:
        owners.add(age)
    return owners


def unit_words(column: str) -> set[str]:
    """Words of a measure's name after the measure itself ("kg" of "Weight (kg)", "mg" and
    "dl" of "LDL mg/dL", "days" of "followup_days"). Coded columns have none: their extra
    words are value labels ("Yes", "No"), not units."""
    if parse_column_name(column).value_mapping:
        return set()
    parts = [part for part in name_tokens(column) if not part.isdigit()]
    return {part for part in parts[1:] if part not in LOGIC_WORDS and part not in NO_LIMIT_WORDS}


def _owner_before(
    tokens: list[str],
    number_at: int,
    columns: list[str],
    numeric_columns: set[str],
    alias_index: dict[str, str] | None,
    ranges: dict[str, tuple[float, float]] | None,
) -> tuple[str | None, bool]:
    """(owner from the words before the number, whether those words name it explicitly)."""
    age = age_column(columns)
    start = number_at - 1
    if start >= 1 and tokens[start] == "__rlo__" and _NUMBER_TOKEN.fullmatch(tokens[start - 1]):
        # The upper end of a range ("50 __rlo__ 65 __rhi__") belongs where the lower end does.
        return number_owner(tokens, start - 1, columns, numeric_columns, alias_index, ranges), True
    if start >= 0 and tokens[start] in DIRECTION_OPS and tokens[start] not in _POSTFIX:
        if tokens[start] in _AGE_WORDS:
            return age, True
        start -= 1
    elif start >= 0 and tokens[start] in BETWEEN_WORDS:
        start -= 1
    for at in range(start, -1, -1):
        word = tokens[at]
        if word in FUNCTION_WORDS:
            continue
        if word in _AGE_WORDS:
            return age, True
        named = _column_for_word(word, columns, alias_index)
        if named is not None:
            # "diabetics over 65": a yes/no or text column cannot own a cutoff. The words name
            # no measure for the number, so it has no owner and the plan holds.
            return (named if named in numeric_columns else None), True
        break
    before = tokens[:number_at]
    measures = {
        named
        for named in (_column_for_word(word, columns, alias_index) for word in before)
        if named is not None and named in numeric_columns and named != age
    }
    if not measures:
        return age, False
    if ranges is None:
        return None, False
    try:
        number = float(tokens[number_at])
    except ValueError:
        return None, False
    inside = [column for column in measures if column in ranges and ranges[column][0] <= number <= ranges[column][1]]
    if inside or any(column not in ranges for column in measures):
        return None, False
    return age, False


def _column_for_word(word: str, columns: list[str], alias_index: dict[str, str] | None) -> str | None:
    if word in LOGIC_WORDS or word in FUNCTION_WORDS or word in SUBJECT_WORDS or word.isdigit():
        return None
    for alias, column in (alias_index or {}).items():
        if " " not in alias and stem_match(word, alias):
            return column
    for column in columns:
        if _names_column(word, column):
            return column
    return None


def _names_column(word: str, column: str) -> bool:
    """The word is the column's name: short names on any word; longer ones ("LDL mg/dL",
    "Smoker 1: Yes 2: No") on their leading word, the measure rather than a unit or label."""
    parts = [part for part in name_tokens(column) if not part.isdigit()]
    if not parts:
        return False
    return stem_match(word, parts[0]) or (len(parts) <= 2 and any(stem_match(word, part) for part in parts))


def unused_words(
    question: str,
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    alias_index: dict[str, str] | None = None,
    context: Any = None,
) -> list[str]:
    """Content words of the question that the plan does not bind, in question order."""
    tokens = tokenize(question)
    unused = [shown(tokens[at]) for at in _unused_at(tokens, plan, columns, values_for, alias_index, context)]
    return [word for word in unused if word]


def explain_hold(
    question: str,
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    alias_index: dict[str, str] | None = None,
    context: Any = None,
) -> str | None:
    """One plain sentence telling the reader what to change, or None when every word is used.

    The cause decides the sentence: a number whose measure the words leave open, values
    joined by "and" (together or each?), two ranges joined by "or", words that name nothing
    in the data, and otherwise words the plan could not use.
    """
    tokens = tokenize(question)
    unused = _unused_at(tokens, plan, columns, values_for, alias_index, context)
    words = [shown(tokens[at]) for at in unused if shown(tokens[at])]
    if not words:
        # Only a range marker is left, which unused_words does not report either: no hold.
        return None
    namer = _ColumnNamer(
        [*columns, *sorted(_referenced_columns(plan, context) - set(columns))], values_for, alias_index
    )
    for explain in (_ambiguous_number, _ranges_joined_by_or, _values_joined_by_and, _statistic_missing):
        sentence = explain(tokens, unused, plan, columns, values_for, alias_index, namer)
        if sentence:
            return sentence
    unknown = [
        tokens[at]
        for at in unused
        if not _NUMBER_TOKEN.fullmatch(tokens[at])
        and tokens[at] not in LOGIC_WORDS
        and tokens[at] not in _INTENT_VOCABULARY
        and not namer.columns_named(tokens[at])
    ]
    if unknown and len(unknown) == len(words):
        return (
            f"I couldn't match {_quoted(unknown)} to a column or value in this dataset; rephrase with a column "
            f"name from your data, or leave {_it_them(unknown)} out."
        )
    return hold_reason(words)


def _quoted(words: list[str]) -> str:
    return _listed([f"'{word}'" for word in words])


def _it_them(words: list[str]) -> str:
    return "it" if len(words) == 1 else "them"


def _ambiguous_number(
    tokens: list[str],
    unused: list[int],
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    alias_index: dict[str, str] | None,
    namer: _ColumnNamer,
) -> str | None:
    """ "average bmi in patients over 30": 30 could limit age or bmi."""
    for at in unused:
        if not _NUMBER_TOKEN.fullmatch(tokens[at]):
            continue
        direction_at = _direction_at(tokens, at)
        if direction_at is None:
            continue
        candidates: list[str] = []
        age = age_column(columns)
        if age:
            candidates.append(age)
        numeric = {column for column in columns if _is_measure(values_for(column))}
        for word in tokens[:at]:
            for column in sorted(namer.columns_named(word) & numeric):
                if column not in candidates:
                    candidates.append(column)
        if len(candidates) < 2:
            continue
        number = tokens[at]
        if tokens[direction_at] in {"__rlo__", "__rhi__"}:
            low, high = (number, tokens[at + 2]) if tokens[direction_at] == "__rlo__" else (tokens[at - 2], number)
            limit = f"between {low} and {high}"
            number = f"{low} to {high}"
        else:
            limit = f"{_DIRECTION_SHOWN.get(tokens[direction_at], shown(tokens[direction_at]))} {number}"
        examples = " or ".join(f"'{_short(column)} {limit}'" for column in candidates[:2])
        return f"Say which measure {number} applies to, for example {examples}."
    return None


def _ranges_joined_by_or(
    tokens: list[str],
    unused: list[int],
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    alias_index: dict[str, str] | None,
    namer: _ColumnNamer,
) -> str | None:
    """ "under 40 or over 80": the plan cannot keep patients who meet either range."""
    if not any(tokens[at] in OR_WORDS for at in unused):
        return None
    numbers = [at for at, word in enumerate(tokens) if _NUMBER_TOKEN.fullmatch(word) and _direction_at(tokens, at)]
    if len(numbers) < 2:
        return None
    return "I can't keep patients who meet either of two ranges; ask about each range in its own question."


def _values_joined_by_and(
    tokens: list[str],
    unused: list[int],
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    alias_index: dict[str, str] | None,
    namer: _ColumnNamer,
) -> str | None:
    """ "how many women and men?": both together, or each one?"""
    if "and" not in tokens:
        return None
    for column in columns:
        words = _value_words(column, values_for(column))
        named = [at for at in unused if _names_value(tokens[at], words)]
        keys = {_value_key(tokens[at], column, values_for) for at in named}
        if len(keys) < 2:
            continue
        first, last = tokens[named[0]], tokens[named[-1]]
        return f"Say '{first} or {last}' to count them together, or 'by {_short(column)}' to see each group."
    return None


# Declared column types (semantic layer / data contract) that settle what a numeric column is.
_DECLARED_MEASURES = frozenset({"continuous", "numeric"})
_DECLARED_CODES = frozenset({"categorical", "coded", "binary"})
GroupingLevels: TypeAlias = Literal["measurement", "coded"]


def numeric_grouping_levels(
    values: list[Any], row_count: int | None = None, declared_type: str | None = None
) -> GroupingLevels | None:
    """What grouping by a numeric column with more than MAX_NUMERIC_GROUP_LEVELS values makes.

    "measurement": one group per exact value of a measure. The column is declared continuous,
    or holds a non-integer value, or has as many distinct values as half the rows (an
    identifier or a raw measurement). "coded": integer codes (site 1..12, a zip code without a
    codebook); grouping by them may be what was meant, so it is confirmed, never refused.
    None: not a numeric column with that many values. Small cells in any grouping are hidden
    by the engine, not here.
    """
    from clinical_analytics.core.nl_query_config import MAX_NUMERIC_GROUP_LEVELS

    present = [value for value in values if value is not None]
    distinct = set(present)
    if not _is_measure(present) or len(distinct) <= MAX_NUMERIC_GROUP_LEVELS:
        return None
    declared = (declared_type or "").strip().lower()
    if declared in _DECLARED_MEASURES:
        return "measurement"
    if declared in _DECLARED_CODES:
        return "coded"
    if any(not float(value).is_integer() for value in distinct):
        return "measurement"
    if row_count is not None and 2 * len(distinct) > row_count:
        return "measurement"
    return "coded"


def _grouping_levels(
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    row_count: int | None,
    declared_type: Callable[[str], str | None] | None,
) -> tuple[str, list[Any], GroupingLevels] | None:
    group_by = getattr(plan, "group_by", None)
    if getattr(plan, "intent", None) not in {"COUNT", "DESCRIBE", "COMPARE_GROUPS"}:
        return None  # a correlation's second variable is not a grouping
    if not group_by or group_by not in columns or parse_column_name(str(group_by)).value_mapping:
        return None
    values = [value for value in values_for(str(group_by)) if value is not None]
    declared = declared_type(str(group_by)) if declared_type is not None else None
    levels = numeric_grouping_levels(values, row_count, declared)
    return (str(group_by), values, levels) if levels is not None else None


def measurement_grouping_reason(
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    row_count: int | None = None,
    declared_type: Callable[[str], str | None] | None = None,
) -> str | None:
    """One sentence when the plan groups by a measurement, or None.

    Grouping (or cross-tabulating) by a continuous column makes one group per exact value:
    a privacy leak and no answer. It is refused whatever the words say, so it never runs.
    Integer codes are not refused here (coded_grouping_reason holds them for confirmation).
    """
    found = _grouping_levels(plan, columns, values_for, row_count, declared_type)
    if found is None or found[2] != "measurement":
        return None
    group_by, values, _levels = found
    name = _short(group_by)
    middle = sorted(float(value) for value in values)[len(values) // 2]
    cutoff = f"{round(middle):g}" if abs(middle) >= 10 else f"{middle:.2g}"
    metric = getattr(plan, "metric", None)
    groups = [column for column in columns if column != group_by and _is_grouping(column, values_for)]
    if metric and metric != group_by and metric in groups:
        example = f"compare {name} by {_short(str(metric))}"
    elif groups:
        measure = _short(str(metric)) if metric and metric != group_by and metric in columns else name
        example = f"compare {measure} by {_short(groups[0])}"
    else:
        example = f"average {name}"
    return (
        f"{name} is a measurement, not a group; ask about a range of it such as '{name} over {cutoff}', "
        f"or ask '{example}'."
    )


def coded_grouping_reason(
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    row_count: int | None = None,
    declared_type: Callable[[str], str | None] | None = None,
) -> str | None:
    """One sentence when the plan groups by many integer codes, or None.

    "count patients by site_id" with sites 1..12 is a fair grouping, but the same integers
    could be a measurement read as codes. Held for the user's confirmation, which releases it.
    """
    found = _grouping_levels(plan, columns, values_for, row_count, declared_type)
    if found is None or found[2] != "coded":
        return None
    group_by, values, _levels = found
    name = _short(group_by)
    return (
        f"{name} holds {len(set(values))} different numbers; the answer shows one group for each. Confirm the plan "
        f"if {name} is a code such as a site or clinic, or ask about a range of it such as '{name} over "
        f"{sorted(values)[len(values) // 2]:g}'."
    )


def indicator_statistic_reason(question: str, plan: Any, values_for: Callable[[str], list[Any]]) -> str | None:
    """One sentence when the question asks a statistic of a 0/1 column with no codebook, or None.

    "average outcome by site" on a column holding only 0 and 1 reports the share coded 1. That
    is the asked rate only if 1 means yes, which nothing in the data says, so the plan is held
    for the user's confirmation. A header codebook ("outcome 0: No 1: Yes") settles it.
    """
    metric = getattr(plan, "metric", None)
    if not metric or not any(word in STATISTIC_WORDS for word in tokenize(question)):
        return None
    if parse_column_name(str(metric)).value_mapping:
        return None
    present = {value for value in values_for(str(metric)) if value is not None}
    numbers = {float(value) for value in present if isinstance(value, int | float) and not isinstance(value, bool)}
    if len(numbers) != len(present) or numbers != {0.0, 1.0}:
        return None
    name = _short(str(metric))
    return (
        f"{name} holds only 0 and 1, and nothing in the data says which one means yes; its average is the "
        f"share coded 1. Confirm the plan if 1 means yes, or ask 'how many patients have {name} 1'."
    )


def _is_grouping(column: str, values_for: Callable[[str], list[Any]]) -> bool:
    """A column with a few stored labels or codes (sex, site, treatment), not an identifier."""
    from clinical_analytics.core.nl_query_config import MAX_NUMERIC_GROUP_LEVELS

    stored = values_for(column)
    count = cached(stored, column, "distinct", lambda: len({value for value in stored if value is not None}))
    return 2 <= count <= MAX_NUMERIC_GROUP_LEVELS


def _statistic_missing(
    tokens: list[str],
    unused: list[int],
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    alias_index: dict[str, str] | None,
    namer: _ColumnNamer,
) -> str | None:
    """ "ldl for women?": a measure is named but no statistic of it is asked for."""
    if (
        getattr(plan, "intent", None) != "DESCRIBE"
        or getattr(plan, "metric", None)
        or any(word in INTENT_WORDS["STAT"] or word in INTENT_WORDS["COUNT"] for word in tokens)
    ):
        return None
    numeric = {column for column in columns if _is_measure(values_for(column))}
    for at in unused:
        # Only a word naming one column, and that a measure: "covid19" names many yes/no items
        # beside one coded number, and "average covid19" would be the wrong next step.
        named = sorted(namer.columns_named(tokens[at]))
        if len(named) == 1 and named[0] in numeric:
            return f"Say which statistic you want, for example 'average {tokens[at]}' or 'median {tokens[at]}'."
    return None


def _short(column: str) -> str:
    from clinical_analytics.analysis.variable_types import short_name

    return short_name(column)


_DIRECTION_SHOWN = {
    "__ge__": "at least",
    "__le__": "at most",
    "__gt__": "over",
    "__lt__": "under",
    "__gt_age__": "over",
    "__lt_age__": "under",
}


def _unused_at(
    tokens: list[str],
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    alias_index: dict[str, str] | None,
    context: Any,
) -> list[int]:
    """Positions of the tokens the plan does not bind."""
    plan = as_executed(plan)  # a grouping execution drops is not applied, so its words are unused
    used = [False] * len(tokens)
    intent = str(getattr(plan, "intent", "") or "")
    filters = list(getattr(plan, "filters", []) or [])
    steps = list(getattr(plan, "cohort_steps", []) or [])
    group_by = getattr(plan, "group_by", None)
    referenced = _referenced_columns(plan, context)
    numeric_columns = {column for column in columns if _is_measure(values_for(column))}
    ranges = numeric_ranges(columns, values_for)

    # Column names, as a whole phrase ("statin use 1 yes 2 no") or by alias/stem.
    for column in referenced:
        phrases = [name_tokens(column), name_tokens(parse_column_name(column).display_name)]
        phrases += [alias.split() for alias, target in (alias_index or {}).items() if target == column]
        for phrase in phrases:
            _mark_phrase(tokens, used, phrase)
    for at, word in enumerate(tokens):
        if used[at] or word in LOGIC_WORDS or word.isdigit():
            continue
        if any(_column_for_word(word, [column], alias_index) == column for column in referenced):
            used[at] = True
        elif any(stem_match(word, alias) for alias, target in (alias_index or {}).items() if target in referenced):
            used[at] = True

    # Bound values: equality/IN filters, equality cohort steps, the proportion/rate value.
    bound_words: dict[str, set[str]] = {}
    for spec in filters:
        if spec.operator in {"==", "IN", "!=", "NOT_IN"}:
            values = spec.value if isinstance(spec.value, list) else [spec.value]
            bound_words.setdefault(spec.column, set()).update(_value_words(spec.column, values))
    for step in steps:
        if step.op == "eq":
            bound_words.setdefault(step.column, set()).update(_value_words(step.column, [step.value]))
    for column_attr, value_attr in (("proportion_column", "proportion_value"), ("rate_column", "rate_value")):
        share_column = getattr(plan, column_attr, None)
        if share_column:
            share_value = getattr(plan, value_attr, None)
            bound_words.setdefault(str(share_column), set()).update(_value_words(str(share_column), [share_value]))
    for at, word in enumerate(tokens):
        if used[at] or word in LOGIC_WORDS:
            continue
        if any(stem_match(word, value) for words in bound_words.values() for value in words):
            used[at] = True

    # Values of the grouping column count only when the question names every stored value
    # ("between drug and placebo", "women and men"): then grouping covers exactly what was
    # asked. Naming some of them ("in the north", "north and south" of three sites) is a
    # limit the plan did not apply.
    # The plan's own columns are always nameable, even if a caller's column list omits one.
    namer = _ColumnNamer([*columns, *sorted(referenced - set(columns))], values_for, alias_index)
    rate_at = tokens.index("__rate__") if "__rate__" in tokens else None
    rate_ok = rate_at is not None and rate_direction_ok(tokens, rate_at, plan, namer)
    if rate_ok and rate_at is not None:
        # "are men more likely ... than women": the named groups are the sides compared, all
        # of which the rate within each group shows; "than" introduces the other side.
        group_values = _value_words(str(group_by), values_for(str(group_by)))
        for at, word in enumerate(tokens):
            if not used[at] and (word == "than" or (_names_value(word, group_values) and at < rate_at)):
                used[at] = True
            elif not used[at] and _names_value(word, group_values) and at > 0 and tokens[at - 1] == "than":
                used[at] = True
    if group_by:
        stored = {str(value) for value in values_for(group_by) if value is not None}
        group_values = _value_words(group_by, values_for(group_by))
        hits = [at for at, word in enumerate(tokens) if not used[at] and _names_value(word, group_values)]
        named = {_value_key(tokens[at], group_by, values_for) for at in hits}
        if stored and named >= stored:
            for at in hits:
                used[at] = True

    # Numbers: bound to a comparison of the same direction on the column they belong to.
    metric = getattr(plan, "metric", None)
    metric_mentions = sum(1 for word in tokens if metric and metric in namer.columns_named(word))
    for at, word in enumerate(tokens):
        if used[at] or not re.fullmatch(r"\d+(?:\.\d+)?", word):
            continue
        direction_at = _direction_at(tokens, at)
        if direction_at is None:
            continue
        owner = number_owner(tokens, at, columns, numeric_columns, alias_index, ranges)
        op = DIRECTION_OPS[tokens[direction_at]]
        # "not over 65", "excluding those with ldl over 160": a comparison a negation may
        # reach is never read, as asked or inverted. The number stays unused, so the plan holds.
        if comparison_scope(tokens, at, direction_at) != "clear":
            continue
        # "median ldl over 70": when the only mention of the summarised measure is also the
        # number's owner, the cutoff could equally mean age. Ambiguous: the number stays unused.
        # Not when the words say age ("older than 50", "aged 40 to 60").
        lead = min(at, direction_at) - 1
        if lead >= 1 and tokens[lead] == "__rlo__" and _NUMBER_TOKEN.fullmatch(tokens[lead - 1]):
            lead -= 2  # upper end of a range: the words before the lower end say what it is
        says_age = tokens[direction_at] in _AGE_WORDS or (lead >= 0 and tokens[lead] in _AGE_WORDS)
        summarised = (
            intent in {"DESCRIBE", "COMPARE_GROUPS"} and owner == metric and metric_mentions <= 1 and not says_age
        )
        if owner is not None and not summarised and _has_comparison(filters, steps, owner, op, float(word)):
            used[at] = True
            used[direction_at] = True
            # "over 100 kg", "above 160 mg/dl", "over 365 days": the unit is part of the comparison.
            units = unit_words(owner) | (_AGE_UNIT_WORDS if owner == age_column(columns) else set())
            after = max(at, direction_at) + 1
            while after < len(tokens) and tokens[after] in units:
                used[after] = True
                after += 1
            # "aged 65 or older": the age word is part of the bound comparison.
            lead = min(at, direction_at) - 1
            if lead >= 0 and tokens[lead] in {"aged", "age"} and owner == age_column(columns):
                used[lead] = True

    # Fixed vocabulary and intent words. A word of the test the plan line names ("logistic
    # regression" for a predictor plan) is used by that plan.
    counting = intent == "COUNT" or share_planned(plan)
    label_words = label_words_of(str(intent))
    for at, word in enumerate(tokens):
        if used[at]:
            continue
        if word in FUNCTION_WORDS or word in SUBJECT_WORDS or word in LINK_WORDS:
            used[at] = True
        elif word in INTENT_WORDS["COUNT"]:
            used[at] = counting
        elif word in INTENT_WORDS["STAT"]:
            used[at] = statistic_planned(word, plan) or (
                word == "median" and any(step.op == "gt_median" for step in steps)
            )
        elif word in INTENT_WORDS["COMPARE_GROUPS"]:
            # A comparison is answered by a test across groups or by the asked statistic per group.
            used[at] = intent == "COMPARE_GROUPS" or grouped_statistic(plan) is not None
        elif any(word in INTENT_WORDS[name] for name in ("CORRELATIONS", "FIND_PREDICTORS", "SURVIVAL")):
            used[at] = any(word in INTENT_WORDS[name] and intent == name for name in INTENT_WORDS)
        elif (
            word in label_words or any(inflection_of(word, own) for own in INTENT_WORDS.get(str(intent), ()))
        ) and not namer.columns_named(word):
            # The test's own name, or an inflection of the plan's intent word ("counted"). Not a
            # word that names a column ("group" of a trial's group column): the plan must read it.
            used[at] = True
        else:
            used[at] = _logic_used(tokens, at, plan, filters, steps, values_for, namer, context)

    # "admitted to hospital": the verb points at the place the next word names; it is used when
    # that word names a column the plan reads ("admitted to the ICU" with no ICU column is not).
    for at, word in enumerate(tokens):
        if used[at] or word not in PLACE_VERBS:
            continue
        place = next(
            (index for index in range(at + 1, min(at + 4, len(tokens))) if tokens[index] not in FUNCTION_WORDS), None
        )
        if place is not None and used[place] and namer.columns_named(tokens[place]) & referenced:
            used[at] = True

    # "<item> versus not <item>": both sides of one yes/no column, shown as its two groups. Used
    # only when the plan counts exactly those groups, with no limit a negation could change.
    tail = complement_tail(tokens)
    if tail is not None and _complement_grouped(tokens, tail, plan, values_for, namer):
        for at in range(tail, len(tokens)):
            used[at] = True

    # A word a negation may or may not reach ("hypertension" in "without diabetes and
    # hypertension", "placebo" in "excluding women on placebo") is never used by a limit,
    # whichever way the plan reads it: the words do not say, so the plan holds. Comparisons
    # were read above from where they start, by the same rule.
    # A word naming the same column as the negated word is part of that one item ("without a
    # normal dexa result": "dexa result" names the column whose value "normal" is negated).
    limited = _limited_columns(plan)
    for at, word in enumerate(tokens):
        if not used[at] or word in NO_LIMIT_WORDS or word in LOGIC_WORDS or _NUMBER_TOKEN.fullmatch(word):
            continue
        named = namer.columns_named(word) & limited
        if not named or negation_scope(tokens[:at]) != "uncertain":
            continue
        governed = _governed_word(tokens, at)
        if governed is None or not named & namer.columns_named(governed):
            used[at] = False

    return [at for at in range(len(tokens)) if not used[at]]


def _complement_grouped(
    tokens: list[str], tail: int, plan: Any, values_for: Callable[[str], list[Any]], namer: _ColumnNamer
) -> bool:
    """The plan counts each value of the two-valued column the words before the tail name."""
    group_by = getattr(plan, "group_by", None)
    if not group_by or not distribution_planned(plan):
        return False
    stored = {value for value in values_for(str(group_by)) if value is not None}
    if len(stored) != 2 or negation_scope(tokens[:tail]) != "clear":
        return False
    return any(group_by in namer.columns_named(word) for word in tokens[:tail])


def _governed_word(tokens: list[str], before: int) -> str | None:
    """The first word after the last negation before ``before`` that carries a limit."""
    cues = [at for at in range(before) if tokens[at] in NEGATION_CUES]
    if not cues:
        return None
    return next((word for word in tokens[cues[-1] + 1 : before] if word not in NO_LIMIT_WORDS), None)


def _limited_columns(plan: Any) -> set[str]:
    """Columns the plan limits the patients on: filters, cohort steps, and a share's own column."""
    limited = {str(spec.column) for spec in getattr(plan, "filters", []) or []}
    limited |= {str(step.column) for step in getattr(plan, "cohort_steps", []) or []}
    for attr in ("proportion_column", "rate_column"):
        value = getattr(plan, attr, None)
        if value:
            limited.add(str(value))
    return limited


class _ColumnNamer:
    """Which dataset columns a question word names: by name, alias, stem, or a stored value."""

    def __init__(
        self, columns: list[str], values_for: Callable[[str], list[Any]], alias_index: dict[str, str] | None
    ) -> None:
        self._columns = columns
        self._alias_index = alias_index
        self._values = {column: _value_words(column, values_for(column)) for column in columns}

    def columns_named(self, word: str) -> set[str]:
        if word in LOGIC_WORDS or word in FUNCTION_WORDS or word in SUBJECT_WORDS or word in LINK_WORDS:
            return set()
        named = {column for column in self._columns if _names_column(word, column)}
        for alias, column in (self._alias_index or {}).items():
            if " " not in alias and stem_match(word, alias):
                named.add(column)
        named |= {column for column, words in self._values.items() if _names_value(word, words)}
        return named

    def neighbour(self, tokens: list[str], at: int, step: int, reach: int = 4) -> set[str]:
        """Columns named by the nearest content word after (step=1) or before (step=-1) ``at``."""
        index = at + step
        while 0 <= index < len(tokens) and abs(index - at) <= reach:
            word = tokens[index]
            if word not in FUNCTION_WORDS and word not in SUBJECT_WORDS and word not in LINK_WORDS:
                named = self.columns_named(word)
                if named or word not in LOGIC_WORDS:
                    return named
            index += step
        return set()


def _names_value(word: str, values: AbstractSet[str]) -> bool:
    if isinstance(values, IndexedWords):
        return values.named_by(word)
    return any(stem_match(word, value) for value in values)


def _value_key(word: str, column: str, values_for: Callable[[str], list[Any]]) -> str:
    """The stored value a word names (so "women" and "females" count as one value)."""
    stored = values_for(column)
    if isinstance(stored, ColumnValues) and stored.column == column:
        return stored.derived("value_keys", lambda: _ValueKeys(column, stored)).key(word)
    for value in stored:
        if value is not None and _names_value(word, _value_words(column, [value])):
            return str(value)
    return word


class _ValueKeys:
    """_value_key by index: the first stored value (in stored order) one of whose words the
    word stem-matches, found without trying every value."""

    def __init__(self, column: str, stored: ColumnValues) -> None:
        self._stored = stored
        self._first: dict[str, int] = {}
        for at, value in enumerate(stored):
            if value is None:
                continue
            for word in _words_of(column, [value], stored.match_stored):
                self._first.setdefault(word, at)
        self._index = StemIndex(self._first, stem_match)

    def key(self, word: str) -> str:
        named = self._index.matches(word)
        if not named:
            return word
        return str(self._stored[min(self._first[target] for target in named)])


def _logic_used(
    tokens: list[str],
    at: int,
    plan: Any,
    filters: list[Any],
    steps: list[Any],
    values_for: Callable[[str], list[Any]],
    namer: _ColumnNamer,
    context: Any,
) -> bool:
    """A logic word is used only when the plan holds its operator on the column it is about."""
    word = tokens[at]
    intent = str(getattr(plan, "intent", "") or "")
    if word in NEGATION_WORDS:
        targets = namer.neighbour(tokens, at, 1)
        return any(_negated_in_plan(plan, filters, steps, column, values_for) for column in targets)
    if word in OR_WORDS:
        targets = namer.neighbour(tokens, at, -1) | namer.neighbour(tokens, at, 1)
        return any(
            spec.operator == "IN" and isinstance(spec.value, list) and len(spec.value) >= 2 and spec.column in targets
            for spec in filters
        )
    if word in BETWEEN_WORDS:
        after = namer.neighbour(tokens, at, 1)
        if intent == "COMPARE_GROUPS" or grouped_statistic(plan) is not None:
            return getattr(plan, "group_by", None) in after
        if intent == "CORRELATIONS":
            variables = {getattr(plan, "metric", None), getattr(plan, "group_by", None)}
            variables |= set(getattr(context, "predictor_variables", None) or [])
            return bool(after & variables)
        return _has_range(filters, steps)
    if word == "__rate__":
        return rate_direction_ok(tokens, at, plan, namer)
    if word in RATE_WORDS:
        return share_planned(plan) or distribution_planned(plan)
    if word in BY_WORDS:
        return getattr(plan, "group_by", None) in namer.neighbour(tokens, at, 1)
    if word in CUTOFF_WORDS:
        return any(step.op == "ge_percent_rank" for step in steps)
    if word in DIRECTION_OPS:
        return DIRECTION_OPS[word] == "gt" and any(step.op == "gt_median" for step in steps)
    return False


def rate_direction_ok(tokens: list[str], at: int, plan: Any, namer: _ColumnNamer) -> bool:
    """ "are <groups> more likely to <rate>": the plan's rate is what follows "more likely" and
    its grouping is named before it. A plan with the two swapped answers the reversed question
    ("are statin users more likely to be women" is not the statin rate of each sex)."""
    rate_column = getattr(plan, "rate_column", None)
    group_by = getattr(plan, "group_by", None)
    if not (share_planned(plan) and rate_column and group_by):
        return False
    before = set().union(*(namer.columns_named(word) for word in tokens[:at]))
    after = set().union(*(namer.columns_named(word) for word in tokens[at + 1 :]))
    return rate_column in after and rate_column not in before and group_by in before


def share_planned(plan: Any) -> bool:
    """The plan computes a share: a proportion, or a rate within each group.

    A rate with no group is not computed (execution refuses it), so its words stay unused.
    """
    if getattr(plan, "proportion_column", None):
        # One share of everyone: a grouping beside it is not computed, so the words stay unused.
        return not getattr(plan, "group_by", None)
    return bool(getattr(plan, "rate_column", None) and getattr(plan, "group_by", None))


def distribution_planned(plan: Any) -> bool:
    """The plan counts every group of one column with no other limit: the share of patients in
    each group ("what share of patients have each MS type") is each count over the stated
    total. Any limit, proportion or rate makes the share a different number, so it is not."""
    return (
        getattr(plan, "intent", None) == "COUNT"
        and bool(getattr(plan, "group_by", None))
        and not getattr(plan, "filters", None)
        and not getattr(plan, "cohort_steps", None)
        and not getattr(plan, "proportion_column", None)
        and not getattr(plan, "rate_column", None)
    )


def statistic_planned(word: str, plan: Any) -> bool:
    """A statistic word is used only when the plan computes that statistic of a measure.

    DESCRIBE computes the plan's own statistic. COMPARE_GROUPS compares means, or computes
    the asked statistic per group (query_plan.grouped_statistic). A plan that computes means
    leaves "median" unused, so the plan is held.
    """
    asked = STATISTIC_WORDS.get(word)
    if asked is None or not getattr(plan, "metric", None):
        return False
    intent = getattr(plan, "intent", None)
    if intent == "DESCRIBE":
        return asked == getattr(plan, "statistic", None)
    if intent == "COMPARE_GROUPS":
        return asked == (grouped_statistic(plan) or "mean")
    return False


def _negated_in_plan(
    plan: Any, filters: list[Any], steps: list[Any], column: str, values_for: Callable[[str], list[Any]]
) -> bool:
    """The plan excludes on this column: != / NOT IN, or equality to the column's "No" code."""
    for spec in filters:
        if spec.column != column:
            continue
        if spec.operator in {"!=", "NOT_IN"}:
            return True
        if spec.operator == "==" and _is_no_value(column, spec.value, values_for):
            return True
    if any(
        step.op == "eq" and step.column == column and _is_no_value(column, step.value, values_for) for step in steps
    ):
        return True
    for column_attr, value_attr in (("proportion_column", "proportion_value"), ("rate_column", "rate_value")):
        if getattr(plan, column_attr, None) == column and _is_no_value(
            column, getattr(plan, value_attr, None), values_for
        ):
            return True
    return False


def _referenced_columns(plan: Any, context: Any) -> set[str]:
    referenced: set[str] = set()
    for attr in ("metric", "group_by", "proportion_column", "rate_column"):
        value = getattr(plan, attr, None)
        if value:
            referenced.add(str(value))
    referenced |= {str(spec.column) for spec in getattr(plan, "filters", []) or []}
    referenced |= {str(step.column) for step in getattr(plan, "cohort_steps", []) or []}
    if context is not None:
        referenced |= {str(item) for item in getattr(context, "predictor_variables", []) or [] if item}
        for attr in ("time_variable", "event_variable"):
            value = getattr(context, attr, None)
            if value:
                referenced.add(str(value))
    return referenced


def _value_words(column: str, values: list[Any]) -> AbstractSet[str]:
    """Words that name a value of the column: its stored labels and their header code labels.

    For a column's own distinct values the set is built once and indexed (IndexedWords).
    """
    if isinstance(values, ColumnValues) and values.column == column:
        return values.derived(
            "value_words", lambda: IndexedWords(_words_of(column, values, values.match_stored), stem_match)
        )
    return _words_of(column, values, True)


def _words_of(column: str, values: list[Any], match_stored: bool) -> set[str]:
    """Value words; with ``match_stored`` False only the header codebook's labels count
    (value_index: identifiers and many-valued measures are never named by a word)."""
    mapping = parse_column_name(column).value_mapping or {}
    words: set[str] = set()
    for value in values:
        if value is None:
            continue
        label = str(value).strip().lower()
        if match_stored:
            if not re.fullmatch(r"\d+(?:\.\d+)?", label):
                # A stored label's digits are part of it ("Type 2"); a bare numeric code is not a word.
                words |= set(name_tokens(label))
            words |= _SEX_INFLECTIONS.get(label, set())
        code = label[:-2] if label.endswith(".0") else label
        if code in mapping:
            words |= set(name_tokens(mapping[code]))
    return {word for word in words if word not in LOGIC_WORDS}


def _mark_phrase(tokens: list[str], used: list[bool], phrase: list[str]) -> None:
    if not phrase:
        return
    size = len(phrase)
    for start in range(len(tokens) - size + 1):
        if tokens[start : start + size] == phrase and (size > 1 or phrase[0] not in LOGIC_WORDS):
            for at in range(start, start + size):
                used[at] = True


def _direction_at(tokens: list[str], at: int) -> int | None:
    if at + 1 < len(tokens) and tokens[at + 1] in _POSTFIX:
        return at + 1
    if at >= 1 and tokens[at - 1] in DIRECTION_OPS and tokens[at - 1] not in _POSTFIX:
        return at - 1
    return None


def _has_comparison(filters: list[Any], steps: list[Any], column: str, op: str, number: float) -> bool:
    filter_op = {"gt": ">", "lt": "<", "ge": ">=", "le": "<="}[op]
    for spec in filters:
        if spec.column == column and spec.operator == filter_op and _same_number(spec.value, number):
            return True
    return any(step.column == column and step.op == op and _same_number(step.value, number) for step in steps)


def _has_range(filters: list[Any], steps: list[Any]) -> bool:
    lows = {spec.column for spec in filters if spec.operator in {">", ">="}} | {
        s.column for s in steps if s.op in {"gt", "ge"}
    }
    highs = {spec.column for spec in filters if spec.operator in {"<", "<="}} | {
        s.column for s in steps if s.op in {"lt", "le"}
    }
    return bool(lows & highs)


def _same_number(value: Any, number: float) -> bool:
    try:
        return float(value) == number
    except (TypeError, ValueError):
        return False


def _is_no_value(column: str, value: Any, values_for: Callable[[str], list[Any]]) -> bool:
    from clinical_analytics.core.cohort_constraints import yes_no_values

    _yes, no = yes_no_values(column, values_for(column))
    return no is not None and str(no) == str(value)


def _is_measure(values: list[Any]) -> bool:
    if isinstance(values, ColumnValues):
        return values.derived("is_measure", lambda: _measure(values))
    return _measure(values)


def _measure(values: list[Any]) -> bool:
    present = [value for value in values if value is not None]
    if len(present) <= 2:
        return False
    return all(isinstance(value, int | float) and not isinstance(value, bool) for value in present)


# ============================================================================
# Reverse accounting: every plan element must be asked for by the question.
# ============================================================================

# Words that say the question is about age without naming the column.
_AGE_JUSTIFY = frozenset({"age", "aged", "old", "older", "oldest", "younger", "youngest", "elderly"}) | {
    "__gt_age__",
    "__lt_age__",
}
_FILTER_OP = {">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
_NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def unjustified_reason(elements: list[str]) -> str:
    """One plain sentence naming the plan elements the question did not ask for."""
    return (
        f"The plan uses {_listed(elements)}, which your question doesn't ask for; confirm the plan if that is "
        "what you meant, or name what you want."
    )


def unjustified_elements(
    question: str,
    plan: Any,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    alias_index: dict[str, str] | None = None,
    context: Any = None,
) -> list[str]:
    """Plan elements no word of the question justifies, in plan order, each written exactly as
    the plan line shows it (execution_gate.limit_text) so the reader can find it on the card.

    The forward check (unused_words) asks whether every word is used; this asks whether every
    element was asked for. A measure, grouping, filter column, value and operator each need a
    word of the question: the column's name, an alias, a word of a stored value or its
    codebook label, the number itself with a comparison word of the same direction, or a
    negation word for an exclusion. Anything else was added by the parser and is not run
    without the user's confirmation.
    """
    check = _Justifier(question, plan, columns, values_for, alias_index)
    missing: list[str] = []
    intent = str(getattr(plan, "intent", "") or "")
    metric = getattr(plan, "metric", None)
    if metric and intent != "COUNT" and not check.column_named(str(metric)):
        missing.append(f"outcome {metric}")
    group_by = getattr(plan, "group_by", None)
    if group_by and not (check.column_named(str(group_by)) or check.any_value_named(str(group_by))):
        missing.append(f"grouping {group_by}")
    for spec in getattr(plan, "filters", []) or []:
        if not check.filter_justified(spec):
            missing.append(limit_text(spec))
    for step in getattr(plan, "cohort_steps", []) or []:
        if not check.step_justified(step):
            missing.append(limit_text(step))
    for column_attr, value_attr in (("proportion_column", "proportion_value"), ("rate_column", "rate_value")):
        share_column = getattr(plan, column_attr, None)
        if share_column and not check.equality_justified(str(share_column), [getattr(plan, value_attr, None)]):
            missing.append(limit_text(CohortStep("eq", str(share_column), getattr(plan, value_attr, None))))
    for predictor in getattr(context, "predictor_variables", None) or []:
        if predictor and predictor not in {metric, group_by} and not check.column_named(str(predictor)):
            missing.append(f"predictor {predictor}")
    return missing


class _Justifier:
    """Answers "did the question ask for this?" for one question and its dataset."""

    def __init__(
        self,
        question: str,
        plan: Any,
        columns: list[str],
        values_for: Callable[[str], list[Any]],
        alias_index: dict[str, str] | None,
    ) -> None:
        self.tokens = tokenize(question)
        self.columns = columns
        self.values_for = values_for
        self.alias_index = alias_index
        self.age = age_column(columns)
        referenced = _referenced_columns(plan, None)
        self.all_columns = [*columns, *sorted(referenced - set(columns))]
        self.numeric = {column for column in self.all_columns if _is_measure(values_for(column))}
        self.ranges = numeric_ranges(self.all_columns, values_for)
        self.label_digits = self._digits_inside_labels()

    def _digits_inside_labels(self) -> dict[int, str]:
        """Number tokens that are part of a stored label the question spells out, by position.

        The 2 of "type 2 diabetes" belongs to the label "Type 2" of the diabetes column. It can
        justify that value only: never a number on another column ("Creatinine = 2.2").
        """
        claimed: dict[int, str] = {}
        for column in self.all_columns:
            stored = self.values_for(column)
            by_first = cached(stored, column, "digit_phrases", lambda: _digit_phrases(column, stored))
            for start, token in enumerate(self.tokens):
                for phrase in by_first.get(token, ()):
                    size = len(phrase)
                    if self.tokens[start : start + size] != phrase:
                        continue
                    for at in range(start, start + size):
                        if _NUMBER_TOKEN.fullmatch(self.tokens[at]):
                            claimed.setdefault(at, column)
        return claimed

    def _number_free_for(self, at: int, column: str) -> bool:
        return self.label_digits.get(at, column) == column

    def column_named(self, column: str) -> bool:
        if column == self.age and any(token in _AGE_JUSTIFY for token in self.tokens):
            return True
        if any(_column_for_word(token, [column], self.alias_index) == column for token in self.tokens):
            return True
        phrases = [name_tokens(column), name_tokens(parse_column_name(column).display_name)]
        phrases += [alias.split() for alias, target in (self.alias_index or {}).items() if target == column]
        return any(_contains(self.tokens, phrase) for phrase in phrases if phrase)

    def any_value_named(self, column: str) -> bool:
        words = _value_words(column, self.values_for(column))
        return any(_names_value(token, words) for token in self.tokens if token not in FUNCTION_WORDS)

    def value_named(self, column: str, value: Any) -> bool:
        """The question names this stored value: every word of its label, or its code label."""
        if value is None:
            return False
        if isinstance(value, bool) or self._is_polarity(column, value):
            # Yes/no columns are bound by naming the column ("diabetics", "statin users").
            return self.column_named(column)
        label = str(value).strip().lower()
        code = label[:-2] if label.endswith(".0") else label
        code_label = (parse_column_name(column).value_mapping or {}).get(code)
        if code_label and self._words_named(name_tokens(code_label.lower())):
            return True
        if _NUMBER_TOKEN.fullmatch(label):
            return self.column_named(column) and self._bare_number_present(column, float(label))
        inflections = _SEX_INFLECTIONS.get(label, set())
        if inflections & set(self.tokens):
            return True
        return self._words_named([word for word in name_tokens(label) if word not in LOGIC_WORDS])

    def _negated_at_value(self, column: str, value: Any) -> bool:
        """Every word naming this value stands right after a negation, by the shared scope rule."""
        words = _value_words(column, [value])
        named = [
            at for at, token in enumerate(self.tokens) if token not in FUNCTION_WORDS and _names_value(token, words)
        ]
        return bool(named) and all(negation_scope(self.tokens[:at]) == "negated" for at in named)

    def equality_justified(self, column: str, values: list[Any]) -> bool:
        if not values or not all(self.value_named(column, value) for value in values):
            return False
        # A stored text value names its own column ("placebo"); a number needs the column named.
        return True

    def filter_justified(self, spec: Any) -> bool:
        values = spec.value if isinstance(spec.value, list) else [spec.value]
        if spec.operator in {"==", "IN"}:
            return self.equality_justified(spec.column, values)
        if spec.operator in {"!=", "NOT_IN"}:
            # Each excluded value must be named right after its own negation ("not on placebo").
            # A negation elsewhere in the question ("on placebo without outcome") never justifies
            # excluding a value the question asks for.
            return self.equality_justified(spec.column, values) and all(
                self._negated_at_value(spec.column, value) for value in values
            )
        if spec.operator in _FILTER_OP:
            return self.comparison_justified(spec.column, _FILTER_OP[spec.operator], spec.value)
        return False

    def step_justified(self, step: Any) -> bool:
        if step.op == "eq":
            return self.equality_justified(step.column, [step.value])
        if step.op in {"gt", "lt", "ge", "le"}:
            return self.comparison_justified(step.column, step.op, step.value)
        if step.op in {"gt_median", "ge_percent_rank"}:
            cutoff = any(token in CUTOFF_WORDS or token in {"median", "older", "higher"} for token in self.tokens)
            return cutoff and self.column_named(step.column)
        if step.op == "not_null":
            return self.column_named(step.column)
        return False  # an op the peel never builds ("ne", "in") is never asked for

    def comparison_justified(self, column: str, op: str, value: Any) -> bool:
        """The number is in the question, with a comparison word of the same direction, on this column."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        between = "between" in self.tokens
        for at, token in enumerate(self.tokens):
            if not _NUMBER_TOKEN.fullmatch(token) or float(token) != number or not self._number_free_for(at, column):
                continue
            ops: set[str] = set()
            for near in (at - 1, at + 1):
                if not (0 <= near < len(self.tokens) and self.tokens[near] in DIRECTION_OPS):
                    continue
                # A comparison a negation may reach justifies nothing, as asked or inverted.
                if comparison_scope(self.tokens, at, near) == "clear":
                    ops.add(DIRECTION_OPS[self.tokens[near]])
            plain = negation_scope(self.tokens[:at]) == "clear"
            if not (op in ops or (between and plain and op in {"ge", "le", "gt", "lt"})):
                continue
            owner = number_owner(self.tokens, at, self.columns, self.numeric, self.alias_index, self.ranges)
            if owner == column or self.column_named(column):
                return True
        return False

    def _bare_number_present(self, column: str, number: float) -> bool:
        """The number is in the question with no comparison word on either side ("creatinine 2.2")."""
        for at, token in enumerate(self.tokens):
            if not _NUMBER_TOKEN.fullmatch(token) or float(token) != number or not self._number_free_for(at, column):
                continue
            if not any(
                0 <= near < len(self.tokens) and self.tokens[near] in DIRECTION_OPS for near in (at - 1, at + 1)
            ):
                return True
        return False

    def _words_named(self, words: list[str]) -> bool:
        return bool(words) and all(any(stem_match(token, word) for token in self.tokens) for word in words)

    def _is_polarity(self, column: str, value: Any) -> bool:
        from clinical_analytics.core.cohort_constraints import yes_no_values

        yes, no = yes_no_values(column, self.values_for(column))
        return value is not None and str(value) in {str(yes), str(no)} and yes is not None


def _digit_phrases(column: str, stored: list[Any]) -> dict[str, list[list[str]]]:
    """Labels of two or more words with a number among them ("Type 2"), by first word."""
    labels = [str(value) for value in stored if isinstance(value, str)] if matches_stored(stored) else []
    labels += list((parse_column_name(column).value_mapping or {}).values())
    by_first: dict[str, list[list[str]]] = {}
    for label in labels:
        phrase = name_tokens(label)
        if len(phrase) < 2 or not any(_NUMBER_TOKEN.fullmatch(part) for part in phrase):
            continue
        if phrase not in by_first.setdefault(phrase[0], []):
            by_first[phrase[0]].append(phrase)
    return by_first


def _contains(tokens: list[str], phrase: list[str]) -> bool:
    size = len(phrase)
    return any(tokens[start : start + size] == phrase for start in range(len(tokens) - size + 1))

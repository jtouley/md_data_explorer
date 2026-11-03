"""Word accounting: every content word of a question must be bound by the plan that runs.

A plan that leaves a word of the question unused answers a different question, whatever its
confidence. This gate runs once, after parsing (any tier) and before execution, and holds
such a plan for confirmation with the unused words named.

A word is used only when:
- it is in a small reviewed list of function words or subject nouns (FUNCTION_WORDS, SUBJECT_WORDS);
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
from dataclasses import dataclass
from typing import Any

from clinical_analytics.core.column_parser import parse_column_name

# Reviewed vocabulary. Adding a word here makes the gate accept it in any question.
FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a", "all", "among", "amongst", "an", "and", "any", "are", "as", "at", "be", "been", "being",
        "did", "do", "does", "for", "from", "had", "has", "have", "having", "i", "in", "is", "it",
        "its", "me", "of", "on", "our", "s", "that", "the", "their", "them", "there", "these", "they",
        "this", "those", "to", "us", "was", "we", "were", "what", "which", "who", "whom", "whose",
        "with", "within",
    }
)  # fmt: skip
SUBJECT_WORDS: frozenset[str] = frozenset(
    {
        "case", "cases", "cohort", "group", "groups", "individuals", "participant", "participants",
        "patient", "patients", "people", "person", "persons", "subject", "subjects", "user", "users",
    }
)  # fmt: skip
# Verbs that only link a patient to a stored value ("patients who received placebo").
LINK_WORDS: frozenset[str] = frozenset({"given", "received", "receiving", "taking", "treated"})
INTENT_WORDS: dict[str, frozenset[str]] = {
    "COUNT": frozenset({"how", "many", "number", "count", "total"}),
    "STAT": frozenset(
        {"average", "avg", "mean", "median", "minimum", "maximum", "min", "max", "std", "standard", "deviation"}
    ),
    "COMPARE_GROUPS": frozenset({"compare", "comparison", "versus", "vs", "difference"}),
    "CORRELATIONS": frozenset({"correlation", "correlated", "correlate", "relationship", "association"}),
    "FIND_PREDICTORS": frozenset({"predict", "predicts", "predictor", "predictors"}),
    "SURVIVAL": frozenset({"survival"}),
}
# Logic words: used only when the plan holds the matching operator.
NEGATION_WORDS: frozenset[str] = frozenset(
    {"not", "non", "no", "without", "excluding", "exclude", "except", "outside", "never", "__neg__"}
)
OR_WORDS: frozenset[str] = frozenset({"or"})
BETWEEN_WORDS: frozenset[str] = frozenset({"between"})
RATE_WORDS: frozenset[str] = frozenset(
    {"proportion", "percent", "percentage", "share", "rate", "fraction", "likely", "__rate__"}
)
BY_WORDS: frozenset[str] = frozenset({"by"})
CUTOFF_WORDS: frozenset[str] = frozenset({"top", "decile", "highest", "oldest"})
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
}
_POSTFIX = {"__ge__", "__le__"}
_AGE_WORDS = {"aged", "age", "old", "__gt_age__", "__lt_age__"}
LOGIC_WORDS: frozenset[str] = (
    NEGATION_WORDS | OR_WORDS | BETWEEN_WORDS | RATE_WORDS | BY_WORDS | CUTOFF_WORDS | frozenset(DIRECTION_OPS)
)

# Multi-word logic phrases, rewritten to one placeholder before tokenising.
_PHRASES: tuple[tuple[str, str], ...] = (
    (r"n't\b", " not"),
    (r"(\d)\s*\+", r"\1 __ge__ "),
    (r"\b(?:or|and) (?:more|older|over|above|greater|higher)\b", " __ge__ "),
    (r"\bat least\b", " __ge__ "),
    (r"\b(?:or|and) (?:less|fewer|younger|under|below|lower)\b", " __le__ "),
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
}
_SEX_INFLECTIONS = {
    "f": {"f", "female", "females", "woman", "women"},
    "female": {"female", "females", "woman", "women"},
    "m": {"m", "male", "males", "man", "men"},
    "male": {"male", "males", "man", "men"},
}
HOLD_PREFIX = "I didn't use:"


@dataclass(frozen=True)
class Token:
    text: str
    index: int


def tokenize(question: str) -> list[str]:
    """Lowercase words, numbers and logic placeholders, in order."""
    text = question.lower()
    for pattern, replacement in _PHRASES:
        text = re.sub(pattern, replacement, text)
    return re.findall(r"__[a-z_]+__|\d+(?:\.\d+)?|[a-z]+", text)


def shown(word: str) -> str:
    return _SHOWN.get(word, word)


def hold_reason(words: Iterable[str]) -> str:
    quoted = ", ".join(f"'{shown(word)}'" for word in words)
    return f"{HOLD_PREFIX} {quoted}. Check the plan before running it."


def name_tokens(name: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?|[a-z]+", name.lower())


def stem_match(word: str, target: str) -> bool:
    """Same word, a plural, or a shared stem of at least five letters covering 75% of the shorter."""
    if word == target or word.rstrip("s") == target.rstrip("s"):
        return True
    if word.isdigit() or target.isdigit():
        return False
    shorter = min(len(word), len(target))
    common = 0
    for left, right in zip(word, target, strict=False):
        if left != right:
            break
        common += 1
    return common >= 5 and common >= 0.75 * shorter


def number_owner(
    tokens: list[str],
    number_at: int,
    columns: list[str],
    numeric_columns: set[str],
    alias_index: dict[str, str] | None = None,
) -> str | None:
    """Column a compared number belongs to, or None when the question does not say.

    The nearest content word before the comparison names the column ("bmi over 30" is bmi).
    Age is used only when the words say age ("older than", "aged"), or the number follows a
    subject ("patients over 65") and no other measured column is named before it.
    """
    age = next((column for column in columns if name_tokens(column) == ["age"]), None)
    start = number_at - 1
    if start >= 0 and tokens[start] in DIRECTION_OPS and tokens[start] not in _POSTFIX:
        if tokens[start] in _AGE_WORDS:
            return age
        start -= 1
    for at in range(start, -1, -1):
        word = tokens[at]
        if word in FUNCTION_WORDS:
            continue
        if word in _AGE_WORDS:
            return age
        named = _column_for_word(word, columns, alias_index)
        if named is not None:
            return named
        break
    before = tokens[:number_at]
    for word in before:
        named = _column_for_word(word, columns, alias_index)
        if named is not None and named in numeric_columns and named != age:
            return None
    return age


def _column_for_word(word: str, columns: list[str], alias_index: dict[str, str] | None) -> str | None:
    if word in LOGIC_WORDS or word in FUNCTION_WORDS or word in SUBJECT_WORDS or word.isdigit():
        return None
    for alias, column in (alias_index or {}).items():
        if " " not in alias and stem_match(word, alias):
            return column
    for column in columns:
        parts = [part for part in name_tokens(column) if not part.isdigit()]
        if parts and len(parts) <= 2 and any(stem_match(word, part) for part in parts):
            return column
    return None


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
    used = [False] * len(tokens)
    intent = str(getattr(plan, "intent", "") or "")
    filters = list(getattr(plan, "filters", []) or [])
    steps = list(getattr(plan, "cohort_steps", []) or [])
    group_by = getattr(plan, "group_by", None)
    referenced = _referenced_columns(plan, context)
    numeric_columns = {column for column in columns if _is_measure(values_for(column))}

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

    # Bound values: equality/IN filters, equality cohort steps, every value of the grouping column.
    value_words: set[str] = set()
    for spec in filters:
        if spec.operator in {"==", "IN", "!=", "NOT_IN"}:
            values = spec.value if isinstance(spec.value, list) else [spec.value]
            value_words |= _value_words(spec.column, values)
    for step in steps:
        if step.op == "eq":
            value_words |= _value_words(step.column, [step.value])
    if group_by:
        value_words |= _value_words(group_by, values_for(group_by))
    for at, word in enumerate(tokens):
        if not used[at] and word not in LOGIC_WORDS and any(stem_match(word, value) for value in value_words):
            used[at] = True

    # Numbers: bound to a comparison of the same direction on the column they belong to.
    for at, word in enumerate(tokens):
        if used[at] or not re.fullmatch(r"\d+(?:\.\d+)?", word):
            continue
        direction_at = _direction_at(tokens, at)
        if direction_at is None:
            continue
        owner = number_owner(tokens, at, columns, numeric_columns, alias_index)
        op = DIRECTION_OPS[tokens[direction_at]]
        # "median ldl over 70": a cutoff on the very measure being summarised could equally
        # mean an age cutoff. Ambiguous, so the number stays unused and the plan holds.
        summarised = (
            intent in {"DESCRIBE", "COMPARE_GROUPS"}
            and owner == getattr(plan, "metric", None)
            and direction_at >= 2
            and tokens[direction_at - 2] in INTENT_WORDS["STAT"]
        )
        if owner is not None and not summarised and _has_comparison(filters, steps, owner, op, float(word)):
            used[at] = True
            used[direction_at] = True

    # Fixed vocabulary and intent words.
    stat_ok = bool(getattr(plan, "metric", None)) and intent in {"DESCRIBE", "COMPARE_GROUPS"}
    counting = intent == "COUNT" or bool(getattr(plan, "proportion_column", None) or getattr(plan, "rate_column", None))
    for at, word in enumerate(tokens):
        if used[at]:
            continue
        if word in FUNCTION_WORDS or word in SUBJECT_WORDS or word in LINK_WORDS:
            used[at] = True
        elif word in INTENT_WORDS["COUNT"]:
            used[at] = counting
        elif word in INTENT_WORDS["STAT"]:
            used[at] = stat_ok or (word == "median" and any(step.op == "gt_median" for step in steps))
        elif any(
            word in INTENT_WORDS[name] for name in ("COMPARE_GROUPS", "CORRELATIONS", "FIND_PREDICTORS", "SURVIVAL")
        ):
            used[at] = any(word in INTENT_WORDS[name] and intent == name for name in INTENT_WORDS)
        else:
            used[at] = _logic_used(word, plan, filters, steps, values_for)

    return [shown(word) for at, word in enumerate(tokens) if not used[at]]


def _logic_used(
    word: str, plan: Any, filters: list[Any], steps: list[Any], values_for: Callable[[str], list[Any]]
) -> bool:
    intent = str(getattr(plan, "intent", "") or "")
    if word in NEGATION_WORDS:
        if any(spec.operator in {"!=", "NOT_IN"} for spec in filters):
            return True
        return any(step.op == "eq" and _is_no_value(step.column, step.value, values_for) for step in steps)
    if word in OR_WORDS:
        return any(spec.operator == "IN" and isinstance(spec.value, list) and len(spec.value) >= 2 for spec in filters)
    if word in BETWEEN_WORDS:
        return intent in {"COMPARE_GROUPS", "CORRELATIONS"} or _has_range(filters, steps)
    if word in RATE_WORDS:
        return bool(getattr(plan, "proportion_column", None) or getattr(plan, "rate_column", None))
    if word in BY_WORDS:
        return bool(getattr(plan, "group_by", None))
    if word in CUTOFF_WORDS:
        return any(step.op == "ge_percent_rank" for step in steps)
    if word in DIRECTION_OPS:
        return DIRECTION_OPS[word] == "gt" and any(step.op == "gt_median" for step in steps)
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


def _value_words(column: str, values: list[Any]) -> set[str]:
    mapping = parse_column_name(column).value_mapping or {}
    words: set[str] = set()
    for value in values:
        if value is None:
            continue
        label = str(value).strip().lower()
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
    lows = {spec.column for spec in filters if spec.operator in {">", ">="}} | {s.column for s in steps if s.op == "gt"}
    highs = {spec.column for spec in filters if spec.operator in {"<", "<="}} | {
        s.column for s in steps if s.op == "lt"
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
    present = [value for value in values if value is not None]
    if len(present) <= 2:
        return False
    return all(isinstance(value, int | float) and not isinstance(value, bool) for value in present)

# SPDX-License-Identifier: MIT
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
from clinical_analytics.core.execution_gate import limit_text
from clinical_analytics.core.query_plan import STATISTIC_WORDS, CohortStep

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
LINK_WORDS: frozenset[str] = frozenset({"given", "received", "receiving", "taking", "treated", "use", "uses", "using"})
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
BY_WORDS: frozenset[str] = frozenset({"by", "each", "per"})
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
    (r"(\d)\s+plus\b", r"\1 __ge__ "),
    (r"\b(?:or|and) (?:more|older|over|above|greater|higher)\b(?! than)", " __ge__ "),
    (r"\bat least\b", " __ge__ "),
    (r"\b(?:or|and) (?:less|fewer|younger|under|below|lower)\b(?! than)", " __le__ "),
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
# A word starts with a letter and keeps its digits ("hba1c", "cd4"); a number stands alone.
_TOKEN = r"__[a-z_]+__|[a-z][a-z0-9]*|\d+(?:\.\d+)?"


@dataclass(frozen=True)
class Token:
    text: str
    index: int


def tokenize(question: str) -> list[str]:
    """Lowercase words, numbers and logic placeholders, in order."""
    text = question.lower()
    for pattern, replacement in _PHRASES:
        text = re.sub(pattern, replacement, text)
    return re.findall(_TOKEN, text)


def shown(word: str) -> str:
    return _SHOWN.get(word, word)


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def hold_reason(words: Iterable[str]) -> str:
    """One plain sentence naming the words of the question that the plan leaves out."""
    quoted = [f"'{shown(word)}'" for word in words]
    if len(quoted) == 1:
        return f"I couldn't place the word {quoted[0]}, so it is not in the plan."
    return f"I couldn't place the words {_listed(quoted)}, so they are not in the plan."


def name_tokens(name: str) -> list[str]:
    return re.findall(_TOKEN, name.lower())


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


_AGE_UNITS = {"years", "year", "yrs", "yr", "y", "at", "baseline", "in", "current"}


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
            numbers = [float(value) for value in values if value is not None]
            ranges[column] = (min(numbers), max(numbers))
    return ranges


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
    Age is used only when the words say age ("older than", "aged"), or the number follows a
    subject ("patients over 65") and either no other measure is named before it, or the
    number is outside the observed range of every measure named before it.
    """
    age = age_column(columns)
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
    measures = {
        named
        for named in (_column_for_word(word, columns, alias_index) for word in before)
        if named is not None and named in numeric_columns and named != age
    }
    if not measures:
        return age
    if ranges is None:
        return None
    try:
        number = float(tokens[number_at])
    except ValueError:
        return None
    inside = [column for column in measures if column in ranges and ranges[column][0] <= number <= ranges[column][1]]
    if inside or any(column not in ranges for column in measures):
        return None
    return age


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
    if group_by:
        stored = {str(value) for value in values_for(group_by) if value is not None}
        group_values = _value_words(group_by, values_for(group_by))
        hits = [at for at, word in enumerate(tokens) if not used[at] and _names_value(word, group_values)]
        named = {_value_key(tokens[at], group_by, values_for) for at in hits}
        if stored and named >= stored:
            for at in hits:
                used[at] = True

    # The plan's own columns are always nameable, even if a caller's column list omits one.
    namer = _ColumnNamer([*columns, *sorted(referenced - set(columns))], values_for, alias_index)

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
        # "median ldl over 70": when the only mention of the summarised measure is also the
        # number's owner, the cutoff could equally mean age. Ambiguous: the number stays unused.
        summarised = intent in {"DESCRIBE", "COMPARE_GROUPS"} and owner == metric and metric_mentions <= 1
        if owner is not None and not summarised and _has_comparison(filters, steps, owner, op, float(word)):
            used[at] = True
            used[direction_at] = True
            # "aged 65 or older": the age word is part of the bound comparison.
            lead = min(at, direction_at) - 1
            if lead >= 0 and tokens[lead] in {"aged", "age"} and owner == age_column(columns):
                used[lead] = True

    # Fixed vocabulary and intent words.
    counting = intent == "COUNT" or share_planned(plan)
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
        elif any(
            word in INTENT_WORDS[name] for name in ("COMPARE_GROUPS", "CORRELATIONS", "FIND_PREDICTORS", "SURVIVAL")
        ):
            used[at] = any(word in INTENT_WORDS[name] and intent == name for name in INTENT_WORDS)
        else:
            used[at] = _logic_used(tokens, at, plan, filters, steps, values_for, namer, context)

    return [shown(word) for at, word in enumerate(tokens) if not used[at]]


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


def _names_value(word: str, values: set[str]) -> bool:
    return any(stem_match(word, value) for value in values)


def _value_key(word: str, column: str, values_for: Callable[[str], list[Any]]) -> str:
    """The stored value a word names (so "women" and "females" count as one value)."""
    for value in values_for(column):
        if value is not None and _names_value(word, _value_words(column, [value])):
            return str(value)
    return word


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
        if intent == "COMPARE_GROUPS":
            return getattr(plan, "group_by", None) in after
        if intent == "CORRELATIONS":
            variables = {getattr(plan, "metric", None), getattr(plan, "group_by", None)}
            variables |= set(getattr(context, "predictor_variables", None) or [])
            return bool(after & variables)
        return _has_range(filters, steps)
    if word in RATE_WORDS:
        return share_planned(plan)
    if word in BY_WORDS:
        return getattr(plan, "group_by", None) in namer.neighbour(tokens, at, 1)
    if word in CUTOFF_WORDS:
        return any(step.op == "ge_percent_rank" for step in steps)
    if word in DIRECTION_OPS:
        return DIRECTION_OPS[word] == "gt" and any(step.op == "gt_median" for step in steps)
    return False


def share_planned(plan: Any) -> bool:
    """The plan computes a share: a proportion, or a rate within each group.

    A rate with no group is not computed (execution refuses it), so its words stay unused.
    """
    if getattr(plan, "proportion_column", None):
        return True
    return bool(getattr(plan, "rate_column", None) and getattr(plan, "group_by", None))


def statistic_planned(word: str, plan: Any) -> bool:
    """A statistic word is used only when the plan computes that statistic of a measure.

    DESCRIBE computes the plan's own statistic; COMPARE_GROUPS compares means. "median age
    by sex" on a plan that computes means leaves "median" unused, so the plan is held.
    """
    asked = STATISTIC_WORDS.get(word)
    if asked is None or not getattr(plan, "metric", None):
        return False
    intent = getattr(plan, "intent", None)
    if intent == "DESCRIBE":
        return asked == getattr(plan, "statistic", None)
    if intent == "COMPARE_GROUPS":
        return asked == "mean"
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
    return f"The plan uses {_listed(elements)}, which your question doesn't ask for."


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
            labels = [str(value) for value in self.values_for(column) if isinstance(value, str)]
            labels += list((parse_column_name(column).value_mapping or {}).values())
            for label in labels:
                phrase = name_tokens(label)
                if len(phrase) < 2 or not any(_NUMBER_TOKEN.fullmatch(part) for part in phrase):
                    continue
                size = len(phrase)
                for start in range(len(self.tokens) - size + 1):
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
            negated = any(token in NEGATION_WORDS for token in self.tokens)
            return negated and self.equality_justified(spec.column, values)
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
        return self.column_named(step.column)

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
            ops = {
                DIRECTION_OPS[self.tokens[near]]
                for near in (at - 1, at + 1)
                if 0 <= near < len(self.tokens) and self.tokens[near] in DIRECTION_OPS
            }
            if not (op in ops or (between and op in {"ge", "le", "gt", "lt"})):
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


def _contains(tokens: list[str], phrase: list[str]) -> bool:
    size = len(phrase)
    return any(tokens[start : start + size] == phrase for start in range(len(tokens) - size + 1))

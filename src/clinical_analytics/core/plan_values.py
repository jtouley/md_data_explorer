# SPDX-License-Identifier: MIT
"""Value presence: a limit on a value its column does not hold is held, never run.

A plan can pass word accounting and still name a value that is not in the data: "type 2"
where the column stores "Type 2", or a code the codebook does not define. It would run and
report zero patients, which reads as a finding. This check runs before execution:

- categorical and coded columns: an equality value must be one of the column's stored values
  or a code in the header's codebook;
- measured columns: a compared number must lie inside the column's plausibility range, when
  the data contracts define one (contracts.config.PLAUSIBILITY_RANGES).

Reasons name the plan's value and column only; stored values are never put in a message.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from clinical_analytics.analysis.variable_types import code_labels, short_name
from clinical_analytics.contracts.builder import find_plausibility_range
from clinical_analytics.core.word_accounting import _is_measure

_FILTER_EQUALITY = frozenset({"==", "!=", "IN", "NOT_IN"})
_FILTER_COMPARISON = frozenset({">", ">=", "<", "<="})
_STEP_COMPARISON = frozenset({"gt", "ge", "lt", "le"})


def absent_value_reason(plan: Any, columns: list[str], values_for: Callable[[str], list[Any]]) -> str | None:
    """One sentence naming the first plan value its column cannot hold, or None."""
    known = set(columns)
    for column, value, compared in _plan_values(plan):
        if column not in known or value is None:
            continue
        stored = [item for item in values_for(column) if item is not None]
        if not code_labels(column) and _is_measure(stored):
            outside = _outside_plausible_range(column, value)
            if outside:
                return outside
        elif not compared and not _is_stored(column, value, stored):
            return f"I couldn't find '{_text(value)}' in {short_name(column)}."
    return None


def _plan_values(plan: Any) -> Iterator[tuple[str, Any, bool]]:
    """(column, value, is a comparison) for every value the plan limits on, in plan order."""
    for spec in getattr(plan, "filters", None) or []:
        if spec.operator in _FILTER_EQUALITY or spec.operator in _FILTER_COMPARISON:
            for value in spec.value if isinstance(spec.value, list) else [spec.value]:
                yield str(spec.column), value, spec.operator in _FILTER_COMPARISON
    for step in getattr(plan, "cohort_steps", None) or []:
        if step.op == "eq" or step.op in _STEP_COMPARISON:
            yield str(step.column), step.value, step.op in _STEP_COMPARISON
    for column_attr, value_attr in (("proportion_column", "proportion_value"), ("rate_column", "rate_value")):
        column = getattr(plan, column_attr, None)
        if column:
            yield str(column), getattr(plan, value_attr, None), False


def _is_stored(column: str, value: Any, stored: list[Any]) -> bool:
    """The value as the query will compare it: exact text, or the same number ("1", 1, 1.0)."""
    wanted = _key(value)
    return any(_key(item) == wanted for item in stored) or wanted in {_key(code) for code in code_labels(column)}


def _key(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _outside_plausible_range(column: str, value: Any) -> str | None:
    bounds = find_plausibility_range(column)
    if bounds is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"I couldn't find '{_text(value)}' in {short_name(column)}."
    if bounds.minimum <= number <= bounds.maximum:
        return None
    return (
        f"{_text(value)} is outside the plausible range for {short_name(column)} "
        f"({bounds.minimum:g} to {bounds.maximum:g} {bounds.unit})."
    )

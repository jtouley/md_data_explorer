# SPDX-License-Identifier: MIT
"""Per-patient logic on visit rows, compiled in code: quantifiers and per-patient summaries.

On a table of one row per visit, "patients who never had X", "patients under 7 on every
draw" and "the average number of visits per patient" are not row filters or listings. Each one
is an aggregate over a per-patient subquery:

- **Quantifiers** (``quantifier``, ``rolled_up``): "ever" is ``BOOL_OR`` of the condition per
  patient, "never" is ``NOT BOOL_OR``, "always" / "on every visit" is ``BOOL_AND``. A condition
  written as its own negation ("never had X" written as ``x = 'no'``, "never above 200" written as
  ``v <= 200``) becomes the equivalent ``BOOL_AND`` of it. When the wording or the sign is unclear,
  the question is asked back, never guessed.
- **Per-patient summaries** (``PER_PATIENT_KINDS``): a value per patient (the number of their
  rows, their first or last value, the days from their first to their last visit) or per pair of
  consecutive visits (the days between them), then the mean, median, minimum or maximum of it.
  The answer is one number (or one per group), never a patient's row.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Final, Literal, TypeAlias

from sqlglot import exp

from clinical_analytics.model_query.context import ColumnInfo, DatasetContext

Quantifier: TypeAlias = Literal["ever", "never", "always"]
PerPatient: TypeAlias = Literal["count", "first", "last", "span_days", "gap_days"]
PER_PATIENT_KINDS: Final[tuple[str, ...]] = ("count", "first", "last", "span_days", "gap_days")

_OCCASION = (
    r"(?:visits?|draws?|tests?|sessions?|measurements?|readings?|checks?|occasions?|admissions?|"
    r"encounters?|cycles?|samples?|appointments?|times?|records?|results?|assessments?|exams?|episodes?|"
    r"doses?|infusions?|courses?|treatments?|stays?|attendances?|check-?ups?|follow-?ups?|reviews?|rounds?)"
)
_NEVER = re.compile(
    rf"\bnever\b|\bnot once\b|\bnot (?:at|on|in) any\b|\b(?:on|at|in|during) no {_OCCASION}\b|"
    rf"\bnone of (?:their|the|his|her|its) {_OCCASION}\b",
    re.I,
)
_ALWAYS = re.compile(
    r"\balways\b|\b(?:every single|every|each|all)(?: of)?(?: (?:their|the|his|her|its))?\s+(?:\w+\s+)?"
    rf"{_OCCASION}\b|\b(?:each|every) time\b|\bconsistently\b|\bthroughout\b|\bwithout exception\b",
    re.I,
)
_EVER = re.compile(rf"\bever\b|\bat least once\b|\b(?:on|at|in) any(?: of (?:their|the))? {_OCCASION}\b", re.I)
_ABOVE = re.compile(r"\b(?:above|over|more than|greater than|higher than|at least|exceed\w*|>=?)\b|>", re.I)
_BELOW = re.compile(r"\b(?:below|under|less than|lower than|fewer than|at most|<=?)\b|<", re.I)
_PER_PATIENT_WORDS = re.compile(
    r"\bper (?:patient|person|participant|subject|pt)\b|\beach (?:patient|person|participant|subject)'s\b|"
    r"\bfor (?:the same|each) (?:patient|person|participant|subject)\b|\bbetween consecutive\b|"
    r"\b(?:patient|person)'s (?:first|last|latest|baseline)\b",
    re.I,
)


class QuantifierError(ValueError):
    """The question's ever/never/always reading cannot be told apart. The message is for the user."""


def quantifier(question: str, row_nouns: frozenset[str] = frozenset()) -> Quantifier | None:
    """ever | never | always from the question's own words, or None. ``row_nouns``: the table's
    own word for a row ("scan"), an occasion like visit.

    Raises:
        QuantifierError: it says both "never" and "always" (or "every time").
    """
    text = question
    for noun in sorted(row_nouns, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(noun)}\b", "visit", text, flags=re.IGNORECASE)
    never, always = bool(_NEVER.search(text)), bool(_ALWAYS.search(text))
    if never and always:
        raise QuantifierError(
            "The question says both 'never' and 'every time', so it is not clear which visits must meet the "
            "condition. Ask again with one of them."
        )
    if never:
        return "never"
    if always:
        return "always"
    return "ever" if _EVER.search(text) else None


def asks_per_patient(question: str) -> bool:
    """Whether the words ask for a value per patient ("per patient", "between consecutive")."""
    return bool(_PER_PATIENT_WORDS.search(question))


def _direction(question: str) -> Literal["above", "below"] | None:
    above, below = bool(_ABOVE.search(question)), bool(_BELOW.search(question))
    if above == below:
        return None
    return "above" if above else "below"


def _term_direction(term: exp.Expr) -> Literal["above", "below"] | None:
    if isinstance(term, exp.GT | exp.GTE):
        return "above" if isinstance(term.expression, exp.Literal) else "below"
    if isinstance(term, exp.LT | exp.LTE):
        return "below" if isinstance(term.expression, exp.Literal) else "above"
    return None


def rollable(term: exp.Expr, context: DatasetContext, flag: Callable[[ColumnInfo], bool]) -> bool:
    """Whether a condition is about what happened at a visit (a yes/no flag or a measured number)
    rather than who the patient is: only those are rolled up over the patient's visits."""
    names = {c.name for c in term.find_all(exp.Column)}
    columns = [c for n in names if (c := context.column(n)) is not None]
    if not columns:
        return False
    for column in columns:
        parts = set(re.findall(r"[a-z]+", column.name.casefold()))
        if parts & {"age", "sex", "gender", "race", "ethnicity", "birth", "dob"}:
            return False
    return all(flag(c) or c.numeric for c in columns)


def rolled_up(term: exp.Expr, how: Quantifier, question: str, no_value_of: Callable[[exp.Expr], bool]) -> exp.Expr:
    """The per-patient aggregate of one visit condition.

    ``no_value_of(term)``: True when the term tests a yes/no column's "no" value.

    Raises:
        QuantifierError: "never" with a number condition and no above/below word to tell the sign.
    """
    if how == "always":
        return exp.func("BOOL_AND", term.copy())
    if how == "ever":
        return exp.func("BOOL_OR", term.copy())
    if no_value_of(term):
        return exp.func("BOOL_AND", term.copy())  # "never had X" written as X = 'no'
    direction = _term_direction(term)
    if direction is not None:
        said = _direction(question)
        if said is None:
            raise QuantifierError(
                "The question says 'never' about a number without saying above or below, so the condition "
                "could be read two ways. Ask again saying, for example, 'never above 200'."
            )
        if said != direction:
            return exp.func("BOOL_AND", term.copy())  # "never above 200" written as v <= 200
    return exp.Not(this=exp.func("BOOL_OR", term.copy()))


def order_column(context: DatasetContext, preferred: str | None = None) -> ColumnInfo | None:
    """The column that orders a patient's visits: the one named, else the only date column, else
    the only within-patient sequence number; None when there is none or more than one."""
    if preferred:
        named = context.column(preferred)
        if named is not None:
            return named
    dates = [c for c in context.columns if c.logical_type in ("date", "timestamp", "datetime")]
    if len(dates) == 1:
        return dates[0]
    sequences = [c for c in context.columns if c.is_sequence]
    if not dates and len(sequences) == 1:
        return sequences[0]
    return None


def date_columns(context: DatasetContext) -> list[ColumnInfo]:
    return [c for c in context.columns if c.logical_type in ("date", "timestamp", "datetime")]

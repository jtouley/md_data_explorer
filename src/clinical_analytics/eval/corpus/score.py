# SPDX-License-Identifier: MIT
"""Grade the answer the app presents (claim and table) against an independent oracle.

Reuses the golden kit's presented-answer reading (``eval.headless.presented_answer``), and adds
what that kit does not check: that every oracle cell of 1-10 patients is hidden (LEAK), the
odds-ratio and p-value read-outs, and the RIGHT / WRONG / HELD / ERROR grades.

- RIGHT: every shown value matches (counts exactly; statistics within 1e-4 relative, or half a
  unit of the 4 decimals the table is rounded to), labels and unit match, small cells hidden.
- WRONG: a shown value differs, a row is missing or unexpected, the unit or grain is wrong, a
  question that must be held is answered, or a LEAK (also flagged).
- HELD: held or refused (reason recorded), or every value withheld.
- ERROR: an exception, a timeout, or a silent fallback of the language-model tier; also a
  question whose oracle could not compute a trustworthy answer (``OracleAnswer.oracle_error``).

A statistic that does not exist on the data (an odds ratio with a zero cell, a mean over a group
with no values) is ``OracleAnswer.undefined`` or a ``None`` group value: showing a number for it
is WRONG, withholding it is HELD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from clinical_analytics.analysis.grouping import FOLD_LABELS, HIDDEN_GROUPS_LABEL
from clinical_analytics.analysis.variable_types import display_level
from clinical_analytics.api.services.query_service import UNEXPECTED_FAILURE_MESSAGE
from clinical_analytics.eval.headless import HIDDEN, TOTAL_COUNT, presented_answer

Grade: TypeAlias = Literal["RIGHT", "WRONG", "HELD", "ERROR"]
Status: TypeAlias = Literal["completed", "failed", "awaiting_confirmation", "timeout", "error"]
# ("female",) is a group's patients; ("female", "events") / ("female", "non_events") the
# patients with and without the outcome in it; ("total",) the whole answer's patients.
CellKey: TypeAlias = tuple[str, ...]

SMALL_MIN, SMALL_MAX = 1, 10  # a count of 1-10 patients must never be shown
_REL_TOL = 1e-4
_ROUNDING_TOL = 5e-5  # half a unit of the 4th decimal the presented table keeps
_COUNT_COLUMNS = {"count", "n"}
_VALUE_COLUMN = {"count_by": "count", "rates": "rate", "mean_by": "mean", "median_by": "median"}


class OracleError(ValueError):
    """The data does not support a trustworthy oracle answer (e.g. a non-numeric metric value)."""


@dataclass(frozen=True)
class OracleAnswer:
    """The true answer: ``value`` (int, float, {group: value or None}, {"or","ci_low","ci_high"},
    {"pearson","yates"} p-values, or None for a question that must be held), its unit, and
    the patients behind each cell. ``undefined`` says why the statistic does not exist on this
    data; ``oracle_error`` why no trustworthy answer could be computed (graded ERROR)."""

    value: Any
    unit: str
    cell_patients: dict[CellKey, int] = field(default_factory=dict)
    undefined: str | None = None
    oracle_error: str | None = None


@dataclass(frozen=True)
class ApiOutcome:
    """What the app returned for one question."""

    status: Status
    presentation: dict[str, Any] | None = None
    failure_reason: str | None = None
    error: str | None = None
    silent_fallback: str | None = None


@dataclass(frozen=True)
class Scored:
    grade: Grade
    leak: bool
    reason: str


def is_small(count: int | None) -> bool:
    return count is not None and SMALL_MIN <= count <= SMALL_MAX


def _close(expected: float | int, shown: Any) -> bool:
    if isinstance(shown, bool) or not isinstance(shown, int | float):
        return False
    if isinstance(expected, int) and not isinstance(expected, bool):
        return isinstance(shown, int) and shown == expected
    return abs(float(expected) - float(shown)) <= max(_REL_TOL * abs(float(expected)), _ROUNDING_TOL)


def _label(column: str | None, key: str) -> str:
    return key if column is None else str(display_level(column, key))


def _small_group(oracle: OracleAnswer, key: str, kind: str) -> bool:
    cells = oracle.cell_patients
    if is_small(cells.get((key,))):
        return True
    return kind == "rates" and any(is_small(cells.get((key, part))) for part in ("events", "non_events"))


def find_leaks(question: dict[str, Any], oracle: OracleAnswer, presentation: dict[str, Any] | None) -> list[str]:
    """Every shown count of 1-10 patients, label of a group of 1-10 patients, or rate whose
    events or non-events are 1-10 (rate and n give both counts)."""
    if not presentation:
        return []
    leaks: list[str] = []
    match = TOTAL_COUNT.match(presentation.get("claim") or "")
    if match and match.group(1).isdigit() and is_small(int(match.group(1))):
        leaks.append(f"total count {match.group(1)} shown")
    table = presentation.get("table") or {}
    columns = list(table.get("columns") or [])
    if not columns:
        return leaks
    group_by = question.get("group_by")
    by_label = {_label(group_by, key[0]): key[0] for key in oracle.cell_patients if len(key) == 1}
    for row in table.get("rows") or []:
        label = str(row.get(columns[0]))
        for column in columns[1:]:
            value = row.get(column)
            countlike = column in _COUNT_COLUMNS or (question.get("kind") == "crosstab")
            if countlike and isinstance(value, int) and not isinstance(value, bool) and is_small(value):
                leaks.append(f"{label}: {column} {value} shown")
        key = by_label.get(label)
        if key is None or label in FOLD_LABELS:
            continue
        if is_small(oracle.cell_patients.get((key,))):
            leaks.append(f"label of a group of {oracle.cell_patients[(key,)]} patients shown: {label}")
        elif "rate" in columns and isinstance(row.get("rate"), int | float):
            small = [p for p in ("events", "non_events") if is_small(oracle.cell_patients.get((key, p)))]
            if small:
                leaks.append(f"{label}: rate shown with {small[0]} of 1-10 patients")
    return leaks


def _grade_grouped(question: dict[str, Any], oracle: OracleAnswer, shown: Any) -> tuple[Grade, str]:
    if not isinstance(shown, dict):
        return "WRONG", f"expected a table by {question.get('group_by')}, got {_kind_of(shown)}"
    kind, group_by = question["kind"], question.get("group_by")
    expected = {_label(group_by, key): (key, value) for key, value in oracle.value.items()}
    matched = 0
    for label, (key, value) in expected.items():
        if label not in shown:
            if value is None or _small_group(oracle, key, kind):
                continue  # no value to show, or folded into "Other"
            if shown.get(HIDDEN_GROUPS_LABEL) == HIDDEN:
                continue  # folded with the small groups it would give away, under a label claiming no size
            return "WRONG", f"row '{label}' missing"
        if shown[label] == HIDDEN:
            continue  # hidden: required for a small cell, allowed as a secondary suppression
        if value is None:
            if shown[label] is None:
                continue
            return "WRONG", f"'{label}': shown {shown[label]}, but no patient in it has a value"
        if not _close(value, shown[label]):
            return "WRONG", f"'{label}': shown {shown[label]}, expected {value}"
        matched += 1
    extra = [
        label for label in shown if label not in expected and not (label in FOLD_LABELS and shown[label] == HIDDEN)
    ]
    if extra:
        return "WRONG", f"unexpected rows {sorted(extra)}"
    if matched == 0:
        return "HELD", "every value withheld"
    return "RIGHT", f"{matched} value(s) match"


def _grade_odds_ratio(oracle: OracleAnswer, presentation: dict[str, Any]) -> tuple[Grade, str]:
    table = presentation.get("table") or {}
    columns, rows = list(table.get("columns") or []), list(table.get("rows") or [])
    claim = presentation.get("claim") or ""
    if oracle.undefined is not None:
        if "odds ratio" in columns and rows:
            return "WRONG", f"odds ratio shown, but {oracle.undefined}"
        return "HELD", f"no odds ratio shown ({oracle.undefined}): {claim[:160]}"
    if "odds ratio" not in columns:
        return "HELD" if not rows else "WRONG", f"no odds ratio shown: {claim[:160]}"
    if len(rows) != 1:
        return "WRONG", f"{len(rows)} model terms shown, expected one exposure"
    row = rows[0]
    pairs = (("or", "odds ratio"), ("ci_low", "95% CI low"), ("ci_high", "95% CI high"))
    for key, column in pairs:
        if row.get(column) == HIDDEN or not _close(oracle.value[key], row.get(column)):
            return "WRONG", f"{column}: shown {row.get(column)}, expected {oracle.value[key]:.4f}"
    return "RIGHT", "odds ratio and CI match"


_P = re.compile(r"\bp(<|=)(\d*\.\d+)")


def _grade_p_value(question: dict[str, Any], oracle: OracleAnswer, claim: str) -> tuple[Grade, str]:
    test = question.get("test")
    if test and test not in claim.lower():
        return "WRONG", f"claim does not name a {test} test: {claim[:160]}"
    match = _P.search(claim)
    if not match:
        return "HELD", f"no p-value shown: {claim[:160]}"
    if oracle.undefined is not None:
        return "WRONG", f"p{match.group(1)}{match.group(2)} shown, but {oracle.undefined}"
    op, shown = match.group(1), float(match.group(2))
    decimals = len(match.group(2).split(".")[1])
    for name, p in oracle.value.items():
        if (op == "<" and p < shown) or (op == "=" and round(p, decimals) == shown):
            return "RIGHT", f"p matches the {name} test"
    return "WRONG", f"p{op}{shown} shown, expected {oracle.value}"


def _kind_of(value: Any) -> str:
    return "nothing" if value is None else type(value).__name__


def score(question: dict[str, Any], oracle: OracleAnswer, outcome: ApiOutcome) -> Scored:
    """Grade one presented answer."""
    expect = question.get("expect", "answer")
    if outcome.status in ("timeout", "error"):
        return Scored("ERROR", False, outcome.error or outcome.status)
    if oracle.oracle_error is not None:
        return Scored("ERROR", False, f"no trustworthy oracle answer: {oracle.oracle_error}")
    if outcome.status == "failed" and UNEXPECTED_FAILURE_MESSAGE in (outcome.error or ""):
        return Scored("ERROR", False, f"exception: {outcome.error}")
    if outcome.status in ("failed", "awaiting_confirmation"):
        reason = outcome.failure_reason or outcome.error or outcome.status
        return Scored("RIGHT" if expect == "held" else "HELD", False, f"{outcome.status}: {reason}")
    presentation = outcome.presentation or {}
    leaks = find_leaks(question, oracle, presentation)
    grade: Grade
    if expect == "held":
        grade, reason = "WRONG", "answered a question that must be held"
    else:
        grade, reason = _grade_value(question, oracle, presentation)
    if leaks:
        return Scored("WRONG", True, f"LEAK {'; '.join(leaks)}; {reason}")
    if outcome.silent_fallback:
        return Scored("ERROR", False, f"silent fallback ({outcome.silent_fallback}); value grade {grade}: {reason}")
    return Scored(grade, False, reason)


def _grade_value(question: dict[str, Any], oracle: OracleAnswer, presentation: dict[str, Any]) -> tuple[Grade, str]:
    kind = question["kind"]
    claim = presentation.get("claim") or ""
    if kind == "odds_ratio":
        return _grade_odds_ratio(oracle, presentation)
    if kind == "p_value":
        return _grade_p_value(question, oracle, claim)
    shown = presented_answer(presentation)
    if kind == "count":
        match = TOTAL_COUNT.match(claim)
        if match and match.group(2) and match.group(2) != oracle.unit:
            return "WRONG", f"grain: counted {match.group(2)}, expected {oracle.unit}"
        if shown == HIDDEN:
            return ("RIGHT", "small total hidden") if is_small(oracle.value) else ("HELD", "total withheld")
        if isinstance(shown, int) and shown == oracle.value:
            return "RIGHT", "count matches"
        return "WRONG", f"shown {_kind_of(shown) if not isinstance(shown, int) else shown}, expected {oracle.value}"
    if kind in ("count_by", "rates", "mean_by", "median_by"):
        columns = list((presentation.get("table") or {}).get("columns") or [])
        wanted = _VALUE_COLUMN[kind]
        if columns and not any(c == wanted or c.startswith(f"{wanted} ") for c in columns[1:]):
            return "WRONG", f"unit: table shows {columns[1:]}, expected a '{wanted}' column"
        return _grade_grouped(question, oracle, shown)
    return "WRONG", f"no grader for kind '{kind}'"

# SPDX-License-Identifier: MIT
"""Grade the answer the app presents (claim and table) against an independent oracle.

Reuses the golden kit's presented-answer reading (``eval.headless.presented_answer``), and adds
the odds-ratio and p-value read-outs and the RIGHT / WRONG / HELD / ERROR grades.

On screen every answer shows its true numbers, small counts included (ADR015), so a count of 1-10
shown on screen is graded on its value like any other. LEAK is about the export boundary only:
the answer as it would leave the machine (``core.export_boundary.export_answer``) must show no
oracle cell of 1-10 patients as a number.

- RIGHT: every shown value matches (counts exactly; statistics within 1e-4 relative, or half a
  unit of the 4 decimals the table is rounded to), labels and unit match.
- WRONG: a shown value differs, a value is withheld, a row is missing or unexpected, the unit or
  grain is wrong, a question that must be held is answered, or a LEAK at the export boundary
  (also flagged).
- HELD: held or refused (reason recorded), or no value shown.
- ERROR: an exception, a timeout, or a silent fallback of the language-model tier; also a
  question whose oracle could not compute a trustworthy answer (``OracleAnswer.oracle_error``).

A statistic that does not exist on the data (an odds ratio with a zero cell, a mean over a group
with no values) is ``OracleAnswer.undefined`` or a ``None`` group value: showing a number for it
is WRONG, withholding it is HELD.

Population (aggregate) datasets, whose rows are already estimates for a place, year or stratum,
add three kinds, graded on the numbers shown whatever the app calls its value column:

- ``value_by``: one value per group ({label: number or None}). With ``top: N`` the oracle holds the
  N highest (``order: asc``: lowest); rows ranked below them may be shown, any other row is WRONG.
- ``values``: named numbers ({"estimate": x, ...}) that must all be shown, in the claim or a table
  of at most two rows. A proportion may be shown as a percent.
- ``ci_by``: per group {"estimate", "ci_low", "ci_high"}; each row must show all three.

A question whose rows are not patients (``grain: aggregate`` in the question file) has no
patient cells to check at the export boundary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from clinical_analytics.analysis.variable_types import display_level
from clinical_analytics.api.services.query_service import UNEXPECTED_FAILURE_MESSAGE
from clinical_analytics.core.export_boundary import SMALL_MAX, SMALL_MIN, export_answer
from clinical_analytics.eval.headless import HIDDEN, TOTAL_COUNT, count_grain, presented_answer

Grade: TypeAlias = Literal["RIGHT", "WRONG", "HELD", "ERROR"]
Status: TypeAlias = Literal["completed", "failed", "awaiting_confirmation", "timeout", "error"]
# ("female",) is a group's patients; ("female", "events") / ("female", "non_events") the
# patients with and without the outcome in it; ("total",) the whole answer's patients.
CellKey: TypeAlias = tuple[str, ...]

_REL_TOL = 1e-4
_ROUNDING_TOL = 5e-5  # half a unit of the 4th decimal the presented table keeps
_VALUE_COLUMN = {"count_by": "count", "rates": "rate", "mean_by": "mean", "median_by": "median"}
# A number as a reader sees it: "1,234", "-0.5", "12". Not part of a word or a date's digits.
_NUMBER = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])-?\d+(?:\.\d+)?")
AGGREGATE = "aggregate"
POPULATION_KINDS = frozenset({"value_by", "values", "ci_by"})


class OracleError(ValueError):
    """The data does not support a trustworthy oracle answer (e.g. a non-numeric metric value)."""


@dataclass(frozen=True)
class OracleAnswer:
    """The true answer: ``value`` (int, float, {group: value or None}, {"or","ci_low","ci_high"},
    {"pearson","yates"} p-values, or None for a question that must be held), its unit, and
    the patients behind each cell (for the export-boundary LEAK check). ``undefined`` says why the
    statistic does not exist on this data; ``oracle_error`` why no trustworthy answer could be
    computed (graded ERROR)."""

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


def find_export_leaks(question: dict[str, Any], oracle: OracleAnswer, presentation: dict[str, Any] | None) -> list[str]:
    """Counts of 1-10 patients the answer would still show as numbers once it leaves the machine:
    every whole number from 1 to 10 left in the exported table, and every oracle cell of 1-10
    patients whose number is left in the exported claim. On screen they are shown (ADR015)."""
    if not presentation or question.get("grain") == AGGREGATE:
        return []
    exported = export_answer(presentation)
    leaks: list[str] = []
    table = exported["table"] or {"columns": [], "rows": []}
    for row in table["rows"]:
        for column, value in row.items():
            if isinstance(value, int) and not isinstance(value, bool) and is_small(value):
                leaks.append(f"exported table shows {column} {value}")
    claim = exported["claim"] or ""
    shown = {int(n) for n in re.findall(r"(?<![\w.,])\d+(?![\w%]|[.,]\d)", claim)}
    small = sorted({n for n in oracle.cell_patients.values() if is_small(n) and n in shown})
    leaks += [f"exported claim shows {n}" for n in small]
    return leaks


def _missing_rows(missing: list[str], shown: Mapping[str, object]) -> str:
    """Every expected row the table lacks, in sorted order, with what it shows instead."""
    named = f"row '{missing[0]}'" if len(missing) == 1 else f"rows {missing}"
    return f"{named} missing (shown {sorted(shown)[:6]})"


def _grade_grouped(question: dict[str, Any], oracle: OracleAnswer, shown: Any) -> tuple[Grade, str]:
    if not isinstance(shown, dict):
        return "WRONG", f"expected a table by {question.get('group_by')}, got {_kind_of(shown)}"
    group_by = question.get("group_by")
    expected = {_label(group_by, key): value for key, value in oracle.value.items()}
    # Sorted, and every missing row named: an oracle's group_by order changes from run to run, so
    # "the first missing row" would name a different group each run for the same answer. Only a
    # row with no value to show may be absent; a small group must be shown (ADR015).
    missing = sorted(label for label, value in expected.items() if label not in shown and value is not None)
    if missing:
        return "WRONG", _missing_rows(missing, shown)
    matched = 0
    for label, value in sorted(expected.items()):
        if label not in shown:
            continue
        if shown[label] == HIDDEN:
            return "WRONG", f"'{label}': value withheld on screen, expected {value}"
        if value is None:
            if shown[label] is None:
                continue
            return "WRONG", f"'{label}': shown {shown[label]}, but no patient in it has a value"
        if not _close(value, shown[label]):
            return "WRONG", f"'{label}': shown {shown[label]}, expected {value}"
        matched += 1
    extra = [label for label in shown if label not in expected]
    if extra:
        return "WRONG", f"unexpected rows {sorted(extra)}"
    if matched == 0:
        return "HELD", "no value shown"
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


def _numbers(value: Any) -> list[float]:
    """Every number a reader sees in one cell or one sentence."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int | float):
        return [float(value)]
    return [float(found.replace(",", "")) for found in _NUMBER.findall(str(value))]


def _shown_rows(presentation: dict[str, Any]) -> dict[str, list[float]]:
    """{row label: the numbers in its other cells}, for a table whose first column labels rows."""
    table = presentation.get("table") or {}
    columns = list(table.get("columns") or [])
    if not columns:
        return {}
    return {
        str(row.get(columns[0])): [n for column in columns[1:] for n in _numbers(row.get(column))]
        for row in table.get("rows") or []
    }


def _matches(expected: float, shown: list[float], percent_ok: bool) -> bool:
    candidates = (expected, expected * 100) if percent_ok else (expected,)
    return any(_close(float(c), n) for c in candidates for n in shown)


def _ranked_below(question: dict[str, Any], expected: dict[str, Any], shown: list[float]) -> bool:
    """A row ranked below every expected row of a top-N question: showing it is allowed."""
    tops = [float(v["estimate"] if isinstance(v, dict) else v) for v in expected.values() if v is not None]
    if "top" not in question or not shown or not tops:
        return False
    return shown[0] <= min(tops) if question.get("order", "desc") == "desc" else shown[0] >= max(tops)


def _grade_values(oracle: OracleAnswer, presentation: dict[str, Any]) -> tuple[Grade, str]:
    rows = _shown_rows(presentation)
    if len(rows) > 2:
        return "WRONG", f"{len(rows)} rows shown, expected one answer"
    shown = _numbers(presentation.get("claim")) + [n for numbers in rows.values() for n in numbers]
    if not shown:
        return "HELD", f"no number shown: {str(presentation.get('claim'))[:160]}"
    percent_ok = oracle.unit == "proportion"
    missing = [name for name, value in oracle.value.items() if not _matches(value, shown, percent_ok)]
    if missing:
        return "WRONG", f"not shown: {missing} (expected {oracle.value}, shown {shown[:6]})"
    return "RIGHT", f"{len(oracle.value)} value(s) match"


def _grade_population(
    question: dict[str, Any], oracle: OracleAnswer, presentation: dict[str, Any]
) -> tuple[Grade, str]:
    """``value_by``, ``ci_by`` and ``values``: the numbers shown against the source rows'."""
    if question["kind"] == "values":
        return _grade_values(oracle, presentation)
    rows = _shown_rows(presentation)
    if not rows:
        return "HELD", f"no table shown: {str(presentation.get('claim'))[:160]}"
    percent_ok = oracle.unit == "proportion"
    group_by = question.get("group_by")
    expected = {_label(group_by, key): value for key, value in oracle.value.items()}
    missing = sorted(label for label, value in expected.items() if label not in rows and value is not None)
    if missing:
        return "WRONG", _missing_rows(missing, rows)
    matched = 0
    for label, value in sorted(expected.items()):
        if label not in rows:
            continue
        shown = rows[label]
        if value is None:
            if shown:
                return "WRONG", f"'{label}': shown {shown[:3]}, but the source has no value for it"
            continue
        parts = value if isinstance(value, dict) else {"value": value}
        absent = [name for name, number in parts.items() if not _matches(number, shown, percent_ok)]
        if absent:
            return "WRONG", f"'{label}': {absent} not shown (shown {shown[:4]}, expected {parts})"
        matched += 1
    extra = sorted(
        label for label in rows if label not in expected and not _ranked_below(question, expected, rows[label])
    )
    if extra:
        return "WRONG", f"unexpected rows {extra[:6]}"
    if matched == 0:
        return "HELD", "every value withheld"
    return "RIGHT", f"{matched} value(s) match"


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
    leaks = find_export_leaks(question, oracle, presentation)
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
    if kind in POPULATION_KINDS:
        return _grade_population(question, oracle, presentation)
    shown = presented_answer(presentation)
    if kind == "count":
        match = TOTAL_COUNT.match(claim)
        said = count_grain(match.group(2)) if match else None
        # "rows" claims no grain: a dataset with no patient ID says it counts rows, never
        # patients. The number decides: equal to the patient count means rows are patients here.
        # A claimed grain that differs ("patients" for visits, "visits" for patients) is wrong.
        if said and said != oracle.unit and not (said == "rows" and oracle.unit == "patients"):
            return "WRONG", f"grain: counted {said}, expected {oracle.unit}"
        if shown == HIDDEN:
            return "WRONG", f"total withheld on screen, expected {oracle.value}"
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

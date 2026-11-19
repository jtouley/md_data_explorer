# SPDX-License-Identifier: MIT
"""Abstract scenarios on top of the acceptance questions: which [blanks] a fellow gets filled.

A question file may carry a ``scenario:`` block (an abstract with [blanks], from the ID fellows'
research guide) and, per question, ``blanks: [<id>, ...]`` (the blanks its answer fills) and
``needs: aggregate_mode`` (the answer needs a public join this app cannot do yet). Grading is
unchanged: every asked question is graded by ``score.score`` as before. This module only

- splits off the questions that need a capability the app lacks (not asked, listed as skipped);
- rolls the graded questions up into one result per blank: a blank is filled right only when
  every question that fills it is RIGHT, and is LEAK or WRONG when any of them is.

Question files without a ``scenario:`` block (covid_ms) are unaffected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

Capability: TypeAlias = Literal["aggregate_mode"]
CAPABILITIES: Final[frozenset[str]] = frozenset({"aggregate_mode"})
BlankStatus: TypeAlias = Literal["RIGHT", "WRONG", "LEAK", "ERROR", "HELD", "WAITS"]
# Worst first: one question's outcome decides the blank in this order.
_ORDER: Final[tuple[BlankStatus, ...]] = ("LEAK", "WRONG", "ERROR", "HELD")


class ScenarioError(ValueError):
    """A question file's scenario block or its blank references are inconsistent."""


class Blank(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    text: str


class Scenario(BaseModel):
    """The abstract a dataset's questions fill: from the guide, with its [blanks] listed."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_study: int
    title: str
    abstract: str
    blanks: list[Blank] = Field(min_length=1)


class Graded(Protocol):
    """What the roll-up reads from a graded question (run.QuestionResult)."""

    id: str
    grade: str
    leak: bool


@dataclass(frozen=True)
class SkippedQuestion:
    id: str
    query: str
    needs: str


@dataclass(frozen=True)
class BlankResult:
    id: str
    text: str
    status: BlankStatus
    questions: tuple[str, ...]
    note: str


def needs(question: dict[str, Any]) -> str | None:
    """The capability a question waits on, or None.

    Raises:
        ScenarioError: ``needs`` names a capability this harness does not know.
    """
    value = question.get("needs")
    if value is None:
        return None
    if value not in CAPABILITIES:
        raise ScenarioError(f"{question['id']}: needs is {value!r}; use one of {sorted(CAPABILITIES)}")
    return str(value)


def parse_scenario(raw: dict[str, Any], questions: Sequence[dict[str, Any]]) -> Scenario | None:
    """The file's scenario, checked against its questions; None when the file has none.

    Raises:
        ScenarioError: the block does not validate, a question fills an unknown blank, a blank
            has no question, or a question has ``blanks`` without a scenario.
    """
    block = raw.get("scenario")
    with_blanks = [q["id"] for q in questions if q.get("blanks")]
    if block is None:
        if with_blanks:
            raise ScenarioError(f"{with_blanks} fill blanks, but the file has no scenario block")
        return None
    try:
        scenario = Scenario.model_validate(block)
    except ValueError as error:  # pydantic ValidationError is a ValueError
        raise ScenarioError(f"scenario block: {error}") from error
    known = {blank.id for blank in scenario.blanks}
    filled: set[str] = set()
    for question in questions:
        needs(question)
        ids = question.get("blanks") or []
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ScenarioError(f"{question['id']}: blanks must be a list of blank ids")
        if unknown := sorted(set(ids) - known):
            raise ScenarioError(f"{question['id']} fills unknown blank(s) {unknown}; known: {sorted(known)}")
        if ids and question.get("expect", "answer") != "answer":
            raise ScenarioError(f"{question['id']}: a question that must be held fills no blank")
        filled |= set(ids)
    if empty := sorted(known - filled):
        raise ScenarioError(f"blank(s) {empty} have no question that fills them")
    return scenario


def split(questions: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[SkippedQuestion]]:
    """(questions to ask, questions skipped because they need a missing capability).

    Raises:
        ScenarioError: a question follows up a skipped one (it would be asked without its parent).
    """
    asked: list[dict[str, Any]] = []
    skipped: list[SkippedQuestion] = []
    for question in questions:
        capability = needs(question)
        parent = question.get("follow_up_of")
        waits_on = {s.id: s.needs for s in skipped}
        if capability is None and parent in waits_on:
            raise ScenarioError(f"{question['id']} follows up {parent}, which is not asked (needs {waits_on[parent]})")
        if capability is None:
            asked.append(question)
        else:
            skipped.append(SkippedQuestion(question["id"], question["query"], capability))
    return asked, skipped


def _status(graded: Sequence[Graded]) -> BlankStatus:
    grades = {"LEAK" if result.leak else result.grade for result in graded}
    return next((status for status in _ORDER if status in grades), "RIGHT")


def blank_results(
    scenario: Scenario, questions: Sequence[dict[str, Any]], results: Sequence[Graded]
) -> list[BlankResult]:
    """One result per blank, in the abstract's order."""
    by_id = {result.id: result for result in results}
    rolled: list[BlankResult] = []
    for blank in scenario.blanks:
        mine = [q for q in questions if blank.id in (q.get("blanks") or [])]
        ids = tuple(q["id"] for q in mine)
        waiting = sorted({str(q["needs"]) for q in mine if q.get("needs")})
        asked = [q["id"] for q in mine if not q.get("needs")]
        graded = [by_id[i] for i in asked if i in by_id]
        notes = [f"{r.id.split('-', 1)[-1]} {'LEAK' if r.leak else r.grade}" for r in graded]
        if missing := sorted(set(asked) - set(by_id)):
            notes.append(f"not graded: {missing}")
        status: BlankStatus = "ERROR" if missing else _status(graded)
        # A failure on an asked question always shows; otherwise a blank that also needs a
        # missing capability waits on it, whatever its asked questions got.
        if waiting and status not in ("LEAK", "WRONG", "ERROR"):
            status = "WAITS"
        if waiting:
            notes.append(f"needs {', '.join(waiting)}")
        rolled.append(BlankResult(blank.id, blank.text, status, ids, ", ".join(notes)))
    return rolled


def blanks_line(rolled: Sequence[BlankResult]) -> str:
    """The headline: blanks filled right out of those the app can be asked today."""
    askable = [r for r in rolled if r.status != "WAITS"]
    right = sum(r.status == "RIGHT" for r in askable)
    waits = len(rolled) - len(askable)
    unsafe = sum(r.status in ("WRONG", "LEAK") for r in askable)
    return (
        f"Blanks filled right {right} of {len(askable)} askable · filled wrong or leaked {unsafe} · "
        f"waiting on aggregate mode {waits} (of {len(rolled)} blanks)"
    )


def blanks_markdown(scenario: Scenario, rolled: Sequence[BlankResult], skipped: Sequence[SkippedQuestion]) -> str:
    lines = [
        f"#### Case study {scenario.case_study}: {scenario.title}",
        "",
        f"**{blanks_line(rolled)}**",
        "",
        "| blank | result | questions (grade) |",
        "|---|---|---|",
    ]
    for result in rolled:
        text, note = result.text.replace("|", "\\|"), result.note.replace("|", "\\|")
        lines.append(f"| {text} | {result.status} | {note} |")
    if skipped:
        lines += ["", "Not asked (the app cannot do this yet): " + "; ".join(f"{s.id} ({s.needs})" for s in skipped)]
    return "\n".join(lines) + "\n"

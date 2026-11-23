# SPDX-License-Identifier: MIT
"""Value linking before the plan call: which column values the question names.

The question's phrases (one to four words) are matched by plain string similarity (``difflib``)
against every listed category's labels and stored values (never a restricted column's). An exact
match goes to the model as a hint ("'dust mite' is a value of extract"); a near miss
("dust mitt") the plan does not resolve is asked back with the nearest values as options. The
idea follows value-matching steps in open-source text-to-SQL tools; the code is this app's own.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Final

from clinical_analytics.model_query.context import DatasetContext

NEAR: Final = 0.85  # similarity of a near miss (difflib ratio, case-insensitive)
MIN_CHARS: Final = 4
MAX_WORDS: Final = 4
MAX_OPTIONS: Final = 3
# Labels too common as words to say the question names that category.
_PLAIN: Final = frozenset({"other", "unknown", "none", "missing", "yes", "no", "not", "all", "any", "both"})


@dataclass(frozen=True)
class Link:
    """A phrase of the question and the column value it names (``exact``) or may mean."""

    said: str
    label: str
    column: str
    exact: bool


def _phrases(question: str) -> list[str]:
    words = re.findall(r"[\w'+-]+", question.casefold())
    return [" ".join(words[i : i + n]) for n in range(MAX_WORDS, 0, -1) for i in range(len(words) - n + 1)]


def _labels(context: DatasetContext) -> dict[str, list[tuple[str, str]]]:
    """folded label -> [(label as written, column)] for every listed, unrestricted category."""
    found: dict[str, list[tuple[str, str]]] = {}
    for column in context.columns:
        if column.phi or column.kind not in ("category", "flag"):
            continue
        for label in column.choices():
            folded = label.casefold()
            if len(folded) >= 3 and folded not in _PLAIN and not folded.isdigit():
                found.setdefault(folded, []).append((label, column.name))
    return found


def link_values(question: str, context: DatasetContext) -> list[Link]:
    """The values the question names exactly, then near misses of phrases that name none."""
    labels = _labels(context)
    links: list[Link] = []
    covered: list[str] = []
    for phrase in _phrases(question):
        if phrase in labels and not any(phrase in c for c in covered):
            links += [Link(phrase, label, column, True) for label, column in labels[phrase]]
            covered.append(phrase)
    for phrase in _phrases(question):
        if len(phrase) < MIN_CHARS or any(phrase in c or c in phrase for c in covered):
            continue
        near = difflib.get_close_matches(phrase, list(labels), n=MAX_OPTIONS, cutoff=NEAR)
        if near:
            links += [Link(phrase, label, column, False) for folded in near for label, column in labels[folded]]
            covered.append(phrase)
    return links


def hints(links: list[Link], context: DatasetContext) -> list[str]:
    """Hint lines for the prompt: exact values first, then what a near miss may mean."""
    lines = [f"'{x.label}' is a value of {x.column}." for x in links if x.exact]
    lines += [f"'{x.said}' may mean '{x.label}' ({x.column})." for x in links if not x.exact]
    return lines


def unresolved(links: list[Link], used: dict[str, set[str]]) -> list[Link]:
    """Near misses whose phrase the plan does not resolve: no value of a matched column is used.
    ``used``: column -> the labels the plan's conditions use."""
    misses: dict[str, list[Link]] = {}
    for link in links:
        if not link.exact:
            misses.setdefault(link.said, []).append(link)
    out: list[Link] = []
    for _said, options in misses.items():
        if not any(o.label.casefold() in used.get(o.column, set()) for o in options):
            out += options
    return out

# SPDX-License-Identifier: MIT
"""Claim-first presentation of analysis results.

Turns the dicts returned by ``analysis.compute`` (via ``SemanticLayer.format_execution_result``)
into one plain-language claim plus a small statistics table. Pure functions: every number
comes from the analysis payload and nothing is recomputed, so the API, CLI and any UI show
the same answer. A clause whose inputs are missing is left out rather than guessed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any, Final, TypedDict, TypeGuard

import numpy as np

from clinical_analytics.analysis.grouping import (
    HIDDEN_GROUPS_LABEL,
    MAX_GROUP_LEVELS,
    OTHER_LABEL,
    SMALL_CELL_TEST_REFUSAL,
    is_small_level,
    predictor_model_refusal,
    table_statistic_refusal,
    too_many_levels_message,
)
from clinical_analytics.analysis.sample_floors import (
    MIN_COMPARISON_N,
    MIN_CORRELATION_N,
    MIN_GROUP_N,
    MIN_REGRESSION_N,
    MIN_SURVIVAL_N,
)
from clinical_analytics.analysis.variable_types import display_level, header_has_code_labels, short_name
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD


class TablePayload(TypedDict):
    columns: list[str]
    rows: list[dict[str, Any]]


class Presentation(TypedDict):
    claim: str | None
    table: TablePayload | None


def to_json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars, NaN/inf and dates into JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


_TEST_LABELS = {
    "welch_t_test": "Welch t-test",
    "t_test": "t-test",
    "anova": "ANOVA",
    "chi_square": "chi-square",
    "fisher_exact": "Fisher's exact test",
}


def suppress_count(value: Any, threshold: int | None = None) -> Any:
    """Show a patient count from 1 to threshold-1 as '<threshold' (0 and larger counts unchanged)."""
    limit = SMALL_CELL_THRESHOLD if threshold is None else threshold
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        return value
    return f"<{limit}" if 0 < int(value) < limit else int(value)


SECONDARY_SUPPRESSED = "suppressed"


def _limit(threshold: int | None) -> int:
    return SMALL_CELL_THRESHOLD if threshold is None else threshold


def _is_count(value: Any) -> TypeGuard[int]:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _as_count(value: Any) -> int:
    """A patient count as int; anything that is not a count (None, NaN, text) counts as 0."""
    return int(value) if _is_count(value) else 0


def _feasible(total: int, low: int, high: int, rest_low: int, rest_high: int) -> int:
    """How many integers x in [low, high] leave the other cells a sum in [rest_low, rest_high]."""
    return max(0, min(high, total - rest_low) - max(low, total - rest_high) + 1)


def _single_value(values: list[int], floors: list[int], caps: list[int], limit: int, one_small: bool) -> bool:
    """Whether some hidden cell of a line has only one value a reader who knows the method sees.

    The hidden cells sum to sum(values) (the line total minus the visible cells). The reader
    knows each cell is at least its floor (an analysis floor, or 1) and at most its cap: a small
    cell is below the threshold, a complement is below the smallest cell of the line that hid
    it. ``one_small`` says every hidden cell was hidden by this line, so at least one of them is
    small. For each cell the feasible values are two intervals (the cell is small; the cell is
    a complement), counted in closed form: the cost is linear in the number of hidden cells and
    does not depend on the size of the counts.
    """
    total = sum(values)
    caps = [min(cap, total) for cap in caps]
    floor_sum, cap_sum = sum(floors), sum(caps)
    # Room above the threshold of each cell that may be the small one; a complement leaves the
    # small cell among the others, so it needs the least room among them (two smallest kept).
    room = sorted((max(0, cap - (limit - 1)), i) for i, cap in enumerate(caps) if floors[i] < limit)[:2]
    for i, (floor, cap) in enumerate(zip(floors, caps, strict=True)):
        rest_low, rest_high = floor_sum - floor, cap_sum - cap
        found = _feasible(total, floor, min(cap, limit - 1), rest_low, rest_high)
        if one_small:
            least = next((r for r, j in room if j != i), None)
            rest_high = rest_high - least if least is not None else -1
        found += _feasible(total, max(floor, limit), cap, rest_low, rest_high)
        if found < 2:
            return True
    return False


class _Lines:
    """Lines of count cells whose totals are visible, and the cells of them that are hidden.

    Built once: which lines run through each cell, each line's cells by rising count, and each
    line's hidden cells. Hiding a cell updates the lines through it, so a check never scans a
    line again.

    The hidden cells of a line are recoverable as total minus the rest, so a line needs at least
    two hidden cells whose sum is at least the threshold ([100, 1, 1] would reveal 1 and 1), and
    no hidden cell may have only one value a reader could give it ([31: 11, 10, 10] would reveal
    10 and 10; a floor of 10 on n with n + filtered = 11 would reveal n=10). ``complete`` hides
    the smallest visible non-zero cells of a line until that holds or none are left; a tie hides
    every cell of that value, so the order of equal cells reveals nothing.

    A hidden cell is small, a complement of its line (below that line's smallest visible cell)
    or a complement of another line through it. Another line can have hidden the cell only if
    it has a second hidden cell, and then the cell is below that line's smallest visible cell,
    which may be far above this line's.
    """

    def __init__(
        self,
        cells: Mapping[Any, Any],
        lines: Sequence[Sequence[Any]],
        hidden: set[Any],
        limit: int,
        floors: Mapping[Any, int] | None,
    ) -> None:
        self.cells, self.hidden, self.limit, self.floors = cells, hidden, limit, floors or {}
        members = [tuple(line) for line in lines]
        self.through: dict[Any, list[int]] = {}
        for i, line in enumerate(members):
            for key in line:
                self.through.setdefault(key, []).append(i)
        self.rising = [
            sorted((k for k in line if _is_count(cells[k]) and int(cells[k]) > 0), key=lambda k: int(cells[k]))
            for line in members
        ]
        self.first = [0] * len(members)  # where each line's visible cells start in ``rising``
        self.hidden_in: list[list[Any]] = [[k for k in line if k in hidden] for line in members]
        self.stale = [bool(found) for found in self.hidden_in]  # lines to check (again)

    def hide(self, key: Any) -> None:
        if key in self.hidden:
            return
        self.hidden.add(key)
        for i in self.through[key]:
            self.hidden_in[i].append(key)
            self.stale[i] = True

    def smallest_visible(self, i: int) -> int | None:
        """The smallest visible non-zero count of line ``i``, or None when it has none."""
        rising = self.rising[i]
        while self.first[i] < len(rising) and rising[self.first[i]] in self.hidden:
            self.first[i] += 1
        return int(self.cells[rising[self.first[i]]]) if self.first[i] < len(rising) else None

    def exposed(self, i: int) -> bool:
        keys = [k for k in self.hidden_in[i] if _is_count(self.cells[k])]
        values = [int(self.cells[k]) for k in keys]
        if len(values) < 2 or sum(values) < self.limit:
            return True
        unbounded = sum(values)
        smallest = self.smallest_visible(i)
        own = max(self.limit - 1, unbounded if smallest is None else smallest - 1)
        caps, one_small = [], True
        for key in keys:
            cap = own
            for j in self.through[key]:
                if j == i or len(self.hidden_in[j]) < 2:
                    continue  # that line hid nothing besides this cell, so it did not hide it
                one_small = False
                other = self.smallest_visible(j)
                cap = max(cap, unbounded if other is None else other - 1)
            caps.append(cap)
        return _single_value(values, [max(1, self.floors.get(k, 1)) for k in keys], caps, self.limit, one_small)

    def complete(self, i: int) -> None:
        """Hide the smallest visible cells of line ``i`` until its hidden cells are safe."""
        while self.exposed(i):
            smallest = self.smallest_visible(i)
            if smallest is None:
                return
            for key in self.rising[i][self.first[i] :]:
                if int(self.cells[key]) != smallest:
                    break
                self.hide(key)


def _suppress_lines(
    cells: Mapping[Any, Any],
    lines: Sequence[Sequence[Any]],
    threshold: int | None = None,
    hidden: set[Any] | None = None,
    floors: Mapping[Any, int] | None = None,
) -> set[Any]:
    """Cells to hide when each line's total is known (primary + secondary).

    Primary: counts from 1 to threshold-1, any value that is not a count (text, a float: it
    cannot be checked, so it is not shown) and any cells already in ``hidden``. A missing value
    (None) has nothing to hide. Secondary: complete the lines until none has a hidden cell that
    its total and the rest reveal. Only a line that gained a hidden cell is checked again: more
    hidden cells in the lines crossing it only widen what its own hidden cells could be.
    """
    limit = _limit(threshold)
    hidden = set(hidden or ()) | {
        k for k, v in cells.items() if (0 < int(v) < limit if _is_count(v) else v is not None)
    }
    state = _Lines(cells, lines, hidden, limit, floors)
    while any(state.stale):
        for i, stale in enumerate(state.stale):
            if stale:
                state.stale[i] = False
                state.complete(i)
    return hidden


def suppress_count_matrix(rows: list[list[Any]], threshold: int | None = None) -> set[tuple[int, int]]:
    """Cells to hide in a table whose row and column totals are known (primary + secondary)."""
    cells = {(i, j): v for i, row in enumerate(rows) for j, v in enumerate(row)}
    lines = [[(i, j) for j in range(len(row))] for i, row in enumerate(rows)]
    lines += [[(i, j) for i in range(len(rows))] for j in range(len(rows[0]) if rows else 0)]
    return _suppress_lines(cells, lines, threshold)


def _mask(value: Any, threshold: int | None = None) -> str:
    """One marker for every hidden cell, so small (primary) cells are not told apart."""
    return SECONDARY_SUPPRESSED


def suppress_counts(values: list[Any], threshold: int | None = None) -> list[Any]:
    """Suppress a list of counts whose total is visible (primary + secondary)."""
    hidden = suppress_count_matrix([list(values)], threshold)
    return [_mask(v, threshold) if (0, j) in hidden else v for j, v in enumerate(values)]


def line_count_text(count: int, line: Sequence[int], threshold: int | None = None) -> str:
    """A count for a sentence: the number, or "some" when it or another cell of its line is from 1
    to threshold-1 (the line's total is visible, so a shown neighbour would give it away)."""
    limit = _limit(threshold)
    if all(value == 0 or value >= limit for value in [count, *line]):
        return str(count)
    return "some"


def _small_or_unknown(value: Any) -> bool:
    """A patient count from 1 to threshold-1, or anything that is not a count (fail closed)."""
    return not _is_count(value) or 0 < int(value) < _limit(None)


def _small_patients(value: Any) -> bool:
    """A count carried beside a shown count is 1-10, or unknown (fail closed: a missing count is
    never read as safe)."""
    return _small_or_unknown(value)


def overlap_hidden(
    counts: Sequence[Any], total: Any, patients: Sequence[Any] | None = None, forced: set[int] | None = None
) -> set[int]:
    """Group counts to hide when a unit can be in more than one group (a patient with positive
    tests in two years), so the groups add up to more than ``total``, which a reader knows (the
    same question without the grouping shows it).

    The groups are no partition, so "total minus the other groups" gives nothing; what a reader
    can work out from the total and the shown groups is each shown group's complement (the units
    with no row in it: ``total - count``) and, with every group shown, the excess (``sum -
    total``), which pins how many units are in two groups when it is small (with two groups it
    is exactly that number). So a group is hidden when its count, its complement or the patients
    behind it are from 1 to threshold-1; with nothing hidden, a small excess hides every group;
    and when the shown groups do not cover the total, the hidden groups must hold the rest, a
    lower bound that can pin a small one, so every group is hidden. A hidden group can then be
    small or nearly the total (its complement small), and the shown counts never pin which.
    ``forced``: groups hidden for another reason (the patients behind an encounter complement).
    """
    every = set(range(len(counts)))
    if not _is_count(total) or any(not _is_count(c) for c in counts):
        return every
    hidden = {i for i, c in enumerate(counts) if _small_or_unknown(c)} | set(forced or ())
    hidden |= {i for i, p in enumerate(patients or ()) if p is not None and _small_patients(p)}
    hidden |= {i for i, c in enumerate(counts) if 0 < int(total) - int(c) < _limit(None)}
    if not hidden and 0 < sum(int(c) for c in counts) - int(total) < _limit(None):
        return every
    shown = sum(int(c) for i, c in enumerate(counts) if i not in hidden)
    if hidden and int(total) - shown > 0:
        # Every unit is in some group, so the hidden groups hold at least total - shown units:
        # beside a small hidden group that bound can pin it (10 when the bound is 10). Withhold.
        return every
    return hidden


def _encounter_groups(analysis: Mapping[str, Any], values: Sequence[Any]) -> list[dict[str, Any]] | None:
    """The patient cells of each group (``encounter_cells``), aligned with ``values``; None when
    the answer counts patients. A group with no cells fails closed (every cell unknown)."""
    cells = analysis.get("encounter_cells")
    if not isinstance(cells, Mapping):
        return None
    found = {repr(g.get("group")): g for g in cells.get("groups") or []}
    return [found.get(repr(value), dict.fromkeys(ENCOUNTER_GROUP_CELLS)) for value in values]


def _encounter_n_hidden(analysis: Mapping[str, Any]) -> bool:
    """Whether n must be hidden because the patients behind a count of the whole answer (the
    units counted, the units excluded, the units the limits removed) are 1-10."""
    cells = analysis.get("encounter_cells")
    if not isinstance(cells, Mapping):
        return False
    counts = {
        N_USED: analysis.get("n_used"),
        N_EXCLUDED: analysis.get("n_excluded"),
        N_FILTERED: _filtered_out(analysis),
    }
    for key, name in _N_PATIENT_CELLS.items():
        shown = counts.get(key)
        if (not _is_count(shown) or int(shown) != 0) and _small_or_unknown(cells.get(name)):
            return True
    used = analysis.get("n_used", analysis.get("total_count"))
    if not _is_count(used) or int(used) != 0:
        return any(_small_or_unknown(cells.get(f"all_total_{name}")) for name in _UNIT_CELLS)
    return False


def _encounter_totals(analysis: Mapping[str, Any]) -> Mapping[str, Any] | None:
    cells = analysis.get("encounter_cells")
    return cells if isinstance(cells, Mapping) else None


def _cells_small(cells: Mapping[str, Any] | None, names: Sequence[str]) -> bool:
    """Any of ``names`` in ``cells`` is 1-10 or missing; no cells at all is nothing to check."""
    return cells is not None and any(_small_patients(cells.get(name)) for name in names)


_MIXED_CELLS: Final = ("mixed", "mixed_patients")


def _rollup_withheld(analysis: Mapping[str, Any]) -> bool:
    """Whether every share or rate of the answer is withheld for its roll-up: 1-10 units (or
    patients) have the value on one row and not on another, so the "ever" answer minus the
    "always" one would give them."""
    cells = analysis.get("rollup_cells")
    return isinstance(cells, Mapping) and bool(cells) and _cells_small(cells, _MIXED_CELLS)


def _rates_hidden_by_lines(
    analysis: Mapping[str, Any], groups: Sequence[Mapping[str, Any]], seed: set[int] | None = None
) -> set[int]:
    """Rates to hide because the ungrouped answer gives a line total for a count behind them.

    The ungrouped share shows the total events (and non-events), and its "ever" and "always"
    versions the units whose value changed. In a partition each of these is the sum of the
    groups', so a rate hidden alone would be given back by subtraction: each count is
    suppressed down its line (primary + secondary). With overlapping groups the line is a union,
    checked like the groups' n (``overlap_hidden``). A group without its counts fails closed.

    ``seed``: rates already hidden for another reason (their n, their patients). They are hidden
    cells of every line, so the line hides a second cell beside them instead of showing the rest
    (total minus the shown groups would give a seeded rate's events back). The units whose value
    changed are one line, in units; a group whose changed units come from 1-10 patients is a
    seed of it.
    """
    every = set(range(len(groups)))
    events = [g.get("events") for g in groups]
    non_events = [
        int(g["n"]) - int(g["events"]) if _is_count(g.get("n")) and _is_count(g.get("events")) else None for g in groups
    ]
    seeded = set(seed or ())
    lines: list[tuple[list[Any], Any]] = []
    cells = analysis.get("rollup_cells")
    if isinstance(cells, Mapping) and cells:
        found = [
            {
                **dict.fromkeys(_MIXED_CELLS),
                **next((c for c in cells.get("groups") or [] if repr(c.get("group")) == repr(g.get("group"))), {}),
            }
            for g in groups
        ]
        seeded |= {i for i, c in enumerate(found) if _small_patients(c.get("mixed_patients")) and c.get("mixed")}
        lines.append(([c.get("mixed") for c in found], cells.get("mixed")))
    hidden: set[int] = set(seeded)
    if analysis.get("overlapping"):
        for counts, total in lines:
            hidden |= overlap_hidden(counts, total, forced=seeded)
        return hidden
    for counts in [events, non_events, *(counts for counts, _total in lines)]:
        if any(not _is_count(c) for c in counts):
            return every
        hidden |= _suppress_lines(dict(enumerate(counts)), [list(range(len(counts)))], hidden=set(seeded))
    return hidden


def _fold_for(analysis: Mapping[str, Any], values: Sequence[Any], counts: Sequence[Any], hidden: set[int]) -> _Fold:
    """Which hidden groups lose their label (folded into one row), by index.

    A group's label is withheld when the group itself is small: its count, or for encounters the
    patients behind it, is 1-10. When no hidden group is small, every hidden group keeps its label
    with its values hidden (hidden for its complement, a readmission count, the other groups);
    which groups are hidden then says nothing about size. Otherwise, in a partition
    of counts the existing split rule decides (``_small_parts``), so folding never tells a small
    group from a complement where that would pin it; with encounters of few patients among them,
    or overlapping groups, every hidden group folds under a label that claims no size, or only
    the small ones do when the groups overlap (no line adds up there to pin them).
    """
    if not hidden:
        return _Fold()
    encounter = _encounter_groups(analysis, values)
    small_count = {i for i in hidden if _small_or_unknown(counts[i])}
    small_people = {i for i in hidden if encounter is not None and _small_patients(encounter[i].get("patients"))}
    if not small_count | small_people:
        return _Fold()  # every hidden group is large: its label names no small group
    if analysis.get("overlapping"):
        return _Fold(small_count | small_people, HIDDEN_GROUPS_LABEL)
    if small_people - small_count:
        return _Fold(hidden, HIDDEN_GROUPS_LABEL)
    return _small_parts(dict(enumerate(counts)), hidden)


def _split_rates_hidden(analysis: Mapping[str, Any]) -> set[int] | None:
    """Rates to hide when a unit is in more than one group: None means every rate.

    A reader has each shown group's events and non-events (rate and n) and the ungrouped share.
    Events are then overlapping counts like the groups' n: ``overlap_hidden`` against the units
    with the value in some group (the ungrouped share's events), and non-events against the units
    without it in some group. Every other sum or difference of them is a sum over units in two or
    more groups (in two groups, with the value in two, without it in two, with and without it);
    when any such set holds 1-10 units or patients, every rate is hidden.
    """
    cells = analysis.get("rollup_cells")
    if not analysis.get("overlapping"):
        return set()
    if not isinstance(cells, Mapping):
        return None  # overlapping groups with nothing to check them against: fail closed
    names = [name for split in _SPLIT_CELLS for name in (split, f"{split}_patients")]
    if _cells_small(cells, names):
        return None
    groups = list(analysis.get("groups") or [])
    events = [g.get("events") for g in groups]
    non_events = [
        int(g["n"]) - int(g["events"]) if _is_count(g.get("n")) and _is_count(g.get("events")) else None for g in groups
    ]
    hidden = overlap_hidden(events, cells.get("event_any"))
    hidden |= overlap_hidden(non_events, cells.get("nonevent_any"))
    return hidden


def unit_groups_hidden(
    analysis: Mapping[str, Any], values: Sequence[Any], counts: Sequence[Any], total: Any, *, uncovered: bool = True
) -> set[int]:
    """Group indices to hide in a count or rate of distinct units (patients or encounters).

    Overlapping groups: ``overlap_hidden``. A partition: the line of groups with its total shown
    (primary + secondary). For encounters, a group is also hidden when the patients behind it or
    behind its complement (total minus it) are 1-10; every group is hidden when the patients
    behind the units in two groups are 1-10, or (``uncovered``, partitions only) when the patients
    behind the units in no shown group (total minus the shown groups) are 1-10.
    """
    every = set(range(len(counts)))
    encounter = _encounter_groups(analysis, values)
    forced: set[int] = set()
    if encounter is not None:
        forced = {i for i, g in enumerate(encounter) if _cells_small(g, _GROUP_SIZE_CELLS)}
    if analysis.get("overlapping"):
        hidden = overlap_hidden(counts, total, None, forced)
    else:
        hidden = _suppress_lines(dict(enumerate(counts)), [list(range(len(counts)))], hidden=forced)
    cells = analysis.get("encounter_cells")
    if isinstance(cells, Mapping):
        if analysis.get("overlapping") and _small_patients(cells.get("multi_patients")):
            return every
        # In a partition the units in no shown group are total minus the shown groups. With
        # overlapping groups that difference is only a bound, and one above 0 already hid all.
        partition = not analysis.get("overlapping")
        if (
            uncovered
            and partition
            and hidden
            and hidden != every
            and _small_or_unknown(cells.get("uncovered_patients"))
        ):
            return every
    return hidden


def encounter_groups_hidden(analysis: Mapping[str, Any]) -> set[int]:
    """The groups a count or rate of encounters hides before the uncovered-patients check: the
    semantic layer counts the patients behind the units in no shown group for exactly these."""
    if analysis.get("type") == "rate":
        groups = list(analysis.get("groups") or [])
        values = [g.get("group") for g in groups]
        return unit_groups_hidden(
            analysis, values, [g.get("n") for g in groups], analysis.get("n_used"), uncovered=False
        )
    column = analysis.get("grouped_by")
    rows = list(analysis.get("group_counts") or [])
    values = [row.get(column) for row in rows]
    return unit_groups_hidden(
        analysis, values, [row.get("count") for row in rows], analysis.get("total_count"), uncovered=False
    )


def _one(analysis: Mapping[str, Any]) -> str:
    """One counted unit, in words: the encounter noun the question used ("admission"), else
    patient or visit by the grain."""
    noun = analysis.get("unit_noun")
    if isinstance(noun, str) and noun:
        return noun
    return {"patients": "patient", "visits": "visit"}.get(_unit(analysis) or "", "unit")


def _article(word: str) -> str:
    return "An" if word[:1].lower() in "aeiou" else "A"


def _overlap_sentence(analysis: Mapping[str, Any], group_col: str, total: Any) -> str:
    one = _one(analysis)
    return (
        f"{_article(one)} {one} with rows in more than one {group_col} is counted in each, so the {group_col} "
        f"counts add up to more than {total}."
    )


_ROLLUP_SENTENCES = {
    "any": "{a} {one} counts as having {event} when at least one of its rows has it.",
    "all": "{a} {one} counts as having {event} only when all of its rows that record it have it.",
}


def _rollup_sentence(analysis: Mapping[str, Any], event: str) -> str | None:
    template = _ROLLUP_SENTENCES.get(str(analysis.get("rollup")))
    return template.format(a=_article(_one(analysis)), one=_one(analysis), event=event) if template else None


def _units_word(analysis: Mapping[str, Any]) -> str | None:
    """What n counts in words: the encounter noun the question used ("tests"), else ``_unit``."""
    noun = analysis.get("unit_noun")
    return f"{noun}s" if isinstance(noun, str) and noun else _unit(analysis)


def _unit(analysis: Mapping[str, Any]) -> str | None:
    """What n counts, when the query service knows the dataset's grain: patients, visits, or rows
    (some rows have no patient ID, so rows are not patients)."""
    unit = analysis.get("unit")
    return unit if unit in ("patients", "visits", "rows") else None


def _counted(value: Any, analysis: Mapping[str, Any]) -> str:
    """'n=377' or 'n=377 patients'."""
    unit = _units_word(analysis)
    return f"n={value} {unit}" if unit else f"n={value}"


def _event_phrase(column: str, level: Any) -> str:
    """'Statin use = Yes', or 'event=1' for a bare code."""
    name = short_name(column)
    label = display_level(column, level)
    return label if label.startswith(f"{name}=") else f"{name} = {label}"


# Counts behind each group of an encounter answer (``SemanticLayer._encounter_cells``), never shown:
# the patients behind its units, the units beyond those patients (readmissions, units with no
# patient ID) and the patients with two or more units; for its units, events and non-events.
_UNIT_CELLS: Final = ("patients", "excess", "readmitted")
_GROUP_SIZE_CELLS: Final = (*_UNIT_CELLS, "complement_patients")  # hide the group (its n)
_GROUP_RATE_CELLS: Final = tuple(f"{part}_{name}" for part in ("event", "nonevent") for name in _UNIT_CELLS)
ENCOUNTER_GROUP_CELLS: Final = (*_GROUP_SIZE_CELLS, *_GROUP_RATE_CELLS)
# The same counts for the whole answer: its units (n or the total), events and non-events.
ENCOUNTER_TOTAL_CELLS: Final = tuple(
    f"all_{part}_{name}" for part in ("total", "event", "nonevent") for name in _UNIT_CELLS
)
_TOTAL_RATE_CELLS: Final = tuple(f"all_{part}_{name}" for part in ("event", "nonevent") for name in _UNIT_CELLS)
# Counts behind a rolled-up share (``SemanticLayer._rollup_cells``); each is checked in units and
# in patients. "mixed": units with the value on one row and not on another (ever minus always).
_SPLIT_CELLS: Final = ("multi", "event_multi", "nonevent_multi", "split")

# Cell keys for the n clause; tuples so they never collide with a group name or a table cell.
N_USED = ("n", "used")
N_EXCLUDED = ("n", "excluded")
N_FILTERED = ("n", "filtered")  # rows the question's filters removed; never displayed
# Patient counts behind the encounter counts of the whole answer a reader can work out.
_N_PATIENT_CELLS: Final = {N_EXCLUDED: "unrecorded_patients", N_FILTERED: "filtered_patients"}


def _filtered_out(analysis: Mapping[str, Any]) -> Any:
    """What the question's filters removed (``n_filtered_out``, counted where the rows were
    filtered, in the unit of the answer's total); 0 when the result does not carry it. A value
    that is not a count is returned as it is, so the suppression hides around it."""
    filtered = analysis.get("n_filtered_out")
    if filtered is None:
        return 0
    if _is_count(filtered) and int(filtered) < 0:
        return "invalid"  # a negative count means the totals disagree: hide around it, not trust it
    return filtered


def _cohort(analysis: Mapping[str, Any]) -> int:
    """The cohort size the reader can learn: n_used + n_excluded.

    Without filters it is the dataset row count the app shows; with filters a COUNT with the
    same filter shows it. It is never larger than the row count, so bounds on n use it.
    """
    return _as_count(analysis.get("n_used")) + _as_count(analysis.get("n_excluded"))


def _unchecked(key: Any, value: Any, hidden: set[Any]) -> bool:
    """Whether a statistic guarded by a count is withheld: the count is hidden, or it is missing
    or not a count, so nothing shows that the statistic covers enough patients."""
    return key in hidden or not _is_count(value)


def suppress_with_exclusions(
    analysis: Mapping[str, Any],
    parts: Mapping[Any, Any],
    lines: Sequence[Sequence[Any]] = (),
    *,
    floor: int = 0,
) -> set[Any]:
    """Hidden cells of one answer: its count cells (``parts``) plus n_used, n_excluded and the
    rows the filters removed.

    n_used + n_excluded + filtered-out is the dataset row count, which the app shows, so the
    three are a line with a visible total; the filtered-out cell is never displayed but is
    suppressed like the others. n_used + n_excluded is the size of the filtered cohort, which a
    COUNT with the same filter shows, so those two are a line as well: a hidden exclusion count
    always hides n (and the reverse). The parts (group sizes, table cells) add up to n_used and
    are first suppressed against it; ``lines`` are further lines among the parts (rows,
    columns). When n_used is hidden every part is hidden, zeros included: a visible part beside
    a hidden n tells a reader which cells the algorithm saw as small. ``floor`` is the smallest
    n_used the analysis runs on; a reader knows it, so when the cohort size leaves n_used only
    two possible values (floor and floor + 1) n is always hidden, even with nothing excluded.
    """
    n_used, excluded = analysis.get("n_used"), analysis.get("n_excluded")
    hidden = _suppress_lines(parts, [*lines, list(parts)])
    cells = {**parts, N_USED: n_used, N_EXCLUDED: excluded, N_FILTERED: _filtered_out(analysis)}
    n_lines = [[N_USED, N_EXCLUDED, N_FILTERED], [N_USED, N_EXCLUDED]]
    hidden = _suppress_lines(cells, [*lines, *n_lines], hidden=hidden, floors={N_USED: floor})
    if _is_count(n_used) and _cohort(analysis) - floor <= 1:
        hidden |= {N_USED, N_EXCLUDED, N_FILTERED}
    if N_EXCLUDED in hidden or N_FILTERED in hidden:
        hidden.add(N_USED)  # whatever the lines found: n beside a hidden complement gives it away
    if N_USED in hidden:
        hidden |= set(parts)
    return hidden


def _n_clause(analysis: Mapping[str, Any], hidden: set[Any]) -> str | None:
    """'n=377; 23 excluded for missing data', 'n=377' or 'n hidden'.

    ``hidden`` comes from ``suppress_with_exclusions``. With n hidden the clause does not say
    whether any rows were excluded: that would split the hidden answers into "n small" and
    "exclusions small", and the analysis floor can pin n inside the first. A hidden exclusion
    count always hides n, so there is no "n=377; some excluded".
    """
    n_used = analysis.get("n_used")
    if not _is_count(n_used):
        return None
    if N_USED in hidden or N_EXCLUDED in hidden:
        return "n hidden"
    excluded = analysis.get("n_excluded")
    if _is_count(excluded) and int(excluded) > 0:
        return f"{_counted(n_used, analysis)}; {excluded} excluded for missing data"
    return _counted(n_used, analysis)


def _overall_statistics_shown(n: Any, n_hidden: bool, cohort: int, floor: int) -> bool:
    """Whether statistics of one whole sample (a mean, a median) may be shown.

    Never for n below the threshold: they can be one patient's values. With n hidden for
    another reason (a small exclusion count) they are shown only if that does not narrow n:
    both kinds of hidden n must still leave at least two values (floor at most threshold - 2,
    and a cohort, n + excluded, of at least threshold + 2: a COUNT with the question's filters
    shows the cohort, and a cohort of 12 with a statistic shown would pin n=11).
    """
    limit = _limit(None)
    if not (_is_count(n) and int(n) >= limit):
        return False
    return not n_hidden or (floor <= limit - 2 and cohort >= limit + 2)


def _with_details(claim: str, details: list[str | None]) -> str:
    kept = [d for d in details if d]
    return f"{claim} ({'; '.join(kept)})" if kept else claim


def _plain(text: str) -> str:
    """Strip the markdown bold markers compute.py puts in headlines."""
    return text.replace("**", "")


def term_label(term: Any) -> str:
    """Readable regression term: drop the boolean-dummy suffix; decode coded-category dummies.

    'DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis_2' -> 'DEXA result: Osteopenia'.
    """
    text = str(term).removesuffix("[T.True]")
    base, sep, level = text.rpartition("_")
    if sep and header_has_code_labels(base):
        return f"{short_name(base)}: {display_level(base, level)}"
    return text


def _round_cell(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _table(rows: list[dict[str, Any]], count_columns: tuple[str, ...] = ()) -> TablePayload | None:
    """Display table: floats rounded; each patient-count column suppressed down its length
    (primary + secondary), because its total is shown or derivable."""
    if not rows:
        return None
    safe_rows = [{k: _round_cell(v) for k, v in row.items()} for row in to_json_safe(rows)]
    for column in count_columns:
        masked = suppress_counts([row.get(column) for row in safe_rows])
        for row, value in zip(safe_rows, masked, strict=True):
            if column in row:
                row[column] = value
    return {"columns": list(rows[0].keys()), "rows": safe_rows}


def _hidden_cell(value: Any) -> bool:
    return isinstance(value, str) and value == SECONDARY_SUPPRESSED


class _Fold(set[Any]):
    """Groups folded into one row, and that row's label."""

    def __init__(self, keys: Iterable[Any] = (), label: str = OTHER_LABEL) -> None:
        super().__init__(keys)
        self.label = label


def _split_fold_allowed(sizes: Mapping[Any, Any], hidden: set[Any], floor: int = 1) -> bool:
    """Whether small groups may fold alone, complements keeping their own rows.

    That split tells a reader which hidden groups are small (1 to threshold-1) and which are
    complements (at least the threshold, and below the smallest visible count, since complements
    are hidden smallest first). It is allowed only when, for every way the hidden groups could
    split into k small and h complements, each small group keeps several possible values given
    the hidden mass ({10, 43, 44}: the 43 is below 44, so the 10 would be pinned: refused). The
    test reads only what a reader sees (the hidden mass, how many groups are hidden, the visible
    counts), so whether the split is used reveals nothing more. ``floor``: the smallest group the
    analysis runs on (a reader knows it: two per group for a comparison of means).
    """
    keys = [k for k in sizes if k in hidden]
    if not keys or not all(_is_count(sizes[k]) for k in keys):
        return False
    limit = _limit(None)
    mass = sum(int(sizes[k]) for k in keys)
    visible = [int(v) for k, v in sizes.items() if k not in hidden and _is_count(v) and int(v) > 0]
    if not visible:
        # Every group hidden: the reader also knows the algorithm had to hide them all, which
        # pins a small group when the complements are near the threshold ({3, 11, 11, 11}).
        return False
    cap = min(visible) - 1
    m = len(keys)
    for k in range(1, m + 1):
        h = m - k
        low, high = max(floor * k, mass - h * cap), min((limit - 1) * k, mass - limit * h)
        if low > high:
            continue  # no such split is possible
        if h > 1 or (h == 1 and k > 1):
            # Two small groups next to a complement say the small ones alone were exposed (at
            # their cap); several tied complements say how the algorithm stopped. Either pins a
            # small group ({10, 10, 11, 15}, {3, 11, 11, 50}), so only one small group with one
            # complement, or small groups alone, may be split.
            return False
        one_low, one_high = max(floor, low - (limit - 1) * (k - 1)), min(limit - 1, high - floor * (k - 1))
        if one_high - one_low < 1 + SPLIT_FOLD_SLACK:
            return False
    return True


# Values beyond two a small group must keep: the reader also knows the algorithm stopped hiding
# as soon as the line was safe, which can remove a value at an edge (brute-force audited).
SPLIT_FOLD_SLACK: Final = 2


def _small_parts(sizes: Mapping[Any, Any], hidden: set[Any], floor: int = 1) -> _Fold:
    """Groups whose label is folded into one row.

    When ``_split_fold_allowed``: only the primary-suppressed groups (true size 1 to threshold-1,
    or not a count: fail closed) fold into ``OTHER_LABEL`` ("groups under 11"); a group hidden
    only as their complement ({200, 150, 5}: the 150) keeps its own row and label with its values
    hidden, never the under-11 label. Otherwise every hidden group folds into
    ``HIDDEN_GROUPS_LABEL``, which claims no size, so the fold follows from the hidden markers a
    reader sees anyway. With n hidden every size is marked hidden whatever the groups are, and a
    fold chosen from the true sizes would tell the reader which groups are small: then no label
    is folded and every group's values stay hidden.
    """
    if N_USED in hidden:
        return _Fold()
    folded = {g for g, n in sizes.items() if g in hidden or is_small_level(n)}
    if not folded:
        return _Fold()
    if _split_fold_allowed(sizes, folded, max(1, floor)):
        return _Fold({g for g in folded if is_small_level(sizes[g])})
    return _Fold(folded, HIDDEN_GROUPS_LABEL)


def _fold_rows(table: TablePayload | None, fold: set[str], label: str = OTHER_LABEL) -> TablePayload | None:
    """The table with the rows labelled in ``fold`` folded into one ``OTHER_LABEL`` row (last).

    A group's label beside its hidden size still names the few patients in it (and, beside
    another group's 0, says which arm they are in). Only groups of 1 to threshold-1 patients are
    folded (``_small_parts``); a group hidden as their complement keeps its row. The Other row
    shows no number: every cell of it is hidden.
    """
    if table is None or not table["rows"] or not fold:
        return table
    label_col = table["columns"][0]
    kept = [row for row in table["rows"] if str(row.get(label_col)) not in fold]
    if len(kept) == len(table["rows"]):
        return table
    other = {col: SECONDARY_SUPPRESSED for col in table["columns"]}
    other[label_col] = label
    return {"columns": table["columns"], "rows": [*kept, other]}


def _fold_crosstab(
    rows: list[dict[str, Any]],
    groups: list[str],
    fold_cols: set[str],
    fold_rows: set[int],
    fold_label: str = HIDDEN_GROUPS_LABEL,
) -> list[dict[str, Any]]:
    """Contingency rows with the ``fold_cols`` columns and ``fold_rows`` rows folded into one
    ``fold_label`` column and row, every cell of which is hidden.

    A column of hidden cells and zeros (drug hidden, placebo 0) under a label such as ldl=80.9
    says which arm the patients with that exact value are in, whatever the cells hide.
    """
    if not rows:
        return rows
    label_col = next(iter(rows[0]))
    kept_cols = [g for g in groups if g not in fold_cols]
    out = []
    for i, row in enumerate(rows):
        if i in fold_rows:
            continue
        shown = {label_col: row[label_col], **{g: row[g] for g in kept_cols}}
        if fold_cols:
            shown[fold_label] = SECONDARY_SUPPRESSED
        out.append(shown)
    if fold_rows:
        out.append({label_col: fold_label, **{g: SECONDARY_SUPPRESSED for g in kept_cols}})
        if fold_cols:
            out[-1][fold_label] = SECONDARY_SUPPRESSED
    return out


def _too_many_levels(column: str, levels: int) -> Presentation:
    """No table for a grouping with more levels than a reader can scan without seeing patients."""
    return {"claim": too_many_levels_message(column, levels), "table": None}


def _present_comparison(analysis: Mapping[str, Any]) -> Presentation:
    outcome = analysis.get("outcome_col") or "outcome"
    group_col = analysis.get("group_col") or "group"
    test_type = str(analysis.get("test_type") or "")
    test = _TEST_LABELS.get(test_type, test_type)
    p = _num(analysis.get("p_value"))
    group_means = analysis.get("group_means") or {}
    group_sizes = analysis.get("group_sizes") or {}
    groups = [str(g) for g in analysis.get("groups") or []]
    if len(groups) > MAX_GROUP_LEVELS or len(analysis.get("contingency") or []) > MAX_GROUP_LEVELS:
        dimension = group_col if len(groups) > MAX_GROUP_LEVELS else outcome
        levels = len(groups) if len(groups) > MAX_GROUP_LEVELS else len(analysis.get("contingency") or [])
        return _too_many_levels(str(dimension), levels)
    label = {g: display_level(group_col, g) for g in groups}
    outcome_name = short_name(outcome)
    p_clause = f"{test} {_fmt_p(p)}" if p is not None else None

    if test_type in ("welch_t_test", "t_test", "anova") and "table_min_cell" in analysis:
        # An outcome with two measured values: each group's mean gives back how many of its
        # patients had each value, so the means and the test follow the table rule compute applied.
        mean_refusal = table_statistic_refusal(analysis.get("table_min_cell"))
        if mean_refusal is not None:
            return {"claim": mean_refusal, "table": None}
    if test_type in ("welch_t_test", "t_test", "anova"):
        sizes = {str(g): group_sizes.get(str(g)) for g in group_means}
        hidden = suppress_with_exclusions(analysis, sizes, floor=MIN_GROUP_N * max(len(sizes), 1))
        small_groups = _small_parts(sizes, hidden, MIN_GROUP_N)
        fold = {label.get(g, g) for g in small_groups}
        n_clause = _n_clause(analysis, hidden)
        # A mean of a handful of patients is close to their values (one patient: that value),
        # so a group's mean is hidden with its size, as the Kaplan-Meier median is. Hiding only
        # the small groups' means would mark the complements as the large groups. A group with
        # no size, or one that is not a count, is treated as small.
        rows = [
            {
                group_col: label.get(str(g), str(g)),
                f"mean {outcome_name}": _mask(m) if _unchecked(str(g), sizes[str(g)], hidden) else _num(m),
                "n": _mask(sizes[str(g)]) if str(g) in hidden else sizes[str(g)],
            }
            for g, m in group_means.items()
        ]

    if test_type in ("welch_t_test", "t_test"):
        diff = _num(analysis.get("mean_diff"))
        lo, hi = _num(analysis.get("ci_lower")), _num(analysis.get("ci_upper"))
        claim: str | None = None
        if small_groups:
            # A difference of means with a group of a few patients is close to their values, and
            # naming the groups names that group: the comparison is withheld.
            claim = _with_details(
                f"The comparison of mean {outcome_name} by {group_col} is not shown because a group has fewer "
                f"than {_limit(None)} patients",
                [n_clause],
            )
            return {"claim": claim + ".", "table": _fold_rows(_table(rows), fold, small_groups.label)}
        if len(groups) == 2 and diff is not None:
            first, second = label[groups[0]], label[groups[1]]
            if round(abs(diff), 2) == 0:
                claim = f"Mean {outcome_name} was almost the same in {first} and {second} (difference under 0.01)"
            else:
                # mean_diff and its CI are groups[0] - groups[1]. When the claim names groups[1]
                # as higher, it states groups[1] - groups[0], so the CI flips sign and order too.
                higher, lower = (first, second) if diff > 0 else (second, first)
                claim = f"Mean {outcome_name} was {_fmt(abs(diff))} higher in {higher} than {lower}"
                if diff < 0 and lo is not None and hi is not None:
                    lo, hi = -hi, -lo
            ci_clause = f"95% CI {_fmt(lo)} to {_fmt(hi)}" if lo is not None and hi is not None else None
            if p is not None and p >= 0.05:
                claim += ", not statistically significant"
            claim = _with_details(claim, [ci_clause, p_clause, n_clause])
            claim += f". Unadjusted comparison by {group_col}."
        return {"claim": claim, "table": _fold_rows(_table(rows), fold, small_groups.label)}

    if test_type == "anova":
        claim = None
        if p is not None:
            verb = "differed" if p < 0.05 else "did not differ significantly"
            n_groups = analysis.get("n_groups") or len(groups) or len(group_means)
            claim = _with_details(
                f"Mean {outcome_name} {verb} across {n_groups} {group_col} groups", [p_clause, n_clause]
            )
            claim += ". Unadjusted."
        return {"claim": claim, "table": _fold_rows(_table(rows), fold, small_groups.label)}

    # chi-square / Fisher on a contingency table (rows = outcome levels, columns = groups)
    # With a cell of 1-10 patients the test is withheld, with the rule compute applied
    # (``table_statistic_refusal`` on the recorded ``table_min_cell``): a payload that skipped
    # compute's check, or one stored before the cells were recorded, shows no p-value, test
    # name, significance or expected-count warning either (fail closed).
    test_refusal = table_statistic_refusal(analysis.get("table_min_cell"))
    contingency = [dict(row) for row in analysis.get("contingency") or []]
    cells = {(i, j): row.get(g) for i, row in enumerate(contingency) for j, g in enumerate(groups)}
    lines = [[(i, j) for j in range(len(groups))] for i in range(len(contingency))]
    lines += [[(i, j) for i in range(len(contingency))] for j in range(len(groups))]
    hidden = suppress_with_exclusions(analysis, cells, lines, floor=MIN_COMPARISON_N)
    n_clause = _n_clause(analysis, hidden)
    any_small_group = any(_is_count(n) and 0 < int(n) < _limit(None) for n in group_sizes.values())
    unchecked = any(not _is_count(v) for v in cells.values())  # the shares cannot be checked
    withheld = bool(hidden & cells.keys()) or any_small_group or unchecked
    proportions = analysis.get("group_proportions") or {}
    event_level = analysis.get("event_level")
    claim = None
    if test_refusal is not None:
        claim = test_refusal
    elif p is not None:
        significance = "statistically significant" if p < 0.05 else "not statistically significant"
        if proportions and event_level is not None and not withheld:
            shares = " and ".join(
                f"{_fmt(100 * float(proportions[g]), 0)}% in {label[g]}" for g in groups if g in proportions
            )
            claim = _with_details(
                f"{_event_phrase(outcome, event_level)} was {shares}; the difference was {significance}",
                [p_clause, n_clause],
            )
        else:
            verb = "differed" if p < 0.05 else "did not differ significantly"
            claim = _with_details(f"The distribution of {outcome_name} {verb} by {group_col}", [p_clause, n_clause])
        claim += ". Unadjusted."
        if withheld:
            claim += f" Group percentages are withheld because a group or cell has fewer than {_limit(None)} patients."
        warning = analysis.get("warning")
        if warning:
            claim += f" {warning}"
    if not contingency:
        return {"claim": claim, "table": None}
    rows = []
    for i, row in enumerate(contingency):
        shown: dict[str, Any] = {outcome_name: display_level(outcome, row.get(outcome))}
        for j, g in enumerate(groups):
            shown[label[g]] = _mask(row.get(g)) if (i, j) in hidden else row.get(g)
        rows.append(shown)

    # Fold every group (column) and level (row) none of whose non-zero cells shows on the cell
    # lines alone: its label would name a handful of patients and, beside 0s, their group. The
    # fold follows from the hidden markers a reader sees, so it can take a group hidden only as a
    # complement; its label (HIDDEN_GROUPS_LABEL) therefore claims no size, never "under 11".
    parts = set() if N_USED in hidden else hidden
    blank_cols = {
        label[g]
        for j, g in enumerate(groups)
        if all((i, j) in parts or not cells[(i, j)] for i in range(len(contingency)))
    }
    # Levels fold only beside a group still shown under its label: with every group in Other,
    # a level's label stands next to hidden cells alone and names nobody.
    blank_rows = (
        {i for i in range(len(contingency)) if all((i, j) in parts or not cells[(i, j)] for j in range(len(groups)))}
        if len(blank_cols) < len(groups)
        else set()
    )
    rows = _fold_crosstab(rows, [label[g] for g in groups], blank_cols, blank_rows, HIDDEN_GROUPS_LABEL)
    return {"claim": claim, "table": _table(rows)}


def _present_predictor(analysis: Mapping[str, Any]) -> Presentation:
    outcome = analysis.get("outcome_col") or "outcome"
    outcome_name = short_name(outcome)
    event_level = analysis.get("event_level")
    event = _event_phrase(outcome, event_level) if event_level is not None else outcome_name
    summary = analysis.get("summary") or {}
    n_clause = _n_clause(analysis, suppress_with_exclusions(analysis, {}, floor=MIN_REGRESSION_N))
    # A term names a level ("site_West") beside an odds ratio that rests on its patients and its
    # reference level's, and with the totals other answers show it gives back how many of them
    # had the outcome. With 1-10 in a level, or in a level with one outcome value, the model is
    # withheld: folding the level into "Other" would be a different model. Compute refuses it
    # first; this applies the same rule (``predictor_model_refusal``) to the recorded
    # ``term_sizes`` and ``cell_sizes``, so a payload that skipped compute's check, or one stored
    # before both were recorded, is withheld too (fail closed). No refusal names a column: for a
    # 0/1 predictor the column is the label of its level.
    named = analysis.get("significant_predictors") or []
    terms = [var for var in summary if var != "Intercept"] + [
        s.get("variable") for s in named if isinstance(s, Mapping)
    ]
    refusal = predictor_model_refusal(terms, analysis.get("term_sizes"), analysis.get("cell_sizes"))
    if refusal is not None:
        return {"claim": refusal, "table": None}
    rows = [
        {
            "variable": term_label(var),
            "odds ratio": _num(stats.get("Odds Ratio")),
            "95% CI low": _num(stats.get("CI Lower")),
            "95% CI high": _num(stats.get("CI Upper")),
            "p": _num(stats.get("P-Value")),
        }
        for var, stats in summary.items()
        if var != "Intercept" and isinstance(stats, Mapping)
    ]
    significant = sorted(analysis.get("significant_predictors") or [], key=lambda s: _num(s.get("p_value")) or 1.0)
    predictors = ", ".join(str(r["variable"]) for r in rows)
    model = f"Logistic regression of {event}" + (f" on {predictors}" if predictors else "")
    if significant:
        top = significant[0]
        odds, lo, hi, p = (_num(top.get(k)) for k in ("odds_ratio", "ci_lower", "ci_upper", "p_value"))
        claim = _with_details(
            f"{term_label(top.get('variable'))} had the strongest evidence of association with {event}",
            [
                f"OR {_fmt(odds)}" if odds is not None else None,
                f"95% CI {_fmt(lo)} to {_fmt(hi)}" if lo is not None and hi is not None else None,
                _fmt_p(p) if p is not None else None,
            ],
        )
        adjustment = "each odds ratio is adjusted for the other predictors" if len(rows) > 1 else "unadjusted"
        claim += f". {_with_details(model, [n_clause])}; {adjustment}."
    elif rows:
        adjustment = "predictors adjusted for each other" if len(rows) > 1 else "unadjusted"
        claim = f"No predictor reached p<0.05 for {event}. {_with_details(model, [n_clause])}; {adjustment}."
    else:
        claim = None
    return {"claim": claim, "table": _table(rows)}


def _present_relationship(analysis: Mapping[str, Any]) -> Presentation:
    # A correlation of two coded variables is a statistic of their table of counts (for two 0/1
    # columns, phi): withheld when a cell holds 1-10 patients, by the rule compute applied
    # (``table_statistic_refusal`` on each pair's ``table_min_cell``; not recorded: withheld).
    # A withheld pair sorts last with its r hidden, so its place in the order says nothing of r.
    refusals: list[str] = []
    pairs: list[dict[str, Any]] = []
    for c in analysis.get("correlations") or []:
        if c.get("var1") == c.get("var2"):
            continue
        refusal = table_statistic_refusal(c.get("table_min_cell"), may_rest_on_no_table=True)
        if refusal is not None:
            refusals.append(refusal)
        pairs.append(
            {
                "variable 1": c.get("var1"),
                "variable 2": c.get("var2"),
                "r": SECONDARY_SUPPRESSED if refusal is not None else _num(c.get("correlation")),
            }
        )
    pairs.sort(key=lambda r: abs(r["r"]) if isinstance(r["r"], float) else -1.0, reverse=True)
    # n_observations + its exclusions (+ filtered-out rows) is the row count, as for n_used
    sample = {**analysis, "n_used": analysis.get("n_observations")}
    n_clause = _n_clause(sample, suppress_with_exclusions(sample, {}, floor=MIN_CORRELATION_N))
    claim = None
    if pairs and isinstance(pairs[0]["r"], float):
        top = pairs[0]
        r = top["r"]
        strength = "strong" if abs(r) >= 0.5 else "moderate" if abs(r) >= 0.3 else "weak"
        shown = "The strongest correlation shown" if refusals else "The strongest correlation"
        claim = _with_details(
            f"{shown} was {strength}: {top['variable 1']} and {top['variable 2']}",
            [f"r={_fmt(r)}", n_clause],
        )
        claim += "."
    if refusals:
        # One sentence for every withheld pair; the small-cell one when any pair has a small cell.
        reason = SMALL_CELL_TEST_REFUSAL if SMALL_CELL_TEST_REFUSAL in refusals else refusals[0]
        claim = f"{claim} {reason}" if claim else reason
    return {"claim": claim, "table": _table(pairs)}


def _present_survival(analysis: Mapping[str, Any]) -> Presentation:
    time_col = analysis.get("time_col") or "time"
    event_col = analysis.get("event_col") or "event"
    group_col = analysis.get("group_col")
    event_level = analysis.get("event_level")
    event = _event_phrase(event_col, event_level) if event_level is not None else short_name(event_col)
    group_medians = analysis.get("group_medians") or {}
    if group_col and group_medians:
        group_sizes = analysis.get("group_sizes") or {}

        def _median_text(value: Any) -> str:
            m = _num(value)
            return f"{_fmt(m, 1)}" if m is not None else "not reached"

        # A median of a handful of patients is one patient's time. Hide the median of every
        # group whose size is hidden: showing only the complements' medians would tell the small
        # groups (fewer than the threshold) apart from the complements (at least the threshold).
        sizes = {g: group_sizes.get(g) for g in group_medians}
        hidden = suppress_with_exclusions(analysis, sizes, floor=MIN_SURVIVAL_N)
        n_clause = _n_clause(analysis, hidden)
        small = {g for g in group_medians if _unchecked(g, sizes[g], hidden)}
        folded = _small_parts(sizes, hidden)
        medians_text = ", ".join(
            [
                f"{display_level(group_col, g)} {'not shown' if g in small else _median_text(m)}"
                for g, m in group_medians.items()
                if g not in folded
            ]
            + ([f"{folded.label} not shown"] if folded else [])
        )
        # The log-rank test, with group sizes and the event total, narrows each group's events:
        # withheld with a group x event cell of 1-10 patients, by the rule compute applied. A
        # payload with a log-rank result but no recorded cells is withheld too (fail closed).
        logrank = _num(analysis.get("logrank_p_value"))
        tested = logrank is not None or "table_min_cell" in analysis
        logrank_refusal = table_statistic_refusal(analysis.get("table_min_cell")) if tested else None
        if logrank_refusal is not None:
            logrank = None
        claim = _with_details(
            f"Median time to {event} ({time_col}) by {group_col}: {medians_text}",
            [f"log-rank {_fmt_p(logrank)}" if logrank is not None else None, n_clause],
        )
        claim += ". Unadjusted."
        if small:
            claim += " Medians are not shown for groups whose size is hidden to protect privacy."
        if logrank_refusal is not None:
            claim += f" {logrank_refusal}"
        rows = [
            {
                group_col: display_level(group_col, g),
                "n": _mask(sizes[g]) if g in hidden else sizes[g],
                f"median {time_col}": SECONDARY_SUPPRESSED if g in small else _num(m),
            }
            for g, m in group_medians.items()
        ]
        return {
            "claim": claim,
            "table": _fold_rows(_table(rows), {display_level(group_col, g) for g in folded}, folded.label),
        }
    n_used, n_events = analysis.get("n_used"), analysis.get("n_events")
    parts: dict[str, Any] = {}
    if _is_count(n_used) and _is_count(n_events):
        parts = {"events": int(n_events), "censored": int(n_used) - int(n_events)}
    counted = bool(parts)
    hidden = suppress_with_exclusions(analysis, parts, floor=MIN_SURVIVAL_N)
    n_clause = _n_clause(analysis, hidden)
    median = _num(analysis.get("median_survival"))
    # With n small the median may be one patient's time (and whether it was reached hints at
    # their events). The survival floor makes a small n exactly that floor, so the median is
    # hidden whenever n is hidden: showing it would tell that n apart from a large one. It is
    # hidden too when n is missing or not a count: nothing then shows the sample is large.
    median_hidden = not _overall_statistics_shown(n_used, N_USED in hidden, _cohort(analysis), MIN_SURVIVAL_N)
    if median_hidden:
        reason = "n is hidden" if _is_count(n_used) else "n is missing"
        claim = _with_details(f"Median time to {event} ({time_col}) is not shown because {reason}", [n_clause]) + "."
    elif median is not None:
        claim = _with_details(f"Median time to {event} was {_fmt(median, 1)} ({time_col})", [n_clause]) + "."
    else:
        claim = _with_details(f"Median time to {event} was not reached during follow-up ({time_col})", [n_clause]) + "."
    # Summary only: the Kaplan-Meier step table lists individual event times and its steps reveal
    # counts in small risk sets.
    if not counted:
        return {"claim": claim, "table": None}
    row = {
        "n": _mask(n_used) if N_USED in hidden else n_used,
        **{k: _mask(v) if k in hidden else v for k, v in parts.items()},
        f"median {time_col}": _mask(median) if median_hidden else median,
    }
    return {"claim": claim, "table": _table([row])}


def _present_descriptive(analysis: Mapping[str, Any]) -> Presentation:
    col = analysis.get("focused_variable")
    if col:
        name = short_name(col)
        if analysis.get("is_numeric"):
            stats: dict[str, Any] = {k: _num(analysis.get(k)) for k in ("mean", "median", "std", "min", "max")}
            n = analysis.get("non_null_count")
            # n + missing (+ filtered-out rows) is the row count: "n=95" of 100 rows says 5 missing.
            sample = {**analysis, "n_used": n, "n_excluded": analysis.get("null_count")}
            n_hidden = N_USED in suppress_with_exclusions(sample, {})
            shown_n = _mask(n) if n_hidden else n
            claim = None
            if not _overall_statistics_shown(n, n_hidden, _cohort(sample), 0):
                # With n from 1 to 10, min/max/median are patients' values and the mean of one
                # patient is that value: show none of the statistics, only that n is hidden.
                # The same when n is missing or not a count: nothing shows the sample is large.
                stats = {k: _mask(v) for k, v in stats.items()}
                reason = "n is hidden (n hidden)" if _is_count(n) else "n is missing"
                claim = f"Statistics for {name} are not shown because {reason}."
            elif (
                str(analysis.get("statistic") or "mean") != "mean"
                and stats.get(str(analysis.get("statistic"))) is not None
            ):
                # Lead with the statistic the question asked for; the table has the rest.
                asked = str(analysis.get("statistic"))
                claim = _with_details(
                    f"{_STATISTIC_LABELS.get(asked, asked)} {name} was {_fmt(stats[asked])}",
                    [_n_clause(sample, {N_USED} if n_hidden else set())],
                )
                claim += "."
            elif stats["mean"] is not None:
                claim = _with_details(
                    f"Mean {name} was {_fmt(stats['mean'])}",
                    [
                        f"SD {_fmt(stats['std'])}" if stats["std"] is not None else None,
                        f"median {_fmt(stats['median'])}" if stats["median"] is not None else None,
                        _n_clause(sample, {N_USED} if n_hidden else set()),
                    ],
                )
                claim += "."
            row: dict[str, Any] = {"variable": name, "n": shown_n, **stats}
            return {"claim": claim, "table": _table([row])}
        counts = list(analysis.get("value_counts") or [])
        rows = [{name: display_level(col, r.get(col)), "count": r.get("count")} for r in counts]
        # The listed levels (value_counts().head(10)), the levels cut from the list and the
        # filtered-out rows add up to the dataset row count; the last two are never displayed.
        listed = {i: r.get("count") for i, r in enumerate(rows)}
        row_count = analysis.get("row_count")
        listed_total = sum(_as_count(v) for v in listed.values())
        rest = _as_count(row_count) - listed_total if _is_count(row_count) else 0
        cells: dict[Any, Any] = {**listed, "rest": rest, N_FILTERED: _filtered_out(analysis)}
        hidden = _suppress_lines(cells, [list(cells)])
        # Folded by true size, before masking: only levels of 1-10 patients, unless that pins one.
        # The rest (levels cut from the list) and the filtered-out rows are part of this line but
        # never displayed, so whether a hidden level is small cannot be shown safely: every hidden
        # level folds, decided by the hidden markers alone, under a label that claims no size.
        fold = _Fold({i for i in listed if i in hidden}, HIDDEN_GROUPS_LABEL)
        small = {str(rows[i][name]) for i in fold}
        for i, row in enumerate(rows):
            if i in hidden:
                row["count"] = _mask(row["count"])
        table = _fold_rows(_table(rows), small, fold.label)
        claim = None
        if table and table["rows"]:
            top = table["rows"][0]
            # A frequency counts patients: distinct patients per level once rows are visits.
            noun = _unit(analysis) or "patients"
            if top[name] == fold.label:
                # The top level's count is hidden: its label is folded with the other small ones.
                claim = f"The most common {name} is not shown because its count is hidden to protect privacy."
            elif _hidden_cell(top["count"]):
                claim = f"Most common {name}: {top[name]} (count hidden to protect privacy)."
            else:
                claim = f"Most common {name}: {top[name]} ({top['count']} {noun})."
        return {"claim": claim, "table": table}
    rows_n = analysis.get("row_count")
    cols = analysis.get("column_count")
    claim = f"The dataset has {rows_n} rows and {cols} columns." if rows_n is not None and cols is not None else None
    return {"claim": claim, "table": _table(list(analysis.get("summary_stats") or []))}


def _present_count(analysis: Mapping[str, Any]) -> Presentation:
    group_col = analysis.get("grouped_by")
    group_counts = analysis.get("group_counts")
    total = analysis.get("total_count")
    if group_col and group_counts:
        if isinstance(group_counts, Mapping):
            rows = [{group_col: k, "count": v} for k, v in group_counts.items()]
        else:
            rows = [dict(r) for r in group_counts]
        if len(rows) > MAX_GROUP_LEVELS:
            return _too_many_levels(str(group_col), len(rows))
        values = [row.get(group_col) for row in rows]
        for row in rows:
            row[group_col] = display_level(group_col, row.get(group_col))
        if _total_hidden(analysis, total):
            # Every group is then small (or the filtered-out rows are), and k groups of at least
            # 1 with a total under the threshold can pin each one (10 groups -> all 1), so the
            # groups are not listed.
            return {"claim": _hidden_total_claim(analysis, total), "table": None}
        noun = "group" if len(rows) == 1 else "groups"
        counted = f"{total} {_units_word(analysis)}" if _units_word(analysis) else f"{total} total"
        claim = f"{counted} across {len(rows)} {group_col} {noun}." if total is not None else None
        if analysis.get("overlapping") or analysis.get("encounter_cells") is not None:
            return _present_unit_groups(analysis, rows, values, str(group_col), total, claim)
        # Folded by true size, before suppression: only groups of 1-10 patients, unless that pins one.
        counts = [row.get("count") for row in rows]
        masked = suppress_counts(counts)
        fold = _small_parts(dict(enumerate(counts)), {i for i, v in enumerate(masked) if _hidden_cell(v)})
        small = {str(rows[i][group_col]) for i in fold}
        return {"claim": claim, "table": _fold_rows(_table(rows, count_columns=("count",)), small, fold.label)}
    if _total_hidden(analysis, total):
        return {"claim": _hidden_total_claim(analysis, total), "table": None}
    # What was counted, in the question's own noun for encounters ("admissions"), else the grain.
    unit = f" {_units_word(analysis)}" if _units_word(analysis) else ""
    claim = f"Total count: {total}{unit}." if total is not None else None
    return {"claim": claim, "table": None}


def _present_unit_groups(
    analysis: Mapping[str, Any],
    rows: list[dict[str, Any]],
    values: list[Any],
    group_col: str,
    total: Any,
    claim: str | None,
) -> Presentation:
    """Distinct units per group when the groups are no plain partition of patients: a unit can be
    in two groups (``overlapping``: ``overlap_hidden``), or the unit is an encounter and each
    group's patients are counted too (a group of 1-10 patients is hidden however many
    admissions it has). Every hidden group is folded into one row whose label claims no size."""
    counts = [row.get("count") for row in rows]
    hidden = unit_groups_hidden(analysis, values, counts, total)
    if analysis.get("overlapping") and claim:
        claim = f"{claim} {_overlap_sentence(analysis, group_col, total)}"
    shown_rows = [
        {group_col: row[group_col], "count": SECONDARY_SUPPRESSED if i in hidden else row.get("count")}
        for i, row in enumerate(rows)
    ]
    if hidden and claim:
        claim += f" Groups are hidden where a count of fewer than {_limit(None)} patients could be worked out."
    folded = _fold_for(analysis, values, counts, hidden)
    fold = {str(rows[i][group_col]) for i in folded}
    return {"claim": claim, "table": _fold_rows(_table(shown_rows), fold, folded.label)}


_STATISTIC_LABELS = {"mean": "Mean", "median": "Median", "min": "Minimum", "max": "Maximum", "std": "SD of"}


def _group_label(column: str, level: Any) -> str:
    return "missing" if level is None else display_level(column, level)


def _counts_small(events: Any, n: Any) -> bool:
    """A share events/n reveals both counts (with and without the value) once n is known.

    True when either is from 1 to threshold-1, or when the counts are missing (fail closed).
    """
    if not _is_count(events) or not _is_count(n):
        return True
    return any(0 < count < _limit(None) for count in (int(events), int(n) - int(events)))


def _percent(value: float) -> str:
    return f"{_fmt(100 * value, 1)}%"


def _present_proportion(analysis: Mapping[str, Any]) -> Presentation:
    """Share of patients with a value. Hidden when either of its counts (with, without) is small:
    the share and n together give both counts."""
    column = str(analysis.get("column") or "value")
    event = _event_phrase(column, analysis.get("value"))
    share = _num(analysis.get("proportion"))
    hidden_n = suppress_with_exclusions(analysis, {})
    encounter_n = _encounter_n_hidden(analysis)
    if encounter_n:
        hidden_n |= {N_USED, N_EXCLUDED, N_FILTERED}
    n_clause = _n_clause(analysis, hidden_n)
    if share is None:
        return {
            "claim": f"No patient who matched has '{column}' recorded, so there is no share to report.",
            "table": None,
        }
    n_used = analysis.get("n_used")
    # Hidden too when n is hidden and that narrows it (as for any whole-sample statistic).
    withheld = (
        _counts_small(analysis.get("events"), n_used)
        or not _overall_statistics_shown(n_used, N_USED in hidden_n, _cohort(analysis), 0)
        # Encounters: n hidden for the patients behind it, or 1-10 patients behind its events
        # or non-events (or their readmissions); only then does a hidden n withhold the share.
        or encounter_n
        or _cells_small(_encounter_totals(analysis), _TOTAL_RATE_CELLS)
        or _rollup_withheld(analysis)
    )
    if withheld:
        claim = _with_details(
            f"The share with {event} is not shown because one of its counts is fewer than {_limit(None)} patients",
            [n_clause],
        )
    else:
        noun = _units_word(analysis) or "patients"
        population = f"{noun} who matched the limits" if analysis.get("limited") else noun
        claim = _with_details(f"{_percent(share)} of {population} had {event}", [n_clause])
        rollup = _rollup_sentence(analysis, event)
        if rollup:
            claim += f". {rollup.removesuffix('.')}"
    row = {
        "proportion": SECONDARY_SUPPRESSED if withheld else share,
        "n": SECONDARY_SUPPRESSED if N_USED in hidden_n else n_used,
    }
    return {"claim": claim + ".", "table": _table([row])}


def _present_rate(analysis: Mapping[str, Any]) -> Presentation:
    """Rate of a value in each group. A group's n is suppressed as group sizes are; its rate is
    hidden with its n, or when either of its counts (with, without) is small."""
    column = str(analysis.get("column") or "value")
    group_col = str(analysis.get("group_col") or "group")
    event = _event_phrase(column, analysis.get("value"))
    groups = list(analysis.get("groups") or [])
    if len(groups) > MAX_GROUP_LEVELS:
        return _too_many_levels(group_col, len(groups))
    sizes = {str(g.get("group")): g.get("n") for g in groups}
    keys = list(sizes)
    values = [g.get("group") for g in groups]
    encounter = _encounter_groups(analysis, values)
    by_units = bool(analysis.get("overlapping")) or encounter is not None
    split = _split_rates_hidden(analysis)
    every_rate = split is None or _rollup_withheld(analysis)
    if by_units:
        # A unit in two groups (the groups do not add up to n), or encounters (the patients
        # behind each count are checked): ``unit_groups_hidden``.
        hidden = suppress_with_exclusions(analysis, {} if analysis.get("overlapping") else sizes)
        if _encounter_n_hidden(analysis):
            hidden |= {N_USED, N_EXCLUDED, N_FILTERED}
        if N_USED in hidden:
            hidden |= set(keys)
        else:
            found = unit_groups_hidden(analysis, values, [sizes[k] for k in keys], analysis.get("n_used"))
            hidden |= {keys[i] for i in found}
    else:
        hidden = suppress_with_exclusions(analysis, sizes)
    n_clause = _n_clause(analysis, hidden)
    # Groups counted by encounter, or overlapping, fold every hidden group under a label that
    # claims no size: which of them are small is never told apart.
    if by_units:
        folded = _fold_for(analysis, values, [sizes[k] for k in keys], {i for i, k in enumerate(keys) if k in hidden})
        small_groups = _Fold({keys[i] for i in folded}, folded.label)
    else:
        small_groups = _small_parts(sizes, hidden)
    rows, parts, withheld = [], [], False
    # Rates hidden on their own, then every line through them completed (``_rates_hidden_by_lines``).
    alone = {
        index
        for index, g in enumerate(groups)
        if every_rate
        or _unchecked(str(g.get("group")), g.get("n"), hidden)
        or _counts_small(g.get("events"), g.get("n"))
        or _cells_small(encounter[index] if encounter is not None else None, _GROUP_RATE_CELLS)
        or index in (split or set())
    }
    by_line = _rates_hidden_by_lines(analysis, groups, alone)
    for index, g in enumerate(groups):
        key, n = str(g.get("group")), g.get("n")
        rate = _num(g.get("rate"))
        label = _group_label(group_col, g.get("group"))
        rate_hidden = index in alone or index in by_line
        withheld = withheld or rate_hidden
        rows.append(
            {
                group_col: label,
                "rate": SECONDARY_SUPPRESSED if rate_hidden else rate,
                "n": SECONDARY_SUPPRESSED if key in hidden else n,
            }
        )
        if key in small_groups:
            continue  # its label is folded into Other, in the table and the claim
        parts.append(f"{label} {'not shown' if rate_hidden else ('no values' if rate is None else _percent(rate))}")
    if small_groups:
        parts.append(f"{small_groups.label} not shown")
    claim = _with_details(f"{event} by {group_col}: {', '.join(parts)}", [n_clause]) + ". Unadjusted."
    if withheld:
        claim += f" A rate is not shown when one of its counts is fewer than {_limit(None)} patients."
    rollup = _rollup_sentence(analysis, event)
    if rollup:
        claim += f" {rollup}"
    if analysis.get("overlapping"):
        claim += f" {_overlap_sentence(analysis, group_col, 'n')}"
    fold = {_group_label(group_col, g.get("group")) for g in groups if str(g.get("group")) in small_groups}
    return {"claim": claim, "table": _fold_rows(_table(rows), fold, small_groups.label)}


def _present_grouped_statistic(analysis: Mapping[str, Any]) -> Presentation:
    """One statistic per group. A group's value is hidden with its size: the median, minimum or
    maximum of a few patients is one patient's value."""
    statistic = str(analysis.get("statistic") or "mean")
    metric = short_name(str(analysis.get("metric") or "value"))
    group_col = str(analysis.get("group_col") or "group")
    groups = list(analysis.get("groups") or [])
    if len(groups) > MAX_GROUP_LEVELS:
        return _too_many_levels(group_col, len(groups))
    sizes = {str(g.get("group")): g.get("n") for g in groups}
    hidden = suppress_with_exclusions(analysis, sizes)
    n_clause = _n_clause(analysis, hidden)
    value_column = f"{statistic} {metric}"
    small_groups = _small_parts(sizes, hidden)
    rows, parts = [], []
    for g in groups:
        key, value = str(g.get("group")), _num(g.get("value"))
        label = _group_label(group_col, g.get("group"))
        shown = SECONDARY_SUPPRESSED if _unchecked(key, g.get("n"), hidden) else value
        rows.append({group_col: label, value_column: shown, "n": SECONDARY_SUPPRESSED if key in hidden else g.get("n")})
        withheld = shown == SECONDARY_SUPPRESSED
        if key not in small_groups:  # a small group's label is folded into Other
            parts.append(f"{label} {'not shown' if withheld else ('no values' if value is None else _fmt(value))}")
    if small_groups:
        parts.append(f"{small_groups.label} not shown")
    label = _STATISTIC_LABELS.get(statistic, statistic)
    claim = _with_details(f"{label} {metric} by {group_col}: {', '.join(parts)}", [n_clause]) + ". Unadjusted."
    if any(key in hidden for key in sizes):
        claim += " Values are not shown for groups whose size is hidden to protect privacy."
    fold = {_group_label(group_col, g.get("group")) for g in groups if str(g.get("group")) in small_groups}
    return {"claim": claim, "table": _fold_rows(_table(rows), fold, small_groups.label)}


def _total_hidden(analysis: Mapping[str, Any], total: Any) -> bool:
    """A count total is hidden when it, or the rows the filters removed, is from 1 to 10."""
    if not _is_count(total):
        return False
    if _encounter_n_hidden(analysis):
        return True  # 11 or more encounters of 1-10 patients (or removed by the limits)
    cells = {"total": total, N_FILTERED: _filtered_out(analysis)}
    return "total" in _suppress_lines(cells, [list(cells)])


def _hidden_total_claim(analysis: Mapping[str, Any], total: Any) -> str:
    # Without filters the total is the row count the app already shows. With filters one wording
    # covers "total small" and "filtered-out rows small", so the claim does not say which.
    if _filtered_out(analysis) != 0 or _encounter_n_hidden(analysis):
        return "Total count hidden."
    unit = f" {_unit(analysis)}" if _unit(analysis) else ""
    return f"Total count: {suppress_count(total)}{unit}."


_PRESENTERS = {
    "comparison": _present_comparison,
    "predictor": _present_predictor,
    "relationship": _present_relationship,
    "survival": _present_survival,
    "descriptive": _present_descriptive,
    "count": _present_count,
    "proportion": _present_proportion,
    "rate": _present_rate,
    "grouped_statistic": _present_grouped_statistic,
}


def present_analysis(analysis: Mapping[str, Any] | None) -> Presentation:
    """Claim sentence and statistics table for a formatted analysis result.

    Returns an empty presentation for missing, failed or unknown results so callers
    never show a claim that the analysis did not support.
    """
    if not analysis or analysis.get("error"):
        return {"claim": None, "table": None}
    presenter = _PRESENTERS.get(str(analysis.get("type")))
    if presenter is None:
        return {"claim": None, "table": None}
    presentation = presenter(analysis)
    claim = presentation["claim"]
    if claim is None:
        return presentation
    notes = [_implausible_note(analysis.get("implausible")), *[str(n) for n in analysis.get("data_notes") or []]]
    kept = [note for note in notes if note]
    if kept:
        claim = " ".join([claim, *kept])
    return {"claim": claim, "table": presentation["table"]}


def _implausible_note(implausible: Any) -> str | None:
    """'Includes 12 values of age outside 0–122 years (likely entry errors).' The count is "some"
    when it, or the rest of the values the statistic used, is from 1 to 10."""
    if not isinstance(implausible, Mapping):
        return None
    count, total = implausible.get("count"), implausible.get("line_total")
    if not _is_count(count) or int(count) <= 0:
        return None
    amount = line_count_text(int(count), [int(total) - int(count)] if _is_count(total) else [1])
    column = short_name(str(implausible.get("column") or "this column"))
    bounds = implausible.get("bounds") or "its plausible range"
    return f"Includes {amount} values of {column} outside {bounds} (likely entry errors)."

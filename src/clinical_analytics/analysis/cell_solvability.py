# SPDX-License-Identifier: MIT
"""Which statistics of a few-valued column can be shown without letting a hidden count be
worked out: an exact solvability test.

A column that behaves like categories (``grouping.behaves_like_categories``: length of stay 1-15
days, a pain score 0-10, ages 51-69) is a table of patients per (value, group). A COUNT shows
each cell but those of 1 to ``SMALL_CELL_THRESHOLD - 1`` patients. A mean adds one equation in
the cells (the sum), an SD another (the sum of squares), a median order constraints. Whether a
hidden cell is given back is a question about the non-negative integer solutions of that system,
not about whether a small cell exists: five hidden ages under a mean are five unknowns in one
equation; one hidden age is given back by n alone.

What the reader knows (``_build``), for the column in the answer's scope:

* every cell of the value x group table that is 0 or at least the threshold, that each other
  cell is 1 to threshold-1, and each group's n;
* each group's statistics, by level (``LEVELS``). Statistics are shown in nested levels: "mean"
  (mean, minimum, maximum: the sum, and which cells are 0, which the reader knows already), "sd"
  (+ SD: the sum of squares) and "median" (+ median: order constraints; for an even n the two
  middle values are fixed at the data's own, a stronger constraint than the reader has). A
  statistic is shown only when its level and every level below it pin nothing, so whatever set
  of statistics a reader collects over several questions is one level's set, and that level was
  checked with all of it. One decision serves the compute step and the presenters, and a level
  never depends on which statistic was asked;
* second, as a defence beyond this answer: each value's total over the groups (a COUNT by value
  in the same scope) and the pooled statistics (at the pooled table's own level). With the
  group sizes those totals alone pin some cells (25 of 33 patients in arm A leaves 8 in B); that
  is the counts' to protect and not changed by withholding a mean, so this pass withholds only
  what the statistics newly pin or narrow.

A constraint the data satisfies may always be added without making the decision unsafe: it only
removes solutions, so a count it leaves free is free for the reader too.

A level is withheld (``_check``) when:

* a hidden cell takes one value over all solutions (``pinned``) or values spanning at most
  ``NARROW_SPREAD`` (``narrow``: the middle one is within 1 of every solution, so the count is
  known to +-1). A cell is never taken as free on the solver's word: two tables are checked
  exactly (integers and fractions) against every constraint, and only two whose counts differ
  by more than ``NARROW_SPREAD`` let it through (any guess is then 2 or more from one of them);
* a union of two or more hidden cells (patients aged 66 or 69) holding 1 to threshold-1 patients
  is pinned or narrow and the counts alone do not pin it (``union_pinned``): with 4 hidden
  evenly spaced values, n, the sum and the sum of squares pin the outer pair's total. Unions the
  counts alone pin (a group's hidden cells together are its n minus its shown cells) are the
  counts' to protect. A single cell the counts alone pin is still withheld (``pinned``);
* the table has more than ``MAX_TABLE_CELLS`` cells (``too_large``), or the solver fails, runs
  past ``SOLVE_BUDGET_S``, or returns a table that does not check out (``unsolved``). Fail closed.

Not modelled: the same column under another filter or another grouping column combined with this
answer (differences of counts across scopes are the shown-count log's to catch), and guesses
weighted by how many solutions put a count at each value.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import structlog

from clinical_analytics.core import nl_query_config
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD

logger = structlog.get_logger(__name__)

# Tuning, centralised in config (see there for why each value).
MAX_TABLE_CELLS: Final = nl_query_config.SOLVABILITY_MAX_CELLS
SOLVE_BUDGET_S: Final = nl_query_config.SOLVABILITY_BUDGET_S
NARROW_SPREAD: Final = nl_query_config.SOLVABILITY_NARROW_SPREAD
UNION_WITNESSES: Final = nl_query_config.SOLVABILITY_UNION_TABLES
# Decided tables kept (least recently used dropped): the same table, asked again, is answered
# at once and the same way, whatever the load. Timeouts are not kept.
DECIDED_TABLES: Final = 256
_DECIDED: OrderedDict[tuple[object, ...], Solvability] = OrderedDict()
_DECIDED_LOCK: Final = threading.Lock()

Level: TypeAlias = Literal["none", "mean", "sd", "median"]
# Nested: each level shows its own statistics and those of every level before it.
LEVELS: Final[tuple[Level, ...]] = ("none", "mean", "sd", "median")
LEVEL_STATISTICS: Final[Mapping[Level, frozenset[str]]] = {
    "none": frozenset(),
    "mean": frozenset({"mean", "min", "max"}),
    "sd": frozenset({"mean", "min", "max", "std"}),
    "median": frozenset({"mean", "min", "max", "std", "median"}),
}

Verdict: TypeAlias = Literal[
    "no_hidden_cell",
    "not_pinned",
    "pinned",
    "narrow",
    "union_pinned",
    "pinned_with_totals",
    "too_large",
    "unsolved",
]
NOT_PINNED: Final = "not_pinned"
NO_HIDDEN_CELL: Final = "no_hidden_cell"


@dataclass(frozen=True)
class CellRange:
    """The counts a hidden cell (``value``, ``group``) can take: ``low`` and ``high`` are each
    realised by a table checked against everything the reader knows."""

    value: Fraction
    group: Hashable
    low: int
    high: int


@dataclass(frozen=True)
class Solvability:
    """The decision: the highest ``level`` whose statistics may be shown, and why the next one
    may not (``verdict``; ``NOT_PINNED`` when every level may be shown). ``ranges``: filled by
    ``hidden_cell_ranges`` only."""

    level: Level
    verdict: Verdict
    hidden: int
    cells: int
    ranges: tuple[CellRange, ...] = ()


class UnsolvedError(Exception):
    """The solver gave no checked answer in time (fail closed)."""


@dataclass(frozen=True)
class _Row:
    """lo <= sum(coefficient * x[index]) <= hi over the hidden cells, exact."""

    terms: tuple[tuple[int, Fraction], ...]
    low: Fraction | None
    high: Fraction | None


@dataclass(frozen=True)
class _Problem:
    hidden: tuple[tuple[int, int], ...]  # (value index, group index) of each unknown
    truth: tuple[int, ...]  # the data's own counts of the hidden cells
    rows: tuple[_Row, ...]  # everything the reader knows, as constraints on the hidden cells
    count_rows: tuple[_Row, ...]  # the equalities of the counts alone (n and shown totals)

    def satisfied(self, x: Sequence[int]) -> bool:
        """Whether integer counts ``x`` meet every constraint exactly."""
        if len(x) != len(self.hidden) or any(not 1 <= int(v) <= SMALL_CELL_THRESHOLD - 1 for v in x):
            return False
        for row in self.rows:
            total = sum((coefficient * int(x[i]) for i, coefficient in row.terms), Fraction(0))
            if (row.low is not None and total < row.low) or (row.high is not None and total > row.high):
                return False
        return True


def _as_fraction(value: object) -> Fraction:
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, int | Fraction):
        return Fraction(value)
    if isinstance(value, Decimal) and value.is_finite():
        return Fraction(value)
    if isinstance(value, np.integer):
        return Fraction(int(value))
    if isinstance(value, float | np.floating) and math.isfinite(float(value)):
        # The value as written (0.8, not the binary 0.8000000000000000444): the reader reasons
        # with the values shown, and decimal values then lie on an exact grid (0.1 steps).
        return Fraction(repr(float(value)))
    raise ValueError(f"A cell value must be a finite number, got {type(value).__name__}")


def _table(counts: Mapping[tuple[object, Hashable], int]) -> tuple[list[Fraction], list[Hashable], list[list[int]]]:
    """Sorted distinct values, groups and the value x group counts (patients)."""
    merged: dict[tuple[Fraction, Hashable], int] = {}
    for (value, group), count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int | np.integer) or int(count) < 0:
            raise ValueError("A cell count must be a non-negative integer")
        key = (_as_fraction(value), group)
        merged[key] = merged.get(key, 0) + int(count)
    values = sorted({v for v, _ in merged})
    groups = sorted({g for _, g in merged}, key=lambda g: (type(g).__name__, repr(g)))
    table = [[merged.get((v, g), 0) for g in groups] for v in values]
    return values, groups, table


def _scaled(values: list[Fraction]) -> list[Fraction]:
    """Values as small integers where they lie on a grid ((v - min) / step), else on [0, 1].

    With each group's n known, sums and sums of squares of these determine those of the raw
    values and back (v = min + step * u), so the constraints are the same; integers keep the
    solver's arithmetic exact. Constraints are checked exactly whatever the scale."""
    if len(values) < 2:
        return [Fraction(0)] * len(values)
    differences = [v - values[0] for v in values]
    denominator = math.lcm(*(d.denominator for d in differences))
    step = Fraction(math.gcd(*(int(d * denominator) for d in differences)), denominator)
    scaled = [d / step for d in differences]
    if scaled[-1] <= 10**6:
        return scaled
    return [d / differences[-1] for d in differences]


def _build(
    values: list[Fraction], groups: list[Hashable], table: list[list[int]], *, level: int, totals: bool, pooled: int = 0
) -> _Problem:
    """What the reader knows, as constraints on the hidden cells: each group's n and shown cells,
    and each group's statistics up to ``level`` (an index into ``LEVELS``). With ``totals``,
    each value's total over the groups (shown, or 1 to threshold-1) and the pooled statistics up
    to ``pooled``. ``count_rows``: the equalities among these that are counts."""
    threshold = SMALL_CELL_THRESHOLD
    hidden = tuple((i, j) for i in range(len(values)) for j in range(len(groups)) if 0 < table[i][j] < threshold)
    truth = tuple(table[i][j] for i, j in hidden)
    index = {cell: k for k, cell in enumerate(hidden)}
    u = _scaled(values)
    rows: list[_Row] = []
    count_rows: list[_Row] = []

    def equal(terms: list[tuple[int, Fraction]]) -> _Row:
        total = sum((c * truth[k] for k, c in terms), Fraction(0))
        return _Row(tuple(terms), total, total)

    def moments(cells: list[int], upto: int) -> None:
        if upto >= 1:
            rows.append(equal([(k, u[hidden[k][0]]) for k in cells]))  # the sum (mean x n)
        if upto >= 2:
            rows.append(equal([(k, u[hidden[k][0]] ** 2) for k in cells]))  # the sum of squares (SD)

    def median(column: list[int], cells: list[int], upto: int) -> None:
        if upto >= 3:
            at_value: dict[int, list[int]] = {}
            for k in cells:
                at_value.setdefault(hidden[k][0], []).append(k)
            rows.extend(_order_rows(column, at_value, truth))

    for j in range(len(groups)):
        cells = [index[(i, j)] for i in range(len(values)) if (i, j) in index]
        if cells:
            counted = equal([(k, Fraction(1)) for k in cells])  # n minus the shown cells
            rows.append(counted)
            count_rows.append(counted)
            moments(cells, level)
        median([table[i][j] for i in range(len(values))], cells, level)
    if totals and len(groups) > 1:
        for i in range(len(values)):
            cells = [index[(i, j)] for j in range(len(groups)) if (i, j) in index]
            if not cells:
                continue
            if sum(table[i]) >= threshold:
                shown = equal([(k, Fraction(1)) for k in cells])  # the value's shown total
                rows.append(shown)
                count_rows.append(shown)
            else:  # a hidden total: 1 to threshold-1, its other cells 0
                rows.append(_Row(tuple((k, Fraction(1)) for k in cells), Fraction(1), Fraction(threshold - 1)))
        everyone = list(range(len(hidden)))
        if everyone:
            moments(everyone, pooled)
        median([sum(line) for line in table], everyone, pooled)
    return _Problem(hidden, truth, tuple(rows), tuple(count_rows))


def _order_rows(column: Sequence[int], at_value: Mapping[int, list[int]], truth: Sequence[int]) -> list[_Row]:
    """Order constraints of the median of a multiset given as the data's counts per sorted value
    (``column``), on the unknowns at each value (``at_value``), the known cells moved to the
    bounds. For each middle position p (one for odd n, two for even n, each fixed at the data's
    own value t): at most p-1 below t, at least p at or below t."""
    n = sum(column)
    if n == 0:
        return []
    positions = [(n + 1) // 2] if n % 2 else [n // 2, n // 2 + 1]
    cumulative = np.cumsum(column)
    rows: list[_Row] = []
    for p in positions:
        t = int(np.searchsorted(cumulative, p))  # the data's value at position p
        for upto, low, high in ((t, None, p - 1), (t + 1, p, None)):
            unknown = [k for i in range(upto) for k in at_value.get(i, [])]
            if not unknown:
                continue
            known = Fraction(sum(column[:upto]) - sum(truth[k] for k in unknown))
            rows.append(
                _Row(
                    tuple((k, Fraction(1)) for k in unknown),
                    None if low is None else Fraction(low) - known,
                    None if high is None else Fraction(high) - known,
                )
            )
    return rows


def _integer_kernel(rows: Sequence[_Row], width: int) -> list[list[int]]:
    """An integer basis of the vectors d with sum(coefficient * d) = 0 for every row (exact
    Gauss-Jordan over fractions; the rows here are 0/1 count rows, a few dozen by a few dozen)."""
    matrix = [[Fraction(0)] * width for _ in rows]
    for r, row in enumerate(rows):
        for k, coefficient in row.terms:
            matrix[r][k] += coefficient
    pivots: list[int] = []
    rank = 0
    for column in range(width):
        pivot = next((r for r in range(rank, len(matrix)) if matrix[r][column] != 0), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        lead = matrix[rank][column]
        matrix[rank] = [value / lead for value in matrix[rank]]
        for r in range(len(matrix)):
            if r != rank and matrix[r][column] != 0:
                factor = matrix[r][column]
                matrix[r] = [a - factor * b for a, b in zip(matrix[r], matrix[rank], strict=True)]
        pivots.append(column)
        rank += 1
    basis: list[list[int]] = []
    for free in (c for c in range(width) if c not in pivots):
        vector = [Fraction(0)] * width
        vector[free] = Fraction(1)
        for r, column in enumerate(pivots):
            vector[column] = -matrix[r][free]
        scale = math.lcm(*(v.denominator for v in vector))
        integers = [int(v * scale) for v in vector]
        divisor = math.gcd(*integers)
        basis.append([v // divisor for v in integers])
    return basis


class _Solver:
    """Integer programs over one ``_Problem``, within one deadline."""

    def __init__(self, problem: _Problem, deadline: float) -> None:
        from scipy.optimize import LinearConstraint

        self.problem = problem
        self.deadline = deadline
        width = len(problem.hidden)
        matrix = np.zeros((len(problem.rows), width))
        low = np.full(len(problem.rows), -np.inf)
        high = np.full(len(problem.rows), np.inf)
        for r, row in enumerate(problem.rows):
            for k, coefficient in row.terms:
                matrix[r, k] += float(coefficient)
            if row.low is not None:
                low[r] = float(row.low)
            if row.high is not None:
                high[r] = float(row.high)
        self.constraint = LinearConstraint(matrix, low, high)

    def _remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise UnsolvedError("time budget spent")
        return remaining

    def extreme(self, objective: np.ndarray) -> tuple[int, ...]:
        """A table minimising ``objective`` over everything the reader knows, checked exactly.

        Raises:
            UnsolvedError: no optimum in time, or one that does not check out (the data's own
                table always satisfies the constraints, so "infeasible" is a solver failure).
        """
        result = self._milp(objective, self._remaining())
        if result.status != 0 or result.x is None:
            raise UnsolvedError(f"milp status {result.status}")
        table = tuple(int(v) for v in np.rint(result.x))
        if not self.problem.satisfied(table):
            raise UnsolvedError("solution does not check out")
        return table

    def _milp(self, objective: np.ndarray, time_limit: float) -> Any:
        from scipy.optimize import Bounds, milp

        width = len(self.problem.hidden)
        return milp(
            objective,
            constraints=self.constraint,
            integrality=np.ones(width),
            bounds=Bounds(np.ones(width), np.full(width, SMALL_CELL_THRESHOLD - 1)),
            options={"time_limit": time_limit, "presolve": True, "mip_rel_gap": 0.0},
        )

    def union_candidate(
        self, witnesses: Sequence[Sequence[int]], scope: Sequence[int], cuts: Sequence[Sequence[int]], *, partial: bool
    ) -> list[int] | None:
        """Two or more hidden cells of ``scope`` holding 1 to threshold-1 patients in the data
        whose total is within ``NARROW_SPREAD`` in every witness table, other than the unions in
        ``cuts``; with ``partial``, not every cell of ``scope``. None when there is none.

        Every union the reader could pin or narrow is among these (its total is the same, or
        within ``NARROW_SPREAD``, in every checked table), so None means there is none. One
        found may be free after all: the caller checks it and adds tables that tell it apart.

        Raises:
            UnsolvedError: no answer in time.
        """
        from scipy.optimize import Bounds, LinearConstraint, milp

        h = len(self.problem.hidden)
        width = h + 1  # z (cells chosen), then L (the lowest total over the witnesses)
        rows: list[np.ndarray] = []
        low: list[float] = []
        high: list[float] = []

        def add(coefficients: np.ndarray, lo: float, hi: float) -> None:
            rows.append(coefficients)
            low.append(lo)
            high.append(hi)

        for table in witnesses:
            line = np.zeros(width)
            line[:h] = table
            line[h] = -1.0
            add(line, 0.0, float(NARROW_SPREAD))
        truth = np.zeros(width)
        truth[:h] = self.problem.truth
        add(truth, 1.0, float(SMALL_CELL_THRESHOLD - 1))
        chosen = np.zeros(width)
        chosen[list(scope)] = 1.0
        add(chosen, 2.0, float(len(scope) - 1) if partial else np.inf)
        for cut in cuts:  # not exactly this union again
            line = np.zeros(width)
            line[:h] = [1.0 if v else -1.0 for v in cut]
            add(line, -np.inf, float(sum(cut) - 1))
        upper = np.zeros(width)
        upper[list(scope)] = 1.0
        lower = np.zeros(width)
        lower[h], upper[h] = -np.inf, np.inf
        result = milp(
            np.zeros(width),
            constraints=LinearConstraint(np.vstack(rows), np.array(low), np.array(high)),
            integrality=np.ones(width),
            bounds=Bounds(lower, upper),
            options={"time_limit": self._remaining(), "presolve": True},
        )
        if result.status == 2:
            return None
        if result.status != 0 or result.x is None:
            raise UnsolvedError(f"union milp status {result.status}")
        return [int(v) for v in np.rint(result.x[:h])]


def _sample(witnesses: Sequence[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """At most ``UNION_WITNESSES`` distinct tables, evenly spaced, the data's own and the latest
    (the solver's, which tell apart the last union found) always among them. Fewer tables only
    let more unions through to be checked: a union is never passed over for want of a table."""
    distinct = list(dict.fromkeys(witnesses))
    if len(distinct) <= UNION_WITNESSES:
        return distinct
    latest = distinct[-UNION_WITNESSES // 2 :]
    rest = distinct[: -UNION_WITNESSES // 2]
    stride = max(1, len(rest) // (UNION_WITNESSES - len(latest)))
    return list(dict.fromkeys([distinct[0], *rest[::stride][: UNION_WITNESSES - len(latest) - 1], *latest]))


def _spread(witnesses: Sequence[Sequence[int]], weights: Sequence[int]) -> int:
    totals = [sum(w * x for w, x in zip(weights, table, strict=True)) for table in witnesses]
    return max(totals) - min(totals)


def _unit(width: int, k: int) -> list[int]:
    unit = [0] * width
    unit[k] = 1
    return unit


@dataclass
class _Pass:
    """One model's verdict, the checked tables it found and (when asked) each cell's range."""

    verdict: Verdict = NOT_PINNED
    witnesses: list[tuple[int, ...]] = field(default_factory=list)
    ranges: list[tuple[int, int, int]] = field(default_factory=list)  # (cell, low, high)


def _check(
    problem: _Problem,
    deadline: float,
    exempt: Mapping[int, int] | None = None,
    *,
    separable: bool,
    every_range: bool = False,
) -> _Pass:
    """Whether a hidden cell or a small union of hidden cells is pinned or narrow over ``problem``;
    unions the counts alone pin (``problem.count_rows``) are passed over. ``exempt``: cells the
    counts alone already pin or narrow, with that spread: such a cell is withheld only when the
    statistics narrow it further (a cell known to be 7-9 that they pin to 8). With
    ``every_range``, each cell is solved both ways and its range recorded.

    Raises:
        UnsolvedError: the solver gave no checked answer in time.
    """
    exempt = exempt or {}
    solver = _Solver(problem, deadline)
    width = len(problem.hidden)
    result = _Pass(witnesses=[problem.truth])
    for k in range(width):
        if exempt.get(k) == 0 and not every_range:
            continue  # pinned by the counts alone: nothing for the statistics to narrow
        unit = _unit(width, k)
        found: list[int] = []
        for sense in (1.0, -1.0):
            if not every_range and k not in exempt and _spread(result.witnesses, unit) > NARROW_SPREAD:
                break  # two checked tables already set it 3 or more apart
            objective = np.zeros(width)
            objective[k] = sense
            best = solver.extreme(objective)
            result.witnesses.append(best)
            found.append(best[k])
        if every_range:
            result.ranges.append((k, min(found), max(found)))
        spread = _spread(result.witnesses, unit)
        # An exempt cell was solved both ways (its exact spread); any other, until set 3 apart.
        limit = exempt[k] - 1 if k in exempt else NARROW_SPREAD
        if spread <= limit and result.verdict == NOT_PINNED:
            result.verdict = "pinned" if spread == 0 else "narrow"
            if not every_range:
                return result
    if result.verdict != NOT_PINNED:
        return result
    _check_unions(problem, solver, result, separable=separable)
    return result


def _check_unions(problem: _Problem, solver: _Solver, result: _Pass, *, separable: bool) -> None:
    """Set ``result.verdict`` to "union_pinned" when a union of two or more hidden cells, with 1
    to threshold-1 patients and not pinned by the counts alone, is pinned or narrow.

    A union is pinned by the counts alone when it is a combination of the count equalities
    (orthogonal to every integer direction they leave free, checked exactly); such a union is
    cut off and the search goes on. ``separable``: every constraint is within one group (this
    answer's own facts), so the solutions are a product over groups and a union's range is the
    sum of its parts' ranges: a narrow union has narrow parts, and each group is searched alone,
    its whole hidden set (its n minus its shown cells) left out.

    Raises:
        UnsolvedError: the solver gave no checked answer in time.
    """
    free = _integer_kernel(problem.count_rows, len(problem.hidden))
    if not free:
        return  # the counts alone pin every union
    groups = sorted({j for _, j in problem.hidden})
    scopes = (
        [[k for k, (_, j) in enumerate(problem.hidden) if j == g] for g in groups]
        if separable
        else [list(range(len(problem.hidden)))]
    )
    for scope in scopes:
        cuts: list[list[int]] = []
        while (union := solver.union_candidate(_sample(result.witnesses), scope, cuts, partial=separable)) is not None:
            if all(sum(d * z for d, z in zip(direction, union, strict=True)) == 0 for direction in free):
                cuts.append(union)  # the counts alone pin it
                continue
            objective = np.array(union, dtype=float)
            low, high = solver.extreme(objective), solver.extreme(-objective)
            result.witnesses += [low, high]
            if _spread([low, high], union) <= NARROW_SPREAD:
                result.verdict = "union_pinned"
                return
            cuts.append(union)  # two checked tables set it 3 or more apart: free for good


def _pinned_by_counts(
    values: list[Fraction], groups: list[Hashable], table: list[list[int]], deadline: float
) -> dict[int, int]:
    """The hidden cells the group sizes and value totals alone pin or narrow, each with the exact
    spread of its counts over those facts (0 to ``NARROW_SPREAD``)."""
    counted = _build(values, groups, table, level=0, totals=True)
    witnesses: list[tuple[int, ...]] = [counted.truth]
    solver = _Solver(counted, deadline)
    width = len(counted.hidden)
    narrow: dict[int, int] = {}
    for k in range(width):
        for sense in (1.0, -1.0):
            if _spread(witnesses, _unit(width, k)) > NARROW_SPREAD:
                break
            objective = np.zeros(width)
            objective[k] = sense
            witnesses.append(solver.extreme(objective))
        if (spread := _spread(witnesses, _unit(width, k))) <= NARROW_SPREAD:
            narrow[k] = spread  # both extremes solved exactly: the exact spread
    return narrow


def _level_verdict(
    values: list[Fraction], groups: list[Hashable], table: list[list[int]], level: int, pooled: int, deadline: float
) -> Verdict:
    """Whether the statistics up to ``level`` pin nothing: by this answer's own facts, then with
    the value totals and pooled statistics (at ``pooled``) for what the statistics newly pin."""
    own = _check(_build(values, groups, table, level=level, totals=False), deadline, separable=True)
    if own.verdict != NOT_PINNED or len(groups) == 1:
        return own.verdict
    exempt = _pinned_by_counts(values, groups, table, deadline)
    full = _check(
        _build(values, groups, table, level=level, totals=True, pooled=pooled), deadline, exempt, separable=False
    )
    if full.verdict == NOT_PINNED:
        return NOT_PINNED
    return "union_pinned" if full.verdict == "union_pinned" else "pinned_with_totals"


def _decide(
    values: list[Fraction], groups: list[Hashable], table: list[list[int]], deadline: float
) -> tuple[int, Verdict]:
    """The highest level (index into ``LEVELS``) that pins nothing, and why the next does not."""
    pooled = len(LEVELS) - 1
    if len(groups) > 1:
        line = [[sum(row)] for row in table]
        if any(0 < n[0] < SMALL_CELL_THRESHOLD for n in line):
            pooled, _ = _decide(values, ["all"], line, deadline)
    # Levels are nested, so a level that pins nothing has no lower level that does: the top one
    # first (most columns that pass pass whole), then the lowest, then the one between.
    top = len(LEVELS) - 1
    verdict = _level_verdict(values, groups, table, top, pooled, deadline)
    if verdict == NOT_PINNED:
        return top, NOT_PINNED
    if (lowest := _level_verdict(values, groups, table, 1, pooled, deadline)) != NOT_PINNED:
        return 0, lowest
    for level in range(top - 1, 1, -1):
        if _level_verdict(values, groups, table, level, pooled, deadline) == NOT_PINNED:
            return level, verdict
    return 1, verdict


def _hidden_count(table: list[list[int]]) -> int:
    return sum(1 for line in table for n in line if 0 < n < SMALL_CELL_THRESHOLD)


def statistics_level(counts: Mapping[tuple[object, Hashable], int]) -> Solvability:
    """Which statistics of a column with these patients per (value, group) may be shown
    (``Solvability.level``; ``LEVEL_STATISTICS`` names them), by the module's model.

    ``counts``: patients per present (value, group) cell, one row per patient (a caller whose
    rows are visits records nothing, and the statistics are withheld). A column with no hidden
    cell shows every level; otherwise each level is checked in turn within ``SOLVE_BUDGET_S``.

    Raises:
        ValueError: a value that is not a finite number, or a count that is not a non-negative
            integer (a caller's bug: the cells are counted, never read from input).
    """
    started = time.monotonic()
    values, groups, table = _table(counts)
    cells, hidden = len(values) * len(groups), _hidden_count(table)
    if not any(n for line in table for n in line):
        raise ValueError("A table with no patient has no statistics to decide")
    if hidden == 0:
        return Solvability(LEVELS[-1], NO_HIDDEN_CELL, 0, cells)
    if cells > MAX_TABLE_CELLS:
        logger.info("hidden_cells_checked", level="none", verdict="too_large", hidden=hidden, cells=cells)
        return Solvability("none", "too_large", hidden, cells)
    key = (tuple(values), tuple(repr(g) for g in groups), tuple(tuple(line) for line in table))
    with _DECIDED_LOCK:
        if (known := _DECIDED.get(key)) is not None:
            _DECIDED.move_to_end(key)
            return known
    try:
        index, verdict = _decide(values, groups, table, started + SOLVE_BUDGET_S)
    except UnsolvedError as reason:
        logger.warning("hidden_cells_unsolved", reason=str(reason), hidden=hidden, cells=cells)
        index, verdict = 0, "unsolved"
    decided = Solvability(LEVELS[index], verdict, hidden, cells)
    if verdict != "unsolved":  # a decision; a timeout under load may go otherwise next time
        with _DECIDED_LOCK:
            _DECIDED[key] = decided
            while len(_DECIDED) > DECIDED_TABLES:
                _DECIDED.popitem(last=False)
    logger.info(
        "hidden_cells_checked",
        level=LEVELS[index],
        verdict=verdict,
        hidden=hidden,
        cells=cells,
        elapsed_ms=round(1000 * (time.monotonic() - started)),
    )
    return decided


def hidden_cell_ranges(counts: Mapping[tuple[object, Hashable], int], level: Level = "median") -> Solvability:
    """Every hidden cell's lowest and highest count given this answer's own facts at ``level``
    (``Solvability.ranges``), each solved both ways: for tests and for explaining a decision.
    ``Solvability.level`` is ``level`` when nothing is pinned or narrow, else "none".

    Raises:
        ValueError: as ``statistics_level``.
        UnsolvedError: the solver gave no checked answer within ``SOLVE_BUDGET_S``.
    """
    values, groups, table = _table(counts)
    cells, hidden = len(values) * len(groups), _hidden_count(table)
    if hidden == 0:
        return Solvability(level, NO_HIDDEN_CELL, 0, cells)
    problem = _build(values, groups, table, level=LEVELS.index(level), totals=False)
    result = _check(problem, time.monotonic() + SOLVE_BUDGET_S, separable=True, every_range=True)
    ranges = tuple(
        CellRange(values[problem.hidden[k][0]], groups[problem.hidden[k][1]], low, high)
        for k, low, high in result.ranges
    )
    return Solvability(level if result.verdict == NOT_PINNED else "none", result.verdict, hidden, cells, ranges)

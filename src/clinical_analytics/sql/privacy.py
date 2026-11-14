# SPDX-License-Identifier: MIT
"""Run a governed query and show only what keeps every count of 1-10 patients hidden.

The query is never run as written. The answer's conditions are every AND term of its WHERE, of
each FILTER and of each count_if, plus ``<column> IS NOT NULL`` for each column a count(column) or
value aggregate reads (ADR013 section 4.5). Each becomes an indicator, ``coalesce(TRY(term),
false)``. One grouped query counts patients per group and per combination of indicators (a bit
mask: 2^k cells per group, at most ``MAX_CONDITIONS`` conditions), split per value of each coded
column a value aggregate reads (at most ``MAX_DISCRETE_LEVELS`` values), over the WHOLE table:
rows the WHERE removes are cells too. Rows are patients (``patient_rows``).

Lines (totals a reader knows or can learn): a group's cells add up to the group; each cell across
groups adds up to the same query without GROUP BY; the step-by-step count (the dataset total,
then each WHERE term in the order written) adds up cells too. ``_suppress_lines`` (primary and
secondary suppression) decides what is hidden. A group's measures are shown only when none of
its cells is hidden; an ungrouped answer with a hidden cell is withheld. Every count a shown
answer reveals about its conditions is then a sum of cells that are each 0 or at least 11, so
dropping any set of filters, differencing FILTERs or count_ifs inside one answer, and a filter
that removes few patients in one group only are all covered.

Not covered (documented residual risk, ``docs/security/THREAT_MODEL.md``): two answers whose
conditions are not nested, or that group differently.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

import structlog
from sqlglot import exp

from clinical_analytics.analysis.grouping import OTHER_LABEL
from clinical_analytics.core.result_presentation import (
    SECONDARY_SUPPRESSED,
    _fold_rows,
    _small_parts,
    _suppress_lines,
    _table,
)
from clinical_analytics.sql import config
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.render import render
from clinical_analytics.sql.sandbox import SandboxConnection
from clinical_analytics.sql.validate import AGG_MARKER, GovernedQuery

logger = structlog.get_logger()


VISIT_ROWS = (
    "Some patient IDs are on more than one row, so rows are visits, not patients. "
    "SQL over visit rows is not supported yet: ask with the question box instead."
)
NO_ID_COLUMN = (
    "This dataset has no patient ID column, so its rows can't be checked to be patients, and hiding "
    "small counts needs counts of patients. If each row is one patient, confirm that and ask again."
)
MISSING_IDS = (
    "Some rows have no {key}, so they can't be checked to be different patients. "
    "If each row is one patient, confirm that and ask again."
)
TOO_MANY_GROUPS = (
    f"The grouping has more than {config.MAX_GROUPS} groups, too many to check for small groups. "
    "Group by something coarser (a range, a year)."
)
TOO_MANY_CONDITIONS = (
    f"This answer has more than {config.MAX_CONDITIONS} conditions (the AND terms of WHERE, FILTER and "
    "count_if, and a not-missing check for each column summarised), and every combination of them is "
    "checked for small groups. Ask with fewer, or combine ranges with BETWEEN or IN."
)
WITHHELD = (
    f"This answer, with its filters, would let a reader work out a count of 1 to {config.SMALL_CELL - 1} "
    "patients. Ask with fewer filters or a coarser grouping."
)
NO_VALUE_LABEL = "(no value)"


class PatientRowsUnconfirmedError(SqlRefusedError):
    """Rows can be counted as patients only once the user confirms one row per patient (no ID
    column, or rows without an ID). ``rows`` is what a confirmation is made on; never shown."""

    def __init__(self, reason: str, *, key: str | None, rows: int) -> None:
        super().__init__(reason)
        self.key = key
        self.rows = rows


@dataclass(frozen=True)
class SqlAnswer:
    sql: str  # the checked query, as shown to the user
    columns: list[str]
    rows: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)


Cell = tuple[int, int, tuple[int, ...]]  # (group, condition bit mask, value index per coded column)


@dataclass(frozen=True)
class _Conditions:
    """The answer's conditions, numbered once (by their rendered text); a set of them is a bit mask."""

    terms: tuple[exp.Expr, ...]
    where: tuple[int, ...]  # the WHERE clause's terms, in the order written
    of_aggregate: tuple[int, ...]  # per aggregate: the mask of its own terms

    @property
    def where_mask(self) -> int:
        return sum(1 << i for i in self.where)

    def indicator(self, i: int) -> exp.Expr:
        """True when the row meets term i; an error or NULL is false (never "not p")."""
        return exp.func("coalesce", exp.Try(this=self.terms[i].copy()), exp.false())

    def predicate(self, mask: int) -> exp.Expr:
        chosen = [self.indicator(i) for i in range(len(self.terms)) if mask >> i & 1]
        return exp.and_(*chosen) if chosen else exp.true()


@dataclass(frozen=True)
class _Coded:
    """Columns read by a value aggregate that have few values: each cell is split per value."""

    columns: tuple[str, ...]
    values: tuple[tuple[Any, ...], ...]


def answer_query(
    query: GovernedQuery,
    sandbox: SandboxConnection,
    catalog: SqlCatalog,
    *,
    confirmed_patient_rows: int | None = None,
) -> SqlAnswer:
    """The query's answer with small cells suppressed and small groups folded.

    ``confirmed_patient_rows``: the row count on which the user confirmed one row per patient for
    this dataset version (None: not confirmed). It is needed only when rows can't be shown to be
    patients by their IDs, and counts only while the table still has that many rows.

    Raises:
        PatientRowsUnconfirmedError: no ID on some or all rows, and no matching confirmation.
        SqlRefusedError: rows are visits, too many groups or conditions, the query could not
            run, or (ungrouped) the answer would let a small count be worked out.
    """
    confirmed = patient_rows(sandbox, catalog, confirmed_patient_rows)
    conditions = _conditions(query)
    coded = _coded(query, sandbox)
    frame = sandbox.run_unsuppressed(_core_sql(query, conditions))
    if frame.height > config.MAX_GROUPS:
        raise SqlRefusedError(TOO_MANY_GROUPS)
    records = frame.to_dicts()
    cells = _cells(query, sandbox, conditions, coded, records)
    lines = _lines(cells, conditions, len(records))
    hidden = _suppress_lines(cells, lines)
    logger.info("governed_sql_answered", groups=frame.height, measures=len(query.measures), hidden=len(hidden))
    if query.dimension is None and hidden:
        raise SqlRefusedError(WITHHELD)
    return _present(query, records, cells, hidden, conditions, confirmed)


def patient_rows(sandbox: SandboxConnection, catalog: SqlCatalog, confirmed_rows: int | None) -> bool:
    """Check that every row is one patient, so every count suppressed is a count of patients.

    Returns False when the patient IDs show it (an ID on every row, none repeated), True when the
    user's confirmation does (``confirmed_rows`` equals the table's row count).

    Raises:
        SqlRefusedError: an ID repeats (rows are visits).
        PatientRowsUnconfirmedError: no ID column, or rows without an ID, and no matching confirmation.
    """
    key = catalog.id_column
    columns = [exp.alias_(exp.Count(this=exp.Star()), "n_rows")]
    if key is not None:
        ref = exp.column(key.name, quoted=True)
        columns += [
            exp.alias_(exp.Count(this=ref.copy()), "n_keyed"),
            exp.alias_(exp.Count(this=exp.Distinct(expressions=[ref.copy()])), "n_patients"),
        ]
    counts = sandbox.run_unsuppressed(render(exp.select(*columns).from_(config.TABLE_NAME))).row(0, named=True)
    rows = int(counts["n_rows"])
    if key is not None:
        if counts["n_patients"] < counts["n_keyed"]:
            raise SqlRefusedError(VISIT_ROWS)
        if counts["n_keyed"] == rows:
            return False
    if confirmed_rows is not None and confirmed_rows == rows:
        return True
    reason = NO_ID_COLUMN if key is None else MISSING_IDS.format(key=key.name)
    raise PatientRowsUnconfirmedError(reason, key=None if key is None else key.name, rows=rows)


def _levels(column: str, sandbox: SandboxConnection) -> list[Any] | None:
    """The column's distinct values when there are few enough to count each, else None.

    Raises:
        SqlRefusedError: NaN or infinite values, which cannot be counted per value.
    """
    ref = exp.column(column, quoted=True)
    probe = (
        exp.select(exp.alias_(ref.copy(), "v"))
        .distinct()
        .from_(config.TABLE_NAME)
        .where(exp.Not(this=exp.Is(this=ref.copy(), expression=exp.Null())))
        .limit(config.MAX_DISCRETE_LEVELS + 1)
    )
    values = sandbox.run_unsuppressed(render(probe)).get_column("v").to_list()
    if len(values) > config.MAX_DISCRETE_LEVELS:
        return None
    if any(isinstance(v, float) and not math.isfinite(v) for v in values):
        raise SqlRefusedError(f"{column} holds NaN or infinite values, so it cannot be summarised.")
    return sorted(values)


def _conjuncts(expression: exp.Expr | None) -> list[exp.Expr]:
    """The top-level AND terms of a predicate (none for no predicate)."""
    if expression is None:
        return []
    node = expression
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _not_null(column: str) -> exp.Expr:
    return exp.Not(this=exp.Is(this=exp.column(column, quoted=True), expression=exp.Null()))


def _conditions(query: GovernedQuery) -> _Conditions:
    """Number every condition of the answer once.

    Raises:
        SqlRefusedError: more than ``MAX_CONDITIONS`` (never checked in part).
    """
    terms: list[exp.Expr] = []
    numbered: dict[str, int] = {}

    def number(term: exp.Expr) -> int:
        key = render(term)
        if key not in numbered:
            numbered[key] = len(terms)
            terms.append(term.copy())
        return numbered[key]

    where = tuple(dict.fromkeys(number(t) for t in _conjuncts(query.where)))
    own: list[int] = []
    for agg in query.aggregates:
        mask = 0
        for term in _conjuncts(agg.condition) + ([_not_null(agg.column)] if agg.column is not None else []):
            mask |= 1 << number(term)
        own.append(mask)
    if len(terms) > config.MAX_CONDITIONS:
        raise SqlRefusedError(TOO_MANY_CONDITIONS)
    return _Conditions(tuple(terms), where, tuple(own))


def _coded(query: GovernedQuery, sandbox: SandboxConnection) -> _Coded:
    columns: list[str] = []
    values: list[tuple[Any, ...]] = []
    for agg in query.aggregates:
        if agg.kind != "value" or agg.column is None or agg.column in columns:
            continue
        found = _levels(agg.column, sandbox)
        if found is not None:
            columns.append(agg.column)
            values.append(tuple(found))
    return _Coded(tuple(columns), tuple(values))


def _try(expression: exp.Expr | None) -> exp.Expr:
    """A row-level expression that never fails: an error becomes NULL (a predicate then excludes
    the row, as it would any other). Without it, CASE WHEN <one patient> THEN CAST('x' AS INT)
    would answer "does this patient exist?" by failing or not."""
    return exp.true() if expression is None else exp.Try(this=expression.copy())


def _core_sql(query: GovernedQuery, conditions: _Conditions) -> str:
    """The measures per group, each aggregate over the rows meeting the WHERE's and its own terms."""
    dimension = _try(query.dimension) if query.dimension is not None else None
    rewritten = [
        exp.Filter(
            this=agg.function.copy(),
            expression=exp.Where(this=conditions.predicate(conditions.where_mask | conditions.of_aggregate[k])),
        )
        for k, agg in enumerate(query.aggregates)
    ]

    def measure(template: exp.Expr) -> exp.Expr:
        if isinstance(template, exp.Column):  # one aggregate as it is: a count stays a whole number
            k = int(template.name.removeprefix(AGG_MARKER))
            if query.aggregates[k].kind == "count":
                return rewritten[k].copy()
            # sum() of an integer is HUGEINT, which arrives as a Decimal and would be sent as text.
            return exp.Cast(this=rewritten[k].copy(), to=exp.DataType.build("DOUBLE"))

        def fill(node: exp.Expr) -> exp.Expr:
            if isinstance(node, exp.Column) and node.name.startswith(AGG_MARKER):
                aggregate = rewritten[int(node.name.removeprefix(AGG_MARKER))].copy()
                return exp.Cast(this=aggregate, to=exp.DataType.build("DOUBLE"))  # no overflow error
            return node

        return template.transform(fill)

    columns: list[exp.Expr] = []
    if dimension is not None:
        columns.append(exp.alias_(dimension.copy(), "__d"))
    columns += [exp.alias_(measure(m.template), f"__m{i}") for i, m in enumerate(query.measures)]
    select = exp.select(*columns).from_(config.TABLE_NAME)
    if dimension is not None:
        select = select.group_by(dimension.copy())
    wrapped = exp.select("*").from_(select.subquery("__core")).limit(config.MAX_GROUPS + 1)
    return render(wrapped)


def _group_key(value: Any) -> Any:
    """A group value as a dict key (NaN never equals itself)."""
    return "__nan__" if isinstance(value, float) and math.isnan(value) else value


def _cells(
    query: GovernedQuery,
    sandbox: SandboxConnection,
    conditions: _Conditions,
    coded: _Coded,
    records: list[dict[str, Any]],
) -> dict[Cell, int]:
    """Patients per group, per combination of conditions and per value of each coded column."""
    keys: list[exp.Expr] = []
    if query.dimension is not None:
        keys.append(exp.alias_(_try(query.dimension), "__d"))
    mask: exp.Expr = exp.Literal.number(0)
    for i in range(len(conditions.terms)):
        bit = exp.Mul(
            this=exp.Cast(this=conditions.indicator(i), to=exp.DataType.build("INTEGER")),
            expression=exp.Literal.number(1 << i),
        )
        mask = exp.Add(this=mask, expression=bit)
    keys.append(exp.alias_(mask, "__k"))
    for c, (column, values) in enumerate(zip(coded.columns, coded.values, strict=True)):
        case = exp.case()
        ref = exp.column(column, quoted=True)
        for j, value in enumerate(values):
            case = case.when(exp.EQ(this=ref.copy(), expression=exp.convert(value)), exp.Literal.number(j))
        keys.append(exp.alias_(case.else_(exp.Literal.number(-1)), f"__v{c}"))
    select = exp.select(*keys, exp.alias_(exp.Count(this=exp.Star()), "__n")).from_(config.TABLE_NAME)
    select = select.group_by(*[exp.column(k.alias) for k in keys])
    index = {_group_key(record.get("__d")): g for g, record in enumerate(records)}
    cells: dict[Cell, int] = {}
    for row in sandbox.run_unsuppressed(render(select)).iter_rows(named=True):
        g = index.get(_group_key(row["__d"])) if query.dimension is not None else 0
        if g is None:  # both queries group by the same TRY(expression): never, so fail closed
            raise SqlRefusedError("The query could not run on this dataset.")
        levels = tuple(int(row[f"__v{c}"]) for c in range(len(coded.columns)))
        cells[(g, int(row["__k"]), levels)] = int(row["__n"])
    return cells


def _steps(conditions: _Conditions) -> list[int]:
    """The step-by-step count's masks: no condition, then each WHERE term added in order."""
    masks, mask = [0], 0
    for i in conditions.where:
        mask |= 1 << i
        masks.append(mask)
    return masks


def _lines(cells: dict[Cell, int], conditions: _Conditions, groups: int) -> list[list[Cell]]:
    """A group's cells (total: the group); each cell across groups (total: the query without
    GROUP BY); the cells of each step of the step-by-step count (total: that step)."""
    by_group: dict[int, list[Cell]] = {}
    by_key: dict[tuple[int, tuple[int, ...]], list[Cell]] = {}
    for cell in cells:
        by_group.setdefault(cell[0], []).append(cell)
        by_key.setdefault((cell[1], cell[2]), []).append(cell)
    lines = [*by_group.values(), *by_key.values()]
    lines += [[c for c in cells if c[1] & step == step] for step in _steps(conditions)]
    return lines


def _label(value: Any) -> str:
    return NO_VALUE_LABEL if value is None else str(value)


def _present(
    query: GovernedQuery,
    records: list[dict[str, Any]],
    cells: dict[Cell, int],
    hidden: set[Any],
    conditions: _Conditions,
    confirmed: bool,
) -> SqlAnswer:
    grouped = query.dimension is not None
    min_values = config.SMALL_CELL
    where = conditions.where_mask
    base = {g: 0 for g in range(len(records))}
    hidden_groups_at = set()
    for (g, mask, _levels_of), count in cells.items():
        if mask & where == where:
            base[g] += count
    for g, mask, _levels_of in hidden:
        hidden_groups_at.add(g)

    def shown(g: int, measure_index: int) -> bool:
        return g not in hidden_groups_at  # a group's measures need every one of its cells shown

    measure_names = [m.name for m in query.measures]
    columns = ([query.dimension_name or ""] if grouped else []) + measure_names
    displayed = [g for g in range(len(records)) if not grouped or base[g] > 0]
    rows: list[dict[str, Any]] = []
    raw: dict[int, Any] = {}
    withheld = False
    for g in displayed:
        row: dict[str, Any] = {}
        if grouped:
            assert query.dimension_name is not None
            row[query.dimension_name] = _label(records[g]["__d"])
        for i, name in enumerate(measure_names):
            visible = shown(g, i)
            withheld |= not visible
            row[name] = records[g][f"__m{i}"] if visible else SECONDARY_SUPPRESSED
        raw[id(row)] = records[g].get("__d")
        rows.append(row)

    rows = _ordered(rows, query, raw)
    notes = [
        "Counts are patients (one row per patient, as you confirmed for this dataset)."
        if confirmed
        else "Counts are patients (one row per patient)."
    ]
    table = _table(rows)
    if grouped and table is not None:
        assert query.dimension_name is not None
        sizes: dict[Hashable, Any] = {_label(records[g]["__d"]): base[g] for g in displayed}
        # The group's count is hidden when a cell it adds up (the rows the WHERE keeps) is hidden.
        hidden_groups = {
            _label(records[g]["__d"]) for g in displayed if any(h[0] == g and h[1] & where == where for h in hidden)
        }
        fold = _small_parts(sizes, hidden_groups)
        folded = _fold_rows(table, set(fold), fold.label)
        assert folded is not None
        table = folded
        if fold:
            notes.append(
                f"Groups with fewer than {min_values} patients are combined in one row, with no numbers."
                if fold.label == OTHER_LABEL
                else "Some groups are combined in one row with no numbers, so small groups can't be worked out."
            )
        table = _limited(table, query.limit, fold.label if fold else None)
    elif table is not None and query.limit is not None:
        table = {"columns": table["columns"], "rows": table["rows"][: query.limit]}
    steps = _step_note(cells, hidden, conditions)
    if steps is not None:
        notes.append(steps)
    if withheld:
        notes.append(
            f'Values marked "{SECONDARY_SUPPRESSED}" are hidden: showing them would show, or let a reader '
            f"work out, a count of 1 to {min_values - 1} patients."
        )
    return SqlAnswer(
        sql=query.shown_sql,
        columns=table["columns"] if table is not None else columns,
        rows=table["rows"] if table is not None else [],
        notes=notes,
    )


def _step_note(cells: dict[Cell, int], hidden: set[Any], conditions: _Conditions) -> str | None:
    """The step-by-step count, CONSORT style: the patients, then each WHERE term in order. A count
    is hidden when any cell it adds up is hidden (the dataset total is shown by the app anyway)."""
    if not conditions.where:
        return None
    parts = []
    for n, step in enumerate(_steps(conditions)):
        summed = [c for c in cells if c[1] & step == step]
        count = sum(cells[c] for c in summed)
        text = SECONDARY_SUPPRESSED if n and any(c in hidden for c in summed) else f"{count:,}"
        label = "patients" if n == 0 else f"with {render(conditions.terms[conditions.where[n - 1]])}"
        parts.append(f"{text} {label}")
    return "Step by step: " + " → ".join(parts) + "."


def _ordered(rows: list[dict[str, Any]], query: GovernedQuery, raw: dict[int, Any]) -> list[dict[str, Any]]:
    """ORDER BY over the shown values (hidden ones last, whatever their true value); without it,
    by the group's value. Never the engine's order, which could follow the hidden counts: the
    group's value breaks every tie, hidden values included."""
    keys = [(k.column, k.descending) for k in query.order]
    if query.dimension_name is not None:
        keys.append((query.dimension_name, False))

    def value(row: dict[str, Any], column: str) -> Any:
        if column == query.dimension_name:
            return raw[id(row)]
        found = row.get(column)
        return None if found == SECONDARY_SUPPRESSED else found

    def compare(a: dict[str, Any], b: dict[str, Any]) -> int:
        for column, descending in keys:
            x, y = value(a, column), value(b, column)
            if (x is None) != (y is None):
                return 1 if x is None else -1
            if x is None or x == y:
                continue
            try:
                result = -1 if x < y else 1
            except TypeError:
                result = -1 if str(x) < str(y) else 1
            return -result if descending else result
        return 0

    return sorted(rows, key=functools.cmp_to_key(compare))


def _limited(table: Any, limit: int | None, fold_label: str | None) -> Any:
    if limit is None:
        return table
    label_col = table["columns"][0]
    kept = [r for r in table["rows"] if fold_label is None or r.get(label_col) != fold_label]
    other = [r for r in table["rows"] if fold_label is not None and r.get(label_col) == fold_label]
    return {"columns": table["columns"], "rows": kept[:limit] + other}

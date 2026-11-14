# SPDX-License-Identifier: MIT
"""Run a governed query and show only what keeps every count of 1-10 patients hidden.

The query is never run as written. One generated query computes, per group, the user's measures
and the patient counts behind them, over the WHOLE table: the WHERE clause becomes a FILTER, so
the same pass counts the patients the WHERE removes. Those counts are cells, and the totals a
reader can get by running the query again without its WHERE or without its GROUP BY are lines.
``core.result_presentation._suppress_lines`` (primary and secondary suppression) then decides what
is hidden, and ``_small_parts`` / ``_fold_rows`` fold small groups into one row, as on the intent
path.

Cells, per group g (rows are patients: the dataset must be one row per patient, shown by a
patient ID on every row that never repeats, or confirmed by the user for this dataset version
when there is no ID; see ``patient_rows``):

* base: patients in g kept by WHERE (w) and removed by it (o);
* per aggregate, its partition of g: the rows it reads (``FILTER``, ``count_if``, non-null
  column), split per value when the column has at most ``MAX_DISCRETE_LEVELS`` values (so avg
  and stddev of a coded column cannot reveal a value held by 3 patients), and the rest; each part
  counted under w and under o.

Lines (totals a reader knows or can learn): the parts of each partition within g (total: g's
count), w + o of each part (total: the query without WHERE), each kind of cell across groups
(total: the query without GROUP BY), and w + o across groups.

A measure is shown for g only when g's own count is shown and no cell of its aggregates'
partitions is hidden; a value aggregate (avg, median, ...) also needs at least 11 patients with a
value. So a difference with the same query without its WHERE, a FILTER that drops 3 patients, or
a count_if that is 1-10 never shows.

Not covered (documented residual risk): differences between two queries with different WHERE
clauses (``count(A)`` then ``count(A and B)``), which no single answer can detect.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Literal

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
from clinical_analytics.sql.validate import AGG_MARKER, Aggregate, GovernedQuery

logger = structlog.get_logger()

Row = Literal["w", "o"]  # kept by the WHERE clause / removed by it
CellKey = tuple[int, int, Row, int | str]  # (group, partition (-1: base), row, part)

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
NO_VALUE_LABEL = "(no value)"
_REST = "rest"


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


@dataclass(frozen=True)
class _Partition:
    parts: tuple[exp.Expr, ...]  # disjoint predicates; the rest of the group is implied


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
        SqlRefusedError: rows are visits, too many groups, or the query could not run.
    """
    confirmed = patient_rows(sandbox, catalog, confirmed_patient_rows)
    partitions, part_of = _partitions(query, sandbox)
    frame = sandbox.run_unsuppressed(_core_sql(query, partitions))
    if frame.height > config.MAX_GROUPS:
        raise SqlRefusedError(TOO_MANY_GROUPS)
    records = frame.to_dicts()
    cells, lines = _cells_and_lines(records, partitions)
    hidden = _suppress_lines(cells, lines)
    logger.info("governed_sql_answered", groups=frame.height, measures=len(query.measures), hidden=len(hidden))
    return _present(query, records, cells, hidden, partitions, part_of, confirmed)


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


def _partitions(query: GovernedQuery, sandbox: SandboxConnection) -> tuple[list[_Partition], list[int]]:
    """One partition per distinct set of parts, and each aggregate's partition."""
    levels: dict[str, list[Any] | None] = {}
    partitions: list[_Partition] = []
    index: dict[tuple[str, ...], int] = {}
    part_of: list[int] = []
    for agg in query.aggregates:
        parts = _parts(agg, levels, sandbox)
        key = tuple(render(p) for p in parts)
        if key not in index:
            index[key] = len(partitions)
            partitions.append(_Partition(parts))
        part_of.append(index[key])
    return partitions, part_of


def _parts(agg: Aggregate, levels: dict[str, list[Any] | None], sandbox: SandboxConnection) -> tuple[exp.Expr, ...]:
    condition = _try(agg.condition)
    if agg.column is None:
        return (condition.copy(),)
    ref = exp.column(agg.column, quoted=True)
    if agg.kind == "value":
        if agg.column not in levels:
            levels[agg.column] = _levels(agg.column, sandbox)
        values = levels[agg.column]
        if values is not None:
            return tuple(_and(condition, exp.EQ(this=ref.copy(), expression=exp.convert(v))) for v in values)
    return (_and(condition, exp.Not(this=exp.Is(this=ref.copy(), expression=exp.Null()))),)


def _try(expression: exp.Expr | None) -> exp.Expr:
    """A row-level expression that never fails: an error becomes NULL (a predicate then excludes
    the row, as it would any other). Without it, CASE WHEN <one patient> THEN CAST('x' AS INT)
    would answer "does this patient exist?" by failing or not."""
    return exp.true() if expression is None else exp.Try(this=expression.copy())


def _and(left: exp.Expr, right: exp.Expr) -> exp.Expr:
    return exp.and_(exp.paren(left.copy()), exp.paren(right.copy()))


def _count_where(predicate: exp.Expr) -> exp.Expr:
    return exp.Filter(this=exp.Count(this=exp.Star()), expression=exp.Where(this=predicate))


def _core_sql(query: GovernedQuery, partitions: list[_Partition]) -> str:
    """Measures and every cell, per group, over the whole table (WHERE as FILTER)."""
    where = _try(query.where)
    dimension = _try(query.dimension) if query.dimension is not None else None
    rewritten = [
        exp.Filter(this=agg.function.copy(), expression=exp.Where(this=_and(where, _try(agg.condition))))
        for agg in query.aggregates
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
    columns += [exp.alias_(exp.Count(this=exp.Star()), "__all"), exp.alias_(_count_where(where.copy()), "__n")]
    for p, partition in enumerate(partitions):
        for j, part in enumerate(partition.parts):
            columns.append(exp.alias_(_count_where(_and(where, part)), f"__w{p}_{j}"))
            columns.append(exp.alias_(_count_where(part.copy()), f"__a{p}_{j}"))
    select = exp.select(*columns).from_(config.TABLE_NAME)
    if dimension is not None:
        select = select.group_by(dimension.copy())
    wrapped = exp.select("*").from_(select.subquery("__core")).limit(config.MAX_GROUPS + 1)
    return render(wrapped)


def _cells_and_lines(
    records: list[dict[str, Any]], partitions: list[_Partition]
) -> tuple[dict[CellKey, int], list[list[CellKey]]]:
    cells: dict[CellKey, int] = {}
    parts_of: dict[int, list[int | str]] = {-1: [0]}
    for p, partition in enumerate(partitions):
        parts_of[p] = [*range(len(partition.parts)), _REST]
    for g, record in enumerate(records):
        n = int(record["__n"])
        o = int(record["__all"]) - n
        cells[(g, -1, "w", 0)], cells[(g, -1, "o", 0)] = n, o
        for p, partition in enumerate(partitions):
            kept = [int(record[f"__w{p}_{j}"]) for j in range(len(partition.parts))]
            removed = [int(record[f"__a{p}_{j}"]) - k for j, k in enumerate(kept)]
            for j, (k, r) in enumerate(zip(kept, removed, strict=True)):
                cells[(g, p, "w", j)], cells[(g, p, "o", j)] = k, r
            cells[(g, p, "w", _REST)], cells[(g, p, "o", _REST)] = n - sum(kept), o - sum(removed)

    rows: tuple[Row, Row] = ("w", "o")
    groups = range(len(records))
    lines: list[list[CellKey]] = []
    for g in groups:
        for p, parts in parts_of.items():
            if p >= 0:
                lines += [[(g, p, r, j) for j in parts] for r in rows]  # the partition of g's w / o
            lines += [[(g, p, "w", j), (g, p, "o", j)] for j in parts]  # same query without WHERE
    for p, parts in parts_of.items():
        for part in parts:
            lines += [[(g, p, r, part) for g in groups] for r in rows]  # same query without GROUP BY
            lines.append([(g, p, r, part) for g in groups for r in rows])  # without either
    return cells, lines


def _label(value: Any) -> str:
    return NO_VALUE_LABEL if value is None else str(value)


def _present(
    query: GovernedQuery,
    records: list[dict[str, Any]],
    cells: dict[CellKey, int],
    hidden: set[Any],
    partitions: list[_Partition],
    part_of: list[int],
    confirmed: bool,
) -> SqlAnswer:
    grouped = query.dimension is not None
    min_values = config.SMALL_CELL

    def partition_hidden(g: int, p: int) -> bool:
        parts: list[int | str] = [*range(len(partitions[p].parts)), _REST]
        return any((g, p, r, j) in hidden for r in ("w", "o") for j in parts)

    def shown(g: int, measure_index: int) -> bool:
        if (g, -1, "w", 0) in hidden:
            return False
        for k in query.measures[measure_index].aggregates:
            p = part_of[k]
            if partition_hidden(g, p):
                return False
            if query.aggregates[k].kind == "value":
                with_value = sum(cells[(g, p, "w", j)] for j in range(len(partitions[p].parts)))
                if 0 < with_value < min_values:
                    return False
        return True

    measure_names = [m.name for m in query.measures]
    columns = ([query.dimension_name or ""] if grouped else []) + measure_names
    displayed = [g for g in range(len(records)) if not grouped or cells[(g, -1, "w", 0)] > 0]
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
        sizes: dict[Hashable, Any] = {_label(records[g]["__d"]): cells[(g, -1, "w", 0)] for g in displayed}
        hidden_groups = {_label(records[g]["__d"]) for g in displayed if (g, -1, "w", 0) in hidden}
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

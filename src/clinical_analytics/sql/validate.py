# SPDX-License-Identifier: MIT
"""Static checks of one user-typed query against the dataset's contract.

Fail closed: the query is parsed by sqlglot and every node must be of an allowed type, so a
construct this module does not know is refused, never passed through. What runs is SQL that this
package generates from the checked tree, never the user's text.

The one shape admitted (v1)::

    SELECT [<dimension>,] <measure>, ... FROM unified
    [WHERE <predicate>] [GROUP BY <dimension>] [ORDER BY <output column>] [LIMIT n]

* at most one dimension: a row-level expression of the contract's columns, never the patient
  identifier, and a PHI column only as its year, quarter or month: the date column itself (or a
  CAST of it to a date or timestamp) as the direct argument of year(), quarter(), month(),
  EXTRACT or date_trunc with a unit of year, quarter or month. Anything computed from a PHI
  column before it reaches the coarsener (string building, date arithmetic, CASE, datediff)
  could encode the full value in a year, so it is refused;
* every other output is a measure: arithmetic over aggregates (count, count_if, sum, avg, median,
  quantile_cont, stddev, variance), each with an optional FILTER; a value aggregate reads one bare
  numeric column, so its argument cannot encode a count (``avg(1000 * (x > 5) + x / 1e6)``);
* no joins, subqueries, CTEs, set operations, windows, HAVING, QUALIFY, DISTINCT, GROUPING SETS,
  table functions, MIN/MAX or list-valued aggregates: each either lists patients or lets a
  small group be worked out from what is shown (see ``sql/privacy.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import duckdb
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from clinical_analytics.sql import config
from clinical_analytics.sql.catalog import ColumnSpec, SqlCatalog
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.render import render

AggregateKind = Literal["count", "value"]

AGG_MARKER = "__agg"  # placeholder column for aggregate k in a measure template: __agg{k}

ONE_PATIENT = "A question about one patient can't be answered: the patient ID can only be counted."
LISTS_PATIENTS = "This would list individual patients. Ask for counts or summaries, grouped if you like."
NOT_IN_GROUP = "Every output must be the GROUP BY column or a summary (count, avg, ...) of the rows."
ONE_TABLE = (
    f"Only the table {config.TABLE_NAME} can be read, on its own: joins, subqueries and WITH are not supported yet."
)
COMBINED_ROWS = (
    "Combined or total rows (UNION, ROLLUP, GROUPING SETS) can reveal small groups, so they are not supported."
)
HAVING_REFUSAL = (
    "HAVING, QUALIFY and window functions (rank, lag, ...) are not supported: filtering or ranking groups "
    "before small groups are hidden would reveal them. Sort the result with ORDER BY instead."
)
MIN_MAX_REFUSAL = "MIN and MAX show one patient's value. Use median(...) or quantile_cont(column, 0.1 or 0.9)."

_SELECT_ARGS = frozenset({"expressions", "from_", "where", "group", "order", "limit"})

_SCALAR_NODES: tuple[type[exp.Expr], ...] = (
    exp.Column, exp.Identifier, exp.Literal, exp.Boolean, exp.Null, exp.Paren,
    exp.Add, exp.Sub, exp.Mul, exp.Div, exp.IntDiv, exp.Mod, exp.Neg,
    exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Is, exp.In, exp.Between,
    exp.Like, exp.ILike, exp.And, exp.Or, exp.Not, exp.Case, exp.If,
    exp.Cast, exp.TryCast, exp.DataType, exp.Coalesce, exp.Nullif,
    exp.Round, exp.Floor, exp.Ceil, exp.Abs, exp.Lower, exp.Upper, exp.Trim, exp.Length, exp.Substring,
    exp.Year, exp.Quarter, exp.Month, exp.Extract, exp.DateDiff, exp.TimestampTrunc, exp.DateTrunc,
    exp.Greatest, exp.Least, exp.StartsWith, exp.Contains, exp.DPipe,
)  # fmt: skip
# Around aggregates in a measure: arithmetic and rounding only, on DOUBLE (sql.privacy casts
# each aggregate), so no value can make it fail. CAST is left out: CAST(1e300 AS INT) fails, and a
# failure that depends on a count would answer "is it 4 or more?" without showing the count.
_MEASURE_NODES: tuple[type[exp.Expr], ...] = (
    exp.Literal, exp.Paren, exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Neg,
    exp.Round, exp.TryCast, exp.DataType, exp.Nullif, exp.Coalesce,
)  # fmt: skip
_VALUE_AGGS: tuple[type[exp.Expr], ...] = (
    exp.Sum, exp.Avg, exp.Median, exp.PercentileCont, exp.Stddev, exp.StddevSamp, exp.Variance,
)  # fmt: skip
_COUNT_AGGS: tuple[type[exp.Expr], ...] = (exp.Count, exp.CountIf)
_UNIT_PARENTS: tuple[type[exp.Expr], ...] = (exp.Extract, exp.DateDiff, exp.TimestampTrunc, exp.DateTrunc)
_COARSE_UNITS = frozenset({"YEAR", "QUARTER", "MONTH"})
_COARSENERS: tuple[type[exp.Expr], ...] = (exp.Year, exp.Quarter, exp.Month)
_TRUNCS: tuple[type[exp.Expr], ...] = (exp.TimestampTrunc, exp.DateTrunc)
# A PHI column is coarsened only when it holds calendar values: a date or timestamp column, or a
# string column CAST to one (a number cannot be cast to a date, and year() of it fails).
_CALENDAR_TYPES = frozenset({"date", "timestamp"})
_CALENDAR_CASTS: tuple[exp.DType, ...] = (
    exp.DataType.Type.DATE, exp.DataType.Type.DATETIME, exp.DataType.Type.TIMESTAMP,
    exp.DataType.Type.TIMESTAMPTZ, exp.DataType.Type.TIMESTAMPNTZ, exp.DataType.Type.TIMESTAMPLTZ,
    exp.DataType.Type.TIMESTAMP_S, exp.DataType.Type.TIMESTAMP_MS, exp.DataType.Type.TIMESTAMP_NS,
)  # fmt: skip


@dataclass(frozen=True)
class Aggregate:
    kind: AggregateKind
    function: exp.Expr  # the aggregate without FILTER, columns canonical
    column: str | None  # value aggregates: the column; count(col): the counted column
    condition: exp.Expr | None  # FILTER and count_if predicates, ANDed


@dataclass(frozen=True)
class Measure:
    name: str
    template: exp.Expr  # aggregates replaced by column __agg{k}
    aggregates: tuple[int, ...]


@dataclass(frozen=True)
class OrderKey:
    column: str  # an output column
    descending: bool


@dataclass(frozen=True)
class GovernedQuery:
    """A query that passed every static check, ready for ``sql.privacy``."""

    shown_sql: str  # the checked query, generated from the tree (what the user is shown)
    dimension: exp.Expr | None
    dimension_name: str | None
    measures: tuple[Measure, ...]
    aggregates: tuple[Aggregate, ...]
    where: exp.Expr | None
    order: tuple[OrderKey, ...]
    limit: int | None
    output_columns: tuple[str, ...]


def validate_sql(sql: str, catalog: SqlCatalog) -> GovernedQuery:
    """Check ``sql`` and return the governed form of it.

    Raises:
        SqlRefusedError: with the plain-language reason, for anything outside the admitted shape.
    """
    tree = _parse(sql)
    _check_select_shape(tree)
    tree = _canonical_columns(tree, catalog)
    _check_nodes(tree)

    where_node = tree.args.get("where")
    where = where_node.this if where_node is not None else None
    if where is not None:
        _check_scalar(where, catalog, "WHERE")

    projections = list(tree.expressions)
    names = [_output_name(p) for p in projections]
    if len(set(n.casefold() for n in names)) != len(names):
        raise SqlRefusedError("Two outputs have the same name: give each one its own alias with AS.")
    bodies = [p.this if isinstance(p, exp.Alias) else p for p in projections]

    dimension = _dimension(tree, bodies, names)
    if dimension is not None:
        _check_scalar(dimension, catalog, "GROUP BY")
        _check_dimension_privacy(dimension, catalog)

    aggregates: list[Aggregate] = []
    measures: list[Measure] = []
    dimension_name: str | None = None
    for body, name in zip(bodies, names, strict=True):
        if dimension is not None and body == dimension:
            if dimension_name is not None:
                raise SqlRefusedError("Show the GROUP BY column once.")
            dimension_name = name
            continue
        measures.append(_measure(body, name, catalog, aggregates, grouped=dimension is not None))
    if dimension is not None and dimension_name is None:
        raise SqlRefusedError("Show the GROUP BY column in the SELECT list, so each row says which group it is.")
    if not measures:
        raise SqlRefusedError(LISTS_PATIENTS)

    return GovernedQuery(
        shown_sql=render(tree, pretty=True),
        dimension=dimension,
        dimension_name=dimension_name,
        measures=tuple(measures),
        aggregates=tuple(aggregates),
        where=where,
        order=_order(tree, bodies, names),
        limit=_limit(tree),
        output_columns=tuple(names),
    )


def _parse(sql: str) -> exp.Select:
    if not sql.strip():
        raise SqlRefusedError("The query is empty.")
    if len(sql) > config.MAX_SQL_CHARS:
        raise SqlRefusedError(f"The query is longer than {config.MAX_SQL_CHARS} characters.")
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as e:
        raise SqlRefusedError("The query is not valid SQL.") from e
    if len(statements) != 1:
        raise SqlRefusedError("Send one SELECT statement at a time.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise SqlRefusedError("Only SELECT queries can run: the data is read-only.")
    try:
        trees = sqlglot.parse(sql, dialect="duckdb")
    except ParseError as e:
        raise SqlRefusedError("The query is not valid SQL.") from e
    if len(trees) != 1 or trees[0] is None:
        raise SqlRefusedError("Send one SELECT statement at a time.")
    tree = trees[0]
    if isinstance(tree, exp.SetOperation):
        raise SqlRefusedError(COMBINED_ROWS)
    if not isinstance(tree, exp.Select):
        raise SqlRefusedError("Only SELECT queries can run: the data is read-only.")
    if sum(1 for _ in tree.walk()) > config.MAX_AST_NODES:
        raise SqlRefusedError("The query is too long to check. Split it into smaller questions.")
    for node in tree.walk():
        node.comments = None  # generated SQL carries no user text outside literals and names
    return tree


def _check_select_shape(tree: exp.Select) -> None:
    used = {k for k, v in tree.args.items() if v not in (None, [], False)}
    if used & {"having", "qualify", "windows"}:
        raise SqlRefusedError(HAVING_REFUSAL)
    if used & {"with_", "with", "joins", "laterals"}:
        raise SqlRefusedError(ONE_TABLE)
    if "distinct" in used:
        raise SqlRefusedError("SELECT DISTINCT lists values row by row. Use GROUP BY with a count instead.")
    extra = used - _SELECT_ARGS
    if extra:
        raise SqlRefusedError(f"This part of the query is not supported: {', '.join(sorted(extra))}.")
    if any(isinstance(n, exp.Subquery | exp.Select) and n is not tree for n in tree.walk()):
        raise SqlRefusedError(ONE_TABLE)
    if any(isinstance(n, exp.Window) for n in tree.walk()):
        raise SqlRefusedError(HAVING_REFUSAL)
    if any(isinstance(n, exp.Rollup | exp.Cube | exp.GroupingSets) for n in tree.walk()):
        raise SqlRefusedError(COMBINED_ROWS)
    if any(isinstance(n, exp.Min | exp.Max) for n in tree.walk()):
        raise SqlRefusedError(MIN_MAX_REFUSAL)
    source = tree.args.get("from_")
    table = source.this if source is not None else None
    if not isinstance(table, exp.Table) or not isinstance(table.this, exp.Identifier):
        raise SqlRefusedError(f"Read from the table {config.TABLE_NAME}.")
    if table.args.get("db") or table.args.get("catalog") or table.name.casefold() != config.TABLE_NAME:
        raise SqlRefusedError(f"Only the table {config.TABLE_NAME} can be read.")
    if any(v for k, v in table.args.items() if k not in ("this", "alias")):
        raise SqlRefusedError(f"Read the table {config.TABLE_NAME} as it is (no sampling or options).")
    group = tree.args.get("group")
    if group is not None and (set(k for k, v in group.args.items() if v) - {"expressions"}):
        raise SqlRefusedError("Name the GROUP BY column (GROUP BY ALL, ROLLUP and CUBE are not supported).")


def _canonical_columns(tree: exp.Select, catalog: SqlCatalog) -> exp.Select:
    """Every column resolved to its contract name, quoted and unqualified."""
    table = tree.args["from_"].this
    alias = table.alias_or_name.casefold()
    aliases = {_output_name(p).casefold() for p in tree.expressions if isinstance(p, exp.Alias)}

    def resolve(node: exp.Expr) -> exp.Expr:
        if isinstance(node, exp.Identifier) and node.name.startswith("__"):
            raise SqlRefusedError("Names starting with __ are reserved.")
        if not isinstance(node, exp.Column):
            return node
        if isinstance(node.this, exp.Star):
            raise SqlRefusedError(LISTS_PATIENTS)
        if node.args.get("db") or node.args.get("catalog"):
            raise SqlRefusedError(f"Columns are read from the table {config.TABLE_NAME} only.")
        qualifier = node.table.casefold()
        if qualifier and qualifier not in (alias, config.TABLE_NAME):
            raise SqlRefusedError(f"Columns are read from the table {config.TABLE_NAME} only.")
        if not qualifier and node.name.casefold() in aliases and isinstance(node.parent, exp.Ordered):
            return node  # ORDER BY <name>: the output alias wins over a column of that name, as in SQL
        spec = catalog.column(node.name)
        if spec is None:
            if not qualifier and node.name.casefold() in aliases and _in_order_or_group(node):
                return node  # an output alias in ORDER BY / GROUP BY, resolved there
            raise SqlRefusedError(f"There is no column named {node.name} in this dataset's contract.")
        return exp.column(spec.name, quoted=True)

    resolved = tree.transform(resolve)
    assert isinstance(resolved, exp.Select)  # the root is never a column
    resolved.args["from_"].set("this", exp.to_table(config.TABLE_NAME))
    return resolved


def _in_order_or_group(node: exp.Expr) -> bool:
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Order | exp.Group):
            return True
        parent = parent.parent
    return False


def _check_nodes(tree: exp.Select) -> None:
    """Fail closed: a node type that is neither structure, scalar nor an allowed aggregate."""
    structure = (exp.Select, exp.From, exp.Table, exp.TableAlias, exp.Where, exp.Group, exp.Order,
                 exp.Ordered, exp.Limit, exp.Alias, exp.Star, exp.Filter, exp.Distinct)  # fmt: skip
    allowed = structure + _SCALAR_NODES + _VALUE_AGGS + _COUNT_AGGS
    for node in tree.walk():
        if isinstance(node, exp.Var) and isinstance(node.parent, _UNIT_PARENTS):
            continue
        if not isinstance(node, allowed) or isinstance(node, exp.Anonymous):
            name = render(node)[:40] if not isinstance(node, exp.Anonymous) else node.name
            raise SqlRefusedError(f"This is not supported in governed SQL: {name or type(node).__name__}.")


def _check_scalar(node: exp.Expr, catalog: SqlCatalog, where: str) -> None:
    """A row-level expression: allowed scalars only, and the patient ID never in it."""
    for sub in node.walk():
        if isinstance(sub, _VALUE_AGGS + _COUNT_AGGS + (exp.Filter, exp.Distinct, exp.Star)):
            raise SqlRefusedError(f"Summaries (count, avg, ...) can't be used in {where}.")
        if isinstance(sub, exp.Var) and isinstance(sub.parent, _UNIT_PARENTS):
            continue
        if not isinstance(sub, _SCALAR_NODES):
            raise SqlRefusedError(f"This is not supported in {where}: {render(sub)[:40]}.")
        if isinstance(sub, exp.Column):
            spec = catalog.column(sub.name)
            if spec is not None and spec.is_id:
                raise SqlRefusedError(ONE_PATIENT)


def _check_dimension_privacy(dimension: exp.Expr, catalog: SqlCatalog) -> None:
    """Every PHI column in the dimension is coarsened where it is read (see the module docstring)."""
    for column in dimension.find_all(exp.Column):
        spec = catalog.column(column.name)
        if spec is not None and spec.phi and not _coarsened(column, spec, dimension):
            raise SqlRefusedError(
                f"{spec.name} is personal information: group by its year, quarter or month only, "
                f"read straight from the column: year({spec.name}), month({spec.name}) or "
                f"date_trunc('month', {spec.name})."
            )


def _coarsened(column: exp.Column, spec: ColumnSpec, root: exp.Expr) -> bool:
    """Whether ``column`` is the direct argument (through casts to a date or timestamp only) of a
    coarsener with a unit of year, quarter or month.

    Only the argument slot counts: a coarsener further up the tree is not enough, since what lies
    between could encode the full value in a year (``year(DATE '0001-01-01' + 366 * datediff(...))``).
    """
    node: exp.Expr = column
    cast = False
    while isinstance(node.parent, exp.Cast | exp.TryCast) and node.arg_key == "this" and node is not root:
        if not node.parent.to.is_type(*_CALENDAR_CASTS):
            return False
        node, cast = node.parent, True
    if spec.logical_type not in _CALENDAR_TYPES and not (cast and spec.logical_type == "string"):
        return False
    parent = node.parent
    if node is root or parent is None:
        return False
    if isinstance(parent, _COARSENERS):
        return node.arg_key == "this"
    if isinstance(parent, exp.Extract):
        return node.arg_key == "expression" and _coarse_unit(parent.this)
    if isinstance(parent, _TRUNCS):
        return node.arg_key == "this" and _coarse_unit(parent.args.get("unit"))
    return False


def _coarse_unit(unit: exp.Expr | None) -> bool:
    """A unit written as a keyword or a string literal: year, quarter or month."""
    if isinstance(unit, exp.Var) or (isinstance(unit, exp.Literal) and unit.is_string):
        return unit.name.upper() in _COARSE_UNITS
    return False


def _output_name(projection: exp.Expr) -> str:
    if isinstance(projection, exp.Alias):
        return projection.alias
    if isinstance(projection, exp.Column):
        return projection.name
    return render(projection)


def _dimension(tree: exp.Select, bodies: list[exp.Expr], names: list[str]) -> exp.Expr | None:
    group = tree.args.get("group")
    if group is None:
        return None
    items = list(group.expressions)
    if len(items) != 1:
        raise SqlRefusedError(
            "Group by one column at a time for now. To split an outcome by a group, "
            "count it with count_if(...) grouped by the group."
        )
    return _output_target(items[0], bodies, names, "GROUP BY")


def _output_target(item: exp.Expr, bodies: list[exp.Expr], names: list[str], clause: str) -> exp.Expr:
    """A GROUP BY / ORDER BY item as the output expression it names (position, alias, or itself).

    GROUP BY reads a name as a column first (columns are quoted once canonical); ORDER BY reads it
    as an output alias first (``_canonical_columns`` leaves such a name as written).
    """
    if isinstance(item, exp.Literal) and not item.is_string:
        position = int(item.name) if item.name.isdigit() else 0
        if not 1 <= position <= len(bodies):
            raise SqlRefusedError(f"{clause} {item.name} is not a column of the output.")
        return bodies[position - 1]
    alias_first = clause == "ORDER BY" and isinstance(item, exp.Column) and not item.table
    if isinstance(item, exp.Column) and (alias_first or not item.this.quoted):
        for body, name in zip(bodies, names, strict=True):
            if name.casefold() == item.name.casefold() and body != item:
                return body
    return item


def _measure(body: exp.Expr, name: str, catalog: SqlCatalog, aggregates: list[Aggregate], *, grouped: bool) -> Measure:
    """A measure: arithmetic over aggregates, each aggregate checked and numbered."""
    found: list[int] = []

    def visit(node: exp.Expr) -> exp.Expr:
        root = node.this if isinstance(node, exp.Filter) else node
        if isinstance(root, _VALUE_AGGS + _COUNT_AGGS):
            condition = None
            if isinstance(node, exp.Filter):
                condition = node.expression.this
                _check_scalar(condition, catalog, "FILTER")
            aggregates.append(_aggregate(root, condition, catalog))
            found.append(len(aggregates) - 1)
            return exp.column(f"{AGG_MARKER}{len(aggregates) - 1}")
        if isinstance(node, exp.Column | exp.Star):
            raise SqlRefusedError(NOT_IN_GROUP if grouped else LISTS_PATIENTS)
        if not isinstance(node, _MEASURE_NODES):
            raise SqlRefusedError(f"This is not supported around a summary: {render(node)[:40]}.")
        return node

    template = _transform_outside_aggregates(body.copy(), visit)
    if not found:
        raise SqlRefusedError(LISTS_PATIENTS)
    return Measure(name=name, template=template, aggregates=tuple(found))


def _transform_outside_aggregates(node: exp.Expr, visit: Callable[[exp.Expr], exp.Expr]) -> exp.Expr:
    """Top-down: an aggregate (or FILTER) is replaced whole; other nodes are checked, then their
    children visited."""
    replaced = visit(node)
    if replaced is not node:
        return replaced
    for key, value in list(node.args.items()):
        if isinstance(value, exp.Expr):
            node.set(key, _transform_outside_aggregates(value, visit))
        elif isinstance(value, list):
            node.set(key, [_transform_outside_aggregates(v, visit) if isinstance(v, exp.Expr) else v for v in value])
    return node


def _aggregate(node: exp.Expr, condition: exp.Expr | None, catalog: SqlCatalog) -> Aggregate:
    if any(isinstance(n, _VALUE_AGGS + _COUNT_AGGS + (exp.Filter,)) for n in node.walk() if n is not node):
        raise SqlRefusedError("A summary of a summary is not supported.")
    if isinstance(node, exp.CountIf):
        predicate = node.this
        _check_scalar(predicate, catalog, "count_if")
        both = predicate if condition is None else exp.and_(exp.paren(condition), exp.paren(predicate))
        return Aggregate("count", exp.Count(this=exp.Star()), None, both)
    if isinstance(node, exp.Count):
        return _count(node, condition, catalog)
    column = node.this
    if not isinstance(column, exp.Column) or len(node.args.get("expressions") or []) > 0:
        raise SqlRefusedError(
            f"{node.key}(...) takes one column as it is. To summarise a condition, count it: "
            "count_if(condition) / count(*)."
        )
    spec = _value_column(column, catalog)
    if isinstance(node, exp.PercentileCont):
        quantile = node.expression
        if (
            not isinstance(quantile, exp.Literal)
            or quantile.is_string
            or float(quantile.name) not in config.ALLOWED_QUANTILES
        ):
            allowed = ", ".join(str(q) for q in sorted(config.ALLOWED_QUANTILES))
            raise SqlRefusedError(f"quantile_cont takes one of these quantiles: {allowed}.")
    elif any(v for k, v in node.args.items() if k != "this"):
        raise SqlRefusedError(f"{node.key}(...) takes one column.")
    function = node.copy()
    if spec.logical_type == "boolean":
        # DuckDB has no avg/median/stddev of BOOLEAN: read true as 1 and false as 0 (a proportion).
        function.set("this", exp.Cast(this=column.copy(), to=exp.DataType.build("INTEGER")))
    return Aggregate("value", function, spec.name, condition)


def _count(node: exp.Count, condition: exp.Expr | None, catalog: SqlCatalog) -> Aggregate:
    arg = node.this
    if isinstance(arg, exp.Star) and not node.args.get("expressions"):
        return Aggregate("count", exp.Count(this=exp.Star()), None, condition)
    if isinstance(arg, exp.Distinct):
        inner = arg.expressions
        spec = catalog.column(inner[0].name) if len(inner) == 1 and isinstance(inner[0], exp.Column) else None
        if spec is None or not spec.is_id:
            raise SqlRefusedError("count(DISTINCT ...) is supported for the patient ID only.")
        return Aggregate("count", exp.Count(this=exp.column(spec.name, quoted=True)), spec.name, condition)
    if isinstance(arg, exp.Column):
        if node.args.get("expressions"):
            raise SqlRefusedError("count takes one column. Count each column on its own: count(a), count(b).")
        spec = catalog.column(arg.name)
        assert spec is not None  # resolved by _canonical_columns
        return Aggregate("count", exp.Count(this=exp.column(spec.name, quoted=True)), spec.name, condition)
    raise SqlRefusedError(
        "count takes *, one column, or DISTINCT of the patient ID. For a condition use count_if(...)."
    )


def _value_column(column: exp.Column, catalog: SqlCatalog) -> ColumnSpec:
    spec = catalog.column(column.name)
    assert spec is not None  # resolved by _canonical_columns
    if spec.is_id:
        raise SqlRefusedError(ONE_PATIENT)
    if spec.phi:
        raise SqlRefusedError(f"{spec.name} is personal information: it can be counted or used in a filter only.")
    if not spec.numeric:
        raise SqlRefusedError(f"{spec.name} is not a number, so it can only be counted.")
    return spec


def _order(tree: exp.Select, bodies: list[exp.Expr], names: list[str]) -> tuple[OrderKey, ...]:
    order = tree.args.get("order")
    if order is None:
        return ()
    keys = []
    for ordered in order.expressions:
        if not isinstance(ordered, exp.Ordered):
            raise SqlRefusedError("ORDER BY takes output columns.")
        target = _output_target(ordered.this, bodies, names, "ORDER BY")
        index = next((i for i, body in enumerate(bodies) if body == target), None)
        if index is None:
            raise SqlRefusedError("ORDER BY takes output columns (by name, alias or position).")
        keys.append(OrderKey(names[index], bool(ordered.args.get("desc"))))
    return tuple(keys)


def _limit(tree: exp.Select) -> int | None:
    limit = tree.args.get("limit")
    if limit is None:
        return None
    value = limit.expression
    if not isinstance(value, exp.Literal) or value.is_string or not value.name.isdigit():
        raise SqlRefusedError("LIMIT takes a whole number.")
    return int(value.name)

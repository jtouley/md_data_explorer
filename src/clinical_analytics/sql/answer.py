# SPDX-License-Identifier: MIT
"""Run a governed query and show its answer as computed.

The query is never run as written: this module generates SQL from the checked tree. The WHERE's
AND terms and each aggregate's own FILTER or count_if terms are its conditions; each runs as an
indicator, ``coalesce(TRY(term), false)``, so no row expression can make the query fail (a failure
that depends on one patient's value would answer "does this patient exist?"). Every aggregate
reads the rows meeting the WHERE's conditions and its own.

Counts are patients: the rows are checked to be one per patient first (``patient_rows``). Every
number is shown as computed, small counts included (ADR015): the user already holds the rows. A
count leaves the machine only through ``core.export_boundary``, which writes 1-10 as "<11".

The answer carries the step-by-step count, CONSORT style: the patients, then each WHERE term in
the order written.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any

import structlog
from sqlglot import exp

from clinical_analytics.core.result_presentation import _table
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
    "This dataset has no patient ID column, so its rows can't be checked to be patients, and every "
    "count would be labelled patients. If each row is one patient, confirm that and ask again."
)
MISSING_IDS = (
    "Some rows have no {key}, so they can't be checked to be different patients. "
    "If each row is one patient, confirm that and ask again."
)
TOO_MANY_GROUPS = (
    f"The grouping has more than {config.MAX_GROUPS} groups, too many to show as a table. "
    "Group by something coarser (a range, a year)."
)
NO_VALUE_LABEL = "(no value)"
# The base count of a group: its patients meeting the WHERE, never shown as a measure.
_BASE = "__base"


class PatientRowsUnconfirmedError(SqlRefusedError):
    """Rows can be counted as patients only once the user confirms one row per patient (no ID
    column, or rows without an ID). ``rows`` is what a confirmation is made on; never shown.
    ``confirmation``: the token a confirmation must send back (set by ``sql.service``, which knows
    the dataset version), so it applies only to the data the user was asked about."""

    def __init__(self, reason: str, *, key: str | None, rows: int, confirmation: str | None = None) -> None:
        super().__init__(reason)
        self.key = key
        self.rows = rows
        self.confirmation = confirmation


@dataclass(frozen=True)
class SqlAnswer:
    sql: str  # the checked query, as shown to the user
    columns: list[str]
    rows: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)


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


def answer_query(
    query: GovernedQuery,
    sandbox: SandboxConnection,
    catalog: SqlCatalog,
    *,
    confirmed_patient_rows: int | None = None,
) -> SqlAnswer:
    """The query's answer, every value as computed.

    ``confirmed_patient_rows``: the row count on which the user confirmed one row per patient for
    this dataset version (None: not confirmed). It is needed only when rows can't be shown to be
    patients by their IDs, and counts only while the table still has that many rows.

    Raises:
        PatientRowsUnconfirmedError: no ID on some or all rows, and no matching confirmation.
        SqlRefusedError: rows are visits, too many groups, or the query could not run.
    """
    confirmed = patient_rows(sandbox, catalog, confirmed_patient_rows)
    conditions = _conditions(query)
    frame = sandbox.run_generated(_core_sql(query, conditions))
    if frame.height > config.MAX_GROUPS:
        raise SqlRefusedError(TOO_MANY_GROUPS)
    records = frame.to_dicts()
    steps = _step_counts(sandbox, conditions)
    logger.info("governed_sql_answered", groups=frame.height, measures=len(query.measures))
    return _present(query, records, conditions, steps, confirmed)


def patient_rows(sandbox: SandboxConnection, catalog: SqlCatalog, confirmed_rows: int | None) -> bool:
    """Check that every row is one patient, so every count is a count of patients.

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
    counts = sandbox.run_generated(render(exp.select(*columns).from_(config.TABLE_NAME))).row(0, named=True)
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


def _conditions(query: GovernedQuery) -> _Conditions:
    """Number every condition of the answer once: the WHERE's AND terms, then each aggregate's own."""
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
        for term in _conjuncts(agg.condition):
            mask |= 1 << number(term)
        own.append(mask)
    return _Conditions(tuple(terms), where, tuple(own))


def _try(expression: exp.Expr) -> exp.Expr:
    """A row-level expression that never fails: an error becomes NULL (a predicate then excludes
    the row, as it would any other)."""
    return exp.Try(this=expression.copy())


def _core_sql(query: GovernedQuery, conditions: _Conditions) -> str:
    """The measures per group, each aggregate over the rows meeting the WHERE's and its own terms,
    and each group's patients meeting the WHERE (``_BASE``: a group none of them is in is not shown).

    The aggregates run in an inner query (a count as a whole number, the rest as DOUBLE: sum() of
    an integer is HUGEINT, which would arrive as a Decimal and be sent as text). The arithmetic
    around them runs in the outer query, on DOUBLE and inside TRY, so no value can make the query
    fail (DuckDB allows no aggregate inside TRY, hence the two levels).
    """
    dimension = _try(query.dimension) if query.dimension is not None else None
    double = exp.DataType.build("DOUBLE")
    inner: list[exp.Expr] = [exp.alias_(dimension.copy(), "__d")] if dimension is not None else []
    inner.append(
        exp.alias_(
            exp.Filter(
                this=exp.Count(this=exp.Star()), expression=exp.Where(this=conditions.predicate(conditions.where_mask))
            ),
            _BASE,
        )
    )
    for k, agg in enumerate(query.aggregates):
        aggregate: exp.Expr = exp.Filter(
            this=agg.function.copy(),
            expression=exp.Where(this=conditions.predicate(conditions.where_mask | conditions.of_aggregate[k])),
        )
        if agg.kind != "count":
            aggregate = exp.Cast(this=aggregate, to=double.copy())
        inner.append(exp.alias_(aggregate, f"__a{k}"))
    select = exp.select(*inner).from_(config.TABLE_NAME)
    if dimension is not None:
        select = select.group_by(dimension.copy())

    def measure(template: exp.Expr) -> exp.Expr:
        if isinstance(template, exp.Column):  # one aggregate as it is: a count stays a whole number
            return exp.column(f"__a{template.name.removeprefix(AGG_MARKER)}")

        def fill(node: exp.Expr) -> exp.Expr:
            if isinstance(node, exp.Column) and node.name.startswith(AGG_MARKER):
                column = exp.column(f"__a{node.name.removeprefix(AGG_MARKER)}")
                return exp.Cast(this=column, to=double.copy())
            return node

        return exp.Try(this=template.copy().transform(fill))

    outer: list[exp.Expr] = [exp.column("__d")] if dimension is not None else []
    outer.append(exp.column(_BASE))
    outer += [exp.alias_(measure(m.template), f"__m{i}") for i, m in enumerate(query.measures)]
    return render(exp.select(*outer).from_(select.subquery("__core")).limit(config.MAX_GROUPS + 1))


def _steps(conditions: _Conditions) -> list[int]:
    """The step-by-step count's masks: no condition, then each WHERE term added in order."""
    masks, mask = [0], 0
    for i in conditions.where:
        mask |= 1 << i
        masks.append(mask)
    return masks


def _step_counts(sandbox: SandboxConnection, conditions: _Conditions) -> list[int]:
    """Patients at each step of the step-by-step count (none without a WHERE), in one scan."""
    if not conditions.where:
        return []
    counts = [
        exp.alias_(
            exp.Filter(this=exp.Count(this=exp.Star()), expression=exp.Where(this=conditions.predicate(mask))),
            f"__s{n}",
        )
        for n, mask in enumerate(_steps(conditions))
    ]
    row = sandbox.run_generated(render(exp.select(*counts).from_(config.TABLE_NAME))).row(0)
    return [int(value) for value in row]


def _label(value: Any) -> str:
    return NO_VALUE_LABEL if value is None else str(value)


def _present(
    query: GovernedQuery,
    records: list[dict[str, Any]],
    conditions: _Conditions,
    steps: list[int],
    confirmed: bool,
) -> SqlAnswer:
    """The answer: one row per group with patients meeting the WHERE (or the one ungrouped row)."""
    grouped = query.dimension is not None
    measure_names = [m.name for m in query.measures]
    columns = ([query.dimension_name or ""] if grouped else []) + measure_names
    rows: list[dict[str, Any]] = []
    raw: dict[int, Any] = {}
    for record in records:
        if grouped and not record.get(_BASE):
            continue  # a group with no patient meeting the WHERE has nothing to show
        row: dict[str, Any] = {}
        if grouped:
            assert query.dimension_name is not None
            row[query.dimension_name] = _label(record["__d"])
        for i, name in enumerate(measure_names):
            row[name] = record[f"__m{i}"]
        raw[id(row)] = record.get("__d")
        rows.append(row)
    rows = _ordered(rows, query, raw)
    if query.limit is not None:
        rows = rows[: query.limit]
    notes = [
        "Counts are patients (one row per patient, as you confirmed for this dataset)."
        if confirmed
        else "Counts are patients (one row per patient)."
    ]
    step_note = _step_note(steps, conditions)
    if step_note is not None:
        notes.append(step_note)
    table = _table(rows)
    return SqlAnswer(
        sql=query.shown_sql,
        columns=table["columns"] if table is not None else columns,
        rows=table["rows"] if table is not None else [],
        notes=notes,
    )


def _step_note(steps: list[int], conditions: _Conditions) -> str | None:
    """The step-by-step count, CONSORT style: the patients, then each WHERE term in order."""
    if not steps:
        return None
    parts = []
    for n, count in enumerate(steps):
        label = "patients" if n == 0 else f"with {render(conditions.terms[conditions.where[n - 1]])}"
        parts.append(f"{count:,} {label}")
    return "Step by step: " + " → ".join(parts) + "."


def _ordered(rows: list[dict[str, Any]], query: GovernedQuery, raw: dict[int, Any]) -> list[dict[str, Any]]:
    """ORDER BY over the shown values (missing ones last); without it, by the group's value. Never
    the engine's order: the group's value breaks every tie, so the order is the same on every run."""
    keys = [(k.column, k.descending) for k in query.order]
    if query.dimension_name is not None:
        keys.append((query.dimension_name, False))

    def value(row: dict[str, Any], column: str) -> Any:
        if column == query.dimension_name:
            return raw[id(row)]
        found = row.get(column)
        return None if isinstance(found, float) and math.isnan(found) else found

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

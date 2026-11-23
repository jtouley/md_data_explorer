# SPDX-License-Identifier: MIT
"""The plan the model fills in, its reply schema, and the compiler from plan to SQL.

The model never writes SQL or free conditions. It fills fixed slots, held by the reply schema
(Ollama's ``format``) to enums built from the contract: the columns, each category's values in
its dictionary's words, the answer kinds, how each patient's records are read (any, every, first,
last, change) and the comparison operators. The compiler turns a valid plan into one SELECT; a
plan it cannot compile raises ``PlanError`` with feedback that only names what it supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from clinical_analytics.model_query.context import Column, DatasetContext

Answer: TypeAlias = Literal[
    "count", "share", "mean", "median", "min", "max", "sum",
    "compare_groups", "odds_ratio", "correlation", "survival", "list_patients",
]  # fmt: skip
WhichVisit: TypeAlias = Literal["any", "every", "first", "last", "change"]
Period: TypeAlias = Literal["none", "year", "month"]
ANSWERS: Final = get_args(Answer)
STATISTICS: Final = {"mean": "AVG", "median": "MEDIAN", "min": "MIN", "max": "MAX", "sum": "SUM"}
TESTS: Final = ("compare_groups", "odds_ratio", "correlation", "survival")
CATEGORY_OPS: Final = ("is", "is not", "is missing", "is recorded")
NUMBER_OPS: Final = ("is", "is not", "<", "<=", ">", ">=", "is missing", "is recorded")
DATE_OPS: Final = ("before", "on or after", "is missing", "is recorded")
MAX_GROUPS: Final = 30
MAX_CUTS: Final = 6
LISTING_REASON: Final = (
    "This app answers with counts, shares and summaries; it does not list individual patients or their IDs."
)


class PlanError(ValueError):
    """The plan cannot be compiled; the message is fed back to the model (never shown raw)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Condition(_Strict):
    column: str
    op: str
    value: float | str | None = None
    values: tuple[float | str, ...] = ()


class Plan(_Strict):
    answer: Answer
    count_of: str = "patients"
    filters: tuple[Condition, ...] = ()
    numerator: tuple[Condition, ...] = ()
    measure: str | None = None
    group_by: str | None = None
    period: Period = "none"
    cut_points: tuple[float, ...] = Field(default=(), max_length=MAX_CUTS)
    which_visit: WhichVisit = "any"


def parse_plan(payload: Any) -> Plan:
    if not isinstance(payload, dict):
        raise PlanError("Reply with one JSON object.")
    try:
        return Plan.model_validate(payload)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise PlanError(f"The reply's {where} is not valid: {first['msg']}.") from e


# ----------------------------------------------------------------------------- reply schema
def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _condition_schema(column: Column) -> dict[str, Any] | None:
    ops: tuple[str, ...]
    value: dict[str, Any]
    values: dict[str, Any]
    if column.kind in ("category", "flag"):
        choices = list(column.choices())
        if not choices:
            return None
        ops, value, values = CATEGORY_OPS, {"type": "null"}, {"type": "array", "items": {"enum": choices}}
    elif column.kind == "number":
        ops, value, values = NUMBER_OPS, _nullable({"type": "number"}), {"type": "array", "items": {"type": "number"}}
    elif column.kind == "date":
        date = {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}
        ops, value, values = DATE_OPS, _nullable(date), {"type": "array", "maxItems": 0}
    else:
        return None
    return {
        "type": "object",
        "properties": {
            "column": {"const": column.name},
            "op": {"enum": list(ops)},
            "value": value,
            "values": {**values, "maxItems": values.get("maxItems", 12)},
        },
        "required": ["column", "op", "value", "values"],
    }


def reply_schema(context: DatasetContext) -> dict[str, Any]:
    """The JSON schema the model's reply is held to: every name and value comes from the table."""
    usable = [c for c in context.columns if c.kind != "key"]
    conditions = [s for c in usable if (s := _condition_schema(c)) is not None]
    condition_list = {"type": "array", "items": {"anyOf": conditions}, "maxItems": 6} if conditions else {
        "type": "array", "maxItems": 0,
    }  # fmt: skip
    names = [c.name for c in usable]
    return {
        "type": "object",
        "properties": {
            "answer": {"enum": list(ANSWERS)},
            "count_of": {"enum": list(context.units)},
            "filters": condition_list,
            "numerator": condition_list,
            "measure": _nullable({"enum": names}) if names else {"type": "null"},
            "group_by": _nullable({"enum": names}) if names else {"type": "null"},
            "period": {"enum": list(get_args(Period))},
            "cut_points": {"type": "array", "items": {"type": "number"}, "maxItems": MAX_CUTS},
            "which_visit": {"enum": list(get_args(WhichVisit))},
        },
        "required": [
            "answer", "count_of", "filters", "numerator", "measure", "group_by", "period", "cut_points", "which_visit",
        ],
    }  # fmt: skip


# ----------------------------------------------------------------------------- compiled form
Kind: TypeAlias = Literal["count", "share", "statistic", "test"]


@dataclass(frozen=True)
class Compiled:
    """One plan as SQL, and what the checks, ask-backs and reading card read about it."""

    plan: Plan
    kind: Kind
    sql: str
    unit: str  # what is counted: "patients" or the records' noun
    per_patient: bool  # patients counted from a visit-level table
    total_sql: str | None = None  # the units the count is out of (N of M)
    group_column: str | None = None
    filters: tuple[Condition, ...] = ()
    numerator: tuple[Condition, ...] = ()
    measure: str | None = None
    columns: frozenset[str] = frozenset()
    test_outcome: str | None = None
    test_group: str | None = None
    test_time: str | None = None

    @property
    def numbers(self) -> tuple[float, ...]:
        """Every number the plan uses (condition values, listed values, cut points)."""
        found: list[float] = []
        for cond in (*self.filters, *self.numerator):
            for v in (cond.value, *cond.values):
                if isinstance(v, int | float) and not isinstance(v, bool):
                    found.append(float(v))
        return (*found, *self.plan.cut_points)


def ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return repr(value) if isinstance(value, float) and not value.is_integer() else str(int(value))
    return "'" + str(value).replace("'", "''") + "'"


def _column(name: str | None, context: DatasetContext, what: str) -> Column:
    column = context.column(name or "")
    if column is None or column.kind == "key":
        raise PlanError(f"{what} must be one of the table's columns (not the ID).")
    return column


def _stored(column: Column, said: float | str) -> Any:
    if column.kind == "number":
        if isinstance(said, str):
            raise PlanError(f"{column.name} is a number: give a number.")
        return said
    value = column.stored(str(said))
    if value is None:
        raise PlanError(f"'{said}' is not a value of {column.name}; use one of {', '.join(column.choices())}.")
    return value


def condition_sql(cond: Condition, context: DatasetContext) -> str:
    column = _column(cond.column, context, "A condition's column")
    ref = ident(column.name)
    if cond.op == "is missing":
        return f"{ref} IS NULL"
    if cond.op == "is recorded":
        return f"{ref} IS NOT NULL"
    if column.kind == "date":
        if not isinstance(cond.value, str) or cond.op not in DATE_OPS:
            raise PlanError(f"{column.name} is a date: use before or on or after with a YYYY-MM-DD value.")
        return f"{ref} {'<' if cond.op == 'before' else '>='} DATE {literal(cond.value)}"
    if cond.op in ("is", "is not"):
        said = cond.values or ((cond.value,) if cond.value is not None else ())
        if not said:
            raise PlanError(f"The condition on {column.name} needs its values.")
        stored = [_stored(column, v) for v in said]
        sql = f"{ref} IN ({', '.join(literal(v) for v in stored)})"
        return f"NOT ({sql})" if cond.op == "is not" else sql
    if column.kind != "number" or cond.op not in ("<", "<=", ">", ">="):
        raise PlanError(f"{column.name} takes is or is not with its listed values.")
    if not isinstance(cond.value, int | float):
        raise PlanError(f"The comparison on {column.name} needs a number in value.")
    return f"{ref} {cond.op} {literal(cond.value)}"


def _all(conditions: tuple[Condition, ...], context: DatasetContext) -> str | None:
    parts = [condition_sql(c, context) for c in conditions]
    return " AND ".join(f"({p})" for p in parts) if parts else None


def _recorded(conditions: tuple[Condition, ...]) -> str:
    """Every column a condition reads is recorded (a share's denominator)."""
    names = dict.fromkeys(c.column for c in conditions if c.op not in ("is missing", "is recorded"))
    return " AND ".join(f"{ident(n)} IS NOT NULL" for n in names) or "TRUE"


def _group_expr(plan: Plan, context: DatasetContext) -> str | None:
    if plan.group_by is None:
        return None
    column = _column(plan.group_by, context, "group_by")
    ref = ident(column.name)
    if column.kind == "date":
        return f"strftime({ref}, '%Y-%m')" if plan.period == "month" else f"year({ref})"
    if column.kind == "number" and plan.cut_points:
        cuts = sorted(set(plan.cut_points))
        whens = [f"WHEN {ref} < {literal(cuts[0])} THEN 'under {_num(cuts[0])}'"]
        for lo, hi in zip(cuts, cuts[1:], strict=False):
            whens.append(f"WHEN {ref} < {literal(hi)} THEN '{_num(lo)} to under {_num(hi)}'")
        whens.append(f"WHEN {ref} >= {literal(cuts[-1])} THEN '{_num(cuts[-1])} or more'")
        return f"CASE {' '.join(whens)} END"
    if column.kind == "number" and column.distinct > MAX_GROUPS:
        raise PlanError(f"{column.name} is a number: give cut_points to break it into groups, or group by a category.")
    return ref


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _where(*parts: str | None) -> str:
    kept = [p for p in parts if p]
    return f" WHERE {' AND '.join(kept)}" if kept else ""


def _ranked(context: DatasetContext, which: WhichVisit, extra: str | None = None) -> str:
    """Each patient's records numbered from the first (or the last) by the contract's order."""
    order = context.order_column
    if order is None:
        raise PlanError("Nothing orders a patient's records in this table: use which_visit any or every.")
    key = ident(context.patient_key or "")
    direction = "DESC" if which == "last" else "ASC"
    return (
        f"SELECT *, ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY {ident(order)} {direction} NULLS LAST) AS _rank "
        f"FROM unified{_where(f'{key} IS NOT NULL', extra)}"
    )


def compile_plan(plan: Plan, context: DatasetContext) -> Compiled:
    """The plan as one SELECT over the ``unified`` view.

    Raises:
        PlanError: the plan uses a column, value or combination the compiler does not support.
    """
    if plan.answer == "list_patients":
        raise PlanError(LISTING_REASON)
    per_patient = plan.count_of == "patients" and context.visit_level
    which = plan.which_visit if context.visit_level else "any"
    if which in ("first", "last", "change") and context.order_column is None:
        raise PlanError("Nothing orders a patient's records in this table: use which_visit any or every.")
    used = {c.column for c in (*plan.filters, *plan.numerator)} | {n for n in (plan.measure, plan.group_by) if n}
    for name in used:
        _column(name, context, f"'{name}'")
    if plan.answer in TESTS:
        return _compile_test(plan, context, which, frozenset(used))
    if plan.answer in STATISTICS:
        return _compile_statistic(plan, context, which, frozenset(used))
    if plan.answer == "share":
        return _compile_share(plan, context, which, per_patient, frozenset(used))
    return _compile_count(plan, context, which, per_patient, frozenset(used))


def _compile_count(
    plan: Plan, context: DatasetContext, which: WhichVisit, per_patient: bool, used: frozenset[str]
) -> Compiled:
    where = _all(plan.filters, context)
    group = _group_expr(plan, context)
    key = ident(context.patient_key or "")
    unit = "patients" if per_patient or context.semantics.one_row_per_patient else context.row_nouns
    if which == "change":
        raise PlanError("change is for a measure (mean, median ...): for a count use any, every, first or last.")
    if not per_patient:
        n, source = "COUNT(*)", "unified"
        total = "SELECT COUNT(*) FROM unified"
    elif which == "every":
        if group is not None:
            raise PlanError("every applies to conditions, not to a breakdown: use any, first or last with group_by.")
        source = (
            f"(SELECT {key} FROM unified WHERE {key} IS NOT NULL GROUP BY {key} HAVING BOOL_AND({where or 'TRUE'}))"
        )
        where, n = None, "COUNT(*)"
        total = f"SELECT COUNT(DISTINCT {key}) FROM unified"
    elif which in ("first", "last"):
        source = f"({_ranked(context, which)})"
        where = " AND ".join(p for p in ("_rank = 1", where) if p)
        n = "COUNT(*)"
        total = f"SELECT COUNT(DISTINCT {key}) FROM unified"
    else:
        n, source = f"COUNT(DISTINCT {key})", "unified"
        total = f"SELECT COUNT(DISTINCT {key}) FROM unified"
    if group is None:
        sql = f"SELECT {n} AS count FROM {source} AS t{_where(where)}"
    else:
        sql = f"SELECT {group} AS grp, {n} AS count FROM {source} AS t{_where(where)} GROUP BY 1 ORDER BY 1 NULLS LAST"
    return Compiled(
        plan=plan, kind="count", sql=sql, unit=unit, per_patient=per_patient, total_sql=total,
        group_column=plan.group_by, filters=plan.filters, columns=used,
    )  # fmt: skip


def _compile_share(
    plan: Plan, context: DatasetContext, which: WhichVisit, per_patient: bool, used: frozenset[str]
) -> Compiled:
    shared = sorted({c.column for c in plan.numerator} & {c.column for c in plan.filters})
    if shared:
        # The same column limiting both the records and the counted event makes the share 100%
        # or 0% whatever the data says.
        raise PlanError(
            f"{', '.join(shared)} is in both filters and numerator: put the condition counted only in numerator, "
            "and keep in filters only the records the share is out of."
        )
    if not plan.numerator:
        if plan.group_by is not None:
            return _compile_count(plan.model_copy(update={"answer": "count"}), context, which, per_patient, used)
        raise PlanError("A share needs numerator: the condition counted among the records kept by filters.")
    where = _all(plan.filters, context)
    event = _all(plan.numerator, context)
    recorded = _recorded(plan.numerator)
    group = _group_expr(plan, context)
    key = ident(context.patient_key or "")
    unit = "patients" if per_patient or context.semantics.one_row_per_patient else context.row_nouns
    if which == "change":
        raise PlanError("change is for a measure (mean, median ...): for a share use any, every, first or last.")
    group_sel = f"{group} AS grp, " if group is not None else ""
    if not per_patient:
        inner = f"SELECT {group_sel}({event}) AS e FROM unified{_where(where, recorded)}"
    elif which in ("first", "last"):
        inner = f"SELECT {group_sel}({event}) AS e FROM ({_ranked(context, which)}) AS r{_where('_rank = 1', where)}"
        inner = f"SELECT * FROM ({inner}) AS x WHERE e IS NOT NULL"
    else:
        agg = "BOOL_AND" if which == "every" else "BOOL_OR"
        by = "grp, " if group is not None else ""
        inner = (
            f"SELECT {by}{key}, {agg}(e) AS e FROM (SELECT {group_sel}{key}, ({event}) AS e FROM unified"
            f"{_where(f'{key} IS NOT NULL', where, recorded)}) AS v GROUP BY {by}{key}"
        )
    head = "grp, " if group is not None else ""
    rate = "COUNT_IF(e) / NULLIF(COUNT(*), 0) AS rate"
    sql = f"SELECT {head}COUNT_IF(e) AS events, COUNT(*) AS n, {rate} FROM ({inner}) AS s"
    if group is not None:
        sql += " GROUP BY 1 ORDER BY 1 NULLS LAST"
    return Compiled(
        plan=plan, kind="share", sql=sql, unit=unit, per_patient=per_patient, group_column=plan.group_by,
        filters=plan.filters, numerator=plan.numerator, columns=used,
    )  # fmt: skip


def _compile_statistic(plan: Plan, context: DatasetContext, which: WhichVisit, used: frozenset[str]) -> Compiled:
    column = _column(plan.measure, context, "measure")
    if not column.numeric:
        raise PlanError(f"{column.name} is not a number column: a {plan.answer} needs one; count or share a category.")
    func = STATISTICS[plan.answer]
    where = _all(plan.filters, context)
    group = _group_expr(plan, context)
    ref = ident(column.name)
    group_sel = f"{group} AS grp, " if group is not None else ""
    head, tail = ("grp, ", " GROUP BY 1 ORDER BY 1 NULLS LAST") if group is not None else ("", "")
    per_patient = context.visit_level and which in ("first", "last", "change")
    if per_patient:
        # One value per patient: the value at their first record (baseline; missing there, the
        # patient is left out), their latest recorded value, or latest minus baseline.
        key, order = ident(context.patient_key or ""), ident(context.order_column or "")
        first, last = f"arg_min_null({ref}, {order})", f"arg_max({ref}, {order})"
        one = {"first": first, "last": last, "change": f"{last} - {first}"}[which]
        enough = f" HAVING COUNT({ref}) >= 2" if which == "change" else ""
        # Each patient in one group: their group at the record the value is read from (the
        # first one for a change).
        pick = "arg_max" if which == "last" else "arg_min"
        grp = f"{pick}({group}, {order}) AS grp, " if group is not None else ""
        source = (
            f"(SELECT {grp}{key}, {one} AS v FROM unified{_where(f'{key} IS NOT NULL', where)} GROUP BY {key}{enough})"
        )
        sql = f"SELECT {head}{func}(v) AS value, COUNT(v) AS n FROM {source} AS t{tail}"
    else:
        sql = f"SELECT {group_sel}{func}({ref}) AS value, COUNT({ref}) AS n FROM unified{_where(where)}{tail}"
    return Compiled(
        plan=plan, kind="statistic", sql=sql, unit="patients" if per_patient else context.row_nouns,
        per_patient=per_patient, group_column=plan.group_by, filters=plan.filters, measure=column.name, columns=used,
    )  # fmt: skip


def _compile_test(plan: Plan, context: DatasetContext, which: WhichVisit, used: frozenset[str]) -> Compiled:
    """A test reads rows; ``analysis.compute`` computes it (never statistics in SQL)."""
    outcome: str | None
    if plan.answer in ("odds_ratio", "survival") and plan.numerator:
        outcome = plan.numerator[0].column
    else:
        outcome = plan.measure
    group = plan.group_by
    time = plan.measure if plan.answer == "survival" else None
    if plan.answer == "survival":
        outcome = plan.numerator[0].column if plan.numerator else None
        if time is None or outcome is None:
            raise PlanError("survival needs measure = the time column and numerator = the event condition.")
    if outcome is None or (plan.answer != "survival" and group is None):
        raise PlanError(f"{plan.answer} needs the outcome (measure, or numerator for odds_ratio) and group_by.")
    names = [n for n in dict.fromkeys((context.patient_key, outcome, group, time)) if n]
    event = _all(plan.numerator, context) if plan.answer in ("odds_ratio", "survival") else None
    # A yes/no outcome is the numerator's condition as 1/0, under the outcome column's own name.
    select = ", ".join(
        f"CASE WHEN {event} THEN 1 WHEN NOT ({event}) THEN 0 END AS {ident(n)}" if event and n == outcome else ident(n)
        for n in names
    )
    where = _all(plan.filters, context)
    if context.visit_level and which in ("first", "last"):
        sql = f"SELECT {select} FROM ({_ranked(context, which)}) AS r{_where('_rank = 1', where)}"
    else:
        sql = f"SELECT {select} FROM unified{_where(where)}"
    return Compiled(
        plan=plan, kind="test", sql=sql, unit=context.row_nouns, per_patient=False, group_column=group,
        filters=plan.filters, numerator=plan.numerator, measure=plan.measure, columns=used, test_outcome=outcome,
        test_group=group, test_time=time,
    )  # fmt: skip


def varies_sql(context: DatasetContext, columns: list[str]) -> str:
    """Per column, then for any of them: patients whose value differs between their records."""
    key = ident(context.patient_key or "")
    counts = ", ".join(f"COUNT(DISTINCT {ident(c)}) AS d{i}" for i, c in enumerate(columns))
    each = ", ".join(f"COUNT_IF(d{i} > 1) AS v{i}" for i in range(len(columns)))
    either = " OR ".join(f"d{i} > 1" for i in range(len(columns)))
    return (
        f"SELECT {each}, COUNT_IF({either}) AS any_varies FROM "
        f"(SELECT {key}, {counts} FROM unified WHERE {key} IS NOT NULL GROUP BY {key}) AS p"
    )

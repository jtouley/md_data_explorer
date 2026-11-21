# SPDX-License-Identifier: MIT
"""The model's reply, checked against the table and compiled to the SQL that runs.

A reply is one JSON object. In the ``plan`` format the model fills a small plan (what to
compute, for which rows, per which group) and this module writes the SQL, so counting distinct
patients, shares and their denominators are always written the same, correct way. In the ``sql``
format the model writes the SELECT itself and this module checks it. Either way the conditions are
SQL the model wrote, checked here: every column must exist, a value compared with a column whose
values are known must be one of them (case and yes/no spellings are repaired), and nothing may
list individual patients.

What a count counts is never the model's choice: it is read from the question's own words and the
table's grain (``resolve_unit``). On visit rows patients are ``COUNT(DISTINCT <patient ID>)``.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias, get_args

import sqlglot
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlglot import exp
from sqlglot.errors import SqlglotError

from clinical_analytics.core import counting_unit
from clinical_analytics.model_query.context import ColumnInfo, DatasetContext
from clinical_analytics.sql.render import render

ReplyFormat: TypeAlias = Literal["plan", "sql"]
Tabulation: TypeAlias = Literal["count", "share", "mean", "median", "min", "max", "sum"]
StatTest: TypeAlias = Literal["compare_groups", "odds_ratio", "correlation", "survival"]
TABULATIONS: Final[tuple[str, ...]] = get_args(Tabulation)
STAT_TESTS: Final[tuple[str, ...]] = get_args(StatTest)
STATISTICS: Final = {"mean": "AVG", "median": "MEDIAN", "min": "MIN", "max": "MAX", "sum": "SUM"}
LISTING: Final = "listing"
SQL: Final = "sql"

LISTING_REASON: Final = (
    "This would list individual patients. The app answers with counts and summaries; "
    "ask how many there are, or for a breakdown by a group."
)
_ROW_WORDS = re.compile(r"\b(rows|records|lines|entries)\b", re.IGNORECASE)
_PEOPLE = r"(?:patients?|people|persons?|individuals|women|men|children|workers|subjects|participants|everyone)"
_RECORDS = r"(?:ids?|mrns?|names|charts?|record numbers?)"
_LISTING_PATTERNS: Final = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"\b(?:which|what) (?:of (?:the|our) )?{_PEOPLE}\b",
        r"^\s*who\b",
        rf"\b(?:list|name|identify|show|give me|pull)\b[^.?!]*\b(?:{_PEOPLE}|{_RECORDS})\b",
        r"\b(?:patient ids?|record numbers?|mrns?|chart numbers?)\b",
    )
)
_SUMMARY_WORDS: Final = re.compile(
    r"\b(how many|number of|count|share|proportion|percent\w*|rate|mean|average|median|odds|compare\w*)\b",
    re.IGNORECASE,
)


def asks_for_listing(question: str) -> bool:
    """Whether the question's own words ask for individual patients ("which patients ...", "list
    the ...", "their record numbers") and for no count or summary. The model is asked the same
    (answer "listing"); this catches the plain cases without it, so they are never answered with
    a table by mistake."""
    if _SUMMARY_WORDS.search(question):
        return False
    return any(pattern.search(question) for pattern in _LISTING_PATTERNS)


_YES = frozenset({"yes", "y", "true", "t", "1"})
_NO = frozenset({"no", "n", "false", "f", "0"})
_AGGREGATES: tuple[type[exp.Expr], ...] = (exp.AggFunc,)


class PlanError(ValueError):
    """The reply cannot be compiled or run as it is. The message is feedback for the model's next
    attempt and, if every attempt fails, the reason shown to the user: it names columns and
    listed category values only, never a row's values."""


class ModelReply(BaseModel):
    """One reply of the model, either format. Unknown keys are ignored; missing ones are None."""

    model_config = ConfigDict(extra="ignore")

    thought: str | None = None
    answer: str
    where: str | None = None
    group_by: str | None = None
    numerator: str | None = None  # share: the condition counted
    number: str | None = None  # mean, median, ...: the number column or expression
    outcome: str | None = None  # tests: the outcome column (or a yes/no condition)
    sql: str | None = None


_PLAN_FIELDS: Final = ("where", "group_by", "numerator", "number", "outcome")


def reply_schema(fmt: ReplyFormat) -> dict[str, Any]:
    """The JSON schema the model server holds the reply to (property order is generation order)."""
    text = {"type": ["string", "null"]}
    answers = [*TABULATIONS, *STAT_TESTS, LISTING] if fmt == "plan" else [SQL, *STAT_TESTS, LISTING]
    properties: dict[str, Any] = {"thought": {"type": "string"}, "answer": {"type": "string", "enum": answers}}
    if fmt == "sql":
        properties["sql"] = text
    properties |= dict.fromkeys(_PLAN_FIELDS, text)
    return {"type": "object", "properties": properties, "required": list(properties)}


def parse_reply(payload: Any) -> ModelReply:
    """Raises: PlanError when the payload is not a reply object."""
    if not isinstance(payload, dict):
        raise PlanError("The reply must be one JSON object.")
    cleaned = {
        k: (None if isinstance(v, str) and v.strip().lower() in ("", "null", "none") else v) for k, v in payload.items()
    }
    try:
        return ModelReply.model_validate(cleaned)
    except ValidationError as e:
        fields = ", ".join(".".join(str(p) for p in err["loc"]) for err in e.errors())
        raise PlanError(f"The reply's fields are not usable: {fields}.") from e


# ---------------------------------------------------------------------------- what is counted


@dataclass(frozen=True)
class Unit:
    """What a count counts: the ID column counted distinct (None: rows), and its words."""

    key: str | None
    word: str  # patients | visits | rows
    noun: str | None = None  # one encounter as the question says it ("admission")

    @property
    def plural(self) -> str:
        return f"{self.noun}s" if self.noun else self.word


def resolve_unit(question: str, context: DatasetContext) -> Unit:
    """The counted unit, from the question's words and the grain: an encounter noun ("admissions",
    "tests") counts its encounter ID; otherwise patients, by the patient ID when rows are visits."""
    if context.rows_are_patients:
        return Unit(None, "patients")
    words = counting_unit.read_units(question)
    if words.unit == "encounters" and words.noun:
        chosen = counting_unit.pick_encounter_key(list(context.encounter_keys), words.noun)
        if chosen is not None:
            return Unit(chosen, "visits", counting_unit.singular(words.noun))
    if context.patient_key is None or (words.unit is None and _ROW_WORDS.search(question)):
        return Unit(None, "rows")
    return Unit(context.patient_key, "patients")


# ---------------------------------------------------------------------------- checked expressions


def _parse_expression(text: str, what: str) -> exp.Expr:
    try:
        tree = sqlglot.parse_one(text, dialect="duckdb")
    except SqlglotError as e:
        raise PlanError(f"{what} is not valid SQL: {text!r}.") from e
    if tree is None or isinstance(tree, exp.Query | exp.Command | exp.DDL | exp.DML):
        raise PlanError(f"{what} must be one SQL expression, not a statement: {text!r}.")
    if any(isinstance(n, exp.Select | exp.Subquery) for n in tree.walk()):
        raise PlanError(f"{what} cannot contain a subquery.")
    for node in tree.walk():
        node.comments = None
    return tree


def _closest(name: str, context: DatasetContext) -> ColumnInfo | None:
    names = {c.name.casefold(): c for c in context.columns}
    near = difflib.get_close_matches(name.casefold(), list(names), n=2, cutoff=0.86)
    return names[near[0]] if len(near) == 1 else None


def _column_list(context: DatasetContext) -> str:
    return ", ".join(c.name for c in context.columns[:60])


def _other_side(column: exp.Column) -> exp.Expr | None:
    parent = column.parent
    if isinstance(parent, exp.EQ | exp.NEQ | exp.Is):
        return parent.expression if parent.this is column else parent.this
    return None


def resolve_columns(tree: exp.Expr, context: DatasetContext) -> exp.Expr:
    """Every column as the table names it, quoted. A double-quoted "value" that is not a column
    but is a value of the column it is compared with is read as the text it was meant to be."""

    def resolve(node: exp.Expr) -> exp.Expr:
        if not isinstance(node, exp.Column):
            return node
        if isinstance(node.this, exp.Star):
            raise PlanError(LISTING_REASON)
        found = context.column(node.name)
        if found is None:
            other = _other_side(node)
            partner = context.column(other.name) if isinstance(other, exp.Column) else None
            if partner is not None and partner.values is not None:
                return exp.Literal.string(node.name)  # sex = "Female": a value, not a column
            found = _closest(node.name, context)
        if found is None:
            raise PlanError(f"There is no column named {node.name}. The columns are: {_column_list(context)}.")
        return exp.column(found.name, quoted=True)

    return tree.transform(resolve)


def _yes_no(values: tuple[Any, ...]) -> dict[bool, Any]:
    """{True: the stored yes value, False: the stored no value} for a yes/no text column."""
    found: dict[bool, Any] = {}
    for value in values:
        folded = str(value).strip().casefold()
        if folded in _YES - {"1"}:
            found[True] = value
        elif folded in _NO - {"0"}:
            found[False] = value
    return found if len(found) == 2 else {}


def _fixed_literal(column: ColumnInfo, literal: exp.Expr) -> exp.Expr:
    """``literal`` as a value ``column`` holds: its stored spelling, a number for a number
    column, yes/no for true/false.

    Raises:
        PlanError: the column's values are all known and this is none of them.
    """
    if column.values is None or isinstance(literal, exp.Null):
        return literal
    if isinstance(literal, exp.Boolean) or (isinstance(literal, exp.Literal) and not literal.is_string):
        raw: Any = literal.this if isinstance(literal, exp.Boolean) else literal.name
    elif isinstance(literal, exp.Literal):
        raw = literal.name
    else:
        return literal
    text = str(raw).strip()
    for value in column.values:
        if str(value) == text:
            return _value_literal(value)
    for value in column.values:
        if str(value).casefold() == text.casefold():
            return _value_literal(value)
    if column.numeric:
        for value in column.values:
            try:
                if float(value) == float(text):
                    return _value_literal(value)
            except ValueError:
                break
    yes_no = _yes_no(column.values)
    if yes_no and text.casefold() in _YES | _NO:
        return _value_literal(yes_no[text.casefold() in _YES])
    # An abbreviation of exactly one value ('M' for 'male', 'F' for 'female').
    starts = [v for v in column.values if text and str(v).casefold().startswith(text.casefold())]
    if len(starts) == 1:
        return _value_literal(starts[0])
    shown = ", ".join(repr(v) for v in column.values)
    raise PlanError(f"Column {column.name} has no value {text!r}. Its values are: {shown}.")


def _value_literal(value: Any) -> exp.Expr:
    if isinstance(value, bool):
        return exp.true() if value else exp.false()
    if isinstance(value, int | float):
        return exp.Literal.number(value)
    return exp.Literal.string(str(value))


def fix_literals(tree: exp.Expr, context: DatasetContext) -> exp.Expr:
    """Repair or refuse every value compared with a column whose values are all known."""

    def spec(node: exp.Expr | None) -> ColumnInfo | None:
        return context.column(node.name) if isinstance(node, exp.Column) else None

    def is_value(node: exp.Expr | None) -> bool:
        return isinstance(node, exp.Literal | exp.Boolean)

    def fix(node: exp.Expr) -> exp.Expr:
        if isinstance(node, exp.EQ | exp.NEQ):
            left, right = node.this, node.expression
            if (column := spec(left)) is not None and is_value(right):
                node.set("expression", _fixed_literal(column, right))
            elif (column := spec(right)) is not None and is_value(left):
                node.set("this", _fixed_literal(column, left))
        elif isinstance(node, exp.In) and (column := spec(node.this)) is not None:
            node.set("expressions", [_fixed_literal(column, v) if is_value(v) else v for v in node.expressions])
        elif isinstance(node, exp.Column) and isinstance(node.parent, exp.And | exp.Or | exp.Not | exp.Where):
            # A bare yes/no column used as a condition (WHERE diabetic): compare it with its yes.
            column = spec(node)
            yes_no = _yes_no(column.values) if column is not None and column.values is not None else {}
            if yes_no:
                return exp.EQ(this=node.copy(), expression=_value_literal(yes_no[True]))
        return node

    return tree.transform(fix)


def _is_predicate(tree: exp.Expr) -> bool:
    """A yes/no condition on a row (a comparison, IN, BETWEEN, IS, or these joined by AND/OR)."""
    while isinstance(tree, exp.Paren):
        tree = tree.this
    return isinstance(tree, exp.Predicate | exp.Connector | exp.Not)


def _yes_condition(column: ColumnInfo | None) -> exp.Expr | None:
    """``column = <its yes value>`` for a yes/no column, else None."""
    yes_no = _yes_no(column.values) if column is not None and column.values is not None else {}
    if not yes_no or column is None:
        return None
    return exp.EQ(this=exp.column(column.name, quoted=True), expression=_value_literal(yes_no[True]))


def _value_owner(text: str, context: DatasetContext, prefer: ColumnInfo | None = None) -> exp.Expr | None:
    """``column = value`` when ``text`` is a listed value of ``prefer`` or of exactly one column."""
    wanted = text.strip().strip("'\"").casefold()

    def holds(column: ColumnInfo) -> Any:
        return next((v for v in column.values or () if str(v).casefold() == wanted), None)

    owners = (
        [prefer]
        if prefer is not None and holds(prefer) is not None
        else [c for c in context.columns if holds(c) is not None]
    )
    if len(owners) != 1 or owners[0] is None:
        return None
    return exp.EQ(this=exp.column(owners[0].name, quoted=True), expression=_value_literal(holds(owners[0])))


def _bare_column(text: str | None, context: DatasetContext) -> ColumnInfo | None:
    """The column ``text`` names when it is just a column name (quoted or not)."""
    if not text:
        return None
    return context.column(text.strip().strip('"`'))


def _condition(text: str | None, what: str, context: DatasetContext) -> exp.Expr | None:
    """A checked row condition, or None for no text.

    Raises:
        PlanError: not a condition, or a column or value the table does not have.
    """
    if not text or not text.strip():
        return None
    text = text.strip().rstrip(";")
    bare = _yes_condition(_bare_column(text, context))
    if bare is not None:
        return bare  # a yes/no column on its own: the rows where it is yes
    tree = _parse_expression(text, what)
    if any(isinstance(n, _AGGREGATES) for n in tree.walk()):
        raise PlanError(f"{what} is a condition on rows: it cannot use COUNT, SUM or another summary.")
    tree = fix_literals(resolve_columns(tree, context), context)
    if not _is_predicate(tree):
        raise PlanError(f"{what} must be a condition such as column = 'value', got {text!r}.")
    return tree


def _plain(node: exp.Expr) -> str:
    """An expression as a reader sees it: SQL without the identifier quotes."""

    def unquote(n: exp.Expr) -> exp.Expr:
        if isinstance(n, exp.Identifier) and re.fullmatch(r"[A-Za-z_]\w*", n.name):
            n.set("quoted", False)
        return n

    return render(node.copy().transform(unquote))


def _columns_of(node: exp.Expr | None) -> set[str]:
    return {c.name for c in node.find_all(exp.Column)} if node is not None else set()


def _terms(condition: exp.Expr | None) -> list[exp.Expr]:
    """The top-level AND terms of a condition."""
    if condition is None:
        return []
    node = condition
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _terms(node.this) + _terms(node.expression)
    return [node]


def _listed_values(term: exp.Expr) -> tuple[str, list[str]] | None:
    """(column, the values it is compared with) for ``c = v``, ``c IN (...)`` or an OR of those on
    one column; None for anything else."""
    while isinstance(term, exp.Paren):
        term = term.this
    if isinstance(term, exp.EQ) and isinstance(term.this, exp.Column) and isinstance(term.expression, exp.Literal):
        return term.this.name, [term.expression.name]
    if isinstance(term, exp.In) and isinstance(term.this, exp.Column) and term.expressions:
        if all(isinstance(v, exp.Literal) for v in term.expressions):
            return term.this.name, [v.name for v in term.expressions]
    if isinstance(term, exp.Or):
        left, right = _listed_values(term.this), _listed_values(term.expression)
        if left and right and left[0] == right[0]:
            return left[0], left[1] + right[1]
    return None


def _split_where(
    where: exp.Expr | None, group_column: str | None, context: DatasetContext
) -> tuple[exp.Expr | None, str | None]:
    """(the where without terms that limit nothing, a column those terms spelled out in full).

    A term that lists every value of a column ("sex = 'female' OR sex = 'male'") keeps every row:
    it was meant as a breakdown, so it is dropped and its column offered as the grouping.

    Raises:
        PlanError: a term pins the group_by column to one value, so the breakdown has one group.
    """
    kept: list[exp.Expr] = []
    spelled: str | None = None
    for term in _terms(where):
        listed = _listed_values(term)
        column = context.column(listed[0]) if listed else None
        if listed and column is not None and column.values is not None:
            named = {v.casefold() for v in listed[1]}
            if {str(v).casefold() for v in column.values} <= named:
                spelled = spelled or column.name
                continue
        if group_column is not None and _columns_of(term) == {group_column}:
            # A limit on the very column the answer is broken down by removes groups from it.
            raise PlanError(
                f"where limits {group_column}, but group_by {group_column} asks for each of its values. "
                f"Remove {group_column} from where to show every group; keep it only if the question "
                "says to leave some out, and then do not group by it."
            )
        if isinstance(term, exp.Or) and any(isinstance(side, exp.And) for side in (term.this, term.expression)):
            raise PlanError(
                "where mixes AND and OR without parentheses, so it is not clear which rows it keeps. "
                "Write it with parentheses, and keep only the limits the question states."
            )
        kept.append(term)
    return (_where_all(*kept), spelled)


_UNIT_SUFFIXES: Final = ("_unit", "_units", "_uom", " unit", " units")


def check_units(trees: list[exp.Expr | None], context: DatasetContext) -> None:
    """A number recorded in more than one unit must be converted before it is compared or
    summarised: a column ``x`` with a column ``x_unit`` holding several units, used without it.

    Raises:
        PlanError: naming the column, its unit column and the units.
    """
    used = set().union(*(_columns_of(t) for t in trees))
    for name in sorted(used):
        for suffix in _UNIT_SUFFIXES:
            unit = context.column(f"{name}{suffix}")
            if unit is None or unit.values is None or len(unit.values) < 2 or unit.name in used:
                continue
            units = ", ".join(repr(v) for v in unit.values)
            raise PlanError(
                f"{name} is recorded in different units (column {unit.name}: {units}). Convert it to one unit "
                f"first, e.g. CASE WHEN {unit.name} = {unit.values[0]!r} THEN {name} * <factor> ELSE {name} END, "
                "using the factor in the data dictionary."
            )


# ---------------------------------------------------------------------------- compiled queries


@dataclass(frozen=True)
class Compiled:
    """A tabulation ready to run: the SQL shown, its plain reading and how to present its rows."""

    kind: Literal["count", "share", "statistic", "table"]
    sql: str
    reading: str
    unit: Unit
    group: str | None = None  # the output's group column name
    group_column: str | None = None  # the table column grouped by, when it is one
    statistic: str | None = None
    value_text: str | None = None
    event_text: str | None = None
    limited: bool = False
    columns: frozenset[str] = frozenset()  # every table column the query reads
    total_sql: str | None = None  # the ungrouped count behind a grouped count
    # A share of patients on visit rows: how many patients meet the event on some rows but not
    # all (then "ever" and "always" give different shares, and the question must say which).
    varies_sql: str | None = None
    rollup: str | None = None  # any | all, for such a share


@dataclass(frozen=True)
class TestCall:
    """A statistical test for ``analysis.compute``: the rows to fetch and the variables to use."""

    test: StatTest
    sql: str  # the rows the test reads (shown as the method)
    reading: str
    outcome: str
    group: str | None
    time: str | None = None
    entity_key: str | None = None
    where_text: str | None = None
    columns: frozenset[str] = frozenset()  # every table column the call reads


def _base_table(context: DatasetContext, *, dedupe: bool) -> exp.Expr:
    """The table, or its distinct rows when exact duplicate rows would be counted twice."""
    table = exp.to_table(context.table)
    if dedupe and context.duplicate_rows:
        return exp.select("*").distinct().from_(table).subquery(context.table)
    return table


def _count(unit: Unit) -> exp.Expr:
    if unit.key is None:
        return exp.Count(this=exp.Star())
    return exp.Count(this=exp.Distinct(expressions=[exp.column(unit.key, quoted=True)]))


def _filtered(aggregate: exp.Expr, condition: exp.Expr) -> exp.Expr:
    return exp.Filter(this=aggregate, expression=exp.Where(this=condition))


def _group(text: str | None, context: DatasetContext) -> tuple[exp.Expr | None, str | None, str | None]:
    """(group expression, output name, table column when the group is one column)."""
    if not text or not text.strip():
        return None, None, None
    tree = resolve_columns(_parse_expression(text.strip(), "group_by"), context)
    if isinstance(tree, exp.Tuple | exp.Paren):
        raise PlanError("group_by takes one column. To split an outcome by a group, use a share with numerator.")
    if isinstance(tree, exp.Column):
        column = context.column(tree.name)
        if column is not None and column.is_patient_id:
            raise PlanError(LISTING_REASON)
        return tree, tree.name, tree.name
    if any(isinstance(n, _AGGREGATES) for n in tree.walk()):
        raise PlanError("group_by is a column or a row expression, not a summary.")
    return tree, _plain(tree), None


def _where_all(*conditions: exp.Expr | None) -> exp.Expr | None:
    """The conditions joined by AND, each term once (a model often states one twice)."""
    seen: dict[str, exp.Expr] = {}
    for condition in conditions:
        for term in _terms(condition):
            seen.setdefault(render(term), exp.paren(term.copy()) if isinstance(term, exp.Or) else term.copy())
    return exp.and_(*seen.values()) if seen else None


def _select(projections: list[exp.Expr], base: exp.Expr, where: exp.Expr | None, group: exp.Expr | None) -> exp.Select:
    select = exp.select(*projections).from_(base)
    if where is not None:
        select = select.where(where)
    if group is not None:
        select = select.group_by(exp.Literal.number(1)).order_by(exp.Literal.number(1))
    return select


def _where_phrase(where: exp.Expr | None) -> str:
    return f" where {_plain(where)}" if where is not None else ""


def _unit_phrase(unit: Unit) -> str:
    if unit.key is None:
        return unit.word
    return f"distinct {unit.plural} ({unit.key})"


def _count_conditions(reply: ModelReply, group_column: str | None, context: DatasetContext) -> list[exp.Expr]:
    """What a count's other fields add to its where: nothing the model wrote is dropped silently.
    A condition there limits the rows; a yes/no column there means its yes rows.

    Raises:
        PlanError: a field names a column without saying which of its values to count.
    """
    extra: list[exp.Expr] = []
    for name in ("numerator", "number", "outcome"):
        text = getattr(reply, name)
        if not text or not text.strip():
            continue
        column = _bare_column(text, context)
        if column is not None and _yes_condition(column) is None:
            if column.is_patient_id or column.is_encounter_id or column.values is None:
                continue  # the ID or a measurement named as what is counted: no limit
            if column.name == group_column:
                continue  # the grouping named twice
            example = column.values[0]
            raise PlanError(
                f"For a count, {name} must be null. To count the rows with one value of {column.name}, put it in "
                f"where (for example {column.name} = {example!r}); to count every value, set group_by to "
                f"{column.name}."
            )
        owned = _value_owner(text, context)
        condition = owned if owned is not None else _condition(text, name, context)
        if condition is not None:
            extra.append(condition)
    return extra


def _numerator(reply: ModelReply, context: DatasetContext) -> exp.Expr | str | None:
    """A share's event: a condition; or the name of a column whose distribution is meant; or None."""
    companion = _bare_column(reply.number, context) or _bare_column(reply.outcome, context)
    text = (reply.numerator or "").strip()
    if not text:
        yes = _yes_condition(companion)
        return yes if yes is not None else (companion.name if companion is not None and companion.values else None)
    column = _bare_column(text, context)
    if column is not None:
        yes = _yes_condition(column)
        return yes if yes is not None else column.name
    owned = _value_owner(text, context, prefer=companion)
    return owned if owned is not None else _condition(text, "numerator", context)


def _share_select(
    unit: Unit,
    event: exp.Expr,
    base: exp.Expr,
    where: exp.Expr | None,
    group: exp.Expr | None,
    group_name: str | None,
    rollup: str,
) -> exp.Select:
    """The share query. Rows: the event is tested on each row. A keyed unit on visit rows: each
    unit is first rolled up over its rows (met on any row, or on all its recorded rows)."""
    met: exp.Expr = event.copy()
    source: exp.Expr = base
    outer_where = where
    group_out: exp.Expr | None = group
    if unit.key is not None:
        key = exp.column(unit.key, quoted=True)
        roll = exp.func("BOOL_AND" if rollup == "all" else "BOOL_OR", event.copy())
        inner_cols: list[exp.Expr] = [key.copy(), exp.alias_(roll, "met")]
        if group is not None:
            inner_cols.insert(0, exp.alias_(group.copy(), group_name, quoted=True))
        keyed = _where_all(where, exp.Is(this=key.copy(), expression=exp.null()).not_())
        inner = exp.select(*inner_cols).from_(base).where(keyed)
        inner = inner.group_by(*[exp.Literal.number(i + 1) for i in range(len(inner_cols) - 1)])
        source = inner.subquery(f"per_{unit.noun or 'patient'}")
        met = exp.column("met")
        outer_where = None
        group_out = exp.column(group_name, quoted=True) if group_name else None
    recorded = exp.Is(this=exp.paren(met.copy()) if unit.key is None else met.copy(), expression=exp.null()).not_()
    events = _filtered(exp.Count(this=exp.Star()), met.copy())
    n = _filtered(exp.Count(this=exp.Star()), recorded)
    rate = exp.Div(
        this=exp.Mul(this=events.copy(), expression=exp.Literal.number("1.0")),
        expression=exp.func("NULLIF", n.copy(), exp.Literal.number(0)),
    )
    projections: list[exp.Expr] = [exp.alias_(events, "events"), exp.alias_(n, "n"), exp.alias_(rate, "rate")]
    if group_out is not None:
        projections.insert(0, exp.alias_(group_out.copy(), group_name, quoted=True))
    return _select(projections, source, outer_where, group_out)


def _varies_sql(unit: Unit, event: exp.Expr, base: exp.Expr, where: exp.Expr | None, group: exp.Expr | None) -> str:
    key = exp.column(unit.key, quoted=True)
    by: list[exp.Expr] = [key.copy()] + ([group.copy()] if group is not None else [])
    having = exp.and_(exp.func("BOOL_OR", event.copy()), exp.Not(this=exp.func("BOOL_AND", event.copy())))
    inner = exp.select(key.copy()).from_(base)
    if where is not None:
        inner = inner.where(where.copy())
    inner = inner.group_by(*by).having(having)
    return render(exp.select(exp.alias_(exp.Count(this=exp.Star()), "n")).from_(inner.subquery("varying")))


def compile_plan(reply: ModelReply, question: str, context: DatasetContext) -> Compiled:
    """The SQL for a tabulation plan.

    Raises:
        PlanError: the plan names a column or value the table does not have, or is incomplete.
    """
    compiled = _compile_plan(reply, question, context)
    return dataclasses.replace(compiled, columns=_sql_columns(compiled.sql, context))


def _sql_columns(sql: str, context: DatasetContext) -> frozenset[str]:
    """The table's columns a generated or checked statement reads."""
    found = {column.name.casefold() for column in sqlglot.parse_one(sql, dialect="duckdb").find_all(exp.Column)}
    return frozenset(c.name for c in context.columns if c.name.casefold() in found)


def _compile_plan(reply: ModelReply, question: str, context: DatasetContext) -> Compiled:
    answer = reply.answer
    if answer not in TABULATIONS:
        raise PlanError(f"answer must be one of: {', '.join(TABULATIONS)}.")
    unit = resolve_unit(question, context)
    group, group_name, group_column = _group(reply.group_by, context)
    where = _condition(reply.where, "where", context)
    event: exp.Expr | None = None
    if answer == "share":
        numerator = _numerator(reply, context)
        if isinstance(numerator, str):
            # "What share of patients is in each category of X": the distribution over X.
            answer = "count"
            if group is None:
                group, group_name, group_column = _group(f'"{numerator}"', context)
        elif numerator is None and where is not None:
            # The model wrote the counted condition as the limit: a share of it among everyone.
            event, where = where, None
        elif numerator is None or (group is not None and _columns_of(numerator) <= _columns_of(group)):
            if group is None:
                raise PlanError("A share needs numerator: the SQL condition that is counted, e.g. col = 'yes'.")
            answer = "count"
        else:
            event = numerator
            # A population limited to those with the event would make every share 100%: a limit
            # that only repeats the numerator is the numerator, not a limit.
            counted = {render(term) for term in _terms(event)}
            where = _where_all(*[term for term in _terms(where) if render(term) not in counted])
    elif answer == "count":
        where = _where_all(where, *_count_conditions(reply, group_column, context))
    where, spelled = _split_where(where, group_column, context)
    if spelled is not None and group is None and answer == "count":
        group, group_name, group_column = _group(f'"{spelled}"', context)
    elif spelled is not None and group_column not in (None, spelled) and answer == "count":
        raise PlanError(
            f"where lists every value of {spelled}, which keeps every row, while group_by is {group_column}. "
            f"To show each value of {spelled}, set group_by to {spelled} and where to null."
        )
    number: exp.Expr | None = None
    if answer in STATISTICS:
        if not reply.number or not reply.number.strip():
            raise PlanError(f"A {answer} needs number: the number column to summarise.")
        number = resolve_columns(_parse_expression(reply.number.strip(), "number"), context)
        if any(isinstance(n, _AGGREGATES) for n in number.walk()):
            raise PlanError("number is a column or row expression; the app applies the summary itself.")
        if _is_predicate(number):
            raise PlanError("number must be a number column, not a condition. Put conditions in where.")
    check_units([where, event, number], context)
    not_null = exp.Is(this=group.copy(), expression=exp.null()).not_() if group is not None else None
    full_where = _where_all(where, not_null)
    by = f", by {group_name}" if group_name else ""
    group_projection = [exp.alias_(group.copy(), group_name, quoted=True)] if group is not None else []

    if answer == "count":
        base = _base_table(context, dedupe=unit.key is None)
        select = _select([*group_projection, exp.alias_(_count(unit), "count")], base, full_where, group)
        total = _select([exp.alias_(_count(unit), "count")], base, full_where, None) if group is not None else None
        return Compiled(
            kind="count",
            sql=render(select, pretty=True),
            reading=f"Counted {_unit_phrase(unit)}{_where_phrase(where)}{by}.",
            unit=unit,
            group=group_name,
            group_column=group_column,
            limited=where is not None,
            total_sql=render(total) if total is not None else None,
        )

    if answer == "share":
        assert event is not None
        words = counting_unit.read_units(question)
        rollup = words.rollup or "any"
        base = _base_table(context, dedupe=unit.key is None)
        select = _share_select(unit, event, base, full_where, group, group_name, rollup)
        rolled = ""
        if unit.key is not None:
            one = unit.noun or "patient"
            rolled = (
                f" One {one} counts only when all of its rows that record it meet it."
                if rollup == "all"
                else f" One {one} counts when at least one of its rows meets it."
            )
        return Compiled(
            kind="share",
            sql=render(select, pretty=True),
            reading=(
                f"Share of {_unit_phrase(unit)} with {_plain(event)}{_where_phrase(where)}{by}, "
                f"among those with it recorded.{rolled}"
            ),
            unit=unit,
            group=group_name,
            group_column=group_column,
            event_text=_plain(event),
            limited=where is not None,
            varies_sql=(
                _varies_sql(unit, event, base, full_where, group)
                if unit.key is not None and words.rollup is None
                else None
            ),
            rollup=rollup if unit.key is not None else None,
        )

    assert number is not None
    value = number
    if isinstance(value, exp.Column):
        column = context.column(value.name)
        if column is not None and (column.is_patient_id or column.is_encounter_id):
            raise PlanError(LISTING_REASON)
        if column is not None and column.text:
            value = exp.TryCast(this=value, to=exp.DataType.build("DOUBLE"))
    # The table's label: the column, or "<column> (converted)" for an expression over one.
    used = sorted(_columns_of(number))
    value_text = _plain(number)
    label = value_text if isinstance(number, exp.Column) or not used else f"{used[0]} (converted)"
    base = _base_table(context, dedupe=True)
    double = exp.DataType.build("DOUBLE")

    def stat(name: str) -> exp.Expr:
        return exp.Cast(this=exp.func(STATISTICS[name], value.copy()), to=double.copy())

    n_values = exp.alias_(exp.Count(this=value.copy()), "n")
    if group is not None:
        projections = [*group_projection, exp.alias_(stat(answer), answer), n_values]
    else:
        names = [answer, *(s for s in ("mean", "median", "min", "max") if s != answer)]
        projections = [exp.alias_(stat(s), s) for s in names]
        projections += [exp.alias_(exp.Cast(this=exp.func("STDDEV_SAMP", value.copy()), to=double.copy()), "std")]
        projections += [n_values, exp.alias_(exp.Count(this=exp.Star()), "rows")]
    select = _select(projections, base, full_where, group)
    over = "" if context.rows_are_patients else f", over {context.row_unit} (rows), not one value per patient"
    return Compiled(
        kind="statistic",
        sql=render(select, pretty=True),
        reading=f"{answer.capitalize()} of {value_text}{_where_phrase(where)}{by}{over}.",
        unit=Unit(None, context.row_unit),
        group=group_name,
        group_column=group_column,
        statistic=answer,
        value_text=label,
        limited=where is not None,
    )


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:48] or "outcome"


def compile_test(reply: ModelReply, context: DatasetContext) -> TestCall:
    """The rows and variables of a statistical test. The outcome is a column, or a yes/no
    condition (``covid19_diagnosis = 'confirmed'``) read as an outcome of its own.

    Raises:
        PlanError: a variable is missing, is not a column of the table, or is used twice.
    """
    test: StatTest = reply.answer  # type: ignore[assignment]

    def column(text: str | None, what: str) -> str | None:
        if not text or not text.strip():
            return None
        tree = resolve_columns(_parse_expression(text.strip(), what), context)
        if not isinstance(tree, exp.Column):
            raise PlanError(f"{what} must be one column name for a {test} test, got {text!r}.")
        return tree.name

    group = column(reply.group_by, "group_by")
    if group is None and test != "survival":
        raise PlanError(f"A {test} test needs group_by: the group or exposure column.")
    derived: exp.Expr | None = None
    outcome: str | None = None
    time: str | None = None
    if test == "survival":
        time, outcome = column(reply.number, "number (the time column)"), column(reply.outcome, "outcome")
        if time is None or outcome is None:
            raise PlanError("A survival test needs number (the time column) and outcome (the event column).")
        time, outcome = outcome, time  # TestCall: outcome holds the time, time the event (as compute reads them)
    else:
        # The outcome as a condition, wherever the model wrote it; a condition on the group
        # column itself only says which group is the exposed one, so it is not the outcome.
        for text in (reply.outcome, reply.number, reply.numerator):
            if not text or not text.strip() or _bare_column(text, context) is not None:
                continue
            owned = _value_owner(text, context, prefer=_bare_column(reply.outcome, context))
            condition = owned if owned is not None else _condition(text, "outcome", context)
            if condition is None or _columns_of(condition) <= {group}:
                continue
            if test in ("compare_groups", "odds_ratio"):
                derived, outcome = condition, _slug(_plain(condition))
                break
        if outcome is None:
            named = next(
                (c for t in (reply.outcome, reply.number, reply.numerator) if (c := _bare_column(t, context))), None
            )
            outcome = named.name if named is not None else None
        if outcome is None:
            raise PlanError(f"A {test} test needs outcome: the outcome column.")
    if outcome == group or (derived is not None and group in _columns_of(derived)):
        raise PlanError(
            f"outcome and group_by are both {group}. The outcome is what is compared; group_by is the column "
            "whose groups are compared."
        )
    where, _spelled = _split_where(_condition(reply.where, "where", context), None, context)
    check_units([where, derived], context)
    key = context.patient_key if not context.rows_are_patients else None
    selected: list[exp.Expr] = []
    if derived is not None:
        yes_no = exp.case().when(derived.copy(), exp.Literal.string("yes"))
        yes_no = yes_no.when(exp.Not(this=exp.paren(derived.copy())), exp.Literal.string("no"))
        selected.append(exp.alias_(yes_no, outcome, quoted=True))
    names = list(dict.fromkeys(n for n in (None if derived is not None else outcome, group, time, key) if n))
    selected += [exp.column(n, quoted=True) for n in names]
    select = exp.select(*selected).from_(_base_table(context, dedupe=True))
    if where is not None:
        select = select.where(where)
    shown = _plain(derived) if derived is not None else outcome
    labels = {
        "compare_groups": f"Compared {shown} between the groups of {group}",
        "odds_ratio": f"Odds ratio of {shown} for {group} (logistic regression, unadjusted)",
        "correlation": f"Correlation between {shown} and {group}",
        "survival": f"Survival: time {outcome}, event {time}" + (f", by {group}" if group else ""),
    }
    return TestCall(
        test=test,
        sql=render(select, pretty=True),
        reading=f"{labels[test]}{_where_phrase(where)}.",
        outcome=outcome,
        group=group,
        time=time,
        entity_key=key,
        where_text=_plain(where) if where is not None else None,
        columns=_sql_columns(render(select), context),
    )


# ---------------------------------------------------------------------------- model-written SQL

_RATE_HINT = re.compile(r"(rate|share|proportion|percent|pct|fraction|prevalence)", re.IGNORECASE)


def compile_sql(reply: ModelReply, question: str, context: DatasetContext) -> Compiled:
    """Check a SELECT the model wrote and read how its result should be presented.

    Raises:
        PlanError: not one SELECT over the table, a column or value the table does not have, a
            listing of individual rows, or rows counted as patients on visit data.
    """
    if not reply.sql or not reply.sql.strip():
        raise PlanError("answer is sql, so the sql field must hold one SELECT statement.")
    try:
        trees = sqlglot.parse(reply.sql.strip().rstrip(";"), dialect="duckdb")
    except SqlglotError as e:
        raise PlanError("The sql is not valid DuckDB SQL.") from e
    if len(trees) != 1 or not isinstance(trees[0], exp.Select):
        raise PlanError("Write exactly one SELECT statement.")
    tree = trees[0]
    for node in tree.walk():
        node.comments = None
    tables = {t.name.casefold() for t in tree.find_all(exp.Table)}
    cte_names = {c.alias.casefold() for c in tree.find_all(exp.CTE)}
    if not tables or tables - cte_names - {context.table.casefold()}:
        raise PlanError(f"Read only the table {context.table}.")
    aliases = {p.alias.casefold() for p in tree.find_all(exp.Alias)} | cte_names

    def resolve(node: exp.Expr) -> exp.Expr:
        if isinstance(node, exp.Column) and not isinstance(node.this, exp.Star):
            if context.column(node.name) is None and node.name.casefold() in aliases:
                return node
            resolved = resolve_columns(exp.column(node.name), context)
            if isinstance(resolved, exp.Column) and node.table:
                resolved.set("table", node.args.get("table"))
            return resolved
        return node

    tree = fix_literals(tree.transform(resolve), context)
    assert isinstance(tree, exp.Select)
    check_units([tree], context)
    projections = list(tree.expressions)
    bodies = [p.this if isinstance(p, exp.Alias) else p for p in projections]
    measures = [i for i, b in enumerate(bodies) if any(isinstance(n, _AGGREGATES) for n in b.walk())]
    dimensions = [i for i in range(len(bodies)) if i not in measures]
    if not measures or any(isinstance(b, exp.Star) for b in bodies):
        raise PlanError(LISTING_REASON)
    for i in dimensions:
        for column in bodies[i].find_all(exp.Column):
            spec = context.column(column.name)
            if spec is not None and spec.is_patient_id:
                raise PlanError(LISTING_REASON)
    unit = resolve_unit(question, context)
    counts = [n for n in tree.find_all(exp.Count)]
    if unit.key is not None:
        for count in counts:
            distinct = isinstance(count.this, exp.Distinct)
            if not distinct:
                raise PlanError(
                    f"Rows are visits, not patients: COUNT(*) counts a patient several times. "
                    f'Use COUNT(DISTINCT "{unit.key}") to count {unit.plural}.'
                )
    kind: Literal["count", "share", "statistic", "table"] = "table"
    statistic: str | None = None
    value_text: str | None = None
    first = bodies[measures[0]]
    name = projections[measures[0]].alias_or_name
    if isinstance(first, exp.Count | exp.CountIf | exp.Filter) and not _RATE_HINT.search(name):
        kind = "count"
    elif isinstance(first, exp.Avg | exp.Median | exp.Min | exp.Max | exp.Sum) and isinstance(
        first.this, exp.Column | exp.Cast | exp.TryCast
    ):
        kind = "statistic"
        statistic = {exp.Avg: "mean", exp.Median: "median", exp.Min: "min", exp.Max: "max", exp.Sum: "sum"}[type(first)]
        value_text = _plain(first.this)
    elif isinstance(first, exp.Div | exp.Mul | exp.Avg | exp.Round | exp.Cast | exp.Paren) or _RATE_HINT.search(name):
        kind = "share"
    if len(dimensions) > 1 or (kind == "table" and len(measures) > 1):
        kind = "table"
    group_name = projections[dimensions[0]].alias_or_name if len(dimensions) == 1 else None
    group_body = bodies[dimensions[0]] if len(dimensions) == 1 else None
    where_node = tree.args.get("where")
    return Compiled(
        kind=kind,
        sql=render(tree, pretty=True),
        reading=(reply.thought or "Ran the query the model wrote.").strip(),
        unit=unit if kind in ("count", "share") else Unit(None, context.row_unit),
        group=group_name,
        group_column=group_body.name if isinstance(group_body, exp.Column) else None,
        statistic=statistic,
        value_text=value_text,
        event_text=name if kind == "share" else None,
        limited=where_node is not None,
        columns=_sql_columns(render(tree), context),
    )


# ---------------------------------------------------------------------------- unused mentions

# Words for the two values a sex column usually holds, whatever it spells them as.
_SEX_WORDS: Final = {
    "female": ("women", "woman", "female", "females", "girls"),
    "male": ("men", "man", "male", "males", "boys"),
}
_SEX_VALUES: Final = {"female": "female", "f": "female", "woman": "female", "male": "male", "m": "male", "man": "male"}


def _mentions(question: str, phrase: str) -> bool:
    words = re.sub(r"[^a-z0-9]+", " ", phrase.casefold()).strip()
    if len(words) < 3:
        return False
    text = " " + re.sub(r"[^a-z0-9]+", " ", question.casefold()) + " "
    return f" {words} " in text


def unused_mentions(question: str, used: frozenset[str], context: DatasetContext) -> list[tuple[str, str]]:
    """(the word of the question, the column) when the question says "women" or "men" while the
    query reads the table's sex column nowhere: the limit a small model most often drops.

    Only these words are read: they name one thing, whatever the table. (Any other category value
    found in a question is as often an ordinary word, "patients" or "confirmed", and a hint about
    it makes the model add a limit that was never asked for.) A hint for the model's next attempt,
    never a reason to hold the question: the model may answer that it limits nothing.
    """
    found: list[tuple[str, str]] = []
    for column in context.columns:
        if column.values is None or not column.text or column.name in used:
            continue
        if {_SEX_VALUES.get(str(v).casefold()) for v in column.values} != {"female", "male"}:
            continue
        for value in column.values:
            phrases = _SEX_WORDS[_SEX_VALUES[str(value).casefold()]]
            said = next((phrase for phrase in phrases if _mentions(question, phrase)), None)
            if said is not None:
                found.append((said, column.name))
                break
    return found

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
from clinical_analytics.model_query.coverage import column_words, stem_word, word_tokens
from clinical_analytics.model_query.grounding import missing_choice
from clinical_analytics.model_query.perpatient import (
    PER_PATIENT_KINDS,
    QuantifierError,
    asks_per_patient,
    date_columns,
    order_column,
    quantifier,
    rollable,
    rolled_up,
)
from clinical_analytics.model_query.units import Unit, UnitError, head_noun, resolve_unit
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


def asks_for_summary(question: str) -> bool:
    """Whether the question's own words ask for a number: how many, a share, a mean, a test."""
    return bool(_SUMMARY_WORDS.search(question))


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
    # Each phrase of the question and what it became (``coverage``): {phrase, role, column}.
    mapping: list[dict[str, Any]] | None = None
    # A value per patient summarised across patients (``perpatient``): count | first | last |
    # span_days | gap_days; and the column that orders a patient's visits, when the table has several.
    per_patient: str | None = None
    order_by: str | None = None


_PLAN_FIELDS: Final = ("where", "group_by", "numerator", "number", "outcome")
MAPPING_ROLES: Final = ("filter", "group", "measure", "unit", "ignored")


def reply_schema(fmt: ReplyFormat) -> dict[str, Any]:
    """The JSON schema the model server holds the reply to (property order is generation order)."""
    text = {"type": ["string", "null"]}
    answers = [*TABULATIONS, *STAT_TESTS, LISTING] if fmt == "plan" else [SQL, *STAT_TESTS, LISTING]
    link = {
        "type": "object",
        "properties": {
            "phrase": {"type": "string"},
            "role": {"type": "string", "enum": list(MAPPING_ROLES)},
            "column": text,
        },
        "required": ["phrase", "role", "column"],
    }
    properties: dict[str, Any] = {
        "thought": {"type": "string"},
        "mapping": {"type": "array", "items": link},
        "answer": {"type": "string", "enum": answers},
    }
    if fmt == "sql":
        properties["sql"] = text
    properties |= dict.fromkeys(_PLAN_FIELDS, text)
    if fmt == "plan":
        properties["per_patient"] = {"type": ["string", "null"], "enum": [*PER_PATIENT_KINDS, None]}
        properties["order_by"] = text
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
        side: exp.Expr = parent.expression if parent.this is column else parent.this
        return side
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


# Words a question or a model uses for a label a dictionary gives a code.
_LABEL_WORDS: Final = {
    "women": "female", "woman": "female", "females": "female", "f": "female", "girls": "female",
    "men": "male", "man": "male", "males": "male", "m": "male", "boys": "male",
    "y": "yes", "true": "yes", "n": "no", "false": "no",
}  # fmt: skip


def yes_no_of(column: ColumnInfo) -> dict[bool, Any]:
    """{True: the stored yes value, False: the stored no value}, from the values or, for a coded
    column (0/1, 1/2), from the dictionary's labels."""
    if column.values is None:
        return {}
    found = _yes_no(column.values)
    if found:
        return found
    yes, no = column.code_for("yes"), column.code_for("no")
    if yes is not None and no is not None:
        return {True: yes, False: no}
    pair = column.negated_pair()
    return {True: pair[0], False: pair[1]} if pair is not None else {}


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
    # A dictionary label ("female" for code 2), or a word for one ("women").
    for wanted in (text, _LABEL_WORDS.get(text.casefold(), "")):
        coded = column.code_for(wanted) if wanted else None
        if coded is not None:
            return _value_literal(coded)
    yes_no = yes_no_of(column)
    if yes_no and text.casefold() in _YES | _NO:
        return _value_literal(yes_no[text.casefold() in _YES])
    # An abbreviation of exactly one value ('M' for 'male', 'F' for 'female').
    starts = [v for v in column.values if text and str(v).casefold().startswith(text.casefold())]
    if len(starts) == 1:
        return _value_literal(starts[0])
    shown = ", ".join(repr(v) for v in column.values)
    raise PlanError(
        f"Column {column.name} has no value {text!r}. Its values are: {shown}. If none of them is what the "
        "question says, remove that condition: a word that names what the whole table is about is not a limit."
    )


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
            column = spec(left)
            if column is not None and isinstance(right, exp.Literal) and right.is_string and not right.name.strip():
                # col = '' / col != '' on any column: "recorded or not", never a value to match.
                missing = exp.Is(this=left.copy(), expression=exp.null())
                return missing if isinstance(node, exp.EQ) else missing.not_()
            if (column := spec(left)) is not None and is_value(right):
                node.set("expression", _fixed_literal(column, right))
            elif (column := spec(right)) is not None and is_value(left):
                node.set("this", _fixed_literal(column, left))
        elif isinstance(node, exp.In) and (column := spec(node.this)) is not None:
            node.set("expressions", [_fixed_literal(column, v) if is_value(v) else v for v in node.expressions])
        elif isinstance(node, exp.Column) and isinstance(node.parent, exp.And | exp.Or | exp.Not | exp.Where):
            # A bare yes/no column used as a condition (WHERE diabetic): compare it with its yes.
            column = spec(node)
            yes_no = yes_no_of(column) if column is not None else {}
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
    yes_no = yes_no_of(column) if column is not None else {}
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
        if group_column is not None and _columns_of(term) == {group_column} and listed is not None:
            # Pinning the very column the answer is broken down by to some of its values leaves
            # those groups only. (A range or a presence test on it, "each year since 2020", is a
            # limit like any other, checked against the question by ``grounding``.)
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
    # Columns the compiler itself reads to order a patient's visits: not the question's.
    helper_columns: frozenset[str] = frozenset()
    # A time operation the plan applies (each patient's first or last value, the latest period).
    time_op: str | None = None
    # A share whose condition reads two or more columns: how many units have it unknown (a value
    # missing), when the question did not say what a missing value means.
    missing_sql: str | None = None
    # A summary over one value per patient on visit rows: how many patients have more than one
    # value (then "each visit" and "each patient" differ, and the question must say which).
    per_patient_varies_sql: str | None = None
    # The dictionary's labels of the group column's codes, shown instead of the codes.
    group_labels: tuple[tuple[str, str], ...] = ()
    # The limits it applies (where's AND terms, and a share's event), for the grounding check.
    conditions: tuple[exp.Expr, ...] = dataclasses.field(default=(), compare=False)
    group_expression: bool = False  # broken down by an expression rather than one column


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
    conditions: tuple[exp.Expr, ...] = dataclasses.field(default=(), compare=False)  # where's AND terms


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


# A number column broken down by its own values must have at most this many.
MAX_NUMBER_GROUPS: Final = 20
_NUM = r"(\d+(?:\.\d+)?)"
_BIN_PATTERNS: Final = (
    ("below", re.compile(rf"\b(?:under|below|less than|younger than|lower than)\s+{_NUM}|<\s*{_NUM}", re.I)),
    (
        "range",
        re.compile(rf"\b{_NUM}\s*(?:to|-|–|through|and)\s*{_NUM}\b(?!\s*(?:or|and)\s+(?:over|older|above|more))", re.I),
    ),
    ("from", re.compile(rf"\b{_NUM}\s*(?:\+|or (?:over|older|more|above)|and (?:over|older|above|up|more))", re.I)),
    ("above", re.compile(rf"\b(?:over|above|older than|more than|greater than)\s+{_NUM}|>\s*{_NUM}", re.I)),
)


def stated_bins(question: str) -> list[tuple[float | None, float | None, bool, str]]:
    """The numeric ranges a question lists for a breakdown ("under 40, 40 to 64, 65 and over"), as
    (low, high, high_inclusive, label) sorted by low; empty unless at least two ranges are stated
    and they do not overlap. A range "A to B" includes B; "under A" excludes A; "A and over"
    includes A; "over A" excludes A."""
    found: list[tuple[int, float | None, float | None, bool, bool, str]] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern in (_BIN_PATTERNS[2], _BIN_PATTERNS[1], _BIN_PATTERNS[0], _BIN_PATTERNS[3]):
        for m in pattern.finditer(question):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            numbers = [float(g) for g in m.groups() if g is not None]
            label = " ".join(m.group(0).split())
            if kind == "below":
                found.append((m.start(), None, numbers[0], False, False, label))
            elif kind == "range" and len(numbers) == 2 and numbers[0] <= numbers[1]:
                found.append((m.start(), numbers[0], numbers[1], True, True, label))
            elif kind == "from":
                found.append((m.start(), numbers[0], None, True, False, label))
            elif kind == "above":
                found.append((m.start(), numbers[0], None, False, False, label))
            else:
                continue
            taken.append((m.start(), m.end()))
    if len(found) < 2:
        return []
    bins = sorted(found, key=lambda b: float("-inf") if b[1] is None else b[1])
    for (_, _lo1, hi1, _li1, hi_inc1, _l1), (_, lo2, _hi2, lo_inc2, _hi_inc2, _l2) in zip(bins, bins[1:], strict=False):
        if hi1 is None or lo2 is None or lo2 < hi1 or (lo2 == hi1 and hi_inc1 and lo_inc2):
            return []  # overlapping or open in the middle: not a clean breakdown
    return (
        [(lo, hi, hi_inc, label) for _at, lo, hi, _lo_inc, hi_inc, label in bins]
        if all(b[1] is not None or b[2] is not None for b in bins)
        else []
    )


def _bins_case(value: exp.Expr, bins: list[tuple[float | None, float | None, bool, str]]) -> exp.Expr:
    """CASE over the stated ranges, labelled as the question words them; other values are NULL
    (and so left out of the breakdown)."""
    case = exp.Case()
    for low, high, high_inclusive, label in bins:
        conditions: list[exp.Expr] = []
        if low is not None:
            conditions.append(exp.GTE(this=value.copy(), expression=exp.Literal.number(low)))
        if high is not None:
            bound = exp.Literal.number(high)
            conditions.append(
                exp.LTE(this=value.copy(), expression=bound)
                if high_inclusive
                else exp.LT(this=value.copy(), expression=bound)
            )
        case = case.when(exp.and_(*conditions), exp.Literal.string(label))
    return case


_YEAR_WORDS = re.compile(r"\b(?:year|years|yearly|annual|annually|calendar year)\b", re.IGNORECASE)
_MONTH_WORDS = re.compile(r"\b(?:month|months|monthly)\b", re.IGNORECASE)


def _group(
    text: str | None, context: DatasetContext, question: str = ""
) -> tuple[exp.Expr | None, str | None, str | None]:
    """(group expression, output name, table column when the group is one column). A date column
    is grouped by its year or month, as the question says; never by each date."""
    if not text or not text.strip():
        return None, None, None
    tree = resolve_columns(_parse_expression(text.strip(), "group_by"), context)
    if isinstance(tree, exp.Tuple | exp.Paren):
        raise PlanError("group_by takes one column. To split an outcome by a group, use a share with numerator.")
    if isinstance(tree, exp.Column):
        column = context.column(tree.name)
        if column is not None and column.is_patient_id:
            raise PlanError(LISTING_REASON)
        if column is not None and column.numeric and not column.is_sequence:
            stated = stated_bins(question)
            if len(stated) >= 2:
                return _bins_case(tree, stated), f"{column.name}_group", None
            if column.distinct > MAX_NUMBER_GROUPS:
                raise PlanError(
                    f"group_by {column.name} has {column.distinct} different values, too many to show. Group by a "
                    "column of categories, or by the ranges the question states."
                )
        if column is not None and column.logical_type in ("date", "timestamp", "datetime"):
            if _MONTH_WORDS.search(question) and not _YEAR_WORDS.search(question):
                month = exp.func("STRFTIME", tree.copy(), exp.Literal.string("%Y-%m"))
                return month, f"{column.name}_month", None
            if _YEAR_WORDS.search(question):
                return exp.func("YEAR", tree.copy()), f"{column.name}_year", None
            raise PlanError(
                f"group_by {column.name} is a date: group by YEAR({column.name}) or by month, as the question says."
            )
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
        return unit.plural if unit.noun else unit.word
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
        roll: exp.Expr = exp.func("BOOL_AND" if rollup == "all" else "BOOL_OR", event.copy())
        if rollup == "none":
            roll = exp.Not(this=exp.paren(roll))
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
    if unit.key is None:
        raise PlanError("A per-patient roll-up needs the patient key column.")
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
    grouped = context.column(compiled.group_column) if compiled.group_column else None
    return dataclasses.replace(
        compiled,
        columns=_sql_columns(compiled.sql, context),
        # Only a code is replaced by its label (2 -> female, "F" -> female); a value that is already
        # a word ("yes", "never_treated") stays as stored, whatever the dictionary says about it.
        group_labels=tuple(
            (code, label) for code, label in (grouped.labels if grouped is not None else ()) if _is_code(code)
        ),
    )


def _is_code(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?|[A-Za-z]", value.strip()))


def _sql_columns(sql: str, context: DatasetContext) -> frozenset[str]:
    """The table's columns a generated or checked statement reads."""
    found = {column.name.casefold() for column in sqlglot.parse_one(sql, dialect="duckdb").find_all(exp.Column)}
    return frozenset(c.name for c in context.columns if c.name.casefold() in found)


_NUMBER_WORDS: Final = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10}
_OCCASIONS = (
    r"(?:visits?|encounters?|admissions?|appointments?|sessions?|tests?|draws?|measurements?|readings?|"
    r"stays?|episodes?|cycles?|records?|rows?)"
)
_AT_LEAST = re.compile(
    rf"\b(?:(more than|over) (one|two|three|\d+)|(at least|(?:a minimum of)) (one|two|three|four|five|\d+)|"
    rf"(two|three|four|five|\d+) or more|(multiple|several|repeat(?:ed)?))\s+(?:[a-z-]+\s+)?{_OCCASIONS}\b",
    re.IGNORECASE,
)
_WITHIN_DAYS = re.compile(
    r"\bwithin (\d+) days? (?:of|after) (?:the same patient's|the patient's|their|his or her|a) "
    r"(?:previous|prior|last|earlier) \w+",
    re.IGNORECASE,
)
_LATEST_PERIOD = re.compile(
    r"\b(?:in |during )?the (?:most recent|latest|last) (calendar )?(month|year)(?: in the (?:data|file|table))?\b",
    re.IGNORECASE,
)


def _min_rows(question: str, row_nouns: frozenset[str] = frozenset()) -> int | None:
    """N from "more than one visit", "at least two visits", "3 or more visits", "multiple visits"
    (``row_nouns``: the table's own word for a row, read like visits)."""
    for noun in sorted(row_nouns, key=len, reverse=True):
        question = re.sub(rf"\b{re.escape(noun)}\b", "visits", question, flags=re.IGNORECASE)
    found = _AT_LEAST.search(question)
    if found is None:
        return None
    more, more_n, least, least_n, plus_n, several = found.groups()

    def number(text: str) -> int:
        return _NUMBER_WORDS.get(text.casefold(), 0) or int(text)

    if more_n:
        return number(more_n) + 1
    if least_n:
        return number(least_n)
    if plus_n:
        return number(plus_n)
    return 2 if several else None


def _repeated_count(
    question: str,
    where: exp.Expr | None,
    group: exp.Expr | None,
    group_name: str | None,
    group_column: str | None,
    unit: Unit,
    context: DatasetContext,
    by: str,
) -> Compiled | None:
    """Patients with at least N rows meeting where ("more than one visit"), or None."""
    least = _min_rows(question, context.row_nouns)
    if least is None or least < 2 or unit.key is None or context.rows_are_patients:
        return None
    # "visit_no > 1" is the model's way of saying "more than one visit": the count says it already.
    sequence = {c.name for c in context.columns if c.is_sequence}
    where = _where_all(*[t for t in _terms(where) if not (_columns_of(t) and _columns_of(t) <= sequence)])
    key = exp.column(unit.key, quoted=True)
    inner_cols: list[exp.Expr] = [key.copy()]
    if group is not None:
        inner_cols.append(exp.alias_(group.copy(), group_name or "group", quoted=True))
    keyed = _where_all(where, exp.Is(this=key.copy(), expression=exp.null()).not_())
    inner = exp.select(*inner_cols).from_(_base_table(context, dedupe=True)).where(keyed)
    inner = inner.group_by(*[exp.Literal.number(i + 1) for i in range(len(inner_cols))]).having(
        exp.GTE(this=exp.Count(this=exp.Star()), expression=exp.Literal.number(least))
    )
    out_group = exp.column(group_name or "group", quoted=True) if group is not None else None
    projections: list[exp.Expr] = [exp.alias_(exp.Count(this=exp.Star()), "count")]
    if out_group is not None:
        projections.insert(0, exp.alias_(out_group.copy(), group_name or "group", quoted=True))
    select = _select(projections, inner.subquery("per_patient"), None, out_group)
    return Compiled(
        kind="count",
        sql=render(select, pretty=True),
        reading=f"Counted distinct patients ({unit.key}) with at least {least} rows{_where_phrase(where)}{by}.",
        unit=unit,
        group=group_name,
        group_column=group_column,
        limited=True,
        conditions=tuple(_terms(where)),
        group_expression=group is not None and group_column is None,
    )


def _return_within_count(
    question: str,
    where: exp.Expr | None,
    group: exp.Expr | None,
    group_name: str | None,
    group_column: str | None,
    unit: Unit,
    context: DatasetContext,
    by: str,
) -> Compiled | None:
    """Rows within N days of the same patient's previous row ("returns within 30 days"), or None.

    Raises:
        PlanError: no patient key or no single date column to measure the days with.
    """
    found = _WITHIN_DAYS.search(question)
    if found is None:
        return None
    dates = date_columns(context)
    if context.patient_key is None or len(dates) != 1:
        raise PlanError("Days since a patient's previous visit need a patient key and one date column.")
    days = int(found.group(1))
    key, date = exp.column(context.patient_key, quoted=True), exp.column(dates[0].name, quoted=True)
    gap = exp.DateDiff(
        this=date.copy(),
        expression=exp.Window(
            this=exp.func("LAG", date.copy()),
            partition_by=[key.copy()],
            order=exp.Order(expressions=[exp.Ordered(this=date.copy())]),
        ),
        unit=exp.var("DAY"),
    )
    cols: list[exp.Expr] = [exp.alias_(gap, "days_since_previous")]
    if group is not None:
        cols.insert(0, exp.alias_(group.copy(), group_name or "group", quoted=True))
    inner = exp.select(*cols).from_(_base_table(context, dedupe=True))
    if where is not None:
        inner = inner.where(where)
    out_group = exp.column(group_name or "group", quoted=True) if group is not None else None
    projections: list[exp.Expr] = [exp.alias_(exp.Count(this=exp.Star()), "count")]
    if out_group is not None:
        projections.insert(0, exp.alias_(out_group.copy(), group_name or "group", quoted=True))
    within = exp.LTE(this=exp.column("days_since_previous"), expression=exp.Literal.number(days))
    select = _select(projections, inner.subquery("visits"), within, out_group)
    return Compiled(
        kind="count",
        sql=render(select, pretty=True),
        reading=(
            f"Counted {_unit_phrase(unit)} within {days} days of the same patient's previous {dates[0].name}"
            f"{_where_phrase(where)}{by}."
        ),
        unit=unit,
        group=group_name,
        group_column=group_column,
        limited=True,
        conditions=tuple(_terms(where)),
        group_expression=group is not None and group_column is None,
    )


def _latest_period(question: str, context: DatasetContext) -> exp.Expr | None:
    """The rows of the most recent calendar month (or year) in the data, when the question asks.

    Raises:
        PlanError: the table has no single date column to tell it by.
    """
    found = _LATEST_PERIOD.search(question)
    if found is None:
        return None
    dates = date_columns(context)
    if len(dates) != 1:
        raise PlanError("'The most recent month' needs one date column; the table has none or several.")
    date = exp.column(dates[0].name, quoted=True)
    part = "month" if found.group(2).casefold() == "month" else "year"
    trunc = exp.func("DATE_TRUNC", exp.Literal.string(part), date.copy())
    latest = exp.select(exp.func("DATE_TRUNC", exp.Literal.string(part), exp.Max(this=date.copy()))).from_(
        exp.to_table(context.table)
    )
    return exp.EQ(this=trunc, expression=latest.subquery())


_AMONG = re.compile(
    r"^(?P<population>.*?\b(?:among|of those|of the|in)\b.*?)"
    r"\b(?P<rest>what (?:share|proportion|percent(?:age)?|fraction)|how many)\b(?P<event>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _among_split(question: str, where: exp.Expr, context: DatasetContext) -> tuple[exp.Expr, exp.Expr] | None:
    """(population, event) from a where of several terms, when the question's "among ..." part
    names the columns of some terms and its "what share ..." part those of the others; else None."""
    found = _AMONG.match(question)
    terms = _terms(where)
    if found is None or len(terms) < 2:
        return None

    def named(text: str, term: exp.Expr) -> bool:
        stems = {stem_word(w) for w in word_tokens(text)}
        columns = [c for name in _columns_of(term) if (c := context.column(name)) is not None]
        return any(column_words(c) & stems for c in columns)

    population = [t for t in terms if named(found.group("population"), t) and not named(found.group("event"), t)]
    event = [t for t in terms if named(found.group("event"), t) and not named(found.group("population"), t)]
    if not population or not event or len(population) + len(event) != len(terms):
        return None
    kept, counted = _where_all(*population), _where_all(*event)
    if kept is None or counted is None:
        return None
    return kept, counted


def _is_presence_test(term: exp.Expr) -> bool:
    """col IS NOT NULL (a value is recorded), however it is written."""
    node = term
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Not):
        inner = node.this
        while isinstance(inner, exp.Paren):
            inner = inner.this
        return isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null)
    return False


_MEASURE_NOUNS: Final = frozenset({"result", "measurement", "reading", "value", "level", "draw", "lab"})


def _measure_named_before(question: str, noun: str, context: DatasetContext) -> str | None:
    """The one number column the words just before the counted noun name ("INR results", "glucose
    readings"), or None."""
    words = re.findall(r"[a-z0-9]+", question.casefold())
    forms = {noun, noun + "s", noun + "es"}
    at = next((i for i, w in enumerate(words) if w in forms), None)
    if at is None:
        return None
    before = {stem_word(w) for w in words[max(0, at - 3) : at]}
    named = [
        c.name
        for c in context.columns
        if c.numeric
        and not c.is_sequence
        and {stem_word(w) for w in re.findall(r"[a-z0-9]+", c.name.casefold())} & before
    ]
    return named[0] if len(named) == 1 else None


def _tests_no_value(term: exp.Expr, context: DatasetContext) -> bool:
    """Whether a condition tests a yes/no column for its "no" value."""
    node = term
    while isinstance(node, exp.Paren):
        node = node.this
    if not isinstance(node, exp.EQ) or not isinstance(node.this, exp.Column):
        return False
    column = context.column(node.this.name)
    flags = yes_no_of(column) if column is not None else {}
    literal = node.expression
    if not flags or not isinstance(literal, exp.Literal | exp.Boolean):
        return False
    said = literal.this if isinstance(literal, exp.Boolean) else literal.name
    return str(said).casefold() == str(flags[False]).casefold()


def _quantified_count(
    question: str,
    where: exp.Expr | None,
    group: exp.Expr | None,
    group_name: str | None,
    group_column: str | None,
    unit: Unit,
    context: DatasetContext,
    by: str,
) -> Compiled | None:
    """Patients counted by a per-patient quantifier ("never", "on every visit"), or None when the
    question names none (then a row condition counts patients with it on any row).

    Raises:
        UnitError: the quantifier cannot be read without a guess.
    """
    try:
        how = quantifier(question, context.row_nouns)
    except QuantifierError as e:
        raise UnitError(str(e)) from e
    if how in (None, "ever") or context.rows_are_patients or unit.key is None:
        return None
    terms = _terms(where)
    rolled_terms = [t for t in terms if rollable(t, context, lambda c: bool(yes_no_of(c)))]
    if not rolled_terms:
        return None
    try:
        aggregates = [rolled_up(t, how, question, lambda t: _tests_no_value(t, context)) for t in rolled_terms]
    except QuantifierError as e:
        raise UnitError(str(e)) from e
    kept = _where_all(*[t for t in terms if t not in rolled_terms])
    key = exp.column(unit.key, quoted=True)
    inner_cols: list[exp.Expr] = [key.copy()]
    if group is not None:
        inner_cols.append(exp.alias_(group.copy(), group_name or "group", quoted=True))
    inner_cols.append(exp.alias_(exp.and_(*aggregates), "met"))
    keyed = _where_all(kept, exp.Is(this=key.copy(), expression=exp.null()).not_())
    if group is not None:
        keyed = _where_all(keyed, exp.Is(this=group.copy(), expression=exp.null()).not_())
    inner = exp.select(*inner_cols).from_(_base_table(context, dedupe=False)).where(keyed)
    inner = inner.group_by(*[exp.Literal.number(i + 1) for i in range(len(inner_cols) - 1)])
    out_group = exp.column(group_name or "group", quoted=True) if group is not None else None
    projections: list[exp.Expr] = [exp.alias_(exp.Count(this=exp.Star()), "count")]
    if out_group is not None:
        projections.insert(0, exp.alias_(out_group.copy(), group_name or "group", quoted=True))
    select = _select(projections, inner.subquery("per_patient"), exp.column("met"), out_group)
    said = " AND ".join(_plain(t) for t in rolled_terms)
    when = "on every one of their rows" if how == "always" else "on none of their rows"
    if how == "never" and all(isinstance(a, exp.Anonymous) or not isinstance(a, exp.Not) for a in aggregates):
        when = "on every one of their rows"
    return Compiled(
        kind="count",
        sql=render(select, pretty=True),
        reading=f"Counted distinct patients ({unit.key}) with {said} {when}{_where_phrase(kept)}{by}.",
        unit=unit,
        group=group_name,
        group_column=group_column,
        limited=True,
        conditions=tuple(terms),
        group_expression=group is not None and group_column is None,
    )


_STATISTIC_WORDS = re.compile(r"\b(?:median|average|mean|typical)\b", re.IGNORECASE)


def _per_patient_repairs(reply: ModelReply, question: str, context: DatasetContext) -> ModelReply:
    """A breakdown by, or a summary of, the patient key is a per-patient count summarised across
    patients ("average number of visits per patient"), never a listing. A question asking for a
    value per patient with no per_patient in the reply is fed back.

    Raises:
        PlanError: the question asks for a per-patient value and the reply has none.
    """
    if context.rows_are_patients or context.patient_key is None or _PER_PATIENT.search(question):
        return reply
    key = context.patient_key.casefold()

    def names_key(text: str | None) -> bool:
        return bool(text) and str(text).strip().strip('"`').casefold() == key

    if reply.per_patient:
        # "per patient" said as a breakdown by the patient key: the summary is across patients.
        return reply.model_copy(update={"group_by": None}) if names_key(reply.group_by) else reply

    asks_summary = bool(_STATISTIC_WORDS.search(question))
    if (names_key(reply.group_by) or names_key(reply.number)) and asks_summary:
        answer = (
            reply.answer if reply.answer in STATISTICS else ("median" if "median" in question.casefold() else "mean")
        )
        return reply.model_copy(update={"per_patient": "count", "answer": answer, "group_by": None, "number": None})
    if asks_per_patient(question) and asks_summary and reply.answer in STATISTICS:
        raise PlanError(
            "The question asks for a value per patient: set per_patient to count (the number of each patient's "
            "rows), first or last (each patient's first or last value of number), span_days (days from each "
            "patient's first to last visit) or gap_days (days between a patient's consecutive visits)."
        )
    return reply


def _per_patient_summary(reply: ModelReply, question: str, context: DatasetContext) -> Compiled:
    """A value per patient (or per consecutive pair of visits), then summarised across patients.

    Raises:
        PlanError: no patient key, no column to order visits by, or a value that is not a number.
    """
    kind = reply.per_patient
    if kind not in PER_PATIENT_KINDS:
        raise PlanError(f"per_patient must be one of: {', '.join(PER_PATIENT_KINDS)}.")
    if context.patient_key is None or context.rows_are_patients:
        raise PlanError("Each row is one patient here, so per_patient does not apply: set it to null.")
    answer = reply.answer if reply.answer in STATISTICS else "mean"
    key = exp.column(context.patient_key, quoted=True)
    where = _condition(reply.where, "where", context)
    group, group_name, group_column = _group(reply.group_by, context, question)
    base = _base_table(context, dedupe=True)
    value: exp.Expr
    what: str
    if kind == "count":
        counted: exp.Expr = exp.Count(this=exp.Star())
        value = _filtered(counted, where) if where is not None else counted
        what = "number of rows per patient" + (f" with {_plain(where)}" if where is not None else "")
        row_filter = None
    order = None
    if kind != "count":
        dates = date_columns(context)
        order = order_column(context, reply.order_by)
        if kind in ("span_days", "gap_days"):
            named = _bare_column(reply.number, context) or _bare_column(reply.order_by, context)
            date = named if named is not None and named in dates else (dates[0] if len(dates) == 1 else None)
            if date is None:
                choices = ", ".join(c.name for c in dates) or "none"
                raise PlanError(f"Days between visits need one date column in number (date columns: {choices}).")
            ref = exp.column(date.name, quoted=True)
            if kind == "span_days":
                value = exp.DateDiff(
                    this=exp.Max(this=ref.copy()), expression=exp.Min(this=ref.copy()), unit=exp.var("DAY")
                )
                what = f"days from each patient's first to last {date.name}"
            else:
                value = ref  # replaced below by the gap to the previous row
                what = f"days between a patient's consecutive {date.name} values"
            row_filter = where
        else:
            number = _bare_column(reply.number, context)
            if number is None or not number.numeric:
                raise PlanError(f"per_patient {kind} needs number: the number column whose {kind} value is taken.")
            if order is None:
                choices = ", ".join(c.name for c in [*dates, *[c for c in context.columns if c.is_sequence]])
                raise PlanError(f"Set order_by to the column that orders a patient's visits (one of: {choices}).")
            ref, by_col = exp.column(number.name, quoted=True), exp.column(order.name, quoted=True)
            func = "ARG_MIN" if kind == "first" else "ARG_MAX"
            value = exp.func(func, ref.copy(), by_col.copy())
            what = f"each patient's {kind} {number.name} (by {order.name})"
            row_filter = _where_all(where, exp.Is(this=ref.copy(), expression=exp.null()).not_())
    group_alias = group_name or "group"
    inner_cols: list[exp.Expr] = [key.copy()]
    inner_group: list[exp.Expr] = [key.copy()]
    if group is not None:
        inner_cols.append(exp.alias_(group.copy(), group_alias, quoted=True))
        inner_group.append(group.copy())
    keyed = _where_all(row_filter, exp.Is(this=key.copy(), expression=exp.null()).not_())
    if kind == "gap_days":
        gap = exp.DateDiff(
            this=value.copy(),
            expression=exp.Window(
                this=exp.func("LAG", value.copy()),
                partition_by=[key.copy()],
                order=exp.Order(expressions=[exp.Ordered(this=value.copy())]),
            ),
            unit=exp.var("DAY"),
        )
        inner = exp.select(*inner_cols, exp.alias_(gap, "patient_value")).from_(base)
        if keyed is not None:
            inner = inner.where(keyed)
        source = (
            exp.select("*")
            .from_(inner.subquery("pairs"))
            .where(exp.Is(this=exp.column("patient_value"), expression=exp.null()).not_())
        )
        source_q = source.subquery("per_pair")
    else:
        inner = exp.select(*inner_cols, exp.alias_(value, "patient_value")).from_(base)
        if keyed is not None:
            inner = inner.where(keyed)
        source_q = inner.group_by(*inner_group).subquery("per_patient")
    out_group = exp.column(group_alias, quoted=True) if group is not None else None
    double = exp.DataType.build("DOUBLE")
    v = exp.column("patient_value")
    projections: list[exp.Expr] = [
        exp.alias_(exp.Cast(this=exp.func(STATISTICS[answer], v.copy()), to=double.copy()), answer),
        exp.alias_(exp.Count(this=v.copy()), "n"),
    ]
    if out_group is not None:
        projections.insert(0, exp.alias_(out_group.copy(), group_alias, quoted=True))
    else:
        for extra in ("mean", "median", "min", "max"):
            if extra != answer:
                projections.insert(
                    -1, exp.alias_(exp.Cast(this=exp.func(STATISTICS[extra], v.copy()), to=double.copy()), extra)
                )
        projections.insert(-1, exp.alias_(exp.Cast(this=exp.func("STDDEV_SAMP", v.copy()), to=double.copy()), "std"))
        projections.append(exp.alias_(exp.Count(this=exp.Star()), "rows"))
    select = _select(projections, source_q, None, out_group)
    by = f", by {group_name}" if group_name else ""
    over = "pairs of consecutive visits" if kind == "gap_days" else "patients"
    return Compiled(
        kind="statistic",
        sql=render(select, pretty=True),
        reading=(
            f"{answer.capitalize()} of the {what}{_where_phrase(where) if kind != 'count' else ''}{by}, over {over}."
        ),
        unit=Unit(context.patient_key, "patients"),
        group=group_name,
        group_column=group_column,
        statistic=answer,
        value_text=what,
        limited=where is not None,
        conditions=tuple(_terms(where)),
        group_expression=group is not None and group_column is None,
        helper_columns=frozenset({order.name} if kind in ("first", "last") and order is not None else ()),
        time_op={"first": "first", "last": "last", "span_days": "both", "gap_days": "both"}.get(kind or ""),
    )


def _compile_plan(reply: ModelReply, question: str, context: DatasetContext) -> Compiled:
    answer = reply.answer
    if answer not in TABULATIONS:
        raise PlanError(f"answer must be one of: {', '.join(TABULATIONS)}.")
    try:
        unit = resolve_unit(question, context)
    except UnitError:
        if answer in ("count", "share"):
            raise
        unit = Unit(None, context.row_unit)  # a mean or median does not count the noun
    if reply.per_patient and answer not in STATISTICS:
        # A per-patient value is summarised by a mean, median, minimum or maximum. A count or share
        # with one is read from its own words ("more than one visit", "on every visit").
        reply = reply.model_copy(update={"per_patient": None})
    reply = _per_patient_repairs(reply, question, context)
    if reply.per_patient:
        return _per_patient_summary(reply, question, context)
    where = _where_all(_condition(reply.where, "where", context), _latest_period(question, context))
    group_text = reply.group_by
    if len(stated_bins(question)) >= 2:
        # "by age: under 40, 40 to 64, 65 and over": the ranges are the breakdown. Range limits
        # the model wrote in where on that column spell the same ranges and are dropped.
        binned = _bare_column(group_text, context) or next(
            (
                c
                for t in _terms(where)
                if len(cols := _columns_of(t)) == 1 and (c := context.column(next(iter(cols)))) and c.numeric
            ),
            None,
        )
        if binned is not None and binned.numeric:
            group_text = f'"{binned.name}"'
            where = _where_all(*[t for t in _terms(where) if _columns_of(t) != {binned.name}])
    group, group_name, group_column = _group(group_text, context, question)
    event: exp.Expr | None = None
    if answer == "share":
        numerator = _numerator(reply, context)
        if isinstance(numerator, str):
            # "What share of patients is in each category of X": the distribution over X.
            answer = "count"
            if group is None:
                group, group_name, group_column = _group(f'"{numerator}"', context)
        elif numerator is None and where is not None and (split := _among_split(question, where, context)):
            # "Among graded visits, what share had drops": the terms the "among" part names keep
            # the population; the rest is what is counted.
            where, event = split
        elif numerator is None and where is not None and len(_terms(where)) > 1:
            # Several conditions and no numerator: which one is counted and which keep the
            # population ("in patients over 50") is the model's to say, never guessed here.
            raise PlanError(
                "A share needs numerator: the condition that is counted (for example infection = 'yes'). Put in "
                "where only the conditions that keep the population the share is taken among (an age limit, "
                "a year, a sex)."
            )
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
        if event is not None:
            overlap = [t for t in _terms(where) if _columns_of(t) & _columns_of(event) and not _is_presence_test(t)]
            if overlap:
                # "Of the visits where X was answered, what share had X >= 1": answered means
                # recorded (IS NOT NULL); a limit on the counted column's values decides the share.
                shown = _plain(overlap[0])
                raise PlanError(
                    f"where limits {shown}, a condition on the column the share counts, so the share is decided "
                    "by the limit. Keep in where only who is counted (for 'answered' or 'recorded', use "
                    "col IS NOT NULL) and put the condition counted in numerator."
                )
        listed = _listed_values(event) if event is not None else None
        every = context.column(listed[0]) if listed else None
        if (
            listed is not None
            and every is not None
            and every.values is not None
            and {str(v).casefold() for v in every.values} <= {v.casefold() for v in listed[1]}
            and group_column in (None, every.name)
        ):
            # "What share is A, B or C" naming every value of one column: its distribution (the
            # count per value, with percentages), not one share that is always 100%.
            answer, event = "count", None
            group, group_name, group_column = _group(f'"{every.name}"', context)
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

    if answer == "count" and unit.key == context.patient_key and unit.key is not None:
        quantified = _quantified_count(question, where, group, group_name, group_column, unit, context, by)
        if quantified is not None:
            return quantified
        repeated = _repeated_count(question, where, group, group_name, group_column, unit, context, by)
        if repeated is not None:
            return repeated
    if answer == "count" and unit.key != context.patient_key:
        returns = _return_within_count(question, where, group, group_name, group_column, unit, context, by)
        if returns is not None:
            return returns
    if answer == "count" and unit.key is None and unit.noun in _MEASURE_NOUNS:
        # "How many INR results": the results of that measure are its recorded values.
        measured = _measure_named_before(question, unit.noun, context)
        if measured is not None:
            where = _where_all(where, exp.Is(this=exp.column(measured, quoted=True), expression=exp.null()).not_())
            full_where = _where_all(
                full_where, exp.Is(this=exp.column(measured, quoted=True), expression=exp.null()).not_()
            )
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
            conditions=tuple(_terms(where)),
            group_expression=group is not None and group_column is None,
        )

    if answer == "share":
        assert event is not None
        words = counting_unit.read_units(question)
        rollup: str = words.rollup or "any"
        try:
            how = quantifier(question, context.row_nouns) if unit.key is not None else None
        except QuantifierError as e:
            raise UnitError(str(e)) from e
        if how == "never":
            never = rolled_up(event, "never", question, lambda t: _tests_no_value(t, context))
            rollup = "none" if isinstance(never, exp.Not) else "all"
        base = _base_table(context, dedupe=unit.key is None)
        # A condition on two or more columns is unknown (NULL) where one of them is missing, and
        # "among those with it recorded" would then drop those rows silently: what a missing value
        # means is the asker's call, unless the question already said it.
        missing = missing_choice(question)
        missing_sql: str | None = None
        recorded_note = ", among those with it recorded"
        if missing in ("meets", "fails"):
            event = exp.func("COALESCE", exp.paren(event.copy()), exp.true() if missing == "meets" else exp.false())
            recorded_note = (
                ", a missing value counted as meeting it" if missing == "meets" else ", a missing value counted as not"
            )
        elif missing is None and len(_columns_of(event)) >= 2:
            unknown = exp.Is(this=exp.paren(event.copy()), expression=exp.null())
            missing_sql = render(
                _select([exp.alias_(_count(unit), "count")], base, _where_all(full_where, unknown), None)
            )
        select = _share_select(unit, event, base, full_where, group, group_name, rollup)
        rolled = ""
        if unit.key is not None:
            one = unit.noun or "patient"
            rolled = {
                "all": f" One {one} counts only when all of its rows that record it meet it.",
                "none": f" One {one} counts only when none of its rows meets it.",
            }.get(rollup, f" One {one} counts when at least one of its rows meets it.")
        return Compiled(
            kind="share",
            sql=render(select, pretty=True),
            reading=(
                f"Share of {_unit_phrase(unit)} with {_plain(event)}{_where_phrase(where)}{by}{recorded_note}.{rolled}"
            ),
            unit=unit,
            group=group_name,
            group_column=group_column,
            event_text=_plain(event),
            limited=where is not None,
            varies_sql=(
                _varies_sql(unit, event, base, full_where, group)
                if unit.key is not None and words.rollup is None and how is None
                else None
            ),
            rollup=rollup if unit.key is not None else None,
            missing_sql=missing_sql,
            conditions=(*_terms(where), *_terms(event)),
            group_expression=group is not None and group_column is None,
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

    def summary(of: exp.Expr, grouped_by: exp.Expr | None, out_group: list[exp.Expr]) -> list[exp.Expr]:
        def stat(name: str) -> exp.Expr:
            return exp.Cast(this=exp.func(STATISTICS[name], of.copy()), to=double.copy())

        n_values = exp.alias_(exp.Count(this=of.copy()), "n")
        if grouped_by is not None:
            return [*out_group, exp.alias_(stat(answer), answer), n_values]
        names = [answer, *(s for s in ("mean", "median", "min", "max") if s != answer)]
        projections = [exp.alias_(stat(s), s) for s in names]
        projections += [exp.alias_(exp.Cast(this=exp.func("STDDEV_SAMP", of.copy()), to=double.copy()), "std")]
        return [*projections, n_values, exp.alias_(exp.Count(this=exp.Star()), "rows")]

    # Over what: each row, or one value per patient. On visit rows a question about patients
    # ("average age of our patients") is about one value per patient; a question about events, or
    # one that says "every visit", is about rows. When a patient's value differs between visits,
    # the engine asks which (``per_patient_varies_sql``).
    mode = statistic_mode(question, context)
    varies_sql: str | None = None
    if mode == "rows":
        select = _select(summary(value, group, group_projection), base, full_where, group)
        over = "" if context.rows_are_patients else f", over {context.row_unit} (rows), not one value per patient"
    else:
        key = exp.column(context.patient_key or "", quoted=True)
        group_alias = group_name or "group"
        inner_projections: list[exp.Expr] = [key.copy()]
        inner_group: list[exp.Expr] = [key.copy()]
        if group is not None:
            inner_projections.append(exp.alias_(group.copy(), group_alias, quoted=True))
            inner_group.append(group.copy())
        inner_projections.append(exp.alias_(exp.func("AVG", value.copy()), "patient_value"))
        inner = exp.select(*inner_projections).from_(base)
        if full_where is not None:
            inner = inner.where(full_where)
        inner = inner.group_by(*inner_group)
        out_group_expr = exp.column(group_alias, quoted=True) if group is not None else None
        out_group = [exp.alias_(out_group_expr.copy(), group_alias, quoted=True)] if out_group_expr is not None else []
        projections = summary(exp.column("patient_value"), out_group_expr, out_group)
        select = _select(projections, inner.subquery("per_patient"), None, out_group_expr)
        over = ", one value per patient (the patient's average over their rows)"
        if mode == "patients_if_constant":
            over = ", one value per patient"
            patients = exp.select(key.copy()).from_(base)
            if full_where is not None:
                patients = patients.where(full_where)
            patients = patients.group_by(key.copy()).having(
                exp.GT(this=exp.Count(this=exp.Distinct(expressions=[value.copy()])), expression=exp.Literal.number(1))
            )
            varies_sql = render(
                exp.select(exp.alias_(exp.Count(this=exp.Star()), "n")).from_(patients.subquery("varying"))
            )
    return Compiled(
        kind="statistic",
        sql=render(select, pretty=True),
        per_patient_varies_sql=varies_sql,
        reading=f"{answer.capitalize()} of {value_text}{_where_phrase(where)}{by}{over}.",
        unit=Unit(None, context.row_unit),
        group=group_name,
        group_column=group_column,
        statistic=answer,
        value_text=label,
        limited=where is not None,
        conditions=tuple(_terms(where)),
        group_expression=group is not None and group_column is None,
    )


# The words that say a summary is over every row ("counting every result", "per visit").
_EVERY_ROW = re.compile(
    r"\b(?:per|each|every|all|across all)\s+(?:\w+\s+)?(?:visits?|results?|measurements?|tests?|admissions?|"
    r"encounters?|rows?|records?|readings?|samples?|sessions?|stays?|episodes?|values?)\b|\bcount every visit\b",
    re.IGNORECASE,
)
_PER_PATIENT = re.compile(r"\buse one value per patient\b", re.IGNORECASE)
_ORDERED = re.compile(r"\b(?:first|last|latest|earliest|most recent|initial|final|baseline)\b", re.IGNORECASE)
StatisticMode: TypeAlias = Literal["rows", "patients", "patients_if_constant"]


def statistic_mode(question: str, context: DatasetContext) -> StatisticMode:
    """Over what a mean, median, minimum or maximum is taken (see ``_compile_plan``)."""
    if context.rows_are_patients or context.patient_key is None or _EVERY_ROW.search(question):
        return "rows"
    if _PER_PATIENT.search(question):
        return "patients"
    found = head_noun(question, context.row_nouns)
    if (found is not None and found[0] == "events") or _ORDERED.search(question):
        return "rows"
    return "patients_if_constant"


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
        pinned = [
            listed for term in _terms(_condition(reply.where, "where", context)) if (listed := _listed_values(term))
        ]
        if test in ("compare_groups", "odds_ratio") and len(pinned) == 1:
            # "in patients with X versus without": X is the exposure, written as a limit.
            name, values = pinned[0]
            raise PlanError(
                f"For a {test} test the exposure goes in group_by, not where: set group_by to {name} and remove "
                f"{name} = {values[0]!r} from where, so both groups are compared."
            )
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
    # A where term pinning the group column to one value says which group is the exposed one; as a
    # limit it would leave one group and nothing to compare, so it is not applied.
    # The same for the outcome: a limit to one outcome value leaves nothing to compare.
    outcome_column = _bare_column(reply.outcome, context)
    pinned_out = {group, outcome_column.name if outcome_column is not None else None} - {None}
    stated = [
        term
        for term in _terms(_condition(reply.where, "where", context))
        if not (_listed_values(term) is not None and _columns_of(term) <= pinned_out and _columns_of(term))
    ]
    where, _spelled = _split_where(_where_all(*stated), None, context)
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
        conditions=tuple(_terms(where)),
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
    parsed: exp.Expr = trees[0]
    for node in parsed.walk():
        node.comments = None
    tables = {t.name.casefold() for t in parsed.find_all(exp.Table)}
    cte_names = {c.alias.casefold() for c in parsed.find_all(exp.CTE)}
    if not tables or tables - cte_names - {context.table.casefold()}:
        raise PlanError(f"Read only the table {context.table}.")
    aliases = {p.alias.casefold() for p in parsed.find_all(exp.Alias)} | cte_names

    def resolve(node: exp.Expr) -> exp.Expr:
        if isinstance(node, exp.Column) and not isinstance(node.this, exp.Star):
            if context.column(node.name) is None and node.name.casefold() in aliases:
                return node
            resolved = resolve_columns(exp.column(node.name), context)
            if isinstance(resolved, exp.Column) and node.table:
                resolved.set("table", node.args.get("table"))
            return resolved
        return node

    tree = fix_literals(parsed.transform(resolve), context)
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

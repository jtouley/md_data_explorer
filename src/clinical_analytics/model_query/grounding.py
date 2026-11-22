# SPDX-License-Identifier: MIT
"""Is every part of the compiled query backed by the question's own words?

A small local model's wrong answers are rarely broken SQL: they are queries that run and answer a
slightly different question. It adds a limit nobody asked for (an IgG cut-off, one pathogen), it
breaks the answer down by a column the question never names, it breaks down a question that asks
for one number, or it keeps one side of "admitted versus not admitted". Each of these leaves a
trace that can be checked against the question without a model:

- a limit (a ``where`` or share condition) comparing with a value the question never states
  (``measles_igg > 120`` for "how many did we check", ``pathogen = 'S. aureus'`` for "pneumonia
  admissions"). A yes/no or presence test is the model's reading of the question's words and is
  not checked;
- a breakdown by a column the question never names, or a breakdown when the question has no
  breakdown words and names no two values of that column;
- no breakdown when the question contrasts two groups ("versus", "or not", "and how many");
- a number the question states in digits (an age, a year, a threshold) used nowhere in the query.

A doubt goes back to the model once as feedback. If the next reply still carries it, the user is
asked: the doubt becomes concrete choices (with or without the limit, broken down or one number),
never a silent pick. A choice is asked as a question of its own; the words it adds
(``NO_LIMIT_PHRASE``, ``ONE_NUMBER_PHRASE``) are then applied to the model's reply as written, so
the choice is honoured even when the model repeats itself.

The check reads names, descriptions and listed category values only, never a row.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from clinical_analytics.model_query.context import ColumnInfo, DatasetContext
from clinical_analytics.sql.render import render

DoubtKind = Literal[
    "limit", "no_group", "missing_group", "dropped", "unlinked", "uncovered", "ambiguous", "share_as_count"
]

# The words a choice adds to the question it re-asks (matched case-insensitively).
NO_LIMIT_PHRASE: Final = "Do not limit it by {column}."
ONE_NUMBER_PHRASE: Final = "Give one overall number."
BREAK_DOWN_PHRASE: Final = "Break it down by {column}."
ONLY_WHERE_PHRASE: Final = "Only where {condition}."

MISSING_MEETS_PHRASE: Final = "Count a missing value as meeting it."
MISSING_FAILS_PHRASE: Final = "Count a missing value as not meeting it."
MISSING_OUT_PHRASE: Final = "Leave out rows with a missing value."
MissingChoice = Literal["meets", "fails", "out"]

_NO_LIMIT = re.compile(r"do not limit it by ([A-Za-z_][\w ,]*?)\.", re.IGNORECASE)
_ONE_NUMBER = re.compile(r"\bgive one overall number\b", re.IGNORECASE)
_BREAKDOWN_WORDS = re.compile(
    r"\b(by|each|per|every|versus|vs|compared?|comparing|across|between|broken|breakdown|split|"
    r"distribution|or not|and how many|which)\b",
    re.IGNORECASE,
)
# "A, B or C" / "A, B, or C": a list of three or more alternatives, one group each.
_ENUMERATION = re.compile(r",[^,.?;]+,?\s+or\s+\w", re.IGNORECASE)
_CONTRAST_WORDS = re.compile(r"\b(versus|vs|or not|and how many)\b", re.IGNORECASE)
_NUMBER_WORDS: Final = {
    "one": "1",
    "once": "1",
    "two": "2",
    "twice": "2",
    "both": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
# Clinical words that state an age limit without a number in the question.
_AGE_WORDS: Final = {
    "adult": (17.0, 18.0),
    "adults": (17.0, 18.0),
    "child": (12.0, 16.0, 17.0, 18.0),
    "children": (12.0, 16.0, 17.0, 18.0),
    "pediatric": (12.0, 16.0, 17.0, 18.0),
    "paediatric": (12.0, 16.0, 17.0, 18.0),
    "elderly": (64.0, 65.0),
    "seniors": (64.0, 65.0),
}
# Clinical abbreviations a column name often uses for a word a question spells out.
_ABBREVIATIONS: Final = {
    "dx": ("diagnosis", "diagnosed", "diagnoses"),
    "tx": ("treatment", "treated", "therapy"),
    "rx": ("prescription", "prescribed", "medication"),
    "hx": ("history",),
    "abx": ("antibiotic", "antibiotics"),
    "yr": ("year",),
    "dob": ("birth", "born"),
    "los": ("length", "stay"),
    "bmi": ("body", "mass"),
    "bp": ("blood", "pressure"),
    "icu": ("intensive",),
    "ed": ("emergency",),
}
_SEX_WORDS: Final = {
    "female": frozenset({"women", "woman", "female", "females", "girls", "girl"}),
    "male": frozenset({"men", "man", "male", "males", "boys", "boy"}),
}
_SEX_VALUES: Final = {"female": "female", "f": "female", "woman": "female", "male": "male", "m": "male", "man": "male"}
_YES_NO: Final = frozenset({"yes", "no", "y", "n", "true", "false", "t", "f", "1", "0"})
# Words too common in questions and column names to say which column is meant.
_COMMON: Final = frozenset(
    {
        "the", "and", "with", "for", "from", "that", "this", "were", "was", "had", "have", "has", "are", "our",
        "what", "how", "many", "much", "who", "any", "all", "not", "per", "each", "patient", "patients",
        "people", "person", "case", "cases", "date", "id", "value", "values", "number", "count", "total",
        "record", "records", "recorded", "flag", "status", "yes", "no", "code", "data", "of", "in",
        "on", "at", "or", "is", "be", "to", "a", "an", "if", "it", "its", "one", "share", "rate",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Doubt:
    """A part of the query the question does not back."""

    kind: DoubtKind
    feedback: str  # for the model's next attempt
    reason: str  # for the user, when the model keeps it
    choices: tuple[tuple[str, str], ...]  # (label, the words added to the question)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> list[str]:
    raw = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.casefold())
    return [_NUMBER_WORDS.get(word, word) for word in raw]


@dataclass(frozen=True)
class _Question:
    text: str
    words: frozenset[str]  # stemmed tokens
    numbers: frozenset[float]

    @classmethod
    def of(cls, question: str) -> _Question:
        tokens = _tokens(question)
        numbers = {float(t) for t in tokens if re.fullmatch(r"\d+(?:\.\d+)?", t)}
        for word in tokens:
            numbers.update(_AGE_WORDS.get(word, ()))
        return cls(question, frozenset(_stem(t) for t in tokens), frozenset(numbers))

    def says(self, word: str) -> bool:
        return _stem(word) in self.words

    def says_sex(self, which: str) -> bool:
        return any(self.says(word) for word in _SEX_WORDS[which])


def _sex_of(value: object) -> str | None:
    return _SEX_VALUES.get(str(value).strip().casefold())


def _column_words(column: ColumnInfo) -> set[str]:
    """The words that name a column: its name's parts, its description, its listed text values."""
    words = set(_tokens(column.name.replace("_", " ")))
    for part in list(words):
        words.update(_ABBREVIATIONS.get(part, ()))
    words.update(_tokens(column.description or ""))
    if column.values is not None and column.text:
        for value in column.values:
            words.update(_tokens(str(value)))
    return {_stem(w) for w in words if w not in _COMMON and not w.isdigit() and len(w) >= 2}


# A word in this share of the table's columns (or more) names what the table is about ("covid" in
# a COVID registry), not one column.
SHARED_WORD_SHARE: Final = 0.2


def _shared_words(context: DatasetContext) -> set[str]:
    counts: dict[str, int] = {}
    for column in context.columns:
        for word in _column_words(column):
            counts[word] = counts.get(word, 0) + 1
    limit = max(3.0, SHARED_WORD_SHARE * len(context.columns))
    return {word for word, n in counts.items() if n >= limit}


def column_named(question: _Question, column: ColumnInfo, context: DatasetContext | None = None) -> bool:
    """Whether the question's words name ``column`` (a word of its name, description or values)."""
    if column.values is not None and {_sex_of(v) for v in column.values} == {"female", "male"}:
        if question.says_sex("female") or question.says_sex("male") or question.says("sex") or question.says("gender"):
            return True
    shared = _shared_words(context) if context is not None else set()
    return bool((_column_words(column) - shared) & question.words)


def _value_said(question: _Question, value: str) -> bool:
    sex = _sex_of(value)
    if sex is not None:
        return question.says_sex(sex)
    if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?.*", value):
        return float(value[:4]) in question.numbers
    words = [w for w in _tokens(value) if len(w) >= 2 or w.isdigit()]
    return bool(words) and all(question.says(w) for w in words)


def _number_said(question: _Question, literal: exp.Literal) -> bool:
    try:
        return float(literal.name) in question.numbers
    except ValueError:
        return False


def _compared_literals(term: exp.Expr) -> list[exp.Literal]:
    """The values a condition compares with: a literal side of a comparison, an IN list, BETWEEN's
    bounds. Literals inside an expression being compared (a unit conversion's CASE and factor) are
    the model's arithmetic, not a value the question has to state."""
    node = term
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Connector):
        return _compared_literals(node.this) + _compared_literals(node.expression)
    if isinstance(node, exp.Not):
        return _compared_literals(node.this)
    if isinstance(node, exp.Between):
        sides = [node.args.get("low"), node.args.get("high")]
    elif isinstance(node, exp.In):
        sides = list(node.expressions)
    elif isinstance(node, exp.Binary):
        sides = [node.this, node.expression]
    else:
        return list(node.find_all(exp.Literal))
    return [side for side in sides if isinstance(side, exp.Literal)]


# A number written in digits that is part of a name ("COVID-19", "H1N1"), a confidence level
# ("95% CI") or a unit ("mmol/L") is not a value the query has to use.
_NAME_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.%/-])")
_CONFIDENCE = re.compile(r"\b(?:9[059])\s*%\s*(?:ci|confidence)", re.IGNORECASE)


def stated_numbers(question: str) -> set[float]:
    """The numbers the question states in digits as values (ages, years, thresholds)."""
    text = _CONFIDENCE.sub(" ", question)
    text = re.sub(r"[A-Za-z]+-\d+\b", " ", text)  # COVID-19, type-2
    return {float(m.group(0)) for m in _NAME_NUMBER.finditer(text)}


def unused_numbers(question: str, sql: str) -> set[float]:
    """The question's stated numbers that appear nowhere in the query: not as a value it compares
    with, and not in a column name it reads ("died_30d" for "within 30 days")."""
    try:
        tree = sqlglot.parse_one(sql, dialect="duckdb")
    except SqlglotError:
        return set()
    used: set[float] = set()
    for literal in tree.find_all(exp.Literal):
        if not literal.is_string:
            try:
                used.add(float(literal.name))
            except ValueError:
                pass
        else:
            used.update(float(t) for t in re.findall(r"\d+(?:\.\d+)?", literal.name))
    for column in tree.find_all(exp.Column):
        used.update(float(t) for t in re.findall(r"\d+", column.name))
    return {number for number in stated_numbers(question) if number not in used}


def dropped_number(question: str, sql: str) -> Doubt | None:
    """A doubt when the question states a number (an age, a year, a threshold) the query never
    uses: the limit it sets was most likely left out."""
    unused = sorted(unused_numbers(question, sql))
    if not unused:
        return None
    said = ", ".join(f"{n:g}" for n in unused)
    return Doubt(
        "dropped",
        feedback=(
            f"The question states {said}, but the reply uses it nowhere. Put the limit it sets into where "
            "(for example an age, a year or a threshold); keep every other part of the reply."
        ),
        reason=(
            f"The question states {said}, but I could not tell which limit it sets, so I did not answer. "
            "Ask again naming the column it applies to."
        ),
        choices=(),
    )


def _as_float(literal: exp.Literal) -> float | None:
    try:
        return float(literal.name)
    except ValueError:
        return None


def fits_range(column: ColumnInfo, literal: exp.Literal) -> bool:
    """Whether a number compared with a column lies within the column's recorded range (a
    threshold at either end fits); True when the range is not known."""
    if not column.numeric or column.low is None or column.high is None:
        return True
    try:
        value, low, high = float(literal.name), float(column.low), float(column.high)
    except (TypeError, ValueError):
        return True
    return low <= value <= high


def limit_backed(term: exp.Expr, question: _Question, context: DatasetContext) -> bool:
    """Whether one condition is backed by the question: a value it compares with is stated (all
    of them, for text values; one of them, for numbers), or it only tests yes/no or presence."""
    columns = [c for name in sorted({c.name for c in term.find_all(exp.Column)}) if (c := context.column(name))]
    if not columns:
        return True
    literals = _compared_literals(term)
    # A coded value ("gender = 2") is stated when its dictionary label is ("women" for "female").
    labelled = [label for lit in literals for column in columns if (label := column.label_of(lit.name)) is not None]
    if labelled and any(_value_said(question, label) for label in labelled):
        return True
    # A value of a two-valued "X" / "not X" column is a yes/no test in words, like yes or no.
    flag_words = {str(v).strip().casefold() for c in columns if (pair := c.negated_pair()) for v in pair}
    texts = [lit.name for lit in literals if lit.is_string and lit.name.strip().casefold() not in _YES_NO | flag_words]
    numbers = [lit for lit in literals if not lit.is_string and lit.name.strip() not in ("0", "1")]
    if texts:
        return all(_value_said(question, text) for text in texts)
    if numbers:
        # A stated threshold must also fit the column it is compared with: "under 200" is not a
        # limit on a percentage that runs from 0 to 100.
        digits = stated_numbers(question.text)
        return any(
            _number_said(question, n) and (_as_float(n) not in digits or all(fits_range(c, n) for c in columns))
            for n in numbers
        )
    # Only a yes/no or presence test: the column is the model's reading of the question's words
    # ("ended up in hospital" for admitted), which is its job. Such a limit is not a guess.
    return True


def _terms(condition: exp.Expr | None) -> list[exp.Expr]:
    if condition is None:
        return []
    node = condition
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _terms(node.this) + _terms(node.expression)
    return [node]


def _plain(node: exp.Expr) -> str:
    def unquote(n: exp.Expr) -> exp.Expr:
        if isinstance(n, exp.Identifier) and re.fullmatch(r"[A-Za-z_]\w*", n.name):
            n.set("quoted", False)
        return n

    return render(node.copy().transform(unquote))


def _breakdown_asked(question: _Question, column: ColumnInfo | None) -> bool:
    if _BREAKDOWN_WORDS.search(question.text) or _ENUMERATION.search(question.text):
        return True
    if column is None or column.values is None:
        return False
    if {_sex_of(v) for v in column.values} == {"female", "male"}:
        return question.says_sex("female") and question.says_sex("male")
    return sum(_value_said(question, str(v)) for v in column.values) >= 2


def find_doubts(
    question: str,
    context: DatasetContext,
    *,
    conditions: Iterable[exp.Expr],
    group_column: str | None,
    group_expression: bool = False,
    tabulation: bool = True,
    kept_group: str | None = None,
) -> list[Doubt]:
    """The doubts about one compiled query, most serious first.

    ``conditions``: the limits it applies (top-level AND terms of where, and a share's event).
    ``group_column``: the table column it breaks the answer down by (None: none, or an expression,
    then ``group_expression`` is True). ``tabulation``: a count, share or summary (a statistical
    test's group is its exposure, and is not checked here). ``kept_group``: for a follow-up, the
    column the previous answer was broken down by (a refinement keeps it).
    """
    asked = _Question.of(question)
    doubts: list[Doubt] = []
    for term in conditions:
        if limit_backed(term, asked, context):
            continue
        plain = _plain(term)
        columns = ", ".join(sorted({c.name for c in term.find_all(exp.Column)}))
        doubts.append(
            Doubt(
                "limit",
                feedback=(
                    f"The reply keeps only rows where {plain}, but the question does not state that. "
                    "Remove that condition unless the question asks for it. A word that names what the whole "
                    "table is about is not a limit."
                ),
                reason=f"I read this as limited to {plain}, which the question does not say. Which did you mean?",
                choices=(
                    (f"Only where {plain}", ONLY_WHERE_PHRASE.format(condition=plain)),
                    ("Without that limit", NO_LIMIT_PHRASE.format(column=columns)),
                ),
            )
        )
    if not tabulation:
        return doubts
    column = context.column(group_column) if group_column else None
    kept = context.column(kept_group) if kept_group else None
    if kept is not None and (column is None or column.name != kept.name) and not group_expression:
        doubts.append(
            Doubt(
                "missing_group",
                feedback=(
                    f"The previous answer was broken down by {kept.name}, and this follow-up does not ask for "
                    f"another breakdown. Repeat the previous reply with only the change the follow-up asks for: "
                    f"keep group_by {kept.name}."
                ),
                reason=f"The previous answer was broken down by {kept.name}; I read this follow-up differently.",
                choices=(
                    (f"Broken down by {kept.name}", BREAK_DOWN_PHRASE.format(column=kept.name)),
                    ("One overall number", ONE_NUMBER_PHRASE),
                ),
            )
        )
    elif (column is not None or group_expression) and not _breakdown_asked(asked, column):
        name = column.name if column is not None else "an expression"
        doubts.append(
            Doubt(
                "no_group",
                feedback=(
                    f"The question asks for one overall number, but group_by is {name}. Set group_by to null. "
                    f"To count the rows with a yes/no column, put that column = 'yes' in where instead."
                ),
                reason=f"I read this as broken down by {name}, but the question asks for one number.",
                choices=(
                    (f"Broken down by {name}", BREAK_DOWN_PHRASE.format(column=name)),
                    ("One overall number", ONE_NUMBER_PHRASE),
                ),
            )
        )
    elif column is None and not group_expression and (said := _CONTRAST_WORDS.search(question)):
        doubts.append(_missing_group(said.group(0), list(conditions), context))
    return doubts


def _missing_group(word: str, conditions: list[exp.Expr], context: DatasetContext) -> Doubt:
    pinned = next(
        (
            (term, column)
            for term in conditions
            for column in [context.column(c.name) for c in term.find_all(exp.Column)]
            if column is not None and column.values is not None
        ),
        None,
    )
    choices: tuple[tuple[str, str], ...] = ()
    if pinned is not None:
        term, column = pinned
        plain = _plain(term)
        choices = (
            (f"Each value of {column.name}", BREAK_DOWN_PHRASE.format(column=column.name)),
            (f"Only where {plain}", f"{ONLY_WHERE_PHRASE.format(condition=plain)} {ONE_NUMBER_PHRASE}"),
        )
    return Doubt(
        "missing_group",
        feedback=(
            f"The question compares groups ('{word}'), but group_by is null. Set group_by to the column whose values "
            "are compared, and remove that column from where."
        ),
        reason=(
            f"The question compares groups ('{word}'), but I read it as one number for one group."
            + ("" if choices else " Ask again naming the groups to compare.")
        ),
        choices=choices,
    )


def asks_breakdown(text: str) -> bool:
    """Whether the words ask for a breakdown ("by", "each", "versus", "A, B or C")."""
    return bool(_BREAKDOWN_WORDS.search(text) or _ENUMERATION.search(text))


def user_choices(question: str) -> tuple[frozenset[str], bool]:
    """(the columns the question says not to limit by, whether it asks for one overall number):
    the words a choice added when it re-asked the question."""
    columns = {name.strip().casefold() for found in _NO_LIMIT.findall(question) for name in re.split(r",\s*", found)}
    return frozenset(columns), bool(_ONE_NUMBER.search(question))


def without_columns(condition: str | None, columns: frozenset[str]) -> str | None:
    """``condition`` without its top-level AND terms that read any of ``columns`` (None when no term
    is left). Text that does not parse is returned as it is (the compiler reports it)."""
    if not condition or not columns:
        return condition
    try:
        tree = sqlglot.parse_one(condition, dialect="duckdb")
    except SqlglotError:
        return condition
    kept = [term for term in _terms(tree) if not {c.name.casefold() for c in term.find_all(exp.Column)} & columns]
    return render(exp.and_(*kept)) if kept else None


def missing_choice(question: str) -> MissingChoice | None:
    """What the question says a missing value means for a share's condition (the words a choice
    added when it re-asked the question), or None when it does not say."""
    folded = question.casefold()
    for phrase, choice in (
        (MISSING_FAILS_PHRASE, "fails"),
        (MISSING_MEETS_PHRASE, "meets"),
        (MISSING_OUT_PHRASE, "out"),
    ):
        if phrase.casefold() in folded:
            return choice  # type: ignore[return-value]
    return None


def columns_with_stated_values(question: str, conditions: list[exp.Expr], context: DatasetContext) -> set[str]:
    """Columns of conditions whose compared value the question states ("65 and up" for
    ``age_years >= 65``, "positive" for ``result = 'positive'``): the value ties them to the question."""
    asked = _Question.of(question)
    found: set[str] = set()
    for condition in conditions:
        for term in _terms(condition):
            literals = _compared_literals(term)
            term_columns = [c for c in (context.column(col.name) for col in term.find_all(exp.Column)) if c]
            said = any(
                _value_said(asked, lit.name)
                if lit.is_string
                else _number_said(asked, lit)
                and (_as_float(lit) not in stated_numbers(question) or all(fits_range(c, lit) for c in term_columns))
                for lit in literals
            ) or any(
                _value_said(asked, label)
                for lit in literals
                for column in [context.column(c.name) for c in term.find_all(exp.Column)]
                if column is not None and (label := column.label_of(lit.name))
            )
            if said and literals:
                found.update(c.name for c in term.find_all(exp.Column))
    return found

# SPDX-License-Identifier: MIT
"""Does the compiled plan compute what the question asks? Checked on the plan, never on the
model's description of it.

Four small, general tables read the question: numbers with their comparison words, negation,
timepoint words (first, last, any, every, change), and the counted noun (a person, or the
contract's own name for one record). Everything else comes from the contract: the grain, the
patient key, the order column, category labels and open questions. There are no per-question word
lists. What the question states explicitly is applied (an operator, a timepoint, the unit); what
it contradicts is fed back once; what it leaves open is asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, Literal

from clinical_analytics.model_query.context import Column, DatasetContext, plural
from clinical_analytics.model_query.plan import Compiled, Condition, Plan, WhichVisit

Cmp = Literal["<", "<=", ">", ">=", "is"]

# Comparison words, before a number ("under 65") or right after it ("65 or older").
_BEFORE: Final[tuple[tuple[str, Cmp], ...]] = (
    (r"at least|no less than|not less than|minimum of|≥|>=", ">="),
    (r"at most|no more than|not more than|up to|maximum of|within|≤|<=", "<="),
    (r"under|below|less than|fewer than|younger than|lower than|shorter than|earlier than|<", "<"),
    (r"over|above|more than|greater than|older than|higher than|longer than|later than|exceed(?:ing|s)?|>", ">"),
)
_AFTER: Final[tuple[tuple[str, Cmp], ...]] = (
    (r"or (?:more|older|above|higher|greater|over|longer|later)|and (?:over|above|older|up)|\+", ">="),
    (r"or (?:less|fewer|younger|below|lower|under|shorter|earlier)|and (?:under|below|younger)", "<="),
)
_NUMBER: Final = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")
_BETWEEN: Final = re.compile(r"\b(?:between|from)\s+(\d+(?:\.\d+)?)\D{0,12}?\s(?:and|to)\s+(\d+(?:\.\d+)?)", re.I)
_NEGATION: Final = re.compile(r"\b(?:not|no|without|never|none|neither|nor|non)\b|n't\b|\bnon-", re.I)
_TIMES: Final[tuple[tuple[WhichVisit, re.Pattern[str]], ...]] = (
    ("change", re.compile(r"\bchanges?d?\b", re.I)),
    ("first", re.compile(r"\b(?:first|baseline|initial|intake|enrol(?:l)?ment|earliest)\b", re.I)),
    ("last", re.compile(r"\b(?:last|latest|final|most recent)\b", re.I)),
    ("every", re.compile(r"\b(?:every|always|throughout|never|all (?:of )?(?:their|his|her|the) )", re.I)),
    ("any", re.compile(r"\b(?:ever|at any|any (?:visit|time|point)|at least once)\b", re.I)),
)
_PERSONS: Final = re.compile(
    r"\b(?:patients?|people|persons?|individuals?|subjects?|participants?|wom[ae]n|m[ae]n|child(?:ren)?|adults?|"
    r"girls?|boys?|mothers?|infants?|bab(?:y|ies)|members?)\b",
    re.I,
)
_WHEN: Final = re.compile(
    r"\b(?:at|on|during|per|first|last|latest|final|baseline|initial|earliest|recent|every|each|any|all)\b", re.I
)
_YEARS: Final = (1900, 2100)
# Words that ask for a breakdown ("by sex", "per arm", "each district", "A versus B"), and the
# period words a breakdown by date needs. "each patient's" names a timepoint, not a breakdown.
_BREAKDOWN: Final = re.compile(
    r"\b(?:by|per|each|for each|across|versus|vs|compared?|comparing|between|breakdown|broken down|split|"
    r"distribution|stratified|grouped|among the)\b(?! patient'?s?\b)",
    re.I,
)
_PERIOD: Final = re.compile(r"\b(?:years?|yearly|annual\w*|months?|monthly|quarters?|over time|trend\w*)\b", re.I)
# Words that ask for a statistical test or model, not side-by-side figures.
_TEST_WORDS: Final = re.compile(
    r"\b(?:test\w*|significan\w*|p[- ]?values?|odds|correlat\w*|associat\w*|regression|hazard|survival|"
    r"time to|kaplan|log[- ]?rank|chi[- ]?square\w*|t-tests?|anova|confidence intervals?|ci|statistic\w*)\b",
    re.I,
)
# Category labels too common as words to say the question names that category.
_PLAIN_LABELS: Final = frozenset({"other", "unknown", "none", "missing", "yes", "no", "not", "all", "any"})


@dataclass(frozen=True)
class Stated:
    """One number in the question and the comparison it is said with."""

    value: float
    cmp: Cmp | None
    year: bool


@dataclass(frozen=True)
class Cues:
    numbers: tuple[Stated, ...]
    negated: bool
    never: bool
    times: frozenset[WhichVisit]
    persons: bool
    records: bool
    episodes: bool  # the contract's episode noun ("admissions"), said as what is counted
    values: tuple[tuple[str, str], ...]  # (category label said, its column)


@dataclass(frozen=True)
class Problem:
    """A meaning the plan gets wrong: fed back once, then held (with the reason)."""

    kind: str
    feedback: str
    reason: str


@dataclass(frozen=True)
class Ask:
    """Something the question leaves open that changes the number: asked in clinical words."""

    kind: str
    reason: str
    choices: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def _mentioned_labels(question: str, context: DatasetContext) -> list[tuple[str, str, tuple[int, int]]]:
    """(label, column, span) of each category value the question names whole, when only one
    column holds that label."""
    owners: dict[str, set[str]] = {}
    for column in context.columns:
        if column.kind in ("category", "flag"):
            for label in column.choices():
                if len(label) >= 3 and label.casefold() not in _PLAIN_LABELS and not label.isdigit():
                    owners.setdefault(label.casefold(), set()).add(column.name)
    found = []
    for label, names in owners.items():
        if len(names) != 1:
            continue
        match = re.search(rf"(?<!\w){re.escape(label)}(?!\w)", question, re.I)
        if match:
            found.append((label, next(iter(names)), match.span()))
    return found


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    for start, end in spans:
        text = text[:start] + " " * (end - start) + text[end:]
    return text


def _cmp_of(question: str, start: int, end: int, previous_end: int) -> Cmp | None:
    after = question[end : end + 24].lower()
    for words, cmp in _AFTER:
        if re.match(rf"\s*(?:\w+\s+){{0,2}}?(?:{words})(?!\w)", after):
            return cmp
    before = question[max(previous_end, start - 30) : start].lower()
    found: tuple[int, Cmp] | None = None
    for words, cmp in _BEFORE:
        for match in re.finditer(rf"(?<!\w)(?:{words})(?!\w)", before):
            if found is None or match.start() > found[0]:
                found = (match.start(), cmp)
    return found[1] if found else None


def read_cues(question: str, context: DatasetContext, used: frozenset[str] = frozenset()) -> Cues:
    """What the question says, with the words that belong to a category label or to a used
    column's own description set aside (so "non-binary" is not a negation, nor "Initial serum
    lactate" a timepoint)."""
    labels = _mentioned_labels(question, context)
    text = _blank(question, [span for *_rest, span in labels])
    between = {float(n): cmp for m in _BETWEEN.finditer(text) for n, cmp in zip(m.groups(), (">=", "<="), strict=True)}
    numbers: list[Stated] = []
    previous = 0
    for match in _NUMBER.finditer(text):
        value = float(match.group(1))
        cmp = between.get(value) or _cmp_of(text, match.start(), match.end(), previous)
        year = value.is_integer() and _YEARS[0] <= value <= _YEARS[1]
        numbers.append(Stated(value=value, cmp=cmp, year=year))  # type: ignore[arg-type]
        previous = match.end()
    own = " ".join(f"{c.name} {c.description}".casefold() for name in used if (c := context.column(name)) is not None)
    times = frozenset(
        which for which, pattern in _TIMES if any(m.group(0).casefold() not in own for m in pattern.finditer(text))
    )
    records_word = context.row_noun.split()[-1]
    noun = rf"(?:{re.escape(records_word)}|{re.escape(plural(records_word))}|records?)"
    records = re.compile(rf"\b{noun}\b", re.I)
    # "over all encounters", "across every visit": every record, in the contract's own noun.
    if context.visit_level and re.search(
        rf"\b(?:all|every|each|across)(?: of)?(?: their| the)?(?: \w+)? {noun}\b", text, re.I
    ):
        times = times | {"every"}
    # A records noun inside a timepoint phrase ("at their latest visit") says when, not what is
    # counted.
    counted_records = [m for m in records.finditer(text) if not _WHEN.search(" ".join(text[: m.start()].split()[-3:]))]
    episode = context.episode
    episodes = False
    if episode is not None and context.visit_level:
        word = episode[1].split()[-1].removesuffix("s")
        found = re.finditer(rf"\b{re.escape(word)}s?\b", text, re.I)
        episodes = any(not _WHEN.search(" ".join(text[: m.start()].split()[-3:])) for m in found)
    negated_text = re.sub(r"\b(?:or|and|vs\.?|versus) not\b", " ", text, flags=re.I)
    return Cues(
        numbers=tuple(numbers),
        negated=bool(_NEGATION.search(negated_text)),
        never=bool(re.search(r"\bnever\b", text, re.I)),
        times=times,
        persons=bool(_PERSONS.search(text)) and records_word not in ("patient",),
        records=context.visit_level and bool(counted_records),
        episodes=episodes,
        values=tuple((label, name) for label, name, _span in labels),
    )


# ----------------------------------------------------------------------------- fixes
def _expected_time(cues: Cues) -> WhichVisit | None:
    if "change" in cues.times or {"first", "last"} <= cues.times:
        return "change"
    for which in ("first", "last", "every", "any"):
        if which in cues.times:
            return which  # type: ignore[return-value]
    return None


def _time_by_column(plan: Plan, context: DatasetContext, which: WhichVisit) -> bool:
    """The plan reads the stated time from a column: the first record by its sequence number, or
    a measure the contract says was taken at that time."""
    for cond in plan.filters:
        start = context.sequence_start(cond.column)
        if start is not None and which == "first" and cond.op == "is" and float(start) in _floats(cond):
            return True
    named = [c for n in (plan.measure, *(f.column for f in plan.filters)) if n and (c := context.column(n))]
    return any(c.timepoint is not None for c in named)


def _floats(cond: Condition) -> list[float]:
    return [float(v) for v in (cond.value, *cond.values) if isinstance(v, int | float)]


def _slots_used(plan: Plan) -> tuple[Plan, list[str]]:
    """Every slot the model filled is used: a count's numerator conditions are its limits, and a
    filter repeated as a share's numerator is only the numerator (else the share is 100%)."""
    if plan.answer == "count" and plan.numerator:
        return plan.model_copy(update={"filters": (*plan.filters, *plan.numerator), "numerator": ()}), [
            "numerator->filters"
        ]
    if plan.answer == "share" and set(plan.filters) & set(plan.numerator):
        kept = tuple(c for c in plan.filters if c not in plan.numerator)
        return plan.model_copy(update={"filters": kept}), ["filter_repeated_in_numerator"]
    return plan, []


def _as_figures(plan: Plan, context: DatasetContext) -> Plan | None:
    """A test the question does not ask for ("compare A and B") as side-by-side figures: the mean
    of a number, or the share of a yes/no, in each group. None when there is no such reading."""
    if plan.answer == "odds_ratio" and plan.numerator:
        return plan.model_copy(update={"answer": "share"})
    if plan.answer != "compare_groups" or plan.measure is None:
        return None
    column = context.column(plan.measure)
    if column is None:
        return None
    if column.numeric:
        return plan.model_copy(update={"answer": "mean"})
    if column.kind == "flag":
        yes = next((v for v in column.values if not column.negative(v)), None)
        if yes is not None:
            event = Condition(column=column.name, op="is", values=(column.label(yes),))
            return plan.model_copy(update={"answer": "share", "measure": None, "numerator": (event,)})
    return None


def apply_stated(plan: Plan, question: str, context: DatasetContext) -> tuple[Plan, list[str]]:
    """The plan with what the question states explicitly put in force: the comparison said with
    each number, the timepoint, and the unit counted. Returns the plan and what was changed."""
    plan, changed = _slots_used(plan)
    if plan.answer in ("compare_groups", "odds_ratio") and not _TEST_WORDS.search(question):
        figures = _as_figures(plan, context)
        if figures is not None:
            changed.append(f"{plan.answer}->{figures.answer}")
            plan = figures
    used = frozenset({c.column for c in (*plan.filters, *plan.numerator)} | {plan.measure or "", plan.group_by or ""})
    cues = read_cues(question, context, used)
    conditions = {"filters": list(plan.filters), "numerator": list(plan.numerator)}
    for stated in cues.numbers:
        if stated.cmp is None or stated.year:
            continue
        hits = [
            (part, i)
            for part, conds in conditions.items()
            for i, c in enumerate(conds)
            if stated.value in _floats(c) and context.column(c.column) and context.column(c.column).numeric  # type: ignore[union-attr]
        ]
        if len(hits) != 1:
            continue
        part, i = hits[0]
        cond = conditions[part][i]
        op = stated.cmp
        if cond.op != op and not (op == "is" and cond.op == "is"):
            conditions[part][i] = cond.model_copy(update={"op": op, "value": stated.value, "values": ()})
            changed.append(f"operator:{cond.op}->{op}")
    update: dict[str, object] = {k: tuple(v) for k, v in conditions.items()}
    if context.visit_level:
        expected = _expected_time(cues)
        if expected == "every" and plan.answer in ("mean", "median", "min", "max", "sum"):
            expected = "any"  # a measure over every visit pools them
        if expected is not None and plan.which_visit != expected and not _time_by_column(plan, context, expected):
            update["which_visit"] = expected
            changed.append(f"which_visit:{plan.which_visit}->{expected}")
        elif expected is None and plan.which_visit in ("first", "last", "every"):
            update["which_visit"] = "any"  # a timepoint the question does not state is asked, not assumed
            changed.append(f"which_visit:{plan.which_visit}->any")
        if cues.never and update.get("which_visit", plan.which_visit) == "any" and _negated(plan, context):
            update["which_visit"] = "every"
            changed.append("which_visit:never->every")
        if plan.answer in ("count", "share"):
            unit = "patients" if cues.persons and not cues.records and not cues.episodes else None
            unit = context.row_nouns if cues.records and not cues.persons and not cues.episodes else unit
            if cues.episodes and not cues.persons and context.episode is not None:
                unit = context.episode[1]
            if unit is not None and plan.count_of != unit:
                update["count_of"] = unit
                changed.append(f"count_of->{unit}")
    return plan.model_copy(update=update), changed


def _negated(plan: Plan, context: DatasetContext) -> bool:
    """The plan excludes something: "is not" a value (not "is not none", a double negative), a
    value that means no, a number of zero, or not recorded."""
    for cond in (*plan.filters, *plan.numerator):
        column = context.column(cond.column)
        if cond.op == "is missing":
            return True
        if column is None or cond.op not in ("is", "is not"):
            continue
        said = [v for v in (*cond.values, cond.value) if v is not None]
        if column.numeric:
            if cond.op == "is" and said and all(float(v) == 0 for v in said):
                return True
            if cond.op == "is not":
                return True
            continue
        stored = [column.stored(str(v)) for v in said]
        negative = bool(stored) and all(s is not None and column.negative(s) for s in stored)
        if negative != (cond.op == "is not"):
            return True
    return False


# ----------------------------------------------------------------------------- problems
def _explained(stated: Stated, compiled: Compiled, context: DatasetContext) -> bool:
    if any(abs(stated.value - n) < 1e-9 for n in compiled.numbers):
        return True
    if stated.year:
        years = {str(int(stated.value)), str(int(stated.value) + 1)}
        dated = [str(c.value)[:4] for c in (*compiled.filters, *compiled.numerator) if isinstance(c.value, str)]
        if years & set(dated) or (compiled.plan.period != "none"):
            return True
    token = _num(stated.value)
    said = " ".join(
        str(v) for c in (*compiled.filters, *compiled.numerator) for v in (c.value, *c.values) if isinstance(v, str)
    )
    if re.search(rf"(?<!\d){re.escape(token)}(?!\d)", said):
        return True  # inside a value the plan names ("ward 7", "MCA M2")
    for name in compiled.columns:
        column = context.column(name)
        if column is not None and re.search(rf"(?<!\d){re.escape(token)}(?!\d)", f"{column.name} {column.description}"):
            return True
    return False


def _num(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def problems(compiled: Compiled, question: str, context: DatasetContext) -> list[Problem]:
    """Meanings of the question the compiled plan does not compute."""
    cues = read_cues(question, context, compiled.columns)
    found: list[Problem] = []
    for stated in cues.numbers:
        if not _explained(stated, compiled, context):
            said = _num(stated.value)
            found.append(
                Problem(
                    "number",
                    feedback=f"The question says {said}, but the reply does not use {said}. Put it in a condition "
                    "on the column it limits (with the comparison the question says), or in cut_points.",
                    reason=f"I could not use '{said}' from your question, so I did not answer.",
                )
            )
    stated_numbers = {s.value for s in cues.numbers}
    for cond in (*compiled.filters, *compiled.numerator):
        column = context.column(cond.column)
        if column is None or not column.numeric:
            continue
        start = context.sequence_start(column.name)
        for n in _floats(cond):
            if n in stated_numbers or (start is not None and n == start) or (n == 0 and cues.negated):
                continue
            found.append(
                Problem(
                    "invented_number",
                    feedback=f"The question gives no {_num(n)}: do not add a limit on {column.name} it does not state.",
                    reason=f"Your question gives no threshold for {column.display}, so I did not pick one.",
                )
            )
    group = context.column(compiled.group_column or "")
    if group is not None and not compiled.plan.cut_points and compiled.kind != "test":
        asked = _PERIOD.search(question) if group.kind == "date" else _BREAKDOWN.search(question)
        if not asked:
            found.append(
                Problem(
                    "group_invented",
                    feedback="The question does not ask for a breakdown: set group_by null (and period none).",
                    reason=f"My reading broke the answer down by {group.display}, which your question does not ask "
                    "for, so I did not answer.",
                )
            )
    if compiled.group_column is None and cues.negated != _negated(compiled.plan, context):
        found.append(
            Problem(
                "negation",
                feedback="The question says not, no, without or never: the reply must exclude it (is not, or the "
                "value that means no)."
                if cues.negated
                else "The question has no not, no or without: do not exclude a value it does not exclude.",
                reason="I could not be sure I read the 'not' in your question right, so I did not answer."
                if cues.negated
                else "My reading excluded something your question does not exclude, so I did not answer.",
            )
        )
    for label, name in cues.values:
        if name not in compiled.columns:
            column = context.column(name)
            found.append(
                Problem(
                    "value",
                    feedback=f"The question names '{label}', a value of {name}: use {name} in the reply.",
                    reason=f"I could not use '{label}' ({column.display if column else name}) in my reading, "
                    "so I did not answer.",
                )
            )
    return found


# ----------------------------------------------------------------------------- ask-backs
def _visit_choices(context: DatasetContext, *, conditions: bool, statistic: bool) -> list[tuple[str, str]]:
    noun = context.row_noun
    choices: list[tuple[str, str]] = []
    if statistic:
        choices.append((f"Every {noun} counts", f"over every {noun}"))
    else:
        choices.append((f"At any {noun} (ever)", f"at any {noun}"))
        if conditions:
            choices.append((f"At every {noun}", f"at every {noun}"))
    if context.order_column is not None:
        choices.append((f"At the first {noun}", f"at the first {noun}"))
        choices.append((f"At the latest {noun}", f"at the latest {noun}"))
    return choices


def ask_open(compiled: Compiled, context: DatasetContext) -> Ask | None:
    """An unresolved open question of the contract that changes this plan's number: any question
    about a column the plan reads (a date order, a value that may mean not recorded), and a
    question about who a line is when the plan counts."""
    for q in context.open_questions():
        if q.column is not None and q.column in compiled.columns:
            return Ask("open_question", f"{q.question} You can answer it in the Data check panel.")
        if q.column is None and compiled.kind in ("count", "share"):
            return Ask("open_question", f"{q.question} You can answer it in the Data check panel.")
    return None


def ask_unit(compiled: Compiled, question: str, context: DatasetContext) -> Ask | None:
    """A count or share on a visit-level table whose question names neither patients nor the
    records' own noun: the two numbers differ."""
    if not context.visit_level or compiled.kind not in ("count", "share"):
        return None
    cues = read_cues(question, context, compiled.columns)
    if cues.persons or cues.records or cues.episodes:
        return None
    rows = context.row_nouns
    return Ask(
        "unit",
        f"Patients here can have several {rows}, so counting patients and counting {rows} give different "
        "numbers. Which do you mean?",
        (("Patients", "(count patients)"), (rows.capitalize(), f"(count {rows})")),
    )


def visit_columns(compiled: Compiled) -> list[str]:
    """The columns whose value per patient decides this plan's number."""
    names = [c.column for c in (*compiled.filters, *compiled.numerator)]
    if compiled.group_column:
        names.append(compiled.group_column)
    if compiled.kind == "statistic" and compiled.measure:
        names.append(compiled.measure)
    return list(dict.fromkeys(names))


def needs_visit_choice(compiled: Compiled, question: str, context: DatasetContext) -> bool:
    """Patients counted (or measured) from a visit-level table, and the question does not say
    which visit: asked when a column the plan reads differs between a patient's visits."""
    if not context.visit_level or compiled.kind == "test":
        return False
    if compiled.kind in ("count", "share") and compiled.unit != "patients":
        return False  # records, or episodes: each is read on its own records
    cues = read_cues(question, context, compiled.columns)
    if _expected_time(cues) is not None or _time_by_column(compiled.plan, context, "first"):
        return False
    if compiled.kind == "statistic" and cues.records and not cues.persons:
        return False  # a summary over the records the question names ("sessions longer than ...")
    return bool(visit_columns(compiled))


def visit_ask(compiled: Compiled, context: DatasetContext, varying: int, columns: list[str] | None = None) -> Ask:
    """Which visit counts, asked about the ``columns`` that differ between a patient's visits."""
    noun = context.row_noun
    named = [context.column(n) for n in (columns or visit_columns(compiled))]
    what = " and ".join(c.display for c in named if isinstance(c, Column))
    statistic = compiled.kind == "statistic"
    return Ask(
        "which_visit",
        f"{varying} patients have a different {what} at different {plural(noun)}. Which {noun} should count?",
        tuple(_visit_choices(context, conditions=bool(compiled.filters or compiled.numerator), statistic=statistic)),
    )


# ----------------------------------------------------------------------------- tier 1
def matcher_may_answer(question: str, context: DatasetContext, used: frozenset[str]) -> bool:
    """Whether the keyword matcher's answer may stand in front of the model: only a question it
    fully reads. One record per patient (no per-patient choice), no number, comparison, negation
    or timepoint, and every category the question names is in the matcher's plan."""
    if not context.semantics.one_row_per_patient or context.aggregate:
        return False
    cues = read_cues(question, context, used)
    if cues.numbers or cues.negated or cues.times:
        return False
    return all(name in used for _label, name in cues.values)

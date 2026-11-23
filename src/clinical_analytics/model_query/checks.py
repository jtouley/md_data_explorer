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

import os
import re
from dataclasses import dataclass, field
from typing import Final, Literal

from clinical_analytics.contracts.holds import held_message, open_question_holds
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
# Timepoint words. A standalone one is a timepoint by itself ("baseline", "ever"); an attached one
# ("first", "latest", "every") only when it names a record (the contract's noun) or a measure, and
# never before a period ("last year" is a date range).
_TIMES: Final[tuple[tuple[WhichVisit, re.Pattern[str], bool], ...]] = (
    ("change", re.compile(r"\bchanges?d?\b", re.I), False),
    ("first", re.compile(r"\b(?:baseline|intake|enrol(?:l)?ment)\b", re.I), False),
    ("first", re.compile(r"\b(?:first|initial|earliest)\b", re.I), True),
    ("last", re.compile(r"\b(?:last|latest|final|most recent)\b", re.I), True),
    ("every", re.compile(r"\b(?:always|throughout|never)\b", re.I), False),
    ("any", re.compile(r"\b(?:ever|at least once)\b", re.I), False),
    ("any", re.compile(r"\bat any\b", re.I), True),
)
_PERIOD_WORD: Final = re.compile(r"^(?:\d+\s+)?(?:years?|months?|weeks?|days?|quarters?|decades?)\b", re.I)
# A comparison said with a negation ("not under 65", "no older than 70") is the opposite comparison.
_NEGATED_CMP: Final = re.compile(
    r"\b(?:not|no)\s+(?=(?:under|below|less|fewer|younger|lower|shorter|earlier|over|above|more|greater|older|"
    r"higher|longer|later)\b)",
    re.I,
)
_INVERTED: Final[dict[str, Cmp]] = {"<": ">=", "<=": ">", ">": "<=", ">=": "<"}
# Words that name a summary: the plan's answer must be one they allow.
_STATISTIC_WORDS: Final[tuple[tuple[re.Pattern[str], frozenset[str]], ...]] = (
    (re.compile(r"\bmedian\b", re.I), frozenset({"median"})),
    (re.compile(r"\b(?:average|mean)\b", re.I), frozenset({"mean"})),
    (re.compile(r"\b(?:highest|maximum|max|largest|longest|oldest)\b", re.I), frozenset({"max", "top"})),
    (re.compile(r"\b(?:lowest|minimum|min|smallest|shortest|youngest)\b", re.I), frozenset({"min"})),
    (re.compile(r"\b(?:how many|number of)\b", re.I), frozenset({"count", "count_distinct", "top", "sum"})),
)
# "The share of the climbers ...", "among the hikers": what follows names the share's denominator.
_DENOMINATOR: Final = re.compile(
    r"\b(?:(?:share|percent(?:age)?|proportion|fraction|rate) of(?: the| all| our)?|among(?: the| all| our)?|"
    r"of all(?: the)?|of the)\s+((?:[\w-]+\s+){0,2}[\w-]+)",
    re.I,
)
# The denominator phrase ends where its clause's verb or relative word starts ("the women who
# died": "women").
_CLAUSE_END: Final = re.compile(
    r"\b(?:who|that|which|whose|were|was|are|is|had|have|has|did|got|with|in|at|by|on|\w{2,}(?:ed|ing))\b.*", re.I
)
_TIME_UNITS: Final = frozenset({"day", "week", "month", "year", "hour", "minute", "second"})
# Glue words that never name a column, and the everyday words for the sexes a dictionary labels.
_GLUE: Final = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "was", "were", "are", "has", "had", "have", "been", "not",
    "who", "how", "many", "what", "which", "per", "all", "any", "its", "their", "his", "her", "our", "one", "there",
    "into", "over", "under", "than", "more", "less", "each", "when", "did", "does", "got", "yes",
})  # fmt: skip
_SEXES: Final = {
    "female": frozenset({"women", "woman", "female", "females", "girls", "girl"}),
    "male": frozenset({"men", "man", "male", "males", "boys", "boy"}),
}
_MONTHS: Final = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|\b(?:since|before|after|until)\b", re.I
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
# Words that ask for a breakdown ("by sex", "per arm", "each depot", "A versus B"), and the
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


def _value_spans(question: str, context: DatasetContext) -> list[tuple[int, int]]:
    spans = []
    for column in context.columns:
        for label in column.choices() if column.kind in ("category", "flag") else ():
            if len(label) >= 3 and label.casefold() not in _PLAIN_LABELS and not label.isdigit():
                spans += [m.span() for m in re.finditer(rf"(?<!\w){re.escape(label)}(?!\w)", question, re.I)]
    return spans


def _names_key_person(text: str, context: DatasetContext) -> bool:
    """The question names the person the contract's patient key names, in the file's own word
    ("carriers" for a carrier column, the key's role)."""
    key = context.semantics.patient_key
    if key is None:
        return False
    words = {w for w in re.split(r"[^a-z]+", f"{key.column} {key.role}".casefold()) if len(w) >= 4}
    return any(re.search(rf"\b{re.escape(w)}s?\b", text, re.I) for w in words)


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
    if found is None:
        return None
    if re.search(r"\b(?:not|no)\s+$", before[: found[0]]) and found[1] in _INVERTED:
        return _INVERTED[found[1]]  # "not under 65" is 65 or more
    return found[1]


def _timepoints(text: str, context: DatasetContext, used: frozenset[str], records_word: str) -> set[WhichVisit]:
    """The timepoints the question states (see ``_TIMES``). A word that is part of a used column's
    own description ("Opening blister score") is that column's name, not a timepoint."""
    own = " ".join(f"{c.name} {c.description}".casefold() for name in used if (c := context.column(name)) is not None)
    nouns = {records_word, plural(records_word), "record", "records"}
    nouns |= {s.noun for s in context.semantics.sequence_columns} | {
        plural(s.noun) for s in context.semantics.sequence_columns
    }
    if context.episode is not None:
        nouns |= {context.episode[1], context.episode[1].removesuffix("s")}
    measures = {w for name in used if (c := context.column(name)) for w in c.words()} - context.common_words - _GLUE
    found: set[WhichVisit] = set()
    for which, pattern, attached in _TIMES:
        for match in pattern.finditer(text):
            after = text[match.end() :].strip()
            phrase = f"{match.group(0)} {after.split(' ', 1)[0] if after else ''}".casefold().strip()
            if phrase in own and " " in phrase:
                continue  # "opening blister ...", the column's own name
            if not attached:
                found.add(which)
                continue
            if _PERIOD_WORD.match(after):
                continue  # "last year", "first 3 months": a date range, not a record
            following = [w.strip(".,;:?!'s").casefold() for w in after.split()[:4]]
            if any(w in nouns or w in measures for w in following):
                found.add(which)
    return found


def read_cues(question: str, context: DatasetContext, used: frozenset[str] = frozenset()) -> Cues:
    """What the question says, with the words that belong to a category label or to a used
    column's own description set aside (so "non-reactive" is not a negation, nor "Opening blister
    score" a timepoint)."""
    labels = _mentioned_labels(question, context)
    # Every category value the question names is set aside, whichever columns hold it: a value
    # ("women" in a column of groups) never sets the counted unit, a negation or a timepoint.
    text = _blank(question, _value_spans(question, context))
    between = {float(n): cmp for m in _BETWEEN.finditer(text) for n, cmp in zip(m.groups(), (">=", "<="), strict=True)}
    numbers: list[Stated] = []
    previous = 0
    for match in _NUMBER.finditer(text):
        value = float(match.group(1))
        cmp = between.get(value) or _cmp_of(text, match.start(), match.end(), previous)
        year = value.is_integer() and _YEARS[0] <= value <= _YEARS[1]
        numbers.append(Stated(value=value, cmp=cmp, year=year))  # type: ignore[arg-type]
        previous = match.end()
    records_word = context.row_noun.split()[-1]
    noun = rf"(?:{re.escape(records_word)}|{re.escape(plural(records_word))}|records?)"
    times = frozenset(_timepoints(text, context, used, records_word))
    records = re.compile(rf"\b{noun}\b", re.I)
    # "over all contacts", "across every visit": every record, in the contract's own noun.
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
    negated_text = _NEGATED_CMP.sub(" ", negated_text)  # its "not" is part of the comparison
    return Cues(
        numbers=tuple(numbers),
        negated=bool(_NEGATION.search(negated_text)),
        never=bool(re.search(r"\bnever\b", text, re.I)),
        times=times,
        persons=(bool(_PERSONS.search(text)) or _names_key_person(text, context)) and records_word != "patient",
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
    for part, conds in conditions.items():
        for i, cond in enumerate(conds):
            column = context.column(cond.column)
            if column is not None and column.unit_column and isinstance(cond.value, int | float):
                # A duration whose unit is per record: compared in the unit the question says.
                conds[i] = cond.model_copy(update={"unit": _stated_unit(question, float(cond.value), context)})
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


def _negated(plan: Plan, context: DatasetContext, named: frozenset[str] = frozenset()) -> bool:
    """The plan excludes something: "is not" a value (not "is not none", a double negative), a
    value that means no, a number of zero, or not recorded. ``named``: values the question names
    itself ("never" as a vaccination status), whose condition is the value, not a negation."""
    for cond in (*plan.filters, *plan.numerator):
        column = context.column(cond.column)
        if cond.op == "is" and cond.values and all(str(v).casefold() in named for v in cond.values):
            continue
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


def _tokens(question: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9-]+", question.casefold())


def _same(a: str, b: str) -> bool:
    """One word, or one stem: a prefix of the other (four letters or more), or five shared first
    letters ("wheezed" and "wheezer", "anaemic" and "anaemia")."""
    if a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a))):
        return True
    return len(os.path.commonprefix([a, b])) >= 5


def anchors(column: Column, question: str, context: DatasetContext) -> set[str]:
    """The words of ``column`` (contract name, description, labels) the question says, plus a sex
    label said in everyday words ("women" for female)."""
    tokens = _tokens(question)
    words = column.words() - context.common_words - _GLUE
    hits = {w for w in words if any(_same(w, t) for t in tokens)}
    spoken = re.sub(r"[_.]+", " ", column.name).strip()
    if len(spoken) >= 3 and re.search(rf"(?<!\w){re.escape(spoken)}(?!\w)", question, re.I):
        hits.add(column.name.casefold())  # the column's own name, said with spaces ("item 007")
    labels = {label.casefold() for label in column.choices()} if column.kind in ("category", "flag") else set()
    for label, said in _SEXES.items():
        if label in labels and said & set(tokens):
            hits.add(label)
    return hits


def _named_strongly(column: Column, question: str, context: DatasetContext) -> bool:
    """The question names the column beyond doubt: two of its words, a part of its own name, or
    one of its category labels whole."""
    hits = anchors(column, question, context)
    name_parts = {p.casefold() for p in re.split(r"[_.\s]+|(?<=[a-z])(?=[A-Z])", column.name) if len(p) >= 4}
    name_parts -= context.common_words
    label_said = column.kind in ("category", "flag") and any(
        len(label) >= 3
        and label.casefold() not in _PLAIN_LABELS
        and re.search(rf"(?<!\w){re.escape(label)}(?!\w)", question, re.I)
        for label in column.choices()
    )
    return len(hits) >= 2 or bool(name_parts & hits) or bool(label_said)


def _stated_unit(question: str, value: float, context: DatasetContext) -> str | None:
    """The unit said right after a number ("600 months"): a time unit or a unit the contract
    gives any column, singular."""
    known = _TIME_UNITS | {c.unit.casefold().rstrip("s") for c in context.columns if c.unit}
    for match in _NUMBER.finditer(question):
        if float(match.group(1)) != value:
            continue
        after = question[match.end() :].split()
        word = after[0].strip(".,;:?!").casefold().rstrip("s") if after else ""
        if word in known:
            return word
    return None


def _anchor_problems(compiled: Compiled, question: str, context: DatasetContext, cues: Cues) -> list[Problem]:
    """Every column a condition reads is named in the question, and every column the question
    names beyond doubt is read (B1). A date condition needs a date, year or period said."""
    found: list[Problem] = []
    stated = {s.value for s in cues.numbers}
    for cond in (*compiled.filters, *compiled.numerator):
        column = context.column(cond.column)
        if column is None:
            continue
        if column.kind == "date":
            if not (cues.numbers or _PERIOD.search(question) or _MONTHS.search(question)):
                found.append(
                    Problem(
                        "invented_date",
                        feedback=f"The question gives no date, year or period: remove the condition on {column.name}.",
                        reason=f"My reading limited {column.display} to dates your question does not give, so I did "
                        "not answer.",
                    )
                )
            continue
        start = context.sequence_start(column.name)
        by_number = column.numeric and bool(set(_floats(cond)) & stated)
        by_time = start is not None and float(start) in _floats(cond) and "first" in cues.times
        if not (anchors(column, question, context) or by_number or by_time):
            found.append(
                Problem(
                    "unanchored",
                    feedback=f"The question says nothing about {column.name}: remove that condition.",
                    reason=f"My reading added a limit on {column.display} that your question does not state, so I "
                    "did not answer.",
                )
            )
    skip = {context.patient_key, context.count_column, context.order_column}
    for column in context.columns:
        if column.kind == "key" or column.name in skip or column.name in compiled.columns:
            continue
        if _named_strongly(column, question, context):
            found.append(
                Problem(
                    "column",
                    feedback=f"The question names {column.name} ({column.display}), but the reply does not use it: "
                    "put it where the question means it (a filter, the numerator, the measure or group_by).",
                    reason=f"I could not use '{column.display}' in my reading, so I did not answer.",
                )
            )
    return found


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
    if compiled.kind == "test" and not _TEST_WORDS.search(question):
        found.append(
            Problem(
                "test_not_asked",
                feedback="The question asks for figures side by side, not a statistical test: answer count, share "
                "or mean with group_by (and the values compared in filters).",
                reason="Your question does not ask for a statistical test, and I could not read it as figures side "
                "by side, so I did not answer.",
            )
        )
    group = context.column(compiled.group_column or "")
    if group is not None and not compiled.plan.cut_points and compiled.kind != "test" and compiled.plan.answer != "top":
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
    named = frozenset(question[a:b].casefold() for a, b in _value_spans(question, context))
    # A breakdown by a yes/no column ("died or not, by ...") says the negation itself.
    by_negation = group is not None and any(group.negative(v) for v in group.values)
    if not by_negation and cues.negated != _negated(compiled.plan, context, named):
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
    found += _anchor_problems(compiled, question, context, cues)
    found += _statistic_problems(compiled, question)
    found += _denominator_problems(compiled, question, context)
    found += _unit_problems(compiled, question, context)
    for label, name in cues.values:
        if name not in compiled.columns:
            column = context.column(name)
            found.append(
                Problem(
                    "value",
                    feedback=f"The question names '{label}', a value of {name}: add the filter "
                    f'{{"column": "{name}", "op": "is", "value": null, "values": ["{label}"]}} '
                    "(or use it in numerator or group_by).",
                    reason=f"I could not use '{label}' ({column.display if column else name}) in my reading, "
                    "so I did not answer.",
                )
            )
    return found


def _statistic_problems(compiled: Compiled, question: str) -> list[Problem]:
    """A summary word of the question (median, average, highest, how many) the plan does not compute."""
    allowed: set[str] = set()
    for pattern, answers in _STATISTIC_WORDS:
        if pattern.search(question):
            allowed |= answers
    if not allowed or compiled.plan.answer in allowed or compiled.kind == "test":
        return []
    words = sorted(allowed)
    return [
        Problem(
            "statistic",
            feedback=f"The question asks for {' or '.join(words)}, but the reply's answer is {compiled.plan.answer}.",
            reason=f"Your question asks for {' or '.join(words)}, and my reading computed something else, so I did "
            "not answer.",
        )
    ]


def _denominator_problems(compiled: Compiled, question: str, context: DatasetContext) -> list[Problem]:
    """A share whose numerator is what the question names as the denominator ("the share of the
    women who ...": women are the denominator, a filter)."""
    if compiled.kind != "share":
        return []
    found: list[Problem] = []
    for match in _DENOMINATOR.finditer(question):
        phrase = _CLAUSE_END.sub("", match.group(1)).strip()
        if not phrase:
            continue
        for cond in compiled.numerator:
            column = context.column(cond.column)
            if column is not None and anchors(column, phrase, context):
                found.append(
                    Problem(
                        "denominator",
                        feedback=f"'{phrase}' is who the share is out of: put the condition on {column.name} in "
                        "filters, and the counted event in numerator.",
                        reason="I could not be sure which group your share is out of, so I did not answer.",
                    )
                )
    return found[:1]


def _unit_problems(compiled: Compiled, question: str, context: DatasetContext) -> list[Problem]:
    """A threshold said in another unit than its column's, or outside the range the dictionary
    gives the column or the values it holds."""
    found: list[Problem] = []
    for cond in (*compiled.filters, *compiled.numerator):
        column = context.column(cond.column)
        if column is None or not column.numeric or column.unit_column:
            continue
        for value in _floats(cond):
            said = _stated_unit(question, value, context)
            unit = (column.unit or "").casefold().rstrip("s")
            if said and unit and said != unit:
                found.append(
                    Problem(
                        "unit",
                        feedback="",
                        reason=f"Your question gives {_num(value)} {said}s, but {column.display} is recorded in "
                        f"{column.unit}, so I did not answer. Ask again in {column.unit}.",
                    )
                )
            low, high = column.stated_range or (column.low, column.high)
            if isinstance(low, int | float) and isinstance(high, int | float) and not low <= value <= high:
                found.append(
                    Problem(
                        "range",
                        feedback="",
                        reason=f"{_num(value)} is outside the values of {column.display} in this file "
                        f"({_num(float(low))} to {_num(float(high))}), so I did not answer. Check the number or its "
                        "unit.",
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
    """The contract's one hold rule (``contracts.holds``): an unanswered question about a column
    the plan reads (implicitly too: the order of first and last, a table's count column), about
    the whole table of groups, or about what a line is when the plan counts lines."""
    holds = open_question_holds(
        context.semantics, columns=compiled.columns, counts_lines=compiled.kind in ("count", "share")
    )
    return Ask("open_question", held_message(holds)) if holds else None


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
    if cues.numbers or cues.negated or cues.times or any(p.search(question) for p, _a in _STATISTIC_WORDS[:4]):
        return False  # a summary word: the model path checks the statistic
    if any(not anchors(c, question, context) for name in used if (c := context.column(name)) is not None):
        return False  # a column the question does not name
    named = [c for c in context.columns if c.kind != "key" and _named_strongly(c, question, context)]
    return all(c.name in used for c in named) and all(name in used for _label, name in cues.values)

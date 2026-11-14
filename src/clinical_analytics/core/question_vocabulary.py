# SPDX-License-Identifier: MIT
"""Words of a question that carry no limit, shared by every check that reads a question.

The cohort peel (cohort_constraints.peel_cohort), the gap check run before execution
(cohort_constraints.unapplied_constraints) and the word-accounting gate
(word_accounting.unused_words) must agree on which words say nothing about the patients.
If one of them skipped a word the others treat as a limit, a question could pass one check
and fail another, or worse, a limit could be dropped in one place and never noticed.

Adding a word here makes every check accept it in any question, so the lists stay small and
reviewed. A word that can narrow the patients (a negation, "or", a direction such as "over",
"only", "currently") never belongs here;
tests/core/test_over_holding.py::test_noLimitWords_vocabulary_sharesNoWordWithLogicOrNegation
enforces that for the logic and negation words the gate knows about.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, TypeAlias

# Articles, pronouns, auxiliaries, prepositions and politeness: grammar, never a limit.
FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a", "all", "among", "amongst", "an", "and", "any", "are", "as", "at", "be", "been", "being",
        "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "having", "i", "in",
        "is", "it", "its", "me", "my", "of", "on", "our", "please", "s", "that", "the", "their", "them",
        "there", "these", "they", "this", "those", "to", "up", "us", "was", "we", "were", "what", "which",
        "who", "whom", "whose", "will", "with", "within", "would", "you", "your",
    }
)  # fmt: skip

# Nouns that name the rows being counted or the dataset itself, not a subset of it.
SUBJECT_WORDS: frozenset[str] = frozenset(
    {
        "case", "cases", "cohort", "data", "dataset", "group", "groups", "individuals", "participant",
        "participants", "patient", "patients", "people", "person", "persons", "study", "subject",
        "subjects", "user", "users",
    }
)  # fmt: skip

# Verbs that only link a patient to a stored value ("patients who received placebo").
LINK_WORDS: frozenset[str] = frozenset({"given", "received", "receiving", "taking", "treated", "use", "uses", "using"})

NO_LIMIT_WORDS: frozenset[str] = FUNCTION_WORDS | SUBJECT_WORDS | LINK_WORDS

# Words that turn "<value>" into "everyone except <value>" when they stand right before it.
NEGATION_WORDS: frozenset[str] = frozenset(
    {"not", "non", "no", "without", "excluding", "exclude", "except", "outside", "never"}
)

# ---------------------------------------------------------------------------
# Negation scope: one rule for the cohort peel, the word-accounting gate and the justifier.
# ---------------------------------------------------------------------------

# Every word that can start an exclusion. No check reads a negation as an inverted comparison
# or as "every value but this one": a value or number a negation may reach is left unbound,
# so the plan holds. "other than" is rewritten to __neg__ before tokenising.
NEGATION_CUES: frozenset[str] = NEGATION_WORDS | {"__neg__", "remove", "removing", "drop", "omit", "rid"}

NegationScope: TypeAlias = Literal["negated", "clear", "uncertain"]


def negation_scope(before: Sequence[str]) -> NegationScope:
    """Whether a negation reaches the word that follows ``before`` (the question's words up to it).

    - "clear": no negation word comes before it.
    - "negated": the last negation stands right before it, with only words that carry no limit
      in between ("without diabetes").
    - "uncertain": a negation comes earlier, with other limits in between ("without diabetes
      and hypertension", "excluding women on placebo"). The words do not say whether the
      negation covers it, so no caller binds it and the plan holds.
    """
    cues = [at for at, word in enumerate(before) if word in NEGATION_CUES]
    if not cues:
        return "clear"
    if all(word in NO_LIMIT_WORDS for word in before[cues[-1] + 1 :]):
        return "negated"
    return "uncertain"


# ---------------------------------------------------------------------------
# Ranges: one pattern for the peel and the gate, so they always read the same two ends.
# ---------------------------------------------------------------------------

_NUM = r"\d+(?:\.\d+)?"
# "between 50 and 65", "40 to 60", "40-60", "40 - 60", "40–60". Both ends included.
RANGE_PATTERN: re.Pattern[str] = re.compile(
    rf"\b(?:between\s+(?P<lo>{_NUM})\s+and\s+(?P<hi>{_NUM})|(?P<lo2>{_NUM})\s*(?:to\b|-|–)\s*(?P<hi2>{_NUM}))\b",
    re.IGNORECASE,
)


def range_ends(match: re.Match[str]) -> tuple[str, str, str, str]:
    """(low text, high text, low group name, high group name) of a RANGE_PATTERN match, as written."""
    lo_group = "lo" if match.group("lo") is not None else "lo2"
    hi_group = "hi" if match.group("hi") is not None else "hi2"
    return match.group(lo_group), match.group(hi_group), lo_group, hi_group


def range_in_order(match: re.Match[str]) -> bool:
    """True when the range's first end is not above its second ("between 65 and 50" is not).

    A reversed range is never normalised: "between 65 and 50" may be a typo for either end,
    so neither the peel nor the gate binds it and the plan holds.
    """
    low, high, _lo, _hi = range_ends(match)
    return float(low) <= float(high)


# ---------------------------------------------------------------------------
# Hyphenated names: "COVID-19" is the word the data stores as "covid19".
# ---------------------------------------------------------------------------

_HYPHENATED = re.compile(r"\b([a-z]+)-(\d+)\b", re.IGNORECASE)
_NAME_WORD = re.compile(r"[a-z][a-z0-9]*")


def join_hyphenated_names(question: str, columns: Sequence[str]) -> str:
    """Write "<word>-<number>" as one word when a column name holds it as one ("COVID-19" ->
    "covid19" for covid19_confirmed_case), so no check reads the number as a cutoff or leaves it
    unused. Only words a column name spells that way are joined; "IL-6" stays as written when
    no column holds "il6", and a range such as "18-65" is never touched."""
    if "-" not in question:
        return question
    name_words = {word for column in columns for word in _NAME_WORD.findall(str(column).lower())}

    def joined(match: re.Match[str]) -> str:
        word = f"{match.group(1)}{match.group(2)}".lower()
        return word if word in name_words else match.group(0)

    return _HYPHENATED.sub(joined, question)


# ---------------------------------------------------------------------------
# Grouping phrases and the "<X> versus not <X>" comparison: one reading for the peel and the gate.
# ---------------------------------------------------------------------------

# "broken down by sex", "split by site" say only "by": rewritten before any check reads them.
_GROUPING_PHRASE = re.compile(r"\b(?:broken|split|stratified|separated)\s+(?:down\s+|out\s+)?by\b", re.IGNORECASE)
COMPARE_LEADS: frozenset[str] = frozenset({"versus", "vs", "or"})
COMPLEMENT_CUES: frozenset[str] = frozenset({"not", "non"})


def plain_grouping(text: str) -> str:
    """Write "broken down by", "split by", "stratified by" as "by"."""
    return _GROUPING_PHRASE.sub("by", text)


def complement_tail(tokens: Sequence[str]) -> int | None:
    """Where "<versus|vs|or> not <words already said>" ends the question, or None.

    "admitted to hospital versus not admitted" asks for both sides of one yes/no item: every
    word after "not" repeats a word said before the comparison, so the tail names nothing new.
    It never says what "not" applies to beyond those words, so a caller only uses it to show
    both stored values of that item side by side, never to compute everyone else.
    """
    for at in range(len(tokens) - 2):
        if tokens[at] not in COMPARE_LEADS or tokens[at + 1] not in COMPLEMENT_CUES:
            continue
        rest = [word for word in tokens[at + 2 :] if word not in FUNCTION_WORDS]
        said = set(tokens[:at])
        if rest and all(word in said for word in rest):
            return at
    return None

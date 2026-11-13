# SPDX-License-Identifier: MIT
"""Words of a question that carry no limit, shared by every check that reads a question.

The cohort peel (cohort_constraints.peel_cohort), the gap check run before execution
(cohort_constraints.unapplied_constraints) and the word-accounting gate
(word_accounting.unused_words) must agree on which words say nothing about the patients.
If one of them skipped a word the others treat as a limit, a question could pass one check
and fail another, or worse, a limit could be dropped in one place and never noticed.

Adding a word here makes every check accept it in any question, so the lists stay small and
reviewed. A word that can narrow the patients (a negation, "or", a direction such as "over",
"only", "currently") never belongs here; tests/core/test_question_vocabulary.py enforces that
for the logic words the gate knows about.
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

# Every word that can start an exclusion. The gate only accepts NEGATION_WORDS as used; the
# others ("drop", "remove") still put the words after them in doubt, so nothing after them
# is bound as asked. "other than" is rewritten to __neg__ before tokenising.
NEGATION_CUES: frozenset[str] = NEGATION_WORDS | {"__neg__", "remove", "removing", "drop", "omit", "rid"}

# "<value> non-users": the "non" excludes the word before it, never the words after it.
NON_USER_NOUNS: frozenset[str] = frozenset({"user", "users", "taker", "takers", "patient", "patients"})

# Words that may stand before "non" without being what it excludes ("how many non-users").
_LEAD_WORDS: frozenset[str] = frozenset(
    {
        "how", "many", "number", "count", "total", "proportion", "share", "percentage", "percent", "fraction",
        "average", "avg", "mean", "median", "minimum", "maximum", "min", "max", "std",
    }
)  # fmt: skip

# Words that join two items, so a negation before the first may or may not reach the second.
_COORDINATORS: frozenset[str] = frozenset({"and", "or", "nor"})

# The inverse of each comparison: "not over 65" is "65 or under".
INVERSE_OP: dict[str, str] = {"gt": "le", "lt": "ge", "ge": "lt", "le": "gt"}

NegationScope: TypeAlias = Literal["negated", "clear", "uncertain"]


def postfix_non(words: Sequence[str], at: int) -> bool:
    """True when ``words[at]`` is the "non" of "<value> non-users", which excludes the word before it."""
    return (
        words[at] == "non"
        and at >= 1
        and at + 1 < len(words)
        and words[at + 1] in NON_USER_NOUNS
        and words[at - 1] not in NO_LIMIT_WORDS
        and words[at - 1] not in _LEAD_WORDS
        and words[at - 1] not in NEGATION_CUES
    )


def negation_scope(before: Sequence[str]) -> NegationScope:
    """Whether a negation governs the word that follows ``before`` (the question's words up to it).

    - "negated": the last negation stands right before it, with only words that carry no limit
      in between ("not on placebo", "excluding those over 65").
    - "clear": no negation reaches it.
    - "uncertain": a negation comes earlier, with other limits in between ("without diabetes
      and hypertension", "excluding women on placebo", "not on placebo with diabetes"). The
      negation may cover the whole phrase or only its first item; the words do not say, so
      every caller leaves the word unbound and the plan holds.

    "non" written onto a word ("non-diabetic women") reaches only that word, unless the next
    word joins another item to it ("non-smokers and drinkers"): then the rest is uncertain.
    """
    cues = [at for at, word in enumerate(before) if word in NEGATION_CUES and not postfix_non(before, at)]
    if not cues:
        return "clear"
    last = cues[-1]
    between = list(before[last + 1 :])
    if all(word in NO_LIMIT_WORDS for word in between):
        return "negated"
    if before[last] == "non" and len(cues) == 1 and between[0] not in NO_LIMIT_WORDS:
        rest = between[1:]
        if not rest or rest[0] not in _COORDINATORS:
            return "clear"
    return "uncertain"


def postfix_negated(after: Sequence[str]) -> bool:
    """True when the word starting ``after`` is followed by "non-users" ("statin non-users")."""
    return len(after) >= 3 and after[1] == "non" and after[2] in NON_USER_NOUNS


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

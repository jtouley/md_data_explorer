# SPDX-License-Identifier: MIT
"""What a count or share counts on visit-level data, read from the question's own words.

Hospital extracts are usually one row per visit or test, with a patient ID and often an
encounter ID (an admission, a test). "How many patients tested positive" counts distinct
patients; "what share of tests were positive" counts distinct tests; "how many admissions"
counts distinct admissions. When a value differs between a patient's rows ("screened" at one
visit, not at the next), a share of patients needs a roll-up the question has to say:

* ``any``: the patient has the value on at least one row ("ever screened", "on any check",
  "at least one positive test");
* ``all``: on every recorded row ("always", "on every visit").

The app never guesses either. A question that names no unit is held by the grain check
(``answer_checks.grain_hold_reason``); a share whose value varies within a patient and names no
roll-up is refused with a plain choice (``rollup_choice``). Every word read here is recorded, so
word accounting treats it as used (``accounted_question``) and no other word is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from clinical_analytics.analysis.variable_types import short_name

Rollup: TypeAlias = Literal["any", "all"]
UnitKind: TypeAlias = Literal["patients", "encounters"]

PATIENT_NOUNS: frozenset[str] = frozenset(
    {"patient", "patients", "people", "person", "persons", "individuals", "adults", "children", "women", "men"}
)
# Nouns that count encounters: a distinct encounter ID when the data has one.
ENCOUNTER_NOUNS: frozenset[str] = frozenset(
    {
        "admission", "admissions", "encounter", "encounters", "test", "tests", "visit", "visits",
        "stay", "stays", "hospitalization", "hospitalizations", "hospitalisation", "hospitalisations",
    }
)  # fmt: skip
# Words that start what the question counts: the first unit noun after one is the counted unit.
_HEAD = re.compile(
    r"\b(?:how many|number of|count of|count|share of|proportion of|percentage of|percent of|"
    r"fraction of|rate of|what share|what proportion|what percentage|what fraction)\b"
)
_WORD = re.compile(r"[a-z][a-z0-9]*")
_OCCASION = r"(?:visit|visits|check|checks|test|tests|time|times|row|rows|admission|admissions|encounter|encounters)"
_ANY_PHRASES = (
    r"\bever\b",
    r"\bat least (?:one|once)\b",
    rf"\b(?:on|at|in|from) any (?:of (?:their|his|her) )?{_OCCASION}\b",
    rf"\bany {_OCCASION}\b",
)
_ALL_PHRASES = (
    r"\balways\b",
    rf"\b(?:on|at|in) (?:every|all(?: of)?(?: (?:their|his|her))?) {_OCCASION}\b",
    rf"\bevery {_OCCASION}\b",
)
_ORDER_PHRASE = re.compile(
    r"\b(?:first|last|latest|earliest|most recent|initial|final) "
    r"(?:visit|visits|check|checks|test|tests|admission|admissions|encounter|encounters|result|results|value|values)\b"
)
# An encounter ID: a column whose name ends in an ID word and names an encounter.
_ID_WORDS = frozenset({"id", "key", "no", "num", "number", "nbr"})
_ENCOUNTER_NAME_WORDS = frozenset(
    {"encounter", "enc", "admission", "admit", "adm", "hadm", "visit", "stay", "episode", "csn", "test", "accession"}
)


@dataclass(frozen=True)
class UnitWords:
    """The counted unit and roll-up the question names, and the phrases that name them."""

    unit: UnitKind | None = None
    noun: str | None = None  # the unit noun as asked ("admissions", "tests")
    rollup: Rollup | None = None
    rollup_phrases: tuple[str, ...] = ()
    both_rollups: bool = False  # "ever" and "always" in one question: refused, never guessed
    ordered: str | None = None  # "first test", "latest check": not supported yet


def _matches(patterns: tuple[str, ...], text: str) -> list[str]:
    return [m.group(0) for p in patterns for m in re.finditer(p, text)]


def read_units(question: str) -> UnitWords:
    """The unit the question counts and the roll-up it asks for (lowercase matching).

    The counted unit is the first patient or encounter noun after the counting words ("how
    many", "share of"), else the first one anywhere: "what share of tests in patients 65 and
    older" counts tests; "in patients 65+, how many admissions" counts admissions.
    """
    text = question.lower()
    head = _HEAD.search(text)
    words = [(m.start(), m.group(0)) for m in _WORD.finditer(text)]
    after = [w for at, w in words if head is not None and at >= head.end()]
    unit: UnitKind | None = None
    noun: str | None = None
    for word in [*after, *(w for _, w in words)]:
        if word in PATIENT_NOUNS:
            unit, noun = "patients", word
            break
        if word in ENCOUNTER_NOUNS:
            unit, noun = "encounters", word
            break
    any_found, all_found = _matches(_ANY_PHRASES, text), _matches(_ALL_PHRASES, text)
    rollup: Rollup | None = "any" if any_found else ("all" if all_found else None)
    ordered = _ORDER_PHRASE.search(text)
    return UnitWords(
        unit=unit,
        noun=noun,
        rollup=rollup,
        rollup_phrases=tuple(any_found or all_found),
        both_rollups=bool(any_found and all_found),
        ordered=ordered.group(0) if ordered else None,
    )


def accounted_question(question: str, words: UnitWords, encounter_counted: bool) -> str:
    """The question without the words this module read and the plan now uses: the roll-up
    phrase when the plan carries the roll-up, and the encounter noun when the plan counts the
    encounter ID. Word accounting checks every other word as before."""
    text = question
    for phrase in words.rollup_phrases:
        text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text, flags=re.IGNORECASE)
    if encounter_counted and words.noun:
        text = re.sub(rf"\b{re.escape(words.noun)}\b", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def encounter_key_candidates(columns: list[str], patient_key: str | None) -> list[str]:
    """Columns named as an encounter ID (``encounter_id``, ``hadm_id``, ``visit_id``, ``test_id``),
    in column order, never the patient key."""
    found = []
    for column in columns:
        if column == patient_key:
            continue
        parts = [p for p in re.split(r"[^a-z0-9]+", column.lower()) if p]
        if len(parts) >= 2 and parts[-1] in _ID_WORDS and _ENCOUNTER_NAME_WORDS & set(parts[:-1]):
            found.append(column)
        elif len(parts) == 1 and parts[0] in {"hadmid", "encounterid", "visitid", "admissionid", "csn"}:
            found.append(column)
    return found


def encounter_spans_patients(layer: Any, encounter: str, patient: str) -> bool:
    """Whether some encounter ID is on rows of more than one patient (a DuckDB aggregate): then
    it is not an encounter of one patient, and counting it would be counting something else."""
    view = layer.get_base_view()
    keyed = view.filter(view[encounter].notnull() & view[patient].notnull())
    per_encounter = keyed.group_by(encounter).aggregate(p=keyed[patient].nunique())
    return int(per_encounter.filter(per_encounter.p > 1).count().execute()) > 0


def no_encounter_key_reason(noun: str) -> str:
    return (
        f"This data has no column that identifies each of the {noun}, so {noun} can't be told apart from rows. "
        f"Ask for patients, or for rows."
    )


def ambiguous_encounter_reason(noun: str, candidates: list[str]) -> str:
    names = " and ".join(f"'{c}'" for c in candidates[:2])
    return (
        f"More than one column could identify the {noun} ({names}), so I can't tell which to count; "
        f"ask for one by name."
    )


def encounter_spans_reason(encounter: str, patient: str, noun: str) -> str:
    return (
        f"Some {short_name(encounter)} values are on rows of more than one {short_name(patient)}, so "
        f"{short_name(encounter)} does not identify one patient's {noun}; ask for patients or rows instead."
    )


def rollup_choice(column: str, unit: str) -> str:
    """The one plain choice for a share whose value differs between rows of one unit."""
    name = short_name(column)
    return (
        f"{name} differs between rows of the same {unit}. Say which {unit}s to count: those with it on at "
        f"least one row (for example 'ever' or 'on any visit'), or on every row ('always')."
    )


def both_rollups_reason() -> str:
    return "The question asks for 'at least one' and 'every' row at once; ask for one of them."


def ordered_rollup_reason(phrase: str) -> str:
    return (
        f"Counting by the {phrase} needs an order of each patient's rows, which isn't supported yet. Ask for "
        f"patients with it on at least one row ('ever') or on every row ('always')."
    )


ALL_ROLLUP_COUNT_REASON = (
    "A count of patients with a value on every row isn't supported yet; ask for the share with it on "
    "every row ('always'), or for patients with it on at least one row ('ever')."
)


ENCOUNTER_ALL_REASON = (
    "A share of admissions or tests with a value on every row isn't supported; ask for those with it on at "
    "least one row ('on any'), or count patients instead."
)

# The name words that identify each encounter noun's column when more than one could.
_NOUN_COLUMN_WORDS: dict[str, frozenset[str]] = {
    "admission": frozenset({"admission", "admit", "adm", "hadm"}),
    "hospitalization": frozenset({"admission", "admit", "adm", "hadm"}),
    "hospitalisation": frozenset({"admission", "admit", "adm", "hadm"}),
    "stay": frozenset({"stay", "admission", "hadm"}),
    "test": frozenset({"test", "accession"}),
    "visit": frozenset({"visit"}),
    "encounter": frozenset({"encounter", "enc", "csn"}),
}


def singular(noun: str | None) -> str:
    """'admissions' -> 'admission'; None -> 'row'."""
    if not noun:
        return "row"
    return noun[:-1] if noun.endswith("s") else noun


def pick_encounter_key(candidates: list[str], noun: str) -> str | None:
    """The one encounter ID column for ``noun``: the only candidate, else the only one whose name
    says the same kind of encounter ("tests" -> test_id beside encounter_id); None when unsure."""
    if len(candidates) == 1:
        return candidates[0]
    words = _NOUN_COLUMN_WORDS.get(singular(noun), frozenset())
    named = [c for c in candidates if words & set(re.split(r"[^a-z0-9]+", c.lower()))]
    return named[0] if len(named) == 1 else None

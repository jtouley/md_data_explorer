# SPDX-License-Identifier: MIT
"""What a count or share counts: the head noun of the question, mapped through the table's grain.

The model never chooses the counting unit. "How many patients", "what share of people" count
patients (``COUNT(DISTINCT <patient key>)`` on visit rows); "how many visits", "% of admissions",
"number of operations each year" count those events (their own ID column when the table has one
for that kind, else the table's rows, each of which is one event). A noun the table cannot count
(events on a table of one row per patient, patients on a table with no patient key, two ID
columns that could both be meant) is a question back to the user, never a fallback to another
unit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from clinical_analytics.core import counting_unit
from clinical_analytics.model_query.context import DatasetContext

HeadKind = Literal["patients", "events"]

# Nouns that count people. Words that also limit them ("women", "children", "smokers") count
# people too: the limit itself is checked by the coverage check.
PATIENT_NOUNS: Final = frozenset(
    {
        "patient", "patients", "pt", "pts", "people", "person", "persons", "individual", "individuals",
        "participant", "participants", "subject", "subjects", "member", "members", "resident", "residents",
        "women", "woman", "men", "man", "children", "child", "kids", "adults", "adult", "infants", "infant",
        "mothers", "newborns", "babies", "girls", "boys", "users", "smokers", "survivors", "enrollees",
        "folks",
    }
)  # fmt: skip
# Nouns that count events: one row (or one ID of that kind) each.
EVENT_NOUNS: Final = frozenset(
    {
        "visit", "visits", "encounter", "encounters", "admission", "admissions", "readmission", "readmissions",
        "hospitalization", "hospitalizations", "hospitalisation", "hospitalisations", "stay", "stays",
        "operation", "operations", "surgery", "surgeries", "procedure", "procedures", "case", "cases",
        "session", "sessions", "cycle", "cycles", "test", "tests", "episode", "episodes", "appointment",
        "appointments", "consultation", "consultations", "consult", "consults", "sample", "samples",
        "specimen", "specimens", "record", "records", "row", "rows", "entries", "entry", "prescriptions",
        "prescription", "discharges", "discharge", "deliveries", "delivery", "births", "transfusions",
        "result", "results", "measurement", "measurements", "reading", "readings", "draw", "draws",
        "assessment", "assessments", "observation", "observations", "lab", "labs", "infusion", "infusions",
        "attendance", "attendances", "exam", "exams",
    }
)  # fmt: skip
_HEAD: Final = re.compile(
    r"\b(?:how many|number of|no\. of|count of|count|total|share of|proportion of|percentage of|percent of|"
    r"fraction of|rate of|% of|what share|what proportion|what percentage|what percent|what fraction|"
    r"average|mean|median)\b|%\s*of",
    re.IGNORECASE,
)
_WORD: Final = re.compile(r"[a-z][a-z0-9]*")
_PREPOSITIONS: Final = frozenset({"on", "at", "in", "per", "during", "from", "across", "over"})
_DETERMINERS: Final = frozenset({"each", "every", "any", "all", "their", "a", "the"})
# Words between the counting words and the noun that do not end the search ("how many unique
# patients", "number of clinic visits", "% of all ED visits").
_HEAD_WINDOW: Final = 5


# Nouns for a measured value or a row: they count rows (each row is one), never an ID.
_ROW_NOUNS: Final = frozenset(
    {"record", "row", "entry", "result", "measurement", "reading", "draw", "assessment", "observation", "lab",
     "value", "exam"}
)  # fmt: skip


class UnitError(ValueError):
    """The question counts something the table cannot count as asked. The message is for the user."""


@dataclass(frozen=True)
class Unit:
    """What a count counts: the ID column counted distinct (None: rows), and its words."""

    key: str | None
    word: str  # patients | visits | rows (the grain word the presentation reads)
    noun: str | None = None  # one event as the question says it ("admission", "surgery")

    @property
    def plural(self) -> str:
        return plural(self.noun) if self.noun else self.word


def singular(noun: str) -> str:
    """'surgeries' -> 'surgery', 'admissions' -> 'admission', 'pts' -> 'pt'."""
    if noun.endswith("ies") and len(noun) > 4:
        return noun[:-3] + "y"
    if noun.endswith("ses") and noun[:-2] in ("case", "course"):
        return noun[:-1]
    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def plural(noun: str) -> str:
    if noun.endswith("y") and noun[-2:-1] not in ("a", "e", "o", "u"):
        return noun[:-1] + "ies"
    return noun if noun.endswith("s") else noun + "s"


def head_noun(question: str, extra_events: frozenset[str] = frozenset()) -> tuple[HeadKind, str] | None:
    """(patients | events, the noun) the question counts: the first unit noun within a few words
    after its counting words ("how many", "% of", "number of", "average"), else the first unit
    noun anywhere; None when it names none ("how many were positive?")."""
    text = question.casefold()
    words = [(m.start(), m.group(0)) for m in _WORD.finditer(text)]

    def kind(word: str) -> HeadKind | None:
        if word in PATIENT_NOUNS:
            return "patients"
        if word in EVENT_NOUNS or word in extra_events:
            return "events"
        return None

    def placed(i: int) -> bool:
        """The noun at ``i`` says when or where ("on record", "at each visit", "per visit")."""
        prev1 = words[i - 1][1] if i >= 1 else ""
        prev2 = words[i - 2][1] if i >= 2 else ""
        return prev1 in _PREPOSITIONS or (prev1 in _DETERMINERS and prev2 in _PREPOSITIONS)

    for head in _HEAD.finditer(text):
        after = [i for i, (at, _w) in enumerate(words) if at >= head.end()][:_HEAD_WINDOW]
        for i in after:
            found = kind(words[i][1])
            if found is not None and not placed(i):
                return found, words[i][1]
    # Elsewhere in the question a noun after a preposition says when or where ("on record", "at
    # each visit", "in 2024 admissions"), not what is counted; "record" there never counts.
    for i, (_at, word) in enumerate(words):
        if placed(i) or singular(word) in ("record", "row", "entry"):
            continue
        found = kind(word)
        if found is not None:
            return found, word
    return None


def resolve_unit(question: str, context: DatasetContext) -> Unit:
    """The counted unit from the question's head noun and the table's grain.

    Raises:
        UnitError: the noun cannot be counted on this table as asked.
    """
    found = head_noun(question, context.row_nouns)
    if context.rows_are_patients:
        if found is None or found[0] == "patients":
            return Unit(None, "patients")
        noun = singular(found[1])
        # One row per person or per event (a delivery, an operation): the rows are what is counted.
        # A number column that counts the noun ("prenatal_visits") is a measure, not the rows.
        counting = [
            c.name
            for c in context.columns
            if c.numeric and plural(noun) in set(re.findall(r"[a-z]+", c.name.casefold()))
        ]
        if counting:
            raise UnitError(
                f"Each row of this table is one record, and {', '.join(counting)} holds a number of {plural(noun)}. "
                f"Ask for the total or the average of {counting[0]}, or for the number of rows."
            )
        return Unit(None, "rows", noun)
    if found is None or found[0] == "patients":
        if context.patient_key is not None:
            return Unit(context.patient_key, "patients")
        if found is not None:
            raise UnitError(
                "No column of this table identifies patients, so patients cannot be told apart from rows. "
                "Ask how many rows, or upload the data with a patient ID column."
            )
        return Unit(None, "rows")
    noun = singular(found[1])
    if noun in _ROW_NOUNS:
        return Unit(None, "rows", noun if noun not in ("record", "row", "entry") else None)
    keys = list(context.encounter_keys)
    chosen = counting_unit.pick_encounter_key(keys, noun)
    if chosen is not None:
        return Unit(chosen, "visits", noun)
    if not keys:
        # No ID column for events: each (distinct) row of the table is one.
        return Unit(None, "visits", noun)
    if len(keys) == 1:
        return Unit(keys[0], "visits", noun)
    names = ", ".join(keys)
    raise UnitError(
        f"More than one column could identify the {plural(noun)} ({names}), so I can't tell which to count. "
        "Ask again naming the column."
    )

# SPDX-License-Identifier: MIT
"""Metamorphic checks of the one inference pass (ADR017): what a file means must not depend on
what its columns are called, the order they come in, how its dates are written, or the layout
of its data dictionary.

Each synthetic table and its dictionary (invented domains: splint fittings, iron infusions, a
table of falls, a shift-work survey, pack-years with only non-response codes, and a file that
only broad name conventions describe) is
transformed by (i) renaming every column to a neutral random token, consistently in the
dictionary; (ii) reordering the columns; (iii) writing its dates another way; (iv) laying the
dictionary out as prose, a Markdown table, or a codebook table. The file goes through staging
as an upload does. The semantics, mapped back through the renaming, must be the same; where the
original relied on a name alone (a broad key convention, a unit or timepoint word in a name, a
count or stay word exempting a column from the "not recorded" question), the transformed file
must ask instead (an open question), or drop a descriptive fact that changes no number. Never a
different fact.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest

from clinical_analytics.contracts.inference import UploadFacts, infer_semantics
from clinical_analytics.contracts.semantics import DatasetSemantics
from clinical_analytics.storage.staging import read_csv_typed

LAYOUTS = ("prose", "markdown", "codebook")
DATE_WRITINGS = ("%Y-%m-%d", "%d/%m/%Y", "%d %b %Y")
# Facts a name alone may give; a renamed file must ask instead, or drop a fact that changes no number.
QUESTION_KINDS_FOR_NAMES = frozenset({"patient_key", "rows_are_patients", "missing_code"})


@dataclass(frozen=True)
class Entry:
    """One dictionary entry, laid out by ``_render``."""

    column: str
    description: str
    codes: tuple[tuple[str, str], ...] = ()
    value_range: tuple[int, int] | None = None
    not_recorded: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Case:
    name: str
    frame: pl.DataFrame  # date columns as dates; written as text on upload
    statements: tuple[str, ...] = ()
    entries: tuple[Entry, ...] = ()
    # Columns whose facts in the original come from their names only (see the module docstring).
    named: frozenset[str] = field(default_factory=frozenset)


def _days(n: int, start: date = date(2021, 1, 14)) -> list[date]:
    return [start + timedelta(days=(i * 37) % 700) for i in range(n)]


def _fittings() -> Case:
    rows = [(o, k) for o in range(40) for k in range(1 + o % 4)]
    frame = pl.DataFrame(
        {
            "owner_ref": [f"OW{o:03d}" for o, _ in rows],
            "fitting_seq": [k + 1 for _, k in rows],
            "fitted_on": _days(len(rows)),
            "limb": [1 + (o + k) % 3 for o, k in rows],
            "grip": [999.0 if (o * 7 + k) % 23 == 0 else 10.5 + (o + 3 * k) % 40 for o, k in rows],
            "fitter_ref": [f"F{o % 5}" for o, _ in rows],
        }
    )
    entries = (
        Entry("owner_ref", "Splint owner"),
        Entry("fitting_seq", "Fitting order for the owner"),
        Entry("fitted_on", "Date the splint was fitted"),
        Entry("limb", "Limb splinted", codes=(("1", "arm"), ("2", "leg"), ("3", "trunk"))),
        Entry("grip", "Grip strength (kg)", value_range=(0, 60), not_recorded=(("999", "Not recorded"),)),
        Entry("fitter_ref", "Fitter identifier"),
    )
    statements = ("One row per splint fitting.", "The patient identifier is owner_ref.")
    return Case("fittings", frame, statements, entries)


def _infusions() -> Case:
    rows = [(p, k) for p in range(50) for k in range(1 + p % 3)]
    frame = pl.DataFrame(
        {
            "recipient": [f"R{(p * 7919) % 100000:05d}" for p, _ in rows],
            "dose_seq": [k + 1 for _, k in rows],
            "given": _days(len(rows), date(2022, 2, 20)),
            "ferritin": [-9 if (p + k) % 17 == 0 else 20 + (p * 11 + k) % 400 for p, k in rows],
            "reaction": [(p + k) % 3 for p, k in rows],
        }
    )
    entries = (
        Entry("recipient", "Infusion recipient"),
        Entry("dose_seq", "Dose order"),
        Entry("given", "Date given"),
        Entry(
            "ferritin",
            "Ferritin before the dose (ng/mL)",
            value_range=(5, 1000),
            not_recorded=(("-9", "Not recorded"),),
        ),
        Entry("reaction", "Reaction", codes=(("0", "none"), ("1", "mild"), ("2", "severe"))),
    )
    return Case("infusions", frame, ("One row per iron infusion.",), entries)


def _falls_table() -> Case:
    cells = [
        (a, y, b)
        for a in ("Northvale", "Eastmoor", "Westfold", "Southby")
        for y in (2019, 2020, 2021)
        for b in (1, 2, 3)
    ]
    frame = pl.DataFrame(
        {
            "area": [a for a, _, _ in cells],
            "period": [y for _, y, _ in cells],
            "band": [b for _, _, b in cells],
            "tally": [(i * 13) % 29 + 1 for i in range(len(cells))],
        }
    )
    entries = (
        Entry("area", "Health area"),
        Entry("period", "Calendar year"),
        Entry("band", "Age band", codes=(("1", "under 65"), ("2", "65 to 79"), ("3", "80 or over"))),
        Entry("tally", "Number of falls"),
    )
    return Case("falls_table", frame, ("Counts of falls by area, year and age band.",), entries)


def _shift_survey() -> Case:
    n = 150
    frame = pl.DataFrame(
        {
            "resp": list(range(5001, 5001 + n)),
            "sleep_h": [77 if i % 19 == 0 else 3 + i % 9 for i in range(n)],
            "rested": [1 + i % 2 for i in range(n)],
            "shift_start": _days(n, date(2023, 5, 15)),
        }
    )
    entries = (
        Entry("resp", "Respondent sequence number"),
        Entry("sleep_h", "Hours slept before the shift", value_range=(0, 16), not_recorded=(("77", "Don't know"),)),
        Entry("rested", "Felt rested", codes=(("1", "yes"), ("2", "no"))),
        Entry("shift_start", "Date the shift started"),
    )
    return Case("shift_survey", frame, ("One row per participant.",), entries)


def _smoking() -> Case:
    """A measure whose code list holds only non-response codes, outside its values."""
    n = 140
    frame = pl.DataFrame(
        {
            "smoker_ref": [f"Q{i:04d}" for i in range(n)],
            "pack": [-9 if i % 5 == 0 else -8 if i % 7 == 0 else 2 + (i * 3) % 55 for i in range(n)],
            "quit_on": _days(n, date(2020, 6, 18)),
        }
    )
    entries = (
        Entry("smoker_ref", "Smoker"),
        Entry(
            "pack", "Pack-years smoked", not_recorded=(("-9", "not applicable (never smoker)"), ("-8", "don't know"))
        ),
        Entry("quit_on", "Date the person stopped smoking"),
    )
    return Case("smoking", frame, ("One row per participant.",), entries)


def _named_only() -> Case:
    n = 120
    frame = pl.DataFrame(
        {
            "person_id": [f"P{i // 3:03d}" for i in range(n)],
            "hb_g_dl": [11.5 + (i % 30) / 10 for i in range(n)],
            "pain_pre": [3 + i % 6 for i in range(n)],
            "pain_post": [1 + i % 4 for i in range(n)],
            "stay_days": [99 if i % 10 == 0 else 2 + i % 8 for i in range(n)],
        }
    )
    named = frozenset({"person_id", "hb_g_dl", "pain_pre", "pain_post", "stay_days"})
    return Case("named_only", frame, named=named)


CASES = (_fittings, _infusions, _falls_table, _shift_survey, _smoking, _named_only)
NAMED_FREE = CASES[:-1]


# --------------------------------------------------------------------------- transforms
def _render(case: Case, layout: str, names: dict[str, str]) -> str | None:
    """The dictionary in one layout, with every column name replaced by ``names``."""
    if not case.entries and not case.statements:
        return None
    lines = [*case.statements, ""]
    if layout == "markdown":
        lines += ["| column | description |", "|---|---|"]
    for e in case.entries:
        col = names[e.column]
        ranged = f" ({e.value_range[0]}-{e.value_range[1]})" if e.value_range else ""
        codes = ", ".join(f"{c} = {label}" for c, label in (*e.codes, *e.not_recorded))
        if layout == "prose":
            lines.append(f"{col}  {e.description}{ranged}." + (f" {codes}" if codes else ""))
        elif layout == "markdown":
            lines.append(f"| {col} | {e.description}{ranged}." + (f" {codes}" if codes else "") + " |")
        else:
            lines += [f"Variable Name: {col}", f"SAS Label: {e.description}"]
            rows = [f"{c}\t{label}" for c, label in (*e.codes, *e.not_recorded)]
            if e.value_range:
                rows.insert(0, f"{e.value_range[0]} to {e.value_range[1]}\tRange of Values")
            if rows:
                lines += ["Code or Value\tValue Description", *rows]
            lines.append("")
    text = "\n".join(lines) + "\n"
    for old, new in sorted(names.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<![\w]){re.escape(old)}(?![\w])", new, text)
    return text


def _neutral_names(columns: list[str], seed: int) -> dict[str, str]:
    """A neutral token per column ("zqkvmx"): no word any rule reads."""
    rng = random.Random(seed)
    letters = "bcdfghjklmnpqrstvwxz"
    return {c: "zq" + "".join(rng.choice(letters) for _ in range(5)) for c in columns}


def _upload(
    case: Case, *, layout: str, writing: str, rename: bool, reorder: bool, seed: int
) -> tuple[DatasetSemantics, dict[str, str]]:
    """Semantics of the case's file as uploaded (written as CSV, staged, then inferred), and the
    map from each column's name in this file back to its original name."""
    names = _neutral_names(case.frame.columns, seed) if rename else {c: c for c in case.frame.columns}
    frame = case.frame.with_columns(pl.col(pl.Date).dt.strftime(writing)).rename(names)
    if reorder:
        frame = frame.select(list(reversed(frame.columns)))
    table = read_csv_typed(frame.write_csv().encode())
    facts = UploadFacts(doc_context=_render(case, layout, names), staged_dates={d.column: d for d in table.dates})
    return infer_semantics(table.data, facts), {new: old for old, new in names.items()}


# --------------------------------------------------------------------------- comparison
def _project(semantics: DatasetSemantics, back: dict[str, str]) -> dict[str, Any]:
    """Every fact, with columns named as in the original file; descriptive wording left out."""

    def m(column: str | None) -> str | None:
        return back.get(column, column) if column is not None else None

    key = semantics.patient_key
    agg = semantics.aggregate
    return {
        "key": (m(key.column), key.source, key.unique, key.confidence) if key else None,
        "grain": (
            semantics.grain.kind,
            semantics.grain.noun,
            semantics.grain.one_row_per_patient,
            semantics.grain.patients,
        ),
        "row_key": (semantics.row_key.kind, tuple(sorted(m(c) or "" for c in semantics.row_key.columns))),
        "sequences": sorted((m(s.column), m(s.within), s.start, s.kind) for s in semantics.sequence_columns),
        "groups": sorted((m(g.column), g.is_id) for g in semantics.grouping_columns),
        "episodes": sorted((m(e.column), m(e.within)) for e in semantics.episode_keys),
        "aggregate": (
            (agg.kind, m(agg.count_column), tuple(sorted(m(c) or "" for c in agg.strata)), agg.repeated_strata_lines)
            if agg
            else None
        ),
        "survey": semantics.survey_sample,
        "columns": {
            m(name): {
                "type": col.logical_type,
                "role": col.role,
                "date": (col.date is not None, col.date.has_time if col.date else None),
                "missing": sorted(str(c.value) for c in col.missing_codes),
                "labels": dict(col.labels),
                "range": (col.stated_range.minimum, col.stated_range.maximum) if col.stated_range else None,
                "unit": (col.unit.value, col.unit.source) if col.unit else None,
                "timepoint": (col.timepoint.measure is not None) if col.timepoint else False,
                "nonresponse": sorted(str(c.value) for c in col.nonresponse_codes),
            }
            for name, col in semantics.columns.items()
        },
        "questions": sorted((q.kind, m(q.column) or "", str(q.value)) for q in semantics.open_questions),
    }


def _name_only_differences(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """What changed, after removing the differences a name-only fact may make (see the module
    docstring). An empty list: the semantics are the same, or asked where a name had decided."""
    problems: list[str] = []
    asked_after = {q[0] for q in after["questions"]}
    for field_name in ("grain", "row_key", "sequences", "groups", "episodes", "aggregate", "survey"):
        if before[field_name] != after[field_name] and not (
            before["key"] != after["key"] and asked_after & {"patient_key", "rows_are_patients"}
        ):
            problems.append(f"{field_name}: {before[field_name]} -> {after[field_name]}")
    if before["key"] != after["key"]:
        name_only = before["key"] is not None and before["key"][1] == "structure" and before["key"][3] == "high"
        if not (name_only and asked_after & {"patient_key", "rows_are_patients"}):
            problems.append(f"key: {before['key']} -> {after['key']}")
    for column, was in before["columns"].items():
        now = after["columns"].get(column)
        if now is None:
            problems.append(f"{column}: gone")
            continue
        for fact, value in was.items():
            if value == now[fact]:
                continue
            if fact == "unit" and value is not None and value[1] == "structure" and now[fact] is None:
                continue  # a unit read from the name's unit grammar: descriptive, never a number
            if fact == "timepoint" and now[fact] is False:
                continue  # a timepoint word in a name: pairs columns for display only
            if fact == "role" and before["key"] != after["key"]:
                continue  # follows the key, which is asked
            problems.append(f"{column}.{fact}: {value} -> {now[fact]}")
    missing_questions = set(before["questions"]) - set(after["questions"])
    extra = {q for q in set(after["questions"]) - set(before["questions"]) if q[0] not in QUESTION_KINDS_FOR_NAMES}
    if before["key"] != after["key"]:
        missing_questions = {q for q in missing_questions if q[0] not in ("patient_key", "rows_are_patients")}
    problems += [f"question gone: {q}" for q in sorted(missing_questions)]
    problems += [f"question added: {q}" for q in sorted(extra)]
    return problems


# --------------------------------------------------------------------------- the checks
@pytest.mark.parametrize("build", CASES, ids=[c.__name__.lstrip("_") for c in CASES])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_inferSemantics_renamedReorderedRewrittenRelaidOut_sameMeaningOrAsked(build: Any, layout: str) -> None:
    # Arrange
    case = build()
    writing = DATE_WRITINGS[LAYOUTS.index(layout)]

    # Act
    original, same = _upload(case, layout="prose", writing="%Y-%m-%d", rename=False, reorder=False, seed=0)
    changed, back = _upload(case, layout=layout, writing=writing, rename=True, reorder=True, seed=len(layout))

    # Assert
    assert _name_only_differences(_project(original, same), _project(changed, back)) == []


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("build", NAMED_FREE, ids=[c.__name__.lstrip("_") for c in NAMED_FREE])
def test_inferSemantics_dictionaryOrStructureCases_identicalUnderEveryTransform(build: Any, layout: str) -> None:
    # Arrange: these files' facts come from their dictionaries and structure only
    case = build()

    # Act
    original, same = _upload(case, layout="prose", writing="%Y-%m-%d", rename=False, reorder=False, seed=0)
    changed, back = _upload(case, layout=layout, writing="%d %b %Y", rename=True, reorder=True, seed=7)

    # Assert: nothing at all differs, questions included
    assert _project(changed, back) == _project(original, same)


def test_inferSemantics_nameOnlyFacts_becomeQuestionsWhenRenamed() -> None:
    # Arrange: a file only broad name conventions describe
    case = _named_only()

    # Act
    original, _same = _upload(case, layout="prose", writing="%Y-%m-%d", rename=False, reorder=False, seed=0)
    changed, back = _upload(case, layout="prose", writing="%Y-%m-%d", rename=True, reorder=False, seed=3)
    renamed = {old: new for new, old in back.items()}

    # Assert: the convention key is asked once its name is gone; the stay's 99 is asked once
    # nothing says it counts days; the unit and the pairing (descriptive) are not guessed
    assert (original.patient_key_column, original.question("patient_key")) == ("person_id", None)
    assert changed.question("patient_key") is not None
    assert changed.patient_key_column in (None, renamed["person_id"])
    assert original.question("missing_code") is None
    assert changed.question("missing_code").column == renamed["stay_days"]
    assert original.columns["hb_g_dl"].unit.value == "g/dL"
    assert changed.columns[renamed["hb_g_dl"]].unit is None


@pytest.mark.parametrize("layout", LAYOUTS)
def test_inferSemantics_nonresponseCodesOnAMeasure_neverAveragedInAnyLayout(layout: str) -> None:
    # Arrange
    case = _smoking()

    # Act
    semantics, back = _upload(case, layout=layout, writing="%d/%m/%Y", rename=True, reorder=True, seed=11)
    pack = {old: new for new, old in back.items()}["pack"]

    # Assert: "don't know" is applied as missing; "not applicable" is asked and holds the column
    assert [str(m.value) for m in semantics.columns[pack].missing_codes] == ["-8"]
    assert semantics.columns[pack].labels == {}
    asked = [q for q in semantics.open_questions if q.kind == "missing_code" and q.column == pack]
    assert [str(q.value) for q in asked] == ["-9"]

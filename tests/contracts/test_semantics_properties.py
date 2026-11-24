# SPDX-License-Identifier: MIT
"""Properties of the contract's semantics that hold for every file (ADR017), over files drawn by
seeded generators (hypothesis is not a dependency; each seed is one reproducible file).

1. A value inside a column's stated range is never blanked without the user's declaration.
2. A frequency table's total row is never added into a SUM by any reader.
3. A confirmed missing-value code is applied by all three readers.
4. An open question holds every answer that reads what it concerns.
5. Renaming every column (and the dictionary with it) never changes an applied fact.

The three readers are the raw-frame read (the aggregate path), the semantic layer's view and
governed SQL's view (``read_by_every_reader`` in ``tests/conftest.py``). Invented domains only:
allergy skin-prick clinics, audiology booths, burns dressings, podiatry nail surgery, dialysis
fluid removal, hearing-aid fittings.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

import polars as pl
import pytest

from clinical_analytics.contracts.aggregates import count_cases
from clinical_analytics.contracts.declarations import (
    AggregateDeclaration,
    Declarations,
    MissingCodeDeclaration,
    QuestionAnswer,
)
from clinical_analytics.contracts.holds import COLUMN_KINDS, COUNT_KINDS, KEY_KINDS, TABLE_KINDS, open_question_holds
from clinical_analytics.contracts.inference import UploadFacts, apply_semantics, infer_semantics
from clinical_analytics.contracts.readings import declared_reads
from clinical_analytics.contracts.semantics import DatasetSemantics, Grain, OpenQuestion, with_declarations
from tests.contracts.conftest import applied_facts

SEEDS = range(24)
EXPLICIT = ("Refused", "Don't know", "Not asked", "Missing", "Not recorded")
OTHER_NONRESPONSE = ("not applicable", "unknown", "not applicable (no reaction)", "skipped")
ORDINARY = ("trace", "no wheal", "largest seen", "reading repeated")
READERS = ("polars", "layer", "sql")


def _frame_with_codes(rng: random.Random) -> tuple[pl.DataFrame, str, str, tuple[int, int]]:
    """A measure with a stated range and a random code list, codes inside or outside the range,
    written as prose or as a codebook table."""
    n = rng.randint(40, 90)
    low = rng.choice((0, 5, 20))
    high = low + rng.choice((15, 40, 120))
    column = rng.choice(("wheal", "booth_db", "dressing_area", "fluid_off"))
    codes: dict[int, str] = {}
    for _ in range(rng.randint(1, 4)):
        inside = rng.random() < 0.4
        code = rng.randint(low, high) if inside else rng.choice((-9, -8, -7, -1, high + 1, 777, 888, 999))
        codes[code] = rng.choice(EXPLICIT + OTHER_NONRESPONSE + ORDINARY)
    values = [rng.choice(list(codes)) if rng.random() < 0.2 else rng.randint(low, high) for _ in range(n)]
    frame = pl.DataFrame({"visit_ref": [f"V{i:04d}" for i in range(n)], column: values})
    if rng.random() < 0.5:
        listed = ", ".join(f"{c} = {label}" for c, label in codes.items())
        doc = f"{column}  Reading at the clinic ({low}-{high}). {listed}\n"
    else:
        rows = "\n".join(f"{c}\t{label}" for c, label in codes.items())
        doc = (
            f"Variable Name: {column}\nSAS Label: Reading at the clinic\nCode or Value\tValue Description\n"
            f"{low} to {high}\tRange of Values\n{rows}\n"
        )
    return frame, column, doc, (low, high)


def _stored(frame: pl.DataFrame, doc: str | None) -> tuple[pl.DataFrame, DatasetSemantics]:
    """The typed Parquet's frame and its semantics, as the upload writes them."""
    semantics = infer_semantics(frame, UploadFacts(doc_context=doc)).model_copy(update={"dataset_version": "v1"})
    return apply_semantics(frame, semantics), semantics


# --------------------------------------------------------------------------- 1. values in range
@pytest.mark.parametrize("seed", SEEDS)
def test_everyReader_valueInsideStatedRange_neverBlankedWithoutDeclaration(
    seed: int, read_by_every_reader: Callable[..., dict[str, pl.DataFrame]]
) -> None:
    # Arrange
    frame, column, doc, (low, high) = _frame_with_codes(random.Random(seed))
    stored, semantics = _stored(frame, doc)

    # Act
    reads = read_by_every_reader(stored, declared_reads(Declarations(), "v1", semantics), "visit_ref")

    # Assert: every line whose value lies in the stated range reads as that value, in every reader
    original = frame.sort("visit_ref").get_column(column)
    inside = [i for i, v in enumerate(original.to_list()) if low <= v <= high]
    for name in READERS:
        read = reads[name].get_column(column).cast(pl.Float64).to_list()
        assert [read[i] for i in inside] == [float(original[i]) for i in inside], (name, doc)


@pytest.mark.parametrize("seed", SEEDS)
def test_inferSemantics_nonresponseCodeOutsideRange_neverACategoryOfAMeasure(seed: int) -> None:
    # Arrange
    frame, column, doc, (low, high) = _frame_with_codes(random.Random(seed))

    # Act
    semantics = infer_semantics(frame, UploadFacts(doc_context=doc))
    col = semantics.columns[column]
    asked = {float(q.value) for q in semantics.open_questions if q.kind == "missing_code" and q.column == column}

    # Assert: a listed non-response code outside the range is missing or asked, never a category
    for code, label in re.findall(r"(-?\d+)\s*(?:=|\t)\s*([^,\n]+)", doc.split("\n", 1)[-1] if "\t" in doc else doc):
        value = int(code)
        if low <= value <= high or label.strip() in ORDINARY or value not in frame.get_column(column).to_list():
            continue
        assert code not in col.labels, (code, label, doc)
        assert value in {m.value for m in col.missing_codes} or float(value) in asked, (code, label, doc)


# --------------------------------------------------------------------------- 2. totals
GRAND_TOTALS = ("All", "Total", "All areas", "Overall")


def _counts_table(rng: random.Random) -> tuple[pl.DataFrame, str, set[str], str]:
    """Nail-surgery counts by area and year, with total lines per year (labelled or not)."""
    areas = [f"Area {chr(65 + i)}" for i in range(rng.randint(4, 9))]
    years = list(range(2010, 2010 + rng.randint(6, 14)))
    rows = [{"area": a, "year": y, "operations": rng.randint(0, 60)} for a in areas for y in years]
    sums = {y: sum(r["operations"] for r in rows if r["year"] == y) for y in years}
    labelled = rng.random() < 0.6
    label = rng.choice(GRAND_TOTALS) if labelled else "Area sum"
    rows += [{"area": label, "year": y, "operations": n} for y, n in sums.items()]
    rng.shuffle(rows)
    return pl.DataFrame(rows), label, set(areas), "operations"


@pytest.mark.parametrize("seed", SEEDS)
def test_everyReader_frequencyTableTotalRow_neverAddedIntoASum(
    seed: int, read_by_every_reader: Callable[..., dict[str, pl.DataFrame]]
) -> None:
    # Arrange: the user said the lines count cases; a total found by its sums alone is confirmed
    # ("yes, it is a total") or left open
    rng = random.Random(seed)
    frame, label, areas, count = _counts_table(rng)
    frame = frame.with_row_index("line_no")
    semantics = infer_semantics(frame.drop("line_no")).model_copy(update={"dataset_version": "v1"})
    answers = tuple(
        QuestionAnswer(question_id=q.id, answer=q.options[0], dataset_version="v1")
        for q in semantics.open_questions
        if q.kind == "total_row" and rng.random() < 0.5
    )
    declarations = Declarations(
        aggregates=(AggregateDeclaration(count_column=count, rows=frame.height, dataset_version="v1"),),
        answers=answers,
    )
    declared = with_declarations(semantics, declarations)
    truth = int(frame.filter(pl.col("area").is_in(sorted(areas)))[count].sum())

    # Act
    held = open_question_holds(declared, columns={count}, counts_lines=False)
    reads = read_by_every_reader(frame, declared_reads(declarations, "v1", declared), "line_no")
    aggregate_path = count_cases(frame, declared.aggregate, {}) if declared.aggregate else None

    # Assert: an answer is held until the total is settled; otherwise no reader adds the total
    if held:
        assert {h.kind for h in held} <= TABLE_KINDS | {"missing_code"}
        return
    for name in READERS:
        assert int(reads[name].get_column(count).sum()) == truth, (name, label)
    assert aggregate_path is not None and aggregate_path.total == truth


# --------------------------------------------------------------------------- 3. confirmed codes
@pytest.mark.parametrize("seed", SEEDS)
def test_everyReader_confirmedMissingCode_appliedByAllThree(
    seed: int, read_by_every_reader: Callable[..., dict[str, pl.DataFrame]]
) -> None:
    # Arrange: hearing-aid fittings with an undocumented sentinel the user confirms
    rng = random.Random(seed)
    n = rng.randint(30, 80)
    code = rng.choice((99, 999, 9999, -1, 777))
    as_float = rng.random() < 0.5
    values = [code if rng.random() < 0.15 else rng.randint(10, 80) for _ in range(n)]
    frame = pl.DataFrame(
        {"fitting_ref": [f"F{i:03d}" for i in range(n)], "gain": [float(v) for v in values] if as_float else values}
    )
    stored, semantics = _stored(frame, None)
    declarations = Declarations(
        missing_codes=(MissingCodeDeclaration(column="gain", value=code, missing=True, dataset_version="v1"),)
    )

    # Act
    reads = read_by_every_reader(stored, declared_reads(declarations, "v1", semantics), "fitting_ref")

    # Assert: the code is missing in every reader, every other value as stored
    expected = [None if v == code else float(v) for v in values]
    for name in READERS:
        assert reads[name].get_column("gain").cast(pl.Float64).to_list() == expected, name


@pytest.mark.parametrize("seed", SEEDS)
def test_everyReader_documentedExplicitCode_missingInEveryReader(
    seed: int, read_by_every_reader: Callable[..., dict[str, pl.DataFrame]]
) -> None:
    # Arrange: dialysis fluid removal (mL) whose own entry lists explicit non-response codes
    rng = random.Random(seed)
    n = rng.randint(40, 90)
    codes = rng.sample((-9, -8, -7, 7777, 8888), k=rng.randint(1, 3))
    values = [rng.choice(codes) if rng.random() < 0.2 else rng.randint(200, 3500) for _ in range(n)]
    frame = pl.DataFrame({"session_ref": [f"S{i:03d}" for i in range(n)], "fluid_off": values})
    listed = ", ".join(f"{c} = {rng.choice(EXPLICIT)}" for c in codes)
    stored, semantics = _stored(frame, f"fluid_off  Fluid removed in the session, mL. {listed}\n")

    # Act
    reads = read_by_every_reader(stored, declared_reads(Declarations(), "v1", semantics), "session_ref")

    # Assert: the mean every reader gives leaves the codes out
    true_mean = sum(v for v in values if v not in codes) / sum(1 for v in values if v not in codes)
    for name in READERS:
        assert reads[name].get_column("fluid_off").mean() == pytest.approx(true_mean), name


# --------------------------------------------------------------------------- 4. holds
KINDS = sorted(COLUMN_KINDS | TABLE_KINDS | COUNT_KINDS | KEY_KINDS)


def _random_questions(rng: random.Random, columns: list[str]) -> tuple[OpenQuestion, ...]:
    questions = []
    for i in range(rng.randint(1, 6)):
        kind = rng.choice(KINDS)
        column = rng.choice(columns)
        options = tuple(rng.sample(columns, k=2)) if kind in KEY_KINDS else ("yes", "no")
        questions.append(
            OpenQuestion(
                id=f"{kind}:{column}:{i}",
                kind=kind,
                question=f"Question {i}?",
                column=None if kind in TABLE_KINDS | COUNT_KINDS else column,
                options=options,
                answered=rng.random() < 0.3,
            )
        )
    return tuple(questions)


@pytest.mark.parametrize("seed", SEEDS)
def test_openQuestionHolds_unansweredQuestion_holdsEveryAnswerReadingItsColumn(seed: int) -> None:
    # Arrange
    rng = random.Random(seed)
    columns = ["burn_ref", "tbsa", "dressing", "pain", "seen_on"]
    questions = _random_questions(rng, columns)
    grain = Grain(kind="record", noun="record", rows=10, one_row_per_patient=False, source="default")
    semantics = DatasetSemantics(rows=10, grain=grain)
    semantics = semantics.model_copy(update={"open_questions": questions})
    read = set(rng.sample(columns, k=rng.randint(1, len(columns))))
    counts_lines = rng.random() < 0.5

    # Act
    held = {h.question_id for h in open_question_holds(semantics, columns=read, counts_lines=counts_lines)}
    held_all = {h.question_id for h in open_question_holds(semantics, columns=None, counts_lines=False)}

    # Assert: each open question holds what reads it; an answered one holds nothing
    for q in questions:
        if q.answered:
            assert q.id not in held and q.id not in held_all
            continue
        reads_it = q.kind in TABLE_KINDS or (q.kind in COUNT_KINDS and counts_lines)
        reads_it |= q.kind in COLUMN_KINDS and q.column in read
        reads_it |= q.kind in KEY_KINDS and (counts_lines or bool({q.column, *q.options} & read))
        assert (q.id in held) is reads_it, (q.kind, q.column, read, counts_lines)
        assert (q.id in held_all) is (q.kind not in COUNT_KINDS), q.kind


@pytest.mark.parametrize("seed", SEEDS)
def test_openQuestionHolds_questionsFromInference_holdTheirColumn(seed: int) -> None:
    # Arrange: audiology booths with an undocumented sentinel and a day/month date question
    rng = random.Random(seed)
    n = rng.randint(40, 80)
    sentinel = rng.choice((999, 9999, -1))
    frame = pl.DataFrame(
        {
            "booth_ref": [f"B{i:03d}" for i in range(n)],
            "threshold": [sentinel if i % 9 == 0 else rng.randint(0, 90) for i in range(n)],
            "tested_on": [f"{1 + i % 12:02d}/{1 + (i * 5) % 12:02d}/2021" for i in range(n)],
        }
    )

    # Act
    semantics = infer_semantics(frame)

    # Assert: every column question holds an answer reading its column, and only that column
    for q in semantics.open_questions:
        if q.kind not in COLUMN_KINDS:
            continue
        assert q.id in {h.question_id for h in open_question_holds(semantics, columns={q.column}, counts_lines=False)}
        others = {c for c in frame.columns if c != q.column}
        assert q.id not in {h.question_id for h in open_question_holds(semantics, columns=others, counts_lines=False)}


# --------------------------------------------------------------------------- 5. renaming
def _podiatry_file(rng: random.Random) -> tuple[pl.DataFrame, str]:
    n = rng.randint(60, 120)
    frame = pl.DataFrame(
        {
            "toe_ref": [f"T{i % (n // 2):03d}" for i in range(n)],
            "nail_score": [rng.choice((-9, -8)) if i % 11 == 0 else rng.randint(0, 30) for i in range(n)],
            "relief": [rng.choice((1, 2, 3, 7)) for _ in range(n)],
            "cut_on": [f"{1 + i % 28:02d}/{1 + (i * 7) % 12:02d}/2022" for i in range(n)],
            "wait": [rng.choice((999, rng.randint(1, 40))) if i % 13 == 0 else rng.randint(1, 40) for i in range(n)],
        }
    )
    doc = (
        "toe_ref  Person whose toe was operated on\n"
        "nail_score  Nail score (0-30). -9 = not applicable, -8 = refused\n"
        "relief  Relief: 1 = none, 2 = some, 3 = full, 7 = don't know\n"
        "cut_on  Date of the nail surgery\n"
        "wait  Days from referral to surgery\n"
    )
    return frame, doc


@pytest.mark.parametrize("seed", SEEDS)
def test_inferSemantics_everyColumnRenamed_appliedFactsUnchanged(seed: int) -> None:
    # Arrange: neutral tokens for every column, in the file and its dictionary alike
    rng = random.Random(seed)
    frame, doc = _podiatry_file(rng)
    letters = "bcdfghjklmnpqrstvwxz"
    names = {c: "zq" + "".join(rng.choice(letters) for _ in range(6)) for c in frame.columns}
    renamed_doc = doc
    for old, new in names.items():
        renamed_doc = re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", new, renamed_doc)

    # Act
    before = infer_semantics(frame, UploadFacts(doc_context=doc))
    after = infer_semantics(frame.rename(names), UploadFacts(doc_context=renamed_doc))

    # Assert
    assert applied_facts(after, {new: old for old, new in names.items()}) == applied_facts(
        before, {c: c for c in frame.columns}
    )

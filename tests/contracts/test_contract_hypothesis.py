# SPDX-License-Identifier: MIT
"""The contract's five invariants (ADR017) as hypothesis properties, searched and shrunk.

``test_semantics_properties`` checks the same invariants over seeded files; here hypothesis draws
the files (values, codes, labels, layouts, questions and column names) and shrinks any failure
to its smallest file. Derandomized, so every run draws the same examples.

1. A value inside a column's stated range is never blanked without a declaration.
2. No reader includes a frequency table's total row in a SUM.
3. Every reader applies a confirmed missing code.
4. An open question holds every answer that reads its column.
5. Renaming columns consistently in the data and the dictionary never changes an applied fact.

Invented domains only: orthoptics squint clinics, lactation consults, tinnitus booths, plaster
rooms, travel-vaccine clinics.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from clinical_analytics.contracts import config
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

# Function-scoped fixtures here are factories with no state shared between examples.
PROPERTY = settings(
    max_examples=40,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow, HealthCheck.data_too_large],
)
READERS = ("polars", "layer", "sql")
EXPLICIT = ("Refused", "Don't know", "Not asked", "Missing", "Not recorded")
OTHER_NONRESPONSE = ("not applicable", "unknown", "skipped")
ORDINARY = ("faint", "repeat reading", "largest seen")
SENTINELS = (-9, -8, -7, -1, 777, 888, 999)


def _stored(frame: pl.DataFrame, doc: str | None) -> tuple[pl.DataFrame, DatasetSemantics]:
    """The typed Parquet's frame and its semantics, as the upload writes them."""
    semantics = infer_semantics(frame, UploadFacts(doc_context=doc)).model_copy(update={"dataset_version": "v1"})
    return apply_semantics(frame, semantics), semantics


# --------------------------------------------------------------------------- 1. values in range
@st.composite
def _measure_files(draw: st.DrawFn) -> tuple[pl.DataFrame, str, str, tuple[int, int]]:
    """A squint-angle measure with a stated range and a code list (codes inside or outside the
    range, any label), written as prose or as a codebook table."""
    low = draw(st.sampled_from((0, 1, 5, 20)))
    high = low + draw(st.integers(15, 150))
    column = draw(st.sampled_from(("squint_angle", "prism_reading", "deviation_near")))
    codes = draw(
        st.dictionaries(
            st.one_of(st.integers(low, high), st.sampled_from(SENTINELS + (high + 1,))),
            st.sampled_from(EXPLICIT + OTHER_NONRESPONSE + ORDINARY),
            min_size=1,
            max_size=4,
        )
    )
    # Spread over the range (a measure, not a coded item), with drawn lines holding the codes.
    n, step, shift = draw(st.integers(40, 80)), draw(st.integers(1, 7)), draw(st.integers(0, 9))
    spread = [low + (i * step + shift) % (high - low + 1) for i in range(n)]
    coded = draw(st.dictionaries(st.integers(0, n - 1), st.sampled_from(sorted(codes)), min_size=1, max_size=n // 4))
    values = [coded.get(i, v) for i, v in enumerate(spread)]
    frame = pl.DataFrame({"eye_ref": [f"E{i:03d}" for i in range(len(values))], column: values})
    if draw(st.booleans()):
        listed = ", ".join(f"{c} = {label}" for c, label in codes.items())
        doc = f"{column}  Angle at the clinic ({low}-{high}). {listed}\n"
    else:
        rows = "\n".join(f"{c}\t{label}" for c, label in codes.items())
        doc = (
            f"Variable Name: {column}\nSAS Label: Angle at the clinic\nCode or Value\tValue Description\n"
            f"{low} to {high}\tRange of Values\n{rows}\n"
        )
    return frame, column, doc, (low, high)


@PROPERTY
@given(drawn=_measure_files())
def test_everyReader_drawnValueInsideStatedRange_neverBlankedWithoutDeclaration(
    drawn: tuple[pl.DataFrame, str, str, tuple[int, int]], read_by_every_reader: Callable[..., dict[str, pl.DataFrame]]
) -> None:
    # Arrange
    frame, column, doc, (low, high) = drawn
    stored, semantics = _stored(frame, doc)

    # Act
    reads = read_by_every_reader(stored, declared_reads(Declarations(), "v1", semantics), "eye_ref")

    # Assert
    original = frame.get_column(column).to_list()
    inside = [i for i, v in enumerate(original) if low <= v <= high]
    for name in READERS:
        read = reads[name].get_column(column).cast(pl.Float64).to_list()
        assert [read[i] for i in inside] == [float(original[i]) for i in inside], (name, doc)


# --------------------------------------------------------------------------- 2. totals
@st.composite
def _feed_counts(draw: st.DrawFn) -> tuple[pl.DataFrame, str, list[str]]:
    """Lactation consults by ward and month, with a total line per month (labelled or not)."""
    wards = [f"Ward {chr(65 + i)}" for i in range(draw(st.integers(4, 7)))]
    months = list(range(1, draw(st.integers(4, 10)) + 1))
    counts = draw(st.lists(st.integers(0, 60), min_size=len(wards) * len(months), max_size=len(wards) * len(months)))
    rows = [{"ward": w, "month": m, "consults": counts[i * len(months) + j]} for i, w in enumerate(wards) for j, m in
            enumerate(months)]  # fmt: skip
    label = draw(st.sampled_from(("All", "Total", "Overall", "Ward sum")))
    rows += [
        {"ward": label, "month": m, "consults": sum(r["consults"] for r in rows if r["month"] == m)} for m in months
    ]
    order = draw(st.permutations(range(len(rows))))
    return pl.DataFrame([rows[i] for i in order]).with_row_index("line_no"), label, wards


@PROPERTY
@given(drawn=_feed_counts(), confirm=st.booleans())
def test_everyReader_drawnFrequencyTotalRow_neverAddedIntoASum(
    drawn: tuple[pl.DataFrame, str, list[str]],
    confirm: bool,
    read_by_every_reader: Callable[..., dict[str, pl.DataFrame]],
) -> None:
    # Arrange: the user says the lines count consults, and may confirm the total line
    frame, label, wards = drawn
    semantics = infer_semantics(frame.drop("line_no")).model_copy(update={"dataset_version": "v1"})
    answers = tuple(
        QuestionAnswer(question_id=q.id, answer=q.options[0], dataset_version="v1")
        for q in semantics.open_questions
        if confirm and q.kind == "total_row" and q.value == label
    )
    declarations = Declarations(
        aggregates=(AggregateDeclaration(count_column="consults", rows=frame.height, dataset_version="v1"),),
        answers=answers,
    )
    declared = with_declarations(semantics, declarations)
    truth = int(frame.filter(pl.col("ward").is_in(wards))["consults"].sum())

    # Act
    held = open_question_holds(declared, columns={"consults"}, counts_lines=False)
    reads = read_by_every_reader(frame, declared_reads(declarations, "v1", declared), "line_no")

    # Assert: held until the total is settled; once read, no reader adds the total line
    if held:
        assert {h.kind for h in held} <= TABLE_KINDS | {"missing_code"}
        return
    for name in READERS:
        assert int(reads[name].get_column("consults").sum()) == truth, (name, label)
    assert declared.aggregate is not None
    assert count_cases(frame, declared.aggregate, {}).total == truth


# --------------------------------------------------------------------------- 3. confirmed codes
@PROPERTY
@given(
    code=st.sampled_from((99, 999, 9999, -1, 777, 0)),
    values=st.lists(st.integers(10, 80), min_size=30, max_size=70),
    coded_lines=st.sets(st.integers(0, 29), min_size=1, max_size=10),
    as_float=st.booleans(),
)
def test_everyReader_drawnConfirmedMissingCode_appliedByAllThree(
    code: int,
    values: list[int],
    coded_lines: set[int],
    as_float: bool,
    read_by_every_reader: Callable[..., dict[str, pl.DataFrame]],
) -> None:
    # Arrange: tinnitus loudness matches with a code the user confirms means not recorded
    values = [code if i in coded_lines else v for i, v in enumerate(values)]
    frame = pl.DataFrame(
        {
            "booth_ref": [f"B{i:03d}" for i in range(len(values))],
            "loudness": [float(v) for v in values] if as_float else values,
        }
    )
    stored, semantics = _stored(frame, None)
    declarations = Declarations(
        missing_codes=(MissingCodeDeclaration(column="loudness", value=code, missing=True, dataset_version="v1"),)
    )

    # Act
    reads = read_by_every_reader(stored, declared_reads(declarations, "v1", semantics), "booth_ref")

    # Assert
    expected = [None if v == code else float(v) for v in values]
    for name in READERS:
        assert reads[name].get_column("loudness").cast(pl.Float64).to_list() == expected, name


# --------------------------------------------------------------------------- 4. holds
COLUMNS = ("cast_ref", "limb", "plaster_kind", "itch_score", "removed_on")
KINDS = sorted(COLUMN_KINDS | TABLE_KINDS | COUNT_KINDS | KEY_KINDS)


@st.composite
def _questions(draw: st.DrawFn) -> tuple[OpenQuestion, ...]:
    found = []
    for i in range(draw(st.integers(1, 6))):
        kind = draw(st.sampled_from(KINDS))
        column = draw(st.sampled_from(COLUMNS))
        options = tuple(draw(st.lists(st.sampled_from(COLUMNS), min_size=2, max_size=2, unique=True)))
        found.append(
            OpenQuestion(
                id=f"{kind}:{column}:{i}",
                kind=kind,
                question=f"Question {i}?",
                column=None if kind in TABLE_KINDS | COUNT_KINDS else column,
                options=options if kind in KEY_KINDS else ("yes", "no"),
                answered=draw(st.booleans()),
            )
        )
    return tuple(found)


@settings(max_examples=200, deadline=None, derandomize=True, database=None)
@given(
    questions=_questions(),
    read=st.one_of(st.none(), st.sets(st.sampled_from(COLUMNS))),
    counts_lines=st.booleans(),
)
def test_openQuestionHolds_drawnQuestions_holdEveryAnswerReadingTheirColumn(
    questions: tuple[OpenQuestion, ...], read: set[str] | None, counts_lines: bool
) -> None:
    # Arrange
    grain = Grain(kind="record", noun="cast", rows=10, one_row_per_patient=False, source="default")
    semantics = DatasetSemantics(rows=10, grain=grain, open_questions=questions)

    # Act
    held = {h.question_id for h in open_question_holds(semantics, columns=read, counts_lines=counts_lines)}

    # Assert: an open question holds exactly the answers that read what it concerns
    every = read is None
    for q in questions:
        if q.answered:
            assert q.id not in held
            continue
        reads_it = q.kind in TABLE_KINDS or (q.kind in COUNT_KINDS and counts_lines)
        reads_it |= q.kind in COLUMN_KINDS and (every or q.column in (read or ()))
        reads_it |= q.kind in KEY_KINDS and (counts_lines or every or bool({q.column, *q.options} & (read or set())))
        assert (q.id in held) is reads_it, (q.kind, q.column, read, counts_lines)


# --------------------------------------------------------------------------- 5. renaming
@st.composite
def _vaccine_files(draw: st.DrawFn) -> tuple[pl.DataFrame, str]:
    """A travel-vaccine clinic file and its dictionary: a coded reaction, a measure with
    non-response codes, a text date and a wait with an undocumented sentinel."""
    n = draw(st.integers(60, 100))
    arm = draw(st.lists(st.integers(0, 40), min_size=n, max_size=n))
    reaction = draw(st.lists(st.sampled_from((1, 2, 3, 8)), min_size=n, max_size=n))
    wait = draw(st.lists(st.integers(1, 30), min_size=n, max_size=n))
    frame = pl.DataFrame(
        {
            "traveller_ref": [f"T{i % (n // 2):03d}" for i in range(n)],
            "arm_swelling": [draw(st.sampled_from((-9, -8))) if i % 9 == 0 else v for i, v in enumerate(arm)],
            "reaction": reaction,
            "jabbed_on": [f"{1 + i % 28:02d}/{1 + (i * 5) % 12:02d}/2023" for i in range(n)],
            "wait": [999 if i % 12 == 0 else v for i, v in enumerate(wait)],
        }
    )
    doc = (
        "traveller_ref  Person given the vaccine\n"
        "arm_swelling  Swelling at the site (0-40). -9 = not applicable, -8 = refused\n"
        "reaction  Reaction: 1 = none, 2 = mild, 3 = marked, 8 = don't know\n"
        "jabbed_on  Date of the vaccine\n"
        "wait  Days from booking to the vaccine\n"
    )
    return frame, doc


NEUTRAL = st.text(alphabet="bcdfghjklmnpqrstvwxz", min_size=6, max_size=6).map(lambda s: "zq" + s)


@PROPERTY
@given(drawn=_vaccine_files(), new_names=st.lists(NEUTRAL, min_size=5, max_size=5, unique=True))
def test_inferSemantics_drawnConsistentRenaming_appliedFactsUnchanged(
    drawn: tuple[pl.DataFrame, str], new_names: list[str]
) -> None:
    # Arrange: the same neutral names in the file and its dictionary
    frame, doc = drawn
    names = dict(zip(frame.columns, new_names, strict=True))
    renamed_doc = doc
    for old, new in names.items():
        renamed_doc = re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", new, renamed_doc)

    # Act
    before = infer_semantics(frame, UploadFacts(doc_context=doc))
    after = infer_semantics(frame.rename(names), UploadFacts(doc_context=renamed_doc))

    # Assert
    back = {new: old for old, new in names.items()}
    assert applied_facts(after, back) == applied_facts(before, {c: c for c in frame.columns})


def test_feedCounts_drawnTotalLabels_areContractTotalLevels() -> None:
    # Assert: the labelled totals property 2 draws are total labels of the contract (else it tests nothing)
    assert {"all", "total", "overall"} <= config.ESTIMATE_TOTAL_LEVELS

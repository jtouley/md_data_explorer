# SPDX-License-Identifier: MIT
"""Tables of groups, documented codes and date answers (ADR017, third and fourth review rounds).

Synthetic tables only, in new domains: Lyme disease notifications, legionella counts, hepatitis A
outbreak tallies, rheumatology clinic blood pressure, sleep-clinic pack-years, osteoporosis
infusions, haemodialysis adequacy, newborn screening.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.contracts.aggregates import count_cases
from clinical_analytics.contracts.declarations import (
    AggregateDeclaration,
    DateReadingDeclaration,
    Declarations,
    MemoryDeclarationStore,
    MissingCodeDeclaration,
)
from clinical_analytics.contracts.dictionary import parse_dictionary
from clinical_analytics.contracts.inference import UploadFacts, apply_semantics, infer_semantics
from clinical_analytics.contracts.readings import apply_declared, declared_reads
from clinical_analytics.contracts.semantics import describe, with_declarations

STATES = [f"State {i:02d}" for i in range(15)]
YEARS = list(range(1995, 2020))


def _facts(doc: str | None = None) -> UploadFacts:
    return UploadFacts(doc_context=doc)


def _lyme(repeat: int = 0, total: bool = False) -> pl.DataFrame:
    """Lyme disease notifications by state, year and sex, a few strata repeated."""
    rows = [
        {"state": s, "year": y, "sex": sex, "cases": (i * 7 + y) % 40}
        for i, s in enumerate(STATES)
        for y in YEARS
        for sex in ("F", "M")
    ]
    rows += rows[:repeat]
    if total:
        sums: dict[tuple[int, str], int] = {}
        for r in rows:
            sums[(r["year"], r["sex"])] = sums.get((r["year"], r["sex"]), 0) + r["cases"]
        rows += [{"state": "United States", "year": y, "sex": sex, "cases": n} for (y, sex), n in sums.items()]
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- frequency vs estimate
def test_inferSemantics_frequencyTableWithRepeatedStrata_recognisedAndAsked() -> None:
    # Act
    semantics = infer_semantics(_lyme(repeat=6))

    # Assert: a real frequency table with a few repeated strata is never rejected
    assert semantics.aggregate is not None
    assert (semantics.aggregate.kind, semantics.aggregate.count_column) == ("frequency", "cases")
    assert semantics.aggregate.repeated_strata_lines == 12
    assert semantics.question("repeated_strata") is not None
    assert semantics.question("rows_are_patients") is None


def test_countCases_countTableWithTotalRows_neverSummedAcrossTheTotal() -> None:
    # Arrange: lyme counts with a "United States" line per year and sex that is the sum of the states
    frame = _lyme(total=True)
    states = _lyme()
    aggregate = infer_semantics(frame).aggregate
    assert aggregate is not None and aggregate.kind == "frequency"

    # Act
    read = count_cases(frame, aggregate, {"year": [2010], "sex": ["F"]})
    open_sex = count_cases(frame, aggregate, {"year": [2010]})

    # Assert: label and sums agree, so the total line is left out of every sum, never added twice
    assert aggregate.total_values("state") == ("United States",)
    assert read.total == states.filter((pl.col("year") == 2010) & (pl.col("sex") == "F"))["cases"].sum()
    assert open_sex.total == states.filter(pl.col("year") == 2010)["cases"].sum()


def test_countCases_frequencyTable_sumsTheCountColumn() -> None:
    # Arrange
    frame = _lyme()
    aggregate = infer_semantics(frame).aggregate
    assert aggregate is not None

    # Act
    result = count_cases(frame, aggregate, {"year": [2010]})

    # Assert
    assert result.total == int(frame.filter(pl.col("year") == 2010).get_column("cases").sum())


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(
            pl.DataFrame(
                {
                    "sex": ["F", "M"] * 100,
                    "ward": ["A", "B", "C", "D"] * 50,
                    "age": [20 + i % 60 for i in range(200)],
                    "smoker": ["Y", "N"] * 100,
                    "n_visits": [i % 7 for i in range(200)],
                    "sbp": [100 + i % 70 for i in range(200)],
                }
            ),
            id="record_level_with_age_and_sbp",
        ),
    ],
)
def test_inferSemantics_recordLevelFileWithCountColumn_notASummaryTable(frame: pl.DataFrame) -> None:
    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert semantics.aggregate is None
    assert semantics.grain.kind != "stratum"


def test_inferSemantics_looksAggregateNotRecognised_askedNeverPatientSilently() -> None:
    # Arrange: hepatitis A tallies keyed by numeric codes only (no names of groups)
    frame = pl.DataFrame(
        {
            "county_code": [i % 12 for i in range(48)],
            "week": [i // 12 for i in range(48)],
            "cases": [i % 9 for i in range(48)],
        }
    )

    # Act
    semantics = infer_semantics(frame)

    # Assert
    question = semantics.question("aggregate_unsure")
    assert question is not None and "in cases" in question.question


def test_inferSemantics_dictionarySaysCountsByGroup_evidence() -> None:
    # Arrange
    frame = pl.DataFrame(
        {"state": STATES * 2, "year": [2019] * 15 + [2020] * 15, "rate_ratio": [1.5] * 30, "cases": list(range(30))}
    )

    # Act
    semantics = infer_semantics(frame, _facts("Each row is a count of cases by state and year.\n"))

    # Assert
    assert semantics.aggregate is not None and semantics.aggregate.source == "dictionary"


def test_withDeclarations_thirdAnswer_groupsWithCountColumn() -> None:
    # Arrange
    frame = pl.DataFrame({"record_code": [f"R{i}" for i in range(40)], "cases": [i % 5 for i in range(40)]})
    semantics = infer_semantics(frame).model_copy(update={"dataset_version": "v1"})
    declared = AggregateDeclaration(count_column="cases", rows=40, dataset_version="v1")

    # Act
    after = with_declarations(semantics, Declarations(aggregates=(declared,)))

    # Assert
    assert after.aggregate is not None and after.aggregate.count_column == "cases"
    assert (after.grain.kind, after.case_count_column, after.patient_key) == ("stratum", "cases", None)


def test_recordAnswer_countsCasesIn_storesTableKindOrRefusesNonCount(make_semantics: Callable[..., Any]) -> None:
    # Arrange
    from clinical_analytics.core.patient_rows import PatientRowsNotAskedError, question_token, record_answer

    frame = pl.DataFrame(
        {
            "record_code": [f"R{i:03d}" for i in range(60)],
            "cases": [i % 5 for i in range(60)],
            "sex": ["F", "M"] * 30,
            "ahi": [5.5 + i for i in range(60)],
        }
    )
    semantics = make_semantics(frame)
    metadata = {"dataset_version": "v1", "row_count": 60}
    store = MemoryDeclarationStore()
    token = str(question_token(metadata, semantics))

    # Act
    counts = record_answer(
        "ds", metadata, semantics, store, confirmation=token, one_row_per_patient=False, counts_cases_in="cases"
    )

    # Assert
    assert counts == "cases"
    assert store.load("ds").aggregate_for("v1") == AggregateDeclaration(
        count_column="cases", rows=60, dataset_version="v1"
    )
    with pytest.raises(PatientRowsNotAskedError):
        record_answer(
            "ds", metadata, semantics, store, confirmation=token, one_row_per_patient=False, counts_cases_in="sex"
        )


# --------------------------------------------------------------------------- documented codes
@pytest.mark.parametrize(
    "line",
    [
        "sbp  Systolic BP, mmHg. 999 = not recorded, 888 = refused",
        "sbp  Systolic BP, mmHg. 999 = not recorded, 998 = missing",
    ],
)
def test_applySemantics_twoCodesOnOneMeasure_bothMissingMeanTrue(line: str) -> None:
    # Arrange: rheumatology clinic blood pressure
    codes = [999, int(line.split(", ")[-1].split(" ")[0])]
    frame = pl.DataFrame({"sbp": [130, 140, codes[0], 150, codes[1], 135]})

    # Act
    applied = apply_semantics(frame, infer_semantics(frame, _facts(line + "\n")))

    # Assert
    assert applied.get_column("sbp").mean() == pytest.approx(138.75)


def test_inferSemantics_sharedNoteForAllMeasurements_askedNeverApplied() -> None:
    # Arrange
    frame = pl.DataFrame(
        {"pt_id": [f"P{i}" for i in range(6)], "alt": [20, 9999, 30, 40, 25, 22], "ast": [18, 25, 9999, 30, 20, 19]}
    )
    doc = "pt_id  Patient identifier\nalt  ALT, U/L\nast  AST, U/L\n\nFor all measurements, 9999 = not recorded.\n"

    # Act
    semantics = infer_semantics(frame, _facts(doc))
    applied = apply_semantics(frame, semantics)

    # Assert: a prose note is a question for each column holding the code (ADR017 principle A)
    assert (applied.get_column("alt").null_count(), applied.get_column("ast").null_count()) == (0, 0)
    asked = {q.column for q in semantics.open_questions if q.kind == "missing_code" and q.value == 9999}
    assert asked == {"alt", "ast"}


@pytest.mark.parametrize(
    ("column", "values", "asked"),
    [
        pytest.param("age_years", [40, 999, 999, 999, 55] * 20, [999], id="age_999_asked"),
        pytest.param("pack_years", [10, -9, -8, -9, -8, 20] * 20, [-9, -8], id="two_negative_codes"),
        pytest.param("los_days", [3, 5, 99, 99, 99, 4] * 20, [], id="stay_99_real"),
    ],
)
def test_inferSemantics_undocumentedCodes_askedPerValue(column: str, values: list[int], asked: list[int]) -> None:
    # Act
    questions = [
        q.value for q in infer_semantics(pl.DataFrame({column: values})).open_questions if q.kind == "missing_code"
    ]

    # Assert
    assert sorted(questions) == sorted(asked)


def test_applyDeclared_confirmedCodeOnEstimates_neverShown() -> None:
    # Arrange: an estimates table; the user confirms -1 deaths means not recorded
    frame = pl.DataFrame({"sex": ["F", "M", "F"], "deaths": [-1.0, 12.0, 30.0]})
    answer = MissingCodeDeclaration(column="deaths", value=-1, missing=True, dataset_version="v1")

    # Act
    applied = apply_declared(frame, declared_reads(Declarations(missing_codes=(answer,)), "v1"))

    # Assert
    assert applied.get_column("deaths").to_list() == [None, 12.0, 30.0]


def test_qualityReport_openMissingCodeQuestion_listedAsNeedingAnswer(make_contract: Callable[..., Any]) -> None:
    # Arrange
    from clinical_analytics.contracts.quality import run_quality

    frame = pl.DataFrame({"mrn": [f"M{i}" for i in range(100)], "trop_ng_l": [4.0, 999.0, 8.0, 12.0] * 25})
    contract = make_contract(frame, semantics=infer_semantics(frame), id_column="mrn")

    # Act
    report = run_quality(frame.lazy(), contract)

    # Assert
    assert any(c.message.startswith("Needs your answer: Does 999 in trop_ng_l") for c in report.checks)


# --------------------------------------------------------------------------- dates, roles, units
def test_chooseTextFormat_twoDigitBirthYearsInThePast_stillACenturyQuestion() -> None:
    # Arrange
    from clinical_analytics.contracts.dates import DateAmbiguity, choose_text_format

    # Act
    found = choose_text_format(pl.Series("dob", ["03/15/21", "04/20/23", "12/30/22"]))

    # Assert: born in 1921 or 2021: never settled silently
    assert isinstance(found, DateAmbiguity) and found.kind == "century"


def test_applyDeclared_centuryAnswer_datesInThe1900s() -> None:
    # Arrange
    frame = pl.DataFrame({"dob": ["03/15/21", "04/20/23"]})
    reading = DateReadingDeclaration(column="dob", format="%m/%d/%y", century="1900s", dataset_version="v1")

    # Act
    applied = apply_declared(frame, declared_reads(Declarations(date_readings=(reading,)), "v1"))

    # Assert
    assert applied.get_column("dob").to_list() == [date(1921, 3, 15), date(1923, 4, 20)]


@pytest.mark.parametrize("column", ["days_of_therapy", "hours_of_sleep"])
def test_inferSemantics_durationNames_neverTimeSinceSequences(column: str) -> None:
    # Arrange: osteoporosis infusions per visit
    frame = pl.DataFrame({"pt_id": [f"P{i // 3}" for i in range(60)], column: [(i * 7) % 11 for i in range(60)]})

    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert all(s.kind != "time_since_index" for s in semantics.sequence_columns)
    assert "since" not in semantics.grain.noun


def test_inferSemantics_babyUniqueMotherIdsWithBlanks_noQuestion() -> None:
    # Act
    frame = pl.DataFrame(
        {"infant_id": [f"B{i}" for i in range(60)], "mat_id": [None if i % 10 == 0 else f"M{i}" for i in range(60)]}
    )

    # Assert
    assert infer_semantics(frame).question("person_role") is None


@pytest.mark.parametrize(
    ("column", "unit"),
    [("bw_g", "g"), ("ga_d", "days"), ("vitamin_d", None), ("iron_g", None)],
)
def test_inferSemantics_singleLetterUnit_onlyAfterAMeasuringStem(column: str, unit: str | None) -> None:
    # Act
    found = infer_semantics(pl.DataFrame({column: [3.5, 2.5]})).column(column).unit

    # Assert
    assert (found.value if found else None) == unit


def test_parseDictionary_domainUnit_ktv() -> None:
    # Act / Assert: haemodialysis adequacy
    assert parse_dictionary("ktv  Dialysis adequacy (Kt/V)\n", ["ktv"]).entry("ktv").unit == "Kt/V"


def test_describe_sequenceNoun_pluralSpelledRight() -> None:
    # Arrange
    rows = [{"mother_ref": f"M{m}", "pregnancy_no": k, "parity": k} for m in range(20) for k in (1, 2)]

    # Act
    lines = describe(infer_semantics(pl.DataFrame(rows)))

    # Assert
    assert not any("pregnancys" in line for line in lines)


def test_applySemantics_frameUnchangedWhenNothingDeclared() -> None:
    # Act / Assert
    frame = pl.DataFrame({"a": [1, 2]})
    assert_frame_equal(apply_declared(frame, declared_reads(Declarations(), "v1")), frame)


def test_sqlCatalogView_confirmedCodeAndDateReading_readByDuckdb(
    make_contract: Callable[..., Any], tmp_path: Any
) -> None:
    # Arrange: governed SQL reads the same confirmations as every other reader
    import duckdb

    from clinical_analytics.sql.catalog import SqlCatalog

    frame = pl.DataFrame(
        {"pt_id": ["P1", "P2", "P3"], "sbp": [120.0, 999.0, 140.0], "seen": ["03/04/21", "05/06/21", "07/08/21"]}
    )
    path = tmp_path / "t.parquet"
    frame.write_parquet(path)
    missing = (MissingCodeDeclaration(column="sbp", value=999, missing=True, dataset_version="v1"),)
    dates = (DateReadingDeclaration(column="seen", format="%m/%d/%y", dataset_version="v1"),)
    catalog = SqlCatalog.from_contract(make_contract(frame, id_column="pt_id"), path, (), missing, dates)

    # Act
    con = duckdb.connect()
    con.execute(catalog.view_sql())
    rows = con.execute('SELECT sbp, seen FROM "unified" ORDER BY pt_id').fetchall()

    # Assert
    assert rows == [(120.0, date(2021, 3, 4)), (None, date(2021, 5, 6)), (140.0, date(2021, 7, 8))]

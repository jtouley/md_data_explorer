# SPDX-License-Identifier: MIT
"""Third check of ADR017 inference, from synthetic probes: a shared dictionary note never nulls a
real value, totals are found by label or by their sums, records are not tables of groups, and
rates are values whatever their type.

Synthetic tables only, in new domains: care-home residents, hand clinics, scabies notifications,
a patient-experience survey, fall-injury rate tables, newborn screening, diabetes clinics.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from clinical_analytics.contracts.aggregates import count_cases
from clinical_analytics.contracts.dictionary import free_missing_pairs, parse_dictionary
from clinical_analytics.contracts.inference import UploadFacts, infer_semantics

REGIONS = ("North", "South", "East", "West")


def _asked(semantics: Any, question_id: str) -> Any:
    return next((q for q in semantics.open_questions if q.id == question_id), None)


def _residents() -> pl.DataFrame:
    """400 care-home residents aged 60-104, 11 of them really 99; a few really take 99 doses a week."""
    ages = [60 + i % 45 for i in range(400)]
    ages = [99 if i < 11 else (a if a != 99 else 98) for i, a in enumerate(ages)]
    return pl.DataFrame(
        {
            "resident_id": [f"R{i:03d}" for i in range(400)],
            "age": ages,
            "n_meds": [99 if i % 100 == 0 else i % 14 for i in range(400)],
            "mood_score": [99 if i % 25 == 0 else i % 28 for i in range(400)],
        }
    )


# --------------------------------------------------------------------------- 1. shared notes
def test_inferSemantics_sharedNoteCode_askedOnEveryColumnNeverApplied() -> None:
    # Arrange
    facts = UploadFacts(doc_context="resident_id  Resident identifier\n\nCoded fields: 99 = unknown.\n")

    # Act
    semantics = infer_semantics(_residents(), facts)

    # Assert: a prose note never applies a code (ADR017 principle A), wherever 99 lies
    for column in ("age", "n_meds", "mood_score"):
        assert semantics.columns[column].missing_codes == ()
        assert _asked(semantics, f"missing_code:{column}:99") is not None


def test_inferSemantics_columnOwnEntryListsCode_appliedEvenInsideValues() -> None:
    # Arrange: the column's own entry documents 99, so it is applied (the dictionary is specific)
    facts = UploadFacts(doc_context="resident_id  Resident identifier\nage  Age in years. 99 = not recorded\n")

    # Act
    semantics = infer_semantics(_residents(), facts)

    # Assert
    assert [m.value for m in semantics.columns["age"].missing_codes] == [99]


# --------------------------------------------------------------------------- 7. notes written as sentences
_TAKEN = "the measurement was not taken"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(f"A value of 999 indicates {_TAKEN}.", ("999", _TAKEN), id="indicates"),
        pytest.param("999 is used where no reading was available.", ("999", "no reading was available"), id="used"),
        pytest.param("Missing values were coded as -9.", ("-9", "Missing values"), id="coded-as"),
        pytest.param("Grip was coded as -8 when the test was refused.", ("-8", "the test was refused"), id="when"),
    ],
)
def test_freeMissingPairs_sentence_codeAndLabel(text: str, expected: tuple[str, str]) -> None:
    # Act
    pairs = free_missing_pairs(text)

    # Assert
    assert pairs == [expected]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("1 means the splint was worn.", id="real-label"),
        pytest.param("Scores of 5 indicate severe pain.", id="score-meaning"),
        pytest.param("", id="empty"),
    ],
)
def test_freeMissingPairs_noNotRecordedWords_nothing(text: str) -> None:
    # Act / Assert
    assert free_missing_pairs(text) == []


def test_parseDictionary_sentenceNamingColumn_notedForThatColumnOtherwiseShared() -> None:
    # Arrange
    named = "In grip_kg, a value of 999 indicates the reading was not taken."
    text = f"{named}\nMissing values were coded as -9.\n"

    # Act
    found = parse_dictionary(text, ["grip_kg", "pinch_kg"])

    # Assert: both are recorded as notes to ask about, never as the column's code list
    assert found.noted_missing == {"grip_kg": {"999": named}}
    assert found.shared_missing == {"-9": "Missing values were coded as -9."}
    assert found.entry("grip_kg") is None


# --------------------------------------------------------------------------- 3. totals
def _scabies(total: str) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in range(2012, 2018):
        for sex in ("Female", "Male"):
            counts = {r: 3 + (year + i) % 5 for i, r in enumerate(REGIONS)}
            rows += [{"region": r, "year": year, "sex": sex, "cases": n} for r, n in counts.items()]
            rows.append({"region": total, "year": year, "sex": sex, "cases": sum(counts.values())})
    return pl.DataFrame(rows)


@pytest.mark.parametrize("total", ["Grand Total", "All ages", "Both sexes", "National", "Total"])
def test_inferSemantics_totalLabelAndSumsAgree_leftOutOfSums(total: str) -> None:
    # Arrange
    frame = _scabies(total)
    groups = frame.filter(pl.col("region") != total)

    # Act
    semantics = infer_semantics(frame)
    counted = count_cases(frame, semantics.aggregate, {"year": [2013]})

    # Assert: a frequency table; the sum over both sexes leaves the total lines out
    assert semantics.aggregate.kind == "frequency"
    assert any(total in t.values for t in semantics.aggregate.total_levels)
    assert counted.total == groups.filter(pl.col("year") == 2013)["cases"].sum()


@pytest.mark.parametrize("total", ["Statewide", "Health board 9", "Subtotal", "Total, all regions"])
def test_inferSemantics_totalBySumsAlone_askedNotApplied(total: str) -> None:
    # Act
    semantics = infer_semantics(_scabies(total))

    # Assert
    assert semantics.aggregate.total_levels == ()
    assert _asked(semantics, f"total_row:region:{total}") is not None


@pytest.mark.parametrize("label", ["All other causes", "Knee replacement, total", "Overall"])
def test_inferSemantics_realCategoryNamedLikeATotal_neverATotal(label: str) -> None:
    # Arrange: a real category whose counts are not the sum of the others
    frame = _scabies(label).with_columns(
        pl.when(pl.col("region") == label).then(pl.lit(2)).otherwise(pl.col("cases")).alias("cases")
    )

    # Act
    semantics = infer_semantics(frame)
    counted = count_cases(frame, semantics.aggregate, {"year": [2013]})

    # Assert: added like every other group ("Overall" is asked about, never applied)
    assert semantics.aggregate.total_levels == ()
    assert counted.total == frame.filter(pl.col("year") == 2013)["cases"].sum()


def test_inferSemantics_rareLevelEqualToASumInTwoCells_askedNotApplied() -> None:
    # Arrange: "Northvale" happens to equal the sum of the other two states in its only two cells
    frame = pl.DataFrame(
        {
            "state": ["Eastmoor", "Westfold", "Northvale", "Eastmoor", "Westfold", "Northvale", "Eastmoor", "Westfold"],
            "year": [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022],
            "cases": [20, 28, 48, 1, 3, 4, 7, 9],
        }
    )

    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert semantics.aggregate.total_levels == ()
    assert _asked(semantics, "total_row:state:Northvale") is not None
    assert count_cases(frame, semantics.aggregate, {"state": ["Northvale"]}).total == 52


def test_inferSemantics_frequencyTableWithoutTotals_noTotalFound() -> None:
    # Arrange
    frame = _scabies("Islands").filter(pl.col("region") != "Islands")

    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert semantics.aggregate.kind == "frequency"
    assert semantics.aggregate.total_levels == ()


# --------------------------------------------------------------------------- 4. records, not groups
def test_inferSemantics_surveyOfBandsAndOneCount_notAFrequencyTableAsked() -> None:
    # Arrange: 150 respondents, banded answers and a visit count each
    frame = pl.DataFrame(
        {
            "region": [REGIONS[i % 4] for i in range(150)],
            "marital": [("single", "married", "widowed")[i % 3] for i in range(150)],
            "age_band": [("18-39", "40-64", "65+")[i % 3 // 2] for i in range(150)],
            "n_visits": [i % 6 for i in range(150)],
        }
    )

    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert semantics.aggregate is None
    assert semantics.question("aggregate_unsure") is not None
    assert semantics.question("rows_are_patients") is not None


# --------------------------------------------------------------------------- 6. integer rates
def test_inferSemantics_integerRateByName_estimateNotStratum() -> None:
    # Arrange
    frame = pl.DataFrame(
        [
            {"state": s, "year": y, "deaths": 10 + i + y % 7, "rate_per_100k": 8 + (i + y) % 9}
            for i, s in enumerate(("Northvale", "Eastmoor", "Westfold", "Southby", "Midmarsh"))
            for y in range(2010, 2020)
        ]
    )

    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert semantics.aggregate.kind == "estimate"
    assert semantics.aggregate.value_columns == ("rate_per_100k",)
    assert "rate_per_100k" not in semantics.aggregate.strata


def test_inferSemantics_integerRateByDictionary_estimateNotStratum() -> None:
    # Arrange: a neutral name; the dictionary says it is an age-adjusted rate
    frame = pl.DataFrame(
        [
            {"state": s, "year": y, "deaths": 10 + i + y % 7, "value_adj": 8 + (i + y) % 9}
            for i, s in enumerate(("Northvale", "Eastmoor", "Westfold", "Southby", "Midmarsh"))
            for y in range(2010, 2020)
        ]
    )
    doc = "value_adj  Age-adjusted fall-injury death rate per 100,000\n"

    # Act
    semantics = infer_semantics(frame, UploadFacts(doc_context=doc))

    # Assert
    assert semantics.aggregate.kind == "estimate"
    assert semantics.aggregate.value_columns == ("value_adj",)


# --------------------------------------------------------------------------- 8. two nested repeating IDs
def test_inferSemantics_twoNestedRepeatingIds_askedWhichNamesThePatient() -> None:
    # Arrange: two hearing screens per child; a carer brings two children, so both IDs repeat
    frame = pl.DataFrame(
        {
            "child_ref": [f"B{i // 2:03d}" for i in range(120)],
            "carer_ref": [f"M{i // 4:03d}" for i in range(120)],
            "screen_pass": [i % 3 > 0 for i in range(120)],
        }
    )

    # Act
    semantics = infer_semantics(frame)

    # Assert: a line may be a screen of a child or of a carer's family; only the user can say
    asked = semantics.question("patient_key")
    assert asked is not None and asked.options[:2] == ("child_ref", "carer_ref")
    assert asked.option_counts == (60, 30)


# --------------------------------------------------------------------------- nit: coded target column
_GOAL_DOC = "bp_goal_mmhg  Blood pressure goal: 1 = below, 2 = in range, 3 = above\n"


@pytest.mark.parametrize(
    ("column", "values", "doc"),
    [
        pytest.param("a1c_target_pct", [1, 2, 3], None, id="target-codes-by-name"),
        pytest.param("sbp_cat_mmhg", [1, 2, 3, 4], None, id="category-by-name"),
        pytest.param("bp_goal_mmhg", [1, 2, 3], _GOAL_DOC, id="dictionary-codes"),
    ],
)
def test_inferSemantics_codedTargetColumn_noMeasurementUnit(column: str, values: list[int], doc: str | None) -> None:
    # Arrange
    frame = pl.DataFrame(
        {"mrn": [f"D{i:03d}" for i in range(120)], column: [values[i % len(values)] for i in range(120)]}
    )

    # Act
    semantics = infer_semantics(frame, UploadFacts(doc_context=doc))

    # Assert
    assert semantics.columns[column].unit is None


# --------------------------------------------------------------------------- round 5: prose is asked
@pytest.mark.parametrize(
    ("note", "top_coded"),
    [
        pytest.param("Ages of 90 or over were coded as 99 where the exact age is unknown.", True, id="top-code"),
        pytest.param("Smoking was asked only under age 18; for those, 99 indicates not asked.", False, id="other"),
    ],
)
def test_inferSemantics_proseCodeOnRealValues_askedNeverNulled(note: str, top_coded: bool) -> None:
    # Arrange: 30 real patients aged 90+ recorded as 99, and real ages up to 89
    frame = pl.DataFrame(
        {"mrn": [f"A{i:03d}" for i in range(300)], "age": [99 if i < 30 else 20 + i % 70 for i in range(300)]}
    )
    doc = f"mrn  Medical record number\nage  Age in years\n\n{note}\n"

    # Act
    semantics = infer_semantics(frame, UploadFacts(doc_context=doc))

    # Assert
    asked = _asked(semantics, "missing_code:age:99")
    assert semantics.columns["age"].missing_codes == ()
    assert asked is not None
    assert any("top-coded" in option for option in asked.options) is top_coded


# --------------------------------------------------------------------------- 4. weekly tables
def test_inferSemantics_weeklyCountTable_weeksAreStrata() -> None:
    # Arrange: state x report week (1-52) x disease, with cases
    frame = pl.DataFrame(
        [
            {"State": s, "Report Week": w, "Disease": d, "Cases": (i + w + j) % 9}
            for i, s in enumerate(("Northvale", "Eastmoor", "Westfold"))
            for w in range(1, 53)
            for j, d in enumerate(("Scabies", "Impetigo"))
        ]
    )

    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert semantics.aggregate is not None
    assert "Report Week" in semantics.aggregate.strata
    assert semantics.question("aggregate_unsure") is not None


def test_inferSemantics_stackedCopiesOfATable_asked() -> None:
    # Arrange
    one = pl.DataFrame(
        [
            {"State": s, "Year": y, "Cases": i + y % 5}
            for i, s in enumerate(("Northvale", "Eastmoor"))
            for y in range(2010, 2020)
        ]
    )

    # Act
    semantics = infer_semantics(pl.concat([one, one, one]))

    # Assert
    assert semantics.aggregate is None
    assert semantics.question("aggregate_unsure") is not None


# --------------------------------------------------------------------------- 5. a band is not a rate
def test_inferSemantics_ageGroupPerTenYears_stratumNotRate() -> None:
    # Arrange
    frame = pl.DataFrame(
        [
            {"region": r, "age_group_per_10y": g, "deaths": 3 + i + g}
            for i, r in enumerate(REGIONS)
            for g in (1, 2, 3, 4)
        ]
    )

    # Act
    aggregate = infer_semantics(frame).aggregate

    # Assert
    assert aggregate is not None and aggregate.kind == "frequency"
    assert "age_group_per_10y" in aggregate.strata


@pytest.mark.parametrize("total", ["Overall", "All regions"])
def test_inferSemantics_overallAndAllStates_sameTreatment(total: str) -> None:
    # Act: label and sums agree for both
    semantics = infer_semantics(_scabies(total))

    # Assert
    assert any(total in t.values for t in semantics.aggregate.total_levels)
    assert _asked(semantics, f"total_row:region:{total}") is None

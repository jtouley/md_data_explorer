# SPDX-License-Identifier: MIT
"""Third check of ADR017 inference, from synthetic probes: a shared dictionary note never nulls a
real value, totals are found by label or by their sums, records are not tables of groups, and
rates are values whatever their type.

Synthetic tables only, in new domains: care-home residents, hand clinics, mumps notifications,
a patient-experience survey, suicide-rate tables, newborn screening, diabetes clinics.
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
            "phq9": [99 if i % 25 == 0 else i % 28 for i in range(400)],
        }
    )


# --------------------------------------------------------------------------- 1. shared notes
def test_inferSemantics_sharedNoteCodeInsideRealValues_askedNeverNulled() -> None:
    # Arrange
    facts = UploadFacts(doc_context="resident_id  Resident identifier\n\nCoded fields: 99 = unknown.\n")

    # Act
    semantics = infer_semantics(_residents(), facts)

    # Assert: 99 is a real age and a real dose count, so it is asked; on phq9 (0-27) it is applied
    assert semantics.columns["age"].missing_codes == ()
    assert semantics.columns["n_meds"].missing_codes == ()
    assert _asked(semantics, "missing_code:age:99") is not None
    assert _asked(semantics, "missing_code:n_meds:99") is not None
    assert [m.value for m in semantics.columns["phq9"].missing_codes] == [99]
    assert _asked(semantics, "missing_code:phq9:99") is None


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


def test_parseDictionary_sentenceNamingColumn_thatColumnsCodeOtherwiseShared() -> None:
    # Arrange
    text = "In grip_kg, a value of 999 indicates the reading was not taken.\nMissing values were coded as -9.\n"

    # Act
    found = parse_dictionary(text, ["grip_kg", "pinch_kg"])

    # Assert
    assert found.entry("grip_kg").missing == {"999": "the reading was not taken"}
    assert found.shared_missing == {"-9": "Missing values"}


# --------------------------------------------------------------------------- 3. totals
def _mumps(total: str) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in range(2012, 2018):
        for sex in ("Female", "Male"):
            counts = {r: 3 + (year + i) % 5 for i, r in enumerate(REGIONS)}
            rows += [{"region": r, "year": year, "sex": sex, "cases": n} for r, n in counts.items()]
            rows.append({"region": total, "year": year, "sex": sex, "cases": sum(counts.values())})
    return pl.DataFrame(rows)


@pytest.mark.parametrize(
    "total", ["Grand Total", "Statewide", "All ages", "Both sexes", "National", "Subtotal", "Total, all regions"]
)
def test_inferSemantics_labelledTotalRow_recordedNeverSummed(total: str) -> None:
    # Arrange
    frame = _mumps(total)
    groups = frame.filter(pl.col("region") != total)

    # Act
    semantics = infer_semantics(frame)
    counted = count_cases(frame, semantics.aggregate, {"year": [2013], "sex": ["Male"]})

    # Assert: the answer is the total line's count, never the groups plus their total
    assert any(total in t.values for t in semantics.aggregate.total_levels)
    assert counted.total == groups.filter((pl.col("year") == 2013) & (pl.col("sex") == "Male"))["cases"].sum()


def test_inferSemantics_unlabelledTotalBySums_recordedAndAsked() -> None:
    # Arrange: "Health board 9" is the sum of the four regions on every year and sex
    frame = _mumps("Health board 9")

    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert ("region", ("Health board 9",)) in [(t.column, t.values) for t in semantics.aggregate.total_levels]
    assert _asked(semantics, "total_row:region:Health board 9") is not None


def test_inferSemantics_frequencyTableWithoutTotals_noTotalFound() -> None:
    # Arrange
    frame = _mumps("Islands").filter(pl.col("region") != "Islands")

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
            for i, s in enumerate(("Ohio", "Utah", "Iowa", "Maine", "Idaho"))
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
            for i, s in enumerate(("Ohio", "Utah", "Iowa", "Maine", "Idaho"))
            for y in range(2010, 2020)
        ]
    )
    doc = "value_adj  Age-adjusted suicide death rate per 100,000\n"

    # Act
    semantics = infer_semantics(frame, UploadFacts(doc_context=doc))

    # Assert
    assert semantics.aggregate.kind == "estimate"
    assert semantics.aggregate.value_columns == ("value_adj",)


# --------------------------------------------------------------------------- 8. twins
def test_inferSemantics_babyIdRepeatsAndMotherRepeats_asked() -> None:
    # Arrange: two screens per baby; the baby ID repeats, so a line may be a screen or a mother
    frame = pl.DataFrame(
        {
            "infant_id": [f"B{i // 2:03d}" for i in range(120)],
            "mat_id": [f"M{i // 4:03d}" for i in range(120)],
            "oae_pass": [i % 3 > 0 for i in range(120)],
        }
    )

    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert semantics.question("person_role") is not None


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

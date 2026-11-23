# SPDX-License-Identifier: MIT
"""Codebook-table dictionaries (SAS/SPSS/NCHS style), non-response and range codes, unit columns,
survey samples and sequence-number keys (ADR017). Synthetic survey files only: a household
wellbeing survey and a sleep-clinic intake form.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from clinical_analytics.contracts.declarations import Declarations, QuestionAnswer
from clinical_analytics.contracts.dictionary import parse_dictionary
from clinical_analytics.contracts.inference import UploadFacts, apply_semantics, infer_semantics
from clinical_analytics.contracts.readings import apply_declared, declared_reads
from clinical_analytics.contracts.semantics import describe, with_declarations
from clinical_analytics.core.identifier_rules import row_key_note

N = 240

CODEBOOK = """Wellbeing survey 2031 codebook. Estimates need the interview sample weights (WTINT).

Variable Name: RESPSEQ
SAS Label: Respondent sequence number
English Text: Respondent sequence number.

Variable Name: AGEYR
SAS Label: Age in years at interview
Code or Value\tValue Description\tCount\tCumulative
18 to 79\tRange of Values\t200\t200
80\t80 years of age and over\t20\t220
777\tRefused\t10\t230
999\tDon't know\t10\t240
.\tMissing\t0\t240

Variable Name: SLEEPOK
SAS Label: Sleeps well most nights
| Code or Value | Value Description | Count | Cumulative |
|---|---|---|---|
| 1 | Yes | 120 | 120 |
| 2 | No | 100 | 220 |
| 7 | Refused | 10 | 230 |
| 9 | Don't know | 10 | 240 |
"""


def _survey() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "RESPSEQ": list(range(50001, 50001 + N)),
            "AGEYR": [80 if i < 20 else 777 if i < 30 else 999 if i < 40 else 18 + i % 60 for i in range(N)],
            "SLEEPOK": [7 if i < 10 else 9 if i < 20 else 1 + i % 2 for i in range(N)],
            "sleep_dur": [3 + i % 9 for i in range(N)],
            "sleep_dur_unit": ["months" if i % 2 else "years" for i in range(N)],
        }
    )


def _inferred() -> Any:
    return infer_semantics(_survey(), UploadFacts(doc_context=CODEBOOK))


def test_parseDictionary_codebookTables_structuredEntries() -> None:
    # Act
    found = parse_dictionary(CODEBOOK, list(_survey().columns))

    # Assert: tab and pipe tables read per variable, with the range row and the special codes
    age, sleep = found.entry("AGEYR"), found.entry("SLEEPOK")
    assert age.stated_range == (18.0, 79.0)
    assert age.missing == {"777": "Refused", "999": "Don't know"}
    assert age.censored == {"80": "80 years of age and over"}
    assert sleep.labels == {"1": "Yes", "2": "No", "7": "Refused", "9": "Don't know"}
    assert sleep.nonresponse == {"7": "Refused", "9": "Don't know"}
    assert found.mentions_weights is True


def test_inferSemantics_measureNonresponse_missingAndTopCodeAsked() -> None:
    # Act
    semantics = _inferred()
    age = semantics.columns["AGEYR"]

    # Assert: 777/999 are missing on a measure; 80 is a real value whose meaning is asked
    assert sorted(m.value for m in age.missing_codes) == [777, 999]
    assert [(c.value, c.meaning) for c in age.censored_codes] == [(80, "80 years of age and over")]
    assert semantics.question("censored_value").id == "censored_value:AGEYR:80"


def test_readings_codedItemNonresponse_keptButLeftOutOfShares() -> None:
    # Arrange
    semantics = _inferred()
    frame = _survey()

    # Act
    read = apply_declared(frame, declared_reads(Declarations(), None, semantics))

    # Assert: 7 and 9 stay in the stored data, every reader leaves them out of shares and means
    assert [c.value for c in semantics.columns["SLEEPOK"].nonresponse_codes] == [7, 9]
    assert frame.get_column("SLEEPOK").is_in([7, 9]).sum() == 20
    assert read.get_column("SLEEPOK").null_count() == 20
    assert any("left out of percentages and averages" in line for line in describe(semantics, max_lines=40))


def test_readings_topCodeAnsweredLeaveOut_leftOutOnlyThen() -> None:
    # Arrange
    semantics = _inferred()
    answer = QuestionAnswer(question_id="censored_value:AGEYR:80", answer="no, leave those lines out")
    declarations = Declarations(answers=(answer,))

    # Act
    settled = with_declarations(semantics, declarations)
    read = apply_declared(_survey(), declared_reads(declarations, None, settled))

    # Assert
    assert settled.question("censored_value").answered is True
    assert read.filter(pl.col("AGEYR") == 80).height == 0


def test_inferSemantics_unitColumnNotStated_askedNotApplied() -> None:
    # Act
    semantics = _inferred()

    # Assert: the codebook never says sleep_dur_unit is the unit of sleep_dur (ADR017 principle A)
    assert semantics.columns["sleep_dur"].unit_column is None
    asked = next(q for q in semantics.open_questions if q.kind == "unit_pair")
    assert (asked.column, asked.value) == ("sleep_dur", "years")


@pytest.mark.parametrize(
    ("doc", "columns", "survey"),
    [
        pytest.param(CODEBOOK, {}, True, id="codebook-mentions-weights"),
        pytest.param(None, {"WTINT2YR": [1.5] * N}, True, id="weight-column"),
        pytest.param(None, {"weight_kg": [70.0] * N}, False, id="body-weight-not-a-survey"),
    ],
)
def test_inferSemantics_surveySample_noted(doc: str | None, columns: dict[str, list[float]], survey: bool) -> None:
    # Arrange
    frame = _survey().with_columns(**{k: pl.Series(v) for k, v in columns.items()})

    # Act
    semantics = infer_semantics(frame, UploadFacts(doc_context=doc))

    # Assert
    assert semantics.survey_sample is survey
    assert (
        "These are counts of the people in this file, not weighted national estimates."
        in describe(semantics, max_lines=40)
    ) is survey


def test_inferSemantics_respondentSequenceNumber_isThePersonKeyNoteAgrees() -> None:
    # Act
    semantics = _inferred()

    # Assert: the dictionary names the respondent; the unit note says nothing contrary
    assert semantics.patient_key_column == "RESPSEQ"
    assert row_key_note(semantics) == ""


# --------------------------------------------------------------------------- round 6: only explicit labels apply
TUMOUR_CODEBOOK = """Variable Name: PRIMARY
SAS Label: Primary tumour site
Code or Value\tValue Description\tCount\tCumulative
1\tLung\t120\t120
2\tColorectal\t100\t220
3\tBreast\t71\t291
9\tUnknown primary (CUP)\t109\t400
"""


def test_inferSemantics_categoryNamedUnknown_keptAndAskedNeverNulled() -> None:
    # Arrange
    frame = pl.DataFrame(
        {"case_id": [f"T{i:03d}" for i in range(400)], "PRIMARY": [1] * 120 + [2] * 100 + [3] * 71 + [9] * 109}
    )

    # Act
    semantics = infer_semantics(frame, UploadFacts(doc_context=TUMOUR_CODEBOOK))
    read = apply_declared(frame, declared_reads(Declarations(), None, semantics))

    # Assert: "Unknown primary (CUP)" is a category until the user says otherwise
    assert semantics.columns["PRIMARY"].nonresponse_codes == ()
    assert semantics.columns["PRIMARY"].labels["9"] == "Unknown primary (CUP)"
    assert read.filter(pl.col("PRIMARY") == 9).height == 109
    asked = next(q for q in semantics.open_questions if q.id == "missing_code:PRIMARY:9")
    assert "Unknown primary (CUP)" in asked.question


@pytest.mark.parametrize(
    ("rows", "range_", "censored"),
    [
        pytest.param("1\tLess than 9th grade\n2\tHigh school\n3\tCollege\n", None, {}, id="category-less-than"),
        pytest.param("12 to 16\tRange of normal values\n17\tAbove normal\n", None, {}, id="clinical-reference-range"),
        pytest.param(
            "1 to 120\tRange of Values\n666\tLess than 1 year\n",
            (1.0, 120.0),
            {"666": "Less than 1 year"},
            id="measure-edge",
        ),
    ],
)
def test_parseDictionary_codebookRangesAndRangeCodes_onlyTheCodebooksOwn(
    rows: str, range_: tuple[float, float] | None, censored: dict[str, str]
) -> None:
    # Arrange
    text = f"Variable Name: ITEM\nSAS Label: An item\nCode or Value\tValue Description\n{rows}"

    # Act
    entry = parse_dictionary(text, ["ITEM"]).entry("ITEM")

    # Assert
    assert entry.stated_range == range_
    assert entry.censored == censored


UNIT_DOC = "dur  Duration of symptoms\ndur_unit  Unit of measure for dur (days, months or years)\n"


@pytest.mark.parametrize("blank", [0, 2], ids=["all-units", "two-blank"])
def test_readings_statedUnitColumn_everyLineInOneUnitBlanksAsked(blank: int) -> None:
    # Arrange: 6 months, 2 years, 18 months, 1 year; ``blank`` lines without a unit
    units = ["months", "years", "months", "years"] + [""] * blank
    frame = pl.DataFrame(
        {"pid": [f"P{i}" for i in range(4 + blank)], "dur": [6, 2, 18, 1] + [5] * blank, "dur_unit": units}
    )

    # Act
    semantics = infer_semantics(frame, UploadFacts(doc_context=UNIT_DOC))
    read = apply_declared(apply_semantics(frame, semantics), declared_reads(Declarations(), None, semantics))

    # Assert: read in years by every reader; the stored data is not rewritten
    assert semantics.columns["dur"].unit.value == "years"
    assert read.get_column("dur").to_list()[:4] == pytest.approx([0.5, 2.0, 1.5, 1.0])
    assert apply_semantics(frame, semantics).get_column("dur").to_list() == frame.get_column("dur").to_list()
    assert any("read in years" in line for line in describe(semantics, max_lines=40))
    missing = next((q for q in semantics.open_questions if q.kind == "unit_missing"), None)
    assert (missing is not None) is (blank > 0)
    assert read.get_column("dur").null_count() == blank


def test_inferSemantics_ratePerColumn_neverAUnitPair() -> None:
    # Arrange: 20 mg per day, 100 mg per week
    frame = pl.DataFrame(
        {"pid": [f"P{i}" for i in range(40)], "dose_mg": [20, 100] * 20, "dose_per": ["day", "week"] * 20}
    )

    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert semantics.columns["dose_mg"].unit_column is None
    assert not [q for q in semantics.open_questions if q.kind in ("unit_pair", "unit_missing")]

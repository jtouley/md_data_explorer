# SPDX-License-Identifier: MIT
"""Codebook-table dictionaries (two layouts: "Variable Name / Code or Value | Value Description"
and "Label / SAS Variable Name / Value  Value Label"), non-response and range codes, unit columns,
survey samples and sequence-number keys (ADR017). Synthetic survey files only, invented: a
shift-work wellbeing survey, a limb-injury register and a sleep-clinic intake form.
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

CODEBOOK = """Shift-work wellbeing survey 2031 codebook. Weights must be applied to produce population estimates.

Variable Name: RESPNO
SAS Label: Respondent sequence number
English Text: Respondent sequence number.

Variable Name: SHIFTHR
SAS Label: Hours of night shift last week
Code or Value\tValue Description\tCount\tCumulative
0 to 64\tRange of Values\t200\t200
65\t65 hours or more\t20\t220
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
            "RESPNO": list(range(50001, 50001 + N)),
            "SHIFTHR": [65 if i < 20 else 777 if i < 30 else 999 if i < 40 else i % 60 for i in range(N)],
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
    hours, sleep = found.entry("SHIFTHR"), found.entry("SLEEPOK")
    assert hours.stated_range == (0.0, 64.0)
    assert hours.missing == {"777": "Refused", "999": "Don't know"}
    assert hours.censored == {"65": "65 hours or more"}
    assert sleep.labels == {"1": "Yes", "2": "No", "7": "Refused", "9": "Don't know"}
    assert sleep.nonresponse == {"7": "Refused", "9": "Don't know"}
    assert found.mentions_weights is True


def test_inferSemantics_measureNonresponse_missingAndTopCodeAsked() -> None:
    # Act
    semantics = _inferred()
    hours = semantics.columns["SHIFTHR"]

    # Assert: 777/999 are missing on a measure; 65 is a real value whose meaning is asked
    assert sorted(m.value for m in hours.missing_codes) == [777, 999]
    assert [(c.value, c.meaning) for c in hours.censored_codes] == [(65, "65 hours or more")]
    assert semantics.question("censored_value").id == "censored_value:SHIFTHR:65"


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
    answer = QuestionAnswer(question_id="censored_value:SHIFTHR:65", answer="no, leave those lines out")
    declarations = Declarations(answers=(answer,))

    # Act
    settled = with_declarations(semantics, declarations)
    read = apply_declared(_survey(), declared_reads(declarations, None, settled))

    # Assert
    assert settled.question("censored_value").answered is True
    assert read.filter(pl.col("SHIFTHR") == 65).height == 0


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
        pytest.param(CODEBOOK, {}, True, id="codebook-says-apply-weights"),
        pytest.param(CODEBOOK.replace("Weights must be applied", "No weights are needed"), {}, False, id="no-weights"),
        pytest.param(None, {"sampling_weight": [1.5] * N}, True, id="weight-column"),
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
        "These are counts of the people in this file, not weighted population estimates."
        in describe(semantics, max_lines=40)
    ) is survey


def test_inferSemantics_respondentSequenceNumber_isThePersonKeyNoteAgrees() -> None:
    # Act
    semantics = _inferred()

    # Assert: the dictionary names the respondent; the unit note says nothing contrary
    assert semantics.patient_key_column == "RESPNO"
    assert row_key_note(semantics) == ""


# --------------------------------------------------------------------------- round 6: only explicit labels apply
LIMB_CODEBOOK = """Variable Name: LIMB
SAS Label: Injured limb
Code or Value\tValue Description\tCount\tCumulative
1\tArm\t120\t120
2\tLeg\t100\t220
3\tTrunk\t71\t291
9\tUnknown limb (not traced)\t109\t400
"""


def test_inferSemantics_categoryNamedUnknown_keptAndAskedNeverNulled() -> None:
    # Arrange
    frame = pl.DataFrame(
        {"case_id": [f"T{i:03d}" for i in range(400)], "LIMB": [1] * 120 + [2] * 100 + [3] * 71 + [9] * 109}
    )

    # Act
    semantics = infer_semantics(frame, UploadFacts(doc_context=LIMB_CODEBOOK))
    read = apply_declared(frame, declared_reads(Declarations(), None, semantics))

    # Assert: "Unknown limb (not traced)" is a category until the user says otherwise
    assert semantics.columns["LIMB"].nonresponse_codes == ()
    assert semantics.columns["LIMB"].labels["9"] == "Unknown limb (not traced)"
    assert read.filter(pl.col("LIMB") == 9).height == 109
    asked = next(q for q in semantics.open_questions if q.id == "missing_code:LIMB:9")
    assert "Unknown limb (not traced)" in asked.question


@pytest.mark.parametrize(
    ("rows", "range_", "censored"),
    [
        pytest.param("1\tLess than one shift\n2\tOne shift\n3\tTwo or more\n", None, {}, id="category-less-than"),
        pytest.param("12 to 16\tRange of normal values\n17\tAbove normal\n", None, {}, id="clinical-reference-range"),
        pytest.param(
            "1 to 64\tRange of Values\n0\tLess than 1 hour\n",
            (1.0, 64.0),
            {"0": "Less than 1 hour"},
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


UNIT_DOC = "dur  Duration of symptoms\ndur_unit  Gives the unit of dur (days, months or years)\n"


@pytest.mark.parametrize("blank", [0, 2], ids=["all-units", "two-blank"])
def test_readings_statedUnitColumn_everyLineInOneUnitBlanksAsked(blank: int) -> None:
    # Arrange: 6 months, 2 years, 18 months, 1 year; ``blank`` lines without a unit
    units = ["months", "years", "months", "years"] + [""] * blank
    frame = pl.DataFrame(
        {"client_ref": [f"P{i}" for i in range(4 + blank)], "dur": [6, 2, 18, 1] + [5] * blank, "dur_unit": units}
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
        {"client_ref": [f"P{i}" for i in range(40)], "dose_mg": [20, 100] * 20, "dose_per": ["day", "week"] * 20}
    )

    # Act
    semantics = infer_semantics(frame)

    # Assert
    assert semantics.columns["dose_mg"].unit_column is None
    assert not [q for q in semantics.open_questions if q.kind in ("unit_pair", "unit_missing")]


# --------------------------------------------------------------------------- a second codebook layout
SECOND_LAYOUT = """Label: Nights woken per week
Section Name: Sleep
Question Number: 4
SAS Variable Name: WOKEN
Question: On how many nights last week did you wake before your alarm?
Value  Value Label  Frequency  Percentage
0 - 7  Number of nights [0-7]  180  75.0
77  Don’t know/Not sure  20  8.3
99  Refused  40  16.7
BLANK  Not asked or Missing  0  0.0
"""


def test_parseDictionary_secondCodebookLayout_labelBeforeVariableAndOwnRangeRow() -> None:
    # Arrange
    frame = pl.DataFrame(
        {"RESPNO": list(range(N)), "WOKEN": [77 if i < 20 else 99 if i < 60 else i % 8 for i in range(N)]}
    )

    # Act
    entry = parse_dictionary(SECOND_LAYOUT, list(frame.columns)).entry("WOKEN")
    semantics = infer_semantics(frame, UploadFacts(doc_context=SECOND_LAYOUT))

    # Assert: the same structured entry as the first layout gives
    assert entry is not None and entry.description == "Nights woken per week"
    assert entry.stated_range == (0.0, 7.0)
    assert entry.missing == {"77": "Don’t know/Not sure", "99": "Refused"}
    assert sorted(m.value for m in semantics.columns["WOKEN"].missing_codes) == [77, 99]

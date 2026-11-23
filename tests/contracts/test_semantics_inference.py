# SPDX-License-Identifier: MIT
"""The contract's semantics section (ADR017): one inference pass decides what a file means.

Synthetic tables only, in domains the gates and earlier work do not use (hip-fracture rehab,
glaucoma clinics, cataract surgery, newborn hearing screening, sleep studies, psoriasis
biologics, cardiac rehab, bone-age films).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.contracts.dates import DateAmbiguity, DateChoice, choose_number_format, choose_text_format
from clinical_analytics.contracts.dictionary import parse_dictionary, parse_pairs
from clinical_analytics.contracts.inference import add_id_columns, apply_semantics, infer_semantics
from clinical_analytics.contracts.semantics import describe

PATIENTS = 40


def _rehab(key: str = "patient_ref", seq: str = "episode_no") -> pl.DataFrame:
    """Hip-fracture rehab: 40 patients with 1-4 episodes each, numbered 1..k within patient."""
    rows: list[dict[str, Any]] = []
    for p in range(PATIENTS):
        for k in range(1 + p % 4):
            rows.append({key: f"HR{p:03d}", seq: k + 1, "barthel": 40 + (p * 7 + k * 3) % 60, "falls": k % 2})
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- 1. two sources of grain
def test_inferSemantics_episodeRows_appPatientIdNeverTheKey(make_semantics: Callable[..., Any]) -> None:
    # Arrange
    df = _rehab()

    # Act
    semantics = make_semantics(df)

    # Assert: the file's own key stays identified; the app's copy is an app column, never the key
    assert semantics.patient_key_column == "patient_ref"
    assert semantics.grain.kind == "episode"
    assert semantics.one_row_per_patient is False
    assert [(a.column, a.kind, a.made_from) for a in semantics.app_columns] == [
        ("patient_id", "copy", ("patient_ref",))
    ]
    assert semantics.column("patient_id").role == "app_added"


# --------------------------------------------------------------------------- 2. keys by name, words and structure
@pytest.mark.parametrize(
    "key",
    ["subj", "patient_ref", "case_id", "study_no", "PtRef", "member_id"],
)
def test_inferSemantics_patientKeyNames_found(key: str) -> None:
    # Arrange
    df = _rehab(key=key)

    # Act
    semantics = infer_semantics(df)

    # Assert
    assert semantics.patient_key_column == key


def test_inferSemantics_dictionaryStatesKey_keyFromDictionary() -> None:
    # Arrange: a glaucoma clinic file whose key has no ID-like name at all
    df = _rehab(key="eye_owner").rename({"episode_no": "visit_seq"})
    doc = "GLAUCOMA IOP CLINIC\neye_owner is the patient identifier.\n"

    # Act
    semantics = infer_semantics(df, _facts(doc))

    # Assert
    assert semantics.patient_key_column == "eye_owner"
    assert semantics.patient_key.source == "dictionary"


def test_inferSemantics_unnamedKeyWithSequence_keyFromStructure() -> None:
    # Arrange: "film" repeats while "read_order" numbers 1..k within each (bone-age films)
    df = _rehab(key="film", seq="read_order")

    # Act
    semantics = infer_semantics(df)

    # Assert
    assert semantics.patient_key_column == "film"
    assert semantics.patient_key.source == "structure"
    assert [s.column for s in semantics.sequence_columns] == ["read_order"]


def test_inferSemantics_newbornScreening_babyIsPatientMotherGroups() -> None:
    # Arrange: newborn hearing screening; twins share a mother
    mothers = [f"M{i // 2 if i < 20 else i:03d}" for i in range(60)]
    df = pl.DataFrame(
        {"infant_id": [f"B{i:03d}" for i in range(60)], "mat_id": mothers, "oae_pass": [i % 3 > 0 for i in range(60)]}
    )

    # Act
    semantics = infer_semantics(df)

    # Assert
    assert semantics.patient_key_column == "infant_id"
    assert [(g.column, g.role) for g in semantics.grouping_columns] == [("mat_id", "mother")]
    assert semantics.question("person_role").question == "Is each line one baby or one mother?"


@pytest.mark.parametrize(
    ("ids", "expected_key", "expected_row_key"),
    [
        pytest.param([f"C{i:03d}" for i in range(60)], "case_id", ("case_id",), id="all_unique_named"),
        pytest.param([f"C{i % 20:03d}" for i in range(60)], "case_id", (), id="repeating_named"),
        pytest.param([None if i % 3 else f"C{i:03d}" for i in range(60)], None, (), id="mostly_missing"),
    ],
)
def test_inferSemantics_cataractCases_uniqueOrRepeating(
    ids: list[str | None], expected_key: str | None, expected_row_key: tuple[str, ...]
) -> None:
    # Arrange: cataract surgery registry
    df = pl.DataFrame({"case_id": ids, "va_logmar": [0.1 * (i % 9) for i in range(60)]})

    # Act
    semantics = infer_semantics(df)

    # Assert
    assert semantics.patient_key_column == expected_key
    assert tuple(semantics.row_key.columns) == expected_row_key or semantics.row_key.kind == "composite"


def test_inferSemantics_siteIdRepeats_groupingNotPatient() -> None:
    # Arrange: sleep studies at 3 sites; no column names a patient
    df = pl.DataFrame({"site_id": [f"S{i % 3}" for i in range(60)], "ahi": [5.5 + i for i in range(60)]})

    # Act
    semantics = infer_semantics(df)

    # Assert
    assert semantics.patient_key is None
    assert [(g.column, g.role) for g in semantics.grouping_columns] == [("site_id", "site")]
    assert semantics.question("rows_are_patients") is None


def test_inferSemantics_uniqueUnnamedRowKey_asksIsEachLinePatient() -> None:
    # Arrange: psoriasis biologic log with only a unique record code
    df = pl.DataFrame({"record_code": [f"R{i:04d}" for i in range(60)], "pasi": [float(i % 30) for i in range(60)]})

    # Act
    semantics = infer_semantics(df)

    # Assert
    assert semantics.patient_key is None
    assert semantics.question("rows_are_patients").question == "Is each line one patient?"


# --------------------------------------------------------------------------- 3. sequences
@pytest.mark.parametrize("seq", ["visit_no", "episode_no", "cycle", "scan_no"])
def test_inferSemantics_sequenceColumn_orderWithinPatientNeverAnId(seq: str) -> None:
    # Arrange
    df = _rehab(seq=seq)

    # Act
    semantics = infer_semantics(df)

    # Assert
    assert [s.column for s in semantics.sequence_columns] == [seq]
    assert semantics.column(seq).role == "sequence"
    assert seq not in [g.column for g in semantics.grouping_columns]


# --------------------------------------------------------------------------- 4. dates
@pytest.mark.parametrize(
    ("values", "fmt", "first"),
    [
        pytest.param(["2021-03-04", "2021-03-05"], "%Y-%m-%d", date(2021, 3, 4), id="iso"),
        pytest.param(
            ["2021-03-04T10:05:00", "2021-03-05T11:00:00"],
            "%Y-%m-%dT%H:%M:%S%.f",
            datetime(2021, 3, 4, 10, 5),
            id="iso_datetime",
        ),
        pytest.param(["13/04/2021", "03/04/2021"], "%d/%m/%Y", date(2021, 4, 13), id="day_first"),
        pytest.param(["04/13/2021", "04/03/2021"], "%m/%d/%Y", date(2021, 4, 13), id="month_first"),
        pytest.param(["04.03.2021", "13.03.2021"], "%d.%m.%Y", date(2021, 3, 4), id="dotted"),
        pytest.param(["13/04/21", "03/04/21"], "%d/%m/%y", date(2021, 4, 13), id="two_digit_year"),
        pytest.param(["20210304", "20211231"], "%Y%m%d", date(2021, 3, 4), id="yyyymmdd_text"),
        pytest.param(["March 4, 2021", "Dec 31, 2020"], "%B %d, %Y", date(2021, 3, 4), id="month_name_first"),
        pytest.param(["7 March 2019", "7 Mar 2019"], "%d %B %Y", date(2019, 3, 7), id="day_month_name"),
        pytest.param(
            ["03/14/2021 14:05", "03/04/2021 09:00"], "%m/%d/%Y %H:%M", datetime(2021, 3, 14, 14, 5), id="us_datetime"
        ),
    ],
)
def test_chooseTextFormat_oneFormatPerColumn_readsEveryValue(values: list[str], fmt: str, first: date) -> None:
    # Arrange
    series = pl.Series("seen_date", values)

    # Act
    choice = choose_text_format(series)

    # Assert
    assert isinstance(choice, DateChoice)
    assert choice.format == fmt
    target = pl.Datetime("us") if choice.has_time else pl.Date
    assert series.str.strptime(target, choice.format)[0] == first


def test_chooseTextFormat_bothOrdersDisagree_ambiguousNotGuessed() -> None:
    # Act
    found = choose_text_format(pl.Series("ref_date", ["03/04/21", "05/06/21"]))

    # Assert
    assert isinstance(found, DateAmbiguity)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(["2021-03-04", "04/03/2021"], id="two_formats"),
        pytest.param([None, None], id="all_empty"),
        pytest.param(["soon", "later"], id="words"),
        pytest.param(["01/01/0021", "02/02/0022"], id="implausible_years"),
    ],
)
def test_chooseTextFormat_noSingleFormat_none(values: list[str | None]) -> None:
    # Act / Assert
    assert choose_text_format(pl.Series("d", values, dtype=pl.String)) is None


@pytest.mark.parametrize(
    ("series", "from_excel", "expected"),
    [
        pytest.param(
            pl.Series("lkw_date", [20210304, 20191231]), False, date(2021, 3, 4), id="yyyymmdd_int_date_named"
        ),
        pytest.param(pl.Series("surg_date", [44259, 44260]), True, date(2021, 3, 4), id="excel_serial"),
        pytest.param(pl.Series("mrn", [20210304, 20191231]), False, None, id="id_digits_never_dates"),
        pytest.param(pl.Series("surg_date", [44259, 44260]), False, None, id="serial_outside_workbook"),
    ],
)
def test_chooseNumberFormat_onlyWithEvidence(series: pl.Series, from_excel: bool, expected: date | None) -> None:
    # Arrange
    df = series.to_frame()

    # Act
    choice = choose_number_format(series, from_excel=from_excel)

    # Assert
    if expected is None:
        assert choice is None
        return
    assert choice is not None
    semantics = infer_semantics(df, _facts(None, from_excel=from_excel))
    applied = apply_semantics(df, semantics)
    assert applied.get_column(series.name).cast(pl.Date)[0] == expected


def test_inferSemantics_ambiguousDateColumn_openQuestionColumnStaysText() -> None:
    # Arrange: cardiac rehab sessions dated 03/04/21-style, nothing above 12
    df = pl.DataFrame(
        {"pt_id": [f"P{i % 10}" for i in range(30)], "sess_date": ["03/04/21", "05/06/21", "07/08/21"] * 10}
    )

    # Act
    semantics = infer_semantics(df)
    applied = apply_semantics(df, semantics)

    # Assert
    question = semantics.question("date_order")
    assert question is not None and question.column == "sess_date"
    assert "3 April 2021 or March 4, 2021" in question.question
    assert applied.schema["sess_date"] == pl.String


def test_inferSemantics_dictionaryStatesFormat_settlesOrderAndRecordsExample() -> None:
    # Arrange
    df = pl.DataFrame(
        {"pt_id": [f"P{i % 10}" for i in range(30)], "sess_date": ["03/04/21", "05/06/21", "07/08/21"] * 10}
    )
    doc = "sess_date   Session date, DD/MM/YY\n"

    # Act
    semantics = infer_semantics(df, _facts(doc))
    applied = apply_semantics(df, semantics)

    # Assert
    fmt = semantics.column("sess_date").date
    assert (fmt.format, fmt.source, fmt.example_in, fmt.example_out) == (
        "%d/%m/%y",
        "dictionary",
        "03/04/21",
        "3 April 2021",
    )
    assert applied.get_column("sess_date")[0] == date(2021, 4, 3)
    assert semantics.question("date_order") is None


# --------------------------------------------------------------------------- 5. missing-value codes
@pytest.mark.parametrize(
    ("column", "values", "expected"),
    [
        pytest.param("end_weight", [70.5, 80.0, 999.0, 999.0, 65.0], [999], id="undocumented_999_far_above"),
        pytest.param("age", [40, 98, 99, 99, 60], [], id="real_99_year_olds"),
        pytest.param("anc_count", [3, 4, 999, 5, 6], [], id="single_999_is_an_entry_error"),
        pytest.param("pain_score", [3, -1, -1, 0, 7], [-1], id="minus_one_in_nonnegative"),
        pytest.param("pain_score", [None, None, None, None, None], [], id="all_empty"),
    ],
)
def test_inferSemantics_sentinels_onlyFarOutsideAndRepeated(
    column: str, values: list[Any], expected: list[Any]
) -> None:
    # Arrange
    df = pl.DataFrame({column: values}, schema={column: pl.Float64 if isinstance(values[0], float) else pl.Int64})

    # Act
    codes = [m.value for m in infer_semantics(df).column(column).missing_codes]

    # Assert
    assert codes == expected


def test_applySemantics_documentedCodes_storedMissingAndSafeTwice() -> None:
    # Arrange: psoriasis biologics; the dictionary documents 99 and "unknown"
    df = pl.DataFrame({"pasi": [3, 99, 12, 99], "smoker": ["yes", "unknown", "no", "yes"]})
    doc = (
        "pasi     PASI score (0-72). 99 = not assessed at this visit\nsmoker   Smoking: yes, no, unknown = not asked\n"
    )

    # Act
    semantics = infer_semantics(df, _facts(doc))
    once = apply_semantics(df, semantics)
    twice = apply_semantics(once, semantics)

    # Assert
    expected = pl.DataFrame({"pasi": [3, None, 12, None], "smoker": ["yes", None, "no", "yes"]})
    assert_frame_equal(once, expected)
    assert_frame_equal(twice, once)
    assert semantics.column("pasi").stated_range.maximum == 72


def test_inferSemantics_unknownOtherLabel_isACategoryNotMissing() -> None:
    # Arrange
    df = pl.DataFrame({"src": [1, 2, 5, 5]})
    doc = "src  Source: 1 = urinary, 2 = chest, 5 = unknown / other\n"

    # Act
    column = infer_semantics(df, _facts(doc)).column("src")

    # Assert
    assert column.missing_codes == ()
    assert column.labels == {"1": "urinary", "2": "chest", "5": "unknown / other"}


# --------------------------------------------------------------------------- 6. labels
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "Regimen:\n1 = ETN (etanercept, 50 mg/wk)\n2 = ADA (adalimumab) / then switch\n3 = score = 0 at start",
            {"1": "ETN (etanercept, 50 mg/wk)", "2": "ADA (adalimumab) / then switch", "3": "score = 0 at start"},
            id="nested_punctuation_and_equals",
        ),
        pytest.param("Smoker (1=yes, 2=no)", {"1": "yes", "2": "no"}, id="inside_parentheses"),
        pytest.param("1: Yes 2: No", {"1": "Yes", "2": "No"}, id="colon_space_separated"),
        pytest.param("Cycle number (1 = first cycle)", {"1": "first cycle"}, id="one_pair_no_stray_paren"),
    ],
)
def test_parsePairs_labelRunsToNextCode(text: str, expected: dict[str, str]) -> None:
    # Act / Assert
    assert dict(parse_pairs(text)) == expected


def test_parseDictionary_singleNote_notACodebook() -> None:
    # Act
    entry = parse_dictionary("cyc   Cycle number (1 = first cycle)\n", ["cyc"]).entry("cyc")

    # Assert
    assert entry is not None and entry.labels == {}


def test_inferSemantics_everyCodeDecodes_contractEnumMatches(make_contract: Callable[..., Any]) -> None:
    # Arrange
    df = pl.DataFrame({"pt_id": [f"P{i}" for i in range(9)], "das": [1, 2, 3] * 3})
    doc = "das  Disease activity:\n  1 = remission (DAS28 < 2.6)\n  2 = low/moderate\n  3 = high, flare\n"

    # Act
    semantics = infer_semantics(df, _facts(doc))
    contract = make_contract(df, semantics=semantics, id_column="pt_id")

    # Assert
    enum = next(p.enum for p in contract.schema_[0].properties if p.physicalName == "das")
    assert {str(e.value): e.label for e in enum} == {
        "1": "remission (DAS28 < 2.6)",
        "2": "low/moderate",
        "3": "high, flare",
    }


# --------------------------------------------------------------------------- 7-8. units and paired measures
@pytest.mark.parametrize(
    ("column", "unit"),
    [
        ("weight_kg", "kg"),
        ("mtx_mg_wk", "mg/week"),
        ("peep_cmH2O", "cmH2O"),
        ("crp_ng_ml", "ng/mL"),
        ("bmd_g_cm2", "g/cm²"),
        ("fev1_pct", "%"),
    ],
)
def test_inferSemantics_unitFromName(column: str, unit: str) -> None:
    # Act
    found = infer_semantics(pl.DataFrame({column: [1.5, 2.5, 3.5]})).column(column).unit

    # Assert
    assert found is not None and (found.value, found.source) == (unit, "structure")


def test_inferSemantics_pairedMeasures_oneMeasureTwoTimesWithDirection() -> None:
    # Arrange
    df = pl.DataFrame(
        {"weight_preop_kg": [90.5, 101.0], "weight_12m_kg": [80.0, 85.5], "dlqi_baseline": [12, 9], "dlqi_m6": [4, 3]}
    )
    doc = "dlqi_baseline  Dermatology Life Quality Index at baseline (0-30); lower is better\n"

    # Act
    semantics = infer_semantics(df, _facts(doc))

    # Assert
    pre, post = semantics.column("weight_preop_kg").timepoint, semantics.column("weight_12m_kg").timepoint
    assert (pre.measure, pre.order, post.order) == ("weight_kg", 0, 1)
    assert semantics.column("dlqi_m6").timepoint.paired_with == ("dlqi_baseline",)
    assert semantics.column("dlqi_baseline").direction == "lower_is_better"


# --------------------------------------------------------------------------- 9. no false alarm
@pytest.mark.parametrize(("frame", "duplicate_rule"), [("visits", False), ("patients", True)])
def test_buildContract_visitLevelRows_noDuplicateIdCheck(
    make_contract: Callable[..., Any], frame: str, duplicate_rule: bool
) -> None:
    # Arrange
    df = _rehab() if frame == "visits" else _rehab().unique("patient_ref", keep="first", maintain_order=True)
    semantics = infer_semantics(df)

    # Act
    contract = make_contract(df, semantics=semantics, id_column="patient_ref")

    # Assert: repeated patient IDs on a visit-level file are expected, never a failed check
    rules = [q.id for p in contract.schema_[0].properties if p.physicalName == "patient_ref" for q in p.quality or []]
    assert ("patient_ref_no_duplicates" in rules) is duplicate_rule


# --------------------------------------------------------------------------- determinism, read API, words
def test_inferSemantics_sameInputTwice_identical() -> None:
    # Arrange
    df, doc = _rehab(), "One row per rehab episode.\n"

    # Act / Assert
    assert infer_semantics(df, _facts(doc)) == infer_semantics(df, _facts(doc))


def test_semanticsFor_contractRoundTrip_declarationOverrides(tmp_path: Any, make_contract: Callable[..., Any]) -> None:
    # Arrange
    from clinical_analytics.contracts.declarations import FileDeclarationStore, GrainDeclaration
    from clinical_analytics.contracts.semantics import semantics_for
    from clinical_analytics.contracts.store import write_contract

    df = pl.DataFrame({"record_code": [f"R{i:04d}" for i in range(60)], "pasi": [float(i % 30) for i in range(60)]})
    frame, _meta, semantics = add_id_columns(df, infer_semantics(df))
    write_contract(tmp_path, "u1", "v1", make_contract(frame, semantics=semantics, id_column=None))
    store = FileDeclarationStore(tmp_path)
    store.update(
        "u1",
        lambda d: d.with_grain(GrainDeclaration(grain="patient", key=None, rows=60, patients=60, dataset_version="v1")),
    )

    # Act
    read = semantics_for(tmp_path, "u1", "v1")

    # Assert
    assert (read.patient_key_column, read.patient_key.source) == ("record_code", "declared")
    assert read.one_row_per_patient and read.question("rows_are_patients").answered


def test_semanticsFor_noContract_raisesSemanticsMissing(tmp_path: Any) -> None:
    # Arrange
    from clinical_analytics.contracts.semantics import SemanticsMissingError, semantics_for

    # Act / Assert
    with pytest.raises(SemanticsMissingError):
        semantics_for(tmp_path, "u1", "v1")


def test_describe_clinicalWords_noJargon() -> None:
    # Arrange
    df = _rehab().with_columns(
        pl.Series("ep_date", ["13/04/21", "03/04/21"] * 50 + ["03/04/21"] * (_rehab().height - 100))
    )
    doc = "One row per rehab episode.\nbarthel  Barthel index (0-100). 999 = not assessed\n"

    # Act
    lines = describe(infer_semantics(df, _facts(doc)))
    text = " ".join(lines).lower()

    # Assert
    assert lines[0].startswith("Each line is one rehab episode; 40 patients")
    assert "Dates like 03/04/21 in ep_date read as 3 April 2021." in lines
    for jargon in ("null", "grain", "distinct", "select ", "sql"):
        assert jargon not in text


def _facts(doc: str | None, **extra: Any) -> Any:
    from clinical_analytics.contracts.inference import UploadFacts

    return UploadFacts(doc_context=doc, **extra)

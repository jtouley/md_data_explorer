# SPDX-License-Identifier: MIT
"""Review round for ADR017: inference that would change a number is asked, never guessed.

Synthetic tables only, in new domains: fibromyalgia, lupus nephritis biopsies, melanoma
treatment, heart surgery, trauma (GCS), pancreatitis length of stay, chest-pain troponin,
vitamin D clinic, obstetric anaesthesia, cystic fibrosis clinics, notifiable-disease counts.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.contracts.dates import DateAmbiguity, choose_text_format
from clinical_analytics.contracts.dictionary import parse_dictionary, parse_pairs
from clinical_analytics.contracts.inference import UploadFacts, apply_semantics, infer_semantics, person_key_role
from clinical_analytics.contracts.semantics import describe

N = 200


def _facts(doc: str | None = None, **extra: Any) -> UploadFacts:
    return UploadFacts(doc_context=doc, **extra)


# --------------------------------------------------------------------------- 1. code typed to the column
@pytest.mark.parametrize(
    ("dtype", "values", "code"),
    [
        pytest.param(pl.Int64, [10, 45, 999, 60, 10], 999, id="int"),
        pytest.param(pl.Float64, [10.0, 45.5, 999.0, 60.0, 10.0], 999.0, id="float"),
        pytest.param(pl.String, ["10", "45", "999", "60", "10"], "999", id="text"),
    ],
)
def test_applySemantics_documentedIntegerCodeAnyColumnType_storedMissingNoCrash(
    dtype: pl.DataType, values: list[Any], code: Any
) -> None:
    # Arrange: fibromyalgia clinic, FIQ total stated 0-100 with 999 = not done
    df = pl.DataFrame({"fiq_score": values}, schema={"fiq_score": dtype})
    doc = "fiq_score  FIQ total (0-100). 999 = not done\n"

    # Act
    semantics = infer_semantics(df, _facts(doc))
    applied = apply_semantics(df, semantics)

    # Assert
    assert [m.value for m in semantics.column("fiq_score").missing_codes] == [code]
    assert applied.get_column("fiq_score").null_count() == 1
    assert applied.schema["fiq_score"] == dtype


# --------------------------------------------------------------------------- 2. labelled categories stay
@pytest.mark.parametrize(
    ("text", "labels", "missing"),
    [
        pytest.param(
            "biopsy  Renal biopsy: 0 = not done, 1 = done", {"0": "not done", "1": "done"}, {}, id="not_done_category"
        ),
        pytest.param(
            "chemo  Chemotherapy: 0 = accepted, 1 = refused",
            {"0": "accepted", "1": "refused"},
            {},
            id="refused_category",
        ),
        pytest.param(
            "tested  HLA-B27: 0 = negative, 1 = positive, 2 = not tested",
            {"0": "negative", "1": "positive", "2": "not tested"},
            {},
            id="not_tested_category",
        ),
        pytest.param(
            "vital  Vital status: 1 = alive, 2 = dead, 9 = not recorded",
            {"1": "alive", "2": "dead"},
            {"9": "not recorded"},
            id="not_recorded_missing",
        ),
        pytest.param(
            "flag  Melanoma flag: 0 = absent, 9 = missing",
            {"0": "absent", "9": "missing"},
            {},
            id="one_category_left_keeps_all",
        ),
    ],
)
def test_parseDictionary_labelledCodes_categoryUnlessNotRecordedAndTwoLeft(
    text: str, labels: dict[str, str], missing: dict[str, str]
) -> None:
    # Arrange
    column = text.split()[0]

    # Act
    entry = parse_dictionary(text + "\n", [column]).entry(column)

    # Assert
    assert entry is not None
    assert (entry.labels, entry.missing) == (labels, missing)


def test_applySemantics_biopsyNotDone_neverNulled() -> None:
    # Arrange: lupus nephritis; 0 = not done is an answer, so the biopsy rate keeps its zeros
    df = pl.DataFrame({"pt_id": [f"L{i}" for i in range(N)], "biopsy": [i % 3 == 0 for i in range(N)]}).with_columns(
        pl.col("biopsy").cast(pl.Int64)
    )

    # Act
    semantics = infer_semantics(df, _facts("biopsy  Renal biopsy: 0 = not done, 1 = done\n"))

    # Assert
    assert_frame_equal(apply_semantics(df, semantics), df)


# --------------------------------------------------------------------------- 3-4, 10. keys
@pytest.mark.parametrize(
    ("frame", "key"),
    [
        pytest.param(
            pl.DataFrame(
                {
                    "patient_no": [f"P{i}" for i in range(N)],
                    "patient_status": [("alive", "dead", "lost")[i % 3] for i in range(N)],
                }
            ),
            "patient_no",
            id="status_attribute",
        ),
        pytest.param(
            pl.DataFrame({"mrn": [f"M{i}" for i in range(N)], "patient_zip": [f"0{i % 7}100" for i in range(N)]}),
            "mrn",
            id="zip_attribute",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "age": [30 + i % 50 for i in range(N)],
                    "num_meds": [i % 15 for i in range(N)],
                    "sbp": [110.5 + i % 40 for i in range(N)],
                }
            ),
            None,
            id="count_never_key",
        ),
    ],
)
def test_inferSemantics_attributeOrCountNamedForPatient_neverTheKey(frame: pl.DataFrame, key: str | None) -> None:
    # Act / Assert
    assert infer_semantics(frame).patient_key_column == key


@pytest.mark.parametrize(
    ("column", "role"),
    [("patient_status", None), ("patient_zip", None), ("num_meds", None), ("number_of_children", None),
     ("study_no", "patient"), ("pt_id", "patient"), ("subj", "patient"), ("infant_id", "infant")],
)  # fmt: skip
def test_personKeyRole_names_idWordOrWholeNameOnly(column: str, role: str | None) -> None:
    # Act / Assert
    assert person_key_role(column) == role


def test_inferSemantics_surgeonSentenceMentionsPatient_surgeonNeverKey() -> None:
    # Arrange: heart surgery; the sentence about the surgeon mentions the patient
    df = pl.DataFrame(
        {
            "mrn": [f"M{i // 2}" for i in range(N)],
            "surgeon_id": [f"S{i % 6}" for i in range(N)],
            "bypass_min": [60 + i % 90 for i in range(N)],
        }
    )
    doc = "surgeon_id  Identifier of the surgeon who operated on the patient\nmrn  Medical record number\n"

    # Act
    semantics = infer_semantics(df, _facts(doc))

    # Assert
    assert semantics.patient_key_column == "mrn"
    assert ("surgeon_id", "surgeon") in [(g.column, g.role) for g in semantics.grouping_columns]


# --------------------------------------------------------------------------- 5-6. dates stay text when unsure
def test_chooseTextFormat_twoDigitBirthYearInTheFuture_centuryQuestion() -> None:
    # Act
    found = choose_text_format(pl.Series("dob", ["03/15/45", "04/20/85", "12/30/60"]))

    # Assert
    assert isinstance(found, DateAmbiguity) and found.kind == "century"


def test_inferSemantics_oneValueSettlesOrder_questionColumnStaysText() -> None:
    # Arrange: 99 dates that read either way and one 13/01/2021
    df = pl.DataFrame({"op_date": ["03/04/2021"] * 99 + ["13/01/2021"]})

    # Act
    semantics = infer_semantics(df)
    applied = apply_semantics(df, semantics)

    # Assert
    assert semantics.question("date_order") is not None
    assert applied.schema["op_date"] == pl.String


def test_inferSemantics_dashedAmbiguousDates_questionExampleInColumnStyle() -> None:
    # Act
    question = infer_semantics(pl.DataFrame({"op_date": ["03-04-2021", "05-06-2021"]})).question("date_order")

    # Assert
    assert question is not None and "03-04-2021" in question.question


# --------------------------------------------------------------------------- 7. undocumented codes are asked
@pytest.mark.parametrize(
    ("column", "values", "asked"),
    [
        pytest.param("los_days", [3, 5, 99, 99, 99, 4] * 20, False, id="length_of_stay_99s_real"),
        pytest.param("pain_change", [2, -1, -1, -1, 0, 3] * 20, False, id="change_minus_one_real"),
        pytest.param("trop_ng_l", [4.0, 12.0, 999.0, 999.0, 999.0, 8.0] * 20, True, id="troponin_999_asked"),
    ],
)
def test_inferSemantics_undocumentedCode_askedOnlyOutsideCountsAndChanges(
    column: str, values: list[Any], asked: bool
) -> None:
    # Arrange
    df = pl.DataFrame({column: values})

    # Act
    semantics = infer_semantics(df)

    # Assert: never nulled without the dictionary or the user
    assert (semantics.question("missing_code") is not None) is asked
    assert_frame_equal(apply_semantics(df, semantics), df)


def test_withDeclarations_confirmedCode_declaredAndAppliedInLayerView(make_semantic_layer: Callable[..., Any]) -> None:
    # Arrange: chest-pain unit; the user confirms 999 means not recorded
    from clinical_analytics.contracts.declarations import Declarations, MissingCodeDeclaration
    from clinical_analytics.contracts.readings import declared_reads, reads_config
    from clinical_analytics.contracts.semantics import with_declarations
    from clinical_analytics.core.semantic import DECLARED_READS_KEY

    df = pl.DataFrame({"mrn": [f"M{i}" for i in range(120)], "trop_ng_l": [4.0, 12.0, 999.0, 8.0] * 30})
    semantics = infer_semantics(df).model_copy(update={"dataset_version": "v1"})
    answer = MissingCodeDeclaration(column="trop_ng_l", value=999.0, missing=True, dataset_version="v1")

    # Act
    declared = with_declarations(semantics, Declarations(missing_codes=(answer,)))
    reads = declared_reads(Declarations(missing_codes=(answer,)), "v1")
    layer = make_semantic_layer(data=df, config_overrides={DECLARED_READS_KEY: reads_config(reads)})
    view = layer.get_base_view().to_polars()

    # Assert
    assert reads.missing == {"trop_ng_l": (999.0,)}
    assert declared.question("missing_code").answered
    assert view.get_column("trop_ng_l").null_count() == 30


# --------------------------------------------------------------------------- 8. units
@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("vitamin_d", [30.5, 12.0]),
        ("sbp_min", [90, 85]),
        ("iop_l", [14, 18]),
        ("birth_yr", [1957, 1988]),
        ("time_in", [3, 4]),
    ],
)
def test_inferSemantics_ambiguousNameSuffix_noUnit(column: str, values: list[Any]) -> None:
    # Act / Assert
    assert infer_semantics(pl.DataFrame({column: values})).column(column).unit is None


@pytest.mark.parametrize(
    ("line", "unit"),
    [
        ("anc  Absolute neutrophil count, x10^9/L", "x10^9/L"),
        ("map  Mean arterial pressure, mmHg", "mmHg"),
        ("onset  Age at onset (years)", "years"),
        ("wait  Minutes from referral to review", "minutes"),
        ("smoker  Current smoker: Yes/No", None),
    ],
)
def test_parseDictionary_statedUnit_pickedUpNeverFromLabels(line: str, unit: str | None) -> None:
    # Arrange
    column = line.split()[0]

    # Act / Assert
    assert parse_dictionary(line + "\n", [column]).entry(column).unit == unit


# --------------------------------------------------------------------------- 9. labels
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "ECOG 0 = fully active; 1 = restricted; 2 = ambulatory",
            {"0": "fully active", "1": "restricted", "2": "ambulatory"},
        ),
        ("Smoking 0=never 1=former 2=current", {"0": "never", "1": "former", "2": "current"}),
        ("severity 5 = severe", {}),
    ],
)
def test_parsePairs_firstCodeAfterAWord_kept(text: str, expected: dict[str, str]) -> None:
    # Act / Assert
    assert dict(parse_pairs(text)) == expected


def test_parseDictionary_markdownTable_entriesWithCodes() -> None:
    # Arrange: cystic fibrosis clinic dictionary as a Markdown table
    text = (
        "| column | description |\n|---|---|\n"
        "| genotype | CFTR class: 1 = F508del homozygous<br>2 = F508del heterozygous<br>3 = other |\n"
        "| fev1_pp | FEV1 percent predicted (0-150) |\n"
    )

    # Act
    found = parse_dictionary(text, ["genotype", "fev1_pp"])

    # Assert
    assert found.entry("genotype").labels == {"1": "F508del homozygous", "2": "F508del heterozygous", "3": "other"}
    assert found.entry("fev1_pp").stated_range == (0.0, 150.0)


# --------------------------------------------------------------------------- 14. integer dates by the dictionary
@pytest.mark.parametrize(
    ("column", "line"),
    [("seen_on_int", "seen_on_int  First seen, YYYYMMDD"), ("first_contact", "first_contact  Date of first contact")],
)
def test_inferSemantics_integerDateStatedInDictionary_read(column: str, line: str) -> None:
    # Arrange
    df = pl.DataFrame({column: [20210304, 20191231, 20200115]})

    # Act
    applied = apply_semantics(df, infer_semantics(df, _facts(line + "\n")))

    # Assert
    assert applied.get_column(column).to_list()[0] == date(2021, 3, 4)


# --------------------------------------------------------------------------- 15. roles
def test_inferSemantics_staffAndPlaces_groupingNotMeasures() -> None:
    # Arrange: obstetric anaesthesia
    rows = 120
    df = pl.DataFrame(
        {
            "mrn": [f"M{i // 2}" for i in range(rows)],
            "anaesthetist": [f"A{i % 5}" for i in range(rows)],
            "ward": [("L1", "L2", "L3")[i % 3] for i in range(rows)],
            "room": [f"R{i % 8}" for i in range(rows)],
            "operator_code": [f"O{i % 4}" for i in range(rows)],
            "reviewer_initials": [("AB", "CD", "EF")[i % 3] for i in range(rows)],
            "region": [("N", "S", "E", "W")[i % 4] for i in range(rows)],
        }
    )

    # Act
    semantics = infer_semantics(df)

    # Assert
    roles = {g.column: g.role for g in semantics.grouping_columns}
    assert set(roles) >= {"anaesthetist", "ward", "room", "operator_code", "reviewer_initials", "region"}
    assert semantics.patient_key_column == "mrn"


def test_inferSemantics_timeSinceIndexAndEpisodeId_sequenceAndEpisodeKey() -> None:
    # Arrange: pancreatitis admissions within an MRN, measured on weeks since start
    rows = [
        {"mrn": f"M{p}", "admission_id": f"A{p}-{a}", "weeks_since_start": w, "crp": 50.5 + w}
        for p in range(30)
        for a in range(2)
        for w in (a * 10, a * 10 + 2)
    ]
    df = pl.DataFrame(rows)

    # Act
    semantics = infer_semantics(df)

    # Assert
    assert semantics.patient_key_column == "mrn"
    assert [(s.column, s.kind) for s in semantics.sequence_columns] == [("weeks_since_start", "time_since_index")]
    assert [(e.column, e.within) for e in semantics.episode_keys] == [("admission_id", "mrn")]


def test_inferSemantics_uniqueBabyUniqueMother_noBabyOrMotherQuestion() -> None:
    # Act
    df = pl.DataFrame(
        {"infant_id": [f"B{i}" for i in range(60)], "mat_id": [f"M{i}" for i in range(60)], "apgar": [9] * 60}
    )
    semantics = infer_semantics(df)

    # Assert
    assert semantics.patient_key_column == "infant_id"
    assert semantics.question("person_role") is None


# --------------------------------------------------------------------------- 16. summary tables
def test_inferSemantics_summaryTable_aggregateGrainSentenceNoPatientQuestion() -> None:
    # Arrange: notifiable-disease counts by region, year and category
    regions, years, cats = ("North", "South", "East"), (2019, 2020, 2021), ("child", "adult")
    df = pl.DataFrame(
        [
            {"region": r, "year": y, "category": c, "cases": (i * 7) % 40}
            for i, (r, y, c) in enumerate((r, y, c) for r in regions for y in years for c in cats)
        ]
    )

    # Act
    semantics = infer_semantics(df)

    # Assert
    assert semantics.grain.kind == "stratum"
    assert semantics.grain.sentence == "Each line seems to count cases for one region, year and category (to confirm)."
    assert semantics.question("aggregate_unsure") is not None
    assert semantics.case_count_column == "cases"
    assert semantics.question("rows_are_patients") is None
    assert describe(semantics)[0].startswith("Each line seems to count cases for one region, year and category")


@pytest.mark.parametrize(
    "statement",
    [
        "One row per patient visit to the emergency department.",
        "Each row = one ward round documented by the team on call.",
    ],
)
def test_parseDictionary_longGrainStatement_nounIsWholeNeverCut(statement: str) -> None:
    # Act
    noun = parse_dictionary(statement + "\n", ["x"]).grain_noun

    # Assert: ends at a word that names a line, never mid-phrase
    assert noun in ("patient visit", "ward round")


# --------------------------------------------------------------------------- 17. pairing
def test_inferSemantics_scoresAtTwoTimesAndEvents_paired() -> None:
    # Arrange: trauma GCS on admission and at 24 h, weight at admission and discharge
    df = pl.DataFrame(
        {"gcs_admit": [8, 12], "gcs_24h": [10, 14], "wt_admit_kg": [70.5, 80.0], "wt_dc_kg": [68.0, 79.5]}
    )

    # Act
    semantics = infer_semantics(df)

    # Assert
    assert (semantics.column("gcs_admit").timepoint.order, semantics.column("gcs_24h").timepoint.order) == (0, 1)
    assert semantics.column("wt_dc_kg").timepoint.paired_with == ("wt_admit_kg",)
    assert semantics.column("gcs_24h").direction is None


# --------------------------------------------------------------------------- 18. case-differing codes
def test_levelVariantResults_codesListedAsDistinct_notHeld(make_contract: Callable[..., Any]) -> None:
    # Arrange: lesion site codes differ only in case and the dictionary lists both
    from clinical_analytics.contracts.quality import level_variant_results

    df = pl.DataFrame({"pt_id": [f"P{i}" for i in range(40)], "site": ["L", "l", "R"] * 13 + ["L"]})
    semantics = infer_semantics(df, _facts("site  Lesion site: L = left limb, l = lower lip, R = right limb\n"))

    # Act
    results = level_variant_results(df.lazy(), make_contract(df, semantics=semantics, id_column="pt_id"))

    # Assert
    assert results == []


# --------------------------------------------------------------------------- 11. no silent fallbacks
def test_contractNamesPatient_unreadableContract_fails500(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    from fastapi import HTTPException

    from clinical_analytics.api.routes import aggregate
    from clinical_analytics.contracts import service
    from clinical_analytics.contracts.errors import ContractDamagedError

    def damaged(*_args: Any, **_kwargs: Any) -> Any:
        raise ContractDamagedError("damaged")

    monkeypatch.setattr(service, "version_semantics", damaged)

    class Storage:
        upload_dir = __import__("pathlib").Path("/nonexistent/uploads")

    # Act / Assert: a patient file is never read as estimates because its contract was unreadable
    with pytest.raises(HTTPException) as raised:
        aggregate._version_semantics(Storage(), "ds1", "v1", {}, pl.DataFrame({"a": [1]}))  # type: ignore[arg-type]
    assert raised.value.status_code == 500


# --------------------------------------------------------------------------- 13. headless applies semantics
def test_headlessLayer_dateColumn_readWithItsFormat() -> None:
    # Arrange
    from clinical_analytics.eval.headless import layer_for, register_frame

    register_frame(
        "review_dates",
        pl.DataFrame(
            {
                "pt_id": [f"P{i}" for i in range(40)],
                "seen_date": ["13/04/2021", "25/12/2020", "14/02/2021", "03/04/2021"] * 10,
            }
        ),
    )

    # Act
    view = layer_for("review_dates").get_base_view().to_polars()

    # Assert
    assert view.schema["seen_date"] == pl.Date


def test_stagingIdNames_oneListWithInference() -> None:
    # Act / Assert
    from clinical_analytics.contracts.inference import KEY_COLUMN_NAMES
    from clinical_analytics.storage.staging import ID_COLUMN_NAMES

    assert ID_COLUMN_NAMES is KEY_COLUMN_NAMES

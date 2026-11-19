"""Fellow-scenario oracles: each trap is answered the way the data dictionary defines it.

Hand-written synthetic extracts, small enough to check every expected value by hand.
"""

from __future__ import annotations

import math

import pytest

from clinical_analytics.eval.corpus.run import load_oracles, load_questions, load_scenario
from clinical_analytics.eval.corpus.scenario import split
from clinical_analytics.eval.corpus.synthetic_hospital import EXTRACTS, generate

CDIFF = "patient_id,encounter_id,test_date,test_year,age_years,sex,cdiff_result,outpatient_abx_90d,ppi_use"
HEPB = "patient_id,visit_id,visit_date,age_years,sex,birth_region,hbv_screen_done,hbsag_result,new_hbv_diagnosis"
PNEUMONIA = (
    "patient_id,encounter_id,admit_date,admit_year,age_years,sex,pathogen,"
    "pneumococcal_vaccine,flu_vaccine_this_season,icu_admission,died_in_hospital"
)
MEASLES = "patient_id,visit_id,visit_date,risk_group,age_years,mmr_doses_documented,measles_igg,measles_igg_unit"


class TestCdiff:
    def test_casesByYear_duplicateRowAndRepeatPositive_patientCountedOncePerYear(self, write_extract, fellow_oracles):
        # Arrange: P1 has the same encounter twice and a second positive in 2021; P2 is under 65.
        path = write_extract(
            CDIFF,
            [
                "P1,E1,2021-02-01,2021,70,female,positive,yes,no",
                "P1,E1,2021-02-01,2021,70,female,positive,yes,no",
                "P1,E2,2021-08-01,2021,70,female,positive,no,no",
                "P1,E3,2022-01-05,2022,71,female,positive,no,no",
                "P2,E4,2021-03-01,2021,50,male,positive,yes,no",
                "P3,E5,2021-04-01,2021,80,male,negative,no,no",
            ],
        )

        # Act
        answer = fellow_oracles.answers(path, "fellow_cdiff")["fellow_cdiff-cases-by-year-01"]

        # Assert
        assert answer.value == {"2021": 1, "2022": 1}

    def test_positivityByYear_duplicateTestCountedOnce(self, write_extract, fellow_oracles):
        # Arrange: three distinct tests in 65+, one positive (written twice).
        path = write_extract(
            CDIFF,
            [
                "P1,E1,2021-02-01,2021,70,female,positive,yes,no",
                "P1,E1,2021-02-01,2021,70,female,positive,yes,no",
                "P2,E2,2021-03-01,2021,75,male,negative,no,no",
                "P3,E3,2021-04-01,2021,80,male,negative,no,no",
            ],
        )

        # Act
        answer = fellow_oracles.answers(path, "fellow_cdiff")["fellow_cdiff-positivity-by-year-02"]

        # Assert
        assert answer.value == {"2021": pytest.approx(1 / 3)}
        assert answer.cell_patients[("2021", "events")] == 1

    def test_oddsRatio_dedupedTestsIn65Plus_matchesHandComputedWoolf(self, write_extract, fellow_oracles):
        # Arrange: a=2, b=1, c=1, d=2 over distinct 65+ tests; the duplicate and the 40-year-old are out.
        rows = [
            "P1,E1,2021-01-01,2021,70,female,positive,yes,no",
            "P1,E1,2021-01-01,2021,70,female,positive,yes,no",
            "P2,E2,2021-01-02,2021,70,female,positive,yes,no",
            "P3,E3,2021-01-03,2021,70,female,negative,yes,no",
            "P4,E4,2021-01-04,2021,70,female,positive,no,no",
            "P5,E5,2021-01-05,2021,70,female,negative,no,no",
            "P6,E6,2021-01-06,2021,70,female,negative,no,no",
            "P7,E7,2021-01-07,2021,40,female,positive,yes,no",
        ]
        path = write_extract(CDIFF, rows)
        se = math.sqrt(1 / 2 + 1 / 1 + 1 / 1 + 1 / 2)

        # Act
        answer = fellow_oracles.answers(path, "fellow_cdiff")["fellow_cdiff-abx-odds-ratio-03"]

        # Assert
        assert answer.value["or"] == pytest.approx(4.0)
        assert answer.value["ci_low"] == pytest.approx(math.exp(math.log(4.0) - 1.959963984540054 * se))

    def test_oddsRatio_zeroCell_undefinedNotCrash(self, write_extract, fellow_oracles):
        # Arrange: no unexposed positive test.
        path = write_extract(
            CDIFF, ["P1,E1,2021-01-01,2021,70,female,positive,yes,no", "P2,E2,2021-01-02,2021,70,male,negative,no,no"]
        )

        # Act
        answer = fellow_oracles.answers(path, "fellow_cdiff")["fellow_cdiff-abx-odds-ratio-03"]

        # Assert
        assert answer.value is None
        assert answer.undefined is not None


class TestHepb:
    def test_counts_visitLevelRows_distinctPatients(self, write_extract, fellow_oracles):
        # Arrange: P1 has three visits (screened and newly diagnosed on one, duplicated); P2 one visit.
        path = write_extract(
            HEPB,
            [
                "P1,V1,2023-01-01,40,female,East Asia,no,,no",
                "P1,V2,2023-05-01,40,female,East Asia,yes,reactive,yes",
                "P1,V2,2023-05-01,40,female,East Asia,yes,reactive,yes",
                "P1,V3,2024-01-01,41,female,East Asia,no,,no",
                "P2,V4,2023-02-01,60,male,United States,no,,no",
            ],
        )

        # Act
        answers = fellow_oracles.answers(path, "fellow_hepb")

        # Assert
        assert answers["fellow_hepb-adults-seen-01"].value == 2
        assert answers["fellow_hepb-ever-screened-02"].value == 1
        assert answers["fellow_hepb-new-diagnoses-03"].value == 1
        assert answers["fellow_hepb-screened-by-birthplace-04"].value == {"East Asia": 1.0, "United States": 0.0}


class TestPneumonia:
    def test_twoPathogensOneAdmission_admissionCountedOnceInTotalOncePerPathogen(self, write_extract, fellow_oracles):
        # Arrange: A1 found influenza and S. aureus (two rows); A2 is a 60-year-old (out).
        path = write_extract(
            PNEUMONIA,
            [
                "P1,A1,2023-01-01,2023,80,female,Influenza,yes,no,no,no",
                "P1,A1,2023-01-01,2023,80,female,S. aureus,yes,no,no,no",
                "P2,A2,2023-02-01,2023,60,male,Influenza,no,no,no,no",
                "P3,A3,2023-03-01,2023,75,male,Influenza,unknown,yes,no,no",
            ],
        )

        # Act
        answers = fellow_oracles.answers(path, "fellow_pneumonia")

        # Assert
        assert answers["fellow_pneumonia-admissions-70-01"].value == 2
        assert answers["fellow_pneumonia-admissions-70-01"].unit == "visits"
        assert answers["fellow_pneumonia-pathogens-70-02"].value == {"Influenza": 2, "S. aureus": 1}
        # "unknown" is not a documented vaccine.
        assert answers["fellow_pneumonia-pneumococcal-vaccine-03"].value == {"Influenza": 0.5, "S. aureus": 1.0}


class TestMeasles:
    @pytest.mark.parametrize(
        ("igg", "unit", "lacks"),
        [
            pytest.param("0.2", "IU/mL", 0.0, id="iu_converted_protective"),
            pytest.param("0.1", "IU/mL", 1.0, id="iu_converted_not_protective"),
            pytest.param("120", "mIU/mL", 0.0, id="boundary_protective"),
            pytest.param("119", "mIU/mL", 1.0, id="just_below"),
            pytest.param("", "", 1.0, id="not_drawn"),
        ],
    )
    def test_noImmunity_oneDoseByIgg_unitAndBoundary(self, write_extract, fellow_oracles, igg, unit, lacks):
        # Arrange: one dose on record, so the IgG decides.
        path = write_extract(MEASLES, [f"P1,M1,2024-01-01,Healthcare worker,30,1,{igg},{unit}"])

        # Act
        answer = fellow_oracles.answers(path, "fellow_measles")["fellow_measles-no-immunity-by-group-04"]

        # Assert
        assert answer.value == {"Healthcare worker": lacks}

    def test_noImmunity_protectiveOnRedrawnCheck_patientImmune(self, write_extract, fellow_oracles):
        # Arrange: the first titer is low, the redrawn one protective.
        path = write_extract(
            MEASLES,
            ["P1,M1,2024-01-01,Pregnant,30,0,40,mIU/mL", "P1,M2,2024-06-01,Pregnant,30,0,0.3,IU/mL"],
        )

        # Act
        answer = fellow_oracles.answers(path, "fellow_measles")["fellow_measles-no-immunity-by-group-04"]

        # Assert
        assert answer.value == {"Pregnant": 0.0}

    def test_medianIgg_mixedUnits_reportedInMiuPerMl(self, write_extract, fellow_oracles):
        # Arrange: 100 mIU/mL, 0.3 IU/mL (300) and 0.5 IU/mL (500), plus a duplicate row.
        path = write_extract(
            MEASLES,
            [
                "P1,M1,2024-01-01,Pregnant,30,0,100,mIU/mL",
                "P2,M2,2024-01-02,Pregnant,31,0,0.3,IU/mL",
                "P3,M3,2024-01-03,Pregnant,32,0,0.5,IU/mL",
                "P3,M3,2024-01-03,Pregnant,32,0,0.5,IU/mL",
            ],
        )

        # Act
        answer = fellow_oracles.answers(path, "fellow_measles")["fellow_measles-median-igg-by-group-03"]

        # Assert
        assert answer.value == {"Pregnant": 300.0}

    def test_unknownIggUnit_onlyUnitQuestionsError(self, write_extract, fellow_oracles):
        # Arrange
        path = write_extract(MEASLES, ["P1,M1,2024-01-01,Pregnant,30,0,7,AU/mL"])

        # Act
        answers = fellow_oracles.answers(path, "fellow_measles")

        # Assert
        assert "AU/mL" in (answers["fellow_measles-no-immunity-by-group-04"].oracle_error or "")
        assert answers["fellow_measles-checked-by-group-01"].oracle_error is None


class TestScenarioFiles:
    @pytest.mark.parametrize("extract", EXTRACTS)
    def test_questionFile_everyAskedQuestionHasAnOracleOnGeneratedData(self, tmp_path, extract):
        # Arrange
        dataset = f"fellow_{extract}"
        path = tmp_path / f"{dataset}.csv"
        path.write_bytes(generate(extract, "table"))
        asked, skipped = split(load_questions(dataset))

        # Act
        answers = load_oracles(dataset).answers(path)

        # Assert
        assert load_scenario(dataset) is not None
        assert skipped and all(s.needs == "aggregate_mode" for s in skipped)
        assert {q["id"] for q in asked} == set(answers)
        assert all(answer.oracle_error is None for answer in answers.values())

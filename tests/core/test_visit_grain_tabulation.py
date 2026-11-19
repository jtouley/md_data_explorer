"""Basic tabulations on visit-level data (one row per visit or test, with a patient ID).

Each rule a fellow's abstract rests on, on small synthetic extracts:
- patients per group when a patient has rows in two groups (counted in each; the total is the
  distinct patients, never the sum; the per-group counts never let 1-10 be worked out);
- roll-ups over a patient's rows ("ever", "always"), never guessed;
- counts and shares by an encounter key (admissions, tests) as distinct from patients and rows.

Expected values are computed here in Polars from the frame, never read from the app.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from clinical_analytics.analysis.variable_types import display_level
from clinical_analytics.core.cohort_constraints import peel_cohort
from clinical_analytics.core.counting_unit import (
    accounted_question,
    encounter_key_candidates,
    pick_encounter_key,
    read_units,
)
from clinical_analytics.core.query_plan import CohortStep, QueryPlan
from clinical_analytics.core.query_service import QueryService
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED, overlap_hidden, present_analysis

# --------------------------------------------------------------------------- synthetic extracts


def _tests_frame(in_both_years: int = 15, null_keys: int = 0) -> pl.DataFrame:
    """C. diff tests: patients P000-P029 test positive in 2021, and the ``in_both_years``
    patients before P030 also test positive in 2022, with P030-P044 positive in 2022 only (their
    only positive row; each also has a negative 2021 test). P045-P059 test negative in 2021.
    One positive 2021 test is written twice (an exact duplicate row). ``null_keys`` positive
    2022 rows have no patient ID."""
    rows: list[dict[str, Any]] = []
    for i in range(30):
        rows.append(
            {"patient_id": f"P{i:03d}", "encounter_id": f"E{i:03d}a", "test_year": "2021", "result": "positive"}
        )
    for i in range(30 - in_both_years, 30):
        rows.append(
            {"patient_id": f"P{i:03d}", "encounter_id": f"E{i:03d}b", "test_year": "2022", "result": "positive"}
        )
    for i in range(30, 45):
        rows.append(
            {"patient_id": f"P{i:03d}", "encounter_id": f"E{i:03d}b", "test_year": "2022", "result": "positive"}
        )
    for i in range(30, 60):
        rows.append(
            {"patient_id": f"P{i:03d}", "encounter_id": f"E{i:03d}a", "test_year": "2021", "result": "negative"}
        )
    rows.append(dict(rows[0]))  # an exact duplicate row
    for j in range(null_keys):
        rows.append({"patient_id": None, "encounter_id": f"N{j:03d}", "test_year": "2022", "result": "positive"})
    return pl.DataFrame(rows, schema={c: pl.String for c in ("patient_id", "encounter_id", "test_year", "result")})


def _screening_frame() -> pl.DataFrame:
    """Two visits for each of 360 patients; region is one value per patient. Screened at the
    first visit when i % 3 == 0, at the second when i % 2 == 0; every 10th patient's second
    visit records nothing. Every region has many patients screened at one visit and not the other."""
    rows: list[dict[str, Any]] = []
    for i in range(360):
        region = ("North", "South", "East")[i % 3]
        first = "yes" if i % 3 == 0 else "no"
        second = None if i % 10 == 0 else ("yes" if i % 2 == 0 else "no")
        rows.append({"patient_id": f"P{i:03d}", "visit_id": f"V{i:03d}a", "region": region, "screened": first})
        rows.append({"patient_id": f"P{i:03d}", "visit_id": f"V{i:03d}b", "region": region, "screened": second})
    return pl.DataFrame(rows, schema={c: pl.String for c in ("patient_id", "visit_id", "region", "screened")})


def _admissions_frame(legionella_patients: int = 3) -> pl.DataFrame:
    """Admissions: 60 patients with one admission each (pathogen by i % 2), plus
    ``legionella_patients`` patients with four Legionella admissions each (12 admissions from 3
    patients), and 15 S. pneumoniae patients readmitted once. Admission A000 has a second
    pathogen written as a second row (Influenza, its only one), and A001-A015 a second one too
    (S. aureus), so 16 admissions have two pathogens."""
    rows: list[dict[str, Any]] = []
    for i in range(60):
        pathogen = ("S. pneumoniae", "None identified")[i % 2]
        rows.append({"patient_id": f"P{i:03d}", "encounter_id": f"A{i:03d}", "pathogen": pathogen})
    for i in range(0, 30, 2):  # 15 S. pneumoniae patients readmitted once with it
        rows.append({"patient_id": f"P{i:03d}", "encounter_id": f"R{i:03d}", "pathogen": "S. pneumoniae"})
    rows.append({"patient_id": "P000", "encounter_id": "A000", "pathogen": "Influenza"})
    for i in range(1, 16):
        rows.append({"patient_id": f"P{i:03d}", "encounter_id": f"A{i:03d}", "pathogen": "S. aureus"})
    for p in range(legionella_patients):
        for k in range(4):
            rows.append({"patient_id": f"L{p:02d}", "encounter_id": f"AL{p}{k}", "pathogen": "Legionella"})
    return pl.DataFrame(rows, schema={c: pl.String for c in ("patient_id", "encounter_id", "pathogen")})


def _ask(service: QueryService, question: str, plan: QueryPlan) -> Any:
    """Ask ``question`` with the parse fixed to ``plan`` (the parser is not under test here) and
    word accounting passed; the counted unit is still read from the question's own words."""
    intent = SimpleNamespace(
        confidence=0.99,
        intent_type=plan.intent,
        primary_variable=plan.metric,
        grouping_variable=plan.group_by,
        predictor_variables=[],
        time_variable=None,
        event_variable=None,
        filters=[],
    )
    service.nl_engine.parse_query = lambda *args, **kwargs: intent  # type: ignore[method-assign]
    service.nl_engine._intent_to_plan = lambda *args: plan  # type: ignore[method-assign]
    service._accounting_hold_reason = lambda *args: None  # type: ignore[method-assign]
    return service.ask(question, dataset_id="visits")


def _table(result: Any) -> dict[str, Any]:
    table = present_analysis(result.analysis)["table"]
    assert table is not None
    label = table["columns"][0]
    value = table["columns"][1]
    return {str(row[label]): row[value] for row in table["rows"]}


def _positive_by_year() -> QueryPlan:
    return QueryPlan(
        intent="COUNT", group_by="test_year", cohort_steps=[CohortStep("eq", "result", "positive")], confidence=0.99
    )


# --------------------------------------------------------------------------- rule 1: patients per group


class TestPatientsPerGroup:
    @pytest.mark.parametrize("null_keys", [pytest.param(0, id="all_keyed"), pytest.param(2, id="null_keys")])
    def test_ask_patientInTwoYears_countedInEachYearTotalDistinct(
        self, make_semantic_layer: Any, null_keys: int
    ) -> None:
        # Arrange
        frame = _tests_frame(null_keys=null_keys)
        service = QueryService(make_semantic_layer(data=frame))
        positive = frame.filter(pl.col("result") == "positive", pl.col("patient_id").is_not_null())
        expected = positive.group_by("test_year").agg(pl.col("patient_id").n_unique().alias("n"))
        total = positive.get_column("patient_id").n_unique()

        # Act
        result = _ask(service, "how many patients ever tested positive each year?", _positive_by_year())

        # Assert: 30 in 2021 and 30 in 2022 (P015-P029 in both, P030-P044 only in 2022), 45 in all
        assert not result.requires_confirmation and result.failure_reason is None
        shown = _table(result)
        assert shown == {display_level("test_year", y): n for y, n in expected.iter_rows()}
        assert (total, sum(shown.values())) == (45, 60)
        assert result.claim.startswith("45 patients across 2 test_year groups.")
        assert "is counted in each, so the test_year counts add up to more than 45" in result.claim

    def test_ask_onlyQualifyingRowInOneYear_countedInThatYearOnly(self, make_semantic_layer: Any) -> None:
        # Arrange: P030-P044 have a negative 2021 test and their only positive test in 2022
        frame = _tests_frame(in_both_years=0)
        service = QueryService(make_semantic_layer(data=frame))

        # Act
        result = _ask(service, "how many patients ever tested positive each year?", _positive_by_year())

        # Assert: no patient is in both years, so the years add up to the total (no overlap note)
        assert _table(result) == {display_level("test_year", "2021"): 30, display_level("test_year", "2022"): 15}
        assert result.claim.startswith("45 patients across 2 test_year groups.")
        assert "counted in each" not in result.claim

    def test_ask_fewPatientsInTwoYears_everyYearHidden(self, make_semantic_layer: Any) -> None:
        # Arrange: 3 patients in both years: the years' sum minus the total would give 3
        service = QueryService(make_semantic_layer(data=_tests_frame(in_both_years=3)))

        # Act
        result = _ask(service, "how many patients ever tested positive each year?", _positive_by_year())

        # Assert
        table = present_analysis(result.analysis)["table"]
        assert table is not None
        assert all(row["count"] == SECONDARY_SUPPRESSED for row in table["rows"])

    def test_ask_unitNotNamed_heldWithAPlainChoice(self, make_semantic_layer: Any) -> None:
        # Arrange
        service = QueryService(make_semantic_layer(data=_tests_frame()))

        # Act: "how many tested positive" says neither patients nor tests
        result = _ask(service, "how many tested positive each year?", _positive_by_year())

        # Assert: asked once, offering both readings, never guessed
        assert result.requires_confirmation is True
        assert "Confirm that rows are visits to count each patient once" in result.failure_reason
        assert "or ask for admissions or tests or visits to count each encounter_id once" in result.failure_reason


class TestOverlapHidden:
    @pytest.mark.parametrize(
        ("counts", "total", "patients", "expected"),
        [
            pytest.param([30, 30], 45, None, set(), id="excess_15_all_shown"),
            pytest.param([30, 30], 50, None, {0, 1}, id="excess_10_all_hidden"),
            pytest.param([30, 30], 49, None, set(), id="excess_11_boundary_shown"),
            pytest.param([5, 30, 30], 50, None, {0}, id="primary_small_only"),
            pytest.param([40, 30], 45, None, {0, 1}, id="complement_5_hidden_then_coverage"),
            pytest.param([40, 30, 30], 45, None, {0}, id="complement_5_hidden_covered"),
            pytest.param([10, 20, 15], 45, None, {0, 1, 2}, id="partition_hidden_group_pinned_by_coverage"),
            pytest.param([34, 30], 45, None, set(), id="complement_11_boundary_shown"),
            pytest.param([30, 30], 45, [30, 3], {0, 1}, id="patients_small_hidden_then_coverage"),
            pytest.param([30, 30, 30], 45, [30, 30, 3], {2}, id="patients_small_hidden_covered"),
            pytest.param([30, None], 45, None, {0, 1}, id="null_count_fails_closed"),
            pytest.param([30, 30], None, None, {0, 1}, id="null_total_fails_closed"),
            pytest.param([], 0, None, set(), id="empty"),
        ],
    )
    def test_overlapHidden_cases_hidesWhatCouldBeWorkedOut(
        self, counts: list[Any], total: Any, patients: list[int] | None, expected: set[int]
    ) -> None:
        # Arrange: inputs as parametrized

        # Act
        hidden = overlap_hidden(counts, total, patients)

        # Assert
        assert hidden == expected


# --------------------------------------------------------------------------- rule 2: roll-ups


def _screened_share() -> QueryPlan:
    return QueryPlan(
        intent="COUNT", proportion_column="screened", proportion_value="yes", group_by="region", confidence=0.99
    )


class TestRollups:
    @pytest.mark.parametrize(
        ("question", "rolled"),
        [
            pytest.param("what share of patients were ever screened, by region?", pl.Expr.any, id="ever_any"),
            pytest.param("what share of patients were screened on every visit, by region?", pl.Expr.all, id="all"),
        ],
    )
    def test_ask_rollupWord_sharePerPatientMatchesPolars(
        self, make_semantic_layer: Any, question: str, rolled: Any
    ) -> None:
        # Arrange: the expected share over recorded rows, one value per patient
        frame = _screening_frame()
        service = QueryService(make_semantic_layer(data=frame))
        per_patient = (
            frame.filter(pl.col("screened").is_not_null())
            .group_by("patient_id")
            .agg(pl.col("region").first(), rolled(pl.col("screened") == "yes").alias("event"))
        )
        expected = per_patient.group_by("region").agg(
            pl.col("event").mean().alias("rate"),
            pl.col("event").sum().alias("events"),
            (~pl.col("event")).sum().alias("non_events"),
        )

        # Act
        result = _ask(service, question, _screened_share())

        # Assert: each region's rate matches, or is hidden because 1-10 patients are behind it
        assert result.failure_reason is None
        shown = _table(result)
        matched = 0
        for region, rate, events, non_events in expected.iter_rows():
            value = shown[display_level("region", region)]
            if value == SECONDARY_SUPPRESSED:
                assert 0 < min(events, non_events) < 11
                continue
            assert value == pytest.approx(rate, abs=5e-5)
            matched += 1
        assert matched >= 1
        assert "when at least one of its rows has it" in result.claim or "all of its rows" in result.claim

    def test_ask_noRollupWord_refusedWithAPlainChoice(self, make_semantic_layer: Any) -> None:
        # Arrange
        service = QueryService(make_semantic_layer(data=_screening_frame()))

        # Act
        result = _ask(service, "what share of patients were screened, by region?", _screened_share())

        # Assert
        assert result.failure_reason.startswith("screened differs between rows of the same patient. Say which patients")

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            pytest.param("what share of patients were ever and always screened?", "asks for 'at least one'", id="both"),
            pytest.param("what share of patients were screened at the first visit?", "first visit", id="ordered"),
        ],
    )
    def test_ask_unsupportedRollup_refused(self, make_semantic_layer: Any, question: str, expected: str) -> None:
        # Arrange
        service = QueryService(make_semantic_layer(data=_screening_frame()))

        # Act
        result = _ask(service, question, _screened_share())

        # Assert
        assert result.claim is None
        assert expected in result.failure_reason


# --------------------------------------------------------------------------- rule 3: encounter key


class TestEncounterCounts:
    def test_ask_admissions_countsDistinctEncountersNotRowsOrPatients(self, make_semantic_layer: Any) -> None:
        # Arrange: 91 + 12 rows, 87 admissions, 63 patients
        frame = _admissions_frame()
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(intent="COUNT", confidence=0.99)

        # Act
        result = _ask(service, "how many admissions were there?", plan)

        # Assert
        assert (frame.height, frame.get_column("encounter_id").n_unique()) == (103, 87)
        assert result.claim == "Total count: 87 visits. Each is one admission (a distinct encounter_id)."

    def test_ask_admissionsByPathogen_groupOfFewPatientsHiddenWhateverItsAdmissions(
        self, make_semantic_layer: Any
    ) -> None:
        # Arrange: Legionella has 12 admissions from 3 patients; A000 has two pathogens
        service = QueryService(make_semantic_layer(data=_admissions_frame()))
        plan = QueryPlan(intent="COUNT", group_by="pathogen", confidence=0.99)

        # Act
        result = _ask(service, "how many admissions were there by pathogen?", plan)

        # Assert: the labels of Legionella and Influenza (1 patient) are never shown
        shown = _table(result)
        assert "Legionella" not in shown and "Influenza" not in shown
        assert (shown["S. pneumoniae"], shown["None identified"], shown["S. aureus"]) == (45, 30, 15)
        assert "An admission with rows in more than one pathogen is counted in each" in result.claim

    def test_ask_encounterSharedByTwoPatients_refused(self, make_semantic_layer: Any) -> None:
        # Arrange: encounter A001 is on rows of P001 and P000
        frame = _admissions_frame().with_columns(
            pl.when(pl.col("encounter_id") == "A000")
            .then(pl.lit("A001"))
            .otherwise(pl.col("encounter_id"))
            .alias("encounter_id")
        )
        service = QueryService(make_semantic_layer(data=frame))

        # Act
        result = _ask(service, "how many admissions were there?", QueryPlan(intent="COUNT", confidence=0.99))

        # Assert
        assert result.claim is None
        assert "does not identify one patient's admissions" in result.failure_reason

    def test_ask_testsShareByYear_rateOverDistinctTests(self, make_semantic_layer: Any) -> None:
        # Arrange: duplicate rows are one test
        frame = _tests_frame()
        service = QueryService(make_semantic_layer(data=frame))
        tests = frame.unique(subset=["encounter_id"])
        expected = tests.group_by("test_year").agg((pl.col("result") == "positive").mean().alias("rate"))
        plan = QueryPlan(
            intent="COUNT",
            proportion_column="result",
            proportion_value="positive",
            group_by="test_year",
            confidence=0.99,
        )

        # Act
        result = _ask(service, "what share of tests were positive each year?", plan)

        # Assert
        assert result.failure_reason is None
        shown = _table(result)
        for year, rate in expected.iter_rows():
            assert shown[display_level("test_year", year)] == pytest.approx(rate, abs=5e-5)


# --------------------------------------------------------------------------- reading the words


class TestReadUnits:
    @pytest.mark.parametrize(
        ("question", "unit", "noun", "rollup"),
        [
            pytest.param("how many patients tested positive each year?", "patients", "patients", None, id="patients"),
            pytest.param(
                "what share of tests in patients 65 and older were positive?", "encounters", "tests", None, id="head"
            ),
            pytest.param("in patients 70+, how many admissions?", "encounters", "admissions", None, id="after_head"),
            pytest.param("how many patients have ever been screened?", "patients", "patients", "any", id="ever"),
            pytest.param("how many women were screened?", None, None, None, id="limiting_noun_not_a_unit"),
            pytest.param(
                "what share of patients with at least one admission had diabetes?",
                "patients",
                "patients",
                None,
                id="population_phrase_not_a_rollup",
            ),
            pytest.param("share with protective igg on any check", None, None, "any", id="on_any"),
            pytest.param("share always screened", None, None, "all", id="always"),
            pytest.param("how many tested positive?", None, None, None, id="none"),
            pytest.param("", None, None, None, id="empty"),
        ],
    )
    def test_readUnits_question_unitAndRollup(self, question: str, unit: Any, noun: Any, rollup: Any) -> None:
        # Arrange: question as parametrized

        # Act
        words = read_units(question)

        # Assert
        assert (words.unit, words.noun, words.rollup) == (unit, noun, rollup)

    def test_accountedQuestion_rollupAndCountedNoun_removedOnly(self) -> None:
        # Arrange
        question = "what share of admissions had a vaccine on any visit by pathogen"

        # Act
        accounted = accounted_question(question, read_units(question), encounter_counted=True)

        # Assert
        assert accounted == "what share of had a vaccine by pathogen"

    @pytest.mark.parametrize(
        ("columns", "noun", "expected"),
        [
            pytest.param(["patient_id", "encounter_id", "age"], "admissions", "encounter_id", id="one"),
            pytest.param(["patient_id", "hadm_id", "test_id"], "tests", "test_id", id="named_by_noun"),
            pytest.param(["patient_id", "visit_id", "encounter_id"], "visits", "visit_id", id="named_beats_generic"),
            pytest.param(["patient_id", "hadm_id"], "tests", None, id="admission_id_never_counts_tests"),
            pytest.param(["patient_id", "visit_id", "stay_id"], "encounters", None, id="no_fit"),
            pytest.param(["patient_id", "hadm_id", "admission_id"], "admissions", None, id="two_fit_ambiguous"),
            pytest.param(["patient_id", "age"], "admissions", None, id="none"),
        ],
    )
    def test_pickEncounterKey_columns_onlyWhenUnambiguous(self, columns: list[str], noun: str, expected: Any) -> None:
        # Arrange
        candidates = encounter_key_candidates(columns, "patient_id")

        # Act
        chosen = pick_encounter_key(candidates, noun) if candidates else None

        # Assert
        assert chosen == expected


class TestShareGrouping:
    @pytest.mark.parametrize(
        "question",
        [
            pytest.param("what share of tests were positive each year?", id="each"),
            pytest.param("what share of tests were positive, by test year? keep the rest", id="by_mid_text"),
        ],
    )
    def test_peelCohort_shareAskedByGroup_keepsTheGrouping(self, question: str) -> None:
        # Arrange
        frame = _tests_frame()

        # Act
        peeled = peel_cohort(question, frame.columns, lambda column: frame.get_column(column).unique().to_list())

        # Assert
        assert (peeled.proportion_column, peeled.locked_group) == ("result", "test_year")


class TestPlanContract:
    @pytest.mark.parametrize("rollup", [pytest.param("first", id="unknown"), pytest.param("", id="empty")])
    def test_queryPlan_unknownRollup_raises(self, rollup: str) -> None:
        # Arrange / Act / Assert: an unknown roll-up never reaches execution (read as "any" there)
        with pytest.raises(ValueError, match="Invalid rollup"):
            QueryPlan(intent="COUNT", rollup=rollup)  # type: ignore[arg-type]

    def test_validateQueryPlan_entityKeyNotAColumn_invalid(self, make_semantic_layer: Any) -> None:
        # Arrange
        layer = make_semantic_layer(data=_tests_frame())

        # Act
        validation = layer._validate_query_plan(QueryPlan(intent="COUNT", entity_key="admission_id"))

        # Assert
        assert validation == {"valid": False, "error": "Column 'admission_id' not found in dataset"}


def _admission_share_frame() -> pl.DataFrame:
    """Vaccination per admission by year. 2023: 30 admissions with vaccination recorded, all of 3
    patients, and 50 unrecorded admissions of 40 other patients; 2024: 40 recorded admissions of
    40 patients."""
    rows: list[dict[str, Any]] = []
    for i in range(30):
        rows.append(
            {
                "patient_id": f"R{i % 3}",
                "encounter_id": f"A{i:03d}",
                "year": "2023",
                "vaccine": "yes" if i % 2 else "no",
            }
        )
    for i in range(50):
        rows.append({"patient_id": f"U{i % 40:02d}", "encounter_id": f"B{i:03d}", "year": "2023", "vaccine": None})
    for i in range(40):
        rows.append(
            {
                "patient_id": f"C{i:02d}",
                "encounter_id": f"C{i:03d}",
                "year": "2024",
                "vaccine": "yes" if i % 2 else "no",
            }
        )
    return pl.DataFrame(rows, schema={c: pl.String for c in ("patient_id", "encounter_id", "year", "vaccine")})


def _two_groups_shared_by_two_patients() -> pl.DataFrame:
    """Admissions in two wards: 85 only in A, 80 only in B (one patient each), and 35 in both,
    all of two patients. A=120 and B=115 of 200 would show the 35 shared admissions."""
    rows: list[dict[str, Any]] = []
    for i in range(85):
        rows.append({"patient_id": f"A{i:03d}", "encounter_id": f"EA{i:03d}", "ward": "A"})
    for i in range(80):
        rows.append({"patient_id": f"B{i:03d}", "encounter_id": f"EB{i:03d}", "ward": "B"})
    for i in range(35):
        for ward in ("A", "B"):
            rows.append({"patient_id": f"S{i % 2}", "encounter_id": f"ES{i:03d}", "ward": ward})
    return pl.DataFrame(rows, schema={c: pl.String for c in ("patient_id", "encounter_id", "ward")})


class TestEncounterPatientCells:
    def test_ask_rateByAdmission_recordedAdmissionsOfFewPatients_groupHidden(self, make_semantic_layer: Any) -> None:
        # Arrange: 2023's 30 recorded admissions are 3 patients' (43 patients with unrecorded ones)
        service = QueryService(make_semantic_layer(data=_admission_share_frame()))
        plan = QueryPlan(
            intent="COUNT", proportion_column="vaccine", proportion_value="yes", group_by="year", confidence=0.99
        )

        # Act
        result = _ask(service, "what share of admissions had a vaccine by year?", plan)

        # Assert: 2023 is never shown, by label, rate or n
        table = present_analysis(result.analysis)["table"]
        assert table is not None
        assert display_level("year", "2023") not in {str(row["year"]) for row in table["rows"]}

    def test_ask_admissionCount_fewPatientsBehindTotal_totalHidden(self, make_semantic_layer: Any) -> None:
        # Arrange: 30 admissions, 3 patients
        frame = _admission_share_frame().filter(pl.col("patient_id").str.starts_with("R"))
        service = QueryService(make_semantic_layer(data=frame))

        # Act
        result = _ask(service, "how many admissions were there?", QueryPlan(intent="COUNT", confidence=0.99))

        # Assert
        assert result.claim == "Total count hidden."

    def test_ask_admissionsInTwoWardsOfTwoPatients_everyWardHidden(self, make_semantic_layer: Any) -> None:
        # Arrange
        service = QueryService(make_semantic_layer(data=_two_groups_shared_by_two_patients()))
        plan = QueryPlan(intent="COUNT", group_by="ward", confidence=0.99)

        # Act
        result = _ask(service, "how many admissions were there by ward?", plan)

        # Assert: without suppression A=120, B=115 and 200 in all would show the 2 patients' 35;
        # the 2 readmitted patients (200 admissions of 167 patients) also hide the total
        assert result.claim == "Total count hidden."
        assert present_analysis(result.analysis)["table"] is None

    def test_ask_testsWithOnlyAnAdmissionId_refused(self, make_semantic_layer: Any) -> None:
        # Arrange: the only encounter ID is the admission's
        frame = _admissions_frame().rename({"encounter_id": "hadm_id"})
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(intent="COUNT", confidence=0.99)

        # Act
        result = _ask(service, "how many tests were there?", plan)

        # Assert
        assert result.claim is None
        assert result.failure_reason.startswith("This data has no column that identifies each of the tests")

    def test_ask_partitionHiddenWardsOfFewPatients_everyWardHidden(self, make_semantic_layer: Any) -> None:
        # Arrange: wards C (12 admissions, 2 patients) and D (13 admissions, 3 patients) are hidden;
        # the total minus A and B would give their 25 admissions, all of 5 patients
        rows: list[dict[str, Any]] = []
        # A: 65 admissions of 50 patients (15 readmitted), so readmissions overall are not few.
        for ward, admissions, patients in (("A", 65, 50), ("B", 40, 40), ("C", 12, 2), ("D", 13, 3)):
            for i in range(admissions):
                rows.append({"patient_id": f"{ward}{i % patients:02d}", "encounter_id": f"{ward}{i:03d}", "ward": ward})
        service = QueryService(make_semantic_layer(data=pl.DataFrame(rows)))
        plan = QueryPlan(intent="COUNT", group_by="ward", confidence=0.99)

        # Act
        result = _ask(service, "how many admissions were there by ward?", plan)

        # Assert
        table = present_analysis(result.analysis)["table"]
        assert table is not None
        assert all(row["count"] == SECONDARY_SUPPRESSED for row in table["rows"])


# --------------------------------------------------------------------------- second review: repros


def _hidden_everywhere(result: Any) -> bool:
    """No share, rate or count of the answer is shown (withheld, refused, or every cell hidden)."""
    if result.claim is None:
        return True
    table = present_analysis(result.analysis)["table"]
    if table is None:
        return True
    shown = [c for c in table["columns"] if c in ("proportion", "rate", "count")]
    return all(row[c] == SECONDARY_SUPPRESSED for row in table["rows"] for c in shown)


class TestSecondReview:
    def test_ask_everRateByOverlappingYear_bothYearEventsNotDerivable(self, make_semantic_layer: Any) -> None:
        # Arrange: 80 patients tested in 2021 and 2022; 30 positive each year, 5 of them in both, so
        # 30 + 30 - 55 (the ungrouped "ever" events) would give the 5
        rows: list[dict[str, Any]] = []
        for i in range(80):
            rows.append(
                {"patient_id": f"P{i:02d}", "test_year": "2021", "result": "positive" if i < 30 else "negative"}
            )
            rows.append(
                {"patient_id": f"P{i:02d}", "test_year": "2022", "result": "positive" if 25 <= i < 55 else "negative"}
            )
        service = QueryService(make_semantic_layer(data=pl.DataFrame(rows)))
        plan = QueryPlan(
            intent="COUNT",
            proportion_column="result",
            proportion_value="positive",
            group_by="test_year",
            confidence=0.99,
        )

        # Act
        result = _ask(service, "what share of patients ever tested positive, by test year?", plan)

        # Assert
        assert _hidden_everywhere(result)

    def test_ask_everAndAlways_fewPatientsChanged_bothWithheld(self, make_semantic_layer: Any) -> None:
        # Arrange: 40 screened at both visits, 5 at one only: ever 45% - always 40% of 100 = 5
        rows: list[dict[str, Any]] = []
        for i in range(100):
            first, second = ("yes", "yes") if i < 40 else (("yes", "no") if i < 45 else ("no", "no"))
            rows += [
                {"patient_id": f"P{i:03d}", "visit_id": f"V{i}a", "screened": first},
                {"patient_id": f"P{i:03d}", "visit_id": f"V{i}b", "screened": second},
            ]
        service = QueryService(make_semantic_layer(data=pl.DataFrame(rows)))
        plan = QueryPlan(intent="COUNT", proportion_column="screened", proportion_value="yes", confidence=0.99)

        # Act
        ever = _ask(service, "what share of patients were ever screened?", plan)
        always = _ask(service, "what share of patients were always screened?", plan)

        # Assert
        assert _hidden_everywhere(ever) and _hidden_everywhere(always)

    @pytest.mark.parametrize("value", [pytest.param(0, id="zero"), pytest.param(1, id="one")])
    def test_ask_encounterShareOfValue_zeroAndOneAlike_fewPatientsHidden(
        self, make_semantic_layer: Any, value: int
    ) -> None:
        # Arrange: 30 unvaccinated admissions (value 0) of 3 patients, 60 vaccinated of 60
        rows = [{"patient_id": f"P{i:03d}", "encounter_id": f"A{i:03d}", "vaccinated": 1} for i in range(60)]
        rows += [{"patient_id": f"R{k % 3}", "encounter_id": f"C{k:03d}", "vaccinated": 0} for k in range(30)]
        service = QueryService(make_semantic_layer(data=pl.DataFrame(rows)))
        plan = QueryPlan(intent="COUNT", proportion_column="vaccinated", proportion_value=value, confidence=0.99)

        # Act
        result = _ask(service, "what share of admissions had vaccinated?", plan)

        # Assert: the 3 patients behind one side hide the share whichever value is asked
        assert _hidden_everywhere(result)

    @pytest.mark.parametrize("group_by", [pytest.param(None, id="total"), pytest.param("ward", id="by_ward")])
    def test_ask_admissionsOneReadmission_notDerivableFromPatients(
        self, make_semantic_layer: Any, group_by: str | None
    ) -> None:
        # Arrange: 100 patients, one readmitted: 101 admissions minus 100 patients gives 1
        rows = [
            {"patient_id": f"P{i:03d}", "encounter_id": f"A{i:03d}", "ward": ("medical", "surgical")[i % 2]}
            for i in range(100)
        ]
        rows.append({"patient_id": "P000", "encounter_id": "A999", "ward": "medical"})
        service = QueryService(make_semantic_layer(data=pl.DataFrame(rows)))

        # Act
        plan = QueryPlan(intent="COUNT", group_by=group_by, confidence=0.99)
        result = _ask(service, "how many admissions were there?", plan)

        # Assert
        assert result.claim == "Total count hidden."

    def test_ask_admissionsWithNoPatientId_failClosedAndNoteAgrees(self, make_semantic_layer: Any) -> None:
        # Arrange: 60 patients, 20 of them readmitted; 15 Legionella admissions with no patient ID
        rows = [
            {"patient_id": f"P{i:03d}", "encounter_id": f"{kind}{i:03d}", "pathogen": ("S. pneumoniae", "None")[i % 2]}
            for kind, count in (("A", 60), ("B", 20))
            for i in range(count)
        ]
        rows += [{"patient_id": None, "encounter_id": f"L{k:02d}", "pathogen": "Legionella"} for k in range(15)]
        frame = pl.DataFrame(rows, schema={"patient_id": pl.String, "encounter_id": pl.String, "pathogen": pl.String})
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(intent="COUNT", group_by="pathogen", confidence=0.99)

        # Act
        result = _ask(service, "how many admissions by pathogen?", plan)

        # Assert: Legionella (0 keyed patients) is never shown, and the note does not say the
        # unkeyed rows are left out of a total that counts them
        assert "Legionella" not in _table(result)
        assert "their encounters are counted" in result.claim
        assert "not counted as patients" not in result.claim

    def test_presentRate_encounterGroupWithoutCells_hidden(self) -> None:
        # Arrange: cells came back for one group only
        analysis = {
            "type": "rate",
            "column": "vaccinated",
            "value": 1,
            "group_col": "ward",
            "groups": [
                {"group": "A", "rate": 0.5, "events": 30, "n": 60},
                {"group": "B", "rate": 0.5, "events": 30, "n": 60},
            ],
            "n_used": 120,
            "n_excluded": 0,
            "n_filtered_out": 0,
            "encounter_cells": {
                **dict.fromkeys(
                    ("all_total_patients", "all_event_patients", "all_nonevent_patients", "unrecorded_patients"), 120
                ),
                **dict.fromkeys(
                    (
                        "all_total_excess",
                        "all_total_readmitted",
                        "all_event_excess",
                        "all_event_readmitted",
                        "all_nonevent_excess",
                        "all_nonevent_readmitted",
                        "multi_patients",
                        "filtered_patients",
                    ),
                    0,
                ),
                "groups": [
                    {
                        "group": "A",
                        **dict.fromkeys(("patients", "event_patients", "nonevent_patients", "complement_patients"), 60),
                        **dict.fromkeys(("excess", "readmitted", "event_excess", "event_readmitted"), 0),
                        **dict.fromkeys(("nonevent_excess", "nonevent_readmitted"), 0),
                    }
                ],
            },
        }

        # Act
        table = present_analysis(analysis)["table"]

        # Assert: B has no cells, so it is hidden (and, with one group left, so is A)
        assert table is not None
        assert all(row["rate"] == SECONDARY_SUPPRESSED for row in table["rows"])

    def test_ask_everShareWithLimitThatVariesByVisit_refused(self, make_semantic_layer: Any) -> None:
        # Arrange: age differs between a patient's visits (60s, then 70s)
        rows: list[dict[str, Any]] = []
        for i in range(100):
            first, second = ("yes", "yes") if i < 30 else (("yes", "no") if i < 60 else ("no", "no"))
            rows += [
                {"patient_id": f"P{i:03d}", "visit_id": f"V{i}a", "smoker": first, "age": 60 + i % 10},
                {"patient_id": f"P{i:03d}", "visit_id": f"V{i}b", "smoker": second, "age": 70 + i % 10},
            ]
        service = QueryService(make_semantic_layer(data=pl.DataFrame(rows)))
        plan = QueryPlan(
            intent="COUNT",
            proportion_column="smoker",
            proportion_value="yes",
            cohort_steps=[CohortStep("gt", "age", 69)],
            confidence=0.99,
        )

        # Act
        result = _ask(service, "what share of patients older than 69 ever smoked?", plan)

        # Assert: never "30%" for a true 60%
        assert result.claim is None
        assert result.failure_reason.startswith("age differs between rows of the same patient")

    @pytest.mark.parametrize(
        ("question", "held"),
        [
            pytest.param("how many patients are smokers?", True, id="no_rollup_held"),
            pytest.param("how many patients ever smoked?", False, id="ever_answered"),
        ],
    )
    def test_ask_countLimitedByVaryingValue_heldUnlessRolledUp(
        self, make_semantic_layer: Any, question: str, held: bool
    ) -> None:
        # Arrange
        rows: list[dict[str, Any]] = []
        for i in range(100):
            first, second = ("yes", "yes") if i < 30 else (("yes", "no") if i < 60 else ("no", "no"))
            rows += [
                {"patient_id": f"P{i:03d}", "smoker": first},
                {"patient_id": f"P{i:03d}", "smoker": second},
            ]
        service = QueryService(make_semantic_layer(data=pl.DataFrame(rows)))
        plan = QueryPlan(intent="COUNT", cohort_steps=[CohortStep("eq", "smoker", "yes")], confidence=0.99)

        # Act
        result = _ask(service, question, plan)

        # Assert
        assert result.requires_confirmation is held
        if held:
            assert "counts each patient with at least one row that meets the limits" in result.failure_reason
        else:
            assert result.claim.startswith("Total count: 60 patients.")

    def test_rollupChoice_suggestedWording_readsBackAsARollup(self) -> None:
        # Arrange
        question = "what share of patients were screened on any visit?"

        # Act
        words = read_units(question)

        # Assert: "on any visit" is when, not what is counted
        assert (words.unit, words.rollup) == ("patients", "any")

    def test_peelCohort_byTimePhrase_neverAGrouping(self) -> None:
        # Arrange
        columns = ["patient_id", "day_of_week", "died"]

        # Act
        peeled = peel_cohort("what share of patients died by day 30?", columns, lambda c: ["yes", "no"])

        # Assert
        assert peeled.locked_group is None

    def test_ask_oneRowPerPatientShare_fewMissing_shareStillShown(self, make_semantic_layer: Any) -> None:
        # Arrange: 500 patients, 5 with the outcome missing (as main shows it)
        rows = [
            {"patient_id": f"P{i:03d}", "died": None if i < 5 else ("yes" if i % 5 == 0 else "no")} for i in range(500)
        ]
        frame = pl.DataFrame(rows, schema={"patient_id": pl.String, "died": pl.String})
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(intent="COUNT", proportion_column="died", proportion_value="yes", confidence=0.99)

        # Act
        result = _ask(service, "what share of patients died?", plan)

        # Assert
        assert result.claim.startswith("20.0% of patients had died = yes (n hidden)")

    def test_presentRate_overlappingWithoutRollupCells_everyRateHidden(self) -> None:
        # Arrange: overlapping groups, but the counts behind them are missing
        analysis = {
            "type": "rate",
            "column": "result",
            "value": "positive",
            "group_col": "test_year",
            "overlapping": True,
            "n_used": 100,
            "n_excluded": 0,
            "n_filtered_out": 0,
            "groups": [
                {"group": "2021", "rate": 0.5, "events": 30, "n": 60},
                {"group": "2022", "rate": 0.5, "events": 30, "n": 60},
            ],
        }

        # Act
        table = present_analysis(analysis)["table"]

        # Assert
        assert table is not None
        assert all(row["rate"] == SECONDARY_SUPPRESSED for row in table["rows"])

    def test_ask_partitionWithHiddenGroupThenNextQuestion_answersBoth(self, make_semantic_layer: Any) -> None:
        # Arrange: a rate by encounter whose hidden groups make the layer count the patients in
        # no shown group (a parameterised query that must not keep the cohort views in use)
        service = QueryService(make_semantic_layer(data=_admission_share_frame()))
        plan = QueryPlan(
            intent="COUNT", proportion_column="vaccine", proportion_value="yes", group_by="year", confidence=0.99
        )

        # Act
        first = _ask(service, "what share of admissions had a vaccine by year?", plan)
        second = _ask(service, "how many admissions were there?", QueryPlan(intent="COUNT", confidence=0.99))

        # Assert: the second question runs (it raised ResourceInUse before)
        assert first.claim is not None and second.claim is not None

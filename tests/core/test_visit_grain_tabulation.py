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
    """Two visits for each of 120 patients; region is one value per patient. Screened at the
    first visit when i % 3 == 0, at the second when i % 2 == 0; every 10th patient's second
    visit records nothing."""
    rows: list[dict[str, Any]] = []
    for i in range(120):
        region = ("North", "South", "East")[i % 3 if i < 60 else i % 2]
        first = "yes" if i % 3 == 0 else "no"
        second = None if i % 10 == 0 else ("yes" if i % 2 == 0 else "no")
        rows.append({"patient_id": f"P{i:03d}", "visit_id": f"V{i:03d}a", "region": region, "screened": first})
        rows.append({"patient_id": f"P{i:03d}", "visit_id": f"V{i:03d}b", "region": region, "screened": second})
    return pl.DataFrame(rows, schema={c: pl.String for c in ("patient_id", "visit_id", "region", "screened")})


def _admissions_frame(legionella_patients: int = 3) -> pl.DataFrame:
    """Admissions: 60 patients with one admission each (pathogen by i % 2), plus
    ``legionella_patients`` patients with four Legionella admissions each (12 admissions from 3
    patients). Admission A000 has a second pathogen written as a second row (Influenza, its only
    one), and A001-A015 a second one too (S. aureus), so 16 admissions have two pathogens."""
    rows: list[dict[str, Any]] = []
    for i in range(60):
        pathogen = ("S. pneumoniae", "None identified")[i % 2]
        rows.append({"patient_id": f"P{i:03d}", "encounter_id": f"A{i:03d}", "pathogen": pathogen})
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
        result = _ask(service, "how many patients tested positive each year?", _positive_by_year())

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
        result = _ask(service, "how many patients tested positive each year?", _positive_by_year())

        # Assert: no patient is in both years, so the years add up to the total (no overlap note)
        assert _table(result) == {display_level("test_year", "2021"): 30, display_level("test_year", "2022"): 15}
        assert result.claim.startswith("45 patients across 2 test_year groups.")
        assert "counted in each" not in result.claim

    def test_ask_fewPatientsInTwoYears_everyYearHidden(self, make_semantic_layer: Any) -> None:
        # Arrange: 3 patients in both years: the years' sum minus the total would give 3
        service = QueryService(make_semantic_layer(data=_tests_frame(in_both_years=3)))

        # Act
        result = _ask(service, "how many patients tested positive each year?", _positive_by_year())

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
        assert "or ask for admissions, tests or visits to count each encounter_id once" in result.failure_reason


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
        # Arrange: 76 + 12 rows, 72 admissions, 63 patients
        frame = _admissions_frame()
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(intent="COUNT", confidence=0.99)

        # Act
        result = _ask(service, "how many admissions were there?", plan)

        # Assert
        assert (frame.height, frame.get_column("encounter_id").n_unique()) == (88, 72)
        assert result.claim == "Total count: 72 visits."

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
        assert (shown["S. pneumoniae"], shown["None identified"], shown["S. aureus"]) == (30, 30, 15)
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
            pytest.param("how many adults have ever been screened?", "patients", "adults", "any", id="ever"),
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
            pytest.param(["patient_id", "visit_id", "encounter_id"], "admissions", None, id="ambiguous"),
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

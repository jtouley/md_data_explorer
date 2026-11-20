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
    ambiguous_encounter_reason,
    encounter_key_candidates,
    encounter_key_ties,
    pick_encounter_key,
    read_units,
)
from clinical_analytics.core.query_plan import CohortStep, QueryPlan
from clinical_analytics.core.query_service import QueryService
from clinical_analytics.core.result_presentation import present_analysis

# --------------------------------------------------------------------------- synthetic extracts


def _tests_frame(in_both_years: int = 15, null_keys: int = 0, duplicates: int = 1) -> pl.DataFrame:
    """C. diff tests: patients P000-P029 test positive in 2021, and the ``in_both_years``
    patients before P030 also test positive in 2022, with P030-P044 positive in 2022 only (their
    only positive row; each also has a negative 2021 test). P045-P059 test negative in 2021.
    ``duplicates`` positive 2021 tests are written twice (exact duplicate rows). ``null_keys``
    positive 2022 rows have no patient ID."""
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
    rows += [dict(rows[i]) for i in range(duplicates)]  # exact duplicate rows
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


def _ask(service: QueryService, question: str, plan: QueryPlan, confirm_grain: bool = True) -> Any:
    """Ask ``question`` with the parse fixed to ``plan`` (the parser is not under test here) and
    word accounting passed; the counted unit is still read from the question's own words.

    ``confirm_grain``: a question held only for the patient grain ("Confirm that rows are
    visits") is confirmed once, as the doctor would, so rows are declared visits."""
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
    result = service.ask(question, dataset_id="visits")
    grain_hold = "Confirm that rows are visits to count each patient once"
    if confirm_grain and result.requires_confirmation and grain_hold in (result.failure_reason or ""):
        result = service.ask(question, dataset_id="visits", confirmed=True, confirm_plan_hash=result.plan_hash)
    return result


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

    def test_ask_unitNotNamed_heldWithAPlainChoice(self, make_semantic_layer: Any) -> None:
        """With no limit on a value that differs by visit, the only question is the grain."""
        # Arrange
        service = QueryService(make_semantic_layer(data=_tests_frame()))

        # Act: "how many were there" says neither patients nor tests
        result = _ask(
            service,
            "how many were there each year?",
            QueryPlan(intent="COUNT", group_by="test_year", confidence=0.99),
            confirm_grain=False,
        )

        # Assert: asked once, offering both readings, never guessed
        assert result.requires_confirmation is True
        assert "Confirm that rows are visits to count each patient once" in result.failure_reason
        assert "or ask for admissions or tests or visits to count each encounter_id once" in result.failure_reason


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

        # Assert: every region's rate matches, small ones included (ADR015)
        assert result.failure_reason is None
        shown = _table(result)
        for region, rate, _events, _non_events in expected.iter_rows():
            assert shown[display_level("region", region)] == pytest.approx(rate, abs=5e-5)
        assert (
            "when at least one of its rows in that region has it" in result.claim
            or "only when all of its rows in that region that record it have it" in result.claim
        )

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
        assert result.claim == "Total count: 87 admissions."

    def test_ask_admissionsByPathogen_distinctAdmissionsPerPathogen(self, make_semantic_layer: Any) -> None:
        # Arrange: A000-A015 have two pathogens, so the pathogens add up to more than 87
        frame = _admissions_frame()
        expected = frame.group_by("pathogen").agg(pl.col("encounter_id").n_unique().alias("n"))
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(intent="COUNT", group_by="pathogen", confidence=0.99)

        # Act
        result = _ask(service, "how many admissions were there by pathogen?", plan)

        # Assert: each shown pathogen counts its distinct admissions, and the claim says why
        # they add up to more than the total
        shown = _table(result)
        for pathogen, n in expected.iter_rows():
            assert shown[pathogen] == n
        assert (shown["S. pneumoniae"], shown["None identified"], shown["S. aureus"]) == (45, 30, 15)
        assert result.claim.startswith("87 admissions across")
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
        # Arrange: each duplicated row is one test
        frame = _tests_frame(duplicates=11)
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
            pytest.param("how many tested positive on any visit?", None, None, "any", id="occasion_noun_not_unit"),
            pytest.param(
                "what share of patients had at least one positive test?",
                "patients",
                "patients",
                "any",
                id="at_least_one",
            ),
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
            pytest.param(
                ["patient_id", "encounter_id", "admission_type_id", "admission_source_id"],
                "admissions",
                "encounter_id",
                id="code_columns_never_ids",
            ),
            pytest.param(["patient_id", "admission_type_id"], "admissions", None, id="only_a_code_column"),
            pytest.param(["patient_id", "visit_code", "visit_status_id"], "visits", None, id="code_and_status"),
        ],
    )
    def test_pickEncounterKey_columns_onlyWhenUnambiguous(self, columns: list[str], noun: str, expected: Any) -> None:
        # Arrange
        candidates = encounter_key_candidates(columns, "patient_id")

        # Act
        chosen = pick_encounter_key(candidates, noun) if candidates else None

        # Assert
        assert chosen == expected

    def test_ambiguousEncounterReason_ties_namesOnlyTheColumnsThatTie(self) -> None:
        # Arrange: hadm_id and admission_id both name admissions; encounter_id is only general
        candidates = encounter_key_candidates(["patient_id", "encounter_id", "hadm_id", "admission_id"], "patient_id")

        # Act
        reason = ambiguous_encounter_reason("admissions", encounter_key_ties(candidates, "admissions"))

        # Assert
        assert "('hadm_id' and 'admission_id')" in reason


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


class TestEncounterChoice:
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


# --------------------------------------------------------------------------- roll-ups and limits that vary by visit


def _smokers_frame(with_age: bool = False) -> pl.DataFrame:
    """Two rows per patient for 100 patients: 30 smoke on both, 30 on the first only, 40 never.
    ``with_age``: the first row at 60-69, the second at 70-79."""
    rows: list[dict[str, Any]] = []
    for i in range(100):
        first, second = ("yes", "yes") if i < 30 else (("yes", "no") if i < 60 else ("no", "no"))
        extra = ({"age": 60 + i % 10}, {"age": 70 + i % 10}) if with_age else ({}, {})
        rows += [
            {"patient_id": f"P{i:03d}", "visit_id": f"V{i}a", "smoker": first, **extra[0]},
            {"patient_id": f"P{i:03d}", "visit_id": f"V{i}b", "smoker": second, **extra[1]},
        ]
    return pl.DataFrame(rows)


class TestRollupsAndVaryingLimits:
    @pytest.mark.parametrize(
        ("question", "rolled"),
        [
            pytest.param("what share of patients ever smoked?", pl.Expr.any, id="ever"),
            pytest.param("what share of patients always smoked?", pl.Expr.all, id="always"),
        ],
    )
    def test_ask_ungroupedRollupShare_matchesPolars(self, make_semantic_layer: Any, question: str, rolled: Any) -> None:
        # Arrange
        frame = _smokers_frame()
        expected = (
            frame.group_by("patient_id")
            .agg(rolled(pl.col("smoker") == "yes").alias("event"))
            .get_column("event")
            .mean()
        )
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(intent="COUNT", proportion_column="smoker", proportion_value="yes", confidence=0.99)

        # Act
        result = _ask(service, question, plan)

        # Assert
        assert result.failure_reason is None
        assert result.claim.startswith(f"{100 * expected:.1f}% of patients had smoker = yes")

    def test_ask_everShareWithLimitThatVariesByVisit_refused(self, make_semantic_layer: Any) -> None:
        # Arrange: age differs between a patient's visits (60s, then 70s)
        service = QueryService(make_semantic_layer(data=_smokers_frame(with_age=True)))
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

    def test_ask_everCountWithAgeLimitThatVaries_refused(self, make_semantic_layer: Any) -> None:
        # Arrange: "older than 69 ever smoked" would read "ever" only on the rows over 69 (30, not 60)
        service = QueryService(make_semantic_layer(data=_smokers_frame(with_age=True)))
        plan = QueryPlan(
            intent="COUNT",
            cohort_steps=[CohortStep("gt", "age", 69), CohortStep("eq", "smoker", "yes")],
            confidence=0.99,
        )

        # Act
        result = _ask(service, "how many patients older than 69 ever smoked?", plan)

        # Assert
        assert result.claim is None
        assert result.failure_reason.startswith("age and smoker differ between rows of the same patient")

    @pytest.mark.parametrize(
        ("question", "value"),
        [
            pytest.param("how many patients are smokers?", "yes", id="no_rollup"),
            pytest.param("how many patients ever smoked?", "yes", id="ever"),
            pytest.param("how many patients were ever non smokers?", "no", id="ever_no"),
        ],
    )
    def test_ask_countLimitedByVaryingValue_patientsWithAMatchingRow(
        self, make_semantic_layer: Any, question: str, value: str
    ) -> None:
        # Arrange
        frame = _smokers_frame()
        expected = frame.filter(pl.col("smoker") == value).get_column("patient_id").n_unique()
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(intent="COUNT", cohort_steps=[CohortStep("eq", "smoker", value)], confidence=0.99)

        # Act
        result = _ask(service, question, plan)

        # Assert: same-visit semantics, said in the answer
        assert result.claim.startswith(f"Total count: {expected} patients.")
        assert "each patient is counted when one of its rows meets every limit" in result.claim

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


# --------------------------------------------------------------------------- same-visit semantics


def _years_frame() -> pl.DataFrame:
    """Patients by year: 35 in 2019 only, 35 in 2020 only, 5 in 2019 and 2020, 20 in 2019 and
    2021, 40 in 2021 only (2019: 60, 2020: 40, 2021: 60; 135 patients)."""
    rows: list[dict[str, Any]] = []
    spans = [([2019], 35), ([2020], 35), ([2019, 2020], 5), ([2019, 2021], 20), ([2021], 40)]
    number = 0
    for years, patients in spans:
        for _ in range(patients):
            rows += [{"patient_id": f"P{number:03d}", "year": year} for year in years]
            number += 1
    return pl.DataFrame(rows)


def _ages_frame(null_ages: int = 0) -> pl.DataFrame:
    """Two tests per patient for 120 patients, at 60-69 and then at 70-79. Positive on the first
    test when i % 3 == 0, on the second when i % 4 == 0. ``null_ages`` patients have no age on
    their second test."""
    rows: list[dict[str, Any]] = []
    for i in range(120):
        second_age = None if i < null_ages else 70 + i % 10
        rows.append({"patient_id": f"P{i:03d}", "age": 60 + i % 10, "result": "positive" if i % 3 == 0 else "neg"})
        rows.append({"patient_id": f"P{i:03d}", "age": second_age, "result": "positive" if i % 4 == 0 else "neg"})
    return pl.DataFrame(rows, schema={"patient_id": pl.String, "age": pl.Int64, "result": pl.String})


class TestSameVisitSemantics:
    @pytest.mark.parametrize(
        "question",
        [
            pytest.param("how many patients with year under 2021?", id="under"),
            pytest.param("how many patients with year below 2021?", id="below"),
        ],
    )
    def test_ask_realParser_countLimitedOnOverlappingYear_patientsWithARowBefore(
        self, make_semantic_layer: Any, question: str
    ) -> None:
        # Arrange: 35 + 35 + 5 + 20 patients have a row before 2021
        frame = _years_frame()
        expected = frame.filter(pl.col("year") < 2021).get_column("patient_id").n_unique()
        service = QueryService(make_semantic_layer(data=frame))

        # Act: the real parser and word accounting; a held plan is confirmed
        first = service.ask(question, dataset_id="years")
        result = (
            service.ask(question, dataset_id="years", confirmed=True, confirm_plan_hash=first.plan_hash)
            if first.requires_confirmation
            else first
        )

        # Assert
        assert result.claim.startswith(f"Total count: {expected} patients.")

    @pytest.mark.parametrize(
        "null_ages",
        [pytest.param(0, id="all_recorded"), pytest.param(12, id="null_ages"), pytest.param(120, id="none")],
    )
    def test_ask_twoLimitsThatVaryByVisit_patientsWithOneRowMeetingBoth(
        self, make_semantic_layer: Any, null_ages: int
    ) -> None:
        # Arrange: "70 or older" and "positive" both differ between a patient's rows
        frame = _ages_frame(null_ages)
        service = QueryService(make_semantic_layer(data=frame))
        expected = frame.filter((pl.col("age") >= 70) & (pl.col("result") == "positive")).get_column("patient_id")
        plan = QueryPlan(
            intent="COUNT",
            cohort_steps=[CohortStep("ge", "age", 70), CohortStep("eq", "result", "positive")],
            confidence=0.99,
        )

        # Act
        result = _ask(service, "how many patients 70 or older tested positive?", plan)

        # Assert: one row meets both limits (never "a row at 70+" and "a positive row" apart)
        assert result.claim.startswith(f"Total count: {expected.n_unique()} patients.")
        assert "age and result differ between rows of the same patient" in result.claim

    def test_ask_unitNotNamedAndLimitVaries_heldForTheGrain(self, make_semantic_layer: Any) -> None:
        # Arrange
        service = QueryService(make_semantic_layer(data=_tests_frame()))

        # Act
        result = _ask(service, "how many tested positive each year?", _positive_by_year(), confirm_grain=False)

        # Assert: only the grain is asked, never guessed
        assert result.claim is None and result.requires_confirmation
        assert "Confirm that rows are visits to count each patient once" in result.failure_reason


# --------------------------------------------------------------------------- review of #149: regressions


def _diabetes_130_like(with_source: bool = True) -> pl.DataFrame:
    """One row per encounter: 100 patients with 1-3 encounters each (199 encounters), and code
    columns whose values are shared by many encounters and patients."""
    rows: list[dict[str, Any]] = []
    number = 0
    for i in range(100):
        for j in range(1 + i % 3):
            row = {"patient_id": f"P{i:03d}", "encounter_id": f"E{number:05d}", "admission_type_id": 1 + (i + j) % 3}
            if with_source:
                row["admission_source_id"] = 7 if j == 0 else 1
            rows.append(row)
            number += 1
    return pl.DataFrame(rows)


class TestReviewRegressions:
    @pytest.mark.parametrize("with_source", [pytest.param(True, id="type_and_source"), pytest.param(False, id="type")])
    def test_ask_admissionsBesideCodeColumns_countsTheEncounterId(
        self, make_semantic_layer: Any, with_source: bool
    ) -> None:
        # Arrange
        frame = _diabetes_130_like(with_source)
        expected = frame.get_column("encounter_id").n_unique()
        service = QueryService(make_semantic_layer(data=frame))

        # Act
        result = _ask(service, "how many admissions?", QueryPlan(intent="COUNT", confidence=0.99))

        # Assert
        assert expected == 199
        assert result.claim == f"Total count: {expected} admissions."

    def test_ask_perPatientVisitNumber_grainHoldDoesNotOfferIt(self, make_semantic_layer: Any) -> None:
        # Arrange: visit_number restarts at 1 for each patient, so it is no encounter ID
        frame = pl.DataFrame(
            [
                {"patient_id": f"P{i:03d}", "visit_number": j + 1, "flag": "yes" if j == 0 else "no"}
                for i in range(60)
                for j in range(1 + i % 3)
            ]
        )
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(intent="COUNT", cohort_steps=[CohortStep("eq", "flag", "yes")], confidence=0.99)

        # Act
        result = _ask(service, "how many were flagged?", plan, confirm_grain=False)

        # Assert: held for the grain, with no route that would then be refused
        assert result.requires_confirmation
        assert "Confirm that rows are visits to count each patient once." in result.failure_reason
        assert "visit_number" not in result.failure_reason

    def test_ask_atLeastOnePositiveTest_readAsEver(self, make_semantic_layer: Any) -> None:
        # Arrange: 84 patients with two tests; 60 positive on at least one
        rows: list[dict[str, Any]] = []
        for i in range(84):
            first = "pos" if i < 40 else "neg"
            second = "pos" if 20 <= i < 60 else "neg"
            rows += [{"patient_id": f"P{i:03d}", "result": first}, {"patient_id": f"P{i:03d}", "result": second}]
        frame = pl.DataFrame(rows)
        ever = frame.group_by("patient_id").agg((pl.col("result") == "pos").any().alias("e")).get_column("e")
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(intent="COUNT", proportion_column="result", proportion_value="pos", confidence=0.99)

        # Act
        result = _ask(service, "what share of patients had at least one positive test?", plan)

        # Assert: 60 of 84
        assert (ever.sum(), ever.len()) == (60, 84)
        assert result.claim.startswith(f"{100 * ever.mean():.1f}% of patients had result = pos")

    def test_ask_admissionsAlwaysShare_rolledUpPerAdmission(self, make_semantic_layer: Any) -> None:
        # Arrange: admissions with one or two pathogen rows
        frame = _admissions_frame()
        always = (
            frame.group_by("encounter_id")
            .agg((pl.col("pathogen") == "S. pneumoniae").all().alias("a"))
            .get_column("a")
            .mean()
        )
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(
            intent="COUNT", proportion_column="pathogen", proportion_value="S. pneumoniae", confidence=0.99
        )

        # Act
        result = _ask(service, "what share of admissions always had S. pneumoniae?", plan)

        # Assert
        assert result.failure_reason is None
        assert result.claim.startswith(f"{100 * always:.1f}% of admissions had pathogen = S. pneumoniae")

    def test_ask_groupedAlwaysShare_sentenceSaysRowsInThatGroup(self, make_semantic_layer: Any) -> None:
        # Arrange: positive in 2021; in 2022 positive for half
        rows: list[dict[str, Any]] = []
        for i in range(60):
            rows.append({"patient_id": f"P{i:03d}", "year": 2021, "result": "positive"})
            rows.append({"patient_id": f"P{i:03d}", "year": 2022, "result": "positive" if i % 2 else "negative"})
        service = QueryService(make_semantic_layer(data=pl.DataFrame(rows)))
        plan = QueryPlan(
            intent="COUNT", proportion_column="result", proportion_value="positive", group_by="year", confidence=0.99
        )

        # Act
        result = _ask(service, "what share of patients were always positive each year?", plan)

        # Assert: per group, not over the whole record; the overlap sentence names the total
        assert _table(result) == {display_level("year", 2021): 1.0, display_level("year", 2022): 0.5}
        assert "only when all of its rows in that year that record it have it" in result.claim
        assert "so the year counts add up to more than 60." in result.claim

    @pytest.mark.parametrize(
        "question",
        [
            pytest.param("how many patients are there?", id="count"),
            pytest.param("how many patients ever tested positive each year?", id="grouped"),
        ],
    )
    def test_ask_patientsNamedOnRepeatedIds_heldUntilRowsAreDeclaredVisits(
        self, make_semantic_layer: Any, question: str
    ) -> None:
        # Arrange: a repeated ID may be visits of one patient or different people sharing an ID;
        # only the doctor can say which
        frame = _tests_frame()
        service = QueryService(make_semantic_layer(data=frame))

        # Act
        held = _ask(service, question, _positive_by_year(), confirm_grain=False)
        confirmed = _ask(service, question, _positive_by_year())

        # Assert: held first, then answered in distinct patients once confirmed
        assert held.requires_confirmation and held.claim is None
        assert "Confirm that rows are visits to count each patient once" in held.failure_reason
        assert confirmed.claim.startswith("45 patients across 2 test_year groups.")

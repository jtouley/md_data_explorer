"""Answers never silently use bad data (independent DQ review: B1, H3, H4, M1, M2, M4).

Small synthetic frames reproduce each probe file of the review:
- dupids: 200 rows, 170 IDs. Main said "Total count: 170." and "25.0% of patients ... (n=200)".
- range: ages 400 and -3. Main said "Mean age was 53.54 ... n=200" with no word about them.
- crp_arm: 30 "<5" all in arm A. Main read "<5" as 5 and reported p<0.001.
- sex: Male / male / M. Main counted "sex = Male" as 60 of 150 males.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from clinical_analytics.analysis.compute import compute_comparison_analysis, compute_descriptive_analysis
from clinical_analytics.analysis.numeric_text import AmbiguousNumberFormatError, parse_numeric_text
from clinical_analytics.contracts.declarations import (
    Declarations,
    FileDeclarationStore,
    GrainDeclaration,
    LevelMapping,
    MemoryDeclarationStore,
)
from clinical_analytics.contracts.errors import ContractStoreError
from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.core.answer_checks import grain_of
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.core.query_service import (
    CONFIRMATION_NOT_SAVED_REASON,
    INVALID_DATASET_KEY_REASON,
    QueryService,
)
from clinical_analytics.core.result_presentation import present_analysis
from clinical_analytics.core.schema_inference import SchemaInferenceEngine, patient_key_by_name

N = 200
IDS = [f"P{i:03d}" for i in range(N)]


def dupids() -> pl.DataFrame:
    return pl.DataFrame(
        {"patient_id": IDS[:170] + IDS[:30], "age": [40] * 170 + [80] * 30, "died": [0] * 150 + [1] * 50}
    )


def range_frame() -> pl.DataFrame:
    ages = [30 + i % 50 for i in range(N)]
    ages[0], ages[1] = 400, -3
    return pl.DataFrame({"patient_id": IDS, "age": ages})


def crp_arm() -> pl.DataFrame:
    crp = [str(20 + i % 7) for i in range(N)]
    for i in range(0, 60, 2):
        crp[i] = "<5"
    for i in (1, 3, 5):
        crp[i] = "N/A"
    return pl.DataFrame({"patient_id": IDS, "arm": ["A" if i < 100 else "B" for i in range(N)], "crp": crp})


def sex_frame() -> pl.DataFrame:
    return pl.DataFrame({"patient_id": IDS, "sex": ["Male"] * 60 + ["male"] * 50 + ["M"] * 40 + ["Female"] * 50})


def _context(intent: AnalysisIntent, primary: str | None = None, group: str | None = None) -> AnalysisContext:
    context = AnalysisContext()
    context.inferred_intent = intent
    context.primary_variable = primary
    context.grouping_variable = group
    return context


def _service(make_semantic_layer: Any, frame: pl.DataFrame, **kwargs: Any) -> QueryService:
    return QueryService(make_semantic_layer(data=frame), **kwargs)


def _ask(service: QueryService, plan: QueryPlan, confirm_hash: str | None = None) -> Any:
    """Ask with the plan fixed (the NL parser is not under test) and word accounting passed."""
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
    return service.ask("q", dataset_id="probe", confirmed=confirm_hash is not None, confirm_plan_hash=confirm_hash)


def _shown(result: Any) -> str:
    if result.requires_confirmation:
        return f"HELD {result.failure_reason}"
    errors = [i["message"] for i in result.issues if i.get("severity") == "error"]
    return f"REFUSED {errors[0]}" if errors else f"ANSWER {result.claim}"


# --- One numeric-text rule (H4) --------------------------------------------------------------


class TestParseNumericText:
    @pytest.mark.parametrize(
        ("values", "expected", "censored", "tokens"),
        [
            pytest.param(["12", "8.5"], [12.0, 8.5], 0, 0, id="plain"),
            pytest.param(["<5", "12"], [None, 12.0], 1, 0, id="below_limit_is_censored_not_5"),
            pytest.param(["> 100", "12"], [None, 12.0], 1, 0, id="above_limit_is_censored_not_100"),
            pytest.param(["N/A", ".", "12"], [None, None, 12.0], 0, 2, id="missing_tokens"),
            pytest.param([None, "12"], [None, 12.0], 0, 0, id="null"),
            pytest.param(["1,234.5", "2.0"], [1234.5, 2.0], 0, 0, id="thousands_with_decimal"),
        ],
    )
    def test_parseNumericText_values_censoredAndMissingSeparated(
        self, values: list[str | None], expected: list[float | None], censored: int, tokens: int
    ) -> None:
        # Arrange
        series = pl.Series("crp", values, dtype=pl.String)

        # Act
        parsed = parse_numeric_text(series)

        # Assert
        assert parsed is not None
        assert parsed.values.to_list() == expected
        assert (parsed.n_censored, parsed.n_missing_tokens) == (censored, tokens)

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(["Grade 2", "Grade 3"], id="grade_is_text_not_2"),
            pytest.param([], id="empty"),
            pytest.param([None, None], id="all_null"),
            pytest.param(["n/a", "."], id="only_tokens"),
        ],
    )
    def test_parseNumericText_notNumbers_returnsNone(self, values: list[str | None]) -> None:
        # Arrange
        series = pl.Series("grade", values, dtype=pl.String)

        # Act
        parsed = parse_numeric_text(series)

        # Assert
        assert parsed is None

    def test_parseNumericText_ambiguousComma_raisesNotOnePointTwo(self) -> None:
        # Arrange: on main the descriptive path read "1,234" as 1.234
        series = pl.Series("crp", ["1,234", "2,500"])

        # Act / Assert
        with pytest.raises(AmbiguousNumberFormatError):
            parse_numeric_text(series)


class TestComputeRefusesCensoredValues:
    def test_compareGroups_crpArmWithLimits_refusesNotSubstitutes(self) -> None:
        # Arrange
        context = _context(AnalysisIntent.COMPARE_GROUPS, "crp", "arm")

        # Act
        result = compute_comparison_analysis(crp_arm(), context)

        # Assert
        assert "p_value" not in result
        assert "30 values recorded as a limit" in result["error"]

    @pytest.mark.parametrize(
        ("limits", "amount"),
        [pytest.param(3, "some", id="small_count_hidden"), pytest.param(30, "30", id="count_shown")],
    )
    def test_describe_textWithLimits_refusesWithSuppressedCount(self, limits: int, amount: str) -> None:
        # Arrange
        frame = pl.DataFrame({"crp": ["<5"] * limits + ["12"] * (N - limits)})

        # Act
        result = compute_descriptive_analysis(frame, _context(AnalysisIntent.DESCRIBE, "crp"))

        # Assert
        assert result["error"].startswith(f"crp has {amount} values recorded as a limit")

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(pl.Series("hba1c", [None] * N, dtype=pl.Float64), id="float_nulls"),
            pytest.param(pl.Series("hba1c", [None] * N, dtype=pl.String), id="text_nulls"),
        ],
    )
    def test_describe_allNull_refusesPlainly(self, values: pl.Series) -> None:
        # Arrange
        frame = values.to_frame()

        # Act
        result = compute_descriptive_analysis(frame, _context(AnalysisIntent.DESCRIBE, "hba1c"))

        # Assert
        assert result["error"] == "Every value of hba1c is missing, so there is nothing to describe."


# --- Plausibility (H3) and descriptive n clause (M4) -----------------------------------------


class TestDescriptiveDisclosure:
    def test_describe_implausibleAges_answerSaysSo(self) -> None:
        # Arrange
        analysis = compute_descriptive_analysis(range_frame(), _context(AnalysisIntent.DESCRIBE, "age"))

        # Act
        claim = present_analysis(analysis)["claim"] or ""

        # Assert: 2 values, so the count is hidden ("some")
        assert "Includes some values of age outside 0–122 years (likely entry errors)." in claim

    def test_describe_missingRows_saysHowManyExcludedAndWhy(self) -> None:
        # Arrange
        frame = pl.DataFrame({"age": [float(30 + i % 50) for i in range(180)] + [None] * 20})
        analysis = compute_descriptive_analysis(frame, _context(AnalysisIntent.DESCRIBE, "age"))

        # Act
        claim = present_analysis(analysis)["claim"] or ""

        # Assert
        assert "n=180; 20 excluded for missing data" in claim


# --- Grain (B1) ------------------------------------------------------------------------------


def test_inferSchema_repeatedPatientIds_keyChosenByName() -> None:
    # Arrange: 170 IDs over 200 rows (85% unique); main mapped no ID column
    frame = dupids()

    # Act
    schema = SchemaInferenceEngine().infer_schema(frame)

    # Assert
    assert schema.patient_id_column == "patient_id"


class TestGrain:
    @pytest.mark.parametrize(
        "plan",
        [
            pytest.param(QueryPlan(intent="COUNT", confidence=0.99), id="row_count"),
            pytest.param(QueryPlan(intent="DESCRIBE", metric="age", confidence=0.99), id="mean"),
        ],
    )
    def test_ask_repeatedIdsUndeclared_heldWithOneSentence(self, make_semantic_layer: Any, plan: QueryPlan) -> None:
        # Arrange
        service = _service(make_semantic_layer, dupids())

        # Act
        result = _ask(service, plan)

        # Assert
        assert result.requires_confirmation
        assert "Confirm that rows are visits" in result.failure_reason

    def test_ask_rowsConfirmedAsVisits_countsPatientsAndVaryingShareRefused(self, make_semantic_layer: Any) -> None:
        # Arrange: in dupids, P000-P029 have died=0 at one visit and died=1 at another
        store = MemoryDeclarationStore()
        service = _service(make_semantic_layer, dupids(), declarations=store)
        count = QueryPlan(intent="COUNT", confidence=0.99)
        held = _ask(service, count)

        # Act
        confirmed = _ask(service, count, confirm_hash=held.plan_hash)
        explicit = _ask(service, QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.99))
        share = _ask(service, QueryPlan(intent="COUNT", proportion_column="died", proportion_value=1, confidence=0.99))

        # Assert: both counts agree on 170 patients; "25.0% of visits had died=1" is never shown
        assert _shown(confirmed) == "ANSWER Total count: 170 patients."
        assert _shown(explicit) == "ANSWER Total count: 170 patients."
        assert _shown(share).startswith("REFUSED died differs between rows of the same patient")
        assert store.load("probe").grain == GrainDeclaration(
            key="patient_id", rows=200, patients=170, dataset_version="probe"
        )

    def test_ask_visitGrainGroupVariesWithinPatient_groupedCountRefused(self, make_semantic_layer: Any) -> None:
        # Arrange: patient P000 has visits at two sites
        frame = dupids().with_columns(pl.Series("site", [("North", "South", "West")[i % 3] for i in range(N)]))
        store = MemoryDeclarationStore()
        store.save(
            "probe",
            Declarations(grain=GrainDeclaration(key="patient_id", rows=200, patients=170, dataset_version="probe")),
        )
        service = _service(make_semantic_layer, frame, declarations=store)

        # Act
        result = _ask(service, QueryPlan(intent="COUNT", group_by="site", confidence=0.99))

        # Assert
        assert _shown(result).startswith("REFUSED site differs between rows of the same patient")

    def test_ask_uniqueIds_answersWithPatientsUnit(self, make_semantic_layer: Any) -> None:
        # Arrange
        service = _service(make_semantic_layer, sex_frame())

        # Act
        result = _ask(service, QueryPlan(intent="COUNT", confidence=0.99))

        # Assert
        assert _shown(result) == "ANSWER Total count: 200 patients."

    def test_declarationStore_staleDeclaration_isAskedAgain(self, make_semantic_layer: Any) -> None:
        # Arrange: declared on another version of the data (other counts)
        store = MemoryDeclarationStore()
        store.save("probe", Declarations(grain=GrainDeclaration(key="patient_id", rows=150, patients=140)))
        service = _service(make_semantic_layer, dupids(), declarations=store)

        # Act
        result = _ask(service, QueryPlan(intent="COUNT", confidence=0.99))

        # Assert
        assert result.requires_confirmation


def test_fileDeclarationStore_roundTrip_safeToSaveTwice(tmp_path: Any) -> None:
    # Arrange
    store = FileDeclarationStore(tmp_path)
    declared = Declarations(grain=GrainDeclaration(key="patient_id", rows=200, patients=170))

    # Act
    store.save("upload_abc", declared)
    store.save("upload_abc", declared)

    # Assert
    assert store.load("upload_abc") == declared
    assert store.load("other_upload") == Declarations()


# --- Plausibility holds (H3), censored refusal (H4), spellings (M2) --------------------------


class TestDataHolds:
    def test_ask_implausibleAges_heldThenAnsweredWithDisclosure(self, make_semantic_layer: Any) -> None:
        # Arrange
        service = _service(make_semantic_layer, range_frame())
        plan = QueryPlan(intent="DESCRIBE", metric="age", confidence=0.99)

        # Act
        held = _ask(service, plan)
        answered = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert
        assert "age has some values outside 0–122 years" in held.failure_reason
        assert "outside 0–122 years (likely entry errors)" in (answered.claim or "")
        assert "n=200 patients" in (answered.claim or "")

    def test_ask_crpWithLimits_refusedNotHeld(self, make_semantic_layer: Any) -> None:
        # Arrange
        service = _service(make_semantic_layer, crp_arm())
        plan = QueryPlan(intent="COMPARE_GROUPS", metric="crp", group_by="arm", confidence=0.99)

        # Act
        result = _ask(service, plan)

        # Assert
        assert _shown(result).startswith("REFUSED crp has 30 values recorded as a limit")

    @pytest.mark.parametrize(
        "plan",
        [
            pytest.param(QueryPlan(intent="COUNT", group_by="sex", confidence=0.99), id="grouped"),
            pytest.param(
                QueryPlan(
                    intent="COUNT",
                    filters=[FilterSpec(column="sex", operator="==", value="Male")],
                    scope="filtered",
                    confidence=0.99,
                ),
                id="filtered",
            ),
        ],
    )
    def test_ask_categorySpellings_heldUntilMappingConfirmed(self, make_semantic_layer: Any, plan: QueryPlan) -> None:
        # Arrange
        service = _service(make_semantic_layer, sex_frame())

        # Act
        result = _ask(service, plan)

        # Assert
        assert result.requires_confirmation
        assert "(M, Male, male). Confirm to count them together as Male." in result.failure_reason

    def test_ask_sexMaleAfterMapping_countsAllMales(self, make_semantic_layer: Any) -> None:
        # Arrange: main counted 60 (exact "Male" only) of 150 males
        service = _service(make_semantic_layer, sex_frame())
        plan = QueryPlan(
            intent="COUNT",
            filters=[FilterSpec(column="sex", operator="==", value="Male")],
            scope="filtered",
            confidence=0.99,
        )
        held = _ask(service, plan)

        # Act
        answered = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert
        assert _shown(answered) == "ANSWER Total count: 150 patients."

    def test_ask_textMetricDescribe_refusesPlainlyNoCrash(self, make_semantic_layer: Any) -> None:
        # Arrange: on main this raised "'StringColumn' object has no attribute 'mean'"
        frame = pl.DataFrame({"patient_id": IDS, "age": [str(40 + i % 30) for i in range(N - 1)] + ["unknown"]})
        service = _service(make_semantic_layer, frame)

        # Act
        result = _ask(service, QueryPlan(intent="DESCRIBE", metric="age", confidence=0.99))

        # Assert
        shown = _shown(result)
        assert shown.startswith("REFUSED")
        assert "StringColumn" not in shown


# --- Review round (PR #96): visit-grain shares, units, merges, holds, keys, declarations --------


def visits_died() -> pl.DataFrame:
    """200 visit rows of 170 patients; died is one value per patient: 40 of 170 patients died."""
    ids = IDS[:170] + IDS[:30]
    return pl.DataFrame(
        {
            "patient_id": ids,
            "died": [1 if int(pid[1:]) < 40 else 0 for pid in ids],
            "sex": ["1: Male" if int(pid[1:]) % 2 else "2: Female" for pid in ids],
        }
    )


def _visit_service(make_semantic_layer: Any, frame: pl.DataFrame) -> QueryService:
    store = MemoryDeclarationStore()
    rows, patients = frame.height, frame.get_column("patient_id").n_unique()
    store.save(
        "probe",
        Declarations(grain=GrainDeclaration(key="patient_id", rows=rows, patients=patients, dataset_version="probe")),
    )
    return _service(make_semantic_layer, frame, declarations=store)


class TestVisitGrainUnits:
    def test_ask_shareOfPatientLevelEventAtVisitGrain_countsPatients(self, make_semantic_layer: Any) -> None:
        # Arrange: by visits the share would be 70 of 200 (35.0%); by patients it is 40 of 170
        service = _visit_service(make_semantic_layer, visits_died())
        plan = QueryPlan(intent="COUNT", proportion_column="died", proportion_value=1, confidence=0.99)

        # Act
        result = _ask(service, plan)

        # Assert
        assert _shown(result) == "ANSWER 23.5% of patients had died=1 (n=170 patients)."

    def test_ask_describeCategoricalAtVisitGrain_countsDistinctPatientsPerLevel(self, make_semantic_layer: Any) -> None:
        # Arrange: each sex is 85 patients; Male has 100 visits (the row count the old line showed)
        service = _visit_service(make_semantic_layer, visits_died())

        # Act
        result = _ask(service, QueryPlan(intent="DESCRIBE", metric="sex", confidence=0.99))

        # Assert
        assert _shown(result) == "ANSWER Most common sex: 1: Male (85 patients)."

    def test_ask_describeLevelVaryingWithinPatient_refused(self, make_semantic_layer: Any) -> None:
        # Arrange: a site per visit, so a patient can have two sites
        sites = [("1: North", "2: South", "3: West")[i % 3] for i in range(N)]
        frame = visits_died().with_columns(pl.Series("site", sites))
        service = _visit_service(make_semantic_layer, frame)

        # Act
        result = _ask(service, QueryPlan(intent="DESCRIBE", metric="site", confidence=0.99))

        # Assert
        assert _shown(result).startswith("REFUSED site differs between rows of the same patient")


class TestConfirmedSpellingMerge:
    @pytest.mark.parametrize(
        ("plan", "expected"),
        [
            pytest.param(
                QueryPlan(
                    intent="COUNT",
                    filters=[FilterSpec(column="sex", operator="==", value="M")],
                    scope="filtered",
                    confidence=0.99,
                ),
                "ANSWER Total count: 150 patients.",
                id="filter_non_canonical",
            ),
            pytest.param(
                QueryPlan(
                    intent="COUNT",
                    filters=[FilterSpec(column="sex", operator="IN", value=["male"])],
                    scope="filtered",
                    confidence=0.99,
                ),
                "ANSWER Total count: 150 patients.",
                id="in_list_non_canonical",
            ),
            pytest.param(
                QueryPlan(intent="COUNT", proportion_column="sex", proportion_value="M", confidence=0.99),
                "ANSWER 75.0% of patients had sex = Male (n=200 patients).",
                id="share_non_canonical",
            ),
        ],
    )
    def test_ask_nonCanonicalSpellingAfterConfirmation_countsMergedLevel(
        self, make_semantic_layer: Any, plan: QueryPlan, expected: str
    ) -> None:
        # Arrange: Male 60, male 50, M 40 are one level once confirmed (150 males)
        service = _service(make_semantic_layer, sex_frame())
        grouped = QueryPlan(intent="COUNT", group_by="sex", confidence=0.99)
        held = _ask(service, grouped)
        _ask(service, grouped, confirm_hash=held.plan_hash)

        # Act
        result = _ask(service, plan)

        # Assert
        assert _shown(result) == expected

    def test_applyLevels_parserReadsMergedLevelsAccountingKnowsSpellings(self, make_semantic_layer: Any) -> None:
        # Arrange: before the merge "male" ties Male and male, so the parser binds neither
        service = _service(make_semantic_layer, sex_frame())
        assert "M" in service.nl_engine._distinct_values("sex")
        grouped = QueryPlan(intent="COUNT", group_by="sex", confidence=0.99)
        held = _ask(service, grouped)

        # Act
        _ask(service, grouped, confirm_hash=held.plan_hash)

        # Assert: the parser sees one level; word accounting still accepts a plan naming "M"
        mappings = service.declarations.load("probe").levels_for("probe")
        assert service.nl_engine._distinct_values("sex") == ["Female", "Male"]
        assert {"M", "Male", "male", "Female"} <= set(service._values_with_spellings(mappings)("sex"))


class TestSpellingHoldPrivacy:
    @pytest.mark.parametrize(
        ("levels", "expected"),
        [
            pytest.param(
                {"Male": 100, "male": 47, "M": 3, "Female": 50},
                "sex has values that may be one category written differently (Male, male, and some rare "
                "spellings). Confirm to count them together as Male.",
                id="three_row_M_not_named",
            ),
            pytest.param(
                {"Male": 147, "M": 3, "Female": 50},
                "sex has values that may be one category written differently (including some rare spellings). "
                "Confirm to count them together as Male.",
                id="one_large_spelling_generic",
            ),
            pytest.param(
                {"m": 4, "M": 3, "Female": 193},
                "sex has values that may be one category written differently (including some rare spellings). "
                "Confirm to count them together as one category.",
                id="all_small_no_label",
            ),
        ],
    )
    def test_ask_spellingHold_namesOnlySpellingsOfElevenOrMore(
        self, make_semantic_layer: Any, levels: dict[str, int], expected: str
    ) -> None:
        # Arrange
        sexes = [level for level, count in levels.items() for _ in range(count)]
        frame = pl.DataFrame({"patient_id": IDS, "sex": sexes})
        service = _service(make_semantic_layer, frame)

        # Act
        result = _ask(service, QueryPlan(intent="COUNT", group_by="sex", confidence=0.99))

        # Assert
        assert result.failure_reason == expected


class TestPlausibilityOnNumericText:
    def test_ask_textGlucoseWithNegative_heldLikeFloat(self, make_semantic_layer: Any) -> None:
        # Arrange: on the PR head, text glucose -5 was answered with only a trailing note
        glucose = [str(round(4.0 + (i % 50) / 10, 1)) for i in range(N)]
        glucose[0], glucose[1] = "-5", "n/a"
        service = _service(make_semantic_layer, pl.DataFrame({"patient_id": IDS, "glucose": glucose}))

        # Act
        result = _ask(service, QueryPlan(intent="DESCRIBE", metric="glucose", confidence=0.99))

        # Assert
        assert result.requires_confirmation
        assert "glucose has some values outside 0–2656 mg/dL or mmol/L" in result.failure_reason

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            pytest.param("pending", "REFUSED crp has values like 'pending' that are not numbers", id="named_word"),
            pytest.param("BLQ", "REFUSED crp has some values recorded as a limit", id="censoring_word"),
        ],
    )
    def test_ask_labWithUnreadableToken_refusesNotFrequencyTable(
        self, make_semantic_layer: Any, token: str, expected: str
    ) -> None:
        # Arrange: one entry in 200 used to turn crp into a categorical frequency table
        crp = [str(10 + i % 30) for i in range(N - 1)] + [token]
        service = _service(make_semantic_layer, pl.DataFrame({"patient_id": IDS, "crp": crp}))

        # Act
        result = _ask(service, QueryPlan(intent="DESCRIBE", metric="crp", confidence=0.99))

        # Assert
        assert _shown(result).startswith(expected)


class TestDeclarationKeys:
    def test_ask_noDatasetId_confirmationNeverPersisted(self, make_semantic_layer: Any) -> None:
        # Arrange
        store = MemoryDeclarationStore()
        service = _service(make_semantic_layer, dupids(), declarations=store)
        plan = QueryPlan(intent="COUNT", confidence=0.99)
        _ask(service, plan)
        held = service.ask("q")  # no dataset id

        # Act
        service.ask("q", confirmed=True, confirm_plan_hash=held.plan_hash)

        # Assert: nothing under a shared fallback key
        assert store._items == {}

    def test_ask_sameShapeOtherVersion_heldAgain(self, make_semantic_layer: Any) -> None:
        # Arrange: declared on version v1 with the same 200 rows and 170 IDs
        store = MemoryDeclarationStore()
        store.save(
            "probe",
            Declarations(grain=GrainDeclaration(key="patient_id", rows=200, patients=170, dataset_version="v1")),
        )
        service = _service(make_semantic_layer, dupids(), declarations=store)

        # Act
        result = _ask(service, QueryPlan(intent="COUNT", confidence=0.99))

        # Assert
        assert result.requires_confirmation

    def test_ask_declarationsUnwritable_confirmRefusesPlainly(self, make_semantic_layer: Any) -> None:
        # Arrange
        class BrokenStore(MemoryDeclarationStore):
            def update(self, dataset_id: str, change: Any) -> Declarations:
                raise ContractStoreError("Could not write declarations: Read-only file system")

        service = _service(make_semantic_layer, dupids(), declarations=BrokenStore())
        plan = QueryPlan(intent="COUNT", confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert
        assert _shown(result) == f"REFUSED {CONFIRMATION_NOT_SAVED_REASON}"


def test_fileDeclarationStore_corruptFile_updateRaisesContractStoreError(tmp_path: Any) -> None:
    # Arrange
    store = FileDeclarationStore(tmp_path)
    store.save("upload_abc", Declarations())
    store._path("upload_abc").write_text("{not json", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ContractStoreError):
        store.update("upload_abc", lambda d: d)


def test_fileDeclarationStore_writeFails_leavesNoTempFile(tmp_path: Any, monkeypatch: Any) -> None:
    # Arrange
    store = FileDeclarationStore(tmp_path)

    def failing_fsync(fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("clinical_analytics.contracts.declarations.os.fsync", failing_fsync)

    # Act
    with pytest.raises(ContractStoreError):
        store.save("upload_abc", Declarations())

    # Assert
    assert list(store._path("upload_abc").parent.glob(".tmp-*.part")) == []


def test_fileDeclarationStore_concurrentUpdates_keepBoth(tmp_path: Any) -> None:
    # Arrange: two stores (two QueryService instances) on one dataset
    import threading

    first, second = FileDeclarationStore(tmp_path), FileDeclarationStore(tmp_path)
    grain = GrainDeclaration(key="patient_id", rows=200, patients=170, dataset_version="v1")
    level = LevelMapping(column="sex", spellings=("M", "Male"), canonical="Male", dataset_version="v1")

    # Act
    threads = [
        threading.Thread(target=first.update, args=("upload_abc", lambda d: d.with_grain(grain))),
        threading.Thread(target=second.update, args=("upload_abc", lambda d: d.with_levels(level))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Assert
    stored = first.load("upload_abc")
    assert (stored.grain, stored.levels) == (grain, (level,))


class TestPatientKeyRanking:
    @pytest.mark.parametrize(
        "columns",
        [
            pytest.param(["id", "patient_id"], id="generic_first"),
            pytest.param(["patient_id", "id"], id="specific_first"),
        ],
    )
    def test_patientKeyByName_bothOrders_specificNameWins(self, columns: list[str]) -> None:
        # Act / Assert
        assert patient_key_by_name(columns) == "patient_id"

    @pytest.mark.parametrize(
        "columns",
        [
            pytest.param(["study_id", "subject_id"], id="study_first"),
            pytest.param(["subject_id", "study_id"], id="subject_first"),
        ],
    )
    def test_patientKeyByName_singleValuedStudyId_neverChosen(self, columns: list[str]) -> None:
        # Arrange: study_id is the trial's one code on every row
        distinct = {"study_id": 1, "subject_id": 170}

        # Act / Assert
        assert patient_key_by_name(columns, distinct=distinct.__getitem__) == "subject_id"

    def test_patientKeyByName_onlySingleValuedCandidate_none(self) -> None:
        # Act / Assert
        assert patient_key_by_name(["study_id"], distinct=lambda column: 1) is None

    @pytest.mark.parametrize(
        "order",
        [pytest.param(("id", "patient_id"), id="row_number_first"), pytest.param(("patient_id", "id"), id="key_first")],
    )
    def test_ask_rowNumberIdBesideRepeatedPatientId_heldOnGrain(
        self, make_semantic_layer: Any, order: tuple[str, str]
    ) -> None:
        # Arrange: "id" is a unique row number; patient_id repeats (visit rows)
        frame = dupids().with_columns(pl.Series("id", list(range(N)))).select(*order, "age", "died")
        layer = make_semantic_layer(data=frame, config_overrides={"column_mapping": {}})

        # Act
        grain = grain_of(layer)

        # Assert
        assert (grain.key, grain.repeated) == ("patient_id", True)


# --- Second review round (PR #96): null keys, units, merged plans, caching, bad ids -----------


def null_key_frame(missing: int, rows: int = 100) -> pl.DataFrame:
    """``rows`` rows, one per patient, of which ``missing`` have no patient ID."""
    ids: list[str | None] = [f"P{i:03d}" for i in range(rows - missing)]
    return pl.DataFrame({"patient_id": ids + [None] * missing, "age": [40] * rows})


def visits_with_null_keys(missing: int) -> pl.DataFrame:
    """200 visit rows of 170 patients (sex one value per patient) plus ``missing`` rows with no ID."""
    ids: list[str | None] = [*IDS[:170], *IDS[:30], *([None] * missing)]
    sex = ["2: Female" if pid is None or int(pid[1:]) % 2 else "1: Male" for pid in ids]
    return pl.DataFrame({"patient_id": ids, "sex": sex})


def _ask_question(service: QueryService, question: str, confirm_hash: str | None = None) -> Any:
    """Ask through the real parser and word accounting (nothing stubbed)."""
    return service.ask(question, dataset_id="probe", confirmed=confirm_hash is not None, confirm_plan_hash=confirm_hash)


class TestNullPatientKeys:
    def test_grainOf_nullKeys_countsDistinctNonNullNotRepeated(self, make_semantic_layer: Any) -> None:
        # Arrange: 100 rows, 98 unique IDs, 2 rows without an ID (no ID repeats)
        layer = make_semantic_layer(data=null_key_frame(2))

        # Act
        grain = grain_of(layer)

        # Assert
        assert (grain.rows, grain.patients, grain.missing_keys, grain.repeated) == (100, 98, 2, False)

    @pytest.mark.parametrize(
        ("missing", "amount"),
        [
            pytest.param(2, "some", id="small_count_suppressed"),
            pytest.param(15, "15", id="count_shown"),
        ],
    )
    def test_ask_nullKeysNoRepeat_answeredInRowsAndNullsNoted(
        self, make_semantic_layer: Any, missing: int, amount: str
    ) -> None:
        # Arrange
        service = _service(make_semantic_layer, null_key_frame(missing))

        # Act
        result = _ask(service, QueryPlan(intent="COUNT", confidence=0.99))

        # Assert: never held as repeated IDs; rows without an ID are not called patients
        note = f"Data check: patient_id is missing on {amount} rows."
        assert _shown(result) == f"ANSWER Total count: 100 rows. {note}"
        assert result.analysis["data_notes"] == [note]

    def test_ask_nullKeysAtVisitGrain_countsNonNullPatientsOnly(self, make_semantic_layer: Any) -> None:
        # Arrange: 170 patients over 200 visit rows, plus 15 rows with no ID
        service = _service(make_semantic_layer, visits_with_null_keys(15))
        plan = QueryPlan(intent="COUNT", confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert: the missing ID is not one more patient
        assert _shown(result) == (
            "ANSWER Total count: 170 patients. "
            "Data check: patient_id is missing on 15 rows; those rows are not counted as patients."
        )
        assert result.analysis["n_filtered_out"] == 0

    def test_ask_frequencyAtVisitGrainWithNullKeys_filteredOutIsZero(self, make_semantic_layer: Any) -> None:
        # Arrange
        service = _service(make_semantic_layer, visits_with_null_keys(15))
        plan = QueryPlan(intent="DESCRIBE", metric="sex", confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert: the table counts 170 patients and no filter removed any
        assert result.analysis is not None, _shown(result)
        assert result.analysis["row_count"] == 170
        assert result.analysis["n_filtered_out"] == 0


class TestUnitOfCountedEntity:
    @pytest.mark.parametrize(
        ("entity", "expected"),
        [
            pytest.param("ward", "ANSWER Total count: 12.", id="other_entity_no_unit"),
            pytest.param("patient_id", "ANSWER Total count: 200 patients.", id="patient_key_patients"),
        ],
    )
    def test_ask_countDistinctEntity_unitOnlyForPatientKey(
        self, make_semantic_layer: Any, entity: str, expected: str
    ) -> None:
        # Arrange: one row per patient, 12 wards
        frame = pl.DataFrame({"patient_id": IDS, "ward": [f"W{i % 12:02d}" for i in range(N)]})
        service = _service(make_semantic_layer, frame)

        # Act
        result = _ask(service, QueryPlan(intent="COUNT", entity_key=entity, confidence=0.99))

        # Assert
        assert _shown(result) == expected


class TestMergedLevelsOnParsedPlan:
    def test_ask_maleTwiceAfterConfirmingSpellings_answers150NoHold(self, make_semantic_layer: Any) -> None:
        # Arrange: nothing stubbed. Male 60, male 50, M 40 are one level once confirmed. ("M"
        # alone is a one-letter word, which the parser never binds to a value.)
        service = _service(make_semantic_layer, sex_frame())
        held = _ask_question(service, "count patients by sex")
        assert held.requires_confirmation
        _ask_question(service, "count patients by sex", confirm_hash=held.plan_hash)

        # Act
        first = _ask_question(service, "how many male patients")
        second = _ask_question(service, "how many male patients")

        # Assert
        assert [_shown(first), _shown(second)] == ["ANSWER Total count: 150 patients."] * 2

    def test_ask_nonCanonicalValueAfterConfirmation_accountingOnParsedPlanNoHold(
        self, make_semantic_layer: Any
    ) -> None:
        # Arrange: the parse is fixed to sex == M; word accounting runs for real
        service = _service(make_semantic_layer, sex_frame())
        grouped = QueryPlan(intent="COUNT", group_by="sex", confidence=0.99)
        held = _ask(service, grouped)
        _ask(service, grouped, confirm_hash=held.plan_hash)
        plan = QueryPlan(
            intent="COUNT",
            filters=[FilterSpec(column="sex", operator="==", value="M")],
            scope="filtered",
            confidence=0.99,
        )
        intent = SimpleNamespace(
            confidence=0.99,
            intent_type="COUNT",
            primary_variable=None,
            grouping_variable=None,
            predictor_variables=[],
            time_variable=None,
            event_variable=None,
            filters=[],
        )
        service.nl_engine.parse_query = lambda *args, **kwargs: intent  # type: ignore[method-assign]
        service.nl_engine._intent_to_plan = lambda *args: plan  # type: ignore[method-assign]

        # Act
        results = [_ask_question(service, "how many M patients") for _ in range(2)]

        # Assert
        assert [_shown(r) for r in results] == ["ANSWER Total count: 150 patients."] * 2

    def test_ask_heldPlanWithMergedSpelling_lineShowsMergedLevelAndConfirmRuns(self, make_semantic_layer: Any) -> None:
        # Arrange: after the merge, a limit on "M" is shown as the merged level, with the word asked
        ages = [30 + i % 50 for i in range(N)]
        ages[0] = 400
        service = _service(make_semantic_layer, sex_frame().with_columns(pl.Series("age", ages)))
        grouped = QueryPlan(intent="COUNT", group_by="sex", confidence=0.99)
        _ask(service, grouped, confirm_hash=_ask(service, grouped).plan_hash)
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            filters=[FilterSpec(column="sex", operator="==", value="M")],
            scope="filtered",
            confidence=0.99,
        )

        # Act
        held = _ask(service, plan)
        answered = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert
        assert held.requires_confirmation
        assert "sex == Male (includes 'M', 'male')" in (held.plan_line or "")
        assert not answered.requires_confirmation
        assert answered.analysis is not None


class TestOneLookupPerAsk:
    def test_ask_afterConfirmation_declarationsLoadedOnce(self, make_semantic_layer: Any) -> None:
        # Arrange
        class CountingStore(MemoryDeclarationStore):
            loads = 0

            def load(self, dataset_id: str) -> Declarations:
                CountingStore.loads += 1
                return super().load(dataset_id)

        service = _service(make_semantic_layer, sex_frame(), declarations=CountingStore())
        grouped = QueryPlan(intent="COUNT", group_by="sex", confidence=0.99)
        _ask(service, grouped, confirm_hash=_ask(service, grouped).plan_hash)
        CountingStore.loads = 0
        plan = QueryPlan(
            intent="COUNT",
            filters=[FilterSpec(column="sex", operator="==", value="M")],
            scope="filtered",
            confidence=0.99,
        )

        # Act
        _ask(service, plan)

        # Assert
        assert CountingStore.loads == 1

    def test_baseValueCounts_cachedUntilLevelsRemapped(self, make_semantic_layer: Any) -> None:
        # Arrange
        layer = make_semantic_layer(data=sex_frame())
        first = layer.base_value_counts("sex")

        # Act
        again = layer.base_value_counts("sex")
        layer.remap_levels("sex", {"male": "Male", "M": "Male"})
        merged = layer.base_value_counts("sex")

        # Assert
        assert again is first
        values, counts = merged
        assert values.name == "sex"
        assert dict(zip(values.to_list(), counts.to_list(), strict=True)) == {"Male": 150, "Female": 50}

    @pytest.mark.parametrize("name", ["__n", "count", "value"])
    def test_ask_groupByColumnNamedLikeCountField_heldNoCrash(self, make_semantic_layer: Any, name: str) -> None:
        # Arrange
        frame = sex_frame().rename({"sex": name})
        service = _service(make_semantic_layer, frame)

        # Act
        result = _ask(service, QueryPlan(intent="COUNT", group_by=name, confidence=0.99))

        # Assert
        assert result.requires_confirmation
        assert "may be one category written differently" in (result.failure_reason or "")

    def test_ask_textLabColumn_valueCountsComputedOncePerColumn(
        self, make_semantic_layer: Any, monkeypatch: Any
    ) -> None:
        # Arrange: glucose as text runs the statistic check and the plausibility check
        glucose = [str(round(4.0 + (i % 50) / 10, 1)) for i in range(N)]
        glucose[1] = "n/a"
        frame = pl.DataFrame({"patient_id": IDS, "glucose": glucose})
        service = _service(make_semantic_layer, frame)
        layer = service.semantic_layer
        computed: list[str] = []
        original = layer._compute_value_counts

        def spy(column: str) -> tuple[pl.Series, pl.Series]:
            computed.append(column)
            return original(column)

        monkeypatch.setattr(layer, "_compute_value_counts", spy)

        # Act
        _ask(service, QueryPlan(intent="DESCRIBE", metric="glucose", confidence=0.99))

        # Assert
        assert computed == ["glucose"]


class TestInvalidDatasetId:
    def test_fileDeclarationStore_invalidId_raisesContractStoreError(self, tmp_path: Any) -> None:
        # Arrange
        store = FileDeclarationStore(tmp_path)

        # Act / Assert
        with pytest.raises(ContractStoreError):
            store.load("../outside")

    @pytest.mark.parametrize(
        "dataset_id",
        [
            pytest.param("../outside", id="path_traversal"),
            pytest.param("a/b", id="slash"),
            pytest.param("-leading-dash", id="bad_first_char"),
            pytest.param("x" * 129, id="too_long"),
        ],
    )
    def test_ask_invalidDatasetId_plainRefusalNotException(
        self, make_semantic_layer: Any, tmp_path: Any, dataset_id: str
    ) -> None:
        # Arrange
        service = _service(make_semantic_layer, dupids(), declarations=FileDeclarationStore(tmp_path))
        plan = QueryPlan(intent="COUNT", confidence=0.99)
        service.nl_engine.parse_query = lambda *args, **kwargs: SimpleNamespace(  # type: ignore[method-assign]
            confidence=0.99,
            intent_type="COUNT",
            primary_variable=None,
            grouping_variable=None,
            predictor_variables=[],
            time_variable=None,
            event_variable=None,
            filters=[],
        )
        service.nl_engine._intent_to_plan = lambda *args: plan  # type: ignore[method-assign]
        service._accounting_hold_reason = lambda *args: None  # type: ignore[method-assign]

        # Act
        result = service.ask("how many patients", dataset_id=dataset_id)

        # Assert
        assert _shown(result) == f"REFUSED {INVALID_DATASET_KEY_REASON}"

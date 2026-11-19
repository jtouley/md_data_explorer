# SPDX-License-Identifier: MIT
"""QueryService on a population-estimate table (ADR014): never pooled, never tested, held or
answered with intervals, and relaxed only when the upload was declared public estimates.

Synthetic tables only (``make_estimate_frame``); every number is made up.
"""

from __future__ import annotations

import pytest

from clinical_analytics.contracts.config import ESTIMATE_UNDECLARED_REFUSAL
from clinical_analytics.contracts.declarations import (
    Declarations,
    MemoryDeclarationStore,
    PopulationEstimatesDeclaration,
)
from clinical_analytics.core.query_service import QueryService
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED, present_analysis

DATASET = "estimates"
VERSION = "v1"


@pytest.fixture
def estimate_service(make_semantic_layer, make_estimate_frame):
    """A query service on a declared synthetic estimate table (25 counties, so > 20 groups)."""

    def _make(declared: bool = True, **frame_args) -> QueryService:
        frame = make_estimate_frame(**({"counties": 25} | frame_args))
        layer = make_semantic_layer(dataset_name=DATASET, data=frame, config_overrides={"column_mapping": {}})
        store = MemoryDeclarationStore()
        if declared:
            store.save(
                DATASET,
                Declarations(
                    population_estimates=PopulationEstimatesDeclaration(statement="test", dataset_version=VERSION)
                ),
            )
        return QueryService(layer, declarations=store)

    return _make


def _ask(service: QueryService, question: str, confirm: bool = False):
    result = service.ask(question, dataset_id=DATASET, dataset_version=VERSION)
    if confirm and result.requires_confirmation:
        result = service.ask(
            question, dataset_id=DATASET, dataset_version=VERSION, confirmed=True, confirm_plan_hash=result.plan_hash
        )
    return result


@pytest.mark.slow
@pytest.mark.integration
class TestEstimateQuestions:
    @pytest.mark.parametrize("confirm", [False, True], ids=["asked", "confirmed"])
    def test_ask_pooledMeanByCause_refusedNotAveraged(self, estimate_service, confirm):
        # Arrange
        service = estimate_service()

        # Act
        result = _ask(service, "Average mx by cause_name", confirm=confirm)

        # Assert
        assert result.analysis is None
        assert result.failure_reason is not None and "Averaging across those rows" in result.failure_reason

    def test_ask_compareBySexInOnePlaceAndYear_intervalsNoTest(self, estimate_service):
        # Arrange
        service = estimate_service()

        # Act
        result = _ask(
            service, "Compare the cardiovascular disease mortality rate between males and females in Colorado in 2014"
        )

        # Assert
        assert result.analysis is not None and result.analysis["type"] == "estimates"
        claim = present_analysis(result.analysis)["claim"] or ""
        assert "Male" in claim and "Female" in claim and "Both" not in claim.split(".")[0]
        assert "ANOVA" not in claim and "p<" not in claim and "p=" not in claim
        assert "no significance test" in claim

    def test_ask_countyRankingOver20Groups_answeredStateLeftOut(self, estimate_service):
        # Arrange
        service = estimate_service()

        # Act
        result = _ask(
            service, "Which Colorado county had the highest ischemic heart disease rate in 2014 for both sexes?"
        )

        # Assert
        assert result.analysis is not None
        table = present_analysis(result.analysis)["table"]
        assert table is not None and len(table["rows"]) == 25
        assert "Colorado" not in {row["location_name"] for row in table["rows"]}

    def test_ask_undeclaredEstimateTable_refusedWithDeclarationHint(self, estimate_service):
        # Arrange
        service = estimate_service(declared=False)

        # Act
        result = _ask(service, "What was the cardiovascular disease mortality rate in Colorado in 2014 for both sexes?")

        # Assert
        assert result.analysis is None
        assert result.failure_reason == ESTIMATE_UNDECLARED_REFUSAL

    def test_ask_openSexConfirmed_bothSavedAndUsedNext(self, estimate_service):
        # Arrange
        service = estimate_service()
        question = "What was the cardiovascular disease mortality rate in Colorado in 2014?"

        # Act
        held = _ask(service, question)
        answered = _ask(service, question, confirm=True)
        next_answer = _ask(service, "What was the ischemic heart disease mortality rate in Colorado in 2014?")

        # Assert
        assert held.requires_confirmation and "Confirm to use sex = Both" in (held.failure_reason or "")
        assert service.declarations.load(DATASET).choices_for(VERSION) == {"sex": "Both"}
        assert answered.analysis is not None and "sex = Both" in (answered.claim or "")
        assert next_answer.analysis is not None and "the choice saved for this dataset" in (next_answer.claim or "")

    def test_ask_countBySex_rowsNotPatients(self, estimate_service):
        # Arrange
        service = estimate_service()

        # Act
        result = _ask(service, "count by sex")

        # Assert
        assert result.analysis is not None
        assert (present_analysis(result.analysis)["claim"] or "").split(" across")[0].endswith("rows")

    def test_ask_wayneCountyOnOutWayneTable_refusedNamingDetroit(self, make_semantic_layer, make_public_table):
        # Arrange: a local table whose Wayne rows exclude Detroit
        import io

        from clinical_analytics.storage.staging import read_csv_typed

        buffer = io.BytesIO()
        make_public_table("mdhhs_local").write_csv(buffer)
        frame = read_csv_typed(buffer.getvalue()).data
        layer = make_semantic_layer(dataset_name=DATASET, data=frame, config_overrides={"column_mapping": {}})
        store = MemoryDeclarationStore()
        store.save(
            DATASET,
            Declarations(
                population_estimates=PopulationEstimatesDeclaration(statement="test", dataset_version=VERSION)
            ),
        )
        service = QueryService(layer, declarations=store)

        # Act
        result = _ask(service, "How many hepatitis C cases were there in Wayne County in 2024?")

        # Assert
        assert result.analysis is None
        assert "Wayne County excluding Detroit" in (result.failure_reason or "")

    def test_estimateTable_visitFileKeyedByPatientId_neverEstimates(self, make_semantic_layer):
        # Arrange: repeated visits per patient, with a column that sounds like a rate
        import polars as pl

        frame = pl.DataFrame(
            {
                "patient_id": [f"P{i:02d}" for i in range(20) for _ in range(3)],
                "visit": ["baseline", "week4", "week12"] * 20,
                "heart_rate": [60.0 + (i % 17) for i in range(60)],
            }
        )
        layer = make_semantic_layer(dataset_name=DATASET, data=frame)
        service = QueryService(layer, declarations=MemoryDeclarationStore())

        # Act
        table = service._estimate_table()

        # Assert
        assert table is None


@pytest.fixture
def declared_frame_service(make_semantic_layer):
    """A query service on a declared synthetic estimate table built from ``frame``."""

    def _make(frame):
        layer = make_semantic_layer(dataset_name=DATASET, data=frame, config_overrides={"column_mapping": {}})
        store = MemoryDeclarationStore()
        store.save(
            DATASET,
            Declarations(
                population_estimates=PopulationEstimatesDeclaration(statement="test", dataset_version=VERSION)
            ),
        )
        return QueryService(layer, declarations=store)

    return _make


def _answer_rows(result) -> list[dict]:
    """The answer table's rows; fails when the question produced no answer table."""
    assert result.analysis is not None, result.failure_reason
    table = result.analysis.get("table")
    assert isinstance(table, dict) and table["rows"], result.analysis.get("claim")
    return table["rows"]


@pytest.mark.slow
@pytest.mark.integration
class TestSecondReviewChatPath:
    def test_ask_gbdOnlyNumberMetric_valuesHidden(self, declared_frame_service):
        # Arrange: finding 1, a GBD export with only metric_name = Number (modelled counts 1-10)
        import polars as pl

        rows = [
            {
                "location_name": s,
                "sex_name": sex,
                "metric_name": "Number",
                "year": 2023,
                "val": 1.4 + i / 2,
                "lower": 1.0 + i / 2,
                "upper": 2.0 + i / 2,
            }  # fmt: skip
            for i, s in enumerate(["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado"])
            for sex in ("Both", "Female", "Male")
        ]
        service = declared_frame_service(pl.DataFrame(rows))

        # Act
        result = _ask(service, "Which location_name had the highest val in 2023 for sex_name Both?", confirm=True)

        # Assert
        rows = _answer_rows(result)
        assert len(rows) == 6
        assert all(row["val"] == SECONDARY_SUPPRESSED for row in rows)

    def test_ask_rateBesideSmallCount_rowWithheld(self, declared_frame_service):
        # Arrange: finding 2, rate x population would give a hidden count back
        import polars as pl

        from clinical_analytics.contracts.geography import STATES

        rows = []
        for i, s in enumerate([x.name for x in STATES][:22]):
            cases = 4 if s == "Alabama" else 150 + i
            rows.append({"Geography": s, "Year": 2023, "Cases": cases, "Population": 100_000,
                         "Rate per 100000": cases / 100_000 * 1e5})  # fmt: skip
        service = declared_frame_service(pl.DataFrame(rows))

        # Act
        result = _ask(service, "Which Geography had the highest Rate per 100000?", confirm=True)

        # Assert
        alabama = next(r for r in _answer_rows(result) if r["Geography"] == "Alabama")
        assert alabama["Rate per 100000"] == SECONDARY_SUPPRESSED

    def test_ask_totalWithSmallPart_totalWithheld(self, declared_frame_service):
        # Arrange: finding 3, Both = 46 and Male = 41 with Female = 5
        import polars as pl

        from clinical_analytics.contracts.geography import STATES

        rows = []
        for i, s in enumerate([x.name for x in STATES][:22]):
            female = 5 if s == "Alaska" else 30 + i
            for sex, c in (("Both", 41 + i + female), ("Female", female), ("Male", 41 + i)):
                rows.append({"State": s, "Year": 2023, "Sex": sex, "Cases": c})
        service = declared_frame_service(pl.DataFrame(rows))

        # Act
        result = _ask(service, "Which State had the highest Cases for Sex Both in 2023?", confirm=True)

        # Assert
        alaska = next(r for r in _answer_rows(result) if r["State"] == "Alaska")
        assert alaska["Cases"] == SECONDARY_SUPPRESSED

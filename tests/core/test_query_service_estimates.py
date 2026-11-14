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
from clinical_analytics.core.result_presentation import present_analysis

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
            store.save(DATASET, Declarations(population_estimates=PopulationEstimatesDeclaration(statement="test")))
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

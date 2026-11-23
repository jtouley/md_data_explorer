# SPDX-License-Identifier: MIT
"""The default question path through the API's own pieces: upload, lease, ask, presented payload.

The model is replaced at its call boundary by recorded replies; everything else is real (the
stored upload, its contract, the SQL sandbox, the statistics code). Synthetic frames only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

from clinical_analytics.api import dependencies
from clinical_analytics.api.services.query_service import AsyncQueryService
from clinical_analytics.contracts.declarations import FileDeclarationStore
from clinical_analytics.core import nl_query_config
from clinical_analytics.core.data_home import uploads_dir
from clinical_analytics.core.query_service import QueryResult, QueryService
from clinical_analytics.model_query.engine import MODEL_UNAVAILABLE_REASON
from tests.model_query import tables

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.usefixtures("isolated_data_home")]


_BASE: dict[str, Any] = {"answer": "count", "count_of": "patients", "filters": [], "numerator": [], "measure": None,
                         "group_by": None, "period": "none", "cut_points": [], "which_visit": "any"}  # fmt: skip


def _plan(answer: str = "count", **fields: Any) -> dict[str, Any]:
    return {**_BASE, "answer": answer, **fields}


def _cond(column: str, op: str, value: Any = None, values: list[Any] | None = None) -> dict[str, Any]:
    return {"column": column, "op": op, "value": value, "values": values or []}


def _asker(upload: Callable[..., Any], frame: Any, name: str) -> Callable[..., tuple[QueryResult, list[str]]]:
    response = upload(frame, name)
    assert response.status_code == 200, response.text
    upload_id = response.json()["upload_id"]

    def _ask(
        question: str, replies: list[Any] | None, history: list[dict[str, Any]] | None = None, **options: Any
    ) -> tuple[QueryResult, list[str]]:
        prompts: list[str] = []
        queue = list(replies or [])

        def model(_system: str, user: str, _schema: dict[str, Any]) -> Any:
            prompts.append(user)
            return queue.pop(0) if queue else None

        data_dir = uploads_dir().parent
        with dependencies.lease_semantic_layer(upload_id) as lease:
            service = QueryService(lease.layer, declarations=FileDeclarationStore(data_dir), data_dir=data_dir)
            service._model_available = lambda: replies is not None
            service._model = model
            result = service.ask(
                question,
                dataset_id=upload_id,
                dataset_version=lease.layer.loaded_version,
                conversation_history=history,
                **options,
            )
        return result, prompts

    return _ask


@pytest.fixture
def ask_cohort(upload: Callable[..., Any]) -> Callable[..., tuple[QueryResult, list[str]]]:
    """Factory: ask on the uploaded one-row-per-patient cohort, with ``replies`` as the model's
    answers (None: no model server). Returns the result and the user prompts the model got."""
    return _asker(upload, tables.cohort(), "cohort")


@pytest.fixture
def ask_clinic(upload: Callable[..., Any]) -> Callable[..., tuple[QueryResult, list[str]]]:
    """The same on the uploaded visit-level clinic table."""
    return _asker(upload, tables.shots(), "clinic")


def test_ask_invertedComparison_answeredWithTheStatedOperatorAndCard(ask_cohort):
    # Arrange: the model writes "under 65" as >= 65
    reply = _plan(filters=[_cond("age", ">=", 65)])

    # Act
    result, prompts = ask_cohort("How many of our patients are under 65?", [reply])
    payload = AsyncQueryService()._analysis_payload(result)

    # Assert: answered as under 65, with the SQL and a reading card in clinical words
    expected = tables.cohort().filter(pl.col("age") < 65).height
    assert (result.requires_confirmation, result.issues) == (False, [])
    assert result.analysis is not None and result.analysis["total_count"] == expected
    assert result.sql is not None and '"age" < 65' in result.sql
    assert result.reading is not None and f"Result: {expected} of 60 patients" in result.reading
    assert payload is not None
    assert (payload["sql"], payload["reading"], payload["analysis_type"]) == (result.sql, result.reading, "count")
    assert len(prompts) == 1


def test_ask_noModelServer_fallsBackToTheMatcherPath(ask_cohort):
    # Act: nothing answers at the model address
    result, prompts = ask_cohort("How many of our older folks died?", None)

    # Assert: the matcher's own outcome, and no model-written fields
    assert prompts == []
    assert (result.sql, result.reading, result.model_reply) == (None, None, None)


def test_ask_modelPathSwitchedOff_usesTheMatcherPath(ask_cohort, monkeypatch):
    # Arrange
    monkeypatch.setattr(nl_query_config, "ENABLE_MODEL_QUERIES", False)

    # Act
    _result, prompts = ask_cohort("How many of our older folks died?", [_plan()])

    # Assert
    assert prompts == []


def test_ask_modelStopsAnswering_isRefusedPlainly(ask_cohort):
    # Act: the server was there, then the call returned nothing
    result, _prompts = ask_cohort("How many of our older folks died?", [None])

    # Assert
    assert result.failure_reason == MODEL_UNAVAILABLE_REASON
    assert result.analysis is None


def test_ask_categoryVariesBetweenVisits_asksWhichVisit(ask_clinic):
    # Act
    result, _prompts = ask_clinic("How many patients wheeze?", [_plan(filters=[_cond("wheeze", "is", values=["yes"])])])

    # Assert: a clinical question back with re-askable choices, never a guessed visit
    assert result.requires_confirmation is True and result.analysis is None
    assert result.choices and all(c["question"].startswith("How many patients wheeze?") for c in result.choices)


def test_ask_followUp_modelSeesThePreviousModelTurn(ask_cohort):
    # Arrange: the session's earlier turn, as the API records a model-written answer
    earlier = _plan(group_by="stage")
    history = [{"query": "How many patients in each stage?", "model_reply": earlier}]
    reply = _plan(group_by="stage", filters=[_cond("sex", "is", values=["female"])])

    # Act
    result, prompts = ask_cohort("Now only the women.", [reply], history=history)

    # Assert
    assert "How many patients in each stage?" in prompts[0]
    assert result.analysis is not None and result.analysis["grouped_by"] == "stage"


def test_recordTurn_modelAnswer_keepsTheReplyForTheNextQuestion(ask_cohort):
    # Arrange
    reply = _plan(group_by="stage")
    result, _prompts = ask_cohort("How many patients in each stage?", [reply])
    service = AsyncQueryService()

    # Act
    service._record_turn("sess_1", "How many patients in each stage?", result)

    # Assert
    assert service.conversation_history("sess_1") == [
        {"query": "How many patients in each stage?", "model_reply": result.model_reply}
    ]
    assert result.model_reply is not None and result.model_reply["group_by"] == "stage"


def test_ask_statisticalTest_isRoutedToTheStatisticsCode(ask_cohort):
    # Act
    result, _prompts = ask_cohort(
        "What is the odds ratio that patients died within a year, women versus men, with a CI?",
        [_plan("odds_ratio", numerator=[_cond("died_1y", "is", values=["yes"])], group_by="sex")],
    )

    # Assert: computed by analysis.compute, not statistics written in SQL
    assert result.analysis is not None and result.analysis["type"] == "predictor"
    assert result.sql is not None and "AVG" not in result.sql


def test_verify_answeredModelQuestion_keepsItsPlanMasked():
    # Arrange: an answered model-written question, as the service records it
    from clinical_analytics.api.services.query_service import AsyncQueryResult
    from clinical_analytics.model_query import verified

    service = AsyncQueryService()
    plan = _plan(filters=[_cond("stage", "is", values=["III"])])
    service._queries["qry_1"] = AsyncQueryResult(
        query_id="qry_1", status="completed", question="How many stage III patients?", dataset_id="ds_a",
        dataset_version="v1", model_reply=plan,
    )  # fmt: skip

    # Act
    saved = service.verify("qry_1")

    # Assert: stored for its dataset, the question masked
    stored = verified._load(uploads_dir().parent, "ds_a")
    assert saved is True and stored[0]["plan"] == plan
    assert stored[0]["masked"] == "How many <column> <value> patients?"


def test_verify_unansweredOrUnknown_isNotKept(test_client):
    # Arrange
    from clinical_analytics.api.services.query_service import AsyncQueryResult

    service = AsyncQueryService()
    service._queries["qry_2"] = AsyncQueryResult(query_id="qry_2", status="failed", question="q", dataset_id="ds_a")

    # Act / Assert
    assert service.verify("qry_2") is False
    assert test_client.post("/api/queries/qry_unknown/verify").status_code == 404

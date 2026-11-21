# SPDX-License-Identifier: MIT
"""The default question path through the API's own pieces: upload, lease, ask, presented payload.

The model is replaced at its call boundary by recorded replies; everything else is real (the
stored upload, its contract, the SQL sandbox, the statistics code). Synthetic frames only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


def _plan(answer: str, **fields: Any) -> dict[str, Any]:
    base = {"thought": "t", "answer": answer, "where": None, "group_by": None}
    return base | {"numerator": None, "number": None, "outcome": None} | fields


@pytest.fixture
def ask_uploaded(upload: Callable[..., Any]) -> Callable[..., tuple[QueryResult, list[str]]]:
    """Factory: upload the synthetic visits table once, then ask ``question`` on it as the API
    does, with ``replies`` as the local model's answers (None: no model server). Returns the
    result and the user prompts the model was sent."""
    response = upload(tables.visits(), "visits")
    assert response.status_code == 200, response.text
    upload_id = response.json()["upload_id"]

    def _ask(
        question: str, replies: list[Any] | None, history: list[dict[str, Any]] | None = None, **options: Any
    ) -> tuple[QueryResult, list[str]]:
        prompts: list[str] = []
        queue = list(replies or [])

        def model(_system: str, user: str, _schema: dict[str, Any]) -> Any:
            prompts.append(user)
            return queue.pop(0)

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


def test_ask_wordsTheMatcherCannotPlace_modelWritesTheQueryAndItIsAnswered(ask_uploaded):
    # Arrange: the keyword matcher holds this question ("folks", "came back") on its own
    reply = _plan("count", where="result = 'Positive' AND age_years >= 65")

    # Act
    result, prompts = ask_uploaded("How many of our older folks, 65 and up, came back positive?", [reply])
    payload = AsyncQueryService()._analysis_payload(result)

    # Assert: answered without a hold, counting distinct patients, with the SQL and its reading
    assert (result.requires_confirmation, result.issues) == (False, [])
    assert result.reading == "Counted distinct patients (patient_id) where result = 'positive' AND age_years >= 65."
    assert result.sql is not None and 'COUNT(DISTINCT "patient_id")' in result.sql
    assert result.claim is not None and result.claim.startswith("Total count: ")
    assert payload is not None
    assert (payload["sql"], payload["reading"], payload["analysis_type"]) == (result.sql, result.reading, "count")
    assert len(prompts) == 1 and "65 and up" in prompts[0]


def test_ask_noModelServer_fallsBackToTheMatcherPath(ask_uploaded):
    # Act: nothing answers at the model address
    result, prompts = ask_uploaded("How many of our older folks came back positive?", None)

    # Assert: the matcher's own outcome (a hold or a refusal), and no model-written fields
    assert prompts == []
    assert (result.sql, result.reading, result.model_reply) == (None, None, None)


def test_ask_modelPathSwitchedOff_usesTheMatcherPath(ask_uploaded, monkeypatch):
    # Arrange
    monkeypatch.setattr(nl_query_config, "ENABLE_MODEL_QUERIES", False)

    # Act
    _result, prompts = ask_uploaded("How many of our older folks came back positive?", [_plan("count")])

    # Assert
    assert prompts == []


def test_ask_modelStopsAnswering_isRefusedPlainly(ask_uploaded):
    # Act: the server was there, then the call returned nothing
    result, _prompts = ask_uploaded("How many of our older folks came back positive?", [None])

    # Assert
    assert result.failure_reason == MODEL_UNAVAILABLE_REASON
    assert result.analysis is None


def test_ask_shareThatVariesWithinPatients_asksBackWithChoices(ask_uploaded):
    # Act
    result, _prompts = ask_uploaded(
        "What fraction of our folks got screened?", [_plan("share", numerator="screened = 'yes'")]
    )

    # Assert: a question back, never an answer with a guessed roll-up
    assert result.requires_confirmation is True and result.analysis is None
    assert [choice["label"].split(":")[0] for choice in result.choices] == ["Ever", "Always"]
    assert all(choice["question"].startswith("What fraction of our folks got screened?") for choice in result.choices)


def test_ask_followUp_modelSeesThePreviousModelTurn(ask_uploaded):
    # Arrange: the session's earlier turn, as the API records a model-written answer
    earlier = _plan("count", where="result = 'positive'", group_by="region")
    history = [{"query": "How many folks came back positive in each region?", "model_reply": earlier}]
    reply = _plan("count", where="result = 'positive' AND sex = 'female'", group_by="region")

    # Act
    result, prompts = ask_uploaded("Now only the women.", [reply], history=history)

    # Assert
    assert "How many folks came back positive in each region?" in prompts[0]
    assert result.reading == (
        "Counted distinct patients (patient_id) where result = 'positive' AND sex = 'female', by region."
    )


def test_recordTurn_modelAnswer_keepsTheReplyForTheNextQuestion(ask_uploaded):
    # Arrange
    reply = _plan("count", where="result = 'positive'", group_by="region")
    result, _prompts = ask_uploaded("How many folks came back positive in each region?", [reply])
    service = AsyncQueryService()

    # Act
    service._record_turn("sess_1", "How many folks came back positive in each region?", result)

    # Assert
    assert service.conversation_history("sess_1") == [
        {"query": "How many folks came back positive in each region?", "model_reply": reply}
    ]


def test_ask_statisticalTest_isRoutedToTheStatisticsCode(ask_uploaded):
    # Act
    result, _prompts = ask_uploaded(
        "Among folks 65 and up, what is the odds ratio for a positive result with antibiotics before, with a CI?",
        [_plan("odds_ratio", outcome="result", group_by="abx", where="age_years >= 65")],
    )

    # Assert: a logistic regression from analysis.compute, not statistics written in SQL
    assert result.analysis is not None and result.analysis["type"] == "predictor"
    assert result.reading == "Odds ratio of result for abx (logistic regression, unadjusted) where age_years >= 65."
    assert result.sql is not None
    flat_sql = " ".join(result.sql.split())
    assert "AVG" not in flat_sql and "SELECT DISTINCT * FROM unified" in flat_sql

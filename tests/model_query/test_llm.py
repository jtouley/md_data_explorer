# SPDX-License-Identifier: MIT
"""The model call: warm-up at start, the cold-load timeout, and loopback only."""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.core import nl_query_config
from clinical_analytics.model_query import llm


class _FakeClient:
    """Stands in for OllamaClient at its network boundary; records each call's timeout."""

    calls: list[dict[str, Any]] = []
    answer: str | None = '{"answer": "count"}'

    def __init__(self, model: str, base_url: str, timeout: float) -> None:
        self.model, self.timeout = model, timeout

    def is_available(self) -> bool:
        return True

    def generate(self, **kwargs: Any) -> str | None:
        type(self).calls.append({"timeout": self.timeout, **kwargs})
        return type(self).answer


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr(llm, "OllamaClient", _FakeClient)
    monkeypatch.setattr(nl_query_config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    _FakeClient.calls, _FakeClient.answer = [], '{"answer": "count"}'
    llm._warm.clear()
    yield _FakeClient
    llm._warm.clear()


def test_ollama_model_coldThenWarm_usesColdTimeoutFirst(fake_client):
    # Act
    llm.ollama_model("s", "u", {})
    llm.ollama_model("s", "u", {})

    # Assert: the first call may wait for the model to load; later calls get the normal timeout
    assert [c["timeout"] for c in fake_client.calls] == [
        nl_query_config.MODEL_QUERY_COLD_TIMEOUT_S, nl_query_config.MODEL_QUERY_TIMEOUT_S,
    ]  # fmt: skip
    assert all(c["keep_alive"] == llm.KEEP_ALIVE for c in fake_client.calls)


def test_warm_model_answered_nextCallIsWarm(fake_client):
    # Act
    warmed = llm.warm_model()

    # Assert: an empty prompt loads the model; the next question gets the normal timeout
    assert warmed is True and fake_client.calls[0]["prompt"] == ""
    assert llm.call_timeout() == nl_query_config.MODEL_QUERY_TIMEOUT_S


def test_ollama_model_timedOut_nextCallIsColdAgain(fake_client):
    # Arrange
    llm._warm.set()
    fake_client.answer = None

    # Act
    reply = llm.ollama_model("s", "u", {})

    # Assert
    assert reply is None and llm.call_timeout() == nl_query_config.MODEL_QUERY_COLD_TIMEOUT_S


def test_ollama_model_remoteAddress_isRefused(fake_client, monkeypatch):
    # Arrange
    monkeypatch.setattr(nl_query_config, "OLLAMA_BASE_URL", "http://10.0.0.5:11434")

    # Act / Assert: nothing is sent off the machine
    assert llm.ollama_model("s", "u", {}) is None and fake_client.calls == []

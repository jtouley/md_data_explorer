"""Tests for Ollama fields on GET /health (fast probe for Electron + ops)."""

from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from clinical_analytics.api.main import app

    return TestClient(app)


class TestGetOllamaHealthSnapshot:
    """Unit tests for clinical_analytics.api.health_llm.get_ollama_health_snapshot."""

    def test_get_ollama_health_snapshot_unreachable_sets_reachable_false(self):
        """When Ollama HTTP probe fails, snapshot reports not reachable."""
        from clinical_analytics.api import health_llm

        with patch.object(health_llm.requests, "get", side_effect=requests.RequestException("refused")):
            snap = health_llm.get_ollama_health_snapshot()

        assert snap["ollama_reachable"] is False
        assert snap["ollama_default_model_available"] is False
        assert snap["ollama_models"] == []

    def test_get_ollama_health_snapshot_reachable_marks_default_when_present(self):
        """When /api/tags succeeds, default model availability is computed."""
        from clinical_analytics.api import health_llm

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "nomic-embed-text:latest"},
            ],
        }

        with (
            patch.object(health_llm.requests, "get", return_value=mock_resp),
            patch.object(health_llm, "OLLAMA_DEFAULT_MODEL", "llama3.1:8b"),
        ):
            snap = health_llm.get_ollama_health_snapshot()

        assert snap["ollama_reachable"] is True
        assert snap["ollama_default_model_available"] is True
        assert "llama3.1:8b" in snap["ollama_models"]

    def test_get_ollama_health_snapshot_non_200_tags_marks_unreachable(self):
        """Non-200 from Ollama is treated as unreachable."""
        from clinical_analytics.api import health_llm

        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch.object(health_llm.requests, "get", return_value=mock_resp):
            snap = health_llm.get_ollama_health_snapshot()

        assert snap["ollama_reachable"] is False
        assert snap["ollama_default_model_available"] is False


class TestHealthEndpointOllamaFields:
    """GET /health includes ollama_* keys for desktop clients."""

    def test_health_check_includes_ollama_snapshot_keys(self, test_client):
        """Health response merges API status with Ollama snapshot fields."""
        from clinical_analytics.api import health_llm

        fake_snap = {
            "ollama_base_url": "http://localhost:11434",
            "ollama_default_model": "llama3.1:8b",
            "ollama_reachable": True,
            "ollama_default_model_available": True,
            "ollama_models": ["llama3.1:8b"],
        }

        with patch.object(health_llm, "get_ollama_health_snapshot", return_value=fake_snap):
            response = test_client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["ollama_reachable"] is True
        assert body["ollama_default_model_available"] is True
        assert body["ollama_models"] == ["llama3.1:8b"]

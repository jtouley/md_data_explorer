"""
Test Tier 3 LLM Fallback (Phase 0 - ADR003).

Tests for local Ollama integration, RAG context building, and structured JSON extraction.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest
from clinical_analytics.core.nl_query_engine import QueryIntent

# ============================================================================
# Fixtures (Phase 0)
# ============================================================================


@pytest.fixture
def mock_ollama_client():
    """Mock OllamaClient for testing without actual Ollama service."""
    client = MagicMock()
    client.is_available.return_value = True
    client.is_model_available.return_value = True
    client.generate.return_value = '{"intent_type": "DESCRIBE", "primary_variable": "age", "confidence": 0.8}'
    return client


@pytest.fixture
def sample_rag_context():
    """Factory for RAG context dict with columns, aliases, examples."""

    def _make(
        columns: list[str] | None = None,
        aliases: dict[str, str] | None = None,
        examples: list[str] | None = None,
    ) -> dict:
        return {
            "columns": columns or ["age", "treatment", "viral_load"],
            "aliases": aliases or {"vl": "viral_load", "tx": "treatment"},
            "examples": examples
            or [
                "What is the average age?",
                "Compare viral load by treatment",
                "How many patients have treatment A?",
            ],
        }

    return _make


@pytest.fixture
def sample_semantic_layer_metadata():
    """Mock semantic layer with metadata for RAG context."""
    semantic_layer = MagicMock()

    # Mock get_base_view() to return a DataFrame with columns
    base_view = pl.DataFrame(
        {
            "age": [30, 35, 40],
            "treatment": ["A", "B", "A"],
            "viral_load": [100, 200, 150],
        }
    )
    semantic_layer.get_base_view.return_value = base_view

    # Mock get_column_alias_index()
    semantic_layer.get_column_alias_index.return_value = {
        "vl": "viral_load",
        "tx": "treatment",
        "viral load": "viral_load",
    }

    return semantic_layer


# ============================================================================
# Phase 0 Tests - Ollama Client
# ============================================================================


def test_ollama_client_connection_success():
    """Verify OllamaClient connects to local Ollama service."""
    from clinical_analytics.core.llm_client import OllamaClient

    # Mock successful connection
    with patch("clinical_analytics.core.llm_client.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = OllamaClient()
        assert client.is_available() is True


def test_ollama_client_connection_failure_handles_gracefully():
    """Verify connection failure returns None without crashing."""
    from clinical_analytics.core.llm_client import OllamaClient

    # Mock connection failure
    with patch("clinical_analytics.core.llm_client.requests.get") as mock_get:
        mock_get.side_effect = ConnectionError("Connection refused")

        client = OllamaClient()
        assert client.is_available() is False


def test_ollama_client_model_available():
    """Verify model availability check (llama3.1:8b or llama3.2:3b)."""
    from clinical_analytics.core.llm_client import OllamaClient

    # Mock successful model check
    with patch("clinical_analytics.core.llm_client.requests.get") as mock_get:
        # Mock connection check
        mock_conn_response = Mock()
        mock_conn_response.status_code = 200

        # Mock model list
        mock_model_response = Mock()
        mock_model_response.status_code = 200
        mock_model_response.json.return_value = {"models": [{"name": "llama3.1:8b"}, {"name": "llama3.2:3b"}]}

        mock_get.side_effect = [mock_conn_response, mock_model_response]

        client = OllamaClient(model="llama3.1:8b")
        assert client.is_model_available("llama3.1:8b") is True


def test_ollama_client_model_not_available_handles_gracefully():
    """Verify missing model handled gracefully."""
    from clinical_analytics.core.llm_client import OllamaClient

    # Mock model not found
    with patch("clinical_analytics.core.llm_client.requests.get") as mock_get:
        # Mock connection check
        mock_conn_response = Mock()
        mock_conn_response.status_code = 200

        # Mock model list (missing llama3.1:8b)
        mock_model_response = Mock()
        mock_model_response.status_code = 200
        mock_model_response.json.return_value = {"models": [{"name": "llama2:7b"}]}

        mock_get.side_effect = [mock_conn_response, mock_model_response]

        client = OllamaClient(model="llama3.1:8b")
        assert client.is_model_available("llama3.1:8b") is False


# ============================================================================
# Phase 0 Tests - RAG Context Builder
# ============================================================================


def test_rag_context_builder_includes_columns(sample_semantic_layer_metadata):
    """Verify RAG context includes available columns from semantic layer."""
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    engine = NLQueryEngine(sample_semantic_layer_metadata)

    context = engine._build_rag_context("What is the average age?")

    assert "columns" in context
    assert "age" in context["columns"]
    assert "treatment" in context["columns"]
    assert "viral_load" in context["columns"]


def test_rag_context_builder_includes_aliases(sample_semantic_layer_metadata):
    """Verify RAG context includes alias mappings."""
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    engine = NLQueryEngine(sample_semantic_layer_metadata)

    context = engine._build_rag_context("What is the average age?")

    assert "aliases" in context
    assert "vl" in context["aliases"]
    assert context["aliases"]["vl"] == "viral_load"


def test_rag_context_builder_includes_examples(sample_semantic_layer_metadata):
    """Verify RAG context includes example queries."""
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    engine = NLQueryEngine(sample_semantic_layer_metadata)

    context = engine._build_rag_context("What is the average age?")

    assert "examples" in context
    assert len(context["examples"]) > 0
    # Examples should include query patterns
    assert any("average" in ex.lower() or "compare" in ex.lower() for ex in context["examples"])


# ============================================================================
# Phase 0 Tests - Structured JSON Extraction
# ============================================================================


def test_structured_json_extraction_valid_schema(sample_semantic_layer_metadata):
    """Verify JSON extraction with valid schema returns QueryIntent."""
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    engine = NLQueryEngine(sample_semantic_layer_metadata)

    # Valid JSON response
    response = json.dumps(
        {
            "intent_type": "DESCRIBE",
            "primary_variable": "age",
            "grouping_variable": None,
            "confidence": 0.8,
        }
    )

    intent = engine._extract_query_intent_from_llm_response(response)

    assert intent is not None
    assert intent.intent_type == "DESCRIBE"
    assert intent.primary_variable == "age"
    assert intent.confidence == pytest.approx(0.8, rel=1e-2)


def test_structured_json_extraction_invalid_json_retries(sample_semantic_layer_metadata):
    """Verify invalid JSON triggers retry (up to 3 attempts)."""
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    engine = NLQueryEngine(sample_semantic_layer_metadata)

    # Invalid JSON (missing quotes, malformed)
    invalid_response = '{intent_type: "DESCRIBE", primary_variable: "age"}'

    intent = engine._extract_query_intent_from_llm_response(invalid_response, max_retries=3)

    # Should return None after all retries fail
    assert intent is None


def test_structured_json_extraction_timeout_handling(mock_ollama_client):
    """Verify timeout handling (5s default)."""
    from clinical_analytics.core.llm_client import OllamaClient

    # Mock timeout
    with patch("clinical_analytics.core.llm_client.requests.post") as mock_post:
        import requests

        mock_post.side_effect = requests.Timeout("Request timed out")

        client = OllamaClient(timeout=5.0)
        response = client.generate("test prompt")

        # Should return None on timeout
        assert response is None


# ============================================================================
# Phase 0 Tests - LLM Parse Integration
# ============================================================================


def test_llm_parse_returns_query_intent_with_confidence(sample_semantic_layer_metadata, mock_ollama_client):
    """Verify _llm_parse() returns QueryIntent with confidence >= 0.5."""
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    engine = NLQueryEngine(sample_semantic_layer_metadata)

    # Mock OllamaClient to return valid JSON
    with patch.object(engine, "_get_ollama_client", return_value=mock_ollama_client):
        intent = engine._llm_parse("What is the average age?")

        assert isinstance(intent, QueryIntent)
        assert intent.confidence >= 0.5
        assert intent.intent_type in [
            "DESCRIBE",
            "COMPARE_GROUPS",
            "FIND_PREDICTORS",
            "SURVIVAL",
            "CORRELATIONS",
            "COUNT",
        ]


def test_llm_parse_fallback_to_stub_on_error(sample_semantic_layer_metadata):
    """Verify _llm_parse() falls back to stub (confidence=0.3) on unrecoverable errors."""
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    engine = NLQueryEngine(sample_semantic_layer_metadata)

    # Mock Ollama unavailable
    with patch.object(engine, "_get_ollama_client", return_value=None):
        intent = engine._llm_parse("What is the average age?")

        # Should fall back to stub
        assert isinstance(intent, QueryIntent)
        assert intent.confidence == pytest.approx(0.3, rel=1e-2)
        assert intent.intent_type == "DESCRIBE"

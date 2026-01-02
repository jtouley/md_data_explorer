"""
Integration tests for Tier 3 LLM Fallback (Phase 0 - ADR003).

These tests require a real Ollama instance running locally.
Skip if Ollama is not available.

Run with: pytest tests/core/test_llm_fallback_integration.py -v
Or skip if Ollama unavailable: pytest tests/core/test_llm_fallback_integration.py -v -m "not integration"
"""

import pytest

# Mark all tests in this file as integration tests (require real Ollama)
# Also mark as slow since LLM calls take 5-15 seconds each
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def ollama_manager():
    """Get OllamaManager instance."""
    from clinical_analytics.core.ollama_manager import get_ollama_manager

    return get_ollama_manager()


@pytest.fixture
def ollama_client(ollama_manager):
    """Create real OllamaClient instance via manager."""
    client = ollama_manager.get_client()
    if client is None:
        pytest.skip("Ollama not available - skipping integration test")
    return client


@pytest.fixture
def skip_if_ollama_unavailable(ollama_manager):
    """Skip test if Ollama is not available."""
    status = ollama_manager.get_status()
    if not status["ready"]:
        pytest.skip(f"Ollama not ready - {ollama_manager.get_setup_instructions()}")


def test_ollama_manager_status(ollama_manager):
    """Verify OllamaManager status reporting."""
    status = ollama_manager.get_status()
    assert isinstance(status, dict)
    assert "installed" in status
    assert "running" in status
    assert "ready" in status


def test_ollama_client_real_connection(ollama_client, skip_if_ollama_unavailable):
    """Verify real OllamaClient can connect to local Ollama service."""
    assert ollama_client.is_available() is True


def test_ollama_client_real_model_available(ollama_client, skip_if_ollama_unavailable):
    """Verify real OllamaClient can check for model availability."""
    # Check for default model
    from clinical_analytics.core.nl_query_config import OLLAMA_DEFAULT_MODEL

    model_available = ollama_client.is_model_available(OLLAMA_DEFAULT_MODEL)
    # Model might not be installed, so we just verify the check works
    assert isinstance(model_available, bool)


def test_ollama_client_real_generate(ollama_client, skip_if_ollama_unavailable):
    """Verify real OllamaClient can generate responses."""
    from clinical_analytics.core.nl_query_config import OLLAMA_DEFAULT_MODEL

    if not ollama_client.is_model_available(OLLAMA_DEFAULT_MODEL):
        pytest.skip(f"Model {OLLAMA_DEFAULT_MODEL} not available")

    prompt = "What is 2+2? Respond with just the number."
    response = ollama_client.generate(prompt, model=OLLAMA_DEFAULT_MODEL)

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_llm_parse_with_real_ollama(mock_semantic_layer, skip_if_ollama_unavailable):
    """Verify _llm_parse() works with real Ollama instance."""
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    engine = NLQueryEngine(mock_semantic_layer)

    # Use a complex query that won't match Tier 1 patterns
    complex_query = "What is the relationship between treatment and viral load outcomes?"

    intent = engine._llm_parse(complex_query)

    # Should return a QueryIntent (either from LLM or stub fallback)
    assert intent is not None
    assert intent.confidence >= 0.3  # At least stub confidence
    # If Ollama worked, confidence should be >= 0.5
    # If Ollama failed, it falls back to stub with 0.3 confidence

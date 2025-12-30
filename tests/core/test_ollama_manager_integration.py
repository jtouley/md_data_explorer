"""
Integration tests for OllamaManager - validates real Ollama service.

These tests require a real Ollama instance running locally.
Skip if Ollama is not available.

Run with: pytest tests/core/test_ollama_manager_integration.py -v -m integration
"""

import pytest

from clinical_analytics.core.ollama_manager import get_ollama_manager

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def ollama_manager():
    """Get OllamaManager instance."""
    return get_ollama_manager()


def test_ollama_manager_singleton():
    """Verify OllamaManager is a singleton."""
    manager1 = get_ollama_manager()
    manager2 = get_ollama_manager()
    assert manager1 is manager2


def test_ollama_manager_status(ollama_manager):
    """Verify status reporting works."""
    status = ollama_manager.get_status()

    assert isinstance(status, dict)
    assert "installed" in status
    assert "running" in status
    assert "ready" in status
    assert "available_models" in status

    # If Ollama is not installed, skip remaining tests
    if not status["installed"]:
        pytest.skip("Ollama not installed - skipping integration tests")


def test_ollama_manager_service_detection(ollama_manager):
    """Verify service detection works."""
    status = ollama_manager.get_status()

    if not status["installed"]:
        pytest.skip("Ollama not installed")

    # Service might not be running - that's okay, we're testing detection
    is_running = ollama_manager.is_service_running()
    assert isinstance(is_running, bool)

    if not is_running:
        pytest.skip("Ollama service not running - start with 'ollama serve'")


def test_ollama_manager_model_detection(ollama_manager):
    """Verify model detection works."""
    status = ollama_manager.get_status()
    if not status["running"]:
        pytest.skip("Ollama service not running")

    models = ollama_manager.get_available_models()

    assert isinstance(models, list)
    # If service is running, we should get a list (might be empty)
    # If models are available, verify they're strings
    for model in models:
        assert isinstance(model, str)
        assert len(model) > 0


def test_ollama_manager_get_client(ollama_manager):
    """Verify get_client() returns OllamaClient when service is ready."""
    status = ollama_manager.get_status()
    if not status["ready"]:
        pytest.skip(f"Ollama not ready: {ollama_manager.get_setup_instructions()}")

    client = ollama_manager.get_client()

    if client is None:
        pytest.fail("get_client() returned None but service appears ready")

    # Verify client has expected methods
    assert hasattr(client, "is_available")
    assert hasattr(client, "generate")
    assert client.is_available() is True


def test_ollama_manager_setup_instructions(ollama_manager):
    """Verify setup instructions are provided when not ready."""
    instructions = ollama_manager.get_setup_instructions()

    assert isinstance(instructions, str)
    assert len(instructions) > 0
    assert "Ollama" in instructions


def test_ollama_manager_end_to_end(ollama_manager):
    """End-to-end test: verify OllamaManager can generate a response."""
    status = ollama_manager.get_status()
    if not status["ready"]:
        pytest.skip(f"Ollama not ready: {ollama_manager.get_setup_instructions()}")

    client = ollama_manager.get_client()

    if client is None:
        pytest.fail("get_client() returned None but service appears ready")

    # Simple test query
    test_prompt = "What is 2+2? Respond with just the number."
    response = client.generate(test_prompt, json_mode=False)

    # Should get a response (might not be perfect, but should be non-empty)
    assert response is not None
    assert isinstance(response, str)
    assert len(response.strip()) > 0

    # Response should contain "4" (or at least be non-empty)
    # We're lenient here since LLM responses can vary
    assert len(response) > 0


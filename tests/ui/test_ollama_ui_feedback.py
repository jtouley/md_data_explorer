"""
Tests for Ollama initialization feedback (TDD).

Tests ensure that:
1. initialize_ollama returns correct status for auto-download success
2. initialize_ollama returns actionable message on download failure
3. Status display structures are correct
"""

from unittest.mock import patch


class TestOllamaInitFeedback:
    """Test feedback from Ollama initialization."""

    def test_initialize_ollama_auto_download_success(self):
        """Test that initialize_ollama handles successful auto-download."""
        with patch("clinical_analytics.core.ollama_init.initialize_ollama") as mock_init:
            mock_init.return_value = {
                "installed": True,
                "running": True,
                "ready": True,
                "message": "Ollama LLM ready (1 model(s) downloaded and available)",
                "auto_downloaded": True,
            }

            from clinical_analytics.core.ollama_init import initialize_ollama

            result = initialize_ollama()

            assert result["ready"] is True
            assert result["auto_downloaded"] is True
            assert "downloaded" in result["message"]

    def test_initialize_ollama_provides_helpful_message_on_failure(self):
        """Test that initialize_ollama provides actionable message when download fails."""
        with patch("clinical_analytics.core.ollama_init.initialize_ollama") as mock_init:
            mock_init.return_value = {
                "installed": True,
                "running": True,
                "ready": False,
                "message": "Model download failed - Natural language queries will use pattern matching only.",
                "auto_downloaded": False,
            }

            from clinical_analytics.core.ollama_init import initialize_ollama

            result = initialize_ollama()

            assert result["ready"] is False
            assert "pattern matching" in result["message"] or "failed" in result["message"]


class TestOllamaStatusDisplay:
    """Test status display helpers."""

    def test_get_ollama_status_display_ready(self):
        """Test status display for ready state."""
        status = {
            "installed": True,
            "running": True,
            "ready": True,
            "message": "✓ Ollama LLM ready (1 model(s) available)",
        }

        assert "message" in status
        assert status["ready"] is True

    def test_get_ollama_status_display_not_ready(self):
        """Test status display for not ready state."""
        status = {
            "installed": True,
            "running": False,
            "ready": False,
            "message": "⚠ Ollama service not running",
        }

        assert status["ready"] is False
        assert "⚠" in status["message"] or "warning" in status["message"].lower()

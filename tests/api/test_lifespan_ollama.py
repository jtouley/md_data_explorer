"""Tests for the API lifespan's opt-in Ollama auto-initialization."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from clinical_analytics.api import main


class TestOllamaAutoinit:
    """Startup checks Ollama only when CLINICAL_ANALYTICS_OLLAMA_AUTOINIT is truthy."""

    @pytest.mark.parametrize(
        ("env_value", "expected_calls"),
        [(None, 0), ("", 0), ("0", 0), ("false", 0), ("1", 1), ("true", 1), ("YES", 1)],
        ids=["unset", "empty", "zero", "false", "one", "true", "yes_uppercase"],
    )
    def test_lifespan_autoinitFlag_startsBackgroundInitOnlyWhenEnabled(self, monkeypatch, env_value, expected_calls):
        # Arrange
        if env_value is None:
            monkeypatch.delenv(main.OLLAMA_AUTOINIT_ENV, raising=False)
        else:
            monkeypatch.setenv(main.OLLAMA_AUTOINIT_ENV, env_value)

        # Act
        with (
            patch.object(main, "create_tables"),
            patch.object(main, "_initialize_ollama_in_background") as start_init,
            TestClient(main.app),
        ):
            pass

        # Assert
        assert start_init.call_count == expected_calls

    def test_initialize_ollama_in_background_runsWithAutoDownload(self):
        # Arrange
        status = {"ready": False, "message": "Ollama not installed"}

        # Act
        with patch.object(main, "initialize_ollama", return_value=status) as init:
            thread = main._initialize_ollama_in_background()
            thread.join(timeout=5)

        # Assert
        assert not thread.is_alive()
        assert thread.daemon
        init.assert_called_once_with(auto_download=True)

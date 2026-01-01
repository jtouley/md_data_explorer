"""
Tests for Ollama initialization and auto-download (TDD).

Tests ensure that:
1. Models are automatically downloaded on first startup
2. UI shows appropriate messages during download
3. App blocks until models are ready
"""

from unittest.mock import MagicMock, patch


class TestOllamaAutoDownload:
    """Test automatic model download on first startup."""

    def test_ensure_models_downloaded_auto_downloads_when_missing(self):
        """Test that missing models are automatically downloaded."""
        # Arrange: Mock manager with no models available
        with patch("clinical_analytics.ui.ollama_init.get_ollama_manager") as mock_manager_fn:
            mock_manager = MagicMock()
            mock_manager_fn.return_value = mock_manager

            # Service is running but no models
            mock_manager.get_status.return_value = {
                "installed": True,
                "running": True,
                "ready": False,
                "available_models": [],
                "default_model": "llama3.1:8b",
                "fallback_model": "llama3.2:3b",
            }
            mock_manager.default_model = "llama3.1:8b"

            # Mock subprocess to prevent actual download
            with patch("clinical_analytics.ui.ollama_init.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Act: Call ensure_models_downloaded
                from clinical_analytics.ui.ollama_init import ensure_models_downloaded

                ensure_models_downloaded()

                # Assert: Should attempt to download default model
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert "ollama" in call_args
                assert "pull" in call_args
                assert "llama3.1:8b" in call_args

    def test_ensure_models_downloaded_returns_ready_when_models_exist(self):
        """Test that no download occurs when models already exist."""
        # Arrange: Mock manager with models already available
        with patch("clinical_analytics.ui.ollama_init.get_ollama_manager") as mock_manager_fn:
            mock_manager = MagicMock()
            mock_manager_fn.return_value = mock_manager

            mock_manager.get_status.return_value = {
                "installed": True,
                "running": True,
                "ready": True,
                "available_models": ["llama3.1:8b"],
                "default_model": "llama3.1:8b",
            }

            with patch("clinical_analytics.ui.ollama_init.subprocess.run") as mock_run:
                # Act
                from clinical_analytics.ui.ollama_init import ensure_models_downloaded

                result = ensure_models_downloaded()

                # Assert: Should not call subprocess (no download needed)
                mock_run.assert_not_called()
                assert result["ready"] is True

    def test_ensure_models_downloaded_tries_fallback_on_default_failure(self):
        """Test that fallback model is tried if default fails."""
        # Arrange
        with patch("clinical_analytics.ui.ollama_init.get_ollama_manager") as mock_manager_fn:
            mock_manager = MagicMock()
            mock_manager_fn.return_value = mock_manager

            mock_manager.get_status.return_value = {
                "installed": True,
                "running": True,
                "ready": False,
                "available_models": [],
                "default_model": "llama3.1:8b",
                "fallback_model": "llama3.2:3b",
            }
            mock_manager.default_model = "llama3.1:8b"
            mock_manager.fallback_model = "llama3.2:3b"

            with patch("clinical_analytics.ui.ollama_init.subprocess.run") as mock_run:
                # First call raises exception (default fails), second succeeds (fallback)
                from subprocess import CalledProcessError

                mock_run.side_effect = [
                    CalledProcessError(1, ["ollama", "pull", "llama3.1:8b"]),  # Default fails
                    MagicMock(returncode=0),  # Fallback succeeds
                ]

                # Act
                from clinical_analytics.ui.ollama_init import ensure_models_downloaded

                ensure_models_downloaded()

                # Assert: Should try both models
                assert mock_run.call_count == 2


class TestOllamaInitializationStatus:
    """Test Ollama initialization status messages."""

    def test_initialize_ollama_shows_download_in_progress_message(self):
        """Test that initialization shows helpful message during download."""
        # Arrange
        with patch("clinical_analytics.ui.ollama_init.get_ollama_manager") as mock_manager_fn:
            mock_manager = MagicMock()
            mock_manager_fn.return_value = mock_manager

            mock_manager.get_status.return_value = {
                "installed": True,
                "running": True,
                "ready": False,
                "available_models": [],
                "default_model": "llama3.1:8b",
            }

            # Act
            from clinical_analytics.ui.ollama_init import initialize_ollama

            result = initialize_ollama()

            # Assert: Should indicate models need to be downloaded
            assert not result["ready"]
            assert "download" in result["message"].lower() or "model" in result["message"].lower()

    def test_initialize_ollama_returns_ready_when_models_available(self):
        """Test that initialization returns ready status when models exist."""
        # Arrange
        with patch("clinical_analytics.ui.ollama_init.get_ollama_manager") as mock_manager_fn:
            mock_manager = MagicMock()
            mock_manager_fn.return_value = mock_manager

            mock_manager.get_status.return_value = {
                "installed": True,
                "running": True,
                "ready": True,
                "available_models": ["llama3.1:8b"],
            }

            # Act
            from clinical_analytics.ui.ollama_init import initialize_ollama

            result = initialize_ollama()

            # Assert
            assert result["ready"] is True
            assert "ready" in result["message"].lower()

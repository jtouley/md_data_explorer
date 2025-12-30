"""
Ollama Manager - Self-contained LLM service management (ADR003 Phase 0).

Similar to how DuckDB is self-contained, this module manages Ollama initialization
and availability checking. Provides lazy initialization and graceful degradation.
"""

import subprocess
from typing import Any

import structlog

from clinical_analytics.core.nl_query_config import (
    OLLAMA_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)

logger = structlog.get_logger()


class OllamaManager:
    """
    Self-contained manager for Ollama LLM service.

    Handles:
    - Service availability checking
    - Model availability verification
    - Graceful degradation when unavailable
    - Auto-detection of Ollama installation

    Similar to DuckDB's self-contained approach - checks availability
    on first use and provides clear feedback.
    """

    _instance: "OllamaManager | None" = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern - one manager per application."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Ollama manager (lazy - only checks when needed)."""
        if self._initialized:
            return

        self.base_url = OLLAMA_BASE_URL
        self.default_model = OLLAMA_DEFAULT_MODEL
        self.fallback_model = OLLAMA_FALLBACK_MODEL
        self.timeout = OLLAMA_TIMEOUT_SECONDS

        # Lazy state (checked on first use)
        self._service_available: bool | None = None
        self._available_models: list[str] | None = None
        self._ollama_installed: bool | None = None

        self._initialized = True
        logger.debug("OllamaManager initialized", base_url=self.base_url)

    def is_service_running(self) -> bool:
        """
        Check if Ollama service is running.

        Returns:
            True if service is reachable, False otherwise
        """
        if self._service_available is not None:
            return self._service_available

        from clinical_analytics.core.llm_client import OllamaClient

        client = OllamaClient(base_url=self.base_url, timeout=self.timeout)
        self._service_available = client.is_available()

        if self._service_available:
            logger.info("ollama_service_detected", base_url=self.base_url)
        else:
            logger.debug("ollama_service_not_running", base_url=self.base_url)

        return self._service_available

    def is_ollama_installed(self) -> bool:
        """
        Check if Ollama binary is installed (even if not running).

        Returns:
            True if Ollama binary found, False otherwise
        """
        if self._ollama_installed is not None:
            return self._ollama_installed

        try:
            result = subprocess.run(
                ["ollama", "version"],
                capture_output=True,
                timeout=2.0,
                check=False,
            )
            self._ollama_installed = result.returncode == 0

            if self._ollama_installed:
                logger.info("ollama_binary_detected")
            else:
                logger.debug("ollama_binary_not_found")

        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._ollama_installed = False
            logger.debug("ollama_binary_not_found")

        return self._ollama_installed

    def get_available_models(self) -> list[str]:
        """
        Get list of available models from Ollama service.

        Returns:
            List of model names, empty list if service unavailable
        """
        if not self.is_service_running():
            return []

        if self._available_models is not None:
            return self._available_models

        from clinical_analytics.core.llm_client import OllamaClient

        try:
            import requests

            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                models_data = response.json()
                self._available_models = [m["name"] for m in models_data.get("models", [])]
                logger.debug("ollama_models_loaded", count=len(self._available_models))
                return self._available_models
        except Exception as e:
            logger.warning("ollama_models_fetch_failed", error=str(e))

        self._available_models = []
        return []

    def get_client(self) -> Any:
        """
        Get OllamaClient instance if service is available.

        Returns:
            OllamaClient instance, or None if unavailable
        """
        if not self.is_service_running():
            return None

        from clinical_analytics.core.llm_client import OllamaClient

        # Try default model first, fallback if not available
        available_models = self.get_available_models()
        model = self.default_model
        if model not in available_models:
            if self.fallback_model in available_models:
                model = self.fallback_model
                logger.info("ollama_using_fallback_model", model=model)
            else:
                logger.warning(
                    "ollama_no_models_available",
                    default=self.default_model,
                    fallback=self.fallback_model,
                    available=available_models,
                )
                return None

        return OllamaClient(model=model, base_url=self.base_url, timeout=self.timeout)

    def get_status(self) -> dict[str, Any]:
        """
        Get comprehensive status of Ollama service.

        Returns:
            Dictionary with status information
        """
        installed = self.is_ollama_installed()
        running = self.is_service_running()
        models = self.get_available_models() if running else []

        return {
            "installed": installed,
            "running": running,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "fallback_model": self.fallback_model,
            "available_models": models,
            "ready": running and len(models) > 0,
        }

    def get_setup_instructions(self) -> str:
        """
        Get setup instructions if Ollama is not ready.

        Returns:
            Human-readable setup instructions
        """
        status = self.get_status()

        if status["ready"]:
            return "Ollama is ready and configured."

        instructions = []
        instructions.append("Ollama LLM service is not ready. Setup instructions:")

        if not status["installed"]:
            instructions.append("")
            instructions.append("1. Install Ollama:")
            instructions.append("   Visit https://ollama.ai and download for your platform")
            instructions.append("   Or use: curl -fsSL https://ollama.ai/install.sh | sh")

        if status["installed"] and not status["running"]:
            instructions.append("")
            instructions.append("2. Start Ollama service:")
            instructions.append("   Run: ollama serve")
            instructions.append("   (This starts the service in the background)")

        if status["running"] and not status["available_models"]:
            instructions.append("")
            instructions.append("3. Download a model:")
            instructions.append(f"   Run: ollama pull {self.default_model}")
            instructions.append(f"   Or: ollama pull {self.fallback_model}")

        return "\n".join(instructions)


def get_ollama_manager() -> OllamaManager:
    """
    Get singleton OllamaManager instance.

    Returns:
        OllamaManager instance
    """
    return OllamaManager()

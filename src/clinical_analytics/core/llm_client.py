# SPDX-License-Identifier: MIT
"""
LLM Client for local Ollama integration (ADR003 Phase 0).

Privacy-preserving: All data stays on-device, no external API calls.
"""

from typing import Any

import requests  # type: ignore
import structlog

from clinical_analytics.core import nl_query_config
from clinical_analytics.core.loopback import model_http_session, require_loopback_url

logger = structlog.get_logger()


def ollama_session() -> requests.Session:
    """The HTTP session for Ollama calls: ignores proxy settings (see core.loopback)."""
    return model_http_session()


class OllamaClient:
    """
    Client for local Ollama LLM service.

    Provides connection handling, model management, and JSON-mode generation.
    All requests timeout after 30 seconds by default for responsiveness.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (default: llama3.1:8b)
            base_url: Ollama service URL (default: http://127.0.0.1:11434)
            timeout: Request timeout in seconds (default: 30.0)

        Raises:
            NonLoopbackURLError: base_url is not on this machine and remote hosts are not allowed.
        """
        self.model = model
        self.base_url = require_loopback_url(base_url, allow_remote=nl_query_config.OLLAMA_ALLOW_REMOTE_HOST)
        self._session = ollama_session()
        self.timeout = timeout
        self._connection_checked = False
        self._is_available = False

    def is_available(self) -> bool:
        """
        Check if Ollama service is running.

        Returns:
            True if Ollama is available, False otherwise
        """
        if self._connection_checked:
            return self._is_available

        self._is_available = self._check_connection()
        self._connection_checked = True
        return self._is_available

    def _check_connection(self) -> bool:
        """
        Internal method to check Ollama service.

        Returns:
            True if service is reachable, False otherwise
        """
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=self.timeout, allow_redirects=False)
            return bool(response.status_code == 200)
        except (requests.RequestException, ConnectionError) as e:
            logger.warning(
                "ollama_connection_failed",
                error=str(e),
                base_url=self.base_url,
            )
            return False

    def is_model_available(self, model: str) -> bool:
        """
        Check if model is downloaded and available.

        Args:
            model: Model name to check

        Returns:
            True if model is available, False otherwise
        """
        if not self.is_available():
            return False

        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=self.timeout, allow_redirects=False)
            if response.status_code != 200:
                return False

            models_data = response.json()
            available_models = [m["name"] for m in models_data.get("models", [])]
            return model in available_models

        except (requests.RequestException, ValueError) as e:
            logger.warning(
                "ollama_model_check_failed",
                error=str(e),
                model=model,
            )
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = True,
        response_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> str | None:
        """
        Generate response with JSON mode.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            json_mode: Enable JSON mode (default: True)
            response_schema: JSON schema the server holds the reply to (overrides json_mode)
            options: Ollama sampling/context options (temperature, seed, num_ctx, ...)
            keep_alive: how long the server keeps the model loaded after this call ("30m")

        Returns:
            Generated text, or None on error/timeout
        """
        if not self.is_available():
            logger.warning("ollama_not_available", model=self.model)
            return None

        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }

            if system_prompt:
                payload["system"] = system_prompt

            if response_schema is not None:
                payload["format"] = response_schema
            elif json_mode:
                payload["format"] = "json"
            if options:
                payload["options"] = options
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive

            response = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                allow_redirects=False,
            )

            if response.status_code != 200:
                logger.warning(
                    "ollama_generate_failed",
                    status_code=response.status_code,
                    model=self.model,
                )
                return None

            result = response.json()
            response_text: str | None = result.get("response")
            return response_text

        except requests.Timeout:
            logger.warning(
                "ollama_timeout",
                timeout_seconds=self.timeout,
                model=self.model,
            )
            return None
        except (requests.RequestException, ValueError) as e:
            logger.warning(
                "ollama_generate_error",
                error=str(e),
                model=self.model,
            )
            return None

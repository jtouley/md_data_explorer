"""
LLM Client for local Ollama integration (ADR003 Phase 0).

Privacy-preserving: All data stays on-device, no external API calls.
"""

import requests
import structlog

logger = structlog.get_logger()


class OllamaClient:
    """
    Client for local Ollama LLM service.

    Provides connection handling, model management, and JSON-mode generation.
    All requests timeout after 5 seconds by default for responsiveness.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 5.0,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (default: llama3.1:8b)
            base_url: Ollama service URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 5.0)
        """
        self.model = model
        self.base_url = base_url
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
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            return response.status_code == 200
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
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
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
    ) -> str | None:
        """
        Generate response with JSON mode.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            json_mode: Enable JSON mode (default: True)

        Returns:
            Generated text, or None on error/timeout
        """
        if not self.is_available():
            logger.warning("ollama_not_available", model=self.model)
            return None

        try:
            payload: dict = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }

            if system_prompt:
                payload["system"] = system_prompt

            if json_mode:
                payload["format"] = "json"

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.warning(
                    "ollama_generate_failed",
                    status_code=response.status_code,
                    model=self.model,
                )
                return None

            result = response.json()
            return result.get("response")

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

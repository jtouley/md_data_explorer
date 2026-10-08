"""Lightweight Ollama reachability probe for GET /health (Electron + ops)."""

from __future__ import annotations

import requests  # type: ignore[import-untyped]

from clinical_analytics.core import nl_query_config

# Module-level copies so tests can patch `health_llm.OLLAMA_DEFAULT_MODEL`.
OLLAMA_BASE_URL: str = nl_query_config.OLLAMA_BASE_URL
OLLAMA_DEFAULT_MODEL: str = nl_query_config.OLLAMA_DEFAULT_MODEL

_HEALTH_TIMEOUT_S = 2.0


def get_ollama_health_snapshot() -> dict[str, object]:
    """
    Probe Ollama /api/tags without initializing OllamaManager singleton state.

    Returns:
        Dict with ollama_base_url, ollama_default_model, ollama_reachable,
        ollama_default_model_available, ollama_models (names).
    """
    base_url = OLLAMA_BASE_URL.rstrip("/")
    snapshot: dict[str, object] = {
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_default_model": OLLAMA_DEFAULT_MODEL,
        "ollama_reachable": False,
        "ollama_default_model_available": False,
        "ollama_models": [],
    }
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=_HEALTH_TIMEOUT_S)
    except requests.RequestException:
        return snapshot

    if response.status_code != 200:
        return snapshot

    try:
        payload = response.json()
    except ValueError:
        return snapshot

    models_raw = payload.get("models", [])
    if not isinstance(models_raw, list):
        return snapshot

    names: list[str] = []
    for item in models_raw:
        if isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
            names.append(item["name"])

    snapshot["ollama_reachable"] = True
    snapshot["ollama_models"] = names
    snapshot["ollama_default_model_available"] = OLLAMA_DEFAULT_MODEL in names
    return snapshot

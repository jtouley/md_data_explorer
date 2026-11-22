# SPDX-License-Identifier: MIT
"""The local model call of the question path: Ollama on this machine, reply held to a JSON schema.

Patient data never leaves the machine: the prompt carries column names, category values and the
data dictionary, so the server must be on a loopback address (``core.loopback``). Any other
address is refused here, whatever the configuration says; the call goes through OllamaClient's
proxy-free session.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Final, TypeAlias

import structlog

from clinical_analytics.core import nl_query_config
from clinical_analytics.core.llm_client import OllamaClient
from clinical_analytics.core.llm_json import parse_json_response
from clinical_analytics.core.loopback import is_loopback_url

logger = structlog.get_logger()

# Room for the table description, the examples and a retry's feedback; the server's default
# (4096 tokens) silently cuts the start of a longer prompt, which is the table's columns.
NUM_CTX: Final = 8192
MAX_REPLY_TOKENS: Final = 600
AVAILABILITY_TIMEOUT_S: Final = 3.0
# (system prompt, user prompt, reply schema) -> the parsed reply object, or None when the model
# could not be reached or did not answer with JSON in time.
ModelFn: TypeAlias = Callable[[str, str, dict[str, Any]], Any]


def _client() -> OllamaClient | None:
    """The app's Ollama client at the configured address, or None when that is not this machine.
    The question path refuses another host even when ``ollama_allow_remote_host`` is set: its
    prompt carries the table's columns, category values and data dictionary."""
    url = nl_query_config.OLLAMA_BASE_URL
    if not is_loopback_url(url):
        logger.warning("model_query_url_not_local")
        return None
    return OllamaClient(
        model=nl_query_config.OLLAMA_DEFAULT_MODEL, base_url=url, timeout=nl_query_config.MODEL_QUERY_TIMEOUT_S
    )


def model_available() -> bool:
    """Whether a local model server answers at the configured loopback address."""
    client = _client()
    if client is None:
        return False
    client.timeout = AVAILABILITY_TIMEOUT_S
    return client.is_available()


def ollama_model(system: str, user: str, schema: dict[str, Any]) -> Any:
    """One reply from the local Ollama server, as parsed JSON (None: unreachable, timed out, or
    not JSON). Deterministic settings: temperature 0 and a fixed seed, so a question asked twice
    gets the same query."""
    client = _client()
    if client is None:
        return None
    started = time.perf_counter()
    raw = client.generate(
        prompt=user,
        system_prompt=system,
        response_schema=schema,
        options={"temperature": 0, "seed": 7, "num_ctx": NUM_CTX, "num_predict": MAX_REPLY_TOKENS},
    )
    parsed = parse_json_response(raw) if raw is not None else None
    logger.info(
        "model_query_call",
        model=client.model,
        latency_ms=round((time.perf_counter() - started) * 1000),
        answered=raw is not None,
        parsed=parsed is not None,
    )
    if raw is not None and parsed is None:
        # It answered, but not with usable JSON (cut short, or text around it): an attempt that
        # failed and is fed back, not a model that is down.
        return {}
    return parsed

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic>=1.0", "structlog>=24.1"]
# ///
"""Dev tool: answer the app's Ollama calls with Claude Haiku, for testing the LLM tier.

NOT a production surface. The app promises that the LLM is local; this shim sends every
prompt (column names, category labels, summary results) to the Anthropic API instead.
Use it only with data you may send to a third party: synthetic cohorts or public data,
never patient files.

It serves the two Ollama endpoints the app calls (``core/llm_client.py``,
``core/ollama_manager.py``, ``api/health_llm.py``):

* ``GET  /api/tags``      lists one model, the Claude model ID.
* ``POST /api/generate``  takes ``{model, prompt, system?, format?, stream: false}`` and
  returns ``{"model", "response", "done": true}``.

Run it (credentials come from ``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile)::

    uv run --script scripts/ollama_claude_shim.py --port 11435
    OLLAMA_BASE_URL=http://127.0.0.1:11435 OLLAMA_DEFAULT_MODEL=claude-haiku-5-5 \\
        OLLAMA_TIMEOUT_SECONDS=60 make run-api

Point the app at it only through ``OLLAMA_BASE_URL`` on localhost; nothing here changes
the app's defaults.
"""

from __future__ import annotations

import argparse
import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final

import anthropic
import structlog

logger = structlog.get_logger("ollama_claude_shim")

DEFAULT_MODEL: Final = "claude-haiku-5-5"
DEFAULT_PORT: Final = 11435
MAX_TOKENS: Final = 16000
MAX_BODY_BYTES: Final = 2_000_000
JSON_ONLY_INSTRUCTION: Final = "Reply with exactly one JSON object and nothing else: no prose, no Markdown code fences."
_FENCE: Final = re.compile(r"^```(?:json)?\s*|\s*```$")


class GenerateRequestError(ValueError):
    """The /api/generate body is not the shape the app sends."""


def parse_generate_body(raw: bytes) -> tuple[str, str | None, bool]:
    """Validate an Ollama /api/generate body; return (prompt, system, json_mode)."""
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GenerateRequestError(f"body is not JSON: {e.msg}") from e
    if not isinstance(body, dict):
        raise GenerateRequestError("body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise GenerateRequestError("'prompt' must be a non-empty string")
    if body.get("stream", False):
        raise GenerateRequestError("streaming is not supported; send stream: false")
    system = body.get("system")
    if system is not None and not isinstance(system, str):
        raise GenerateRequestError("'system' must be a string")
    return prompt, system, body.get("format") == "json"


def strip_json_fences(text: str) -> str:
    """Remove a Markdown code fence around a JSON reply, if the model added one."""
    return _FENCE.sub("", text.strip())


class ClaudeBackend:
    """One Messages API call per Ollama generate call."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.client = anthropic.Anthropic()

    def generate(self, prompt: str, system: str | None, json_mode: bool) -> str:
        system_parts = [p for p in (system, JSON_ONLY_INSTRUCTION if json_mode else None) if p]
        kwargs: dict[str, Any] = {}
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined the request (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"reply cut off at max_tokens={MAX_TOKENS}")
        text = "".join(block.text for block in response.content if block.type == "text")
        logger.info(
            "claude_generate",
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )
        return strip_json_fences(text) if json_mode else text


def make_handler(backend: ClaudeBackend) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            logger.debug("http", line=format % args)

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            if self.path.rstrip("/") == "/api/tags":
                self._send(HTTPStatus.OK, {"models": [{"name": backend.model, "model": backend.model}]})
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            if self.path.rstrip("/") != "/api/generate":
                self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(HTTPStatus.BAD_REQUEST, {"error": f"body must be 1..{MAX_BODY_BYTES} bytes"})
                return
            try:
                prompt, system, json_mode = parse_generate_body(self.rfile.read(length))
                text = backend.generate(prompt, system, json_mode)
            except GenerateRequestError as e:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            except anthropic.RateLimitError as e:
                logger.warning("claude_rate_limited", status=e.status_code)
                self._send(HTTPStatus.TOO_MANY_REQUESTS, {"error": "rate limited by the Anthropic API"})
                return
            except anthropic.APIStatusError as e:
                logger.error("claude_api_error", status=e.status_code, error_type=type(e).__name__)
                self._send(HTTPStatus.BAD_GATEWAY, {"error": f"Anthropic API returned {e.status_code}"})
                return
            except anthropic.APIConnectionError as e:
                logger.error("claude_unreachable", error_type=type(e).__name__)
                self._send(HTTPStatus.BAD_GATEWAY, {"error": "could not reach the Anthropic API"})
                return
            except RuntimeError as e:
                logger.warning("claude_no_usable_reply", reason=str(e))
                self._send(HTTPStatus.BAD_GATEWAY, {"error": str(e)})
                return
            self._send(HTTPStatus.OK, {"model": backend.model, "response": text, "done": True})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model ID (default {DEFAULT_MODEL})")
    args = parser.parse_args()
    backend = ClaudeBackend(args.model)
    # Loopback only: this forwards prompts to a third party and must not be reachable from the network.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(backend))
    logger.warning("shim_started_prompts_leave_this_machine", port=args.port, model=args.model)
    server.serve_forever()


if __name__ == "__main__":
    main()

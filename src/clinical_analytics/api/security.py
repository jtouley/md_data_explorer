# SPDX-License-Identifier: MIT
"""Local API access control: a per-launch bearer token and a Host allow-list.

The API listens on 127.0.0.1, but any process on the machine, and any web page the clinician
opens (via DNS rebinding or a simple cross-site request), can reach a loopback port. Two layers
close that:

* ``BearerTokenMiddleware``: when ``CLINICAL_ANALYTICS_API_TOKEN`` is set, every request except
  ``GET /health`` and CORS preflights must carry ``Authorization: Bearer <token>``. The launcher
  (scripts/run_app.sh, or Electron main when it is started on its own) generates a fresh random
  token per launch and hands it to this process and to the renderer's preload script. The query
  stream is read with ``fetch`` streaming, not ``EventSource``, so it sends the same header and
  the token never appears in a URL or an access log.
* ``TrustedHostMiddleware`` (Starlette) rejects requests whose ``Host`` header is not loopback,
  which defeats DNS rebinding.

Running without a token needs an explicit opt-out, ``CLINICAL_ANALYTICS_DEV_NO_AUTH=1`` (set by
``make run-api`` and the test suite): with neither the token nor the opt-out the API refuses to
start, and if it is started anyway (lifespan skipped) every request except ``/health`` gets 503.
``/health`` reports ``auth_required`` so the Electron app can refuse a backend that does not
check tokens. It also proves the server knows the token without revealing it: the client sends a
random ``nonce`` and the server returns ``token_proof = HMAC-SHA256(token, context + nonce)``. A
program squatting on the port cannot compute it, so the app and the launcher refuse to send the
token or any data to it (``python -m clinical_analytics.api.security verify-health URL``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import structlog

logger = structlog.get_logger()

API_TOKEN_ENV = "CLINICAL_ANALYTICS_API_TOKEN"
DEV_NO_AUTH_ENV = "CLINICAL_ANALYTICS_DEV_NO_AUTH"
ALLOWED_HOSTS_ENV = "CLINICAL_ANALYTICS_ALLOWED_HOSTS"
DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost")
MIN_TOKEN_LENGTH = 32
# Paths reachable without a token: the launcher polls /health before it has handed out the token.
PUBLIC_PATHS: frozenset[str] = frozenset({"/health"})

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


HEALTH_PROOF_CONTEXT = b"clinical-analytics-health-v1:"
NONCE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


def token_proof(token: str, nonce: str) -> str:
    """HMAC-SHA256(token, context + nonce) as hex: shows knowledge of the token, not the token."""
    return hmac.new(token.encode("utf-8"), HEALTH_PROOF_CONTEXT + nonce.encode("ascii"), hashlib.sha256).hexdigest()


class ApiTokenConfigError(ValueError):
    """The configured API token is unusable (too short), or missing without the dev opt-out."""


def configured_api_token() -> str | None:
    """Token from the environment, or None in dev mode. Rejects weak tokens loudly."""
    token = os.environ.get(API_TOKEN_ENV, "").strip()
    if not token:
        return None
    if len(token) < MIN_TOKEN_LENGTH:
        raise ApiTokenConfigError(f"{API_TOKEN_ENV} must be at least {MIN_TOKEN_LENGTH} characters (got {len(token)})")
    return token


def dev_no_auth() -> bool:
    """True only when the developer explicitly opted out of the token check."""
    return os.environ.get(DEV_NO_AUTH_ENV, "").strip() == "1"


def auth_required() -> bool:
    """Whether requests must carry the token (False only in explicit dev mode)."""
    return configured_api_token() is not None or not dev_no_auth()


def allowed_hosts() -> list[str]:
    """Host header values accepted; override with a comma list in CLINICAL_ANALYTICS_ALLOWED_HOSTS."""
    raw = os.environ.get(ALLOWED_HOSTS_ENV, "")
    hosts = [h.strip() for h in raw.split(",") if h.strip()]
    return hosts or list(DEFAULT_ALLOWED_HOSTS)


def log_auth_mode() -> None:
    """Startup check: a token, or the explicit dev opt-out. Fails fast otherwise."""
    if configured_api_token() is not None:
        logger.info("api_auth_enabled", scheme="bearer")
        return
    if not dev_no_auth():
        raise ApiTokenConfigError(
            f"{API_TOKEN_ENV} is not set. Start the app with scripts/run_app.sh, or set "
            f"{DEV_NO_AUTH_ENV}=1 to run an unauthenticated development server (make run-api)."
        )
    logger.warning(
        "api_auth_disabled",
        reason=f"{DEV_NO_AUTH_ENV}=1 (development only)",
        action="Never use this with real patient data",
    )


def _bearer_token(scope: Scope) -> str | None:
    headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
    for name, value in headers:
        if name == b"authorization":
            scheme, _, credential = value.decode("latin-1").partition(" ")
            if scheme.lower() == "bearer" and credential.strip():
                return credential.strip()
            return None
    return None


def _is_cors_preflight(scope: Scope) -> bool:
    if scope.get("method") != "OPTIONS":
        return False
    names = {name for name, _ in scope.get("headers", [])}
    return b"origin" in names and b"access-control-request-method" in names


class BearerTokenMiddleware:
    """Pure ASGI middleware (keeps streaming responses unbuffered)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Read per request so tests can toggle it; the value is fixed for a real launch.
        expected = configured_api_token()
        if scope.get("path") in PUBLIC_PATHS or _is_cors_preflight(scope):
            await self.app(scope, receive, send)
            return
        if expected is None:
            if dev_no_auth():
                await self.app(scope, receive, send)
                return
            # Fail closed: no token and no opt-out (startup check skipped or bypassed).
            await _send_json(send, 503, b'{"detail":"API token not configured; restart the app"}')
            return

        presented = _bearer_token(scope)
        if presented is not None and hmac.compare_digest(presented.encode(), expected.encode()):
            await self.app(scope, receive, send)
            return

        logger.warning(
            "api_request_unauthorized",
            path=scope.get("path"),
            method=scope.get("method"),
            reason="missing_token" if presented is None else "wrong_token",
        )
        await _send_json(
            send,
            401,
            b'{"detail":"Missing or invalid API token"}',
            [(b"www-authenticate", b'Bearer realm="clinical-analytics"')],
        )


async def _send_json(
    send: Send, status: int, body: bytes, extra_headers: list[tuple[bytes, bytes]] | None = None
) -> None:
    headers = [(b"content-type", b"application/json"), *(extra_headers or [])]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def verify_health(base_url: str, token: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Check that the server at ``base_url`` knows ``token`` (used by scripts/run_app.sh)."""
    nonce = secrets.token_urlsafe(24)
    try:
        with urllib.request.urlopen(f"{base_url}/health?nonce={nonce}", timeout=timeout_s) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, f"health check failed ({type(e).__name__})"
    if body.get("auth_required") is not True:
        return False, "server does not require the API token"
    proof = body.get("token_proof")
    if not isinstance(proof, str) or not hmac.compare_digest(proof, token_proof(token, nonce)):
        return False, "server on this port does not know this launch's token"
    return True, "ok"


def _main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "verify-health":
        sys.stderr.write("usage: python -m clinical_analytics.api.security verify-health BASE_URL\n")
        return 2
    token = configured_api_token()
    if token is None:
        sys.stderr.write(f"{API_TOKEN_ENV} is not set\n")
        return 2
    ok, reason = verify_health(argv[1], token)
    sys.stderr.write(f"{reason}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

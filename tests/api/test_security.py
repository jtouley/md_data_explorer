"""Local API access control: per-launch bearer token and loopback Host allow-list (S4)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from clinical_analytics.api.security import (
    ALLOWED_HOSTS_ENV,
    API_TOKEN_ENV,
    DEV_NO_AUTH_ENV,
    ApiTokenConfigError,
    allowed_hosts,
    configured_api_token,
    log_auth_mode,
)

TOKEN = "t" * 43
PROTECTED = "/api/datasets"


@pytest.fixture
def token_set(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(API_TOKEN_ENV, TOKEN)
    return TOKEN


def test_middleware_devModeNoToken_allowsRequestWithoutHeader(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.delenv(API_TOKEN_ENV, raising=False)
    monkeypatch.setenv(DEV_NO_AUTH_ENV, "1")

    # Act
    response = test_client.get(PROTECTED)

    # Assert
    assert response.status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token-of-the-wrong-value-xxxxxxxxxx"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": "Bearer "},
        {"Authorization": TOKEN},
    ],
    ids=["missing", "wrong", "wrong-scheme", "empty", "no-scheme"],
)
def test_middleware_tokenSetBadCredential_returns401(
    test_client: TestClient, token_set: str, headers: dict[str, str]
) -> None:
    # Arrange / Act
    response = test_client.get(PROTECTED, headers=headers)

    # Assert
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_middleware_tokenSetCorrectBearer_allowsRequest(test_client: TestClient, token_set: str) -> None:
    # Arrange
    headers = {"Authorization": f"Bearer {token_set}"}

    # Act
    response = test_client.get(PROTECTED, headers=headers)

    # Assert
    assert response.status_code == 200


@pytest.mark.parametrize("path", ["/api/queries/qry_missing/stream", "/api/openapi.json", "/api/sessions"])
def test_middleware_tokenSetNoHeader_protectsStreamDocsAndSessions(
    test_client: TestClient, token_set: str, path: str
) -> None:
    # Arrange / Act
    response = test_client.get(path)

    # Assert
    assert response.status_code == 401


def test_middleware_tokenSetHealth_isPublicForLauncherPolling(test_client: TestClient, token_set: str) -> None:
    # Arrange / Act
    response = test_client.get("/health")

    # Assert
    assert response.status_code == 200


def test_middleware_tokenSetCorsPreflight_answeredWithoutToken(test_client: TestClient, token_set: str) -> None:
    # Arrange
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    }

    # Act
    response = test_client.options(PROTECTED, headers=headers)

    # Assert
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


@pytest.mark.parametrize("host", ["evil.example", "attacker.localhost.example", "192.168.1.5"])
def test_trustedHost_nonLoopbackHostHeader_returns400(test_client: TestClient, host: str) -> None:
    # Arrange / Act
    response = test_client.get("/health", headers={"Host": host})

    # Assert
    assert response.status_code == 400


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000"])
def test_trustedHost_loopbackHostHeader_allowed(test_client: TestClient, host: str) -> None:
    # Arrange / Act
    response = test_client.get("/health", headers={"Host": host})

    # Assert
    assert response.status_code == 200


@pytest.mark.parametrize(("raw", "expected"), [("", None), ("   ", None), (TOKEN, TOKEN), (f"  {TOKEN} ", TOKEN)])
def test_configuredApiToken_envValues_returnsTokenOrNone(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str | None
) -> None:
    # Arrange
    monkeypatch.setenv(API_TOKEN_ENV, raw)

    # Act
    token = configured_api_token()

    # Assert
    assert token == expected


def test_configuredApiToken_tooShort_raisesApiTokenConfigError(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv(API_TOKEN_ENV, "x" * 31)

    # Act / Assert
    with pytest.raises(ApiTokenConfigError, match=API_TOKEN_ENV):
        configured_api_token()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", ["127.0.0.1", "localhost"]), (" a , b ,", ["a", "b"]), (",", ["127.0.0.1", "localhost"])],
)
def test_allowedHosts_envValues_parsesListWithLoopbackDefault(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]
) -> None:
    # Arrange
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, raw)

    # Act
    hosts = allowed_hosts()

    # Assert
    assert hosts == expected


@pytest.mark.parametrize("opt_out", [None, "", "0", "true", "yes"])
def test_middleware_noTokenNoExplicitOptOut_failsClosed503(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch, opt_out: str | None
) -> None:
    # Arrange
    monkeypatch.delenv(API_TOKEN_ENV, raising=False)
    if opt_out is None:
        monkeypatch.delenv(DEV_NO_AUTH_ENV, raising=False)
    else:
        monkeypatch.setenv(DEV_NO_AUTH_ENV, opt_out)

    # Act
    response = test_client.get(PROTECTED)
    health = test_client.get("/health")

    # Assert
    assert response.status_code == 503
    assert health.status_code == 200
    assert health.json()["auth_required"] is True


def test_logAuthMode_noTokenNoOptOut_refusesToStart(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv(API_TOKEN_ENV, raising=False)
    monkeypatch.delenv(DEV_NO_AUTH_ENV, raising=False)

    # Act / Assert
    with pytest.raises(ApiTokenConfigError, match=DEV_NO_AUTH_ENV):
        log_auth_mode()


@pytest.mark.parametrize(
    ("token", "opt_out", "expected"),
    [(TOKEN, None, True), (TOKEN, "1", True), (None, "1", False)],
)
def test_health_authRequiredFlag_reflectsMode(
    test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    token: str | None,
    opt_out: str | None,
    expected: bool,
) -> None:
    # Arrange
    for name, value in ((API_TOKEN_ENV, token), (DEV_NO_AUTH_ENV, opt_out)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    # Act
    body = test_client.get("/health").json()

    # Assert
    assert body["auth_required"] is expected


def test_health_nonceWithToken_returnsHmacProofNotToken(test_client: TestClient, token_set: str) -> None:
    # Arrange
    from clinical_analytics.api.security import token_proof

    nonce = "n" * 24

    # Act
    body = test_client.get("/health", params={"nonce": nonce}).json()

    # Assert
    assert body["token_proof"] == token_proof(token_set, nonce)
    assert token_set not in str(body)


@pytest.mark.parametrize("nonce", ["short", "x" * 129, "bad nonce with spaces!!"])
def test_health_malformedNonce_rejected422(test_client: TestClient, token_set: str, nonce: str) -> None:
    # Arrange / Act
    response = test_client.get("/health", params={"nonce": nonce})

    # Assert
    assert response.status_code == 422


def test_health_noNonceOrDevMode_noProof(test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv(API_TOKEN_ENV, raising=False)
    monkeypatch.setenv(DEV_NO_AUTH_ENV, "1")

    # Act
    dev = test_client.get("/health", params={"nonce": "n" * 24}).json()
    monkeypatch.setenv(API_TOKEN_ENV, TOKEN)
    no_nonce = test_client.get("/health").json()

    # Assert
    assert dev["token_proof"] is None
    assert no_nonce["token_proof"] is None


def _squatter(body: dict[str, object]) -> tuple[str, object]:
    """A local HTTP server that answers /health like the API but does not know the token."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            payload = json.dumps(body).encode()
            self.send_response(200 if self.path.startswith("/health") else 401)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


@pytest.mark.parametrize(
    "body",
    [
        {"status": "healthy", "auth_required": True},
        {"status": "healthy", "auth_required": True, "token_proof": "0" * 64},
        {"status": "healthy", "auth_required": False},
    ],
    ids=["no-proof", "wrong-proof", "no-auth"],
)
def test_verifyHealth_squatterOnPort_refused(body: dict[str, object]) -> None:
    # Arrange
    from clinical_analytics.api.security import verify_health

    url, server = _squatter(body)

    # Act
    try:
        ok, reason = verify_health(url, TOKEN)
    finally:
        server.shutdown()  # type: ignore[attr-defined]

    # Assert
    assert ok is False
    assert reason


def _recording_server(status: int, location: str | None = None) -> tuple[str, list[str], object]:
    """A local server that records each request line and answers ``status`` (with a Location)."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            seen.append(self.requestline)
            self.send_response(status)
            if location is not None:
                self.send_header("location", location)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", seen, server


def test_verifyHealth_proxyEnvSet_neverUsesProxy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: a proxy configured in every spelling, and the local "API" that should answer
    from clinical_analytics.api.security import verify_health

    proxy_url, proxy_seen, proxy = _recording_server(200)
    api_url, api_seen, api = _recording_server(200)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, proxy_url)

    # Act
    try:
        ok, _reason = verify_health(api_url, TOKEN)
    finally:
        proxy.shutdown()  # type: ignore[attr-defined]
        api.shutdown()  # type: ignore[attr-defined]

    # Assert: the local server got the probe; the proxy got nothing
    assert ok is False  # the stand-in API does not know the token
    assert proxy_seen == []
    assert len(api_seen) == 1 and api_seen[0].startswith("GET /health?nonce=")


def test_verifyHealth_redirect_notFollowed() -> None:
    # Arrange: the "API" redirects to another server, which must never be contacted
    from clinical_analytics.api.security import verify_health

    target_url, target_seen, target = _recording_server(200)
    api_url, _api_seen, api = _recording_server(302, location=f"{target_url}/health")

    # Act
    try:
        ok, reason = verify_health(api_url, TOKEN)
    finally:
        target.shutdown()  # type: ignore[attr-defined]
        api.shutdown()  # type: ignore[attr-defined]

    # Assert
    assert ok is False
    assert "HTTPError" in reason
    assert target_seen == []


def test_verifyHealth_realBackendWithSameToken_accepted(real_server: str, token_set: str) -> None:
    # Arrange
    from clinical_analytics.api.security import verify_health

    # Act
    ok, reason = verify_health(real_server, token_set)
    wrong_ok, _ = verify_health(real_server, "w" * 43)

    # Assert
    assert ok is True, reason
    assert wrong_ok is False

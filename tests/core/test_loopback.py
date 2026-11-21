# SPDX-License-Identifier: MIT
"""Model server calls stay on this machine: loopback URLs only, and proxy settings are ignored.

A recording HTTP proxy and a fake Ollama both run on 127.0.0.1. With HTTP_PROXY, http_proxy and
ALL_PROXY pointing at the proxy (and NO_PROXY unset), every model call must reach the fake Ollama
directly and the proxy must see nothing.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from clinical_analytics.core.config_loader import load_nl_query_config
from clinical_analytics.core.loopback import (
    NonLoopbackURLError,
    is_loopback_url,
    model_http_session,
    require_loopback_url,
)

FAKE_MODEL = "synthetic-model:1b"


@dataclass
class _Recorder:
    """Request lines one local server received."""

    lines: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)


@dataclass
class _LocalServer:
    url: str
    recorder: _Recorder


def _serve(handler_body: bytes, status: int, location: str | None = None) -> Iterator[_LocalServer]:
    recorder = _Recorder()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            recorder.add(self.requestline)
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if location is not None:
                self.send_header("Location", location)
            self.send_header("Content-Length", str(len(handler_body)))
            self.end_headers()
            self.wfile.write(handler_body)

        do_GET = _answer  # noqa: N815 - http.server's naming
        do_POST = _answer  # noqa: N815
        do_CONNECT = _answer  # noqa: N815

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - base signature
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _LocalServer(url=f"http://127.0.0.1:{server.server_address[1]}", recorder=recorder)
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def fake_ollama() -> Iterator[_LocalServer]:
    """A stand-in Ollama: /api/tags lists FAKE_MODEL, /api/generate answers JSON."""
    body = json.dumps({"models": [{"name": FAKE_MODEL}], "response": '{"ok": true}'}).encode()
    yield from _serve(body, 200)


@pytest.fixture
def recording_proxy(monkeypatch: pytest.MonkeyPatch) -> Iterator[_LocalServer]:
    """A proxy that answers with a forged reply; HTTP_PROXY, http_proxy and ALL_PROXY point at it."""
    forged = json.dumps({"models": [{"name": FAKE_MODEL}], "response": '{"forged": true}'}).encode()
    for server in _serve(forged, 200):
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy"):
            monkeypatch.setenv(name, server.url)
        yield server


class TestModelCallsIgnoreProxy:
    def test_ollama_client_generate_withProxyEnv_reachesOllamaDirectly(self, recording_proxy, fake_ollama):
        # Arrange
        from clinical_analytics.core.llm_client import OllamaClient

        client = OllamaClient(model=FAKE_MODEL, base_url=fake_ollama.url, timeout=5.0)

        # Act
        available = client.is_model_available(FAKE_MODEL)
        reply = client.generate("synthetic question about column_a")

        # Assert
        assert available is True
        assert reply == '{"ok": true}'
        assert recording_proxy.recorder.lines == []
        assert any(line.startswith("POST /api/generate") for line in fake_ollama.recorder.lines)

    def test_health_snapshot_withProxyEnv_reachesOllamaDirectly(self, recording_proxy, fake_ollama, monkeypatch):
        # Arrange
        from clinical_analytics.api import health_llm

        monkeypatch.setattr(health_llm, "OLLAMA_BASE_URL", fake_ollama.url)
        monkeypatch.setattr(health_llm, "OLLAMA_DEFAULT_MODEL", FAKE_MODEL)

        # Act
        snapshot = health_llm.get_ollama_health_snapshot()

        # Assert
        assert snapshot["ollama_reachable"] is True
        assert snapshot["ollama_default_model_available"] is True
        assert recording_proxy.recorder.lines == []
        assert fake_ollama.recorder.lines == ["GET /api/tags HTTP/1.1"]

    def test_ollama_manager_calls_withProxyEnv_reachOllamaDirectly(self, recording_proxy, fake_ollama, monkeypatch):
        # Arrange
        from clinical_analytics.core.ollama_manager import OllamaManager

        monkeypatch.setattr(OllamaManager, "_instance", None)
        monkeypatch.setattr(OllamaManager, "_initialized", False)
        manager = OllamaManager()
        manager.base_url = fake_ollama.url
        manager.default_model = FAKE_MODEL
        manager.timeout = 5.0

        # Act
        running = manager.is_service_running()
        models = manager.get_available_models()
        client = manager.get_client()

        # Assert
        assert running is True
        assert models == [FAKE_MODEL]
        assert client is not None
        assert recording_proxy.recorder.lines == []
        assert len(fake_ollama.recorder.lines) == 2

    @pytest.mark.parametrize("allow_redirects", [False, True])
    def test_model_http_session_redirect_raisesAndIsNotFollowed(self, recording_proxy, allow_redirects):
        # Arrange: a "model server" that redirects; the redirect target must never be contacted
        import requests

        for target in _serve(b"{}", 200):
            for redirecting in _serve(b"{}", 307, location=f"{target.url}/elsewhere"):
                # Act / Assert: a 3xx is an error (a RequestException every caller handles)
                with model_http_session() as session, pytest.raises(requests.TooManyRedirects):
                    session.get(f"{redirecting.url}/api/tags", timeout=5.0, allow_redirects=allow_redirects)
                assert redirecting.recorder.lines == ["GET /api/tags HTTP/1.1"]
                assert target.recorder.lines == []
                assert recording_proxy.recorder.lines == []

    def test_ollama_client_generate_redirectingServer_returnsNone(self, recording_proxy):
        # Arrange
        from clinical_analytics.core.llm_client import OllamaClient

        for redirecting in _serve(b"{}", 307, location="http://127.0.0.1:9/elsewhere"):
            client = OllamaClient(model=FAKE_MODEL, base_url=redirecting.url, timeout=5.0)

            # Act
            reply = client.generate("synthetic question")

            # Assert
            assert reply is None
            assert recording_proxy.recorder.lines == []

    def test_model_http_session_trustEnv_isOff(self):
        # Arrange / Act
        session = model_http_session()

        # Assert: trust_env=False also turns off the OS proxy settings requests reads on Windows/macOS
        assert session.trust_env is False
        assert session.proxies == {}
        assert session.max_redirects == 0


class TestIsLoopbackUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434",
            "http://LOCALHOST:11434",
            "https://localhost",
            "http://127.0.0.1:11434",
            "http://127.1.2.3:11434",
            "http://[::1]:11434",
            "http://127.0.0.1:11434/",
        ],
    )
    def test_is_loopback_url_localAddresses_returnsTrue(self, url):
        # Arrange / Act
        result = is_loopback_url(url)

        # Assert
        assert result is True

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "localhost:11434",
            "ftp://127.0.0.1",
            "file:///etc/passwd",
            "http://",
            "http://10.0.0.5:11434",
            "http://192.168.1.10:11434",
            "http://0.0.0.0:11434",
            "http://ollama.example.test:11434",
            "http://localhost.example.test:11434",
            "http://127.0.0.1.nip.io:11434",
            "http://localhost@evil.example.test:11434",
            "http://[::ffff:127.0.0.1]:11434",
            "http://[fe80::1%25eth0]:11434",
            "http://[::]:11434",
            "http://127.0.0.1:notaport",
            "http://[::1",
            # parser differentials (urllib vs urllib3 read different hosts): refused outright
            "http://127.0.0.1:11434\\@evil.example.test/",
            "http://127.0.0.1#@evil.example.test",
            "http://127.0.0.1?@evil.example.test",
            "http://user@127.0.0.1:11434",
            "http://localhost%2e:11434",
            " http://127.0.0.1:11434",
            "http://127.0.0.1:11434\n",
            "http://127.0.0.1\t:11434",
            "http://127.0.0.1.:11434",
            "http://2130706433:11434",
            "http://0x7f.1:11434",
        ],
    )
    def test_is_loopback_url_remoteOrMalformed_returnsFalse(self, url):
        # Arrange / Act
        result = is_loopback_url(url)

        # Assert
        assert result is False

    def test_require_loopback_url_remoteWithoutOptIn_raises(self):
        # Arrange
        url = "http://10.0.0.5:11434"

        # Act / Assert
        with pytest.raises(NonLoopbackURLError, match="ollama_allow_remote_host"):
            require_loopback_url(url, allow_remote=False)

    @pytest.mark.parametrize("opt_in", ["true", 1, "yes", object()])
    def test_require_loopback_url_truthyNonBoolOptIn_stillRefused(self, opt_in):
        # Arrange
        url = "http://10.0.0.5:11434"

        # Act / Assert
        with pytest.raises(NonLoopbackURLError):
            require_loopback_url(url, allow_remote=opt_in)  # type: ignore[arg-type]

    def test_require_loopback_url_remoteWithOptIn_returnsUrl(self):
        # Arrange
        url = "http://10.0.0.5:11434"

        # Act
        result = require_loopback_url(url, allow_remote=True)

        # Assert
        assert result == url


class TestConfigRefusesRemoteModelServer:
    @staticmethod
    def _write(tmp_path: Path, text: str) -> Path:
        path = tmp_path / "nl_query.yaml"
        path.write_text(text)
        return path

    @pytest.mark.parametrize("url", ["http://10.0.0.5:11434", "http://ollama.example.test:11434"])
    def test_load_nl_query_config_remoteYamlUrl_raises(self, tmp_path, monkeypatch, url):
        # Arrange
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_ALLOW_REMOTE_HOST", raising=False)
        path = self._write(tmp_path, f'ollama_base_url: "{url}"\n')

        # Act / Assert
        with pytest.raises(NonLoopbackURLError):
            load_nl_query_config(path)

    def test_load_nl_query_config_remoteEnvUrl_raises(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434")
        monkeypatch.delenv("OLLAMA_ALLOW_REMOTE_HOST", raising=False)
        path = self._write(tmp_path, 'ollama_base_url: "http://localhost:11434"\n')

        # Act / Assert
        with pytest.raises(NonLoopbackURLError):
            load_nl_query_config(path)

    def test_load_nl_query_config_remoteUrlWithOptIn_loadsAndWarns(self, tmp_path, monkeypatch, caplog):
        # Arrange
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434")
        monkeypatch.setenv("OLLAMA_ALLOW_REMOTE_HOST", "true")
        path = self._write(tmp_path, "")

        # Act
        with caplog.at_level("WARNING"):
            config = load_nl_query_config(path)

        # Assert
        assert config["ollama_base_url"] == "http://10.0.0.5:11434"
        assert config["ollama_allow_remote_host"] is True
        assert "SECURITY WARNING" in caplog.text

    def test_load_nl_query_config_envOptInFalse_staysRefused(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434")
        monkeypatch.setenv("OLLAMA_ALLOW_REMOTE_HOST", "false")
        path = self._write(tmp_path, "")

        # Act / Assert
        with pytest.raises(NonLoopbackURLError):
            load_nl_query_config(path)

    @pytest.mark.parametrize("opt_in", ["no", "", "maybe", "True", "TRUE", "1", "yes", " true"])
    def test_load_nl_query_config_envOptInNotExact_raises(self, tmp_path, monkeypatch, opt_in):
        # Arrange: a loopback URL, so only the malformed opt-in can fail
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.setenv("OLLAMA_ALLOW_REMOTE_HOST", opt_in)
        path = self._write(tmp_path, "")

        # Act / Assert
        with pytest.raises(ValueError, match="OLLAMA_ALLOW_REMOTE_HOST must be true or false"):
            load_nl_query_config(path)

    @pytest.mark.parametrize("yaml_value", ["[false]", "{a: 1}", "2", "1", '"True"', '"yes"', "null", '""'])
    def test_load_nl_query_config_yamlOptInNotABool_raises(self, tmp_path, monkeypatch, yaml_value):
        # Arrange: values the generic coercion would have turned into True (or quietly False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_ALLOW_REMOTE_HOST", raising=False)
        path = self._write(tmp_path, f"ollama_allow_remote_host: {yaml_value}\n")

        # Act / Assert
        with pytest.raises(ValueError, match="ollama_allow_remote_host in nl_query.yaml must be true or false"):
            load_nl_query_config(path)

    @pytest.mark.parametrize(("yaml_value", "expected"), [("true", True), ("false", False), ('"false"', False)])
    def test_load_nl_query_config_yamlOptInExact_parsed(self, tmp_path, monkeypatch, yaml_value, expected):
        # Arrange
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        monkeypatch.delenv("OLLAMA_ALLOW_REMOTE_HOST", raising=False)
        path = self._write(tmp_path, f"ollama_allow_remote_host: {yaml_value}\n")

        # Act
        config = load_nl_query_config(path)

        # Assert
        assert config["ollama_allow_remote_host"] is expected

    def test_load_nl_query_config_defaults_useLoopbackIp(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_ALLOW_REMOTE_HOST", raising=False)
        path = tmp_path / "missing.yaml"

        # Act
        config = load_nl_query_config(path)

        # Assert
        assert config["ollama_base_url"] == "http://127.0.0.1:11434"
        assert config["ollama_allow_remote_host"] is False

    def test_load_nl_query_config_loopbackUrl_loads(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
        monkeypatch.delenv("OLLAMA_ALLOW_REMOTE_HOST", raising=False)
        path = self._write(tmp_path, "")

        # Act
        config = load_nl_query_config(path)

        # Assert
        assert config["ollama_base_url"] == "http://127.0.0.1:11435"
        assert config["ollama_allow_remote_host"] is False

    def test_ollama_client_remoteUrlWithoutOptIn_raises(self, monkeypatch):
        # Arrange
        from clinical_analytics.core import nl_query_config
        from clinical_analytics.core.llm_client import OllamaClient

        monkeypatch.setattr(nl_query_config, "OLLAMA_ALLOW_REMOTE_HOST", False)

        # Act / Assert
        with pytest.raises(NonLoopbackURLError):
            OllamaClient(base_url="http://10.0.0.5:11434")

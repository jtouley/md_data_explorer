# SPDX-License-Identifier: MIT
"""A local HTTPS server standing in for public data websites (no real internet in tests).

``LocalHttpsSource`` serves canned responses on 127.0.0.1 with a throwaway self-signed
certificate for made-up ``*.example.test`` names. The ``local_https_source`` fixture
(tests/conftest.py) points the URL fetcher's DNS answers, TCP connect and trust store at it.
The fetcher's address check stays real: the fake names resolve to public-looking addresses
(or private ones, to test refusals), and only connections to the public ones reach the server.
"""

from __future__ import annotations

import shutil
import ssl
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Made-up names → the addresses the fake DNS gives for them. 93.184.215.0/24 is public
# address space, used here only as a label: connections to it are routed to the local server.
FAKE_DNS: dict[str, list[str]] = {
    "data.example.test": ["93.184.215.14"],
    "mirror.example.test": ["93.184.215.15"],
    "internal.example.test": ["10.0.0.5"],
    "metadata.example.test": ["169.254.169.254"],
    "mixed.example.test": ["93.184.215.14", "127.0.0.1"],
}
CERT_NAMES = ("data.example.test", "mirror.example.test", "internal.example.test", "mixed.example.test")


@dataclass
class CannedResponse:
    """What the server sends for one path."""

    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    delay_before_body_s: float = 0.0
    trickle_chunks: int = 0  # >0: send the body in this many pieces with a pause between
    trickle_pause_s: float = 0.0
    chunked: bool = False  # Transfer-Encoding: chunked, no Content-Length


def make_certificate(directory: Path) -> tuple[Path, Path]:
    """Self-signed certificate and key for CERT_NAMES (openssl CLI)."""
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("openssl is needed to make the test certificate")
    cert, key = directory / "cert.pem", directory / "key.pem"
    san = ",".join(f"DNS:{name}" for name in CERT_NAMES)
    subprocess.run(  # noqa: S603 - fixed arguments, test-only
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "2",
            "-subj",
            "/CN=data.example.test",
            "-addext",
            f"subjectAltName={san}",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


class LocalHttpsSource:
    """Serves ``routes`` (path with query → CannedResponse) over HTTPS on 127.0.0.1."""

    def __init__(self, cert: Path, key: Path) -> None:
        self.cert = cert
        self.routes: dict[str, CannedResponse] = {}
        self.requests: list[dict[str, str]] = []
        self.connections: list[str] = []  # addresses the fetcher opened TCP connections to
        source = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
                return

            def do_GET(self) -> None:  # noqa: N802 - stdlib name
                source.requests.append({"path": self.path, **{k.lower(): v for k, v in self.headers.items()}})
                canned = source.routes.get(self.path, CannedResponse(status=404, body=b"no such file"))
                self.send_response(canned.status)
                for name, value in canned.headers.items():
                    self.send_header(name, value)
                if canned.chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                elif "Content-Length" not in canned.headers:
                    self.send_header("Content-Length", str(len(canned.body)))
                self.end_headers()
                if canned.delay_before_body_s:
                    time.sleep(canned.delay_before_body_s)
                pieces = _split(canned.body, canned.trickle_chunks)
                try:
                    for piece in pieces:
                        if canned.chunked:
                            self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                        else:
                            self.wfile.write(piece)
                        self.wfile.flush()
                        if canned.trickle_pause_s:
                            time.sleep(canned.trickle_pause_s)
                    if canned.chunked:
                        self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    return  # the client gave up (size cap, timeout): expected in those tests

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        self._server.socket = context.wrap_socket(self._server.socket, server_side=True)
        self.port: int = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def client_context(self) -> ssl.SSLContext:
        """A verifying client context that trusts only this server's certificate."""
        context = ssl.create_default_context(cafile=str(self.cert))
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        return context

    def url(self, path: str, host: str = "data.example.test") -> str:
        return f"https://{host}{path}"


def _split(body: bytes, pieces: int) -> list[bytes]:
    if pieces <= 1 or not body:
        return [body]
    size = max(1, -(-len(body) // pieces))
    return [body[i : i + size] for i in range(0, len(body), size)]

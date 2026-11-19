# SPDX-License-Identifier: MIT
"""Download one public data file from an HTTPS link, for the upload pipeline.

A fellow pastes a link to a public table (CDC, CMS, WHO, IHME); the bytes that come back are
handed to the same upload path a chosen file takes, so every format and check applies
unchanged. This module only fetches. It parses nothing.

Fetching a URL a person typed is a classic server-side request forgery (SSRF) hazard, so the
safe behaviour is the only behaviour; nothing here can be switched off by a caller:

- HTTPS on port 443 only, with certificate and host-name checks; no user:password in links.
- Every address a host name resolves to must be a public internet address. Loopback, private
  networks, link-local (cloud metadata), carrier NAT, multicast, reserved ranges and IPv6
  forms that embed one (IPv4-mapped, NAT64, 6to4, Teredo) are refused.
- The connection goes to the exact address that was checked (DNS rebinding cannot swap it),
  and each redirect is checked again from the start, at most ``MAX_REDIRECTS`` of them.
- No proxy, no cookies, no Referer, no credentials: the only thing sent is a GET for the link.
- Size is capped at the upload limit while reading (Content-Length is checked first, but not
  trusted). Compressed transfers are inflated with the same cap, so a small gzip cannot
  expand into gigabytes. Archives inside the file are the upload pipeline's job.
- Connect and per-read timeouts, plus one deadline for the whole download, so a slow server
  cannot hold the request open.
- A web page (HTML) is refused even when the link ends in ``.csv``: sites answer a missing or
  signed-in-only file with a page, and that page is not data.

Failures raise :class:`UrlFetchError` with a ``kind`` and a sentence written for a clinician.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import socket
import ssl
import time
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from pathlib import PurePosixPath
from typing import Final, Literal
from urllib.parse import SplitResult, unquote, urljoin, urlsplit

import structlog
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.ui.storage.user_datasets import UploadSecurityValidator

logger = structlog.get_logger()

# Limits. Module constants, not parameters: a caller cannot loosen them by accident.
MAX_DOWNLOAD_BYTES: Final[int] = UploadSecurityValidator.MAX_FILE_SIZE_BYTES
MAX_URL_LENGTH: Final[int] = 2048
MAX_REDIRECTS: Final[int] = 5
HTTPS_PORT: Final[int] = 443
CONNECT_TIMEOUT_S: Final[float] = 15.0
READ_TIMEOUT_S: Final[float] = 30.0
# Whole download, redirects included: 100 MB at under 1 MB/s still finishes.
TOTAL_DEADLINE_S: Final[float] = 300.0
CHUNK_BYTES: Final[int] = 64 * 1024
USER_AGENT: Final[str] = "MD-Data-Explorer/0.1 (public dataset download)"

REDIRECT_STATUSES: Final[frozenset[int]] = frozenset({301, 302, 303, 307, 308})

# Content types that name a data format when neither the server's file name nor the link has
# an extension (e.g. an API endpoint). Used only for extensions the upload accepts.
CONTENT_TYPE_EXTENSIONS: Final[dict[str, str]] = {
    "text/csv": ".csv",
    "application/csv": ".csv",
    "text/comma-separated-values": ".csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/x-spss-sav": ".sav",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "text/tab-separated-values": ".tsv",
}
HTML_CONTENT_TYPES: Final[frozenset[str]] = frozenset({"text/html", "application/xhtml+xml"})
HTML_PREFIXES: Final[tuple[bytes, ...]] = (b"<!doctype html", b"<html")

# Prefixes whose IPv6 addresses carry an IPv4 address that decides where packets go.
_NAT64_PREFIX: Final = ipaddress.IPv6Network("64:ff9b::/96")

FetchFailureKind = Literal[
    "invalid_link",
    "blocked_address",
    "unreachable",
    "insecure",
    "timeout",
    "refused",
    "missing",
    "site_error",
    "too_large",
    "not_data",
    "too_many_redirects",
]


class UrlFetchError(Exception):
    """The link could not be downloaded as a data file. ``message`` is for the user."""

    def __init__(self, kind: FetchFailureKind, message: str) -> None:
        super().__init__(message)
        self.kind: FetchFailureKind = kind
        self.message = message


class SourceProvenance(BaseModel):
    """Where a dataset came from, for the user and the methods section."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(..., description="The link as the user gave it")
    final_url: str = Field(..., description="The address the file came from, after redirects")
    fetched_at: str = Field(..., description="When the download finished (UTC, ISO 8601)")
    size_bytes: int = Field(..., ge=0, description="Size of the file as stored")
    sha256: str = Field(..., min_length=64, max_length=64, description="SHA-256 of the file as stored")
    content_type: str | None = Field(None, description="Content type the site reported")
    last_modified: str | None = Field(None, description="Last-Modified the site reported, if any")
    filename: str = Field(..., description="File name the dataset was read as")

    def methods_sentence(self) -> str:
        """One sentence for a methods section: what was downloaded, from where, and when."""
        fetched = datetime.fromisoformat(self.fetched_at)
        when = f"{fetched.day} {fetched:%B %Y}"
        return f"Data downloaded from {self.final_url} on {when} ({self.size_bytes:,} bytes; SHA-256 {self.sha256})."


@dataclass(frozen=True)
class FetchedFile:
    """A downloaded file, ready for the upload pipeline."""

    content: bytes
    filename: str
    provenance: SourceProvenance


# ---------------------------------------------------------------------------
# Link and address checks
# ---------------------------------------------------------------------------


def validate_link(url: str) -> SplitResult:
    """Parse a pasted link; refuse anything but a plain public HTTPS link."""
    text = url.strip()
    if not text:
        raise UrlFetchError("invalid_link", "Paste a link that starts with https://.")
    if len(text) > MAX_URL_LENGTH:
        raise UrlFetchError("invalid_link", "That link is too long. Copy the file's download link again.")
    if any(ord(ch) < 0x21 or ord(ch) == 0x7F for ch in text):
        raise UrlFetchError("invalid_link", "That link has spaces or hidden characters in it. Copy it again.")
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError as e:
        raise UrlFetchError("invalid_link", "That doesn't look like a web link. Copy it again.") from e
    if parts.scheme.lower() == "http":
        raise UrlFetchError(
            "insecure",
            "Only secure links (https://) can be loaded. Try the same link with https:// instead of http://.",
        )
    if parts.scheme.lower() != "https":
        raise UrlFetchError("invalid_link", "Paste a web link that starts with https://.")
    if not parts.hostname:
        raise UrlFetchError("invalid_link", "That link has no website name in it. Copy it again.")
    if parts.username is not None or parts.password is not None:
        raise UrlFetchError(
            "invalid_link",
            "That link has a user name or password in it. Only public links that need no sign-in can be loaded.",
        )
    if port is not None and port != HTTPS_PORT:
        raise UrlFetchError("invalid_link", "Only standard https:// links can be loaded (no port number).")
    try:
        parts.hostname.encode("idna")
    except UnicodeError as e:
        raise UrlFetchError("invalid_link", "That website name isn't valid. Copy the link again.") from e
    return parts


def is_public_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True only for a globally routable unicast address (never this machine or a private network)."""
    if isinstance(ip, ipaddress.IPv6Address):
        embedded: ipaddress.IPv4Address | None = ip.ipv4_mapped
        if embedded is None and ip in _NAT64_PREFIX:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            return is_public_address(embedded)
        if ip.sixtofour is not None or ip.teredo is not None:
            return False
    return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_loopback or ip.is_link_local)


_BLOCKED_MESSAGE: Final[str] = (
    "That link points to this computer or a private network, not a public website. "
    "Only public data on the internet can be loaded from a link."
)


def _resolve_addresses(host: str) -> list[str]:
    """Every address a host name resolves to, for TCP on the HTTPS port."""
    try:
        infos = socket.getaddrinfo(host, HTTPS_PORT, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as e:
        raise UrlFetchError(
            "unreachable",
            "Couldn't find that website. Check the link for typos and that this computer is online.",
        ) from e
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def public_addresses(host: str) -> list[str]:
    """Resolve a host and return its addresses, refusing the host if any of them is not public.

    All of them, not just one: a name that mixes a public and a private address could
    otherwise be steered to the private one.
    """
    raw = _resolve_addresses(host)
    if not raw:
        raise UrlFetchError("unreachable", "Couldn't find that website. Check the link for typos.")
    for text in raw:
        try:
            ip = ipaddress.ip_address(text.split("%", 1)[0])
        except ValueError as e:
            raise UrlFetchError("blocked_address", _BLOCKED_MESSAGE) from e
        if not is_public_address(ip):
            raise UrlFetchError("blocked_address", _BLOCKED_MESSAGE)
    return raw


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def _tls_context() -> ssl.SSLContext:
    """Verifying TLS context (certificate chain and host name), TLS 1.2 or newer."""
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:  # pragma: no cover - certifi ships with the app's HTTP dependencies
        pass
    else:
        # Python.org macOS builds have no system CA bundle until a post-install step is run.
        context.load_verify_locations(cafile=certifi.where())
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


def _open_tcp(address: str, port: int, timeout: float) -> socket.socket:
    """TCP connection to one already-checked address."""
    return socket.create_connection((address, port), timeout=timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS to a host name, over a TCP connection to one address chosen (and checked) earlier.

    The Host header, SNI and certificate check all use the host name; only the TCP target is
    pinned, so a second DNS answer can never redirect the connection.
    """

    def __init__(self, host: str, addresses: list[str], timeout: float, context: ssl.SSLContext) -> None:
        super().__init__(host, HTTPS_PORT, timeout=timeout, context=context)
        self._addresses = addresses
        self._tls = context

    def connect(self) -> None:
        last_error: OSError | None = None
        for address in self._addresses:
            try:
                raw = _open_tcp(address, HTTPS_PORT, self.timeout)  # type: ignore[arg-type]
            except OSError as e:
                last_error = e
                continue
            try:
                self.sock = self._tls.wrap_socket(raw, server_hostname=self.host)
            except BaseException:
                raw.close()
                raise
            return
        raise last_error or OSError("no address to connect to")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _media_type(value: str | None) -> str | None:
    if not value:
        return None
    return value.split(";", 1)[0].strip().lower() or None


def _disposition_filename(value: str | None) -> str | None:
    if not value:
        return None
    message = Message()
    message["content-disposition"] = value
    name = message.get_filename()
    if not name:
        return None
    return PurePosixPath(str(name).replace("\\", "/")).name or None


def _url_filename(path: str) -> str | None:
    return PurePosixPath(unquote(path)).name or None


def choose_filename(
    *,
    url_path: str,
    content_disposition: str | None,
    content_type: str | None,
    accepted_extensions: frozenset[str],
) -> str:
    """Name to hand the upload pipeline: its extension decides how the file is read."""
    candidates = [n for n in (_disposition_filename(content_disposition), _url_filename(url_path)) if n]
    for name in candidates:
        if PurePosixPath(name).suffix.lower() in accepted_extensions:
            return name
    by_type = CONTENT_TYPE_EXTENSIONS.get(_media_type(content_type) or "")
    if by_type and by_type in accepted_extensions:
        stem = PurePosixPath(candidates[0]).stem if candidates else "download"
        return f"{stem or 'download'}{by_type}"
    shown = sorted(ext.lstrip(".").upper() for ext in accepted_extensions)
    raise UrlFetchError(
        "not_data",
        f"That link isn't a data file this app can read ({', '.join(shown)}). "
        "On the website, look for a Download or Export button and copy that link instead.",
    )


def _status_error(status: int) -> UrlFetchError:
    if status in (404, 410):
        return UrlFetchError(
            "missing",
            f"The website says there is no file at that link (error {status}). "
            "It may have moved: open the page in your browser and copy the download link again.",
        )
    if status in (401, 403, 407):
        return UrlFetchError(
            "refused",
            f"The website refused the download (error {status}); it may need you to sign in. "
            "Download the file in your browser and upload it instead.",
        )
    if status == 429:
        return UrlFetchError(
            "refused", "The website is limiting downloads right now (error 429). Wait a few minutes and try again."
        )
    if status >= 500:
        return UrlFetchError(
            "site_error", f"The website had a problem sending the file (error {status}). Try again later."
        )
    return UrlFetchError("site_error", f"The website answered in an unexpected way (status {status}). Try again later.")


def _too_large() -> UrlFetchError:
    limit_mb = MAX_DOWNLOAD_BYTES // (1024 * 1024)
    return UrlFetchError(
        "too_large",
        f"That file is larger than {limit_mb} MB, the most this app loads. "
        "Choose fewer years, places or columns on the website and download a smaller file.",
    )


class _Deadline:
    def __init__(self, seconds: float) -> None:
        self._end = time.monotonic() + seconds

    def remaining(self) -> float:
        left = self._end - time.monotonic()
        if left <= 0:
            raise UrlFetchError(
                "timeout",
                "The website took too long to send the file. Try again later, or download it in your browser.",
            )
        return left


class _BoundedBody:
    """Collects the response body, inflating gzip/deflate, never past MAX_DOWNLOAD_BYTES."""

    def __init__(self, content_encoding: str | None) -> None:
        encoding = (content_encoding or "identity").strip().lower()
        if encoding in ("", "identity"):
            self._inflater = None
        elif encoding in ("gzip", "x-gzip"):
            self._inflater = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        elif encoding == "deflate":
            self._inflater = zlib.decompressobj(wbits=zlib.MAX_WBITS)
        else:
            raise UrlFetchError(
                "not_data",
                "The website sent the file in a form this app can't read. Download it in your browser instead.",
            )
        self._parts: list[bytes] = []
        self._size = 0
        self._wire = 0

    def _keep(self, data: bytes) -> None:
        self._size += len(data)
        if self._size > MAX_DOWNLOAD_BYTES:
            raise _too_large()
        if data:
            self._parts.append(data)

    def feed(self, chunk: bytes) -> None:
        self._wire += len(chunk)
        if self._wire > MAX_DOWNLOAD_BYTES:
            raise _too_large()
        if self._inflater is None:
            self._keep(chunk)
            return
        pending = chunk
        try:
            while pending:
                # Never ask for more than one byte past the cap, so a bomb stops early.
                self._keep(self._inflater.decompress(pending, MAX_DOWNLOAD_BYTES - self._size + 1))
                pending = self._inflater.unconsumed_tail
        except zlib.error as e:
            raise UrlFetchError("site_error", "The file arrived damaged. Try again.") from e

    def finish(self) -> bytes:
        if self._inflater is not None:
            if not self._inflater.eof:
                raise UrlFetchError("site_error", "The file arrived incomplete. Try again.")
            self._keep(self._inflater.flush())
        return b"".join(self._parts)


def _read_body(response: http.client.HTTPResponse, sock: socket.socket | None, deadline: _Deadline) -> bytes:
    length = response.getheader("Content-Length")
    if length is not None and length.strip().isdigit() and int(length) > MAX_DOWNLOAD_BYTES:
        raise _too_large()
    body = _BoundedBody(response.getheader("Content-Encoding"))
    while True:
        if sock is not None:
            sock.settimeout(min(READ_TIMEOUT_S, deadline.remaining()))
        chunk = response.read(CHUNK_BYTES)
        if not chunk:
            break
        body.feed(chunk)
        deadline.remaining()
    return body.finish()


def _looks_like_html(content: bytes) -> bool:
    head = content[:512].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return head.startswith(HTML_PREFIXES)


_WEB_PAGE_MESSAGE: Final[str] = (
    "That link opens a web page, not a data file. On the page, find the Download or Export "
    "button, right-click it, choose Copy Link, and paste that link here."
)


def fetch_public_file(url: str, *, accepted_extensions: frozenset[str]) -> FetchedFile:
    """Download a public data file over HTTPS, following checked redirects. See the module doc.

    Args:
        url: The link the user pasted.
        accepted_extensions: Extensions the upload pipeline reads (e.g. {".csv", ".xlsx"});
            decides the file name the bytes are handed over with.

    Raises:
        UrlFetchError: with a plain-language ``message`` for the user.
    """
    if not accepted_extensions:
        raise ValueError("accepted_extensions must name at least one extension")
    deadline = _Deadline(TOTAL_DEADLINE_S)
    context = _tls_context()
    current = url.strip()
    for hop in range(MAX_REDIRECTS + 1):
        parts = validate_link(current)
        host = parts.hostname or ""
        log = logger.bind(url_host=host, hop=hop)
        addresses = public_addresses(host)
        connection = _PinnedHTTPSConnection(
            host, addresses, timeout=min(CONNECT_TIMEOUT_S, deadline.remaining()), context=context
        )
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "*/*",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "close",
                },
            )
            connection.sock.settimeout(min(READ_TIMEOUT_S, deadline.remaining()))  # type: ignore[union-attr]
            response = connection.getresponse()
            status = response.status
            if status in REDIRECT_STATUSES:
                location = response.getheader("Location")
                if not location:
                    raise _status_error(status)
                current = urljoin(current, location)
                log.info("dataset_url_redirect", status=status)
                continue
            if status != 200:
                raise _status_error(status)
            content_type = response.getheader("Content-Type")
            if _media_type(content_type) in HTML_CONTENT_TYPES:
                raise UrlFetchError("not_data", _WEB_PAGE_MESSAGE)
            filename = choose_filename(
                url_path=parts.path,
                content_disposition=response.getheader("Content-Disposition"),
                content_type=content_type,
                accepted_extensions=accepted_extensions,
            )
            content = _read_body(response, connection.sock, deadline)
            last_modified = response.getheader("Last-Modified")
        except ssl.SSLError as e:
            raise UrlFetchError(
                "insecure",
                "The website's security certificate couldn't be verified, so nothing was downloaded. "
                "Check the link, or download the file in your browser.",
            ) from e
        except TimeoutError as e:
            raise UrlFetchError(
                "timeout", "The website took too long to answer. Try again later, or download the file in your browser."
            ) from e
        except (OSError, http.client.HTTPException) as e:
            raise UrlFetchError(
                "unreachable",
                "Couldn't download from that website: the connection failed. "
                "Check that this computer is online, then try again.",
            ) from e
        finally:
            connection.close()

        if _looks_like_html(content):
            raise UrlFetchError("not_data", _WEB_PAGE_MESSAGE)
        provenance = SourceProvenance(
            url=url.strip(),
            final_url=current,
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            content_type=content_type,
            last_modified=last_modified,
            filename=filename,
        )
        log.info("dataset_url_fetched", size_bytes=len(content), redirects=hop)
        return FetchedFile(content=content, filename=filename, provenance=provenance)

    raise UrlFetchError(
        "too_many_redirects",
        "That link kept sending the download somewhere else, so it was stopped. "
        "Copy the file's direct download link instead.",
    )

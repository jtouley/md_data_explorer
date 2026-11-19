# SPDX-License-Identifier: MIT
"""URL fetch layer: safe defaults (SSRF, redirects, rebinding, size, time) and provenance.

Every download goes to the local HTTPS server (``local_https_source``); nothing reaches the
internet. Data is synthetic.
"""

from __future__ import annotations

import gzip
import hashlib
import ipaddress
import socket
import ssl

import pytest
from fixtures.https_server import CannedResponse

from clinical_analytics.storage import url_fetch
from clinical_analytics.storage.url_fetch import (
    UrlFetchError,
    choose_filename,
    fetch_public_file,
    is_public_address,
    validate_link,
)

ACCEPTED = frozenset({".csv", ".xlsx", ".xls", ".sav"})
CSV = ("state,year,rate\n" + "".join(f"S{i:02d},{2000 + i % 24},{i * 1.5}\n" for i in range(200))).encode()


# ---------------------------------------------------------------------------
# Link and address checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("", "invalid_link"),
        ("   ", "invalid_link"),
        ("http://data.example.test/a.csv", "insecure"),
        ("ftp://data.example.test/a.csv", "invalid_link"),
        ("file:///etc/passwd", "invalid_link"),
        ("javascript:alert(1)", "invalid_link"),
        ("https://", "invalid_link"),
        ("https://user:pw@data.example.test/a.csv", "invalid_link"),
        ("https://token@data.example.test/a.csv", "invalid_link"),
        ("https://data.example.test:8443/a.csv", "invalid_link"),
        ("https://data.example.test:99999/a.csv", "invalid_link"),
        ("https://data.example.test/a b.csv", "invalid_link"),
        ("https://data.example.test/a\r\nX-Injected: 1", "invalid_link"),
        ("https://data.example.test/" + "a" * 2100, "invalid_link"),
    ],
)
def test_validateLink_unsafeOrMalformed_raisesWithKind(url: str, kind: str):
    # Arrange: url

    # Act
    with pytest.raises(UrlFetchError) as caught:
        validate_link(url)

    # Assert
    assert caught.value.kind == kind
    assert caught.value.message


@pytest.mark.parametrize(
    "url",
    ["https://data.example.test/a.csv", "https://data.example.test:443/a.csv?x=1", "  https://DATA.example.test/  "],
)
def test_validateLink_plainHttps_accepted(url: str):
    # Arrange: url

    # Act
    parts = validate_link(url)

    # Assert
    assert parts.hostname == "data.example.test"


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("93.184.215.14", True),
        ("2606:4700:4700::1111", True),
        ("127.0.0.1", False),
        ("0.0.0.0", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("192.168.1.1", False),
        ("169.254.169.254", False),  # cloud metadata
        ("100.64.0.1", False),  # carrier-grade NAT
        ("224.0.0.1", False),  # multicast
        ("240.0.0.1", False),  # reserved
        ("255.255.255.255", False),
        ("192.0.2.1", False),  # documentation range
        ("::1", False),
        ("::", False),
        ("fe80::1", False),
        ("fc00::1", False),
        ("ff02::1", False),
        ("::ffff:127.0.0.1", False),  # IPv4-mapped loopback
        ("::ffff:93.184.215.14", True),
        ("64:ff9b::7f00:1", False),  # NAT64 of 127.0.0.1
        ("64:ff9b::a00:1", False),  # NAT64 of 10.0.0.1
        ("2002:7f00:1::1", False),  # 6to4
        ("2001:0:4136:e378:8000:63bf:3fff:fdd2", False),  # Teredo
    ],
)
def test_isPublicAddress_rangeTable_matchesExpected(address: str, public: bool):
    # Arrange
    ip = ipaddress.ip_address(address)

    # Act
    result = is_public_address(ip)

    # Assert
    assert result is public


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/a.csv",
        "https://[::1]/a.csv",
        "https://169.254.169.254/latest/meta-data/",
        "https://[::ffff:127.0.0.1]/a.csv",
        "https://localhost/a.csv",
        "https://internal.example.test/a.csv",
        "https://metadata.example.test/a.csv",
        "https://mixed.example.test/a.csv",  # one public, one loopback answer
    ],
)
def test_fetchPublicFile_nonPublicHost_blockedBeforeConnecting(local_https_source, url: str):
    # Arrange: local_https_source records every connection

    # Act
    with pytest.raises(UrlFetchError) as caught:
        fetch_public_file(url, accepted_extensions=ACCEPTED)

    # Assert
    assert caught.value.kind == "blocked_address"
    assert local_https_source.connections == []
    assert local_https_source.requests == []


def test_fetchPublicFile_unknownHost_unreachable(monkeypatch):
    # Arrange
    def no_such_host(*_args: object, **_kwargs: object) -> list[object]:
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(url_fetch.socket, "getaddrinfo", no_such_host)

    # Act
    with pytest.raises(UrlFetchError) as caught:
        fetch_public_file("https://nowhere.example.test/a.csv", accepted_extensions=ACCEPTED)

    # Assert
    assert caught.value.kind == "unreachable"


def test_fetchPublicFile_dnsRebinding_connectsOnlyToCheckedAddress(local_https_source, monkeypatch):
    # Arrange: the name answers public first, then loopback (a rebinding attacker)
    local_https_source.routes["/rates.csv"] = CannedResponse(headers={"Content-Type": "text/csv"}, body=CSV)
    answers = iter([["93.184.215.14"], ["127.0.0.1"], ["127.0.0.1"]])
    calls: list[str] = []
    real_getaddrinfo = socket.getaddrinfo

    def rebinding(host: str, port: int, *args: object, **kwargs: object) -> list[tuple[object, ...]]:
        if host == "127.0.0.1":  # the fixture's own connection to the local server
            return real_getaddrinfo(host, port, *args, **kwargs)
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (a, port)) for a in next(answers)]

    monkeypatch.setattr(url_fetch.socket, "getaddrinfo", rebinding)

    # Act
    fetched = fetch_public_file("https://data.example.test/rates.csv", accepted_extensions=ACCEPTED)

    # Assert: one lookup, and the TCP connection went to the address that was checked
    assert calls == ["data.example.test"]
    assert local_https_source.connections == ["93.184.215.14"]
    assert fetched.content == CSV


# ---------------------------------------------------------------------------
# Successful downloads and provenance
# ---------------------------------------------------------------------------


def test_fetchPublicFile_csv_returnsBytesAndProvenance(local_https_source):
    # Arrange
    local_https_source.routes["/api/views/abcd/rows.csv?accessType=DOWNLOAD"] = CannedResponse(
        headers={
            "Content-Type": "text/csv; charset=utf-8",
            "Content-Disposition": 'attachment; filename="PLACES_County_Data.csv"',
            "Last-Modified": "Tue, 03 Nov 2026 10:00:00 GMT",
        },
        body=CSV,
    )
    url = local_https_source.url("/api/views/abcd/rows.csv?accessType=DOWNLOAD")

    # Act
    fetched = fetch_public_file(url, accepted_extensions=ACCEPTED)

    # Assert
    provenance = fetched.provenance
    assert fetched.content == CSV
    assert fetched.filename == "PLACES_County_Data.csv"
    assert provenance.url == url
    assert provenance.final_url == url
    assert provenance.size_bytes == len(CSV)
    assert provenance.sha256 == hashlib.sha256(CSV).hexdigest()
    assert provenance.content_type == "text/csv; charset=utf-8"
    assert provenance.last_modified == "Tue, 03 Nov 2026 10:00:00 GMT"
    assert provenance.fetched_at.endswith("+00:00")
    assert url in provenance.methods_sentence()
    assert provenance.sha256 in provenance.methods_sentence()


def test_fetchPublicFile_requestHeaders_sendNothingButTheGet(local_https_source):
    # Arrange
    local_https_source.routes["/a.csv"] = CannedResponse(headers={"Content-Type": "text/csv"}, body=CSV)

    # Act
    fetch_public_file(local_https_source.url("/a.csv"), accepted_extensions=ACCEPTED)

    # Assert: no cookies, credentials or referrer; Host is the name, not the address
    request = local_https_source.requests[0]
    assert request["host"] == "data.example.test"
    assert not {"cookie", "authorization", "referer", "proxy-authorization"} & set(request)


def test_fetchPublicFile_gzipTransfer_inflatedAndHashedAsStored(local_https_source):
    # Arrange
    local_https_source.routes["/a.csv"] = CannedResponse(
        headers={"Content-Type": "text/csv", "Content-Encoding": "gzip"}, body=gzip.compress(CSV)
    )

    # Act
    fetched = fetch_public_file(local_https_source.url("/a.csv"), accepted_extensions=ACCEPTED)

    # Assert
    assert fetched.content == CSV
    assert fetched.provenance.sha256 == hashlib.sha256(CSV).hexdigest()


def test_fetchPublicFile_chunkedTransfer_readFully(local_https_source):
    # Arrange
    local_https_source.routes["/a.csv"] = CannedResponse(
        headers={"Content-Type": "text/csv"}, body=CSV, chunked=True, trickle_chunks=4
    )

    # Act
    fetched = fetch_public_file(local_https_source.url("/a.csv"), accepted_extensions=ACCEPTED)

    # Assert
    assert fetched.content == CSV


def test_fetchPublicFile_redirectToOtherPublicHost_followedAndRecorded(local_https_source):
    # Arrange
    local_https_source.routes["/latest"] = CannedResponse(
        status=302, headers={"Location": "https://mirror.example.test/files/2026.csv"}
    )
    local_https_source.routes["/files/2026.csv"] = CannedResponse(headers={"Content-Type": "text/csv"}, body=CSV)

    # Act
    fetched = fetch_public_file(local_https_source.url("/latest"), accepted_extensions=ACCEPTED)

    # Assert
    assert fetched.provenance.url == "https://data.example.test/latest"
    assert fetched.provenance.final_url == "https://mirror.example.test/files/2026.csv"
    assert fetched.filename == "2026.csv"
    assert local_https_source.connections == ["93.184.215.14", "93.184.215.15"]


@pytest.mark.parametrize(
    ("location", "kind"),
    [
        ("https://internal.example.test/a.csv", "blocked_address"),
        ("https://127.0.0.1/a.csv", "blocked_address"),
        ("https://169.254.169.254/latest/meta-data/", "blocked_address"),
        ("http://data.example.test/a.csv", "insecure"),
        ("file:///etc/passwd", "invalid_link"),
        ("https://data.example.test:8080/a.csv", "invalid_link"),
    ],
)
def test_fetchPublicFile_redirectToUnsafeTarget_refused(local_https_source, location: str, kind: str):
    # Arrange
    local_https_source.routes["/go"] = CannedResponse(status=301, headers={"Location": location})

    # Act
    with pytest.raises(UrlFetchError) as caught:
        fetch_public_file(local_https_source.url("/go"), accepted_extensions=ACCEPTED)

    # Assert: the first hop was the only connection
    assert caught.value.kind == kind
    assert local_https_source.connections == ["93.184.215.14"]


def test_fetchPublicFile_redirectLoop_stoppedAfterLimit(local_https_source):
    # Arrange
    local_https_source.routes["/loop"] = CannedResponse(status=307, headers={"Location": "/loop"})

    # Act
    with pytest.raises(UrlFetchError) as caught:
        fetch_public_file(local_https_source.url("/loop"), accepted_extensions=ACCEPTED)

    # Assert
    assert caught.value.kind == "too_many_redirects"
    assert len(local_https_source.requests) == url_fetch.MAX_REDIRECTS + 1


# ---------------------------------------------------------------------------
# What the website answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (404, "missing"),
        (410, "missing"),
        (401, "refused"),
        (403, "refused"),
        (429, "refused"),
        (500, "site_error"),
        (503, "site_error"),
        (204, "site_error"),
    ],
)
def test_fetchPublicFile_errorStatus_plainKind(local_https_source, status: int, kind: str):
    # Arrange
    local_https_source.routes["/a.csv"] = CannedResponse(status=status, body=b"" if status == 204 else b"nope")

    # Act
    with pytest.raises(UrlFetchError) as caught:
        fetch_public_file(local_https_source.url("/a.csv"), accepted_extensions=ACCEPTED)

    # Assert
    assert caught.value.kind == kind
    assert str(status) in caught.value.message


@pytest.mark.parametrize(
    ("path", "headers", "body"),
    [
        ("/data", {"Content-Type": "text/html; charset=utf-8"}, b"<p>Download page</p>"),
        ("/a.csv", {"Content-Type": "text/csv"}, b"\n  <!DOCTYPE html><html><body>Sign in</body></html>" * 30),
        ("/report.pdf", {"Content-Type": "application/pdf"}, b"%PDF-1.7 synthetic"),
        ("/export", {"Content-Type": "application/octet-stream"}, b"\x00\x01binary"),
        ("/a.csv", {"Content-Type": "text/csv", "Content-Encoding": "br"}, b"brotli bytes"),
    ],
)
def test_fetchPublicFile_notADataFile_refusedAsNotData(
    local_https_source, path: str, headers: dict[str, str], body: bytes
):
    # Arrange
    local_https_source.routes[path] = CannedResponse(headers=headers, body=body)

    # Act
    with pytest.raises(UrlFetchError) as caught:
        fetch_public_file(local_https_source.url(path), accepted_extensions=ACCEPTED)

    # Assert
    assert caught.value.kind == "not_data"


def test_fetchPublicFile_untrustedCertificate_insecure(local_https_source, monkeypatch):
    # Arrange: the real trust store, which does not know the test certificate
    monkeypatch.setattr(url_fetch, "_tls_context", ssl.create_default_context)
    local_https_source.routes["/a.csv"] = CannedResponse(headers={"Content-Type": "text/csv"}, body=CSV)

    # Act
    with pytest.raises(UrlFetchError) as caught:
        fetch_public_file(local_https_source.url("/a.csv"), accepted_extensions=ACCEPTED)

    # Assert
    assert caught.value.kind == "insecure"
    assert local_https_source.requests == []


# ---------------------------------------------------------------------------
# Size and time limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "canned",
    [
        CannedResponse(headers={"Content-Type": "text/csv"}, body=CSV),  # Content-Length over the cap
        CannedResponse(headers={"Content-Type": "text/csv"}, body=CSV, chunked=True, trickle_chunks=8),
        # A gzip bomb: about 10 kB on the wire, 10 MB inflated.
        CannedResponse(
            headers={"Content-Type": "text/csv", "Content-Encoding": "gzip"}, body=gzip.compress(b"0" * 10_000_000)
        ),
    ],
    ids=["content-length", "chunked", "gzip-bomb"],
)
def test_fetchPublicFile_overSizeCap_tooLarge(local_https_source, monkeypatch, canned: CannedResponse):
    # Arrange
    monkeypatch.setattr(url_fetch, "MAX_DOWNLOAD_BYTES", 1000)
    local_https_source.routes["/big.csv"] = canned

    # Act
    with pytest.raises(UrlFetchError) as caught:
        fetch_public_file(local_https_source.url("/big.csv"), accepted_extensions=ACCEPTED)

    # Assert
    assert caught.value.kind == "too_large"
    assert "MB" in caught.value.message


def test_fetchPublicFile_exactlyAtSizeCap_accepted(local_https_source, monkeypatch):
    # Arrange: the boundary itself is allowed
    monkeypatch.setattr(url_fetch, "MAX_DOWNLOAD_BYTES", len(CSV))
    local_https_source.routes["/a.csv"] = CannedResponse(headers={"Content-Type": "text/csv"}, body=CSV)

    # Act
    fetched = fetch_public_file(local_https_source.url("/a.csv"), accepted_extensions=ACCEPTED)

    # Assert
    assert fetched.provenance.size_bytes == len(CSV)


def test_fetchPublicFile_serverStallsBeforeBody_timeout(local_https_source, monkeypatch):
    # Arrange
    monkeypatch.setattr(url_fetch, "READ_TIMEOUT_S", 0.3)
    local_https_source.routes["/slow.csv"] = CannedResponse(
        headers={"Content-Type": "text/csv"}, body=CSV, delay_before_body_s=2.0
    )

    # Act
    with pytest.raises(UrlFetchError) as caught:
        fetch_public_file(local_https_source.url("/slow.csv"), accepted_extensions=ACCEPTED)

    # Assert
    assert caught.value.kind == "timeout"


def test_fetchPublicFile_tricklingServer_stoppedByOverallDeadline(local_https_source, monkeypatch):
    # Arrange: every read is quick, but the whole download never ends in time
    monkeypatch.setattr(url_fetch, "TOTAL_DEADLINE_S", 0.6)
    local_https_source.routes["/drip.csv"] = CannedResponse(
        headers={"Content-Type": "text/csv"}, body=CSV, trickle_chunks=40, trickle_pause_s=0.1
    )

    # Act
    with pytest.raises(UrlFetchError) as caught:
        fetch_public_file(local_https_source.url("/drip.csv"), accepted_extensions=ACCEPTED)

    # Assert
    assert caught.value.kind == "timeout"


def test_fetchPublicFile_noAcceptedExtensions_rejectedAsProgrammingError():
    # Arrange
    empty: frozenset[str] = frozenset()

    # Act / Assert
    with pytest.raises(ValueError, match="accepted_extensions"):
        fetch_public_file("https://data.example.test/a.csv", accepted_extensions=empty)


# ---------------------------------------------------------------------------
# File name the upload pipeline sees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url_path", "disposition", "content_type", "expected"),
    [
        ("/rows.csv", 'attachment; filename="PLACES.csv"', "text/csv", "PLACES.csv"),
        ("/rows.csv", None, "text/csv", "rows.csv"),
        ("/rows.csv", "attachment; filename*=UTF-8''Vaccination%20Coverage.xlsx", None, "Vaccination Coverage.xlsx"),
        ("/rows.csv", 'attachment; filename="../../etc/passwd.csv"', None, "passwd.csv"),
        ("/rows.csv", 'attachment; filename="..\\\\..\\\\evil.csv"', None, "evil.csv"),
        ("/api/export", None, "text/csv", "export.csv"),
        ("/", None, "text/csv; charset=utf-8", "download.csv"),
        ("/sheet", None, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "sheet.xlsx"),
        ("/data%20file.CSV", None, None, "data file.CSV"),
        ("/notes.txt", 'attachment; filename="table.csv"', "text/plain", "table.csv"),
    ],
)
def test_chooseFilename_sources_pickNameWithReadableExtension(
    url_path: str, disposition: str | None, content_type: str | None, expected: str
):
    # Arrange: inputs above

    # Act
    name = choose_filename(
        url_path=url_path, content_disposition=disposition, content_type=content_type, accepted_extensions=ACCEPTED
    )

    # Assert
    assert name == expected


@pytest.mark.parametrize(
    ("url_path", "content_type"),
    [("/notes.txt", "text/plain"), ("/archive.zip", "application/zip"), ("/x", None), ("/x", "")],
)
def test_chooseFilename_unreadableType_notData(url_path: str, content_type: str | None):
    # Arrange: .zip is not in ACCEPTED here, so a ZIP is refused like any other type

    # Act
    with pytest.raises(UrlFetchError) as caught:
        choose_filename(
            url_path=url_path, content_disposition=None, content_type=content_type, accepted_extensions=ACCEPTED
        )

    # Assert
    assert caught.value.kind == "not_data"
    assert "CSV" in caught.value.message

# SPDX-License-Identifier: MIT
"""The one way the backend talks to the local model server (Ollama): loopback only, no proxies.

SECURITY.md promises that patient data never leaves the machine. A prompt carries the question,
column names and category values, so two things must hold for every call to the model server:

* The URL names this machine (``is_loopback_url``). Config refuses any other host at load time
  unless ``ollama_allow_remote_host`` is set, which logs a loud warning.
* The HTTP client ignores proxy settings (``model_http_session``). Plain ``requests`` calls read
  ``HTTP_PROXY``/``ALL_PROXY`` (and, on Windows and macOS, the OS proxy settings), and send a
  request for ``http://localhost:11434`` through the proxy unless ``NO_PROXY`` lists localhost.
  ``trust_env = False`` turns all of that off, as well as ``.netrc`` credentials.

Callers also pass ``allow_redirects=False`` and a timeout on every request: a redirect could
send the prompt to another host.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Final
from urllib.parse import urlsplit

import requests  # type: ignore[import-untyped]

LOOPBACK_HOSTNAMES: Final = frozenset({"localhost"})
ALLOWED_SCHEMES: Final = frozenset({"http", "https"})
# Characters on which URL parsers disagree about the host (urllib vs urllib3): a URL that holds
# any of them is refused outright rather than trusting one parser's reading.
_AMBIGUOUS: Final = re.compile(r"[\s\\@#?%\x00-\x1f\x7f]")
# Setting that lets the model server live on another host (config key and environment variable).
ALLOW_REMOTE_SETTING: Final = "ollama_allow_remote_host"
ALLOW_REMOTE_ENV: Final = "OLLAMA_ALLOW_REMOTE_HOST"


class NonLoopbackURLError(ValueError):
    """A model server URL that is not on this machine, without the explicit opt-in."""


def is_loopback_url(url: str) -> bool:
    """True if ``url`` is http(s) on ``localhost``, 127.0.0.0/8 or ``::1``.

    Anything that cannot be parsed, or that only looks local (``localhost.example.com``,
    ``127.0.0.1.nip.io``, ``user@localhost@evil``, IPv4-mapped or scoped IPv6), is not loopback.
    So is any URL with userinfo, a query, a fragment, a backslash, a percent-escape or
    whitespace: parsers read the host of such URLs differently, so none of them is trusted.
    """
    if not isinstance(url, str) or _AMBIGUOUS.search(url):
        return False
    try:
        parts = urlsplit(url)
        hostname = parts.hostname  # lower-cased, brackets and userinfo stripped
        parts.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError:
        return False
    if parts.scheme not in ALLOWED_SCHEMES or not hostname:
        return False
    if hostname in LOOPBACK_HOSTNAMES:
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and (address.ipv4_mapped is not None or address.scope_id):
        return False
    return address.is_loopback


def require_loopback_url(url: str, *, allow_remote: bool, setting: str = "ollama_base_url") -> str:
    """Return ``url`` if it is loopback, or if ``allow_remote`` is set; refuse it otherwise.

    Raises:
        NonLoopbackURLError: ``url`` is not on this machine and ``allow_remote`` is False.
    """
    if is_loopback_url(url):
        return url
    if allow_remote is True:  # only an explicit True opts in, never a truthy string or object
        return url
    raise NonLoopbackURLError(
        f"{setting} must point at this machine (localhost, 127.0.0.1 or ::1), got {url!r}. "
        f"Prompts carry column names and category values from your data. To send them to another "
        f"host anyway, set {ALLOW_REMOTE_SETTING}: true (or {ALLOW_REMOTE_ENV}=true)."
    )


def model_http_session() -> requests.Session:
    """A requests session for the model server that never uses proxy or netrc settings.

    Redirects are never followed: with ``max_redirects = 0`` any 3xx reply raises
    ``requests.TooManyRedirects`` (a ``RequestException``), with or without the callers'
    ``allow_redirects=False``, so a model server cannot send a prompt to another host.
    """
    session = requests.Session()
    session.trust_env = False
    session.max_redirects = 0
    return session

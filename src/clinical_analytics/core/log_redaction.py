# SPDX-License-Identifier: MIT
"""Keep patient-level text out of logs.

Questions typed by clinicians, filter values and row values can identify patients, so no log
line at any level may carry them. Call sites log a keyed hash plus a length or count instead
(``stable_hash`` / ``text_fingerprint``). ``redact_sensitive_fields`` is a structlog processor
that enforces the same rule as a backstop when a call site slips: any event key that names raw
text (``query``, ``question``, any ``*_preview`` ...) is replaced by its keyed hash and size, keys that hold data
values (``value``, ``filters``, ``rows`` ...) by their size only, and rendered tracebacks keep
frames and exception types but drop exception messages (they often quote values); a
``traceback`` key gets the same treatment.

Hashes are HMAC-SHA256 with a random per-install key stored in the data home (``.log-key``), so
a short value such as a diagnosis code cannot be recovered by hashing a dictionary of
candidates; the same text still hashes the same way across runs on one install.

``configure_logging`` routes structlog and the standard library (``logging``, uvicorn,
warnings) through one redacting handler and sets the level from
``CLINICAL_ANALYTICS_LOG_LEVEL`` (default INFO).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import sys
import threading
import traceback
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import structlog

from clinical_analytics.core.data_home import data_home, ensure_data_home

LOG_LEVEL_ENV = "CLINICAL_ANALYTICS_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

# Event keys whose values are raw user text or data values. Matched exactly.
SENSITIVE_LOG_KEYS: frozenset[str] = frozenset(
    {
        "query",
        "query_text",
        "normalized_query",
        "question",
        "previous_query",
        "user_prompt",
        "prompt",
        "response",
        "raw",
        "filter",
        "filters",
        "filters_extracted",
        "previous_filters",
        "filter_value",
        "value",
        "values",
        "row",
        "rows",
        "sample_values",
        "exception_message",
        "error_message",
        # Free text that can quote values: error strings, SQL, messages, LLM output, plans.
        "error",
        "errors",
        "sql",
        "message",
        "detail",
        "details",
        "text",
        "content",
        "payload",
        "result",
        "results",
        "data",
        "plan",
        "context",
        "history",
        "conversation_history",
        "filters_applied",
        "interpretation",
        "answer",
        "suggestion",
        "suggested_value",
        "current_value",
        # Fragments of the question the pattern tier extracted, and names that can carry
        # identifiers (an upload's file name, a failure reason quoting LLM-chosen columns).
        "variable_term",
        "primary_term",
        "group_term",
        "group_phrase",
        "subject_phrase",
        "original",
        "filename",
        "failure_reason",
    }
)
# Keys whose values are data values: even a keyed hash is not logged, only the size.
VALUE_LOG_KEYS: frozenset[str] = frozenset(
    {
        "filter",
        "filters",
        "filters_extracted",
        "previous_filters",
        "filter_value",
        "value",
        "values",
        "row",
        "rows",
        "sample_values",
        "filters_applied",
        "suggested_value",
        "current_value",
    }
)

# Key families matched by shape, so a new ``*_preview`` field needs no allowlist edit.
SENSITIVE_LOG_KEY_SUFFIXES: tuple[str, ...] = ("_preview",)
# Keys that hold rendered tracebacks: kept as frames and exception types, messages dropped.
TRACEBACK_LOG_KEYS: frozenset[str] = frozenset({"traceback"})


def is_sensitive_log_key(key: object) -> bool:
    """True for keys that can carry user text or data values (exact names or a known suffix)."""
    if not isinstance(key, str):
        return False
    return (
        key in SENSITIVE_LOG_KEYS
        or key == "preview"
        or key.endswith(SENSITIVE_LOG_KEY_SUFFIXES)
        or key in TRACEBACK_LOG_KEYS
    )


REDACTED = "[redacted]"
LOG_KEY_FILENAME = ".log-key"
_LOG_KEY_BYTES = 32
_log_keys: dict[Path, bytes] = {}
_log_keys_lock = threading.Lock()


class LogKeyUnavailableError(OSError):
    """The per-install log key cannot be read or created (read-only or odd filesystem)."""


def _load_or_create_key(path: Path) -> bytes:
    """Read the install's key, creating it atomically (0600) on first use."""
    try:
        key = path.read_bytes()
        if len(key) == _LOG_KEY_BYTES:
            return key
        raise LogKeyUnavailableError(f"{path.name} is corrupt ({len(key)} bytes); delete it to make a new key")
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, secrets.token_bytes(_LOG_KEY_BYTES))
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(tmp, path)  # atomic and fails if another process created the key first
    except FileExistsError:
        pass
    except OSError:
        # Filesystems without hard links (exFAT, some network shares): best-effort rename.
        if not path.exists():
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    key = path.read_bytes()
    if len(key) != _LOG_KEY_BYTES:
        raise LogKeyUnavailableError(f"{path.name} has the wrong size after creation")
    return key


_key_warning_lock = threading.Lock()
_key_warned = False


def _warn_key_unavailable_once(error: OSError) -> None:
    """Written straight to stderr: logging from inside the log processor would recurse."""
    global _key_warned
    with _key_warning_lock:
        if _key_warned:
            return
        _key_warned = True
    sys.stderr.write(
        f"clinical_analytics: log hashing disabled ({type(error).__name__}); "
        "logs show sizes only until the data home is writable\n"
    )


def log_hash_key() -> bytes | None:
    """The per-install HMAC key for log hashes (cached per data home), or None in safe mode."""
    path = data_home() / LOG_KEY_FILENAME
    with _log_keys_lock:
        key = _log_keys.get(path)
        if key is None:
            try:
                ensure_data_home()
                key = _load_or_create_key(path)
            except OSError as e:
                _warn_key_unavailable_once(e)
                return None
            _log_keys[path] = key
        return key


NO_KEY_HASH = "unavailable"


def keyed_digest(text: str) -> str:
    """Full HMAC-SHA256 hex digest of ``text`` under the install key ("unavailable" without one).

    Never falls back to an unkeyed hash: that would be reversible for short values.
    """
    key = log_hash_key()
    if key is None:
        return NO_KEY_HASH
    return hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()


def stable_hash(text: str) -> str:
    """First 12 hex chars of the keyed digest: stable on this install, not reversible offline."""
    return keyed_digest(text)[:12]


def error_fields(error: BaseException) -> dict[str, str]:
    """What to log about an exception: its type and a keyed hash of its message, never the text."""
    return {"error_type": type(error).__name__, "error_hash": stable_hash(str(error))}


_FRAME_LINE = re.compile(r'^[\s|]*File "[^"]*", line \d+, in \S+$')
_EXC_HEADER = re.compile(r"^[\s|+-]*([A-Za-z_][\w.]*(?:Error|Exception|Warning|Group|Exit|Interrupt|Iteration))\b")


def format_exception_redacted(error: BaseException) -> str:
    """Traceback built from the exception objects: frames and types only, never messages,
    notes or source lines. Follows causes, contexts and exception-group members."""
    lines: list[str] = []
    seen: set[int] = set()

    def walk(exc: BaseException, indent: str) -> None:
        if id(exc) in seen:
            return
        seen.add(id(exc))
        chained = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
        if chained is not None:
            walk(chained, indent)
            lines.append(f"{indent}(then, while handling it:)")
        lines.append(f"{indent}Traceback (most recent call last):")
        for frame in traceback.extract_tb(exc.__traceback__):
            lines.append(f'{indent}  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
        lines.append(f"{indent}{type(exc).__module__}.{type(exc).__qualname__}: [message redacted]")
        if isinstance(exc, BaseExceptionGroup):
            for i, member in enumerate(exc.exceptions, 1):
                lines.append(f"{indent}  +-- member {i}")
                walk(member, indent + "  | ")

    walk(error, "")
    return "\n".join(lines)


def redact_traceback(text: str) -> str:
    """For traceback text that was already rendered: keep frame lines and exception types only.
    Messages, continuation lines of multi-line messages, notes and source lines are dropped."""
    kept = []
    for line in text.splitlines():
        if line.startswith("Traceback") or _FRAME_LINE.match(line):
            kept.append(line)
            continue
        header = _EXC_HEADER.match(line)
        if header and (line.rstrip().endswith(header.group(1)) or line[header.end() :].startswith(":")):
            kept.append(f"{header.group(1)}: [message redacted]")
    return "\n".join(kept)


def _exc_from_exc_info(exc_info: Any) -> BaseException | None:
    if isinstance(exc_info, BaseException):
        return exc_info
    if isinstance(exc_info, tuple) and len(exc_info) == 3:
        return exc_info[1] if isinstance(exc_info[1], BaseException) else None
    if exc_info:
        return sys.exc_info()[1]
    return None


def format_exc_info_redacted(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    """Replaces structlog's format_exc_info: renders ``exc_info`` without exception text."""
    exc_info = event_dict.pop("exc_info", None)
    error = _exc_from_exc_info(exc_info)
    if error is not None:
        event_dict["exception"] = format_exception_redacted(error)
        event_dict["_exception_redacted"] = True
    return event_dict


def _redact_nested(value: Any, depth: int = 0) -> Any:
    """Copy of a dict/list with sensitive keys at any depth replaced by a size marker."""
    if depth > 8:
        return REDACTED
    if isinstance(value, Mapping):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            if is_sensitive_log_key(k):
                size = _size_of(v)
                out[k] = REDACTED if size is None else f"{REDACTED} size={size}"
            else:
                out[k] = _redact_nested(v, depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact_nested(v, depth + 1) for v in value]
    return value


# stdlib loggers whose messages are known not to carry data (uvicorn startup/shutdown lines).
FOREIGN_MESSAGE_ALLOWLIST: tuple[str, ...] = ("uvicorn", "uvicorn.error")


def redact_foreign_record(_logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]) -> Mapping[str, Any]:
    """For stdlib records: the rendered message can quote values (``%s`` args, f-strings), so it
    is replaced unless the logger is allowlisted. A ``%``-style template is kept (it is static);
    uvicorn access lines keep the path without its query string; warnings keep their category."""
    record = event_dict.get("_record")
    if record is None or event_dict.get("_from_structlog"):
        return event_dict
    # ProcessorFormatter clears record.args; pass_foreign_args=True hands them over here.
    args = event_dict.pop("positional_args", None) or ()
    event_dict["logger"] = record.name
    event_dict["lineno"] = record.lineno
    if record.name == "uvicorn.access" and isinstance(args, tuple) and len(args) == 5:
        client, method, full_path, http_version, status = args
        event_dict["event"] = f'{client} - "{method} {str(full_path).split("?", 1)[0]} HTTP/{http_version}" {status}'
        return event_dict
    if record.name == "py.warnings":
        first = str(record.msg).splitlines()[0] if record.msg else ""
        match = re.match(r"^(.*?:\d+): (\w+):", first)
        event_dict["event"] = (
            f"{match.group(1)}: {match.group(2)}: [message redacted]" if match else "warning [message redacted]"
        )
        return event_dict
    if record.name in FOREIGN_MESSAGE_ALLOWLIST:
        return event_dict
    if args:
        event_dict["event"] = f"{record.msg} [args redacted: {len(args) if isinstance(args, tuple) else 1}]"
    else:
        event_dict["event"] = f"[message redacted, {len(str(record.msg))} chars]"
    return event_dict


def text_fingerprint(text: str | None, prefix: str = "query") -> dict[str, str | int | None]:
    """``{prefix}_hash`` and ``{prefix}_len`` for a piece of user text, safe to log."""
    if text is None:
        return {f"{prefix}_hash": None, f"{prefix}_len": 0}
    return {f"{prefix}_hash": stable_hash(text), f"{prefix}_len": len(text)}


def _size_of(value: Any) -> int | None:
    if isinstance(value, (str, bytes, list, tuple, set, frozenset, dict)):
        return len(value)
    return None


def redact_sensitive_fields(_logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]) -> Mapping[str, Any]:
    """structlog processor: replace sensitive keys with ``<key>_hash`` / ``<key>_size``.

    Must run after ``format_exc_info`` so the rendered ``exception`` text is redacted too.
    """
    exception = event_dict.get("exception")
    # Text from format_exc_info_redacted is built without messages; anything else is parsed.
    if isinstance(exception, str) and not event_dict.pop("_exception_redacted", False):
        event_dict["exception"] = redact_traceback(exception)
    for key in TRACEBACK_LOG_KEYS:
        rendered = event_dict.get(key)
        if isinstance(rendered, str):
            event_dict[key] = redact_traceback(rendered)
    for key, value in list(event_dict.items()):
        if not is_sensitive_log_key(key) and isinstance(value, (Mapping, list, tuple, set, frozenset)):
            event_dict[key] = _redact_nested(value)
    for key in [k for k in event_dict if is_sensitive_log_key(k) and k not in TRACEBACK_LOG_KEYS]:
        value = event_dict.pop(key)
        if value is None:
            continue
        event_dict[f"{key}_redacted"] = True
        if isinstance(value, str) and key not in VALUE_LOG_KEYS:
            event_dict.setdefault(f"{key}_hash", stable_hash(value))
        size = _size_of(value)
        if size is not None:
            event_dict.setdefault(f"{key}_size", size)
    return event_dict


def resolve_log_level(raw: str | None = None) -> int:
    """Level name from ``raw`` or the env var; unknown names fail loudly."""
    name = (raw if raw is not None else os.environ.get(LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL)).strip().upper()
    level = logging.getLevelNamesMapping().get(name or DEFAULT_LOG_LEVEL)
    if level is None:
        raise ValueError(f"{LOG_LEVEL_ENV}={name!r} is not a logging level (use DEBUG, INFO, WARNING, ERROR)")
    return level


# Loggers that install their own handlers (uvicorn's dictConfig); they are pointed at ours.
_THIRD_PARTY_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "py.warnings")


class _RedactingHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Marker class so configure_logging can replace its own handler when called twice."""


def _shared_processors() -> list[Any]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        format_exc_info_redacted,
        redact_foreign_record,
        redact_sensitive_fields,
    ]


def configure_logging(level: int | None = None) -> None:
    """Route structlog and stdlib logging through one redacting handler. Safe to call twice."""
    resolved = level if level is not None else resolve_log_level()
    structlog.configure(
        processors=[*_shared_processors(), structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(resolved),
        cache_logger_on_first_use=False,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        pass_foreign_args=True,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=False, exception_formatter=structlog.dev.plain_traceback),
        ],
    )
    handler = _RedactingHandler(sys.stderr)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    for existing in [h for h in root.handlers if isinstance(h, _RedactingHandler)]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)
    for name in _THIRD_PARTY_LOGGERS:
        third_party = logging.getLogger(name)
        third_party.handlers.clear()
        third_party.propagate = True
        # uvicorn's config sets its own levels (log_level="error" silences access lines); the
        # root level decides instead, so nothing depends on which configured logging last.
        third_party.setLevel(logging.NOTSET)
        third_party.disabled = False
    # Re-hook every time: something (a test runner) may have replaced warnings.showwarning.
    logging.captureWarnings(False)
    logging.captureWarnings(True)

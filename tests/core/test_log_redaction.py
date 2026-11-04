"""No question text, filter value or row value may reach a log line (security item S2)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import structlog
from structlog.testing import LogCapture

from clinical_analytics.core.data_home import DATA_HOME_ENV
from clinical_analytics.core.log_redaction import (
    LOG_KEY_FILENAME,
    LOG_LEVEL_ENV,
    VALUE_LOG_KEYS,
    configure_logging,
    redact_sensitive_fields,
    resolve_log_level,
    stable_hash,
    text_fingerprint,
)
from clinical_analytics.core.query_service import QueryService

SECRET_VALUE = "quokkavirus"
QUESTION = f"how many patients have diagnosis {SECRET_VALUE}"
PREVIOUS_QUESTION = "count patients with marsupialfever"


@contextmanager
def _capture_structlog_all_levels() -> Iterator[list[dict[str, Any]]]:
    """Capture every structlog event, DEBUG included, with no redacting processor in the chain.

    Leaving the backstop processor out means the test checks what call sites hand to structlog.
    """
    previous = structlog.get_config()
    capture = LogCapture()
    structlog.configure(
        processors=[capture],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        cache_logger_on_first_use=False,
    )
    try:
        yield capture.entries
    finally:
        structlog.configure(**previous)


def test_ask_fullPipelineAllLevels_logsNeitherQuestionNorFilterValue(
    make_semantic_layer: Any, mock_llm_calls: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    layer = make_semantic_layer(
        data={
            "patient_id": [1, 2, 3, 4],
            "age": [45, 62, 38, 51],
            "diagnosis": [SECRET_VALUE, "flu", SECRET_VALUE, "flu"],
        }
    )
    service = QueryService(layer)
    history = [{"query": PREVIOUS_QUESTION, "intent": "COUNT", "filters_applied": [{"value": "marsupialfever"}]}]

    # Act
    with caplog.at_level(logging.DEBUG), _capture_structlog_all_levels() as events:
        service.ask(QUESTION, dataset_id="ds", conversation_history=history)

    # Assert
    structlog_text = repr(events)
    assert events, "the ask pipeline should log something"
    assert QUESTION not in structlog_text
    assert SECRET_VALUE not in structlog_text
    assert PREVIOUS_QUESTION not in structlog_text
    assert "marsupialfever" not in structlog_text
    assert SECRET_VALUE not in caplog.text
    assert QUESTION not in caplog.text
    assert any(event.get("query_hash") == stable_hash(QUESTION) for event in events)


@pytest.mark.parametrize(
    ("key", "value", "expected_size"),
    [
        ("query", "how many with HbA1c over 9", 26),
        ("question", "", 0),
        ("filters", [{"column": "dx", "value": "rare"}], 1),
        ("rows", [{"id": 1}, {"id": 2}], 2),
        ("value", 42, None),
    ],
)
def test_redactSensitiveFields_sensitiveKey_replacedByHashAndSize(
    key: str, value: Any, expected_size: int | None
) -> None:
    # Arrange
    event = {"event": "x", key: value, "dataset_id": "ds"}

    # Act
    out = redact_sensitive_fields(None, "info", event)

    # Assert
    assert key not in out
    assert out[f"{key}_redacted"] is True
    assert out.get(f"{key}_size") == expected_size
    assert out["dataset_id"] == "ds"
    if isinstance(value, str) and value:
        assert value not in repr(out)
        assert out[f"{key}_hash"] == stable_hash(value)


def test_redactSensitiveFields_noneValue_dropsKeyWithoutMarker() -> None:
    # Arrange
    event = {"event": "x", "query": None}

    # Act
    out = redact_sensitive_fields(None, "info", event)

    # Assert
    assert out == {"event": "x"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, {"query_hash": None, "query_len": 0}),
        ("", {"query_hash": stable_hash(""), "query_len": 0}),
        ("abc", {"query_hash": stable_hash("abc"), "query_len": 3}),
    ],
)
def test_textFingerprint_edgeCases_returnsHashAndLength(text: str | None, expected: dict[str, Any]) -> None:
    # Arrange / Act
    out = text_fingerprint(text)

    # Assert
    assert out == expected


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, logging.INFO), ("", logging.INFO), ("debug", logging.DEBUG), ("WARNING", logging.WARNING)],
)
def test_resolveLogLevel_envValues_returnsLevel(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: int
) -> None:
    # Arrange
    if env_value is None:
        monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    else:
        monkeypatch.setenv(LOG_LEVEL_ENV, env_value)

    # Act
    level = resolve_log_level()

    # Assert
    assert level == expected


def test_resolveLogLevel_unknownName_raisesValueError(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv(LOG_LEVEL_ENV, "loud")

    # Act / Assert
    with pytest.raises(ValueError, match=LOG_LEVEL_ENV):
        resolve_log_level()


def test_stableHash_keyedPerInstall_notPlainSha256AndDiffersAcrossDataHomes(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    import hashlib

    monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path / "a"))
    first = stable_hash("E11.9")
    monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path / "b"))

    # Act
    second = stable_hash("E11.9")

    # Assert
    assert first != hashlib.sha256(b"E11.9").hexdigest()[:12]
    assert first != second
    key_file = tmp_path / "a" / LOG_KEY_FILENAME
    assert key_file.stat().st_mode & 0o077 == 0
    monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path / "a"))
    assert stable_hash("E11.9") == first


@pytest.mark.parametrize("key", sorted(VALUE_LOG_KEYS))
def test_redactSensitiveFields_valueKey_logsSizeOnlyNoHash(key: str) -> None:
    # Arrange
    event = {"event": "x", key: "E11"}

    # Act
    out = redact_sensitive_fields(None, "info", event)

    # Assert
    assert f"{key}_hash" not in out
    assert out[f"{key}_size"] == 3


def test_redactSensitiveFields_renderedException_keepsTypeDropsMessage() -> None:
    # Arrange
    secret = "patient MRN " + "4471"  # built at runtime: traceback source lines show code, not values
    try:
        raise ValueError(secret)
    except ValueError:
        event = structlog.processors.format_exc_info(None, "error", {"event": "boom", "exc_info": True})

    # Act
    out = redact_sensitive_fields(None, "error", event)

    # Assert
    assert "4471" not in out["exception"]
    assert "ValueError: [message redacted]" in out["exception"]
    assert "Traceback" in out["exception"]


def test_configureLogging_stdlibRecordWithException_rendersRedacted(capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    previous = structlog.get_config()
    configure_logging(logging.INFO)
    stdlib_logger = logging.getLogger("clinical_analytics.test_stdlib_route")

    # Act
    try:
        value = "diagnosis=" + "quokka" + "virus"
        try:
            raise KeyError(value)
        except KeyError:
            stdlib_logger.exception("stdlib failure")
        structlog.get_logger().info("structlog line", question="how many with quokkavirus")
        err = capsys.readouterr().err
    finally:
        root = logging.getLogger()
        for handler in [h for h in root.handlers if type(h).__name__ == "_RedactingHandler"]:
            root.removeHandler(handler)
        structlog.configure(**previous)

    # Assert
    assert "stdlib failure" not in err  # f-string-style stdlib messages are redacted
    assert "[message redacted" in err
    assert "structlog line" in err
    assert "KeyError: [message redacted]" in err
    assert "quokkavirus" not in err


def _phi(tag: str) -> str:
    """Built at runtime so the value never appears in a traceback's source line."""
    return "".join([tag, "PHI"])


def test_configureLogging_everyKnownLeakPath_noValueReachesOutput(capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    import warnings

    import uvicorn

    previous = structlog.get_config()
    # As if a real uvicorn started earlier in this process (tests/api, or a CLI launch):
    # its default config installs its own handlers and sets uvicorn.access to ERROR.
    uvicorn.Config("clinical_analytics.api.main:app", log_level="error")
    configure_logging(logging.INFO)
    log = structlog.get_logger()
    std = logging.getLogger("clinical_analytics.leak_paths")
    noted = ValueError("x")
    noted.add_note(_phi("Note") + ": patient")

    # Act
    try:
        for error in (
            ExceptionGroup("grp", [ValueError(_phi("Group"))]),
            ValueError("line1 ok\n" + _phi("Multi") + " no colon here"),
            noted,
        ):
            try:
                raise error
            except Exception:  # noqa: BLE001 - every kind of exception is under test
                log.error("failure", exc_info=True)
                std.error("std failure", exc_info=True)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(_phi("Warn"), stacklevel=1)
        std.warning("std %s", _phi("Arg"))
        std.warning(f"std f-string {_phi('Fstr')}")
        log.info(
            "ev",
            error=_phi("Err"),
            sql=f"SELECT * WHERE name='{_phi('Sql')}'",
            message=_phi("Msg"),
            detail=_phi("Detail"),
            context={"plan": {"filters": [{"value": _phi("Nested")}]}},
            columns=[{"name": "dx", "value": _phi("Deep")}],
        )
        logging.getLogger("uvicorn.access").info(
            '%s - "%s %s HTTP/%s" %d', "127.0.0.1:5", "GET", f"/api/sessions?dataset_id={_phi('Qs')}", "1.1", 200
        )
        err = capsys.readouterr().err
    finally:
        root = logging.getLogger()
        for handler in [h for h in root.handlers if type(h).__name__ == "_RedactingHandler"]:
            root.removeHandler(handler)
        structlog.configure(**previous)

    # Assert
    assert "PHI" not in err
    assert "ExceptionGroup: [message redacted]" in err
    assert "UserWarning: [message redacted]" in err
    assert "std %s [args redacted: 1]" in err
    assert '"GET /api/sessions HTTP/1.1" 200' in err


@pytest.mark.parametrize("failure", [PermissionError(13, "denied"), OSError(95, "Operation not supported")])
def test_stableHash_keyFileCannotBeCreated_safeModeNoUnkeyedHash(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: OSError
) -> None:
    # Arrange
    from clinical_analytics.core import log_redaction

    monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path / "ro"))
    monkeypatch.setattr(log_redaction, "_key_warned", False)

    def broken(_path: Any) -> bytes:
        raise failure

    monkeypatch.setattr(log_redaction, "_load_or_create_key", broken)

    # Act
    first = stable_hash("E11.9")
    second = stable_hash("I10")
    out = redact_sensitive_fields(None, "info", {"event": "x", "query": "E11.9"})
    warned = capsys.readouterr().err

    # Assert
    assert first == second == log_redaction.NO_KEY_HASH[:12]
    assert out["query_size"] == 5
    assert warned.count("log hashing disabled") == 1


def test_loadOrCreateKey_noHardLinks_fallsBackToRename(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: exFAT-like filesystem where os.link is not supported.
    from clinical_analytics.core import log_redaction

    def no_link(*_a: Any) -> None:
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(log_redaction.os, "link", no_link)
    path = tmp_path / LOG_KEY_FILENAME

    # Act
    key = log_redaction._load_or_create_key(path)

    # Assert
    assert len(key) == 32
    assert path.read_bytes() == key
    assert [p.name for p in tmp_path.iterdir()] == [LOG_KEY_FILENAME]


CLINICAL_PHRASE = "".join(["HIV-associated ", "nephropathy in ward 7"])


@pytest.mark.parametrize(
    "key",
    ["raw_preview", "response_preview", "body_preview", "preview"],
    ids=["raw_preview", "response_preview", "any_suffix", "bare_preview"],
)
def test_redactSensitiveFields_previewKey_replacedByHashAndSize(key: str) -> None:
    # Arrange
    event = {"event": "x", key: CLINICAL_PHRASE, "nested": {key: CLINICAL_PHRASE}}

    # Act
    out = redact_sensitive_fields(None, "warning", event)

    # Assert
    assert CLINICAL_PHRASE not in repr(out)
    assert out[f"{key}_redacted"] is True
    assert out[f"{key}_size"] == len(CLINICAL_PHRASE)


@pytest.mark.parametrize("traceback_value", ["rendered", "", None], ids=["rendered", "empty", "none"])
def test_redactSensitiveFields_tracebackKey_keepsFramesDropsMessage(traceback_value: str | None) -> None:
    # Arrange
    import traceback as tb

    text: str | None = traceback_value
    if traceback_value == "rendered":
        try:
            raise ValueError(CLINICAL_PHRASE)
        except ValueError:
            text = tb.format_exc()
    event = {"event": "x", "traceback": text}

    # Act
    out = redact_sensitive_fields(None, "warning", event)

    # Assert
    assert CLINICAL_PHRASE not in repr(out)
    if traceback_value == "rendered":
        assert "ValueError: [message redacted]" in out["traceback"]
        assert 'File "' in out["traceback"]


def test_parseJsonResponse_malformedClinicalText_renderedWarningOmitsPhrase(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    from clinical_analytics.core.llm_json import parse_json_response

    previous = structlog.get_config()
    configure_logging(logging.INFO)
    malformed = '{"intent": "COUNT", "explanation": "' + CLINICAL_PHRASE

    # Act
    try:
        parsed = parse_json_response(malformed)
        err = capsys.readouterr().err
    finally:
        root = logging.getLogger()
        for handler in [h for h in root.handlers if type(h).__name__ == "_RedactingHandler"]:
            root.removeHandler(handler)
        structlog.configure(**previous)

    # Assert
    assert parsed is None
    assert "llm_json_parse_failed" in err
    assert "nephropathy" not in err
    assert "HIV" not in err


@pytest.mark.parametrize(
    "question",
    [
        f"what is the average {SECRET_VALUE}",
        f"count patients broken down by diagnosis {SECRET_VALUE}",
        f"how many patients per diagnosis {SECRET_VALUE}",
    ],
    ids=["what_is_average", "broken_down_by", "per_group"],
)
def test_ask_patternTierQuestionFragments_neverLoggedAtAnyLevel(
    make_semantic_layer: Any, mock_llm_calls: Any, question: str
) -> None:
    # Arrange
    layer = make_semantic_layer(
        data={"patient_id": [1, 2, 3, 4], "age": [45, 62, 38, 51], "diagnosis": ["a", "flu", "a", "flu"]}
    )
    service = QueryService(layer)

    # Act
    with _capture_structlog_all_levels() as events:
        service.ask(question, dataset_id="ds")

    # Assert
    assert events, "the ask pipeline should log something"
    assert SECRET_VALUE not in repr(events)


@pytest.mark.parametrize(
    "key",
    ["variable_term", "primary_term", "group_term", "group_phrase", "original", "filename", "failure_reason"],
)
def test_redactSensitiveFields_questionFragmentKey_replacedByHashAndSize(key: str) -> None:
    # Arrange
    event = {"event": "x", key: CLINICAL_PHRASE}

    # Act
    out = redact_sensitive_fields(None, "debug", event)

    # Assert
    assert CLINICAL_PHRASE not in repr(out)
    assert out[f"{key}_redacted"] is True
    assert out[f"{key}_size"] == len(CLINICAL_PHRASE)

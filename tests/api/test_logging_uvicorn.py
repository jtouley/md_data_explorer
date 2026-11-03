"""uvicorn's own logging setup must never bypass redaction (final review N1)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
import structlog
import uvicorn
from fastapi.testclient import TestClient


def _remove_our_handlers() -> None:
    root = logging.getLogger()
    for handler in [h for h in root.handlers if type(h).__name__ == "_RedactingHandler"]:
        root.removeHandler(handler)


def test_lifespan_uvicornReconfiguredLoggingAfterImport_accessLineRedacted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange: our handler first, then uvicorn's default dictConfig on top (a plain CLI launch).
    from clinical_analytics.api.main import app
    from clinical_analytics.core.log_redaction import configure_logging

    previous = structlog.get_config()
    configure_logging(logging.INFO)
    uvicorn.Config("clinical_analytics.api.main:app", log_level="error")

    # Act: app startup, then an access line carrying a value in its query string.
    try:
        with TestClient(app):
            pass
        logging.getLogger("uvicorn.access").info(
            '%s - "%s %s HTTP/%s" %d', "127.0.0.1:5", "GET", "/api/sessions?dataset_id=" + "Qs" + "PHI", "1.1", 200
        )
        err = capsys.readouterr().err
    finally:
        _remove_our_handlers()
        structlog.configure(**previous)

    # Assert
    assert '"GET /api/sessions HTTP/1.1" 200' in err
    assert "PHI" not in err


@pytest.mark.parametrize("argv", [[], ["--reload"], ["--port", "8123", "--log-level", "warning"]])
def test_serveMain_anyArgs_runsUvicornWithLogConfigNoneOnLoopback(argv: list[str]) -> None:
    # Arrange
    from clinical_analytics.api import serve

    # Act
    with patch.object(serve.uvicorn, "run") as run:
        serve.main(argv)

    # Assert
    kwargs: dict[str, Any] = run.call_args.kwargs
    assert kwargs["log_config"] is None
    assert kwargs["host"] == "127.0.0.1"
    assert run.call_args.args[0] == "clinical_analytics.api.main:app"


def test_serveMain_nonLoopbackHost_rejected() -> None:
    # Arrange
    from clinical_analytics.api import serve

    # Act / Assert
    with patch.object(serve.uvicorn, "run") as run, pytest.raises(SystemExit):
        serve.main(["--host", "0.0.0.0"])  # noqa: S104 - the point of the test
    run.assert_not_called()

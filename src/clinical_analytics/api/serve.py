"""Start the API with uvicorn's own logging config turned off.

uvicorn installs handlers (and levels) for its ``uvicorn.error`` / ``uvicorn.access`` loggers
unless ``log_config=None``; those handlers print access lines with query strings and raw
exception text, bypassing the redacting handler from ``core.log_redaction``. Every launcher
(scripts/run_app.sh, ``make run-api``, the e2e harnesses) starts the API through this module:

    python -m clinical_analytics.api.serve [--reload] [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

APP_IMPORT_PATH = "clinical_analytics.api.main:app"
DEFAULT_HOST = "127.0.0.1"  # loopback only: uploads may contain patient data
DEFAULT_PORT = 8000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Clinical Analytics API (loopback only).")
    parser.add_argument("--host", default=DEFAULT_HOST, choices=["127.0.0.1"], help="bind address (loopback only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true", help="development: restart on code changes")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    uvicorn.run(
        APP_IMPORT_PATH,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        log_config=None,  # keep our redacting handler in charge of uvicorn's loggers
    )


if __name__ == "__main__":
    main()

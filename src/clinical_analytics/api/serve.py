"""Start the API with uvicorn's own logging config turned off.

uvicorn installs handlers (and levels) for its ``uvicorn.error`` / ``uvicorn.access`` loggers
unless ``log_config=None``; those handlers print access lines with query strings and raw
exception text, bypassing the redacting handler from ``core.log_redaction``. Every launcher
(scripts/run_app.sh, ``make run-api``, the e2e harnesses) starts the API through this module:

    python -m clinical_analytics.api.serve [--reload] [--host 127.0.0.1] [--port 8000]

Before uvicorn starts, the data store is checked against the running DuckDB
(``storage.store_format.prepare_store_for_engine``). If it cannot be used (open in another
program, unreadable, or no room for the check's copy), or a store rewrite left pending by a
failed delete cannot finish, one plain line goes to stderr and the process exits with
``STORE_CHECK_FAILED_EXIT``, instead of a uvicorn lifespan traceback. The same happens when the
bundled embedding model is missing (``core.embeddings``; it is never downloaded).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import uvicorn

from clinical_analytics.api.parent_watchdog import start_parent_watchdog, start_stdin_watchdog
from clinical_analytics.core.data_home import DataHomeOwnershipError, data_home, ensure_data_home, uploads_dir
from clinical_analytics.core.embeddings import EmbeddingModelMissingError, check_embedding_model
from clinical_analytics.core.log_redaction import configure_logging
from clinical_analytics.storage.dataset_deletion import StoreCompactionPendingError, finish_pending_compaction
from clinical_analytics.storage.datastore import StoreInUseError
from clinical_analytics.storage.store_format import StoreFormatError, prepare_store_for_engine

APP_IMPORT_PATH = "clinical_analytics.api.main:app"
DEFAULT_HOST = "127.0.0.1"  # loopback only: uploads may contain patient data
DEFAULT_PORT = 8000
STORE_CHECK_FAILED_EXIT = 2
MODEL_CHECK_CACHE = "embedding_model_check.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Clinical Analytics API (loopback only).")
    parser.add_argument("--host", default=DEFAULT_HOST, choices=["127.0.0.1"], help="bind address (loopback only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true", help="development: restart on code changes")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    parser.add_argument(
        "--exit-with-parent",
        type=int,
        default=None,
        metavar="PID",
        help="shut down when process PID exits (the desktop app passes its own PID)",
    )
    parser.add_argument(
        "--stop-on-stdin-close",
        action="store_true",
        help="shut down gracefully when stdin closes (the desktop app's stop request)",
    )
    return parser


def check_store() -> str | None:
    """Run the startup store check; return a one-line reason if the API must not start."""
    data_dir = uploads_dir().parent
    try:
        ensure_data_home()
        prepare_store_for_engine(data_dir)
        # A rewrite left pending by a failed delete must finish before anything reads the store.
        finish_pending_compaction(data_dir)
    except (StoreFormatError, StoreCompactionPendingError, DataHomeOwnershipError) as e:
        return str(e)
    except StoreInUseError:
        return "The data store is in use by this app. Close every Clinical Analytics window and start it again."
    except OSError as e:
        return f"Could not check the data store before starting ({e.strerror or type(e).__name__})."
    return None


def check_bundled_model() -> str | None:
    """One-line reason if the bundled embedding model is missing or not the pinned bytes."""
    try:
        check_embedding_model(verify=True, cache_path=data_home() / "cache" / MODEL_CHECK_CACHE)
    except EmbeddingModelMissingError as e:
        return str(e)
    return None


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging()
    # Store first: it creates the data home owner-only, where the model check keeps its cache.
    reason = check_store() or check_bundled_model()
    if reason is not None:
        print(f"Clinical Analytics cannot start: {' '.join(reason.split())}", file=sys.stderr)
        sys.exit(STORE_CHECK_FAILED_EXIT)
    if args.exit_with_parent is not None:
        start_parent_watchdog(args.exit_with_parent)
    if args.stop_on_stdin_close:
        start_stdin_watchdog()
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

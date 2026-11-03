"""FastAPI application entry point.

Main application setup with CORS, startup hooks, and route registration.

To run:
    uvicorn clinical_analytics.api.main:app --reload --port 8000

Reference: docs/architecture/LIGHTWEIGHT_UI_ARCHITECTURE.md
"""

import asyncio
import os
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import structlog
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from clinical_analytics.api import health_llm
from clinical_analytics.api.db.database import create_tables

# Import routes
from clinical_analytics.api.routes import datasets, enrichments, queries, sessions
from clinical_analytics.api.security import (
    NONCE_PATTERN,
    BearerTokenMiddleware,
    allowed_hosts,
    auth_required,
    configured_api_token,
    log_auth_mode,
    token_proof,
)
from clinical_analytics.core.data_home import uploads_dir
from clinical_analytics.core.log_redaction import configure_logging
from clinical_analytics.core.ollama_init import initialize_ollama
from clinical_analytics.storage.dataset_deletion import (
    DUCKDB_FILENAME,
    DatasetDeletionError,
    compaction_marker,
    resume_pending_compaction,
)

# Redacting processor chain + INFO default (CLINICAL_ANALYTICS_LOG_LEVEL overrides).
configure_logging()
logger = structlog.get_logger()

# Opt-in so tests and bare uvicorn runs never start downloads; `make run-api` sets it.
OLLAMA_AUTOINIT_ENV = "CLINICAL_ANALYTICS_OLLAMA_AUTOINIT"


STORE_RECOVERY_DOC = "docs/security/LOCAL_DATA_SECURITY.md (Recovering from a pending store rewrite)"


class StoreCompactionPendingError(RuntimeError):
    """A delete left dropped-table bytes in the store and the rewrite still cannot finish.

    The API refuses to start rather than serve queries from that file.
    """


def _ollama_autoinit_enabled() -> bool:
    return os.getenv(OLLAMA_AUTOINIT_ENV, "").strip().lower() in {"1", "true", "yes"}


def _initialize_ollama_in_background() -> threading.Thread:
    """Check Ollama and pull a missing model without blocking startup (a pull can take minutes)."""

    def _run() -> None:
        status = initialize_ollama(auto_download=True)
        logger.info("ollama_autoinit_finished", ready=status["ready"], message=status["message"])

    thread = threading.Thread(target=_run, name="ollama-autoinit", daemon=True)
    thread.start()
    return thread


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Runs on startup and shutdown:
    - Startup: Create database tables; when CLINICAL_ANALYTICS_OLLAMA_AUTOINIT is set,
      check Ollama and pull a missing model in a background thread
    - Shutdown: Close connections, cleanup resources

    Args:
        app: FastAPI application instance

    Yields:
        None: Control returns to application during runtime
    """
    # Startup. Re-apply the redacting log setup: a uvicorn started without log_config=None has
    # by now installed its own handlers and levels on uvicorn.error / uvicorn.access.
    configure_logging()
    logger.info("api_starting")
    # Fails fast on a weak token; warns when running without one (dev mode).
    log_auth_mode()

    # Create database tables (in production, use Alembic migrations)
    create_tables()
    logger.info("api_database_ready")

    # A delete that failed between DROP and the store rewrite leaves a marker: finish it now,
    # so dropped rows do not linger in analytics.duckdb. If it cannot finish, the API does not
    # start: serving would read the file that still holds a deleted dataset's bytes.
    data_dir = uploads_dir().parent
    try:
        if await asyncio.to_thread(resume_pending_compaction, data_dir):
            logger.info("api_store_compaction_resumed")
    except DatasetDeletionError as e:
        marker = compaction_marker(data_dir / DUCKDB_FILENAME)
        logger.error(
            "api_store_compaction_failed",
            error_type=type(e).__name__,
            cause_type=type(e.__cause__).__name__ if e.__cause__ else None,
            marker=marker.name,
            recovery=f"API not started; see {STORE_RECOVERY_DOC}",
        )
        raise StoreCompactionPendingError(
            f"A deleted dataset is still in the data store ({marker.name} is present) and the "
            f"store rewrite failed ({type(e.__cause__ or e).__name__}). The API will not start "
            f"until it succeeds; see {STORE_RECOVERY_DOC}."
        ) from e

    if _ollama_autoinit_enabled():
        _initialize_ollama_in_background()

    yield

    # Shutdown
    logger.info("api_shutting_down")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Clinical Analytics API",
    description="REST API for clinical data analytics platform (replaces Streamlit UI)",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ============================================================================
# CORS Middleware
# ============================================================================

# Allow frontend origins (Next.js + Electron dev servers)
# Note: Electron in dev mode uses localhost, not file://
# The packaged Electron app serves its page from the privileged app:// scheme (electron/src/main.js),
# so its requests carry this exact Origin. "null" (file:// pages, sandboxed iframes) is never allowed.
ELECTRON_APP_ORIGIN = "app://clinical-analytics"
DEFAULT_ALLOWED_ORIGINS = [
    ELECTRON_APP_ORIGIN,
    "http://localhost:5173",  # Electron/Vite dev server
    "http://127.0.0.1:5173",  # Electron/Vite dev (IP)
    "http://localhost:3000",  # Next.js dev
    "http://127.0.0.1:3000",  # Next.js dev (IP)
]


def cors_origins() -> list[str]:
    """Allowed origins from CORS_ORIGINS (comma list) or the defaults; "null" is rejected."""
    raw = os.getenv("CORS_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()] or DEFAULT_ALLOWED_ORIGINS
    if any(o.lower() == "null" or o == "*" for o in origins):
        raise ValueError('CORS_ORIGINS may not contain "null" or "*"')
    return origins


ALLOWED_ORIGINS = cors_origins()

# Order matters: the last middleware added runs first. CORS answers preflights (which carry no
# Authorization header) and adds CORS headers to 401s; TrustedHost then rejects non-loopback
# Host headers (DNS rebinding); the bearer check runs last. See api/security.py.
app.add_middleware(BearerTokenMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts())
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # bearer header, no cookies
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ============================================================================
# Route Registration
# ============================================================================


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check(
    nonce: Annotated[
        str | None,
        Query(pattern=NONCE_PATTERN.pattern, description="Random value; the reply proves the server knows the token"),
    ] = None,
) -> dict[str, Any]:
    """Health check endpoint for monitoring and desktop LLM status."""
    token = configured_api_token()
    body: dict[str, Any] = {
        "status": "healthy",
        "service": "clinical-analytics-api",
        # Electron refuses a backend that reports False (unless it was itself started in dev mode).
        "auth_required": auth_required(),
        # HMAC(token, nonce): lets the app tell this server from a program squatting on the port.
        "token_proof": token_proof(token, nonce) if token and nonce else None,
    }
    body.update(health_llm.get_ollama_health_snapshot())
    return body


# Register API routes
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
app.include_router(datasets.router, prefix="/api", tags=["datasets"])
app.include_router(queries.router, prefix="/api", tags=["queries"])
app.include_router(enrichments.router, prefix="/api", tags=["enrichments"])


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    from clinical_analytics.api.serve import main

    main(["--reload"])

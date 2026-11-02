"""FastAPI application entry point.

Main application setup with CORS, startup hooks, and route registration.

To run:
    uvicorn clinical_analytics.api.main:app --reload --port 8000

Reference: docs/architecture/LIGHTWEIGHT_UI_ARCHITECTURE.md
"""

import os
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from clinical_analytics.api import health_llm
from clinical_analytics.api.db.database import create_tables

# Import routes
from clinical_analytics.api.routes import datasets, enrichments, queries, sessions
from clinical_analytics.core.ollama_init import initialize_ollama

logger = structlog.get_logger()

# Opt-in so tests and bare uvicorn runs never start downloads; `make run-api` sets it.
OLLAMA_AUTOINIT_ENV = "CLINICAL_ANALYTICS_OLLAMA_AUTOINIT"


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
    # Startup
    logger.info("api_starting")

    # Create database tables (in production, use Alembic migrations)
    create_tables()
    logger.info("api_database_ready")

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
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    ",".join(
        [
            "http://localhost:3000",  # Next.js dev
            "http://127.0.0.1:3000",  # Next.js dev (IP)
            "http://localhost:8000",  # FastAPI (for SSE same-origin)
            "http://127.0.0.1:8000",  # FastAPI (IP)
            "http://localhost:5173",  # Electron/Vite dev server
            "http://127.0.0.1:5173",  # Electron/Vite dev (IP)
        ]
    ),
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ============================================================================
# Route Registration
# ============================================================================


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint for monitoring and desktop LLM status."""
    body: dict[str, Any] = {
        "status": "healthy",
        "service": "clinical-analytics-api",
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
    import uvicorn

    uvicorn.run(
        "clinical_analytics.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

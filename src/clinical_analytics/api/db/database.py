"""Database configuration and session management.

Provides SQLAlchemy engine, session factory, and dependency injection
for FastAPI routes.

Development: SQLite (file-based, no external service)
Production: PostgreSQL (scalable, persistent)
"""

import os
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from clinical_analytics.api.models.database import Base

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/clinical_analytics.db")

# Create engine
# For SQLite: check_same_thread=False allows multi-threaded access (FastAPI default)
# For PostgreSQL: use pool_size and max_overflow for connection pooling
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",  # Log SQL for debugging
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def create_tables() -> None:
    """Create all database tables.

    Called on application startup to ensure schema exists.
    In production, use Alembic migrations instead.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database session injection.

    Usage:
        @app.get("/api/sessions")
        async def get_sessions(db: Annotated[Session, Depends(get_db)]):
            ...

    Yields:
        Session: SQLAlchemy session for database operations

    Note:
        Session is automatically closed after request completes (finally block).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Type alias for dependency injection
DBSession = Annotated[Session, Depends(get_db)]

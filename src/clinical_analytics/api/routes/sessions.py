"""Session management API routes.

Endpoints:
- POST /api/sessions - Create new session
- GET /api/sessions/{session_id} - Retrieve session
- GET /api/sessions - List sessions (with filters)
- DELETE /api/sessions/{session_id} - Delete session
"""

import secrets
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from clinical_analytics.api.db.database import get_db
from clinical_analytics.api.models import database as db_models
from clinical_analytics.api.models import schemas

router = APIRouter()


def generate_session_id() -> str:
    """Generate unique session ID.

    Format: sess_{random_hex}
    """
    return f"sess_{secrets.token_hex(8)}"


# ============================================================================
# POST /api/sessions - Create Session
# ============================================================================


@router.post("/sessions", response_model=schemas.SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: schemas.SessionCreate,
    db: Annotated[Session, Depends(get_db)],
) -> schemas.SessionResponse:
    """Create a new conversation session.

    Creates a new session associated with a dataset. Sessions persist across
    browser refreshes and can be browsed/searched later.

    Args:
        request: Session creation request with dataset_id and optional metadata
        db: Database session (injected)

    Returns:
        SessionResponse: Created session with session_id, timestamps

    Example:
        POST /api/sessions
        {
            "dataset_id": "upload_abc123",
            "metadata": {"user_agent": "Mozilla/5.0", "theme": "dark"}
        }

        Response (201):
        {
            "session_id": "sess_a1b2c3d4e5f6g7h8",
            "dataset_id": "upload_abc123",
            "created_at": "2026-01-03T15:30:00Z",
            "updated_at": "2026-01-03T15:30:00Z",
            "message_count": 0
        }
    """
    # Generate unique session ID
    session_id = generate_session_id()

    # Create database record
    db_session = db_models.Session(
        session_id=session_id,
        dataset_id=request.dataset_id,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    # Return response
    return schemas.SessionResponse(
        session_id=db_session.session_id,
        dataset_id=db_session.dataset_id,
        created_at=db_session.created_at,
        updated_at=db_session.updated_at,
        message_count=0,  # New session has no messages
    )


# ============================================================================
# GET /api/sessions/{session_id} - Retrieve Session
# ============================================================================


@router.get("/sessions/{session_id}", response_model=schemas.SessionResponse)
async def get_session(
    session_id: Annotated[str, Path(..., description="Session ID to retrieve")],
    db: Annotated[Session, Depends(get_db)],
) -> schemas.SessionResponse:
    """Retrieve a session by ID.

    Args:
        session_id: Session identifier
        db: Database session (injected)

    Returns:
        SessionResponse: Session details with message count

    Raises:
        HTTPException: 404 if session not found

    Example:
        GET /api/sessions/sess_a1b2c3d4e5f6g7h8

        Response (200):
        {
            "session_id": "sess_a1b2c3d4e5f6g7h8",
            "dataset_id": "upload_abc123",
            "created_at": "2026-01-03T15:30:00Z",
            "updated_at": "2026-01-03T15:35:00Z",
            "message_count": 4
        }
    """
    # Query database
    db_session = db.query(db_models.Session).filter(db_models.Session.session_id == session_id).first()

    if not db_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )

    # Count messages in session
    message_count = len(db_session.messages)

    return schemas.SessionResponse(
        session_id=db_session.session_id,
        dataset_id=db_session.dataset_id,
        created_at=db_session.created_at,
        updated_at=db_session.updated_at,
        message_count=message_count,
    )


# ============================================================================
# GET /api/sessions - List Sessions
# ============================================================================


@router.get("/sessions", response_model=schemas.SessionListResponse)
async def list_sessions(
    db: Annotated[Session, Depends(get_db)],
    dataset_id: Annotated[str | None, Query(description="Filter by dataset ID")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Max sessions to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Number of sessions to skip")] = 0,
) -> schemas.SessionListResponse:
    """List sessions with optional filters.

    Args:
        dataset_id: Optional dataset filter
        limit: Maximum number of sessions to return (1-100, default 50)
        offset: Number of sessions to skip for pagination (default 0)
        db: Database session (injected)

    Returns:
        SessionListResponse: List of sessions and total count

    Example:
        GET /api/sessions?dataset_id=upload_abc123&limit=10&offset=0

        Response (200):
        {
            "sessions": [
                {
                    "session_id": "sess_xyz789",
                    "dataset_id": "upload_abc123",
                    "created_at": "2026-01-03T14:00:00Z",
                    "updated_at": "2026-01-03T14:30:00Z",
                    "message_count": 10
                },
                ...
            ],
            "total": 42
        }
    """
    # Build query
    query = db.query(db_models.Session)

    # Apply dataset filter
    if dataset_id:
        query = query.filter(db_models.Session.dataset_id == dataset_id)

    # Get total count (before pagination)
    total = query.count()

    # Apply pagination and ordering
    db_sessions = query.order_by(db_models.Session.updated_at.desc()).offset(offset).limit(limit).all()

    # Convert to response models
    sessions = [
        schemas.SessionResponse(
            session_id=s.session_id,
            dataset_id=s.dataset_id,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=len(s.messages),
        )
        for s in db_sessions
    ]

    return schemas.SessionListResponse(sessions=sessions, total=total)


# ============================================================================
# DELETE /api/sessions/{session_id} - Delete Session
# ============================================================================


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: Annotated[str, Path(..., description="Session ID to delete")],
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete a session and all its messages.

    Args:
        session_id: Session identifier
        db: Database session (injected)

    Raises:
        HTTPException: 404 if session not found

    Example:
        DELETE /api/sessions/sess_a1b2c3d4e5f6g7h8

        Response (204): No content
    """
    # Query database
    db_session = db.query(db_models.Session).filter(db_models.Session.session_id == session_id).first()

    if not db_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )

    # Delete session (cascade deletes messages)
    db.delete(db_session)
    db.commit()

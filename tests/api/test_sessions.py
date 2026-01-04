"""Tests for session management API endpoints.

Tests cover:
- POST /api/sessions - Create new session
- GET /api/sessions/{session_id} - Retrieve session
- GET /api/sessions - List sessions (with filters)
- DELETE /api/sessions/{session_id} - Delete session

TDD Workflow:
1. Write failing test (RED) ✅ This file
2. Implement route (GREEN)
3. Verify test passes
"""

import pytest
from clinical_analytics.api.db.database import get_db
from clinical_analytics.api.models.database import Base
from clinical_analytics.api.routes import sessions
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# ============================================================================
# Test Database Setup
# ============================================================================


@pytest.fixture(scope="function")
def test_app():
    """Create FastAPI test app without lifespan."""
    # Create test app without lifespan (skip startup table creation)
    app = FastAPI(
        title="Clinical Analytics API (Test)",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Register routes
    app.include_router(sessions.router, prefix="/api", tags=["sessions"])

    return app


@pytest.fixture(scope="function")
def test_db(test_app):
    """Create test database for each test function.

    Uses in-memory SQLite for fast, isolated tests.
    """
    # Create in-memory SQLite database with StaticPool to share connection
    # This ensures all sessions use the same in-memory database
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session factory
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Override get_db dependency
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    test_app.dependency_overrides[get_db] = override_get_db

    yield engine

    # Cleanup
    Base.metadata.drop_all(bind=engine)
    test_app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(test_app, test_db):
    """FastAPI test client with test database."""
    return TestClient(test_app)


# ============================================================================
# POST /api/sessions - Create Session
# ============================================================================


def test_unit_sessions_createSession_returnsSessionId(client):
    """Test creating a new session returns session_id and metadata.

    Arrange: Valid session creation request
    Act: POST /api/sessions
    Assert: 201 Created, response contains session_id, dataset_id, timestamps
    """
    # Arrange
    request_data = {
        "dataset_id": "test_dataset_123",
        "metadata": {"user_agent": "test-client", "theme": "dark"},
    }

    # Act
    response = client.post("/api/sessions", json=request_data)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["session_id"].startswith("sess_")  # Session ID format
    assert data["dataset_id"] == "test_dataset_123"
    assert "created_at" in data
    assert "updated_at" in data
    assert data["message_count"] == 0  # New session has no messages


def test_unit_sessions_createSessionMinimal_succeeds(client):
    """Test creating session with minimal data (no metadata).

    Arrange: Request with only dataset_id (no metadata)
    Act: POST /api/sessions
    Assert: 201 Created, session created successfully
    """
    # Arrange
    request_data = {"dataset_id": "minimal_dataset"}

    # Act
    response = client.post("/api/sessions", json=request_data)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["dataset_id"] == "minimal_dataset"


def test_unit_sessions_createSessionMissingDataset_returns422(client):
    """Test creating session without dataset_id fails validation.

    Arrange: Request with missing dataset_id
    Act: POST /api/sessions
    Assert: 422 Unprocessable Entity (Pydantic validation error)
    """
    # Arrange
    request_data = {}  # Missing dataset_id

    # Act
    response = client.post("/api/sessions", json=request_data)

    # Assert
    assert response.status_code == 422  # Validation error


# ============================================================================
# GET /api/sessions/{session_id} - Retrieve Session
# ============================================================================


def test_unit_sessions_getSession_returnsSessionDetails(client):
    """Test retrieving an existing session by ID.

    Arrange: Create a session first
    Act: GET /api/sessions/{session_id}
    Assert: 200 OK, response contains session details
    """
    # Arrange: Create session
    create_response = client.post("/api/sessions", json={"dataset_id": "test_dataset"})
    session_id = create_response.json()["session_id"]

    # Act: Retrieve session
    response = client.get(f"/api/sessions/{session_id}")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["dataset_id"] == "test_dataset"
    assert "created_at" in data
    assert "updated_at" in data


def test_unit_sessions_getSessionNotFound_returns404(client):
    """Test retrieving non-existent session returns 404.

    Arrange: Non-existent session_id
    Act: GET /api/sessions/{session_id}
    Assert: 404 Not Found
    """
    # Arrange
    session_id = "sess_nonexistent_123"

    # Act
    response = client.get(f"/api/sessions/{session_id}")

    # Assert
    assert response.status_code == 404
    data = response.json()
    assert "error" in data or "detail" in data


# ============================================================================
# GET /api/sessions - List Sessions
# ============================================================================


def test_unit_sessions_listSessions_returnsAllSessions(client):
    """Test listing all sessions.

    Arrange: Create multiple sessions
    Act: GET /api/sessions
    Assert: 200 OK, response contains all sessions
    """
    # Arrange: Create 3 sessions
    client.post("/api/sessions", json={"dataset_id": "dataset_1"})
    client.post("/api/sessions", json={"dataset_id": "dataset_2"})
    client.post("/api/sessions", json={"dataset_id": "dataset_1"})

    # Act
    response = client.get("/api/sessions")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    assert "total" in data
    assert data["total"] == 3
    assert len(data["sessions"]) == 3


def test_unit_sessions_listSessionsFilterByDataset_returnsFiltered(client):
    """Test listing sessions filtered by dataset_id.

    Arrange: Create sessions for different datasets
    Act: GET /api/sessions?dataset_id=dataset_1
    Assert: 200 OK, only sessions for dataset_1 returned
    """
    # Arrange: Create sessions for different datasets
    client.post("/api/sessions", json={"dataset_id": "dataset_1"})
    client.post("/api/sessions", json={"dataset_id": "dataset_2"})
    client.post("/api/sessions", json={"dataset_id": "dataset_1"})

    # Act
    response = client.get("/api/sessions", params={"dataset_id": "dataset_1"})

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2  # Only 2 sessions for dataset_1
    for session in data["sessions"]:
        assert session["dataset_id"] == "dataset_1"


def test_unit_sessions_listSessionsEmpty_returnsEmptyList(client):
    """Test listing sessions when none exist.

    Arrange: No sessions created
    Act: GET /api/sessions
    Assert: 200 OK, empty sessions list
    """
    # Arrange: (no sessions created)

    # Act
    response = client.get("/api/sessions")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert len(data["sessions"]) == 0


# ============================================================================
# DELETE /api/sessions/{session_id} - Delete Session
# ============================================================================


def test_unit_sessions_deleteSession_succeeds(client):
    """Test deleting a session.

    Arrange: Create a session
    Act: DELETE /api/sessions/{session_id}
    Assert: 204 No Content, session deleted
    """
    # Arrange: Create session
    create_response = client.post("/api/sessions", json={"dataset_id": "test_dataset"})
    session_id = create_response.json()["session_id"]

    # Act: Delete session
    response = client.delete(f"/api/sessions/{session_id}")

    # Assert
    assert response.status_code == 204  # No Content

    # Verify session is deleted
    get_response = client.get(f"/api/sessions/{session_id}")
    assert get_response.status_code == 404


def test_unit_sessions_deleteSessionNotFound_returns404(client):
    """Test deleting non-existent session returns 404.

    Arrange: Non-existent session_id
    Act: DELETE /api/sessions/{session_id}
    Assert: 404 Not Found
    """
    # Arrange
    session_id = "sess_nonexistent_456"

    # Act
    response = client.delete(f"/api/sessions/{session_id}")

    # Assert
    assert response.status_code == 404

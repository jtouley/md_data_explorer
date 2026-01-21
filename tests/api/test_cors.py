"""Tests for CORS configuration.

Phase 4 of Electron UI Migration: Verify CORS allows Electron origins.

Tests cover:
- CORS headers allow Electron dev ports (localhost:5173, localhost:8000)
- Health endpoint accessible from Electron origins
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from clinical_analytics.api.main import app

    return TestClient(app)


class TestCORSConfiguration:
    """Tests for CORS middleware configuration."""

    def test_cors_allows_electron_dev_origin(self, test_client):
        """CORS headers allow Electron dev server origin (localhost:5173)."""
        # Act: Make request with Electron dev server Origin header
        response = test_client.get(
            "/health",
            headers={"Origin": "http://localhost:5173"},
        )

        # Assert: Response includes CORS headers for Electron origin
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_allows_backend_origin(self, test_client):
        """CORS headers allow backend origin for SSE (localhost:8000)."""
        # Act: Make request with backend Origin header
        response = test_client.get(
            "/health",
            headers={"Origin": "http://localhost:8000"},
        )

        # Assert: Response includes CORS headers for backend origin
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:8000"

    def test_cors_preflight_allows_electron_origin(self, test_client):
        """CORS preflight (OPTIONS) allows Electron origin with required headers."""
        # Act: Make OPTIONS preflight request
        response = test_client.options(
            "/api/datasets",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        # Assert: Preflight response allows the origin
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert "GET" in response.headers.get("access-control-allow-methods", "")

    def test_cors_still_allows_nextjs_origin(self, test_client):
        """CORS still allows existing Next.js origin (backward compatibility)."""
        # Act: Make request with Next.js Origin header
        response = test_client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )

        # Assert: Response includes CORS headers for Next.js origin
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

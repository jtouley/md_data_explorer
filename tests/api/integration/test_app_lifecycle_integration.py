"""
Integration tests for FastAPI application lifecycle.

Phase 5: Tests startup/shutdown hooks and middleware.
"""

import pytest
import requests
from sqlalchemy import inspect


@pytest.mark.integration
@pytest.mark.slow
class TestAppLifecycleIntegration:
    """Integration tests for FastAPI application lifecycle."""

    def test_integration_app_startup_createsTables(self, real_server, test_db):
        """Test that app startup creates database tables."""
        # Tables should be created by lifespan hook

        inspector = inspect(test_db)
        tables = inspector.get_table_names()

        assert "sessions" in tables
        assert "messages" in tables

    def test_integration_app_healthCheck_respondsWhenReady(self, real_server):
        """Test health check endpoint responds correctly."""
        response = requests.get(f"{real_server}/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "clinical-analytics-api"

    def test_integration_app_cors_allowsConfiguredOrigins(self, real_server):
        """Test CORS middleware allows configured origins."""
        response = requests.options(f"{real_server}/api/sessions", headers={"Origin": "http://localhost:3000"})

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

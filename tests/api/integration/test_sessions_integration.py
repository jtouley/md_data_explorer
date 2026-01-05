"""
Integration tests for Sessions API with real FastAPI server.

Phase 3: Tests complete session lifecycle with real server.
"""

import concurrent.futures

import pytest
import requests


@pytest.mark.integration
@pytest.mark.slow
class TestSessionsAPIIntegration:
    """Integration tests for Sessions API with real FastAPI server."""

    def test_integration_sessions_createRetrieveDelete_fullWorkflow(self, real_server, test_db):
        """Test complete session lifecycle with real server.

        Arrange: Real FastAPI server running
        Act: Create → Retrieve → Delete session via HTTP
        Assert: All operations succeed, session persists across requests
        """
        # Create session
        response = requests.post(f"{real_server}/api/sessions", json={"dataset_id": "test_dataset"})
        assert response.status_code == 201
        session_id = response.json()["session_id"]

        # Retrieve session (separate request)
        response = requests.get(f"{real_server}/api/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["session_id"] == session_id

        # Delete session
        response = requests.delete(f"{real_server}/api/sessions/{session_id}")
        assert response.status_code == 204

        # Verify deleted
        response = requests.get(f"{real_server}/api/sessions/{session_id}")
        assert response.status_code == 404

    def test_integration_sessions_concurrent_isolationWorks(self, real_server, test_db):
        """Test concurrent session creation with real database."""

        def create_session(i):
            return requests.post(f"{real_server}/api/sessions", json={"dataset_id": f"dataset_{i}"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_session, i) for i in range(20)]
            responses = [f.result() for f in futures]

        # All should succeed
        assert all(r.status_code == 201 for r in responses)

        # All should have unique session IDs
        session_ids = [r.json()["session_id"] for r in responses]
        assert len(set(session_ids)) == 20

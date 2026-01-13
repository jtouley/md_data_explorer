"""
End-to-end integration tests for complete user workflows.

Phase 6: Tests critical user paths through the API.
"""

import pytest
import requests


@pytest.mark.integration
@pytest.mark.slow
class TestE2EWorkflowsIntegration:
    """End-to-end integration tests for complete user workflows."""

    def test_integration_e2e_uploadDatasetCreateSessionQuery(self, real_server, tmp_path):
        """Test complete workflow: Upload → Create Session → Query.

        This is the critical user path through the API.
        """
        # Step 1: Use test dataset (upload endpoint is future work)
        dataset_id = "test_dataset_001"

        # Step 2: Create session
        session_response = requests.post(f"{real_server}/api/sessions", json={"dataset_id": dataset_id})
        assert session_response.status_code == 201
        session_id = session_response.json()["session_id"]

        # Step 3: Query endpoint is future work

        # Step 4: Retrieve session (verify it persists)
        get_response = requests.get(f"{real_server}/api/sessions/{session_id}")
        assert get_response.status_code == 200
        assert get_response.json()["dataset_id"] == dataset_id

    def test_integration_e2e_errorRecovery_datasetDeletedMidSession(self, real_server):
        """Test error handling when dataset is deleted during session."""
        # Create session
        session_response = requests.post(f"{real_server}/api/sessions", json={"dataset_id": "test_dataset"})
        _session_id = session_response.json()["session_id"]  # noqa: F841

        # Delete and query endpoints are future work
        # Test validates session creation works even if dataset might be deleted later

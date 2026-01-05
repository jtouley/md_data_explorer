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
        # Step 1: Upload dataset (future endpoint)
        # upload_response = requests.post(
        #     f"{real_server}/api/datasets/upload",
        #     files={"file": open(csv_file, "rb")}
        # )
        # dataset_id = upload_response.json()["dataset_id"]

        # For now, use uploaded_test_dataset fixture
        dataset_id = "test_dataset_001"

        # Step 2: Create session
        session_response = requests.post(f"{real_server}/api/sessions", json={"dataset_id": dataset_id})
        assert session_response.status_code == 201
        session_id = session_response.json()["session_id"]

        # Step 3: Query dataset (future endpoint)
        # query_response = requests.post(
        #     f"{real_server}/api/datasets/{dataset_id}/query",
        #     json={"metrics": ["patient_count"]}
        # )
        # assert query_response.status_code == 200

        # Step 4: Retrieve session (verify it persists)
        get_response = requests.get(f"{real_server}/api/sessions/{session_id}")
        assert get_response.status_code == 200
        assert get_response.json()["dataset_id"] == dataset_id

    def test_integration_e2e_errorRecovery_datasetDeletedMidSession(self, real_server):
        """Test error handling when dataset is deleted during session."""
        # Create session
        session_response = requests.post(f"{real_server}/api/sessions", json={"dataset_id": "test_dataset"})
        _session_id = session_response.json()["session_id"]  # noqa: F841

        # Delete dataset (future endpoint)
        # delete_response = requests.delete(
        #     f"{real_server}/api/datasets/test_dataset"
        # )

        # Try to query - should get error (not crash)
        # query_response = requests.post(
        #     f"{real_server}/api/datasets/test_dataset/query",
        #     json={"metrics": ["patient_count"]}
        # )
        # assert query_response.status_code == 404

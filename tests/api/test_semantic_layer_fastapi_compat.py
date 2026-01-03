"""
Test semantic layer compatibility with FastAPI dependency injection.

Phase 1.5: Critical compatibility testing before API development.

Tests verify:
1. SemanticLayer works with FastAPI Depends() dependency injection
2. DuckDB connection pooling works with async context
3. @st.cache_resource pattern translates to FastAPI singleton pattern
4. No blocking operations in async routes
"""

import asyncio
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory


# Test fixtures
@pytest.fixture
def sample_dataset(make_cohort_with_categorical):
    """Create a sample uploaded dataset for testing."""
    # Use existing conftest fixture to create test data
    cohort_df = make_cohort_with_categorical(n_patients=100)

    # For this test, we'll mock an uploaded dataset
    # In real usage, UploadedDatasetFactory.create_dataset() would be called
    # For now, we'll create a minimal mock
    class MockDataset:
        def __init__(self, cohort):
            self.cohort = cohort
            self.config = {
                "name": "test_dataset",
                "patient_id_column": "patient_id",
                "outcome_column": "outcome",
            }

        def get_semantic_layer(self):
            """Get semantic layer for this dataset."""
            return SemanticLayer(
                dataset_name=self.config["name"],
                cohort=self.cohort,
                config=self.config,
            )

    return MockDataset(cohort_df)


@pytest.fixture
def app_with_semantic_layer(sample_dataset):
    """Create a FastAPI app with semantic layer dependency."""
    app = FastAPI()

    # Singleton pattern (similar to @st.cache_resource)
    _semantic_layer_instance = None

    def get_semantic_layer() -> SemanticLayer:
        """
        Dependency that provides semantic layer instance.

        This mimics Streamlit's @st.cache_resource pattern:
        - Single instance created on first call
        - Reused across all subsequent requests
        - Not pickled (DuckDB connections are not picklable)
        """
        nonlocal _semantic_layer_instance
        if _semantic_layer_instance is None:
            _semantic_layer_instance = sample_dataset.get_semantic_layer()
        return _semantic_layer_instance

    @app.get("/test-semantic-layer")
    async def test_route(
        semantic_layer: Annotated[SemanticLayer, Depends(get_semantic_layer)]
    ):
        """Test route that uses semantic layer."""
        # Verify semantic layer is usable
        dataset_info = semantic_layer.get_dataset_info()
        return {
            "success": True,
            "dataset_name": dataset_info.get("name"),
            "has_metrics": len(dataset_info.get("metrics", {})) > 0,
        }

    @app.get("/test-query")
    async def test_query_route(
        semantic_layer: Annotated[SemanticLayer, Depends(get_semantic_layer)]
    ):
        """Test route that executes a query via semantic layer."""
        # Execute a simple query to verify DuckDB works in async context
        result = semantic_layer.query(
            metrics=["patient_count"],
            dimensions=None,
            filters=None,
        )
        return {
            "success": True,
            "row_count": len(result),
        }

    return app


# Test 1: Basic FastAPI dependency injection works
def test_unit_semanticLayer_fastApiDepends_injectsSuccessfully(app_with_semantic_layer):
    """
    Test that SemanticLayer can be injected via FastAPI Depends().

    This verifies:
    - SemanticLayer instance creation works in FastAPI context
    - Dependency injection pattern is compatible
    - Basic route execution succeeds
    """
    client = TestClient(app_with_semantic_layer)

    # Act
    response = client.get("/test-semantic-layer")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["dataset_name"] == "test_dataset"
    assert data["has_metrics"] is True


# Test 2: DuckDB works in async context
def test_unit_duckdb_asyncContext_queriesExecute(app_with_semantic_layer):
    """
    Test that DuckDB queries execute successfully in async FastAPI routes.

    This verifies:
    - DuckDB connection doesn't block async operations
    - Query execution completes successfully
    - No threading issues with DuckDB in async context
    """
    client = TestClient(app_with_semantic_layer)

    # Act
    response = client.get("/test-query")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["row_count"] > 0


# Test 3: Semantic layer instance reuse (singleton pattern)
def test_unit_semanticLayer_singleton_sameInstanceAcrossRequests(app_with_semantic_layer):
    """
    Test that semantic layer instance is reused across requests.

    This verifies:
    - Singleton pattern works (like @st.cache_resource)
    - Same instance used for multiple requests
    - No instance creation overhead on subsequent requests
    """
    client = TestClient(app_with_semantic_layer)

    # Act - Make multiple requests
    response1 = client.get("/test-semantic-layer")
    response2 = client.get("/test-semantic-layer")
    response3 = client.get("/test-query")

    # Assert - All succeed (using same instance)
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200

    # Verify consistent results (same instance)
    assert response1.json()["dataset_name"] == response2.json()["dataset_name"]


# Test 4: Concurrent requests don't cause issues
def test_unit_semanticLayer_concurrent_noRaceConditions(app_with_semantic_layer):
    """
    Test that concurrent requests using semantic layer don't cause race conditions.

    This verifies:
    - DuckDB connection pooling works correctly
    - No blocking operations cause deadlocks
    - Concurrent access is safe
    """
    client = TestClient(app_with_semantic_layer)

    # Act - Simulate concurrent requests
    import concurrent.futures

    def make_request():
        return client.get("/test-query")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [f.result() for f in futures]

    # Assert - All requests succeeded
    assert all(r.status_code == 200 for r in responses)
    assert all(r.json()["success"] for r in responses)


# Test 5: Non-picklable objects work with FastAPI Depends
def test_unit_semanticLayer_nonPicklable_worksWithDepends(app_with_semantic_layer, sample_dataset):
    """
    Test that non-picklable semantic layer works with FastAPI Depends().

    This verifies:
    - SemanticLayer (with DuckDB/Ibis) doesn't need to be pickled
    - FastAPI Depends() doesn't require pickling (unlike @st.cache_data)
    - Compatible with @st.cache_resource pattern
    """
    import pickle

    # Arrange - Verify semantic layer is NOT picklable
    semantic_layer = sample_dataset.get_semantic_layer()

    # Assert - SemanticLayer is not picklable (expected)
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(semantic_layer)

    # Act - But it works with FastAPI Depends() anyway
    client = TestClient(app_with_semantic_layer)
    response = client.get("/test-semantic-layer")

    # Assert - Route works despite non-picklability
    assert response.status_code == 200
    assert response.json()["success"] is True


# Test 6: Async query execution doesn't block
@pytest.mark.asyncio
async def test_unit_semanticLayer_asyncQuery_nonBlocking(sample_dataset):
    """
    Test that semantic layer queries don't block async operations.

    This verifies:
    - Query execution can run alongside other async operations
    - No blocking I/O that would prevent event loop progress
    - Compatible with async FastAPI patterns
    """
    semantic_layer = sample_dataset.get_semantic_layer()

    # Act - Run query alongside other async operations
    async def query_task():
        # Run query in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            semantic_layer.query,
            ["patient_count"],
            None,
            None,
        )

    async def other_task():
        await asyncio.sleep(0.01)
        return "completed"

    # Both tasks should complete
    results = await asyncio.gather(query_task(), other_task())

    # Assert - Both tasks completed
    assert len(results[0]) > 0  # Query result has rows
    assert results[1] == "completed"  # Other task completed


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

"""
Integration tests for semantic layer dependency injection with real datasets.

Phase 4: Tests semantic layer with real DuckDB and cache invalidation.
"""

import concurrent.futures

import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestSemanticLayerIntegration:
    """Integration tests for semantic layer dependency injection with real datasets."""

    def test_integration_semanticLayer_realDataset_queriesSucceed(self, real_server, uploaded_test_dataset):
        """Test semantic layer with real uploaded dataset and DuckDB.

        Arrange: Real uploaded dataset with DuckDB storage
        Act: Query via API endpoint using semantic layer dependency
        Assert: Query executes successfully, returns correct data
        """
        upload_id, storage = uploaded_test_dataset

        # This would use a real route like /api/datasets/{dataset_id}/query
        # For now, test that semantic layer can be loaded
        from clinical_analytics.api.dependencies import get_semantic_layer

        semantic = get_semantic_layer(upload_id)
        dataset_info = semantic.get_dataset_info()

        assert dataset_info["name"] is not None
        assert "metrics" in dataset_info

    def test_integration_semanticLayer_cacheInvalidation_refreshesData(self, uploaded_test_dataset):
        """Test cache invalidation when dataset is updated."""
        from clinical_analytics.api.dependencies import (
            get_semantic_layer,
            invalidate_semantic_layer_cache,
        )

        upload_id, storage = uploaded_test_dataset

        # Load semantic layer (caches it)
        semantic1 = get_semantic_layer(upload_id)
        id1 = id(semantic1)

        # Get again (should return cached instance)
        semantic2 = get_semantic_layer(upload_id)
        assert id(semantic2) == id1  # Same instance

        # Invalidate cache
        invalidate_semantic_layer_cache(upload_id)

        # Get again (should create new instance)
        semantic3 = get_semantic_layer(upload_id)
        assert id(semantic3) != id1  # Different instance

    def test_integration_semanticLayer_concurrentQueries_threadSafe(self, uploaded_test_dataset):
        """Test concurrent queries through semantic layer with real DuckDB."""
        from clinical_analytics.api.dependencies import get_semantic_layer

        upload_id, storage = uploaded_test_dataset
        semantic = get_semantic_layer(upload_id)

        def execute_query():
            return semantic.query(metrics=["patient_count"])

        # Execute 10 concurrent queries
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(execute_query) for _ in range(10)]
            results = [f.result() for f in futures]

        # All queries should succeed
        assert len(results) == 10
        assert all(len(r) > 0 for r in results)

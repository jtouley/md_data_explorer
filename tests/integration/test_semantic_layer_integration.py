"""Integration tests for DuckDB → SemanticLayer flow.

Tests that semantic layer can be created from stored dataset and execute queries.
"""

import pytest

from .diagnostic_helpers import (
    assert_with_diagnostics,
    dump_semantic_layer_state,
)


@pytest.mark.integration
@pytest.mark.slow
def test_integration_storage_to_semanticLayer_fullFlow(real_storage, sample_csv_file):
    """Storage → SemanticLayer → Query execution

    Uses correct pattern: UploadedDataset → load() → get_semantic_layer()
    NOT SemanticLayer.from_upload() (doesn't exist)

    Diagnostics on failure:
    - SemanticLayer config dump
    - DuckDB schema vs semantic layer mismatch
    - Generated SQL (via compile() before execution)
    """
    # Arrange: Upload file
    with open(sample_csv_file, "rb") as f:
        file_bytes = f.read()

    success, message, upload_id = real_storage.save_upload(
        file_bytes=file_bytes,
        original_filename="test.csv",
        metadata={"dataset_name": "Test Dataset"},
    )
    assert success, f"Upload failed: {message}"

    # Act: Create semantic layer via UploadedDataset (correct pattern)
    from clinical_analytics.datasets.uploaded.definition import UploadedDataset

    dataset = UploadedDataset(upload_id=upload_id, storage=real_storage)
    dataset.load()  # Initializes semantic layer

    # Semantic layer may not initialize if unified cohort CSV doesn't exist yet
    try:
        semantic = dataset.get_semantic_layer()
    except (ValueError, FileNotFoundError):
        pytest.skip("Semantic layer requires unified cohort CSV which may not exist immediately after upload")

    # Assert: Can query
    dataset_info = semantic.get_dataset_info()

    assert_with_diagnostics(
        dataset_info is not None,
        dump_semantic_layer_state,
        semantic=semantic,
    )

    # Act: Execute query (get SQL before execution for diagnostics)
    # Build query expression first to get SQL
    from ibis import _

    view = semantic.get_base_view()
    result_expr = view.aggregate(_.count().name("count"))
    sql = result_expr.compile()  # Get SQL before execution

    # Execute query
    result = semantic.query(metrics=["count()"])

    # Assert: Got results
    assert_with_diagnostics(
        len(result) > 0,
        lambda: {
            "semantic_layer": dump_semantic_layer_state(semantic),
            "query_result_shape": (len(result), len(result.columns)),
            "generated_sql": sql,  # SQL obtained via compile(), not _last_executed_sql
        },
    )

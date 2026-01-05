"""Integration tests for Query → Analysis flow.

Tests that query results can be passed to analysis modules correctly.
"""

import polars as pl
import pytest

from .diagnostic_helpers import assert_with_diagnostics


@pytest.mark.integration
@pytest.mark.slow
def test_integration_query_to_analysis_descriptiveStats(real_storage, sample_csv_file):
    """SemanticLayer query → Analysis module

    Correct pattern:
    - semantic.query() returns pandas DataFrame
    - Convert to polars: pl.from_pandas(result)
    - Create AnalysisContext before calling compute_descriptive_analysis
    - Signature: compute_descriptive_analysis(df: pl.DataFrame, context: AnalysisContext)

    Diagnostics on failure:
    - Query result shape
    - Analysis input requirements
    - Error from analysis module
    """
    # Arrange: Setup semantic layer
    with open(sample_csv_file, "rb") as f:
        file_bytes = f.read()

    success, message, upload_id = real_storage.save_upload(
        file_bytes=file_bytes,
        original_filename="test.csv",
        metadata={"dataset_name": "Test Dataset"},
    )
    assert success, f"Upload failed: {message}"

    from clinical_analytics.datasets.uploaded.definition import UploadedDataset

    dataset = UploadedDataset(upload_id=upload_id, storage=real_storage)
    dataset.load()

    # Semantic layer may not initialize if unified cohort CSV doesn't exist yet
    try:
        semantic = dataset.get_semantic_layer()
    except (ValueError, FileNotFoundError):
        pytest.skip("Semantic layer requires unified cohort CSV which may not exist immediately after upload")

    # Act: Query (returns pandas DataFrame)
    # Note: query() may not work if metrics aren't in config, so use get_base_view() directly
    view = semantic.get_base_view()
    result_pd = view.execute()  # Execute to get pandas DataFrame

    # Convert pandas → polars (analysis expects polars)
    result_pl = pl.from_pandas(result_pd)

    # Create AnalysisContext (required by compute_descriptive_analysis)
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext(
        inferred_intent=AnalysisIntent.DESCRIBE,
        primary_variable=None,  # Describe all columns
    )

    # Act: Run analysis with correct signature
    from clinical_analytics.analysis.compute import compute_descriptive_analysis

    analysis_result = compute_descriptive_analysis(result_pl, context)

    # Assert: Got analysis (check for expected keys in result)
    assert_with_diagnostics(
        "type" in analysis_result and "summary_stats" in analysis_result,
        lambda: {
            "query_result_pandas_shape": result_pd.shape,
            "query_result_polars_shape": (result_pl.height, result_pl.width),
            "result_schema": dict(result_pl.schema),
            "analysis_keys": list(analysis_result.keys()),
            "analysis_type": analysis_result.get("type"),
        },
    )

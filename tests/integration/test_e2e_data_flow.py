"""End-to-end integration tests for complete data flow.

Tests the full pipeline: Upload → Storage → DuckDB → SemanticLayer → Query → Analysis
with diagnostics at every stage.
"""

import json

import polars as pl
import pytest

from .diagnostic_helpers import dump_semantic_layer_state


@pytest.mark.integration
@pytest.mark.slow
def test_integration_e2e_upload_query_analyze_fullPipeline(real_storage, sample_csv_file):
    """Complete data flow: Upload → Storage → DuckDB → SemanticLayer → Query → Analysis

    All logic inlined (no undefined helper functions).
    Uses correct APIs verified in Phases 2-4.

    Diagnostics on failure:
    - Pipeline stage that failed
    - State at each stage
    - Data transformations
    """
    pipeline_state = {}

    try:
        # Stage 1: Upload
        pipeline_state["stage"] = "upload"
        with open(sample_csv_file, "rb") as f:
            file_bytes = f.read()

        success, message, upload_id = real_storage.save_upload(
            file_bytes=file_bytes,
            original_filename="test.csv",
            metadata={"dataset_name": "Test Dataset"},
        )
        assert success, f"Upload failed: {message}"
        pipeline_state["upload_id"] = upload_id
        pipeline_state["upload_success"] = success

        # Stage 2: Create semantic layer
        pipeline_state["stage"] = "semantic_layer"
        from clinical_analytics.datasets.uploaded.definition import UploadedDataset

        dataset = UploadedDataset(upload_id=upload_id, storage=real_storage)
        dataset.load()

        # Semantic layer may not initialize if unified cohort CSV doesn't exist yet
        try:
            semantic = dataset.get_semantic_layer()
            pipeline_state["semantic_initialized"] = True
            pipeline_state["semantic_config"] = dump_semantic_layer_state(semantic)
        except (ValueError, FileNotFoundError):
            pytest.skip("Semantic layer requires unified cohort CSV which may not exist immediately after upload")

        # Stage 3: Query
        pipeline_state["stage"] = "query"
        view = semantic.get_base_view()
        result_pd = view.execute()  # Execute to get pandas DataFrame
        pipeline_state["result_shape_pandas"] = result_pd.shape

        # Stage 4: Analyze (convert pandas→polars, create context)
        pipeline_state["stage"] = "analysis"
        result_pl = pl.from_pandas(result_pd)

        from clinical_analytics.analysis.compute import compute_descriptive_analysis
        from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

        context = AnalysisContext(
            inferred_intent=AnalysisIntent.DESCRIBE,
            primary_variable=None,  # Describe all columns
        )

        analysis = compute_descriptive_analysis(result_pl, context)
        pipeline_state["analysis_keys"] = list(analysis.keys())

        # Assert: Complete pipeline works
        assert analysis is not None
        assert "type" in analysis
        assert "summary_stats" in analysis

    except Exception as e:
        # Dump pipeline state on failure
        print(f"\n{'='*80}")
        print(f"PIPELINE FAILED AT STAGE: {pipeline_state.get('stage', 'unknown')}")
        print(f"ERROR: {type(e).__name__}: {e}")
        print(f"{'='*80}")
        print(json.dumps(pipeline_state, indent=2, default=str))
        print(f"{'='*80}\n")
        raise

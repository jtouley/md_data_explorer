"""
Tests for schema inference engine.

Tests schema inference with documentation context integration.
"""

import polars as pl

from clinical_analytics.core.schema_inference import InferredSchema, SchemaInferenceEngine


class TestInferSchemaWithDocContext:
    """Test suite for infer_schema() with doc_context parameter."""

    def test_infer_schema_accepts_doc_context_parameter(self):
        """Test that infer_schema() accepts doc_context parameter."""
        # Arrange: Create test DataFrame
        df = pl.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "age": [25, 30, 35],
                "status": [1, 0, 1],
            }
        )

        engine = SchemaInferenceEngine()
        doc_context = "Column descriptions: patient_id is unique identifier, age is patient age in years"

        # Act: Call infer_schema with doc_context
        schema = engine.infer_schema(df, doc_context=doc_context)

        # Assert: Schema is returned (test passes if signature accepts parameter)
        assert isinstance(schema, InferredSchema)
        assert schema.patient_id_column == "patient_id"

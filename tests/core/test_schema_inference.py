"""
Tests for schema inference engine.

Tests schema inference with documentation context integration.
"""


import polars as pl

from clinical_analytics.core.schema_inference import (
    DictionaryMetadata,
    InferredSchema,
    SchemaInferenceEngine,
)


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


class TestParseDictionaryText:
    """Test suite for parse_dictionary_text() method."""

    def test_parse_dictionary_text_accepts_path(self, tmp_path):
        """Test that parse_dictionary_text() accepts Path parameter."""
        # Arrange: Create test text file
        text_file = tmp_path / "dictionary.txt"
        text_file.write_text("patient_id: Unique patient identifier\nage: Patient age in years")

        engine = SchemaInferenceEngine()

        # Act: Call parse_dictionary_text with Path
        result = engine.parse_dictionary_text(text_file)

        # Assert: DictionaryMetadata is returned
        assert isinstance(result, DictionaryMetadata)
        assert "patient_id" in result.column_descriptions

    def test_parse_dictionary_text_accepts_string(self):
        """Test that parse_dictionary_text() accepts str parameter."""
        # Arrange: Create test text content
        text_content = "patient_id: Unique patient identifier\nage: Patient age in years"

        engine = SchemaInferenceEngine()

        # Act: Call parse_dictionary_text with str
        result = engine.parse_dictionary_text(text_content)

        # Assert: DictionaryMetadata is returned
        assert isinstance(result, DictionaryMetadata)
        assert "patient_id" in result.column_descriptions
        assert result.column_descriptions["patient_id"] == "Unique patient identifier"

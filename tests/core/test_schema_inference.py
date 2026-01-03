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


class TestExtractCodebooksFromDocs:
    """Test suite for extract_codebooks_from_docs() function."""

    def test_extract_codebooks_from_docs_parses_comma_separated_pattern(self):
        """Test that extract_codebooks_from_docs() parses '1: Biktarvy, 2: Symtuza' patterns."""
        from clinical_analytics.core.schema_inference import extract_codebooks_from_docs

        # Arrange: Documentation text with codebook pattern
        doc_text = "Current Regimen: 1: Biktarvy, 2: Symtuza, 3: Triumeq"

        # Act: Extract codebooks
        codebooks = extract_codebooks_from_docs(doc_text)

        # Assert: Codebooks extracted correctly
        assert isinstance(codebooks, dict)
        assert "current_regimen" in codebooks or "Current Regimen" in codebooks
        # Check that codebook has correct structure
        codebook_key = "current_regimen" if "current_regimen" in codebooks else "Current Regimen"
        assert codebooks[codebook_key] == {"1": "Biktarvy", "2": "Symtuza", "3": "Triumeq"}

    def test_extract_codebooks_from_docs_parses_space_separated_pattern(self):
        """Test that extract_codebooks_from_docs() parses '1: Yes 2: No' patterns."""
        from clinical_analytics.core.schema_inference import extract_codebooks_from_docs

        # Arrange: Documentation text with space-separated codebook pattern
        doc_text = "Status: 1: Yes 2: No 0: n/a"

        # Act: Extract codebooks
        codebooks = extract_codebooks_from_docs(doc_text)

        # Assert: Codebooks extracted correctly
        assert isinstance(codebooks, dict)
        codebook_key = "status" if "status" in codebooks else "Status"
        assert codebooks[codebook_key] == {"1": "Yes", "2": "No", "0": "n/a"}

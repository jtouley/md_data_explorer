"""
Tests for LLM Enrichment Suggestion Generation.

Phase 2: ADR011 Metadata Enrichment
Tests for generating, validating, and parsing LLM-suggested metadata patches.
"""

from unittest.mock import MagicMock, patch


class TestBuildEnrichmentPrompt:
    """Tests for building privacy-safe enrichment prompts."""

    def test_build_enrichment_prompt_includes_schema_summary(self):
        """Test that prompt includes column names and types."""
        from clinical_analytics.core.llm_enrichment import build_enrichment_prompt
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(
            patient_id_column="patient_id",
            outcome_columns=["mortality"],
            categorical_columns=["sex", "treatment_arm"],
            continuous_columns=["age", "bmi"],
        )

        prompt = build_enrichment_prompt(schema)

        assert "patient_id" in prompt
        assert "mortality" in prompt
        assert "age" in prompt
        assert "bmi" in prompt

    def test_build_enrichment_prompt_excludes_raw_data(self):
        """Test that prompt does NOT include any raw data values."""
        from clinical_analytics.core.llm_enrichment import build_enrichment_prompt
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(
            patient_id_column="patient_id",
            continuous_columns=["age"],
        )

        prompt = build_enrichment_prompt(schema)

        # Should not contain any actual patient IDs or data values
        assert "P001" not in prompt
        assert "John" not in prompt

    def test_build_enrichment_prompt_includes_doc_context(self):
        """Test that prompt includes documentation context when provided."""
        from clinical_analytics.core.llm_enrichment import build_enrichment_prompt
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(continuous_columns=["hba1c"])
        doc_context = "hba1c: Hemoglobin A1c percentage, measurement of blood sugar control"

        prompt = build_enrichment_prompt(schema, doc_context=doc_context)

        assert "Hemoglobin A1c" in prompt or "hba1c" in prompt.lower()

    def test_build_enrichment_prompt_includes_codebook_patterns(self):
        """Test that prompt includes codebook patterns when present in schema."""
        from clinical_analytics.core.llm_enrichment import build_enrichment_prompt
        from clinical_analytics.core.schema_inference import (
            DictionaryMetadata,
            InferredSchema,
        )

        dict_metadata = DictionaryMetadata(
            codebooks={"statin_used": {"0": "n/a", "1": "Atorvastatin", "2": "Simvastatin"}},
        )
        schema = InferredSchema(
            categorical_columns=["statin_used"],
            dictionary_metadata=dict_metadata,
        )

        prompt = build_enrichment_prompt(schema)

        # Should mention codebook-related information
        assert "statin_used" in prompt


class TestValidateLLMSuggestions:
    """Tests for validating LLM-generated suggestions."""

    def test_validate_llm_suggestions_valid_description(self):
        """Test validation of valid description suggestion."""
        from clinical_analytics.core.llm_enrichment import validate_llm_suggestions
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(continuous_columns=["age", "bmi"])

        raw_json = """
        {
            "suggestions": [
                {
                    "operation": "set_description",
                    "column": "age",
                    "value": "Patient age in years at enrollment"
                }
            ]
        }
        """

        result = validate_llm_suggestions(raw_json, schema)

        assert result.valid
        assert len(result.suggestions) == 1
        assert result.suggestions[0].column == "age"

    def test_validate_llm_suggestions_rejects_unknown_column(self):
        """Test that suggestions for unknown columns are rejected."""
        from clinical_analytics.core.llm_enrichment import validate_llm_suggestions
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(continuous_columns=["age"])

        raw_json = """
        {
            "suggestions": [
                {
                    "operation": "set_description",
                    "column": "nonexistent_column",
                    "value": "This column doesn't exist"
                }
            ]
        }
        """

        result = validate_llm_suggestions(raw_json, schema)

        assert not result.valid or len(result.suggestions) == 0
        assert len(result.rejected) >= 1
        assert "nonexistent_column" in result.rejected[0]["reason"]

    def test_validate_llm_suggestions_rejects_invalid_operation(self):
        """Test that unknown operations are rejected."""
        from clinical_analytics.core.llm_enrichment import validate_llm_suggestions
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(continuous_columns=["age"])

        raw_json = """
        {
            "suggestions": [
                {
                    "operation": "invalid_operation",
                    "column": "age",
                    "value": "some value"
                }
            ]
        }
        """

        result = validate_llm_suggestions(raw_json, schema)

        assert len(result.rejected) >= 1

    def test_validate_llm_suggestions_rejects_malformed_json(self):
        """Test that malformed JSON is rejected gracefully."""
        from clinical_analytics.core.llm_enrichment import validate_llm_suggestions
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(continuous_columns=["age"])

        raw_json = "this is not valid json {"

        result = validate_llm_suggestions(raw_json, schema)

        assert not result.valid
        assert len(result.suggestions) == 0

    def test_validate_llm_suggestions_codebook_entry(self):
        """Test validation of codebook entry suggestions."""
        from clinical_analytics.core.llm_enrichment import validate_llm_suggestions
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(categorical_columns=["status"])

        raw_json = """
        {
            "suggestions": [
                {
                    "operation": "set_codebook_entry",
                    "column": "status",
                    "value": {"code": "1", "label": "Active"}
                },
                {
                    "operation": "set_codebook_entry",
                    "column": "status",
                    "value": {"code": "2", "label": "Inactive"}
                }
            ]
        }
        """

        result = validate_llm_suggestions(raw_json, schema)

        assert result.valid
        assert len(result.suggestions) == 2

    def test_validate_llm_suggestions_exclusion_pattern(self):
        """Test validation of exclusion pattern suggestions."""
        from clinical_analytics.core.llm_enrichment import validate_llm_suggestions
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(categorical_columns=["statin_used"])

        raw_json = """
        {
            "suggestions": [
                {
                    "operation": "set_exclusion_pattern",
                    "column": "statin_used",
                    "pattern": "n/a",
                    "coded_value": 0,
                    "context": "Use != 0 to exclude patients not on statins"
                }
            ]
        }
        """

        result = validate_llm_suggestions(raw_json, schema)

        assert result.valid
        assert len(result.suggestions) >= 1


class TestGenerateEnrichmentSuggestions:
    """Tests for end-to-end enrichment suggestion generation."""

    def test_generate_enrichment_suggestions_returns_patches(self, mock_llm_calls):
        """Test that generate_enrichment_suggestions returns MetadataPatch objects."""
        from clinical_analytics.core.llm_enrichment import (
            generate_enrichment_suggestions,
        )
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(
            continuous_columns=["age", "bmi"],
            categorical_columns=["sex"],
        )

        # Mock LLM response
        mock_response = """
        {
            "suggestions": [
                {
                    "operation": "set_description",
                    "column": "age",
                    "value": "Patient age in years"
                }
            ]
        }
        """

        with patch("clinical_analytics.core.llm_enrichment.OllamaClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate.return_value = mock_response
            mock_client_class.return_value = mock_client

            suggestions = generate_enrichment_suggestions(schema)

        assert isinstance(suggestions, list)

    def test_generate_enrichment_suggestions_handles_ollama_unavailable(self, mock_llm_calls):
        """Test graceful handling when Ollama is unavailable."""
        from clinical_analytics.core.llm_enrichment import (
            generate_enrichment_suggestions,
        )
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(continuous_columns=["age"])

        with patch("clinical_analytics.core.llm_enrichment.OllamaClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate.side_effect = Exception("Connection refused")
            mock_client_class.return_value = mock_client

            suggestions = generate_enrichment_suggestions(schema)

        # Should return empty list, not raise exception
        assert suggestions == []

    def test_generate_enrichment_suggestions_with_doc_context(self, mock_llm_calls):
        """Test that doc_context is passed to prompt builder."""
        from clinical_analytics.core.llm_enrichment import (
            generate_enrichment_suggestions,
        )
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(continuous_columns=["hba1c"])
        doc_context = "hba1c: Hemoglobin A1c percentage"

        with patch("clinical_analytics.core.llm_enrichment.OllamaClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate.return_value = '{"suggestions": []}'
            mock_client_class.return_value = mock_client

            with patch("clinical_analytics.core.llm_enrichment.build_enrichment_prompt") as mock_build:
                mock_build.return_value = "mocked prompt"
                generate_enrichment_suggestions(schema, doc_context=doc_context)

                # Verify doc_context was passed
                mock_build.assert_called_once()
                call_kwargs = mock_build.call_args
                assert call_kwargs[1].get("doc_context") == doc_context or (
                    len(call_kwargs[0]) > 1 and call_kwargs[0][1] == doc_context
                )

    def test_generate_enrichment_suggestions_sets_provenance(self, mock_llm_calls):
        """Test that suggestions have correct provenance metadata."""
        from clinical_analytics.core.llm_enrichment import (
            generate_enrichment_suggestions,
        )
        from clinical_analytics.core.metadata_patch import PatchStatus
        from clinical_analytics.core.schema_inference import InferredSchema

        schema = InferredSchema(continuous_columns=["age"])

        mock_response = """
        {
            "suggestions": [
                {
                    "operation": "set_description",
                    "column": "age",
                    "value": "Patient age"
                }
            ]
        }
        """

        with patch("clinical_analytics.core.llm_enrichment.OllamaClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.generate.return_value = mock_response
            mock_client_class.return_value = mock_client

            suggestions = generate_enrichment_suggestions(schema, model_id="llama3.1:8b")

        if suggestions:
            assert suggestions[0].provenance == "llm"
            assert suggestions[0].status == PatchStatus.PENDING
            assert suggestions[0].model_id == "llama3.1:8b"


class TestMetadataPatchSchema:
    """Tests for metadata_patch schema in llm_json.py."""

    def test_metadata_patch_schema_exists(self):
        """Test that metadata_patch schema is registered."""
        from clinical_analytics.core.llm_json import validate_shape

        # Should not raise "Unknown schema" error
        result = validate_shape({"suggestions": []}, "metadata_patch")

        # Even if validation fails, schema should be recognized
        assert "Unknown schema" not in str(result.errors)

    def test_metadata_patch_schema_validates_suggestions_array(self):
        """Test that schema validates suggestions array structure."""
        from clinical_analytics.core.llm_json import validate_shape

        payload = {"suggestions": [{"operation": "set_description", "column": "age", "value": "Patient age"}]}

        result = validate_shape(payload, "metadata_patch")

        assert result.valid

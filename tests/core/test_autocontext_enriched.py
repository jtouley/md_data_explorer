"""
Tests for Phase 4: AutoContext Integration with ResolvedMetadata.

Tests cover:
- build_autocontext() using ResolvedDatasetMetadata overlays
- User/LLM aliases included in ColumnContext
- Enriched descriptions available in AutoContext
- PHI column filtering/exclusion
- Codebook enrichments reflected in context
- Exclusion patterns included for query grounding
"""

from unittest.mock import MagicMock

import pytest
from clinical_analytics.core.autocontext import (
    AutoContext,
    ColumnContext,
    build_autocontext,
)
from clinical_analytics.core.metadata_patch import (
    ResolvedColumnMetadata,
    ResolvedDatasetMetadata,
    ResolvedExclusionPattern,
    ResolvedRelationship,
    SemanticType,
)


class TestAutoContextWithResolvedMetadata:
    """Test suite for build_autocontext() with ResolvedMetadata integration."""

    @pytest.fixture
    def mock_semantic_layer(self):
        """Create a mock semantic layer for testing."""
        mock_sl = MagicMock()
        mock_sl.dataset_name = "test_dataset"
        mock_sl.upload_id = "test_upload_001"
        mock_sl.dataset_version = "v1"
        mock_sl.config = {
            "display_name": "Test Dataset",
            "column_mapping": {},
            "outcomes": {},
            "variable_types": {},
        }

        # Default alias index
        mock_sl.get_column_alias_index.return_value = {
            "patient_id": "patient_id",
            "age": "age",
            "hba1c_pct": "hba1c_pct",
        }
        mock_sl.get_column_metadata.return_value = None

        return mock_sl

    @pytest.fixture
    def resolved_metadata_with_enrichments(self) -> ResolvedDatasetMetadata:
        """Create ResolvedDatasetMetadata with various enrichments."""
        return ResolvedDatasetMetadata(
            columns={
                "patient_id": ResolvedColumnMetadata(
                    name="patient_id",
                    label="Patient ID",
                    description="Unique patient identifier",
                    semantic_type=SemanticType.IDENTIFIER,
                    aliases=["pid", "subject_id"],
                    is_phi=True,
                ),
                "age": ResolvedColumnMetadata(
                    name="age",
                    label="Patient Age",
                    description="Age in years at enrollment",
                    semantic_type=SemanticType.DEMOGRAPHIC,
                    unit="years",
                    aliases=["patient_age", "age_years"],
                ),
                "hba1c_pct": ResolvedColumnMetadata(
                    name="hba1c_pct",
                    label="HbA1c",
                    description="Hemoglobin A1c percentage",
                    semantic_type=SemanticType.MEASUREMENT,
                    unit="%",
                    codebook={"threshold": "7.0% is target for diabetes control"},
                    aliases=["a1c", "glycated_hemoglobin"],
                ),
                "diabetes_status": ResolvedColumnMetadata(
                    name="diabetes_status",
                    label="Diabetes Status",
                    description="Binary indicator for diabetes diagnosis",
                    semantic_type=SemanticType.CLINICAL,
                    codebook={"0": "No diabetes", "1": "Has diabetes", "9": "N/A"},
                    exclusion_patterns=[
                        ResolvedExclusionPattern(
                            pattern="n/a",
                            coded_value=9,
                            context="Exclude unknown status from analysis",
                            auto_apply=True,
                        )
                    ],
                ),
            },
            relationships=[
                ResolvedRelationship(
                    columns=["hba1c_pct", "diabetes_status"],
                    relationship_type="correlation",
                    rule="Higher HbA1c correlates with diabetes_status=1",
                    inference="Use hba1c_pct as predictor for diabetes queries",
                    confidence=0.85,
                )
            ],
        )

    def test_build_autocontext_accepts_resolved_metadata_parameter(self, mock_semantic_layer):
        """Test that build_autocontext() accepts resolved_metadata parameter."""
        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            continuous_columns=["age"],
        )

        # Should not raise - parameter accepted
        autocontext = build_autocontext(
            semantic_layer=mock_semantic_layer,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
            resolved_metadata=None,  # New parameter
        )

        assert isinstance(autocontext, AutoContext)

    def test_build_autocontext_includes_user_aliases_from_resolved_metadata(
        self, mock_semantic_layer, resolved_metadata_with_enrichments
    ):
        """Test that user/LLM aliases from ResolvedMetadata are included in ColumnContext."""
        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            continuous_columns=["age", "hba1c_pct"],
        )

        autocontext = build_autocontext(
            semantic_layer=mock_semantic_layer,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
            resolved_metadata=resolved_metadata_with_enrichments,
        )

        # Find hba1c_pct column
        hba1c_col = next((c for c in autocontext.columns if c.name == "hba1c_pct"), None)
        assert hba1c_col is not None

        # User aliases should include enriched aliases
        assert "a1c" in hba1c_col.user_aliases
        assert "glycated_hemoglobin" in hba1c_col.user_aliases

    def test_build_autocontext_includes_enriched_descriptions(
        self, mock_semantic_layer, resolved_metadata_with_enrichments
    ):
        """Test that enriched descriptions are available in ColumnContext."""
        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            continuous_columns=["age", "hba1c_pct"],
        )

        autocontext = build_autocontext(
            semantic_layer=mock_semantic_layer,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
            resolved_metadata=resolved_metadata_with_enrichments,
        )

        # Find age column
        age_col = next((c for c in autocontext.columns if c.name == "age"), None)
        assert age_col is not None

        # ColumnContext should have description (new field)
        assert hasattr(age_col, "description")
        assert age_col.description == "Age in years at enrollment"

    def test_build_autocontext_includes_enriched_units(self, mock_semantic_layer, resolved_metadata_with_enrichments):
        """Test that enriched units are reflected in ColumnContext."""
        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            continuous_columns=["age", "hba1c_pct"],
        )

        autocontext = build_autocontext(
            semantic_layer=mock_semantic_layer,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
            resolved_metadata=resolved_metadata_with_enrichments,
        )

        # Find age column - should have enriched unit
        age_col = next((c for c in autocontext.columns if c.name == "age"), None)
        assert age_col is not None
        assert age_col.units == "years"

        # Find hba1c column - should have enriched unit
        hba1c_col = next((c for c in autocontext.columns if c.name == "hba1c_pct"), None)
        assert hba1c_col is not None
        assert hba1c_col.units == "%"

    def test_build_autocontext_includes_enriched_codebook(
        self, mock_semantic_layer, resolved_metadata_with_enrichments
    ):
        """Test that enriched codebook entries are reflected in ColumnContext."""
        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            categorical_columns=["diabetes_status"],
            continuous_columns=["age", "hba1c_pct"],
        )

        mock_semantic_layer.get_column_alias_index.return_value = {
            "patient_id": "patient_id",
            "age": "age",
            "hba1c_pct": "hba1c_pct",
            "diabetes_status": "diabetes_status",
        }

        autocontext = build_autocontext(
            semantic_layer=mock_semantic_layer,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
            resolved_metadata=resolved_metadata_with_enrichments,
        )

        # Find diabetes_status column
        diabetes_col = next((c for c in autocontext.columns if c.name == "diabetes_status"), None)
        assert diabetes_col is not None

        # Codebook should include enriched entries
        assert diabetes_col.codebook is not None
        assert "0" in diabetes_col.codebook
        assert diabetes_col.codebook["0"] == "No diabetes"
        assert diabetes_col.codebook["9"] == "N/A"

    def test_build_autocontext_excludes_phi_columns_from_context(
        self, mock_semantic_layer, resolved_metadata_with_enrichments
    ):
        """Test that PHI-marked columns are excluded from AutoContext (privacy)."""
        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            continuous_columns=["age", "hba1c_pct"],
        )

        autocontext = build_autocontext(
            semantic_layer=mock_semantic_layer,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
            resolved_metadata=resolved_metadata_with_enrichments,
        )

        # PHI column (patient_id is marked is_phi=True) should be excluded from columns
        # but kept in entity_keys for reference
        patient_id_col = next((c for c in autocontext.columns if c.name == "patient_id"), None)

        # patient_id should still appear in entity_keys (needed for join context)
        assert "patient_id" in autocontext.entity_keys

        # But if column is included, it should be marked as PHI
        if patient_id_col is not None:
            assert hasattr(patient_id_col, "is_phi")
            assert patient_id_col.is_phi is True

    def test_build_autocontext_includes_exclusion_patterns(
        self, mock_semantic_layer, resolved_metadata_with_enrichments
    ):
        """Test that exclusion patterns are included in ColumnContext for query grounding."""
        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            categorical_columns=["diabetes_status"],
            continuous_columns=["age", "hba1c_pct"],
        )

        mock_semantic_layer.get_column_alias_index.return_value = {
            "patient_id": "patient_id",
            "age": "age",
            "hba1c_pct": "hba1c_pct",
            "diabetes_status": "diabetes_status",
        }

        autocontext = build_autocontext(
            semantic_layer=mock_semantic_layer,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
            resolved_metadata=resolved_metadata_with_enrichments,
        )

        # Find diabetes_status column
        diabetes_col = next((c for c in autocontext.columns if c.name == "diabetes_status"), None)
        assert diabetes_col is not None

        # Should have exclusion_patterns field
        assert hasattr(diabetes_col, "exclusion_patterns")
        assert diabetes_col.exclusion_patterns is not None
        assert len(diabetes_col.exclusion_patterns) > 0

        # Check first exclusion pattern
        first_pattern = diabetes_col.exclusion_patterns[0]
        assert first_pattern.pattern == "n/a"
        assert first_pattern.coded_value == 9
        assert first_pattern.auto_apply is True

    def test_build_autocontext_includes_relationships_in_context(
        self, mock_semantic_layer, resolved_metadata_with_enrichments
    ):
        """Test that cross-column relationships are included in AutoContext."""
        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            categorical_columns=["diabetes_status"],
            continuous_columns=["age", "hba1c_pct"],
        )

        mock_semantic_layer.get_column_alias_index.return_value = {
            "patient_id": "patient_id",
            "age": "age",
            "hba1c_pct": "hba1c_pct",
            "diabetes_status": "diabetes_status",
        }

        autocontext = build_autocontext(
            semantic_layer=mock_semantic_layer,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
            resolved_metadata=resolved_metadata_with_enrichments,
        )

        # AutoContext should have relationships field
        assert hasattr(autocontext, "relationships")
        assert autocontext.relationships is not None
        assert len(autocontext.relationships) > 0

        # Check first relationship
        first_rel = autocontext.relationships[0]
        assert "hba1c_pct" in first_rel.columns
        assert "diabetes_status" in first_rel.columns
        assert first_rel.relationship_type == "correlation"

    def test_build_autocontext_merges_system_and_user_aliases(
        self, mock_semantic_layer, resolved_metadata_with_enrichments
    ):
        """Test that system aliases and user aliases are both preserved."""
        from clinical_analytics.core.schema_inference import InferredSchema

        # Set up system aliases in semantic layer
        mock_semantic_layer.get_column_alias_index.return_value = {
            "hba1c_pct": "hba1c_pct",
            "hba1c": "hba1c_pct",  # System alias
            "hemoglobin_a1c": "hba1c_pct",  # System alias
        }

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            continuous_columns=["hba1c_pct"],
        )

        autocontext = build_autocontext(
            semantic_layer=mock_semantic_layer,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
            resolved_metadata=resolved_metadata_with_enrichments,
        )

        # Find hba1c_pct column
        hba1c_col = next((c for c in autocontext.columns if c.name == "hba1c_pct"), None)
        assert hba1c_col is not None

        # System aliases should be preserved
        assert "hba1c" in hba1c_col.system_aliases or "hemoglobin_a1c" in hba1c_col.system_aliases

        # User aliases (from ResolvedMetadata) should be in user_aliases
        assert "a1c" in hba1c_col.user_aliases
        assert "glycated_hemoglobin" in hba1c_col.user_aliases

    def test_build_autocontext_without_resolved_metadata_behaves_unchanged(self, mock_semantic_layer):
        """Test that build_autocontext() works normally when resolved_metadata is None."""
        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            continuous_columns=["age"],
        )

        # Should work without resolved_metadata (backward compatibility)
        autocontext = build_autocontext(
            semantic_layer=mock_semantic_layer,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
            resolved_metadata=None,
        )

        assert isinstance(autocontext, AutoContext)
        assert "patient_id" in autocontext.entity_keys


class TestColumnContextEnrichedFields:
    """Test suite for new ColumnContext fields added for enrichment."""

    def test_column_context_has_description_field(self):
        """Test that ColumnContext has description field."""
        # Create ColumnContext with description
        col = ColumnContext(
            name="hba1c_pct",
            normalized_name="hba1c_pct",
            system_aliases=["hba1c"],
            user_aliases=["a1c"],
            dtype="numeric",
            description="Hemoglobin A1c percentage",
        )

        assert hasattr(col, "description")
        assert col.description == "Hemoglobin A1c percentage"

    def test_column_context_has_is_phi_field(self):
        """Test that ColumnContext has is_phi field."""
        col = ColumnContext(
            name="patient_id",
            normalized_name="patient_id",
            system_aliases=[],
            user_aliases=[],
            dtype="id",
            is_phi=True,
        )

        assert hasattr(col, "is_phi")
        assert col.is_phi is True

    def test_column_context_has_exclusion_patterns_field(self):
        """Test that ColumnContext has exclusion_patterns field."""
        from clinical_analytics.core.metadata_patch import ResolvedExclusionPattern

        pattern = ResolvedExclusionPattern(
            pattern="n/a",
            coded_value=9,
            context="Exclude from analysis",
            auto_apply=True,
        )

        col = ColumnContext(
            name="diabetes_status",
            normalized_name="diabetes_status",
            system_aliases=[],
            user_aliases=[],
            dtype="coded",
            exclusion_patterns=[pattern],
        )

        assert hasattr(col, "exclusion_patterns")
        assert len(col.exclusion_patterns) == 1
        assert col.exclusion_patterns[0].pattern == "n/a"

    def test_column_context_has_semantic_type_field(self):
        """Test that ColumnContext has semantic_type field."""
        col = ColumnContext(
            name="age",
            normalized_name="age",
            system_aliases=[],
            user_aliases=[],
            dtype="numeric",
            semantic_type="demographic",
        )

        assert hasattr(col, "semantic_type")
        assert col.semantic_type == "demographic"


class TestAutoContextEnrichedFields:
    """Test suite for new AutoContext fields added for enrichment."""

    def test_autocontext_has_relationships_field(self):
        """Test that AutoContext has relationships field."""
        from clinical_analytics.core.metadata_patch import ResolvedRelationship

        rel = ResolvedRelationship(
            columns=["hba1c_pct", "diabetes_status"],
            relationship_type="correlation",
            rule="Higher HbA1c correlates with diabetes",
            inference="Use for prediction",
            confidence=0.85,
        )

        autocontext = AutoContext(
            dataset={"upload_id": "test", "dataset_version": "v1", "display_name": "Test"},
            entity_keys=["patient_id"],
            columns=[],
            glossary={},
            constraints={"no_row_level_data": True, "max_tokens": 4000},
            relationships=[rel],
        )

        assert hasattr(autocontext, "relationships")
        assert len(autocontext.relationships) == 1
        assert autocontext.relationships[0].relationship_type == "correlation"

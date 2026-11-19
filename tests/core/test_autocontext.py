"""
Tests for ADR004 Phase 3: AutoContext Packager

Tests cover:
- ColumnContext and AutoContext dataclass structures
- Entity key extraction from schema
- Column catalog building with aliases
- Glossary extraction from doc_context
- Token budget enforcement
- Privacy validation (no row-level data)
"""

# Test will fail until autocontext.py is created
import pytest
from ibis.common.exceptions import IbisTypeError
from structlog.testing import capture_logs

from clinical_analytics.core.autocontext import (
    AutoContext,
    ColumnContext,
    _reconstruct_inferred_schema_from_semantic_layer,
    build_autocontext,
)


class TestColumnContext:
    """Test suite for ColumnContext dataclass."""

    def test_column_context_dataclass_has_required_fields(self):
        """Test that ColumnContext dataclass has all required fields."""
        # Arrange: Create ColumnContext instance
        column_context = ColumnContext(
            name="Current Regimen",
            normalized_name="current_regimen",
            system_aliases=["regimen", "current_regimen"],
            user_aliases=[],
            dtype="coded",
            units=None,
            codebook={"1": "Biktarvy", "2": "Symtuza"},
        )

        # Assert: Verify all fields exist and are correct type
        assert column_context.name == "Current Regimen"
        assert column_context.normalized_name == "current_regimen"
        assert isinstance(column_context.system_aliases, list)
        assert isinstance(column_context.user_aliases, list)
        assert column_context.dtype in ("numeric", "categorical", "datetime", "id", "coded")
        assert column_context.units is None or isinstance(column_context.units, str)
        assert column_context.codebook is None or isinstance(column_context.codebook, dict)

    def test_column_context_optional_fields_default_to_none(self):
        """Test that optional fields default to None."""
        # Arrange: Create ColumnContext with only required fields
        column_context = ColumnContext(
            name="age",
            normalized_name="age",
            system_aliases=["age"],
            user_aliases=[],
            dtype="numeric",
        )

        # Assert: Optional fields should default to None
        assert column_context.units is None
        assert column_context.codebook is None
        assert not hasattr(column_context, "stats")  # observed values never reach the model


class TestAutoContext:
    """Test suite for AutoContext dataclass."""

    def test_autocontext_dataclass_has_required_fields(self):
        """Test that AutoContext dataclass has all required fields."""
        # Arrange: Create AutoContext instance
        autocontext = AutoContext(
            dataset={"upload_id": "test_001", "dataset_version": "v1", "display_name": "Test Dataset"},
            entity_keys=["patient_id", "encounter_id"],
            columns=[],
            glossary={"BMI": "Body Mass Index", "LDL": "Low-density lipoprotein"},
            constraints={"no_row_level_data": True, "max_tokens": 4000},
        )

        # Assert: Verify all fields exist and are correct type
        assert isinstance(autocontext.dataset, dict)
        assert "upload_id" in autocontext.dataset
        assert isinstance(autocontext.entity_keys, list)
        assert isinstance(autocontext.columns, list)
        assert isinstance(autocontext.glossary, dict)
        assert isinstance(autocontext.constraints, dict)
        assert autocontext.constraints["no_row_level_data"] is True


class TestBuildAutoContext:
    """Test suite for build_autocontext() function."""

    def test_build_autocontext_extracts_entity_keys(self, mock_semantic_layer):
        """Test that build_autocontext() extracts entity keys from schema."""
        # Arrange: Create mock semantic layer and inferred schema
        from clinical_analytics.core.schema_inference import InferredSchema

        mock_sl = mock_semantic_layer(columns={"patient_id": "patient_id", "encounter_id": "encounter_id"})
        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            time_zero="2020-01-01",
            outcome_columns=["outcome"],
            categorical_columns=["treatment"],
            continuous_columns=["age"],
        )

        # Act: Build AutoContext
        autocontext = build_autocontext(
            semantic_layer=mock_sl,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
        )

        # Assert: Entity keys should include patient_id (highest priority)
        assert isinstance(autocontext.entity_keys, list)
        assert len(autocontext.entity_keys) > 0
        assert "patient_id" in autocontext.entity_keys
        # patient_id should be first (highest priority)
        assert autocontext.entity_keys[0] == "patient_id"

    def test_build_autocontext_builds_column_catalog(self, mock_semantic_layer):
        """Test that build_autocontext() builds column catalog with aliases."""
        # Arrange: Create mock semantic layer with aliases
        mock_sl = mock_semantic_layer(
            columns={
                "patient_id": "patient_id",
                "age": "age",
                "current_regimen": "Current Regimen: 1=Biktarvy, 2=Symtuza",
            }
        )
        mock_sl.get_column_alias_index.return_value = {
            "patient_id": "patient_id",
            "age": "age",
            "regimen": "current_regimen",
            "current_regimen": "current_regimen",
        }

        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            patient_id_column="patient_id",
            categorical_columns=["current_regimen"],
            continuous_columns=["age"],
        )

        # Act: Build AutoContext
        autocontext = build_autocontext(
            semantic_layer=mock_sl,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=4000,
        )

        # Assert: Column catalog should contain columns with aliases
        assert len(autocontext.columns) > 0
        # Find age column
        age_col = next((c for c in autocontext.columns if c.name == "age"), None)
        assert age_col is not None
        assert age_col.dtype == "numeric"
        assert "age" in age_col.system_aliases

    def test_build_autocontext_extracts_glossary(self, mock_semantic_layer):
        """Test that build_autocontext() extracts glossary from doc_context."""
        # Arrange: Create doc_context with glossary terms
        doc_context = """
        Abbreviations:
        - BMI: Body Mass Index
        - LDL: Low-density lipoprotein
        - HIV: Human Immunodeficiency Virus

        Definitions:
        - Regimen: Antiretroviral treatment combination
        """
        mock_sl = mock_semantic_layer(columns={"patient_id": "patient_id"})

        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(patient_id_column="patient_id")

        # Act: Build AutoContext
        autocontext = build_autocontext(
            semantic_layer=mock_sl,
            inferred_schema=inferred_schema,
            doc_context=doc_context,
            query_terms=None,
            max_tokens=4000,
        )

        # Assert: Glossary should contain extracted terms
        assert isinstance(autocontext.glossary, dict)
        assert len(autocontext.glossary) > 0
        # Should extract BMI and LDL
        assert "BMI" in autocontext.glossary or "bmi" in autocontext.glossary.lower()
        assert "LDL" in autocontext.glossary or "ldl" in autocontext.glossary.lower()

    def test_build_autocontext_glossary_limited_to_top_50(self, mock_semantic_layer):
        """Test that glossary is limited to top 50 terms for token budget."""
        # Arrange: Create doc_context with many terms
        doc_context = "\n".join([f"- Term{i}: Definition {i}" for i in range(100)])
        mock_sl = mock_semantic_layer(columns={"patient_id": "patient_id"})

        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(patient_id_column="patient_id")

        # Act: Build AutoContext
        autocontext = build_autocontext(
            semantic_layer=mock_sl,
            inferred_schema=inferred_schema,
            doc_context=doc_context,
            query_terms=None,
            max_tokens=4000,
        )

        # Assert: Glossary should be limited to top 50
        assert len(autocontext.glossary) <= 50

    def test_build_autocontext_enforces_token_budget(self, mock_semantic_layer):
        """Test that build_autocontext() enforces token budget with truncation."""
        # Arrange: Create schema with many columns (exceeds token budget)
        mock_sl = mock_semantic_layer(
            columns={f"col_{i}": f"col_{i}" for i in range(100)}  # 100 columns
        )
        mock_sl.get_column_alias_index.return_value = {f"col_{i}": f"col_{i}" for i in range(100)}

        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(
            categorical_columns=[f"col_{i}" for i in range(100)],
        )

        # Act: Build AutoContext with small token budget
        autocontext = build_autocontext(
            semantic_layer=mock_sl,
            inferred_schema=inferred_schema,
            doc_context=None,
            query_terms=None,
            max_tokens=1000,  # Small budget
        )

        # Assert: Should truncate columns to fit budget
        # Token count approximation: ~4 chars per token
        # 100 columns * ~50 chars each = ~5000 chars = ~1250 tokens
        # With 1000 token budget, should truncate to ~80 columns
        assert len(autocontext.columns) < 100

    def test_build_autocontext_metadataWithMinMaxTopValues_carriesNoObservedValues(self, mock_semantic_layer):
        """min, max and value counts in the column metadata never reach the model's context."""
        # Arrange: column metadata as uploads store it, with observed statistics
        mock_sl = mock_semantic_layer(columns={"patient_id": "patient_id", "smoker": "smoker"})
        observed = {"min": 0, "max": 1, "top_values": {"rare_level": 3, "common_level": 400}}
        mock_sl.get_column_metadata.return_value = {"type": "categorical", "metadata": observed}

        from clinical_analytics.core.schema_inference import InferredSchema

        inferred_schema = InferredSchema(patient_id_column="patient_id", categorical_columns=["smoker"])

        # Act
        autocontext = build_autocontext(semantic_layer=mock_sl, inferred_schema=inferred_schema, max_tokens=4000)

        # Assert
        assert autocontext.constraints["no_row_level_data"] is True
        assert "rare_level" not in repr(autocontext)
        assert all(not hasattr(col, "stats") for col in autocontext.columns)


class TestEstimateTokens:
    """Test suite for _estimate_tokens() function."""

    def test_estimate_tokens_accuracy_with_tiktoken(self):
        """Test token estimation accuracy using tiktoken."""
        # Arrange: tiktoken downloads its encoding on first use; skip when it cannot be loaded
        tiktoken = pytest.importorskip("tiktoken")
        try:
            tiktoken.get_encoding("cl100k_base")
        except (OSError, ValueError) as e:
            pytest.skip(f"tiktoken cl100k_base encoding unavailable (not cached, no network): {e}")
        from clinical_analytics.core.autocontext import _estimate_tokens

        # Text that exposes character approximation inaccuracy:
        # Long text with punctuation has different char/token ratio than simple text
        long_text = "The quick brown fox jumps over the lazy dog. " * 10
        # Character approximation: 450 chars / 4 = 112 tokens
        # Actual tiktoken count: 101 tokens (10% difference)

        # Act
        result = _estimate_tokens(long_text)

        # Assert: Should be close to tiktoken count (not character approximation)
        # With tiktoken: 101 tokens
        # With char approximation: 112 tokens
        # Test expects tiktoken accuracy (within 2 tokens of 101)
        assert 99 <= result <= 103, (
            f"Token count should be ~101 with tiktoken (got {result}). "
            f"Character approximation gives ~{len(long_text) // 4}"
        )


class TestReconstructInferredSchemaFromConfig:
    """Tier 3 rebuilds the schema from the layer config: bad entries classify by default, view errors are logged."""

    @pytest.mark.parametrize(
        "age_entry",
        [
            pytest.param(None, id="none"),
            pytest.param("numeric", id="string"),
            pytest.param(["numeric"], id="list"),
        ],
    )
    def test_reconstruct_variableTypesEntryNotDict_classifiesCategoricalAndWarns(self, mock_semantic_layer, age_entry):
        # Arrange: a YAML-style config whose "age" entry is not a dict; "mx" is a well-formed one
        mock_sl = mock_semantic_layer(columns={"mx": "mx", "age": "age"})
        mock_sl.config = {
            "column_mapping": {},
            "outcomes": {},
            "variable_types": {"mx": {"type": "numeric"}, "age": age_entry},
        }

        # Act
        with capture_logs() as logs:
            schema = _reconstruct_inferred_schema_from_semantic_layer(mock_sl)

        # Assert
        assert schema.continuous_columns == ["mx"]
        assert schema.categorical_columns == ["age"]
        warned = [e for e in logs if e["event"] == "autocontext_variable_type_not_a_dict"]
        assert len(warned) == (0 if age_entry is None else 1)

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(IbisTypeError("column 'x' not found"), id="ibis_missing_column"),
            pytest.param(KeyError("source_column"), id="outcome_without_source_column"),
        ],
    )
    def test_reconstruct_baseViewConfigError_logsWarningAndHasNoColumns(self, mock_semantic_layer, error):
        # Arrange
        mock_sl = mock_semantic_layer()
        mock_sl.config = {"column_mapping": {}, "outcomes": {}, "variable_types": {}}
        mock_sl.get_base_view.side_effect = error

        # Act
        with capture_logs() as logs:
            schema = _reconstruct_inferred_schema_from_semantic_layer(mock_sl)

        # Assert: not silent, and no column is classified
        assert schema.categorical_columns == [] and schema.continuous_columns == []
        events = [e for e in logs if e["event"] == "autocontext_base_view_unavailable"]
        assert [e["error_type"] for e in events] == [type(error).__name__]

    def test_reconstruct_baseViewBug_propagates(self, mock_semantic_layer):
        # Arrange: an AttributeError is a bug (e.g. a None config field), not a config the data lacks
        mock_sl = mock_semantic_layer()
        mock_sl.config = {"column_mapping": {}, "outcomes": {}, "variable_types": {}}
        mock_sl.get_base_view.side_effect = AttributeError("'NoneType' object has no attribute 'items'")

        # Act / Assert
        with pytest.raises(AttributeError):
            _reconstruct_inferred_schema_from_semantic_layer(mock_sl)

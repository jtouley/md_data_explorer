"""
Tests for centralized patient ID regeneration logic.

Tests the patient_id_regeneration module which is the single source of truth
for all patient ID regeneration decisions and validation.
"""

import polars as pl
import pytest

from clinical_analytics.datasets.uploaded.patient_id_regeneration import (
    PatientIdRegenerationError,
    can_regenerate_patient_id,
    regenerate_patient_id,
    validate_synthetic_id_metadata,
)


class TestValidateSyntheticIdMetadata:
    """Tests for validate_synthetic_id_metadata function."""

    def test_none_metadata_returns_empty_dict(self):
        """None metadata should return empty dict."""
        # Act
        result = validate_synthetic_id_metadata(None)

        # Assert
        assert result == {}

    def test_valid_metadata_returns_unchanged(self):
        """Valid metadata should be returned unchanged."""
        # Arrange
        metadata = {
            "patient_id": {
                "patient_id_source": "composite",
                "patient_id_columns": ["race", "gender"],
            }
        }

        # Act
        result = validate_synthetic_id_metadata(metadata)

        # Assert
        assert result == metadata

    def test_invalid_type_raises_error(self):
        """Non-dict metadata should raise PatientIdRegenerationError."""
        # Arrange
        invalid_metadata = "not a dict"

        # Act & Assert
        with pytest.raises(PatientIdRegenerationError, match="must be a dict"):
            validate_synthetic_id_metadata(invalid_metadata)

    def test_invalid_patient_id_type_raises_error(self):
        """Non-dict patient_id sub-structure should raise error."""
        # Arrange
        metadata = {"patient_id": "not a dict"}

        # Act & Assert
        with pytest.raises(PatientIdRegenerationError, match="patient_id.*must be a dict"):
            validate_synthetic_id_metadata(metadata)

    def test_invalid_patient_id_source_raises_error(self):
        """Invalid patient_id_source should raise error."""
        # Arrange
        metadata = {
            "patient_id": {
                "patient_id_source": "invalid_source",
            }
        }

        # Act & Assert
        with pytest.raises(PatientIdRegenerationError, match="Invalid patient_id_source"):
            validate_synthetic_id_metadata(metadata)

    def test_valid_sources_accepted(self):
        """Valid patient_id_source values should be accepted."""
        # Arrange
        valid_sources = ["composite", "single_column", "existing"]

        for source in valid_sources:
            metadata = {"patient_id": {"patient_id_source": source}}

            # Act
            result = validate_synthetic_id_metadata(metadata)

            # Assert
            assert result["patient_id"]["patient_id_source"] == source

    def test_invalid_patient_id_columns_type_raises_error(self):
        """Non-list patient_id_columns should raise error."""
        # Arrange
        metadata = {
            "patient_id": {
                "patient_id_source": "composite",
                "patient_id_columns": "not a list",
            }
        }

        # Act & Assert
        with pytest.raises(PatientIdRegenerationError, match="patient_id_columns must be a list"):
            validate_synthetic_id_metadata(metadata)


class TestCanRegeneratePatientId:
    """Tests for can_regenerate_patient_id function."""

    def test_no_metadata_returns_cannot_regenerate(self):
        """No metadata should return cannot regenerate."""
        # Arrange
        df = pl.DataFrame({"col1": [1, 2, 3]})

        # Act
        result = can_regenerate_patient_id(df, None)

        # Assert
        assert result.can_regenerate is False
        assert result.source_type is None
        assert result.source_columns == []
        assert result.error_message is None

    def test_composite_with_all_columns_present_returns_can_regenerate(self):
        """Composite metadata with all columns present should return can_regenerate=True."""
        # Arrange
        df = pl.DataFrame(
            {
                "race": ["White", "Black", "Asian"],
                "gender": ["M", "F", "M"],
                "age": [45, 52, 38],
            }
        )
        metadata = {
            "patient_id": {
                "patient_id_source": "composite",
                "patient_id_columns": ["race", "gender"],
            }
        }

        # Act
        result = can_regenerate_patient_id(df, metadata)

        # Assert
        assert result.can_regenerate is True
        assert result.source_type == "composite"
        assert result.source_columns == ["race", "gender"]
        assert result.error_message is None

    def test_composite_with_missing_column_returns_cannot_regenerate_with_error(self):
        """Composite metadata with missing column should fail fast with error."""
        # Arrange
        df = pl.DataFrame(
            {
                "gender": ["M", "F", "M"],
                "age": [45, 52, 38],
            }
        )
        metadata = {
            "patient_id": {
                "patient_id_source": "composite",
                "patient_id_columns": ["race", "gender"],  # race is missing
            }
        }

        # Act
        result = can_regenerate_patient_id(df, metadata)

        # Assert
        assert result.can_regenerate is False
        assert result.source_type == "composite"
        assert result.source_columns == ["race", "gender"]
        assert result.error_message is not None
        assert "race" in result.error_message  # Should mention missing column
        assert "not found" in result.error_message

    def test_composite_with_empty_columns_returns_cannot_regenerate(self):
        """Composite metadata with empty columns should fail fast."""
        # Arrange
        df = pl.DataFrame({"race": ["White", "Black"]})
        metadata = {
            "patient_id": {
                "patient_id_source": "composite",
                "patient_id_columns": [],  # Empty columns
            }
        }

        # Act
        result = can_regenerate_patient_id(df, metadata)

        # Assert
        assert result.can_regenerate is False
        assert "patient_id_columns" in result.error_message

    def test_single_column_with_column_present_returns_can_regenerate(self):
        """Single column metadata with column present should return can_regenerate=True."""
        # Arrange
        df = pl.DataFrame(
            {
                "subject_id": ["S001", "S002", "S003"],
                "age": [45, 52, 38],
            }
        )
        metadata = {
            "patient_id": {
                "patient_id_source": "single_column",
                "patient_id_columns": ["subject_id"],
            }
        }

        # Act
        result = can_regenerate_patient_id(df, metadata)

        # Assert
        assert result.can_regenerate is True
        assert result.source_type == "single_column"
        assert result.source_columns == ["subject_id"]
        assert result.error_message is None

    def test_single_column_with_missing_column_returns_cannot_regenerate(self):
        """Single column metadata with missing column should fail fast with error."""
        # Arrange
        df = pl.DataFrame({"age": [45, 52, 38]})
        metadata = {
            "patient_id": {
                "patient_id_source": "single_column",
                "patient_id_columns": ["subject_id"],  # Missing
            }
        }

        # Act
        result = can_regenerate_patient_id(df, metadata)

        # Assert
        assert result.can_regenerate is False
        assert "subject_id" in result.error_message
        assert "not found" in result.error_message

    def test_existing_source_returns_cannot_regenerate(self):
        """'existing' source type means no regeneration needed."""
        # Arrange
        df = pl.DataFrame({"patient_id": [1, 2, 3]})
        metadata = {
            "patient_id": {
                "patient_id_source": "existing",
            }
        }

        # Act
        result = can_regenerate_patient_id(df, metadata)

        # Assert
        assert result.can_regenerate is False
        assert result.error_message is None  # Not an error, just no regeneration


class TestRegeneratePatientId:
    """Tests for regenerate_patient_id function."""

    def test_composite_regeneration_creates_patient_id(self):
        """Composite regeneration should create patient_id from multiple columns."""
        # Arrange
        df = pl.DataFrame(
            {
                "race": ["White", "Black", "Asian"],
                "gender": ["M", "F", "M"],
                "age": [45, 52, 38],
            }
        )
        metadata = {
            "patient_id": {
                "patient_id_source": "composite",
                "patient_id_columns": ["race", "gender"],
            }
        }

        # Act
        df_with_id, id_metadata = regenerate_patient_id(df, metadata)

        # Assert
        assert "patient_id" in df_with_id.columns
        assert id_metadata["patient_id_source"] == "composite"
        assert id_metadata["patient_id_columns"] == ["race", "gender"]
        # Original columns preserved
        assert "race" in df_with_id.columns
        assert "gender" in df_with_id.columns
        assert "age" in df_with_id.columns

    def test_single_column_regeneration_creates_patient_id(self):
        """Single column regeneration should create patient_id from source column."""
        # Arrange
        df = pl.DataFrame(
            {
                "subject_id": ["S001", "S002", "S003"],
                "age": [45, 52, 38],
            }
        )
        metadata = {
            "patient_id": {
                "patient_id_source": "single_column",
                "patient_id_columns": ["subject_id"],
            }
        }

        # Act
        df_with_id, id_metadata = regenerate_patient_id(df, metadata)

        # Assert
        assert "patient_id" in df_with_id.columns
        assert id_metadata["patient_id_source"] == "single_column"
        assert id_metadata["patient_id_columns"] == ["subject_id"]
        # Patient ID values should match source column
        assert df_with_id["patient_id"].to_list() == ["S001", "S002", "S003"]

    def test_regeneration_with_nulls_is_deterministic(self):
        """Regeneration with null values should be deterministic."""
        # Arrange
        df = pl.DataFrame(
            {
                "race": ["White", None, "Asian"],
                "gender": ["M", "F", None],
            }
        )
        metadata = {
            "patient_id": {
                "patient_id_source": "composite",
                "patient_id_columns": ["race", "gender"],
            }
        }

        # Act - regenerate twice
        df_with_id_1, _ = regenerate_patient_id(df, metadata)
        df_with_id_2, _ = regenerate_patient_id(df, metadata)

        # Assert - should be identical
        assert df_with_id_1["patient_id"].to_list() == df_with_id_2["patient_id"].to_list()

    def test_invalid_source_type_raises_error(self):
        """Invalid source type should raise PatientIdRegenerationError."""
        # Arrange
        df = pl.DataFrame({"col1": [1, 2, 3]})
        metadata = {
            "patient_id": {
                "patient_id_source": "invalid",
            }
        }

        # Act & Assert
        with pytest.raises(PatientIdRegenerationError, match="unsupported source_type"):
            regenerate_patient_id(df, metadata)


class TestPatientIdRegenerationIntegration:
    """Integration tests for the full regeneration flow."""

    def test_full_flow_composite_regeneration(self):
        """Test full flow: validate → can_regenerate → regenerate."""
        # Arrange
        df = pl.DataFrame(
            {
                "race": ["White", "Black", "Asian"],
                "gender": ["M", "F", "M"],
                "outcome": [0, 1, 0],
            }
        )
        raw_metadata = {
            "patient_id": {
                "patient_id_source": "composite",
                "patient_id_columns": ["race", "gender"],
            }
        }

        # Act - Step 1: Validate
        validated = validate_synthetic_id_metadata(raw_metadata)
        assert validated == raw_metadata

        # Act - Step 2: Check if can regenerate
        result = can_regenerate_patient_id(df, validated)
        assert result.can_regenerate is True

        # Act - Step 3: Regenerate
        df_with_id, id_metadata = regenerate_patient_id(df, validated)

        # Assert
        assert "patient_id" in df_with_id.columns
        assert len(df_with_id) == 3
        # Original columns preserved
        assert "race" in df_with_id.columns
        assert "gender" in df_with_id.columns
        assert "outcome" in df_with_id.columns

    def test_full_flow_fails_fast_on_missing_column(self):
        """Test that missing column is caught early with clear error."""
        # Arrange
        df = pl.DataFrame(
            {
                "gender": ["M", "F", "M"],
                "outcome": [0, 1, 0],
                # 'race' is missing
            }
        )
        raw_metadata = {
            "patient_id": {
                "patient_id_source": "composite",
                "patient_id_columns": ["race", "gender"],
            }
        }

        # Act - Step 1: Validate (should pass - structure is valid)
        validated = validate_synthetic_id_metadata(raw_metadata)
        assert validated == raw_metadata

        # Act - Step 2: Check if can regenerate (should fail fast)
        result = can_regenerate_patient_id(df, validated)

        # Assert - Fail fast with clear error
        assert result.can_regenerate is False
        assert result.error_message is not None
        assert "race" in result.error_message

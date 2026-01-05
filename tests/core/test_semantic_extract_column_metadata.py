"""
Tests for SemanticLayer.extract_column_metadata() method (ADR004 Phase 3).

Tests that extract_column_metadata() returns dict compatible with ColumnContext construction.
"""


class TestExtractColumnMetadata:
    """Test suite for extract_column_metadata() method."""

    def test_extract_column_metadata_returns_dict_compatible_with_column_context(self, make_semantic_layer):
        """Test that extract_column_metadata() returns dict compatible with ColumnContext."""
        # Arrange: Create semantic layer with variable_types metadata
        config = {
            "variable_types": {
                "current_regimen": {
                    "type": "categorical",
                    "codebook": {"1": "Biktarvy", "2": "Symtuza"},
                    "metadata": {"numeric": True},
                },
                "ldl_mg_dl": {
                    "type": "numeric",
                    "units": "mg/dL",
                    "metadata": {"numeric": True},
                },
            }
        }
        semantic_layer = make_semantic_layer(
            dataset_name="test",
            config_overrides=config,
            data={"current_regimen": [1, 2], "ldl_mg_dl": [100, 120]},
        )

        # Act: Extract metadata for coded column
        metadata = semantic_layer.extract_column_metadata("current_regimen")

        # Assert: Should return dict with required fields
        assert metadata is not None
        assert "name" in metadata
        assert "normalized_name" in metadata
        assert "dtype" in metadata
        assert metadata["name"] == "current_regimen"
        assert metadata["dtype"] == "coded"
        assert "codebook" in metadata
        assert metadata["codebook"] == {"1": "Biktarvy", "2": "Symtuza"}

    def test_extract_column_metadata_maps_dtype_correctly(self, make_semantic_layer):
        """Test that extract_column_metadata() maps dtype correctly."""
        # Arrange: Create semantic layer with different column types
        config = {
            "variable_types": {
                "age": {"type": "numeric", "metadata": {"numeric": True}},
                "treatment": {"type": "categorical", "metadata": {"numeric": False}},
                "patient_id": {"type": "categorical", "metadata": {"numeric": False}},
            }
        }
        semantic_layer = make_semantic_layer(
            dataset_name="test",
            config_overrides=config,
            data={"age": [25, 30], "treatment": ["A", "B"], "patient_id": [1, 2]},
        )

        # Act & Assert: Test dtype mapping
        age_meta = semantic_layer.extract_column_metadata("age")
        assert age_meta is not None
        assert age_meta["dtype"] == "numeric"

        treatment_meta = semantic_layer.extract_column_metadata("treatment")
        assert treatment_meta is not None
        assert treatment_meta["dtype"] == "categorical"

        patient_id_meta = semantic_layer.extract_column_metadata("patient_id")
        assert patient_id_meta is not None
        assert patient_id_meta["dtype"] == "id"  # "id" in name

    def test_extract_column_metadata_returns_none_if_no_metadata(self, make_semantic_layer):
        """Test that extract_column_metadata() returns None if metadata unavailable."""
        # Arrange: Create semantic layer without variable_types
        semantic_layer = make_semantic_layer(
            dataset_name="test",
            config_overrides={},
            data={"age": [25, 30]},
        )

        # Act: Extract metadata for column without metadata
        metadata = semantic_layer.extract_column_metadata("age")

        # Assert: Should return None
        assert metadata is None

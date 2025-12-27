"""
Tests for MultiTableHandler table classification and aggregate-before-join pipeline.

Acceptance criteria for Milestone 1:
1. Bridge detection on synthetic many-to-many fixture
2. Byte estimates within sane range on sampled rows
3. Grain key detection is deterministic
"""

import polars as pl
import pytest
from clinical_analytics.core.multi_table_handler import (
    MultiTableHandler,
    TableClassification,
    TableRelationship
)


class TestTableClassification:
    """Test suite for Milestone 1: Table Classification System."""

    def test_bridge_detection_on_many_to_many_fixture(self):
        """
        M1 Acceptance Test 1: Bridge table identified in synthetic many-to-many fixture.

        Setup:
        - patients (dimension): unique patient_id
        - medications (dimension): unique medication_id
        - patient_medications (bridge): patient_id + medication_id composite unique

        Expected:
        - patient_medications classified as "bridge"
        - patients classified as "dimension"
        - medications classified as "dimension" or "reference"
        """
        # Arrange: Create synthetic many-to-many dataset
        patients = pl.DataFrame({
            "patient_id": ["P1", "P2", "P3"],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 45, 28]
        })

        medications = pl.DataFrame({
            "medication_id": ["M1", "M2", "M3"],
            "drug_name": ["Aspirin", "Metformin", "Lisinopril"],
            "dosage": ["100mg", "500mg", "10mg"]
        })

        # Bridge table: many-to-many relationship
        # - patient_id is NOT unique (P1 has 2 medications)
        # - medication_id is NOT unique (M1 prescribed to 2 patients)
        # - BUT composite (patient_id, medication_id) IS unique
        patient_medications = pl.DataFrame({
            "patient_id": ["P1", "P1", "P2", "P3"],
            "medication_id": ["M1", "M2", "M1", "M3"],
            "start_date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-03-01"],
            "dosage_override": [None, "250mg", None, None]
        })

        tables = {
            "patients": patients,
            "medications": medications,
            "patient_medications": patient_medications
        }

        # Act: Initialize handler and classify
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        classifications = handler.classify_tables()

        # Assert: Bridge detection
        assert "patient_medications" in classifications, "Bridge table should be classified"

        bridge_class = classifications["patient_medications"]
        assert bridge_class.classification == "bridge", (
            f"patient_medications should be classified as bridge, got {bridge_class.classification}"
        )

        # Verify bridge characteristics
        assert bridge_class.relationship_degree >= 2, (
            f"Bridge should have 2+ foreign keys, got {bridge_class.relationship_degree}"
        )

        # Verify patients is dimension
        assert classifications["patients"].classification == "dimension", (
            f"patients should be dimension, got {classifications['patients'].classification}"
        )

        # Verify medications is dimension or reference
        assert classifications["medications"].classification in ["dimension", "reference"], (
            f"medications should be dimension or reference, got {classifications['medications'].classification}"
        )

        handler.close()

    def test_byte_estimates_within_sane_range(self):
        """
        M1 Acceptance Test 2: Byte estimates within sane range on sampled rows.

        Expected ranges:
        - Int64 column (1000 rows): ~8,000 bytes
        - Utf8 column (1000 rows, avg 10 chars): ~10,000 bytes
        - Total estimate should be within 2x of theoretical minimum
        """
        # Arrange: Create table with known characteristics
        num_rows = 1000
        df = pl.DataFrame({
            "id": range(num_rows),  # Int64: 8 bytes/row
            "name": [f"Name_{i:04d}" for i in range(num_rows)],  # Utf8: ~9 chars = 9 bytes/row
            "value": [float(i) * 1.5 for i in range(num_rows)],  # Float64: 8 bytes/row
        })

        tables = {"test_table": df}
        handler = MultiTableHandler(tables)

        # Act: Estimate bytes
        estimated_bytes = handler._estimate_table_bytes(df)

        # Assert: Within reasonable range
        # Theoretical minimum: (8 + 9 + 8) * 1000 = 25,000 bytes
        theoretical_min = 25_000
        theoretical_max = theoretical_min * 2  # Allow 2x overhead

        assert theoretical_min <= estimated_bytes <= theoretical_max, (
            f"Byte estimate {estimated_bytes:,} outside sane range "
            f"[{theoretical_min:,}, {theoretical_max:,}]"
        )

        # Verify estimate is reasonable per row
        bytes_per_row = estimated_bytes / num_rows
        assert 20 <= bytes_per_row <= 50, (
            f"Bytes per row {bytes_per_row:.1f} outside expected range [20, 50]"
        )

        handler.close()

    def test_grain_key_detection_is_deterministic(self):
        """
        M1 Acceptance Test 3: Grain key detection is deterministic.

        Same DataFrame should always return same grain key, regardless of:
        - Column order
        - Multiple runs
        - Data order
        """
        # Arrange: Create DataFrame with multiple ID columns
        df_original = pl.DataFrame({
            "encounter_id": ["E1", "E2", "E3"],
            "patient_id": ["P1", "P1", "P2"],
            "visit_id": ["V1", "V2", "V3"],
            "name": ["Alice", "Alice", "Bob"]
        })

        # Create reordered version (different column order)
        df_reordered = df_original.select(["name", "patient_id", "visit_id", "encounter_id"])

        tables_original = {"test": df_original}
        tables_reordered = {"test": df_reordered}

        handler_original = MultiTableHandler(tables_original)
        handler_reordered = MultiTableHandler(tables_reordered)

        # Act: Detect grain key multiple times
        grain_key_1 = handler_original._detect_grain_key(df_original)
        grain_key_2 = handler_original._detect_grain_key(df_original)  # Same handler
        grain_key_3 = handler_reordered._detect_grain_key(df_reordered)  # Different order

        # Assert: Deterministic grain key
        assert grain_key_1 == grain_key_2, (
            f"Same DataFrame should return same grain key: {grain_key_1} != {grain_key_2}"
        )

        assert grain_key_1 == grain_key_3, (
            f"Column order should not affect grain key: {grain_key_1} != {grain_key_3}"
        )

        # Verify grain key follows priority rules (patient_id should be chosen)
        assert grain_key_1 == "patient_id", (
            f"patient_id should be prioritized, got {grain_key_1}"
        )

        handler_original.close()
        handler_reordered.close()

    def test_grain_level_detection(self):
        """Test grain level detection from grain key names."""
        # Arrange
        tables = {"dummy": pl.DataFrame({"id": [1]})}
        handler = MultiTableHandler(tables)

        # Act & Assert: Patient grain
        assert handler._detect_grain_level("patient_id") == "patient"
        assert handler._detect_grain_level("subject_id") == "patient"

        # Admission grain
        assert handler._detect_grain_level("hadm_id") == "admission"
        assert handler._detect_grain_level("encounter_id") == "admission"
        assert handler._detect_grain_level("visit_id") == "admission"

        # Event grain (fallback)
        assert handler._detect_grain_level("row_id") == "event"
        assert handler._detect_grain_level("charttime_id") == "event"

        handler.close()

    def test_time_column_detection(self):
        """Test detection of time columns."""
        # Arrange: DataFrame with time column
        df_with_time = pl.DataFrame({
            "id": [1, 2, 3],
            "charttime": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [100, 200, 300]
        })

        # DataFrame without time column
        df_no_time = pl.DataFrame({
            "id": [1, 2, 3],
            "value": [100, 200, 300]
        })

        # DataFrame with constant time column (should not detect)
        df_constant_time = pl.DataFrame({
            "id": [1, 2, 3],
            "timestamp": ["2024-01-01", "2024-01-01", "2024-01-01"]
        })

        tables = {"dummy": pl.DataFrame({"id": [1]})}
        handler = MultiTableHandler(tables)

        # Act & Assert
        time_col, has_time = handler._detect_time_column(df_with_time)
        assert has_time is True
        assert time_col == "charttime"

        time_col, has_time = handler._detect_time_column(df_no_time)
        assert has_time is False
        assert time_col is None

        time_col, has_time = handler._detect_time_column(df_constant_time)
        assert has_time is False, "Constant time columns should not be detected"

        handler.close()

    def test_classification_rules(self):
        """Test classification rule priority."""
        # Arrange
        patients = pl.DataFrame({
            "patient_id": ["P1", "P2", "P3"],
            "age": [30, 45, 28]
        })

        # High cardinality fact table (make it larger to avoid reference classification)
        # Need > 10 MB to avoid reference classification
        # Create enough rows to exceed 10 MB threshold
        num_vitals_rows = 100_000  # Ensure > 10 MB
        vitals = pl.DataFrame({
            "patient_id": [f"P{i % 100}" for i in range(num_vitals_rows)],
            "charttime": [f"2024-01-{(i % 30) + 1:02d}" for i in range(num_vitals_rows)],
            "heart_rate": [70 + (i % 30) for i in range(num_vitals_rows)]
        })

        tables = {
            "patients": patients,
            "vitals": vitals
        }

        # Act
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        classifications = handler.classify_tables(anchor_table="patients")

        # Assert
        patients_class = classifications["patients"]
        assert patients_class.classification == "dimension"
        assert patients_class.is_unique_on_grain is True
        assert patients_class.cardinality_ratio <= 1.1

        vitals_class = classifications["vitals"]
        # Small table with high cardinality → reference
        # Large table with high cardinality AND N-side → event or fact
        assert vitals_class.classification in ["event", "fact", "reference"]
        assert vitals_class.cardinality_ratio > 1.1
        assert vitals_class.has_time_column is True

        # Verify estimated bytes is reasonable
        assert vitals_class.estimated_bytes > 0

        handler.close()

    def test_null_rate_calculation(self):
        """Test null rate calculation in grain key."""
        # Arrange: DataFrame with NULLs in grain key
        df_with_nulls = pl.DataFrame({
            "patient_id": ["P1", "P2", None, "P3", None],
            "value": [100, 200, 300, 400, 500]
        })

        tables = {"test": df_with_nulls}

        # Act
        handler = MultiTableHandler(tables)
        classifications = handler.classify_tables()

        # Assert
        test_class = classifications["test"]
        assert test_class.null_rate_in_grain == 0.4, (  # 2 out of 5 are NULL
            f"Expected null_rate 0.4, got {test_class.null_rate_in_grain}"
        )

        handler.close()


class TestTableClassificationEdgeCases:
    """Edge cases for table classification."""

    def test_empty_dataframe(self):
        """Test classification with empty DataFrame."""
        # Arrange
        empty_df = pl.DataFrame({
            "patient_id": pl.Series([], dtype=pl.Utf8),
            "value": pl.Series([], dtype=pl.Int64)
        })

        tables = {"empty": empty_df}

        # Act
        handler = MultiTableHandler(tables)
        classifications = handler.classify_tables()

        # Assert: Should handle gracefully
        if "empty" in classifications:
            empty_class = classifications["empty"]
            assert empty_class.estimated_bytes == 0

        handler.close()

    def test_single_row_dataframe(self):
        """Test classification with single row."""
        # Arrange
        single_row = pl.DataFrame({
            "patient_id": ["P1"],
            "age": [30]
        })

        tables = {"single": single_row}

        # Act
        handler = MultiTableHandler(tables)
        classifications = handler.classify_tables()

        # Assert
        single_class = classifications["single"]
        assert single_class.is_unique_on_grain is True
        assert single_class.cardinality_ratio == 1.0

        handler.close()

    def test_all_nulls_grain_key(self):
        """Test classification when grain key is all NULLs."""
        # Arrange
        all_nulls = pl.DataFrame({
            "patient_id": [None, None, None],
            "value": [100, 200, 300]
        })

        tables = {"nulls": all_nulls}

        # Act
        handler = MultiTableHandler(tables)
        classifications = handler.classify_tables()

        # Assert
        nulls_class = classifications["nulls"]
        assert nulls_class.null_rate_in_grain == 1.0

        handler.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

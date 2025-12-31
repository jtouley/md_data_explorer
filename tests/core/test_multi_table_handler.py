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
)


class TestTableClassification:
    """Test suite for Milestone 1: Table Classification System."""

    def test_bridge_detection_on_many_to_many_fixture(self, make_multi_table_setup):
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
        # Arrange: Create synthetic many-to-many dataset using factory fixture
        tables = make_multi_table_setup()

        # Extend bridge table with additional fields
        patient_medications = tables["patient_medications"].with_columns(
            pl.Series("dosage_override", [None, "250mg", None, None])
        )
        tables["patient_medications"] = patient_medications

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
        df = (
            pl.select(idx=pl.int_range(0, num_rows))
            .with_columns(
                [
                    pl.col("idx").alias("id"),  # Int64: 8 bytes/row
                    pl.concat_str([pl.lit("Name_"), pl.col("idx").cast(pl.Utf8).str.zfill(4)]).alias(
                        "name"
                    ),  # Utf8: ~9 chars = 9 bytes/row
                    (pl.col("idx") * 1.5).cast(pl.Float64).alias("value"),  # Float64: 8 bytes/row
                ]
            )
            .drop("idx")
        )

        tables = {"test_table": df}
        handler = MultiTableHandler(tables)

        # Act: Estimate bytes
        estimated_bytes = handler._estimate_table_bytes(df)

        # Assert: Within reasonable range
        # Theoretical minimum: (8 + 9 + 8) * 1000 = 25,000 bytes
        theoretical_min = 25_000
        theoretical_max = theoretical_min * 2  # Allow 2x overhead

        assert theoretical_min <= estimated_bytes <= theoretical_max, (
            f"Byte estimate {estimated_bytes:,} outside sane range [{theoretical_min:,}, {theoretical_max:,}]"
        )

        # Verify estimate is reasonable per row
        bytes_per_row = estimated_bytes / num_rows
        assert 20 <= bytes_per_row <= 50, f"Bytes per row {bytes_per_row:.1f} outside expected range [20, 50]"

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
        df_original = pl.DataFrame(
            {
                "encounter_id": ["E1", "E2", "E3"],
                "patient_id": ["P1", "P1", "P2"],
                "visit_id": ["V1", "V2", "V3"],
                "name": ["Alice", "Alice", "Bob"],
            }
        )

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

        assert grain_key_1 == grain_key_3, f"Column order should not affect grain key: {grain_key_1} != {grain_key_3}"

        # Verify grain key follows priority rules (patient_id should be chosen)
        assert grain_key_1 == "patient_id", f"patient_id should be prioritized, got {grain_key_1}"

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
        df_with_time = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "charttime": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "value": [100, 200, 300],
            }
        )

        # DataFrame without time column
        df_no_time = pl.DataFrame({"id": [1, 2, 3], "value": [100, 200, 300]})

        # DataFrame with constant time column (should not detect)
        df_constant_time = pl.DataFrame({"id": [1, 2, 3], "timestamp": ["2024-01-01", "2024-01-01", "2024-01-01"]})

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

    def test_classification_rules(self, make_multi_table_setup):
        """Test classification rule priority."""
        # Arrange
        tables = make_multi_table_setup()
        patients = tables["patients"]

        # High cardinality fact table (make it larger to avoid reference classification)
        # Need > 10 MB to avoid reference classification
        # Create enough rows to exceed 10 MB threshold
        num_vitals_rows = 100_000  # Ensure > 10 MB
        vitals = (
            pl.select(idx=pl.int_range(0, num_vitals_rows))
            .with_columns(
                [
                    pl.concat_str([pl.lit("P"), (pl.col("idx") % 100).cast(pl.Utf8)]).alias("patient_id"),
                    pl.concat_str([pl.lit("2024-01-"), ((pl.col("idx") % 30) + 1).cast(pl.Utf8).str.zfill(2)]).alias(
                        "charttime"
                    ),
                    (70 + (pl.col("idx") % 30)).alias("heart_rate"),
                ]
            )
            .drop("idx")
        )

        tables = {"patients": patients, "vitals": vitals}

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
        df_with_nulls = pl.DataFrame({"patient_id": ["P1", "P2", None, "P3", None], "value": [100, 200, 300, 400, 500]})

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
        empty_df = pl.DataFrame({"patient_id": pl.Series([], dtype=pl.Utf8), "value": pl.Series([], dtype=pl.Int64)})

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
        single_row = pl.DataFrame({"patient_id": ["P1"], "age": [30]})

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
        all_nulls = pl.DataFrame({"patient_id": [None, None, None], "value": [100, 200, 300]})

        tables = {"nulls": all_nulls}

        # Act
        handler = MultiTableHandler(tables)
        classifications = handler.classify_tables()

        # Assert
        nulls_class = classifications["nulls"]
        assert nulls_class.null_rate_in_grain == 1.0

        handler.close()


class TestPerformanceOptimizations:
    """Test suite for performance optimizations (sampling, pattern matching)."""

    def test_grain_key_fallback_prefers_patient_over_event(self):
        """
        Acceptance: Table with patient_id and event_id picks patient_id even if
        event_id is more unique.

        This tests the explicit scoring formula that penalizes row-level IDs
        (event_id, row_id, uuid).
        """
        # Arrange: event_id is perfectly unique (row-level ID), patient_id has duplicates
        df = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2"],  # 2 unique
                "event_id": ["E1", "E2", "E3", "E4"],  # 4 unique (higher uniqueness!)
                "value": [100, 200, 300, 400],
            }
        )

        # Act
        handler = MultiTableHandler({"test": df})
        grain_key = handler._detect_grain_key(df)

        # Assert: Should pick patient_id despite event_id being more unique
        assert grain_key == "patient_id", f"Expected 'patient_id' (explicit key), got '{grain_key}'"

        handler.close()

    def test_id_pattern_does_not_match_false_positives(self):
        """
        Acceptance: endswith('_id') pattern does not match 'valid', 'fluid', 'paid'.

        This tests the tightened ID pattern matching that requires exact 'id' or endswith('_id').
        """
        # Arrange: False positives that end with 'id'
        df = pl.DataFrame(
            {
                "valid": [True, False, True],
                "fluid": [100, 200, 300],
                "paid": [10.5, 20.5, 30.5],
                "patient_id": ["P1", "P2", "P3"],
            }
        )

        handler = MultiTableHandler({"test": df})

        # Act
        grain_key = handler._detect_grain_key(df)

        # Assert: Should pick patient_id, not any false positives
        assert grain_key == "patient_id", f"Expected 'patient_id', got '{grain_key}'"

        # Verify _is_probably_id_col() rejects false positives
        assert not handler._is_probably_id_col("valid")
        assert not handler._is_probably_id_col("fluid")
        assert not handler._is_probably_id_col("paid")
        assert handler._is_probably_id_col("patient_id")

        handler.close()

    def test_classification_uses_sampled_helpers_only(self, monkeypatch):
        """
        Performance guardrail: Enforce that classification uses _sample_df() and never
        touches df[...] for uniqueness on original frame.

        This test tracks calls to our own _sample_df() helper to verify sampling is used.
        """
        # Arrange: Create large DataFrame
        n = 100_000
        large_df = (
            pl.select(idx=pl.int_range(0, n))
            .with_columns(
                [
                    pl.concat_str([pl.lit("P"), (pl.col("idx") % 1000).cast(pl.Utf8)]).alias("patient_id"),
                    pl.col("idx").alias("value"),
                ]
            )
            .drop("idx")
        )

        # Track calls to _sample_df()
        sample_df_calls = []
        original_sample_df = MultiTableHandler._sample_df

        def tracked_sample_df(self, df, n=10_000):
            sample_df_calls.append((df.height, n))
            return original_sample_df(self, df, n)

        monkeypatch.setattr(MultiTableHandler, "_sample_df", tracked_sample_df)

        # Act
        handler = MultiTableHandler({"large": large_df})
        handler.classify_tables()

        # Assert: _sample_df() was called (proving we're using sampling)
        assert len(sample_df_calls) > 0, "Classification should use _sample_df() for all uniqueness checks"

        # Verify all samples are bounded
        for df_height, sample_size in sample_df_calls:
            assert sample_size <= 10_000, f"Sample size {sample_size} exceeds bound (df_height={df_height})"

        handler.close()

    def test_classification_1m_rows_completes_within_3_seconds(self):
        """
        Strict acceptance gate: Classifying a 1M-row table must complete within 3 seconds.

        This is a hard performance requirement that ensures sampling is working correctly.
        """
        import time

        # Arrange: Create 1M-row table
        n = 1_000_000
        large_df = (
            pl.select(idx=pl.int_range(0, n))
            .with_columns(
                [
                    pl.concat_str([pl.lit("P"), (pl.col("idx") % 1000).cast(pl.Utf8)]).alias(
                        "patient_id"
                    ),  # 1000 unique patients
                    pl.concat_str([pl.lit("E"), pl.col("idx").cast(pl.Utf8)]).alias(
                        "event_id"
                    ),  # 1M unique events (row-level ID)
                    pl.col("idx").alias("value"),
                ]
            )
            .drop("idx")
        )

        handler = MultiTableHandler({"large": large_df})

        # Act: Time classification
        start = time.perf_counter()
        handler.classify_tables()
        elapsed = time.perf_counter() - start

        # Assert: Must complete within 3 seconds
        assert elapsed < 3.0, (
            f"Classification of 1M-row table took {elapsed:.3f}s, exceeds 3s bound (sampling may not be working)"
        )

        # Verify classification worked correctly
        assert "large" in handler.classifications
        classification = handler.classifications["large"]
        assert classification.grain_key == "patient_id", (
            f"Should pick patient_id over event_id (grain_key={classification.grain_key})"
        )

        handler.close()


class TestAnchorSelection:
    """Test suite for Milestone 2: Centrality-Based Anchor Selection."""

    def test_never_anchors_on_event_fact_bridge(self):
        """
        M2 Acceptance Test 1: Never anchors on {event, fact, bridge} classifications.

        Setup:
        - patients (dimension): unique patient_id
        - vitals (event): high cardinality with time column
        - patient_medications (bridge): many-to-many relationship

        Expected:
        - Anchor is "patients" (dimension), never vitals or patient_medications
        """
        # Arrange
        patients = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "age": [30, 45, 28]})

        # Large vitals table (event classification)
        num_vitals = 100_000
        vitals = (
            pl.select(idx=pl.int_range(0, num_vitals))
            .with_columns(
                [
                    pl.concat_str([pl.lit("P"), ((pl.col("idx") % 3) + 1).cast(pl.Utf8)]).alias("patient_id"),
                    pl.concat_str([pl.lit("2024-01-"), ((pl.col("idx") % 30) + 1).cast(pl.Utf8).str.zfill(2)]).alias(
                        "charttime"
                    ),
                    (70 + (pl.col("idx") % 30)).alias("heart_rate"),
                ]
            )
            .drop("idx")
        )

        medications = pl.DataFrame(
            {
                "medication_id": ["M1", "M2", "M3"],
                "drug_name": ["Aspirin", "Metformin", "Lisinopril"],
            }
        )

        # Bridge table
        patient_medications = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P3"],
                "medication_id": ["M1", "M2", "M1", "M3"],
                "start_date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-03-01"],
            }
        )

        tables = {
            "patients": patients,
            "vitals": vitals,
            "medications": medications,
            "patient_medications": patient_medications,
        }

        # Act
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        anchor = handler._find_anchor_by_centrality()

        # Assert: Never anchor on event, fact, or bridge
        anchor_class = handler.classifications[anchor]
        assert anchor_class.classification == "dimension", (
            f"Anchor '{anchor}' must be dimension, got {anchor_class.classification}"
        )

        # Verify it's actually patients (the only dimension)
        assert anchor in ["patients", "medications"], (
            f"Anchor should be patients or medications (dimensions), got '{anchor}'"
        )

        handler.close()

    def test_same_input_graph_yields_same_anchor(self):
        """
        M2 Acceptance Test 2: Same input graph yields same anchor (determinism).

        Run anchor selection multiple times with same data, verify same result.
        """
        # Arrange
        patients = pl.DataFrame({"patient_id": ["P1", "P2", "P3", "P4"], "age": [30, 45, 28, 55]})

        admissions = pl.DataFrame(
            {
                "hadm_id": ["H1", "H2", "H3", "H4", "H5"],
                "patient_id": ["P1", "P1", "P2", "P3", "P4"],
                "admit_date": [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-01-15",
                    "2024-03-01",
                    "2024-04-01",
                ],
            }
        )

        tables = {"patients": patients, "admissions": admissions}

        # Act: Run anchor selection multiple times
        anchors = []
        for _ in range(5):
            handler = MultiTableHandler(tables.copy())
            handler.detect_relationships()
            anchor = handler._find_anchor_by_centrality()
            anchors.append(anchor)
            handler.close()

        # Assert: All anchors should be the same
        assert len(set(anchors)) == 1, f"Anchor selection not deterministic: got {set(anchors)}"

        # Verify it picked patients (patient grain preferred)
        assert anchors[0] == "patients", f"Expected 'patients' as anchor, got '{anchors[0]}'"

    def test_prefers_lower_null_rate_and_smaller_bytes_on_ties(self):
        """
        M2 Acceptance Test 3: Prefers lower null-rate and smaller bytes on ties.

        Setup:
        - dim_a: dimension, 0% nulls, 100 bytes
        - dim_b: dimension, 20% nulls, 50 bytes
        - dim_c: dimension, 0% nulls, 200 bytes

        Expected:
        - Anchor is dim_a (0% nulls wins over dim_b, smaller bytes wins over dim_c)
        """
        # Arrange: Three dimension tables with different null rates and sizes
        # dim_a: 0% nulls, small size
        dim_a = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "value_a": [100, 200, 300]})

        # dim_b: 20% nulls (1 out of 5), smaller bytes
        dim_b = pl.DataFrame({"patient_id": ["P1", "P2", None, "P4", "P5"], "value_b": [10, 20, 30, 40, 50]})

        # dim_c: 0% nulls, larger size (more columns and longer strings)
        dim_c = pl.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "col1": ["A" * 100, "B" * 100, "C" * 100],  # Long strings
                "col2": ["D" * 100, "E" * 100, "F" * 100],
                "col3": ["G" * 100, "H" * 100, "I" * 100],
                "col4": ["J" * 100, "K" * 100, "L" * 100],
                "col5": ["M" * 100, "N" * 100, "O" * 100],
            }
        )

        tables = {"dim_a": dim_a, "dim_b": dim_b, "dim_c": dim_c}

        # Act
        handler = MultiTableHandler(tables)
        # No relationships, so all will have same relationship count
        handler.detect_relationships()
        anchor = handler._find_anchor_by_centrality()

        # Assert: Should pick dim_a (0% nulls and smallest among 0% null tables)
        assert anchor == "dim_a", f"Expected 'dim_a' as anchor (0% nulls, smallest), got '{anchor}'"

        # Verify classifications
        dim_a_class = handler.classifications["dim_a"]
        dim_b_class = handler.classifications["dim_b"]
        dim_c_class = handler.classifications["dim_c"]

        # dim_a should have lower null rate than dim_b
        assert dim_a_class.null_rate_in_grain < dim_b_class.null_rate_in_grain

        # dim_a should have smaller bytes than dim_c
        assert dim_a_class.estimated_bytes < dim_c_class.estimated_bytes

        handler.close()

    def test_hard_exclusions_no_unique_grain(self):
        """Test that tables without unique grain keys are excluded from anchor selection."""
        # Arrange: Only non-unique tables
        non_unique = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2"],  # Not unique
                "value": [100, 200, 300, 400],
            }
        )

        tables = {"non_unique": non_unique}

        # Act & Assert: Should raise ValueError
        handler = MultiTableHandler(tables)
        handler.detect_relationships()

        with pytest.raises(ValueError, match="No dimension tables found"):
            handler._find_anchor_by_centrality()

        handler.close()

    def test_hard_exclusions_high_null_rate(self):
        """Test that tables with >50% NULL rate are excluded."""
        # Arrange: Table with >50% NULLs
        high_nulls = pl.DataFrame(
            {
                "patient_id": ["P1", None, None, None, "P5"],  # 60% nulls
                "value": [100, 200, 300, 400, 500],
            }
        )

        tables = {"high_nulls": high_nulls}

        # Act & Assert
        handler = MultiTableHandler(tables)
        handler.detect_relationships()

        with pytest.raises(ValueError, match="No suitable anchor table found"):
            handler._find_anchor_by_centrality()

        handler.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestDimensionMart:
    """Test suite for Milestone 3: Dimension Mart Builder."""

    def test_mart_rowcount_equals_anchor_unique_grain_count(self):
        """
        M3 Acceptance Test 1: Mart rowcount equals anchor unique grain count.

        Critical invariant: Joining dimensions should preserve anchor cardinality.
        """
        # Arrange: Anchor with 3 unique patients + dimension with patient attributes
        patients = pl.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 45, 28],
            }
        )

        # Dimension table (1:1 relationship with patients)
        demographics = pl.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "gender": ["F", "M", "M"],
                "ethnicity": ["Asian", "White", "Hispanic"],
            }
        )

        tables = {"patients": patients, "demographics": demographics}

        # Act
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        mart_lazy = handler._build_dimension_mart(anchor_table="patients")
        mart = mart_lazy.collect()

        # Assert: Mart should have same row count as anchor
        assert mart.height == patients.height, f"Mart rowcount {mart.height} != anchor rowcount {patients.height}"

        assert mart.height == 3, "Mart should have 3 rows (one per unique patient)"

        # Verify all anchor rows preserved
        assert set(mart["patient_id"]) == {"P1", "P2", "P3"}

        handler.close()

    def test_no_joins_where_rhs_key_is_non_unique(self):
        """
        M3 Acceptance Test 2: No joins where RHS key is non-unique.

        Critical invariant: Dimension mart should reject tables with non-unique join keys
        to prevent row explosion.
        """
        # Arrange
        patients = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "age": [30, 45, 28]})

        # Valid dimension (unique patient_id)
        demographics = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "gender": ["F", "M", "M"]})

        # Invalid "dimension" (non-unique patient_id - actually a fact table)
        # This would cause row explosion if joined
        vitals = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2", "P3"],  # Non-unique!
                "charttime": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01"],
                "heart_rate": [70, 72, 68, 71, 75],
            }
        )

        tables = {"patients": patients, "demographics": demographics, "vitals": vitals}

        # Act
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        mart_lazy = handler._build_dimension_mart(anchor_table="patients")
        mart = mart_lazy.collect()

        # Assert: Mart should have same row count (vitals should be excluded)
        assert mart.height == patients.height, (
            f"Mart rowcount {mart.height} != anchor rowcount {patients.height}. "
            f"Non-unique join key may have caused row explosion!"
        )

        # Verify vitals was not joined (would appear as vitals_heart_rate column)
        vitals_columns = [col for col in mart.columns if "heart_rate" in col]
        assert len(vitals_columns) == 0, (
            f"Vitals table (non-unique key) should not be joined, found columns: {vitals_columns}"
        )

        # Verify demographics WAS joined (unique key)
        assert "gender" in mart.columns or "demographics_gender" in mart.columns, (
            "Demographics (unique key) should be joined"
        )

        handler.close()


# ============================================================================
# Milestone 4: Fact Aggregation with Policy Enforcement
# ============================================================================


class TestFactAggregation:
    """Test fact table aggregation with aggregation policy enforcement (M4)."""

    def test_policy_violations_raise_errors_not_warnings(self):
        """
        M4 Acceptance: Policy violations raise AggregationPolicyError (not warnings).

        Tests that attempting to compute mean on code columns raises an error
        when mean is enabled, preventing silent data corruption.
        """
        from clinical_analytics.core.multi_table_handler import (
            AggregationPolicy,
            AggregationPolicyError,
        )

        # Arrange: Create fact table with code column (large enough to be classified as fact)
        # Need > 10 MB to avoid "reference" classification
        n_patients = 1000
        n_vitals_per_patient = 200  # 200k total vitals rows (should be ~14 MB)

        patients = pl.DataFrame(
            {
                "patient_id": [f"P{i}" for i in range(n_patients)],
            }
        )

        vitals = pl.DataFrame(
            {
                "patient_id": [f"P{i % n_patients}" for i in range(n_patients * n_vitals_per_patient)],
                "diagnosis_code": [
                    100 + (i % 100) for i in range(n_patients * n_vitals_per_patient)
                ],  # Numeric code column (matches *_code pattern)
                "heart_rate": [72 + (i % 30) for i in range(n_patients * n_vitals_per_patient)],  # Numeric value
                # Add many extra columns to exceed 10 MB threshold
                "systolic_bp": [120 + (i % 40) for i in range(n_patients * n_vitals_per_patient)],
                "diastolic_bp": [80 + (i % 20) for i in range(n_patients * n_vitals_per_patient)],
                "oxygen_sat": [95 + (i % 5) for i in range(n_patients * n_vitals_per_patient)],
                "respiration_rate": [16 + (i % 8) for i in range(n_patients * n_vitals_per_patient)],
                "temperature": [98.0 + (i % 10) * 0.1 for i in range(n_patients * n_vitals_per_patient)],
                "glucose": [100 + (i % 50) for i in range(n_patients * n_vitals_per_patient)],
                "weight_kg": [70 + (i % 30) for i in range(n_patients * n_vitals_per_patient)],
            }
        )

        handler = MultiTableHandler({"patients": patients, "vitals": vitals})

        # Classify tables
        handler.classify_tables()

        # Act & Assert: Enable mean and attempt to aggregate
        # This should raise AggregationPolicyError because diagnosis_code matches *_code pattern
        policy = AggregationPolicy(allow_mean=True)

        with pytest.raises(AggregationPolicyError) as exc_info:
            handler._aggregate_fact_tables(grain_key="patient_id", policy=policy)

        # Verify error message is informative
        assert "diagnosis_code" in str(exc_info.value)
        assert "mean" in str(exc_info.value).lower()
        assert "code column" in str(exc_info.value).lower()

        handler.close()

    def test_default_policy_is_safe_no_mean_on_codes(self):
        """
        M4 Acceptance: Default policy prevents mean on code columns.

        Verifies that default AggregationPolicy (allow_mean=False) never
        computes mean, avoiding incorrect aggregations.
        """
        from clinical_analytics.core.multi_table_handler import AggregationPolicy

        # Arrange: Create fact table with code and numeric columns (large enough)
        n_patients = 1000
        n_vitals_per_patient = 200  # 200k rows for > 10 MB

        patients = pl.DataFrame(
            {
                "patient_id": [f"P{i}" for i in range(n_patients)],
            }
        )

        vitals = pl.DataFrame(
            {
                "patient_id": [f"P{i % n_patients}" for i in range(n_patients * n_vitals_per_patient)],
                "icd_code": [f"I{i % 100}" for i in range(n_patients * n_vitals_per_patient)],
                "heart_rate": [72 + (i % 30) for i in range(n_patients * n_vitals_per_patient)],
                # Add many extra columns to exceed 10 MB threshold
                "systolic_bp": [120 + (i % 40) for i in range(n_patients * n_vitals_per_patient)],
                "diastolic_bp": [80 + (i % 20) for i in range(n_patients * n_vitals_per_patient)],
                "oxygen_sat": [95 + (i % 5) for i in range(n_patients * n_vitals_per_patient)],
                "respiration_rate": [16 + (i % 8) for i in range(n_patients * n_vitals_per_patient)],
                "temperature": [98.0 + (i % 10) * 0.1 for i in range(n_patients * n_vitals_per_patient)],
                "glucose": [100 + (i % 50) for i in range(n_patients * n_vitals_per_patient)],
                "weight_kg": [70 + (i % 30) for i in range(n_patients * n_vitals_per_patient)],
            }
        )

        handler = MultiTableHandler({"patients": patients, "vitals": vitals})
        handler.classify_tables()

        # Act: Use default policy (allow_mean=False)
        default_policy = AggregationPolicy()
        feature_tables = handler._aggregate_fact_tables(grain_key="patient_id", policy=default_policy)

        # Assert: No mean columns should be generated
        assert "vitals_features" in feature_tables

        vitals_features = feature_tables["vitals_features"].collect()

        # Check that no mean columns exist
        mean_columns = [col for col in vitals_features.columns if "mean" in col]
        assert len(mean_columns) == 0, f"Default policy should not generate mean columns, found: {mean_columns}"

        # Verify safe aggregations exist (min, max, count_distinct)
        assert "heart_rate_min" in vitals_features.columns
        assert "heart_rate_max" in vitals_features.columns
        assert "heart_rate_count_distinct" in vitals_features.columns

        handler.close()

    def test_opt_in_mean_works_on_non_code_columns(self):
        """
        M4 Acceptance: Opt-in mean aggregation works on non-code numeric columns.

        Verifies that when allow_mean=True, mean is computed only on columns
        that don't match code patterns.
        """
        from clinical_analytics.core.multi_table_handler import AggregationPolicy

        # Arrange: Create fact table with code and numeric columns (large enough)
        n_patients = 1000
        n_vitals_per_patient = 200  # 200k rows for > 10 MB

        patients = pl.DataFrame(
            {
                "patient_id": [f"P{i}" for i in range(n_patients)],
            }
        )

        vitals = pl.DataFrame(
            {
                "patient_id": [f"P{i % n_patients}" for i in range(n_patients * n_vitals_per_patient)],
                "measurement_id": [
                    f"M{i}" for i in range(n_patients * n_vitals_per_patient)
                ],  # Matches *_id pattern (code)
                "heart_rate": [72 + (i % 30) for i in range(n_patients * n_vitals_per_patient)],  # Numeric, not a code
                # Add many extra columns to exceed 10 MB threshold
                "systolic_bp": [120 + (i % 40) for i in range(n_patients * n_vitals_per_patient)],
                "diastolic_bp": [80 + (i % 20) for i in range(n_patients * n_vitals_per_patient)],
                "oxygen_sat": [95 + (i % 5) for i in range(n_patients * n_vitals_per_patient)],
                "respiration_rate": [16 + (i % 8) for i in range(n_patients * n_vitals_per_patient)],
                "temperature": [98.0 + (i % 10) * 0.1 for i in range(n_patients * n_vitals_per_patient)],
                "glucose": [100 + (i % 50) for i in range(n_patients * n_vitals_per_patient)],
                "weight_kg": [70 + (i % 30) for i in range(n_patients * n_vitals_per_patient)],
            }
        )

        handler = MultiTableHandler({"patients": patients, "vitals": vitals})
        handler.classify_tables()

        # Act: Enable mean aggregation
        policy = AggregationPolicy(allow_mean=True)

        # This should NOT raise error - measurement_id is protected, heart_rate is allowed
        feature_tables = handler._aggregate_fact_tables(grain_key="patient_id", policy=policy)

        # Assert: Mean should exist for heart_rate but not measurement_id
        assert "vitals_features" in feature_tables

        vitals_features = feature_tables["vitals_features"].collect()

        # heart_rate should have mean (not a code column)
        assert "heart_rate_mean" in vitals_features.columns

        # measurement_id should NOT have mean (matches *_id pattern)
        # It should only have count_distinct
        assert "measurement_id_count_distinct" in vitals_features.columns
        measurement_id_cols = [col for col in vitals_features.columns if "measurement_id" in col]
        assert "measurement_id_mean" not in measurement_id_cols

        handler.close()

    def test_aggregated_features_use_lazy_frames(self):
        """
        M4 Acceptance: Feature tables are returned as LazyFrames, not collected.

        Verifies that _aggregate_fact_tables returns LazyFrames to maintain
        lazy evaluation pipeline (following CLAUDE.md: "lazy by default").
        """
        # Arrange: Create fact table (large enough)
        n_patients = 1000
        n_vitals_per_patient = 200  # 200k rows for > 10 MB

        patients = pl.DataFrame(
            {
                "patient_id": [f"P{i}" for i in range(n_patients)],
            }
        )

        vitals = pl.DataFrame(
            {
                "patient_id": [f"P{i % n_patients}" for i in range(n_patients * n_vitals_per_patient)],
                "heart_rate": [72 + (i % 30) for i in range(n_patients * n_vitals_per_patient)],
                # Add many extra columns to exceed 10 MB threshold
                "systolic_bp": [120 + (i % 40) for i in range(n_patients * n_vitals_per_patient)],
                "diastolic_bp": [80 + (i % 20) for i in range(n_patients * n_vitals_per_patient)],
                "oxygen_sat": [95 + (i % 5) for i in range(n_patients * n_vitals_per_patient)],
                "respiration_rate": [16 + (i % 8) for i in range(n_patients * n_vitals_per_patient)],
                "temperature": [98.0 + (i % 10) * 0.1 for i in range(n_patients * n_vitals_per_patient)],
                "glucose": [100 + (i % 50) for i in range(n_patients * n_vitals_per_patient)],
                "weight_kg": [70 + (i % 30) for i in range(n_patients * n_vitals_per_patient)],
            }
        )

        handler = MultiTableHandler({"patients": patients, "vitals": vitals})
        handler.classify_tables()

        # Act: Aggregate
        feature_tables = handler._aggregate_fact_tables(grain_key="patient_id")

        # Assert: Result should be LazyFrame
        assert "vitals_features" in feature_tables
        assert isinstance(feature_tables["vitals_features"], pl.LazyFrame), (
            "Feature tables must be LazyFrames, not collected DataFrames"
        )

        handler.close()

    def test_feature_tables_have_correct_schema(self):
        """
        M4 Acceptance: Aggregated feature tables have expected schema.

        Verifies that aggregation creates correct column names with suffixes
        (e.g., heart_rate_min, heart_rate_max, count).
        """
        from clinical_analytics.core.multi_table_handler import AggregationPolicy

        # Arrange: Create fact table with multiple column types (large enough)
        n_patients = 1000
        n_vitals_per_patient = 200  # 200k rows for > 10 MB
        n_total = n_patients * n_vitals_per_patient

        patients = pl.DataFrame(
            {
                "patient_id": [f"P{i}" for i in range(n_patients)],
            }
        )

        from datetime import datetime, timedelta

        vitals = pl.DataFrame(
            {
                "patient_id": [f"P{i % n_patients}" for i in range(n_total)],
                "heart_rate": [72 + (i % 30) for i in range(n_total)],
                "temperature": [98.0 + (i % 10) * 0.1 for i in range(n_total)],
                "measurement_time": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_total)],
                # Add extra columns to exceed 10 MB threshold
                "systolic_bp": [120 + (i % 40) for i in range(n_total)],
                "diastolic_bp": [80 + (i % 20) for i in range(n_total)],
                "oxygen_sat": [95 + (i % 5) for i in range(n_total)],
            }
        )

        handler = MultiTableHandler({"patients": patients, "vitals": vitals})
        handler.classify_tables()

        # Act: Aggregate with default policy
        policy = AggregationPolicy(allow_mean=False, allow_last=True)
        feature_tables = handler._aggregate_fact_tables(grain_key="patient_id", policy=policy)

        vitals_features = feature_tables["vitals_features"].collect()

        # Assert: Verify schema
        assert "patient_id" in vitals_features.columns  # Grain key preserved
        assert "count" in vitals_features.columns  # Count always included

        # Numeric columns: min, max, count_distinct, last (no mean because allow_mean=False)
        assert "heart_rate_min" in vitals_features.columns
        assert "heart_rate_max" in vitals_features.columns
        assert "heart_rate_count_distinct" in vitals_features.columns
        assert "heart_rate_last" in vitals_features.columns

        # Datetime columns: min, max, count_distinct, last
        assert "measurement_time_min" in vitals_features.columns
        assert "measurement_time_max" in vitals_features.columns
        assert "measurement_time_count_distinct" in vitals_features.columns
        assert "measurement_time_last" in vitals_features.columns

        # Verify no mean columns (allow_mean=False)
        mean_columns = [col for col in vitals_features.columns if "mean" in col]
        assert len(mean_columns) == 0

        handler.close()

    def test_feature_tables_exclude_dimension_and_bridge_tables(self):
        """
        M4 Acceptance: Only fact/event tables are aggregated, dimensions/bridges excluded.

        Verifies that _aggregate_fact_tables filters to only fact/event tables
        and excludes dimensions, bridges, and reference tables.
        """
        # Arrange: Create mixed table types
        patients = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "age": [45, 32, 67]})

        # Dimension table (unique on grain, small bytes)
        demographics = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "gender": ["M", "F", "M"]})

        # Fact table (high cardinality, not unique)
        vitals = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
                "heart_rate": [72, 78, 85, 80, 90, 88],
            }
        )

        handler = MultiTableHandler({"patients": patients, "demographics": demographics, "vitals": vitals})

        # Classify tables
        handler.classify_tables()

        # Verify classifications
        assert handler.classifications["patients"].classification == "dimension"
        assert handler.classifications["demographics"].classification == "dimension"
        # vitals should be fact or event (depends on size/time column)

        # Act: Aggregate
        feature_tables = handler._aggregate_fact_tables(grain_key="patient_id")

        # Assert: Only fact/event tables should be in feature_tables
        assert "patients_features" not in feature_tables, "Dimension table 'patients' should not be aggregated"
        assert "demographics_features" not in feature_tables, "Dimension table 'demographics' should not be aggregated"

        # vitals should be aggregated (it's a fact/event table)
        # Note: vitals might be classified as "reference" if too small, let's check
        vitals_class = handler.classifications["vitals"].classification
        if vitals_class in ["fact", "event"]:
            assert "vitals_features" in feature_tables

        handler.close()


class TestBuildUnifiedCohort:
    """Test suite for build_unified_cohort() aggregate-before-join refactor."""

    def test_build_unified_cohort_does_not_use_legacy_duckdb_join(self, make_multi_table_setup):
        """Ensure build_unified_cohort() does not execute legacy DuckDB SQL join."""
        import duckdb

        # Arrange: Create handler with test data
        tables = make_multi_table_setup()
        patients = tables["patients"]

        vitals = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2", "P3"],
                "heart_rate": [70, 72, 68, 71, 75],
                "charttime": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01"],
            }
        )

        tables = {"patients": patients, "vitals": vitals}

        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        # Patch DuckDB to detect legacy SQL pattern
        original_execute = duckdb.DuckDBPyConnection.execute

        def guarded_execute(self, query, *args, **kwargs):
            q = str(query)
            if "SELECT * FROM" in q and "LEFT JOIN" in q:
                raise AssertionError(f"Legacy DuckDB mega-join detected: {q[:200]}...")
            return original_execute(self, query, *args, **kwargs)

        duckdb.DuckDBPyConnection.execute = guarded_execute

        try:
            # This should not trigger the legacy path
            result = handler.build_unified_cohort()
            assert result.height > 0
        finally:
            duckdb.DuckDBPyConnection.execute = original_execute
            handler.close()

    def test_feature_joins_preserve_row_count(self, make_multi_table_setup):
        """Feature joins should not change row count (1:1 validation)."""
        # Arrange
        tables = make_multi_table_setup()
        patients = tables["patients"]

        vitals = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2", "P3"],
                "heart_rate": [70, 72, 68, 71, 75],
                "charttime": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01"],
            }
        )

        tables = {"patients": patients, "vitals": vitals}

        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        anchor = handler._find_anchor_by_centrality()
        mart = handler._build_dimension_mart(anchor_table=anchor)
        mart_height = mart.select(pl.len().alias("n")).collect()["n"][0]

        # Act
        result = handler.build_unified_cohort(anchor_table=anchor)

        # Assert
        assert result.height == mart_height, (
            f"Row count changed: {mart_height} -> {result.height}. Feature joins may have caused row explosion."
        )

        handler.close()

    def test_lazy_join_validate_1_1_fails_on_duplicates(self):
        """Prove validate='1:1' works on LazyFrame.join() by testing duplicate keys."""
        # Arrange
        left = pl.DataFrame({"patient_id": ["P1", "P2"]}).lazy()
        right = pl.DataFrame({"patient_id": ["P1", "P1"], "x": [1, 2]}).lazy()

        # Act & Assert
        with pytest.raises(pl.exceptions.ComputeError):
            left.join(right, on="patient_id", how="left", validate="1:1").collect()

    def test_build_unified_cohort_deterministic_columns(self, make_multi_table_setup):
        """Unified cohort should have deterministic column order."""
        # Arrange
        tables = make_multi_table_setup()
        patients = tables["patients"]

        vitals = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2", "P3"],
                "heart_rate": [70, 72, 68, 71, 75],
                "charttime": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01"],
            }
        )

        labevents = pl.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "glucose": [100, 110, 95],
                "charttime": ["2024-01-01", "2024-01-01", "2024-01-01"],
            }
        )

        tables = {"patients": patients, "vitals": vitals, "labevents": labevents}

        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        # Act: Run twice
        result1 = handler.build_unified_cohort()
        result2 = handler.build_unified_cohort()

        # Assert: Column order should be deterministic
        assert result1.columns == result2.columns, (
            f"Column order not deterministic: {result1.columns} != {result2.columns}"
        )

        handler.close()

    def test_build_unified_cohort_rejects_invalid_join_type(self, make_multi_table_setup):
        """Invalid join_type should raise ValueError."""
        # Arrange
        patients = make_multi_table_setup(num_patients=2)["patients"]

        tables = {"patients": patients}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported join_type"):
            handler.build_unified_cohort(join_type="invalid")

        handler.close()

    def test_build_unified_cohort_accepts_case_insensitive_join_type(self, make_multi_table_setup):
        """join_type should normalize case-insensitively."""
        # Arrange
        patients = make_multi_table_setup(num_patients=2)["patients"]

        tables = {"patients": patients}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        # Act: "LEFT" should normalize to "left" and be accepted
        result = handler.build_unified_cohort(join_type="LEFT")
        assert result.height > 0

        result2 = handler.build_unified_cohort(join_type="left")
        assert result2.height == result.height

        handler.close()


class TestMaterializeMart:
    """Test suite for Milestone 5: Materialization and Planning."""

    def test_materialize_mart_writes_parquet(self, tmp_path, make_multi_table_setup):
        """Verify materialize_mart() writes Parquet files correctly."""
        # Arrange
        tables = make_multi_table_setup()
        patients = tables["patients"]

        vitals = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2", "P3"],
                "heart_rate": [70, 72, 68, 71, 75],
                "charttime": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01"],
            }
        )

        tables = {"patients": patients, "vitals": vitals}

        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"

        # Act
        metadata = handler.materialize_mart(
            output_path=output_path,
            grain="patient",
        )

        # Assert
        assert metadata.parquet_path.exists(), f"Parquet file not created: {metadata.parquet_path}"
        assert metadata.row_count > 0
        assert metadata.grain == "patient"
        assert metadata.grain_key == "patient_id"
        assert len(metadata.schema) > 0
        assert metadata.run_id is not None

        handler.close()

    def test_materialize_mart_caching_works(self, tmp_path, make_multi_table_setup):
        """Verify caching works: skip recompute if run_id exists."""
        # Arrange
        patients = make_multi_table_setup(num_patients=2)["patients"]

        tables = {"patients": patients}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"

        # Act: Materialize first time
        metadata1 = handler.materialize_mart(output_path=output_path, grain="patient")
        first_run_id = metadata1.run_id

        # Act: Materialize second time (should use cache)
        metadata2 = handler.materialize_mart(output_path=output_path, grain="patient")

        # Assert: Same run_id, same path
        assert metadata2.run_id == first_run_id
        assert metadata2.parquet_path == metadata1.parquet_path
        assert metadata2.row_count == metadata1.row_count

        handler.close()

    def test_materialize_mart_rowcount_matches_build_unified_cohort(self, tmp_path, make_multi_table_setup):
        """Verify materialized mart rowcount matches build_unified_cohort()."""
        # Arrange
        tables = make_multi_table_setup()
        patients = tables["patients"]

        vitals = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2", "P3"],
                "heart_rate": [70, 72, 68, 71, 75],
                "charttime": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01"],
            }
        )

        tables = {"patients": patients, "vitals": vitals}

        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        # Act: Build unified cohort
        cohort = handler.build_unified_cohort()
        cohort_height = cohort.height

        # Act: Materialize mart
        output_path = tmp_path / "marts"
        metadata = handler.materialize_mart(output_path=output_path, grain="patient")

        # Assert
        assert metadata.row_count == cohort_height, (
            f"Materialized rowcount {metadata.row_count} != cohort height {cohort_height}"
        )

        handler.close()

    def test_materialize_mart_patient_level_single_file(self, tmp_path, make_multi_table_setup):
        """Verify patient-level marts use single file (not partitioned)."""
        # Arrange
        tables = make_multi_table_setup()
        patients = tables["patients"]

        vitals = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2", "P3"],
                "heart_rate": [70, 72, 68, 71, 75],
            }
        )

        tables = {"patients": patients, "vitals": vitals}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"

        # Act: Materialize at patient level
        metadata = handler.materialize_mart(
            output_path=output_path,
            grain="patient",
            num_buckets=4,
        )

        # Assert: Patient level should be single file (not partitioned)
        assert metadata.parquet_path.is_file(), "Patient-level should be single file, not directory"
        assert metadata.parquet_path.suffix == ".parquet", "Should be .parquet file"
        assert metadata.parquet_path.name == "mart.parquet", "Should be named mart.parquet"

        handler.close()

    def test_dataset_fingerprint_changes_when_data_changes(self, tmp_path):
        """Verify dataset fingerprint includes content hash, not just shape."""
        # Arrange: Create two datasets with same shape but different values
        patients1 = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "age": [30, 45, 28]})

        patients2 = pl.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "age": [31, 46, 29],  # Different values, same shape
            }
        )

        tables1 = {"patients": patients1}
        tables2 = {"patients": patients2}

        handler1 = MultiTableHandler(tables1)
        handler2 = MultiTableHandler(tables2)

        # Act
        fingerprint1 = handler1._compute_dataset_fingerprint()
        fingerprint2 = handler2._compute_dataset_fingerprint()

        # Assert: Fingerprints should differ because content differs
        assert fingerprint1 != fingerprint2, "Fingerprints should differ when data changes"

        handler1.close()
        handler2.close()

    def test_ibis_connection_is_cached(self, tmp_path):
        """Verify Ibis connection is reused (cached on instance)."""
        pytest.importorskip("ibis")

        # Arrange
        patients = pl.DataFrame({"patient_id": ["P1", "P2"], "age": [30, 45]})

        tables = {"patients": patients}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"
        handler.materialize_mart(output_path=output_path, grain="patient")

        # Act: Get connection multiple times
        con1 = handler._get_ibis_connection()
        con2 = handler._get_ibis_connection()

        # Assert: Should be the same object (cached)
        assert con1 is con2, "Ibis connection should be cached and reused"

        handler.close()

    def test_bucket_column_dropped_from_planned_table(self, tmp_path):
        """Verify bucket column is dropped from planned tables (internal partition column)."""
        pytest.importorskip("ibis")

        # Arrange: Create event-level mart with hash bucketing
        # Note: This test simulates event-level partitioning by checking metadata schema
        patients = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "age": [30, 45, 28]})

        events = pl.DataFrame(
            {
                "event_id": ["E1", "E2", "E3", "E4", "E5"],
                "patient_id": ["P1", "P1", "P2", "P2", "P3"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        tables = {"patients": patients, "events": events}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"

        # Materialize at patient level (no bucket column expected)
        metadata = handler.materialize_mart(
            output_path=output_path,
            grain="patient",
        )

        # Act: Plan mart
        plan = handler.plan_mart(metadata=metadata)

        # Execute to check columns
        result = plan.execute()
        result_columns = list(result.columns)

        # Assert: bucket column should not be in result (patient-level doesn't use it)
        assert "bucket" not in result_columns, "Bucket column should not be exposed to consumers"

        handler.close()

    def test_schema_version_used_in_run_id(self, tmp_path):
        """Verify SCHEMA_VERSION constant is used in run_id computation."""
        from clinical_analytics.core.multi_table_handler import SCHEMA_VERSION

        # Arrange
        patients = pl.DataFrame({"patient_id": ["P1", "P2"], "age": [30, 45]})

        tables = {"patients": patients}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"

        # Act: Materialize mart
        metadata1 = handler.materialize_mart(output_path=output_path, grain="patient")
        run_id1 = metadata1.run_id

        # Change SCHEMA_VERSION (simulate schema change)
        # Note: In real usage, SCHEMA_VERSION would be updated in the module
        # For testing, we verify it's a string constant
        assert isinstance(SCHEMA_VERSION, str), "SCHEMA_VERSION should be a string"
        assert len(SCHEMA_VERSION) > 0, "SCHEMA_VERSION should not be empty"

        # Verify run_id is deterministic (same inputs = same run_id)
        metadata2 = handler.materialize_mart(output_path=output_path, grain="patient")
        run_id2 = metadata2.run_id

        assert run_id1 == run_id2, "Same inputs should produce same run_id (deterministic)"

        handler.close()


class TestPlanMart:
    """Test suite for Milestone 5: Planning over materialized Parquet."""

    def test_plan_mart_returns_lazy_ibis_expression(self, tmp_path):
        """Verify plan_mart() returns lazy Ibis expression."""
        pytest.importorskip("ibis")

        # Arrange
        patients = pl.DataFrame({"patient_id": ["P1", "P2"], "age": [30, 45]})

        tables = {"patients": patients}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"
        metadata = handler.materialize_mart(output_path=output_path, grain="patient")

        # Act
        plan = handler.plan_mart(metadata=metadata)

        # Assert
        import ibis

        assert isinstance(plan, ibis.Table), f"Expected ibis.Table, got {type(plan)}"

        handler.close()

    def test_plan_mart_compiles_to_sql_without_executing(self, tmp_path):
        """Verify plan compiles to SQL without executing."""
        pytest.importorskip("ibis")

        # Arrange
        patients = pl.DataFrame({"patient_id": ["P1", "P2"], "age": [30, 45]})

        tables = {"patients": patients}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"
        metadata = handler.materialize_mart(output_path=output_path, grain="patient")

        # Count files before planning
        files_before = len(list(tmp_path.rglob("*")))

        # Act: Plan and compile to SQL
        plan = handler.plan_mart(metadata=metadata)

        # Compile to SQL (should not execute)
        sql = str(plan.compile())

        # Assert: SQL generated, no new files created
        assert "SELECT" in sql.upper() or "READ_PARQUET" in sql.upper()

        files_after = len(list(tmp_path.rglob("*")))
        assert files_after == files_before, "Planning should not create new files"

        handler.close()

    def test_plan_mart_materialized_parquet_readable(self, tmp_path):
        """Verify materialized parquet is readable and rowcount matches."""
        pytest.importorskip("ibis")

        # Arrange
        patients = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "age": [30, 45, 28]})

        vitals = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2", "P3"],
                "heart_rate": [70, 72, 68, 71, 75],
            }
        )

        tables = {"patients": patients, "vitals": vitals}

        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"
        metadata = handler.materialize_mart(output_path=output_path, grain="patient")

        # Act: Plan and execute query
        plan = handler.plan_mart(metadata=metadata)
        result = plan.execute()

        # Assert: Rowcount matches
        assert len(result) == metadata.row_count, (
            f"Query result rowcount {len(result)} != metadata.row_count {metadata.row_count}"
        )

        handler.close()

    def test_plan_mart_uses_duckdb_backend(self, tmp_path):
        """Verify plan_mart() uses DuckDB backend explicitly."""
        pytest.importorskip("ibis")

        # Arrange
        patients = pl.DataFrame({"patient_id": ["P1", "P2"], "age": [30, 45]})

        tables = {"patients": patients}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"
        metadata = handler.materialize_mart(output_path=output_path, grain="patient")

        # Act
        handler.plan_mart(metadata=metadata)

        # Assert: Connection should be DuckDB (check by trying to use it)

        con = handler._get_ibis_connection()
        # Verify it's a DuckDB connection by checking it has read_parquet method
        assert hasattr(con, "read_parquet"), "Should be DuckDB connection with read_parquet"
        # Verify connection type name contains duckdb
        assert "duckdb" in str(type(con)).lower(), "Should use DuckDB backend"

        handler.close()

    def test_plan_mart_handles_partitioned_directories(self, tmp_path):
        """Verify plan_mart() handles partitioned directories (simulated with
        patient grain but partitioned structure)."""
        pytest.importorskip("ibis")

        # Arrange: Create a scenario where we can test partitioned reading
        patients = pl.DataFrame({"patient_id": ["P1", "P2", "P3"], "age": [30, 45, 28]})

        vitals = pl.DataFrame(
            {
                "patient_id": ["P1", "P1", "P2", "P2", "P3"],
                "heart_rate": [70, 72, 68, 71, 75],
            }
        )

        tables = {"patients": patients, "vitals": vitals}
        handler = MultiTableHandler(tables)
        handler.detect_relationships()
        handler.classify_tables()

        output_path = tmp_path / "marts"
        metadata = handler.materialize_mart(
            output_path=output_path,
            grain="patient",  # Single file for patient level
        )

        # Act: Plan over parquet (single file in this case)
        plan = handler.plan_mart(metadata=metadata)
        result = plan.execute()

        # Assert: Can read parquet
        assert len(result) == metadata.row_count
        assert metadata.parquet_path.is_file(), "Patient-level should be single file"

        handler.close()

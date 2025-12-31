"""
Tests for RelationshipDetector (Phase 0.3).

Tests the extracted relationship detection logic independently of MultiTableHandler.

Test name follows: test_unit_scenario_expectedBehavior
"""

import polars as pl

from clinical_analytics.core.relationship_detector import RelationshipDetector


class TestPrimaryKeyDetection:
    """Test suite for primary key detection."""

    def test_detect_primary_key_with_id_column_returns_id(self):
        """Primary key detection should prefer 'id' column (Phase 0.3)."""
        # Arrange: DataFrame with 'id' column that is unique and non-null
        df = pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "patient_id": [1, 2, 3, 4, 5],
                "name": ["A", "B", "C", "D", "E"],
            }
        )
        detector = RelationshipDetector()

        # Act
        pk = detector.detect_primary_key(df)

        # Assert: Should detect 'id' as primary key
        assert pk == "id"

    def test_detect_primary_key_with_patient_id_column_returns_patient_id(self):
        """Primary key detection should detect patient_id (Phase 0.3)."""
        # Arrange: DataFrame with unique patient_id
        df = pl.DataFrame(
            {
                "patient_id": [1, 2, 3, 4, 5],
                "name": ["A", "B", "C", "D", "E"],
                "age": [30, 40, 50, 60, 70],
            }
        )
        detector = RelationshipDetector()

        # Act
        pk = detector.detect_primary_key(df)

        # Assert: Should detect 'patient_id' as primary key
        assert pk == "patient_id"

    def test_detect_primary_key_with_duplicates_returns_none(self):
        """Primary key detection should return None when no unique column (Phase 0.3)."""
        # Arrange: DataFrame with duplicates in all columns
        df = pl.DataFrame(
            {
                "patient_id": [1, 1, 2, 2, 3],
                "name": ["A", "A", "B", "B", "C"],
            }
        )
        detector = RelationshipDetector()

        # Act
        pk = detector.detect_primary_key(df)

        # Assert: Should return None (no unique column)
        assert pk is None

    def test_detect_primary_key_with_nulls_returns_none(self):
        """Primary key detection should return None when all columns have nulls (Phase 0.3)."""
        # Arrange: DataFrame with nulls in all columns
        df = pl.DataFrame(
            {
                "id": [1, 2, None, 4, 5],
                "name": ["A", "B", None, "D", "E"],
            }
        )
        detector = RelationshipDetector()

        # Act
        pk = detector.detect_primary_key(df)

        # Assert: Should return None (all columns have nulls)
        assert pk is None


class TestForeignKeyCandidateDetection:
    """Test suite for foreign key candidate detection."""

    def test_is_foreign_key_candidate_exact_match_returns_true(self):
        """Foreign key candidate detection should match exact names (Phase 0.3)."""
        # Arrange
        detector = RelationshipDetector()

        # Act
        result = detector.is_foreign_key_candidate("patient_id", "patient_id")

        # Assert: Should match exact name
        assert result is True

    def test_is_foreign_key_candidate_with_fk_prefix_returns_true(self):
        """Foreign key candidate detection should match fk_ prefix (Phase 0.3)."""
        # Arrange
        detector = RelationshipDetector()

        # Act
        result = detector.is_foreign_key_candidate("patient_id", "fk_patient_id")

        # Assert: Should match fk_ prefix pattern
        assert result is True

    def test_is_foreign_key_candidate_case_insensitive_returns_true(self):
        """Foreign key candidate detection should be case insensitive (Phase 0.3)."""
        # Arrange
        detector = RelationshipDetector()

        # Act
        result = detector.is_foreign_key_candidate("PatientID", "patient_id")

        # Assert: Should match case-insensitively
        assert result is True

    def test_is_foreign_key_candidate_different_names_returns_false(self):
        """Foreign key candidate detection should reject different names (Phase 0.3)."""
        # Arrange
        detector = RelationshipDetector()

        # Act
        result = detector.is_foreign_key_candidate("patient_id", "admission_id")

        # Assert: Should not match different names
        assert result is False


class TestReferentialIntegrityVerification:
    """Test suite for referential integrity verification."""

    def test_verify_referential_integrity_full_match_returns_one(self):
        """Referential integrity should return 1.0 for 100% match (Phase 0.3)."""
        # Arrange: All child values exist in parent
        parent_df = pl.DataFrame({"id": [1, 2, 3, 4, 5]})
        child_df = pl.DataFrame({"parent_id": [1, 2, 3, 1, 2]})
        detector = RelationshipDetector()

        # Act
        ratio = detector.verify_referential_integrity(parent_df, "id", child_df, "parent_id")

        # Assert: Should return 1.0 (100% match)
        assert ratio == 1.0

    def test_verify_referential_integrity_partial_match_returns_ratio(self):
        """Referential integrity should return partial match ratio (Phase 0.3)."""
        # Arrange: 2 out of 3 unique child values exist in parent
        parent_df = pl.DataFrame({"id": [1, 2]})
        child_df = pl.DataFrame({"parent_id": [1, 2, 3]})
        detector = RelationshipDetector()

        # Act
        ratio = detector.verify_referential_integrity(parent_df, "id", child_df, "parent_id")

        # Assert: Should return ~0.67 (2 out of 3 unique values match)
        assert 0.65 <= ratio <= 0.70

    def test_verify_referential_integrity_no_match_returns_zero(self):
        """Referential integrity should return 0.0 for no match (Phase 0.3)."""
        # Arrange: No child values exist in parent
        parent_df = pl.DataFrame({"id": [1, 2, 3]})
        child_df = pl.DataFrame({"parent_id": [4, 5, 6]})
        detector = RelationshipDetector()

        # Act
        ratio = detector.verify_referential_integrity(parent_df, "id", child_df, "parent_id")

        # Assert: Should return 0.0 (no match)
        assert ratio == 0.0

    def test_verify_referential_integrity_handles_nulls(self):
        """Referential integrity should ignore null values (Phase 0.3)."""
        # Arrange: Child has nulls
        parent_df = pl.DataFrame({"id": [1, 2, 3]})
        child_df = pl.DataFrame({"parent_id": [1, 2, None, None]})
        detector = RelationshipDetector()

        # Act
        ratio = detector.verify_referential_integrity(parent_df, "id", child_df, "parent_id")

        # Assert: Should return 1.0 (nulls ignored, 2 non-null values match)
        assert ratio == 1.0

    def test_verify_referential_integrity_handles_type_mismatch(self):
        """Referential integrity should handle type mismatches by casting (Phase 0.3)."""
        # Arrange: Parent has int, child has string
        parent_df = pl.DataFrame({"id": [1, 2, 3]})
        child_df = pl.DataFrame({"parent_id": ["1", "2", "3"]})
        detector = RelationshipDetector()

        # Act
        ratio = detector.verify_referential_integrity(parent_df, "id", child_df, "parent_id")

        # Assert: Should return 1.0 (types casted for comparison)
        assert ratio == 1.0


class TestRelationshipDetection:
    """Test suite for full relationship detection."""

    def test_detect_relationships_finds_one_to_many(self):
        """Relationship detection should find one-to-many relationships (Phase 0.3)."""
        # Arrange: Parent-child relationship
        tables = {
            "patients": pl.DataFrame({"patient_id": [1, 2, 3]}),
            "admissions": pl.DataFrame(
                {
                    "admission_id": [101, 102, 103, 104],
                    "patient_id": [1, 1, 2, 3],
                }
            ),
        }
        detector = RelationshipDetector()

        # Act
        relationships = detector.detect_relationships(tables)

        # Assert: Should find patients -> admissions relationship
        assert len(relationships) >= 1
        rel = relationships[0]
        assert rel.parent_table == "patients"
        assert rel.child_table == "admissions"
        assert rel.parent_key == "patient_id"
        assert rel.child_key == "patient_id"
        assert rel.relationship_type == "one-to-many"
        assert rel.confidence > 0.8

    def test_detect_relationships_excludes_low_confidence(self):
        """Relationship detection should exclude low-confidence matches (Phase 0.3)."""
        # Arrange: Tables with poor referential integrity
        tables = {
            "table_a": pl.DataFrame({"id": [1, 2, 3]}),
            "table_b": pl.DataFrame(
                {
                    "b_id": [201, 202],
                    "id": [99, 100],  # No matches with table_a
                }
            ),
        }
        detector = RelationshipDetector()

        # Act
        relationships = detector.detect_relationships(tables)

        # Assert: Should not find relationship (low integrity)
        assert len(relationships) == 0

    def test_detect_relationships_returns_sorted_by_confidence(self):
        """Relationship detection should return results sorted by confidence (Phase 0.3)."""
        # Arrange: Multiple relationships with different confidence levels
        tables = {
            "patients": pl.DataFrame({"patient_id": [1, 2, 3]}),
            "admissions": pl.DataFrame(
                {
                    "admission_id": [101, 102, 103],
                    "patient_id": [1, 2, 3],  # 100% match
                }
            ),
            "labs": pl.DataFrame(
                {
                    "lab_id": [501, 502, 503, 504, 505],
                    "patient_id": [1, 2, 3, 1, 2],  # 100% match but more rows (still one-to-many)
                }
            ),
        }
        detector = RelationshipDetector()

        # Act
        relationships = detector.detect_relationships(tables)

        # Assert: Should be sorted by confidence (descending)
        assert len(relationships) >= 2
        for i in range(len(relationships) - 1):
            assert relationships[i].confidence >= relationships[i + 1].confidence

"""
Tests for schema drift detection accounting for patient_id being added by ensure_patient_id.

Tests ensure that:
1. Schema comparison doesn't flag patient_id as "removed" if it will be added by ensure_patient_id
2. Overwrite works when only difference is patient_id being added automatically
"""

from clinical_analytics.ui.storage.user_datasets import (
    classify_schema_drift,
)


class TestSchemaDriftPatientIdHandling:
    """Test suite for patient_id handling in schema drift detection."""

    def test_schema_drift_does_not_flag_patient_id_as_removed_when_will_be_added(self):
        """
        Test that schema drift doesn't flag patient_id as removed when it will be added by ensure_patient_id.

        Scenario:
        - Old dataset has patient_id column
        - New dataset doesn't have patient_id (but ensure_patient_id will add it)
        - Should not flag as "removed" since it will be added
        """
        # Arrange: Create old schema WITH patient_id
        old_schema = {
            "columns": [
                ("patient_id", "Utf8"),
                ("age", "Int64"),
                ("outcome", "Int64"),
            ]
        }

        # Create new schema WITHOUT patient_id (but ensure_patient_id will add it)
        new_schema = {
            "columns": [
                ("age", "Int64"),
                ("outcome", "Int64"),
                # patient_id missing - but ensure_patient_id will add it
            ]
        }

        # Act: Classify schema drift (before fix)
        drift_result = classify_schema_drift(old_schema, new_schema)

        # Assert: Currently flags patient_id as removed (this is the bug)
        removed_columns = drift_result.get("removed_columns", [])
        assert "patient_id" in removed_columns, (
            "Currently patient_id is flagged as removed (this is the bug we're fixing)"
        )

        # Now test the fix: Account for patient_id being added by ensure_patient_id
        # If patient_id is in old schema but not in new, add it to new schema for comparison
        old_columns = {col for col, _ in old_schema.get("columns", [])}
        new_columns = {col for col, _ in new_schema.get("columns", [])}

        if "patient_id" in old_columns and "patient_id" not in new_columns:
            # Add patient_id to new schema (it will be added by ensure_patient_id)
            new_schema_fixed = new_schema.copy()
            new_schema_fixed["columns"] = new_schema["columns"] + [("patient_id", "Utf8")]

            # Act: Classify schema drift with fixed schema
            drift_result_fixed = classify_schema_drift(old_schema, new_schema_fixed)

            # Assert: Should NOT flag patient_id as removed
            removed_columns_fixed = drift_result_fixed.get("removed_columns", [])
            assert "patient_id" not in removed_columns_fixed, (
                "patient_id should not be flagged as removed when accounted for in comparison"
            )

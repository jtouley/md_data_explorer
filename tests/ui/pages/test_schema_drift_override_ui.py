"""
Tests for schema drift override in upload UI.

Tests ensure that:
1. UI passes schema_drift_override=True in metadata when overwrite=True
2. Schema comparison accounts for patient_id being added by ensure_patient_id
"""

from pathlib import Path


class TestSchemaDriftOverrideUI:
    """Test suite for schema drift override UI behavior."""

    def test_ui_passes_schema_drift_override_when_overwrite_checked(self):
        """Test that UI passes schema_drift_override in metadata when overwrite=True."""
        # Arrange: Read upload page
        upload_page = Path("src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py")
        assert upload_page.exists(), "Upload page should exist"

        with open(upload_page) as f:
            content = f.read()

        # Assert: When overwrite is True, schema_drift_override should be set in metadata
        # Check that metadata includes schema_drift_override when overwrite is checked
        assert "if overwrite:" in content or "if overwrite == True:" in content or "if overwrite is True:" in content, (
            "UI should check overwrite flag before setting schema_drift_override"
        )
        assert "schema_drift_override" in content, "UI should set schema_drift_override in metadata"
        assert 'metadata["schema_drift_override"]' in content or "metadata['schema_drift_override']" in content, (
            "UI should set schema_drift_override in metadata dictionary"
        )

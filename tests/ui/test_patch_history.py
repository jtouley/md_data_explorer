"""
Tests for Phase 3b: Patch History Viewer for HIPAA Audit Trail.

Tests cover:
- Loading patch history from overlay store
- Formatting patches for display with timestamps
- Filtering by status (accepted/rejected/pending)
- Filtering by column
- Filtering by date range
- Export to CSV/JSON for compliance reporting
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from clinical_analytics.core.metadata_patch import (
    MetadataPatch,
    PatchOperation,
    PatchStatus,
)


class TestPatchHistoryLoader:
    """Test suite for loading patch history from overlay store."""

    def test_load_patch_history_returns_list(self, tmp_path: Path):
        """Test that load_patch_history returns a list of patch records."""
        from clinical_analytics.ui.components.patch_history import load_patch_history

        result = load_patch_history(
            overlay_store=MagicMock(load_patches=MagicMock(return_value=[])),
            upload_id="test_upload",
            version="v1",
        )

        assert isinstance(result, list)

    def test_load_patch_history_includes_all_fields(self, tmp_path: Path):
        """Test that loaded patches include all display fields."""
        from clinical_analytics.ui.components.patch_history import load_patch_history

        patch = MetadataPatch(
            patch_id="patch_001",
            operation=PatchOperation.SET_DESCRIPTION,
            column="hba1c_pct",
            value="Hemoglobin A1c percentage",
            status=PatchStatus.ACCEPTED,
            created_at=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
            provenance="llm",
            model_id="llama3.2",
            confidence=0.92,
            accepted_by="dr_smith",
            accepted_at=datetime(2025, 1, 15, 11, 0, tzinfo=UTC),
        )

        mock_store = MagicMock()
        mock_store.load_patches.return_value = [patch]

        result = load_patch_history(
            overlay_store=mock_store,
            upload_id="test_upload",
            version="v1",
        )

        assert len(result) == 1
        record = result[0]

        # Required display fields
        assert "patch_id" in record
        assert "operation" in record
        assert "column" in record
        assert "value" in record
        assert "status" in record
        assert "created_at" in record
        assert "provenance" in record

        # Optional display fields
        assert "accepted_by" in record
        assert "accepted_at" in record

    def test_load_patch_history_formats_timestamps(self, tmp_path: Path):
        """Test that timestamps are formatted as human-readable strings."""
        from clinical_analytics.ui.components.patch_history import load_patch_history

        patch = MetadataPatch(
            patch_id="patch_001",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age in years",
            status=PatchStatus.ACCEPTED,
            created_at=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
            provenance="llm",
            accepted_by="admin",
            accepted_at=datetime(2025, 1, 15, 11, 0, tzinfo=UTC),
        )

        mock_store = MagicMock()
        mock_store.load_patches.return_value = [patch]

        result = load_patch_history(
            overlay_store=mock_store,
            upload_id="test_upload",
            version="v1",
        )

        record = result[0]
        # Timestamps should be formatted strings
        assert isinstance(record["created_at"], str)
        assert "2025-01-15" in record["created_at"]


class TestPatchHistoryFiltering:
    """Test suite for filtering patch history."""

    @pytest.fixture
    def sample_patches(self) -> list[MetadataPatch]:
        """Create sample patches for filtering tests."""
        now = datetime.now(UTC)
        return [
            MetadataPatch(
                patch_id="patch_001",
                operation=PatchOperation.SET_DESCRIPTION,
                column="age",
                value="Patient age",
                status=PatchStatus.ACCEPTED,
                created_at=now - timedelta(days=5),
                provenance="llm",
                accepted_by="dr_smith",
                accepted_at=now - timedelta(days=4),
            ),
            MetadataPatch(
                patch_id="patch_002",
                operation=PatchOperation.SET_UNIT,
                column="hba1c_pct",
                value="%",
                status=PatchStatus.REJECTED,
                created_at=now - timedelta(days=3),
                provenance="llm",
                rejected_reason="Incorrect unit",
            ),
            MetadataPatch(
                patch_id="patch_003",
                operation=PatchOperation.ADD_ALIAS,
                column="age",
                value="patient_age",
                status=PatchStatus.ACCEPTED,
                created_at=now - timedelta(days=1),
                provenance="user",
                accepted_by="admin",
                accepted_at=now,
            ),
        ]

    def test_filter_by_status_accepted(self, sample_patches: list[MetadataPatch]):
        """Test filtering patches by accepted status."""
        from clinical_analytics.ui.components.patch_history import filter_patch_history

        mock_store = MagicMock()
        mock_store.load_patches.return_value = sample_patches

        result = filter_patch_history(
            patches=sample_patches,
            status_filter="accepted",
        )

        assert len(result) == 2
        assert all(p.status == PatchStatus.ACCEPTED for p in result)

    def test_filter_by_status_rejected(self, sample_patches: list[MetadataPatch]):
        """Test filtering patches by rejected status."""
        from clinical_analytics.ui.components.patch_history import filter_patch_history

        result = filter_patch_history(
            patches=sample_patches,
            status_filter="rejected",
        )

        assert len(result) == 1
        assert result[0].status == PatchStatus.REJECTED

    def test_filter_by_column(self, sample_patches: list[MetadataPatch]):
        """Test filtering patches by column name."""
        from clinical_analytics.ui.components.patch_history import filter_patch_history

        result = filter_patch_history(
            patches=sample_patches,
            column_filter="age",
        )

        assert len(result) == 2
        assert all(p.column == "age" for p in result)

    def test_filter_by_date_range(self, sample_patches: list[MetadataPatch]):
        """Test filtering patches by date range."""
        from clinical_analytics.ui.components.patch_history import filter_patch_history

        now = datetime.now(UTC)
        start_date = now - timedelta(days=4)
        end_date = now - timedelta(days=2)

        result = filter_patch_history(
            patches=sample_patches,
            start_date=start_date,
            end_date=end_date,
        )

        # Should include only patches created within the date range
        assert len(result) == 1
        assert result[0].patch_id == "patch_002"

    def test_filter_combined(self, sample_patches: list[MetadataPatch]):
        """Test combining multiple filters."""
        from clinical_analytics.ui.components.patch_history import filter_patch_history

        result = filter_patch_history(
            patches=sample_patches,
            status_filter="accepted",
            column_filter="age",
        )

        assert len(result) == 2
        assert all(p.status == PatchStatus.ACCEPTED for p in result)
        assert all(p.column == "age" for p in result)


class TestPatchHistoryDisplay:
    """Test suite for displaying patch history."""

    def test_format_patch_for_display_returns_dict(self):
        """Test that format_patch_for_display returns a dictionary."""
        from clinical_analytics.ui.components.patch_history import (
            format_patch_for_display,
        )

        patch = MetadataPatch(
            patch_id="patch_001",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age",
            status=PatchStatus.ACCEPTED,
            created_at=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
            provenance="llm",
        )

        result = format_patch_for_display(patch)

        assert isinstance(result, dict)
        assert "patch_id" in result
        assert "operation" in result
        assert "column" in result

    def test_format_patch_shows_operation_label(self):
        """Test that operation is shown as human-readable label."""
        from clinical_analytics.ui.components.patch_history import (
            format_patch_for_display,
        )

        patch = MetadataPatch(
            patch_id="patch_001",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age",
            status=PatchStatus.ACCEPTED,
            created_at=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
            provenance="llm",
        )

        result = format_patch_for_display(patch)

        # Operation should be human-readable
        assert result["operation"] == "Set Description"

    def test_format_patch_shows_status_with_color(self):
        """Test that status includes color indicator for display."""
        from clinical_analytics.ui.components.patch_history import (
            format_patch_for_display,
        )

        patch = MetadataPatch(
            patch_id="patch_001",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age",
            status=PatchStatus.ACCEPTED,
            created_at=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
            provenance="llm",
        )

        result = format_patch_for_display(patch)

        assert "status_color" in result
        assert result["status_color"] == "green"  # Accepted = green

    def test_get_unique_columns_from_patches(self):
        """Test getting unique column names for filter dropdown."""
        from clinical_analytics.ui.components.patch_history import (
            get_unique_columns,
        )

        patches = [
            MetadataPatch(
                patch_id="p1",
                operation=PatchOperation.SET_DESCRIPTION,
                column="age",
                value="v1",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="llm",
            ),
            MetadataPatch(
                patch_id="p2",
                operation=PatchOperation.SET_DESCRIPTION,
                column="hba1c",
                value="v2",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="llm",
            ),
            MetadataPatch(
                patch_id="p3",
                operation=PatchOperation.ADD_ALIAS,
                column="age",
                value="v3",
                status=PatchStatus.REJECTED,
                created_at=datetime.now(UTC),
                provenance="user",
            ),
        ]

        result = get_unique_columns(patches)

        assert isinstance(result, list)
        assert set(result) == {"age", "hba1c"}


class TestPatchHistoryExport:
    """Test suite for exporting patch history for compliance."""

    @pytest.fixture
    def sample_patches(self) -> list[MetadataPatch]:
        """Create sample patches for export tests."""
        return [
            MetadataPatch(
                patch_id="patch_001",
                operation=PatchOperation.SET_DESCRIPTION,
                column="age",
                value="Patient age",
                status=PatchStatus.ACCEPTED,
                created_at=datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
                provenance="llm",
                model_id="llama3.2",
                confidence=0.92,
                accepted_by="dr_smith",
                accepted_at=datetime(2025, 1, 15, 11, 0, tzinfo=UTC),
            ),
        ]

    def test_export_to_csv_returns_string(self, sample_patches: list[MetadataPatch]):
        """Test that export_to_csv returns CSV string."""
        from clinical_analytics.ui.components.patch_history import export_to_csv

        result = export_to_csv(sample_patches)

        assert isinstance(result, str)
        assert "patch_id" in result  # Header
        assert "patch_001" in result  # Data

    def test_export_to_csv_includes_all_audit_fields(self, sample_patches: list[MetadataPatch]):
        """Test that CSV includes all fields required for audit."""
        from clinical_analytics.ui.components.patch_history import export_to_csv

        result = export_to_csv(sample_patches)

        # Required audit fields
        assert "patch_id" in result
        assert "operation" in result
        assert "column" in result
        assert "value" in result
        assert "status" in result
        assert "created_at" in result
        assert "provenance" in result
        assert "accepted_by" in result
        assert "accepted_at" in result

    def test_export_to_json_returns_valid_json(self, sample_patches: list[MetadataPatch]):
        """Test that export_to_json returns valid JSON string."""
        import json

        from clinical_analytics.ui.components.patch_history import export_to_json

        result = export_to_json(sample_patches)

        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["patch_id"] == "patch_001"

    def test_get_export_filename_includes_timestamp(self):
        """Test that export filename includes timestamp for uniqueness."""
        from clinical_analytics.ui.components.patch_history import get_export_filename

        result = get_export_filename(
            upload_id="test_upload",
            format="csv",
        )

        assert "test_upload" in result
        assert ".csv" in result
        # Should include date/time component
        assert "202" in result  # Year prefix


class TestPatchHistoryStats:
    """Test suite for patch history statistics."""

    def test_get_patch_stats_returns_dict(self):
        """Test that get_patch_stats returns statistics dictionary."""
        from clinical_analytics.ui.components.patch_history import get_patch_stats

        patches = [
            MetadataPatch(
                patch_id="p1",
                operation=PatchOperation.SET_DESCRIPTION,
                column="age",
                value="v1",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="llm",
            ),
            MetadataPatch(
                patch_id="p2",
                operation=PatchOperation.SET_DESCRIPTION,
                column="hba1c",
                value="v2",
                status=PatchStatus.REJECTED,
                created_at=datetime.now(UTC),
                provenance="llm",
            ),
        ]

        result = get_patch_stats(patches)

        assert isinstance(result, dict)
        assert "total" in result
        assert "accepted" in result
        assert "rejected" in result
        assert result["total"] == 2
        assert result["accepted"] == 1
        assert result["rejected"] == 1

    def test_get_patch_stats_includes_by_provenance(self):
        """Test that stats include breakdown by provenance."""
        from clinical_analytics.ui.components.patch_history import get_patch_stats

        patches = [
            MetadataPatch(
                patch_id="p1",
                operation=PatchOperation.SET_DESCRIPTION,
                column="age",
                value="v1",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="llm",
            ),
            MetadataPatch(
                patch_id="p2",
                operation=PatchOperation.ADD_ALIAS,
                column="age",
                value="v2",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="user",
            ),
        ]

        result = get_patch_stats(patches)

        assert "by_provenance" in result
        assert result["by_provenance"]["llm"] == 1
        assert result["by_provenance"]["user"] == 1

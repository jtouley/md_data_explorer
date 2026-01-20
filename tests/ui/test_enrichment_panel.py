"""
Tests for Enrichment Panel UI Component.

Phase 3a: ADR011 Metadata Enrichment
Tests for the diff view panel showing pending enrichment suggestions.
"""

from datetime import UTC, datetime

import pytest


@pytest.fixture
def sample_pending_patches():
    """Create sample pending patches for testing."""
    from clinical_analytics.core.metadata_patch import (
        MetadataPatch,
        PatchOperation,
        PatchStatus,
    )

    return [
        MetadataPatch(
            patch_id="patch-1",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age in years at enrollment",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
            model_id="llama3.1:8b",
            confidence=0.9,
        ),
        MetadataPatch(
            patch_id="patch-2",
            operation=PatchOperation.SET_SEMANTIC_TYPE,
            column="mortality",
            value="outcome",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
            model_id="llama3.1:8b",
            confidence=0.85,
        ),
        MetadataPatch(
            patch_id="patch-3",
            operation=PatchOperation.SET_CODEBOOK_ENTRY,
            column="status",
            value={"code": "1", "label": "Active"},
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
            model_id="llama3.1:8b",
            confidence=0.75,
        ),
    ]


@pytest.fixture
def sample_resolved_metadata():
    """Create sample resolved metadata for diff comparison."""
    from clinical_analytics.core.metadata_patch import (
        ResolvedColumnMetadata,
        ResolvedDatasetMetadata,
    )

    return ResolvedDatasetMetadata(
        columns={
            "age": ResolvedColumnMetadata(
                name="age",
                description=None,  # No description yet
            ),
            "mortality": ResolvedColumnMetadata(
                name="mortality",
                description="Death indicator",
            ),
            "status": ResolvedColumnMetadata(
                name="status",
                codebook=None,
            ),
        }
    )


class TestEnrichmentPanelData:
    """Tests for enrichment panel data preparation."""

    def test_prepare_diff_view_data_returns_list(self, sample_pending_patches, sample_resolved_metadata):
        """Test that prepare_diff_view_data returns list of diff items."""
        from clinical_analytics.ui.components.enrichment_panel import (
            prepare_diff_view_data,
        )

        diff_data = prepare_diff_view_data(
            pending=sample_pending_patches,
            resolved=sample_resolved_metadata,
        )

        assert isinstance(diff_data, list)
        assert len(diff_data) == 3

    def test_prepare_diff_view_data_includes_before_after(self, sample_pending_patches, sample_resolved_metadata):
        """Test that diff data includes before/after values."""
        from clinical_analytics.ui.components.enrichment_panel import (
            prepare_diff_view_data,
        )

        diff_data = prepare_diff_view_data(
            pending=sample_pending_patches,
            resolved=sample_resolved_metadata,
        )

        # Find the age patch
        age_diff = next(d for d in diff_data if d["column"] == "age")

        assert "before" in age_diff
        assert "after" in age_diff
        assert age_diff["before"] is None  # No previous description
        assert age_diff["after"] == "Patient age in years at enrollment"

    def test_prepare_diff_view_data_includes_confidence(self, sample_pending_patches, sample_resolved_metadata):
        """Test that diff data includes confidence scores."""
        from clinical_analytics.ui.components.enrichment_panel import (
            prepare_diff_view_data,
        )

        diff_data = prepare_diff_view_data(
            pending=sample_pending_patches,
            resolved=sample_resolved_metadata,
        )

        age_diff = next(d for d in diff_data if d["column"] == "age")
        assert age_diff["confidence"] == 0.9

    def test_prepare_diff_view_data_includes_operation(self, sample_pending_patches, sample_resolved_metadata):
        """Test that diff data includes operation type."""
        from clinical_analytics.ui.components.enrichment_panel import (
            prepare_diff_view_data,
        )

        diff_data = prepare_diff_view_data(
            pending=sample_pending_patches,
            resolved=sample_resolved_metadata,
        )

        age_diff = next(d for d in diff_data if d["column"] == "age")
        assert age_diff["operation"] == "set_description"


class TestEnrichmentPanelRender:
    """Tests for enrichment panel rendering."""

    def test_render_diff_item_returns_html(self, sample_pending_patches):
        """Test that render_diff_item returns HTML string."""
        from clinical_analytics.ui.components.enrichment_panel import render_diff_item

        diff_item = {
            "patch_id": "patch-1",
            "column": "age",
            "operation": "set_description",
            "before": None,
            "after": "Patient age in years",
            "confidence": 0.9,
        }

        html = render_diff_item(diff_item)

        assert isinstance(html, str)
        assert "age" in html
        assert "Patient age in years" in html

    def test_render_confidence_indicator_high(self):
        """Test confidence indicator for high confidence (>0.8)."""
        from clinical_analytics.ui.components.enrichment_panel import (
            render_confidence_indicator,
        )

        html = render_confidence_indicator(0.95)

        assert "green" in html.lower() or "high" in html.lower() or "✓" in html

    def test_render_confidence_indicator_medium(self):
        """Test confidence indicator for medium confidence (0.6-0.8)."""
        from clinical_analytics.ui.components.enrichment_panel import (
            render_confidence_indicator,
        )

        html = render_confidence_indicator(0.7)

        assert "yellow" in html.lower() or "medium" in html.lower() or "orange" in html.lower()

    def test_render_confidence_indicator_low(self):
        """Test confidence indicator for low confidence (<0.6)."""
        from clinical_analytics.ui.components.enrichment_panel import (
            render_confidence_indicator,
        )

        html = render_confidence_indicator(0.4)

        assert "red" in html.lower() or "low" in html.lower() or "⚠" in html


class TestEnrichmentPanelActions:
    """Tests for accept/reject action handling."""

    def test_get_accept_reject_buttons_returns_tuple(self, sample_pending_patches):
        """Test that get_accept_reject_buttons returns button config."""
        from clinical_analytics.ui.components.enrichment_panel import (
            get_accept_reject_buttons,
        )

        buttons = get_accept_reject_buttons("patch-1")

        assert isinstance(buttons, dict)
        assert "accept_key" in buttons
        assert "reject_key" in buttons

    def test_handle_accept_calls_overlay_store(self, sample_pending_patches, tmp_path):
        """Test that accepting a patch updates overlay store."""
        from clinical_analytics.core.overlay_store import OverlayStore
        from clinical_analytics.ui.components.enrichment_panel import handle_accept

        store = OverlayStore(base_dir=tmp_path)
        store.save_pending("upload-1", "v1", sample_pending_patches)

        handle_accept(
            store=store,
            upload_id="upload-1",
            version="v1",
            patch_id="patch-1",
            accepted_by="test_user",
        )

        # Verify patch was moved from pending to patches log
        patches = store.load_patches("upload-1", "v1")
        assert len(patches) == 1
        assert patches[0].patch_id == "patch-1"

    def test_handle_reject_calls_overlay_store(self, sample_pending_patches, tmp_path):
        """Test that rejecting a patch updates overlay store."""
        from clinical_analytics.core.metadata_patch import PatchStatus
        from clinical_analytics.core.overlay_store import OverlayStore
        from clinical_analytics.ui.components.enrichment_panel import handle_reject

        store = OverlayStore(base_dir=tmp_path)
        store.save_pending("upload-1", "v1", sample_pending_patches)

        handle_reject(
            store=store,
            upload_id="upload-1",
            version="v1",
            patch_id="patch-1",
            reason="Not accurate",
        )

        # Verify patch was recorded as rejected
        patches = store.load_patches("upload-1", "v1")
        assert len(patches) == 1
        assert patches[0].status == PatchStatus.REJECTED


class TestBatchOperations:
    """Tests for batch accept/reject operations."""

    def test_handle_batch_accept(self, sample_pending_patches, tmp_path):
        """Test accepting multiple patches at once."""
        from clinical_analytics.core.overlay_store import OverlayStore
        from clinical_analytics.ui.components.enrichment_panel import (
            handle_batch_accept,
        )

        store = OverlayStore(base_dir=tmp_path)
        store.save_pending("upload-1", "v1", sample_pending_patches)

        handle_batch_accept(
            store=store,
            upload_id="upload-1",
            version="v1",
            patch_ids=["patch-1", "patch-2"],
            accepted_by="test_user",
        )

        patches = store.load_patches("upload-1", "v1")
        assert len(patches) == 2

    def test_handle_batch_reject(self, sample_pending_patches, tmp_path):
        """Test rejecting multiple patches at once."""
        from clinical_analytics.core.overlay_store import OverlayStore
        from clinical_analytics.ui.components.enrichment_panel import (
            handle_batch_reject,
        )

        store = OverlayStore(base_dir=tmp_path)
        store.save_pending("upload-1", "v1", sample_pending_patches)

        handle_batch_reject(
            store=store,
            upload_id="upload-1",
            version="v1",
            patch_ids=["patch-1", "patch-2"],
            reason="Bulk rejection",
        )

        patches = store.load_patches("upload-1", "v1")
        assert len(patches) == 2
        for p in patches:
            assert p.rejected_reason == "Bulk rejection"

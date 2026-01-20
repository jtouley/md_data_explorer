"""
Tests for Phase 5: Enrichment UI Integration.

Tests cover:
- EnrichmentService for coordinating enrichment workflow
- Session state management for pending suggestions
- Integration with overlay store and LLM enrichment
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from clinical_analytics.core.metadata_patch import (
    MetadataPatch,
    PatchOperation,
    PatchStatus,
)


class TestEnrichmentService:
    """Tests for EnrichmentService integration layer."""

    @pytest.fixture
    def overlay_store(self, tmp_path: Path):
        """Create overlay store with temp directory."""
        from clinical_analytics.core.overlay_store import OverlayStore

        return OverlayStore(base_dir=tmp_path)

    @pytest.fixture
    def sample_schema(self):
        """Create sample inferred schema."""
        from clinical_analytics.core.schema_inference import InferredSchema

        return InferredSchema(
            patient_id_column="patient_id",
            outcome_columns=["mortality"],
            categorical_columns=["sex", "treatment"],
            continuous_columns=["age", "bmi"],
        )

    def test_enrichment_service_exists(self):
        """Test that EnrichmentService class exists."""
        from clinical_analytics.ui.components.enrichment_integration import (
            EnrichmentService,
        )

        assert EnrichmentService is not None

    def test_enrichment_service_init_with_overlay_store(self, overlay_store):
        """Test EnrichmentService initialization with overlay store."""
        from clinical_analytics.ui.components.enrichment_integration import (
            EnrichmentService,
        )

        service = EnrichmentService(overlay_store=overlay_store)

        assert service.overlay_store == overlay_store

    def test_has_pending_suggestions_false_initially(self, overlay_store):
        """Test that has_pending_suggestions returns False when no suggestions."""
        from clinical_analytics.ui.components.enrichment_integration import (
            EnrichmentService,
        )

        service = EnrichmentService(overlay_store=overlay_store)

        result = service.has_pending_suggestions(upload_id="test_upload", version="v1")

        assert result is False

    def test_has_pending_suggestions_true_after_generation(self, overlay_store, sample_schema):
        """Test that has_pending_suggestions returns True after generation."""
        from clinical_analytics.ui.components.enrichment_integration import (
            EnrichmentService,
        )

        service = EnrichmentService(overlay_store=overlay_store)

        # Mock LLM to return suggestions
        mock_patch = MetadataPatch(
            patch_id="p1",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age in years",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )

        with patch(
            "clinical_analytics.ui.components.enrichment_integration.generate_enrichment_suggestions"
        ) as mock_gen:
            mock_gen.return_value = [mock_patch]

            service.trigger_enrichment(
                upload_id="test_upload",
                version="v1",
                schema=sample_schema,
            )

        result = service.has_pending_suggestions(upload_id="test_upload", version="v1")

        assert result is True

    def test_trigger_enrichment_calls_llm(self, overlay_store, sample_schema):
        """Test that trigger_enrichment calls LLM generation."""
        from clinical_analytics.ui.components.enrichment_integration import (
            EnrichmentService,
        )

        service = EnrichmentService(overlay_store=overlay_store)

        with patch(
            "clinical_analytics.ui.components.enrichment_integration.generate_enrichment_suggestions"
        ) as mock_gen:
            mock_gen.return_value = []

            service.trigger_enrichment(
                upload_id="test_upload",
                version="v1",
                schema=sample_schema,
            )

            mock_gen.assert_called_once()

    def test_trigger_enrichment_saves_to_pending(self, overlay_store, sample_schema):
        """Test that trigger_enrichment saves suggestions to pending."""
        from clinical_analytics.ui.components.enrichment_integration import (
            EnrichmentService,
        )

        service = EnrichmentService(overlay_store=overlay_store)

        mock_patch = MetadataPatch(
            patch_id="p1",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age in years",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )

        with patch(
            "clinical_analytics.ui.components.enrichment_integration.generate_enrichment_suggestions"
        ) as mock_gen:
            mock_gen.return_value = [mock_patch]

            service.trigger_enrichment(
                upload_id="test_upload",
                version="v1",
                schema=sample_schema,
            )

        # Check pending was saved
        pending = overlay_store.load_pending("test_upload", "v1")
        assert len(pending) == 1
        assert pending[0].column == "age"

    def test_get_pending_suggestions_returns_list(self, overlay_store):
        """Test that get_pending_suggestions returns list of patches."""
        from clinical_analytics.ui.components.enrichment_integration import (
            EnrichmentService,
        )

        service = EnrichmentService(overlay_store=overlay_store)

        result = service.get_pending_suggestions(upload_id="test_upload", version="v1")

        assert isinstance(result, list)

    def test_accept_suggestion_moves_to_accepted(self, overlay_store, sample_schema):
        """Test that accepting a suggestion moves it from pending to accepted."""
        from clinical_analytics.ui.components.enrichment_integration import (
            EnrichmentService,
        )

        service = EnrichmentService(overlay_store=overlay_store)

        mock_patch = MetadataPatch(
            patch_id="p1",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age in years",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )

        # Save pending directly
        overlay_store.save_pending("test_upload", "v1", [mock_patch])

        # Accept
        service.accept_suggestion(
            upload_id="test_upload",
            version="v1",
            patch_id="p1",
            accepted_by="test_user",
        )

        # Check pending is empty
        pending = overlay_store.load_pending("test_upload", "v1")
        assert len(pending) == 0

        # Check patch is in log
        patches = overlay_store.load_patches("test_upload", "v1")
        assert len(patches) == 1
        assert patches[0].status == PatchStatus.ACCEPTED

    def test_reject_suggestion_removes_from_pending(self, overlay_store, sample_schema):
        """Test that rejecting a suggestion removes it from pending."""
        from clinical_analytics.ui.components.enrichment_integration import (
            EnrichmentService,
        )

        service = EnrichmentService(overlay_store=overlay_store)

        mock_patch = MetadataPatch(
            patch_id="p1",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age in years",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )

        # Save pending directly
        overlay_store.save_pending("test_upload", "v1", [mock_patch])

        # Reject
        service.reject_suggestion(
            upload_id="test_upload",
            version="v1",
            patch_id="p1",
            reason="Not accurate",
        )

        # Check pending is empty
        pending = overlay_store.load_pending("test_upload", "v1")
        assert len(pending) == 0

        # Check rejection is logged
        patches = overlay_store.load_patches("test_upload", "v1")
        assert len(patches) == 1
        assert patches[0].status == PatchStatus.REJECTED

    def test_get_enrichment_stats_returns_dict(self, overlay_store):
        """Test that get_enrichment_stats returns statistics."""
        from clinical_analytics.ui.components.enrichment_integration import (
            EnrichmentService,
        )

        service = EnrichmentService(overlay_store=overlay_store)

        result = service.get_enrichment_stats(upload_id="test_upload", version="v1")

        assert isinstance(result, dict)
        assert "pending_count" in result
        assert "accepted_count" in result
        assert "rejected_count" in result


class TestEnrichmentUIHelpers:
    """Tests for UI helper functions."""

    def test_format_enrichment_button_text(self):
        """Test button text formatting based on state."""
        from clinical_analytics.ui.components.enrichment_integration import (
            format_enrichment_button_text,
        )

        # No pending
        result = format_enrichment_button_text(pending_count=0, is_generating=False)
        assert "Enrich" in result

        # With pending
        result = format_enrichment_button_text(pending_count=5, is_generating=False)
        assert "5" in result

        # Generating
        result = format_enrichment_button_text(pending_count=0, is_generating=True)
        assert "Generating" in result or "..." in result

    def test_get_confidence_badge_color(self):
        """Test confidence badge color mapping."""
        from clinical_analytics.ui.components.enrichment_integration import (
            get_confidence_badge_color,
        )

        assert get_confidence_badge_color(0.9) == "green"
        assert get_confidence_badge_color(0.7) == "orange"
        assert get_confidence_badge_color(0.4) == "red"

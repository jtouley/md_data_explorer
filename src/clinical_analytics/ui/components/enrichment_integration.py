"""
Enrichment Integration Service for ADR011 Phase 5.

This module provides the integration layer between the Streamlit UI pages
and the ADR011 enrichment infrastructure. It coordinates:
- LLM enrichment generation
- Pending suggestion management
- Accept/reject workflows
- Session state integration

Usage in Streamlit pages:
    from clinical_analytics.ui.components.enrichment_integration import (
        EnrichmentService,
    )

    service = EnrichmentService(overlay_store=overlay_store)
    service.trigger_enrichment(upload_id, version, schema)
"""

from typing import Any

import structlog

from clinical_analytics.core.llm_enrichment import generate_enrichment_suggestions
from clinical_analytics.core.metadata_patch import (
    ExclusionPatternPatch,
    MetadataPatch,
    PatchStatus,
    RelationshipPatch,
)
from clinical_analytics.core.overlay_store import OverlayStore
from clinical_analytics.core.schema_inference import InferredSchema

logger = structlog.get_logger(__name__)

PatchType = MetadataPatch | ExclusionPatternPatch | RelationshipPatch


class EnrichmentService:
    """
    Service layer for coordinating metadata enrichment workflow.

    Provides a clean interface for Streamlit pages to interact with
    the enrichment infrastructure without tight coupling.
    """

    def __init__(self, overlay_store: OverlayStore):
        """
        Initialize enrichment service.

        Args:
            overlay_store: OverlayStore instance for persistence
        """
        self.overlay_store = overlay_store

    def has_pending_suggestions(self, upload_id: str, version: str) -> bool:
        """
        Check if there are pending suggestions for this dataset.

        Args:
            upload_id: Upload identifier
            version: Dataset version

        Returns:
            True if pending suggestions exist
        """
        pending = self.overlay_store.load_pending(upload_id, version)
        return len(pending) > 0

    def get_pending_suggestions(self, upload_id: str, version: str) -> list[PatchType]:
        """
        Get pending suggestions for this dataset.

        Args:
            upload_id: Upload identifier
            version: Dataset version

        Returns:
            List of pending patches
        """
        return self.overlay_store.load_pending(upload_id, version)

    def trigger_enrichment(
        self,
        upload_id: str,
        version: str,
        schema: InferredSchema,
        doc_context: str | None = None,
        model_id: str | None = None,
    ) -> list[PatchType]:
        """
        Trigger LLM enrichment for a dataset.

        Args:
            upload_id: Upload identifier
            version: Dataset version
            schema: Inferred schema for the dataset
            doc_context: Optional documentation context
            model_id: Optional LLM model ID

        Returns:
            List of generated suggestions
        """
        logger.info(
            "enrichment_triggered",
            upload_id=upload_id,
            version=version,
        )

        suggestions = generate_enrichment_suggestions(
            schema=schema,
            doc_context=doc_context,
            model_id=model_id,
        )

        if suggestions:
            self.overlay_store.save_pending(upload_id, version, suggestions)
            logger.info(
                "enrichment_suggestions_saved",
                upload_id=upload_id,
                version=version,
                count=len(suggestions),
            )

        return suggestions

    def accept_suggestion(
        self,
        upload_id: str,
        version: str,
        patch_id: str,
        accepted_by: str,
    ) -> None:
        """
        Accept a pending suggestion.

        Args:
            upload_id: Upload identifier
            version: Dataset version
            patch_id: ID of patch to accept
            accepted_by: User accepting the patch
        """
        self.overlay_store.accept_patch(upload_id, version, patch_id, accepted_by)
        logger.info(
            "suggestion_accepted",
            upload_id=upload_id,
            patch_id=patch_id,
            accepted_by=accepted_by,
        )

    def reject_suggestion(
        self,
        upload_id: str,
        version: str,
        patch_id: str,
        reason: str,
    ) -> None:
        """
        Reject a pending suggestion.

        Args:
            upload_id: Upload identifier
            version: Dataset version
            patch_id: ID of patch to reject
            reason: Rejection reason
        """
        self.overlay_store.reject_patch(upload_id, version, patch_id, reason)
        logger.info(
            "suggestion_rejected",
            upload_id=upload_id,
            patch_id=patch_id,
            reason=reason,
        )

    def accept_all_suggestions(
        self,
        upload_id: str,
        version: str,
        accepted_by: str,
    ) -> int:
        """
        Accept all pending suggestions.

        Args:
            upload_id: Upload identifier
            version: Dataset version
            accepted_by: User accepting the patches

        Returns:
            Number of patches accepted
        """
        pending = self.overlay_store.load_pending(upload_id, version)
        count = 0

        for patch in pending:
            self.overlay_store.accept_patch(upload_id, version, patch.patch_id, accepted_by)
            count += 1

        logger.info(
            "all_suggestions_accepted",
            upload_id=upload_id,
            count=count,
        )

        return count

    def reject_all_suggestions(
        self,
        upload_id: str,
        version: str,
        reason: str,
    ) -> int:
        """
        Reject all pending suggestions.

        Args:
            upload_id: Upload identifier
            version: Dataset version
            reason: Rejection reason

        Returns:
            Number of patches rejected
        """
        pending = self.overlay_store.load_pending(upload_id, version)
        count = 0

        for patch in pending:
            self.overlay_store.reject_patch(upload_id, version, patch.patch_id, reason)
            count += 1

        logger.info(
            "all_suggestions_rejected",
            upload_id=upload_id,
            count=count,
        )

        return count

    def get_enrichment_stats(self, upload_id: str, version: str) -> dict[str, Any]:
        """
        Get enrichment statistics for a dataset.

        Args:
            upload_id: Upload identifier
            version: Dataset version

        Returns:
            Dictionary with stats (pending_count, accepted_count, rejected_count)
        """
        pending = self.overlay_store.load_pending(upload_id, version)
        patches = self.overlay_store.load_patches(upload_id, version)

        accepted = sum(1 for p in patches if p.status == PatchStatus.ACCEPTED)
        rejected = sum(1 for p in patches if p.status == PatchStatus.REJECTED)

        return {
            "pending_count": len(pending),
            "accepted_count": accepted,
            "rejected_count": rejected,
            "total_patches": len(patches),
        }


def format_enrichment_button_text(
    pending_count: int,
    is_generating: bool,
) -> str:
    """
    Format button text based on enrichment state.

    Args:
        pending_count: Number of pending suggestions
        is_generating: Whether LLM is currently generating

    Returns:
        Button text string
    """
    if is_generating:
        return "Generating suggestions..."

    if pending_count > 0:
        return f"Review {pending_count} suggestions"

    return "Enrich Metadata with AI"


def get_confidence_badge_color(confidence: float | None) -> str:
    """
    Get badge color for confidence score.

    Args:
        confidence: Confidence score (0-1) or None

    Returns:
        Color string (green, orange, red)
    """
    if confidence is None:
        return "gray"

    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"

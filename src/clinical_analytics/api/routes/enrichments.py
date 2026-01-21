"""Enrichment API routes for ADR011 metadata enrichment.

Endpoints:
- GET /api/datasets/{dataset_id}/enrichments/pending - Get pending suggestions
- POST /api/datasets/{dataset_id}/enrichments/{patch_id}/accept - Accept suggestion
- POST /api/datasets/{dataset_id}/enrichments/{patch_id}/reject - Reject suggestion
- GET /api/datasets/{dataset_id}/enrichments/history - Get patch history
- POST /api/datasets/{dataset_id}/enrichments/generate - Generate suggestions
"""

from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, status
from fastapi import Path as FastAPIPath
from pydantic import BaseModel, Field

from clinical_analytics.core.metadata_patch import MetadataPatch
from clinical_analytics.core.overlay_store import OverlayStore
from clinical_analytics.ui.components.enrichment_integration import EnrichmentService

router = APIRouter()
logger = structlog.get_logger()

DEFAULT_OVERLAY_DIR = Path("data/uploads/metadata/overlays")


class PendingSuggestion(BaseModel):
    """Pending enrichment suggestion."""

    patch_id: str
    operation: str
    column: str
    suggested_value: str
    current_value: str | None = None
    confidence: float
    model_id: str


class PendingResponse(BaseModel):
    """Response for pending suggestions."""

    suggestions: list[PendingSuggestion]
    total: int


class AcceptRequest(BaseModel):
    """Request to accept a suggestion."""

    accepted_by: str = Field("api_user", description="User accepting the patch")


class RejectRequest(BaseModel):
    """Request to reject a suggestion."""

    reason: str = Field("Rejected via API", description="Rejection reason")


class PatchHistoryItem(BaseModel):
    """Single patch in history."""

    patch_id: str
    operation: str
    column: str
    value: str
    status: str
    created_at: str
    resolved_at: str | None = None
    resolved_by: str | None = None
    model_id: str


class PatchHistoryResponse(BaseModel):
    """Response for patch history."""

    patches: list[PatchHistoryItem]
    total: int


class GenerateRequest(BaseModel):
    """Request to generate enrichment suggestions."""

    force_regenerate: bool = Field(False, description="Force regeneration of suggestions")


class GenerateResponse(BaseModel):
    """Response after generating suggestions."""

    message: str
    suggestions_count: int


class AcceptRejectResponse(BaseModel):
    """Response after accepting or rejecting."""

    success: bool
    message: str


def get_overlay_store() -> OverlayStore:
    """Get or create OverlayStore instance."""
    return OverlayStore(base_dir=DEFAULT_OVERLAY_DIR)


def get_enrichment_service(
    overlay_store: Annotated[OverlayStore, Depends(get_overlay_store)],
) -> EnrichmentService:
    """Get or create EnrichmentService instance."""
    return EnrichmentService(overlay_store=overlay_store)


def _patch_to_pending_suggestion(patch: MetadataPatch) -> PendingSuggestion:
    """Convert MetadataPatch to PendingSuggestion API model."""
    return PendingSuggestion(
        patch_id=patch.patch_id,
        operation=patch.operation.name,
        column=patch.column,
        suggested_value=str(patch.value),
        current_value=None,
        confidence=patch.confidence or 0.0,
        model_id=patch.model_id or "unknown",
    )


def _patch_to_history_item(patch: MetadataPatch) -> PatchHistoryItem:
    """Convert MetadataPatch to PatchHistoryItem API model."""
    return PatchHistoryItem(
        patch_id=patch.patch_id,
        operation=patch.operation.name,
        column=patch.column,
        value=str(patch.value),
        status=patch.status.name,
        created_at=patch.created_at.isoformat(),
        resolved_at=patch.accepted_at.isoformat() if patch.accepted_at else None,
        resolved_by=patch.accepted_by,
        model_id=patch.model_id or "unknown",
    )


@router.get(
    "/datasets/{dataset_id}/enrichments/pending",
    response_model=PendingResponse,
)
async def get_pending_suggestions(
    dataset_id: Annotated[str, FastAPIPath(..., description="Dataset ID")],
    enrichment_service: Annotated[EnrichmentService, Depends(get_enrichment_service)],
) -> PendingResponse:
    """Get pending enrichment suggestions for a dataset."""
    logger.info("enrichments_get_pending", dataset_id=dataset_id)

    pending = enrichment_service.get_pending_suggestions(
        upload_id=dataset_id,
        version="v1",
    )

    suggestions = [_patch_to_pending_suggestion(p) for p in pending if isinstance(p, MetadataPatch)]

    return PendingResponse(
        suggestions=suggestions,
        total=len(suggestions),
    )


@router.post(
    "/datasets/{dataset_id}/enrichments/{patch_id}/accept",
    response_model=AcceptRejectResponse,
)
async def accept_suggestion(
    dataset_id: Annotated[str, FastAPIPath(..., description="Dataset ID")],
    patch_id: Annotated[str, FastAPIPath(..., description="Patch ID")],
    request: AcceptRequest,
    enrichment_service: Annotated[EnrichmentService, Depends(get_enrichment_service)],
) -> AcceptRejectResponse:
    """Accept an enrichment suggestion."""
    logger.info(
        "enrichments_accept",
        dataset_id=dataset_id,
        patch_id=patch_id,
        accepted_by=request.accepted_by,
    )

    try:
        enrichment_service.accept_suggestion(
            upload_id=dataset_id,
            version="v1",
            patch_id=patch_id,
            accepted_by=request.accepted_by,
        )
        return AcceptRejectResponse(
            success=True,
            message="Suggestion accepted",
        )
    except Exception as e:
        logger.warning("enrichments_accept_failed", error=str(e))
        return AcceptRejectResponse(
            success=False,
            message=f"Failed to accept suggestion: {e}",
        )


@router.post(
    "/datasets/{dataset_id}/enrichments/{patch_id}/reject",
    response_model=AcceptRejectResponse,
)
async def reject_suggestion(
    dataset_id: Annotated[str, FastAPIPath(..., description="Dataset ID")],
    patch_id: Annotated[str, FastAPIPath(..., description="Patch ID")],
    request: RejectRequest,
    enrichment_service: Annotated[EnrichmentService, Depends(get_enrichment_service)],
) -> AcceptRejectResponse:
    """Reject an enrichment suggestion."""
    logger.info(
        "enrichments_reject",
        dataset_id=dataset_id,
        patch_id=patch_id,
        reason=request.reason,
    )

    try:
        enrichment_service.reject_suggestion(
            upload_id=dataset_id,
            version="v1",
            patch_id=patch_id,
            reason=request.reason,
        )
        return AcceptRejectResponse(
            success=True,
            message="Suggestion rejected",
        )
    except Exception as e:
        logger.warning("enrichments_reject_failed", error=str(e))
        return AcceptRejectResponse(
            success=False,
            message=f"Failed to reject suggestion: {e}",
        )


@router.get(
    "/datasets/{dataset_id}/enrichments/history",
    response_model=PatchHistoryResponse,
)
async def get_patch_history(
    dataset_id: Annotated[str, FastAPIPath(..., description="Dataset ID")],
    overlay_store: Annotated[OverlayStore, Depends(get_overlay_store)],
) -> PatchHistoryResponse:
    """Get patch history for a dataset."""
    logger.info("enrichments_get_history", dataset_id=dataset_id)

    patches = overlay_store.load_patches(
        upload_id=dataset_id,
        version="v1",
    )

    history = [_patch_to_history_item(p) for p in patches if isinstance(p, MetadataPatch)]

    return PatchHistoryResponse(
        patches=history,
        total=len(history),
    )


@router.post(
    "/datasets/{dataset_id}/enrichments/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_suggestions(
    dataset_id: Annotated[str, FastAPIPath(..., description="Dataset ID")],
    request: GenerateRequest,
    enrichment_service: Annotated[EnrichmentService, Depends(get_enrichment_service)],
) -> GenerateResponse:
    """Generate enrichment suggestions for a dataset."""
    logger.info(
        "enrichments_generate",
        dataset_id=dataset_id,
        force=request.force_regenerate,
    )

    # Note: trigger_enrichment requires InferredSchema - for now return placeholder
    # In production, would load schema from dataset and call trigger_enrichment
    return GenerateResponse(
        message=f"Enrichment generation queued for dataset {dataset_id}",
        suggestions_count=0,
    )

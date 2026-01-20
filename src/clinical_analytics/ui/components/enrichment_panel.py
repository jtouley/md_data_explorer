"""
Enrichment Panel UI Component for ADR011.

This module provides the diff view panel for reviewing and accepting/rejecting
LLM-generated metadata enrichment suggestions.

Key Features:
- Side-by-side diff view (current vs suggested)
- Confidence indicators per suggestion
- Batch accept/reject controls
- Edit before accept capability
"""

from typing import Any

import structlog

from clinical_analytics.core.metadata_patch import (
    ExclusionPatternPatch,
    MetadataPatch,
    PatchOperation,
    RelationshipPatch,
    ResolvedDatasetMetadata,
)
from clinical_analytics.core.overlay_store import OverlayStore

logger = structlog.get_logger(__name__)


def prepare_diff_view_data(
    pending: list[MetadataPatch | ExclusionPatternPatch | RelationshipPatch],
    resolved: ResolvedDatasetMetadata,
) -> list[dict[str, Any]]:
    """
    Prepare diff view data for rendering.

    Args:
        pending: List of pending patches
        resolved: Current resolved metadata

    Returns:
        List of diff items with before/after values
    """
    diff_items = []

    for patch in pending:
        if isinstance(patch, RelationshipPatch):
            # Relationship patches don't have before/after for a single column
            diff_items.append(
                {
                    "patch_id": patch.patch_id,
                    "column": ", ".join(patch.columns),
                    "operation": PatchOperation.SET_RELATIONSHIP.value,
                    "before": None,
                    "after": patch.rule,
                    "confidence": patch.confidence,
                    "provenance": patch.provenance,
                    "model_id": getattr(patch, "model_id", None),
                }
            )
        elif isinstance(patch, ExclusionPatternPatch):
            diff_items.append(
                {
                    "patch_id": patch.patch_id,
                    "column": patch.column,
                    "operation": PatchOperation.SET_EXCLUSION_PATTERN.value,
                    "before": None,
                    "after": f"{patch.pattern} → exclude {patch.coded_value}",
                    "confidence": patch.confidence,
                    "provenance": patch.provenance,
                    "model_id": patch.model_id,
                }
            )
        elif isinstance(patch, MetadataPatch):
            # Get current value from resolved metadata
            before = _get_current_value(resolved, patch.column, patch.operation)
            after = _format_patch_value(patch.value, patch.operation)

            diff_items.append(
                {
                    "patch_id": patch.patch_id,
                    "column": patch.column,
                    "operation": patch.operation.value,
                    "before": before,
                    "after": after,
                    "confidence": patch.confidence,
                    "provenance": patch.provenance,
                    "model_id": patch.model_id,
                }
            )

    return diff_items


def _get_current_value(
    resolved: ResolvedDatasetMetadata,
    column: str,
    operation: PatchOperation,
) -> Any:
    """Get current value for a column/operation from resolved metadata."""
    if column not in resolved.columns:
        return None

    col_meta = resolved.columns[column]

    if operation == PatchOperation.SET_DESCRIPTION:
        return col_meta.description
    elif operation == PatchOperation.SET_LABEL:
        return col_meta.label
    elif operation == PatchOperation.SET_SEMANTIC_TYPE:
        return col_meta.semantic_type.value if col_meta.semantic_type else None
    elif operation == PatchOperation.SET_UNIT:
        return col_meta.unit
    elif operation == PatchOperation.MARK_PHI:
        return col_meta.is_phi
    elif operation == PatchOperation.SET_CODEBOOK_ENTRY:
        return col_meta.codebook
    elif operation == PatchOperation.ADD_ALIAS:
        return col_meta.aliases

    return None


def _format_patch_value(value: Any, operation: PatchOperation) -> str:
    """Format patch value for display."""
    if value is None:
        return ""

    if operation == PatchOperation.SET_CODEBOOK_ENTRY:
        if isinstance(value, dict):
            code = value.get("code", "")
            label = value.get("label", "")
            return f"{code}: {label}"

    return str(value)


def render_diff_item(diff_item: dict[str, Any]) -> str:
    """
    Render a single diff item as HTML.

    Args:
        diff_item: Dict with patch_id, column, operation, before, after, confidence

    Returns:
        HTML string for rendering
    """
    column = diff_item["column"]
    operation = diff_item["operation"]
    before = diff_item.get("before") or "(none)"
    after = diff_item.get("after") or "(none)"
    confidence = diff_item.get("confidence", 0.0)

    confidence_html = render_confidence_indicator(confidence)

    html = f"""
    <div class="diff-item" data-patch-id="{diff_item['patch_id']}">
        <div class="diff-header">
            <span class="column-name">{column}</span>
            <span class="operation">{operation}</span>
            {confidence_html}
        </div>
        <div class="diff-content">
            <div class="before">
                <span class="label">Current:</span>
                <span class="value">{before}</span>
            </div>
            <div class="after">
                <span class="label">Suggested:</span>
                <span class="value">{after}</span>
            </div>
        </div>
    </div>
    """

    return html


def render_confidence_indicator(confidence: float | None) -> str:
    """
    Render confidence indicator HTML.

    Args:
        confidence: Confidence score (0-1)

    Returns:
        HTML string with colored indicator
    """
    if confidence is None:
        return '<span class="confidence">?</span>'

    if confidence >= 0.8:
        color = "green"
        icon = "✓"
        level = "high"
    elif confidence >= 0.6:
        color = "orange"
        icon = "~"
        level = "medium"
    else:
        color = "red"
        icon = "⚠"
        level = "low"

    return f'<span class="confidence {level}" style="color: {color}">{icon} {confidence:.0%}</span>'


def get_accept_reject_buttons(patch_id: str) -> dict[str, str]:
    """
    Get button configuration for a patch.

    Args:
        patch_id: Patch identifier

    Returns:
        Dict with button keys
    """
    return {
        "accept_key": f"accept_{patch_id}",
        "reject_key": f"reject_{patch_id}",
        "patch_id": patch_id,
    }


def handle_accept(
    store: OverlayStore,
    upload_id: str,
    version: str,
    patch_id: str,
    accepted_by: str,
) -> None:
    """
    Handle accepting a pending patch.

    Args:
        store: OverlayStore instance
        upload_id: Upload identifier
        version: Dataset version
        patch_id: Patch to accept
        accepted_by: User accepting the patch
    """
    store.accept_patch(upload_id, version, patch_id, accepted_by)
    logger.info(
        "ui_patch_accepted",
        upload_id=upload_id,
        version=version,
        patch_id=patch_id,
        accepted_by=accepted_by,
    )


def handle_reject(
    store: OverlayStore,
    upload_id: str,
    version: str,
    patch_id: str,
    reason: str,
) -> None:
    """
    Handle rejecting a pending patch.

    Args:
        store: OverlayStore instance
        upload_id: Upload identifier
        version: Dataset version
        patch_id: Patch to reject
        reason: Rejection reason
    """
    store.reject_patch(upload_id, version, patch_id, reason)
    logger.info(
        "ui_patch_rejected",
        upload_id=upload_id,
        version=version,
        patch_id=patch_id,
        reason=reason,
    )


def handle_batch_accept(
    store: OverlayStore,
    upload_id: str,
    version: str,
    patch_ids: list[str],
    accepted_by: str,
) -> None:
    """
    Handle accepting multiple patches at once.

    Args:
        store: OverlayStore instance
        upload_id: Upload identifier
        version: Dataset version
        patch_ids: List of patch IDs to accept
        accepted_by: User accepting the patches
    """
    for patch_id in patch_ids:
        store.accept_patch(upload_id, version, patch_id, accepted_by)

    logger.info(
        "ui_batch_accept",
        upload_id=upload_id,
        version=version,
        count=len(patch_ids),
        accepted_by=accepted_by,
    )


def handle_batch_reject(
    store: OverlayStore,
    upload_id: str,
    version: str,
    patch_ids: list[str],
    reason: str,
) -> None:
    """
    Handle rejecting multiple patches at once.

    Args:
        store: OverlayStore instance
        upload_id: Upload identifier
        version: Dataset version
        patch_ids: List of patch IDs to reject
        reason: Rejection reason
    """
    for patch_id in patch_ids:
        store.reject_patch(upload_id, version, patch_id, reason)

    logger.info(
        "ui_batch_reject",
        upload_id=upload_id,
        version=version,
        count=len(patch_ids),
        reason=reason,
    )

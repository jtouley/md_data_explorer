"""
Patch History Viewer for ADR011 - HIPAA Audit Trail.

This module provides UI components for viewing, filtering, and exporting
the patch history for compliance and audit purposes.

Key Functions:
- load_patch_history: Load patches from overlay store
- filter_patch_history: Filter patches by status, column, date
- export_to_csv/json: Export for compliance reporting

Design Principles:
- Complete audit trail visibility
- Human-readable timestamps
- Export in standard formats for HIPAA compliance
- Statistics for quick overview
"""

import csv
import io
import json
from datetime import datetime
from typing import Any, Literal

import structlog

from clinical_analytics.core.metadata_patch import (
    ExclusionPatternPatch,
    MetadataPatch,
    PatchOperation,
    PatchStatus,
    RelationshipPatch,
)
from clinical_analytics.core.overlay_store import OverlayStore

logger = structlog.get_logger(__name__)

# Type alias for patch union
PatchType = MetadataPatch | ExclusionPatternPatch | RelationshipPatch


# =============================================================================
# Operation Label Mapping
# =============================================================================

OPERATION_LABELS: dict[PatchOperation, str] = {
    PatchOperation.SET_LABEL: "Set Label",
    PatchOperation.ADD_ALIAS: "Add Alias",
    PatchOperation.SET_DESCRIPTION: "Set Description",
    PatchOperation.SET_SEMANTIC_TYPE: "Set Semantic Type",
    PatchOperation.MARK_PHI: "Mark PHI",
    PatchOperation.SET_UNIT: "Set Unit",
    PatchOperation.SET_CODEBOOK_ENTRY: "Set Codebook Entry",
    PatchOperation.SET_RELATIONSHIP: "Set Relationship",
    PatchOperation.SET_EXCLUSION_PATTERN: "Set Exclusion Pattern",
}

STATUS_COLORS: dict[PatchStatus, str] = {
    PatchStatus.PENDING: "yellow",
    PatchStatus.ACCEPTED: "green",
    PatchStatus.REJECTED: "red",
}


# =============================================================================
# Patch History Loading
# =============================================================================


def load_patch_history(
    overlay_store: OverlayStore,
    upload_id: str,
    version: str,
) -> list[dict[str, Any]]:
    """
    Load patch history from overlay store and format for display.

    Args:
        overlay_store: OverlayStore instance
        upload_id: Upload identifier
        version: Dataset version

    Returns:
        List of formatted patch records for display
    """
    patches = overlay_store.load_patches(upload_id, version)

    return [format_patch_for_display(patch) for patch in patches]


def format_patch_for_display(patch: PatchType) -> dict[str, Any]:
    """
    Format a patch for display in the UI.

    Args:
        patch: Patch to format

    Returns:
        Dictionary with all display fields
    """
    # Get operation label
    if isinstance(patch, MetadataPatch):
        operation = patch.operation
        operation_label = OPERATION_LABELS.get(operation, operation.value)
        column = patch.column
        value = patch.value
        model_id = patch.model_id
        confidence = patch.confidence
        accepted_by = patch.accepted_by
        accepted_at = patch.accepted_at
        rejected_reason = patch.rejected_reason
    elif isinstance(patch, ExclusionPatternPatch):
        operation = PatchOperation.SET_EXCLUSION_PATTERN
        operation_label = OPERATION_LABELS.get(operation, operation.value)
        column = patch.column
        value = f"{patch.pattern} → {patch.coded_value}"
        model_id = patch.model_id
        confidence = patch.confidence
        accepted_by = None
        accepted_at = None
        rejected_reason = None
    elif isinstance(patch, RelationshipPatch):
        operation = PatchOperation.SET_RELATIONSHIP
        operation_label = OPERATION_LABELS.get(operation, operation.value)
        column = ", ".join(patch.columns)
        value = patch.rule
        model_id = patch.model_id
        confidence = patch.confidence
        accepted_by = None
        accepted_at = None
        rejected_reason = None
    else:
        operation_label = "Unknown"
        column = "unknown"
        value = str(patch)
        model_id = None
        confidence = None
        accepted_by = None
        accepted_at = None
        rejected_reason = None

    # Format timestamps
    created_at_str = _format_timestamp(patch.created_at)
    accepted_at_str = _format_timestamp(accepted_at) if accepted_at else None

    return {
        "patch_id": patch.patch_id,
        "operation": operation_label,
        "column": column,
        "value": value,
        "status": patch.status.value,
        "status_color": STATUS_COLORS.get(patch.status, "gray"),
        "created_at": created_at_str,
        "provenance": patch.provenance,
        "model_id": model_id,
        "confidence": confidence,
        "accepted_by": accepted_by,
        "accepted_at": accepted_at_str,
        "rejected_reason": rejected_reason,
    }


def _format_timestamp(dt: datetime | None) -> str | None:
    """Format datetime as human-readable string."""
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


# =============================================================================
# Patch History Filtering
# =============================================================================


def filter_patch_history(
    patches: list[PatchType],
    status_filter: str | None = None,
    column_filter: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[PatchType]:
    """
    Filter patches by various criteria.

    Args:
        patches: List of patches to filter
        status_filter: Filter by status (accepted, rejected, pending)
        column_filter: Filter by column name
        start_date: Filter by created_at >= start_date
        end_date: Filter by created_at <= end_date

    Returns:
        Filtered list of patches
    """
    result = list(patches)

    # Filter by status
    if status_filter:
        status_enum = PatchStatus(status_filter)
        result = [p for p in result if p.status == status_enum]

    # Filter by column
    if column_filter:
        filtered: list[PatchType] = []
        for p in result:
            if isinstance(p, MetadataPatch | ExclusionPatternPatch):
                if p.column == column_filter:
                    filtered.append(p)
            elif isinstance(p, RelationshipPatch):
                if column_filter in p.columns:
                    filtered.append(p)
        result = filtered

    # Filter by date range
    if start_date:
        result = [p for p in result if p.created_at >= start_date]

    if end_date:
        result = [p for p in result if p.created_at <= end_date]

    return result


def get_unique_columns(patches: list[PatchType]) -> list[str]:
    """
    Get unique column names from patches for filter dropdown.

    Args:
        patches: List of patches

    Returns:
        Sorted list of unique column names
    """
    columns = set()

    for patch in patches:
        if isinstance(patch, MetadataPatch | ExclusionPatternPatch):
            columns.add(patch.column)
        elif isinstance(patch, RelationshipPatch):
            columns.update(patch.columns)

    return sorted(columns)


# =============================================================================
# Patch History Export
# =============================================================================


def export_to_csv(patches: list[PatchType]) -> str:
    """
    Export patches to CSV format for compliance reporting.

    Args:
        patches: List of patches to export

    Returns:
        CSV string with all audit fields
    """
    output = io.StringIO()

    # Define CSV fields (audit-complete)
    fieldnames = [
        "patch_id",
        "operation",
        "column",
        "value",
        "status",
        "created_at",
        "provenance",
        "model_id",
        "confidence",
        "accepted_by",
        "accepted_at",
        "rejected_reason",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for patch in patches:
        record = format_patch_for_display(patch)
        # Filter to only include CSV fields
        row = {k: record.get(k, "") for k in fieldnames}
        writer.writerow(row)

    return output.getvalue()


def export_to_json(patches: list[PatchType]) -> str:
    """
    Export patches to JSON format for compliance reporting.

    Args:
        patches: List of patches to export

    Returns:
        JSON string with all audit fields
    """
    records = [format_patch_for_display(patch) for patch in patches]
    return json.dumps(records, indent=2)


def get_export_filename(
    upload_id: str,
    format: Literal["csv", "json"] = "csv",
) -> str:
    """
    Generate export filename with timestamp.

    Args:
        upload_id: Upload identifier
        format: Export format (csv or json)

    Returns:
        Filename with timestamp for uniqueness
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"patch_history_{upload_id}_{timestamp}.{format}"


# =============================================================================
# Patch History Statistics
# =============================================================================


def get_patch_stats(patches: list[PatchType]) -> dict[str, Any]:
    """
    Calculate statistics for patch history.

    Args:
        patches: List of patches

    Returns:
        Dictionary with statistics
    """
    total = len(patches)
    accepted = sum(1 for p in patches if p.status == PatchStatus.ACCEPTED)
    rejected = sum(1 for p in patches if p.status == PatchStatus.REJECTED)
    pending = sum(1 for p in patches if p.status == PatchStatus.PENDING)

    # Count by provenance
    by_provenance: dict[str, int] = {}
    for patch in patches:
        prov = patch.provenance
        by_provenance[prov] = by_provenance.get(prov, 0) + 1

    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "pending": pending,
        "by_provenance": by_provenance,
    }

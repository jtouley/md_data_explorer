"""
Overlay Store for ADR011 - LLM-Assisted Metadata Enrichment.

This module provides JSONL-based storage for metadata patches with append-only
semantics for audit trail integrity.

Storage Structure:
    data/uploads/metadata/overlays/{upload_id}/{dataset_version}/
    ├── patches.jsonl      # Append-only patch log
    ├── pending.json       # Current pending LLM suggestions
    └── resolved.json      # Cached resolved metadata

Key Design Decisions:
- JSONL for patch log (append-only, human-readable, git-friendly)
- JSON for pending/resolved (single document, overwritable)
- Directory per upload/version for isolation
- No auto-apply - all patches require explicit acceptance
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from clinical_analytics.core.metadata_patch import (
    ExclusionPatternPatch,
    MetadataPatch,
    PatchOperation,
    PatchStatus,
    RelationshipPatch,
    ResolvedColumnMetadata,
    ResolvedDatasetMetadata,
    ResolvedExclusionPattern,
    ResolvedRelationship,
    SemanticType,
)

logger = structlog.get_logger(__name__)


class OverlayStore:
    """
    JSONL-based storage for metadata patches.

    Provides append-only patch log with full audit trail,
    pending suggestion storage, and resolved metadata caching.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize overlay store.

        Args:
            base_dir: Base directory for overlay storage
        """
        self.base_dir = Path(base_dir)

    def get_overlay_path(self, upload_id: str, version: str) -> Path:
        """Get path to overlay directory for upload/version."""
        return self.base_dir / "overlays" / upload_id / version

    def overlay_exists(self, upload_id: str, version: str) -> bool:
        """Check if overlay directory exists."""
        return self.get_overlay_path(upload_id, version).exists()

    def _ensure_overlay_dir(self, upload_id: str, version: str) -> Path:
        """Ensure overlay directory exists and return path."""
        overlay_path = self.get_overlay_path(upload_id, version)
        overlay_path.mkdir(parents=True, exist_ok=True)
        return overlay_path

    # =========================================================================
    # Patch Log Operations (append-only JSONL)
    # =========================================================================

    def append_patch(
        self,
        upload_id: str,
        version: str,
        patch: MetadataPatch | ExclusionPatternPatch | RelationshipPatch,
    ) -> None:
        """
        Append a patch to the JSONL log.

        Args:
            upload_id: Upload identifier
            version: Dataset version
            patch: Patch to append
        """
        overlay_path = self._ensure_overlay_dir(upload_id, version)
        jsonl_path = overlay_path / "patches.jsonl"

        patch_dict = patch.to_dict()
        # Add patch type for deserialization
        patch_dict["_patch_type"] = type(patch).__name__

        with open(jsonl_path, "a") as f:
            f.write(json.dumps(patch_dict) + "\n")

        logger.info(
            "patch_appended",
            upload_id=upload_id,
            version=version,
            patch_id=patch.patch_id,
        )

    def load_patches(
        self,
        upload_id: str,
        version: str,
    ) -> list[MetadataPatch | ExclusionPatternPatch | RelationshipPatch]:
        """
        Load all patches from JSONL log.

        Args:
            upload_id: Upload identifier
            version: Dataset version

        Returns:
            List of patches in chronological order
        """
        overlay_path = self.get_overlay_path(upload_id, version)
        jsonl_path = overlay_path / "patches.jsonl"

        if not jsonl_path.exists():
            return []

        patches: list[MetadataPatch | ExclusionPatternPatch | RelationshipPatch] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                patch_dict = json.loads(line)
                patch = self._deserialize_patch(patch_dict)
                if patch:
                    patches.append(patch)

        return patches

    def _deserialize_patch(
        self, data: dict[str, Any]
    ) -> MetadataPatch | ExclusionPatternPatch | RelationshipPatch | None:
        """Deserialize patch from dictionary."""
        patch_type = data.pop("_patch_type", None)

        if patch_type == "ExclusionPatternPatch":
            return ExclusionPatternPatch.from_dict(data)
        elif patch_type == "RelationshipPatch":
            return RelationshipPatch.from_dict(data)
        elif patch_type == "MetadataPatch":
            return MetadataPatch.from_dict(data)
        else:
            # Try to infer from operation
            op = data.get("operation")
            if op == PatchOperation.SET_EXCLUSION_PATTERN.value:
                return ExclusionPatternPatch.from_dict(data)
            elif op == PatchOperation.SET_RELATIONSHIP.value:
                return RelationshipPatch.from_dict(data)
            else:
                return MetadataPatch.from_dict(data)

    # =========================================================================
    # Pending Suggestions Operations
    # =========================================================================

    def save_pending(
        self,
        upload_id: str,
        version: str,
        suggestions: list[MetadataPatch | ExclusionPatternPatch | RelationshipPatch],
    ) -> None:
        """
        Save pending LLM suggestions (overwrites existing).

        Args:
            upload_id: Upload identifier
            version: Dataset version
            suggestions: List of pending patches
        """
        overlay_path = self._ensure_overlay_dir(upload_id, version)
        pending_path = overlay_path / "pending.json"

        data = []
        for patch in suggestions:
            patch_dict = patch.to_dict()
            patch_dict["_patch_type"] = type(patch).__name__
            data.append(patch_dict)

        with open(pending_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "pending_saved",
            upload_id=upload_id,
            version=version,
            count=len(suggestions),
        )

    def load_pending(
        self,
        upload_id: str,
        version: str,
    ) -> list[MetadataPatch | ExclusionPatternPatch | RelationshipPatch]:
        """
        Load pending LLM suggestions.

        Args:
            upload_id: Upload identifier
            version: Dataset version

        Returns:
            List of pending patches
        """
        overlay_path = self.get_overlay_path(upload_id, version)
        pending_path = overlay_path / "pending.json"

        if not pending_path.exists():
            return []

        with open(pending_path) as f:
            data = json.load(f)

        patches: list[MetadataPatch | ExclusionPatternPatch | RelationshipPatch] = []
        for patch_dict in data:
            patch = self._deserialize_patch(patch_dict)
            if patch:
                patches.append(patch)

        return patches

    def clear_pending(self, upload_id: str, version: str) -> None:
        """
        Clear pending suggestions.

        Args:
            upload_id: Upload identifier
            version: Dataset version
        """
        overlay_path = self.get_overlay_path(upload_id, version)
        pending_path = overlay_path / "pending.json"

        if pending_path.exists():
            pending_path.unlink()
            logger.info("pending_cleared", upload_id=upload_id, version=version)

    # =========================================================================
    # Accept/Reject Patch Operations
    # =========================================================================

    def accept_patch(
        self,
        upload_id: str,
        version: str,
        patch_id: str,
        accepted_by: str,
    ) -> None:
        """
        Accept a pending patch and add to patch log.

        Args:
            upload_id: Upload identifier
            version: Dataset version
            patch_id: ID of patch to accept
            accepted_by: User who accepted
        """
        pending = self.load_pending(upload_id, version)

        for patch in pending:
            if patch.patch_id == patch_id:
                # Create accepted version of patch
                accepted_patch = self._create_accepted_patch(patch, accepted_by)
                self.append_patch(upload_id, version, accepted_patch)

                # Remove from pending
                remaining = [p for p in pending if p.patch_id != patch_id]
                if remaining:
                    self.save_pending(upload_id, version, remaining)
                else:
                    self.clear_pending(upload_id, version)

                # Invalidate cache
                self.invalidate_cache(upload_id, version)

                logger.info(
                    "patch_accepted",
                    upload_id=upload_id,
                    version=version,
                    patch_id=patch_id,
                    accepted_by=accepted_by,
                )
                return

        logger.warning(
            "patch_not_found_in_pending",
            upload_id=upload_id,
            version=version,
            patch_id=patch_id,
        )

    def reject_patch(
        self,
        upload_id: str,
        version: str,
        patch_id: str,
        reason: str,
    ) -> None:
        """
        Reject a pending patch and record rejection in patch log.

        Args:
            upload_id: Upload identifier
            version: Dataset version
            patch_id: ID of patch to reject
            reason: Rejection reason
        """
        pending = self.load_pending(upload_id, version)

        for patch in pending:
            if patch.patch_id == patch_id:
                # Create rejected version of patch
                rejected_patch = self._create_rejected_patch(patch, reason)
                self.append_patch(upload_id, version, rejected_patch)

                # Remove from pending
                remaining = [p for p in pending if p.patch_id != patch_id]
                if remaining:
                    self.save_pending(upload_id, version, remaining)
                else:
                    self.clear_pending(upload_id, version)

                logger.info(
                    "patch_rejected",
                    upload_id=upload_id,
                    version=version,
                    patch_id=patch_id,
                    reason=reason,
                )
                return

        logger.warning(
            "patch_not_found_in_pending",
            upload_id=upload_id,
            version=version,
            patch_id=patch_id,
        )

    def _create_accepted_patch(
        self,
        patch: MetadataPatch | ExclusionPatternPatch | RelationshipPatch,
        accepted_by: str,
    ) -> MetadataPatch | ExclusionPatternPatch | RelationshipPatch:
        """Create accepted version of patch."""
        now = datetime.now(UTC)

        if isinstance(patch, MetadataPatch):
            return MetadataPatch(
                patch_id=patch.patch_id,
                operation=patch.operation,
                column=patch.column,
                value=patch.value,
                status=PatchStatus.ACCEPTED,
                created_at=patch.created_at,
                provenance=patch.provenance,
                model_id=patch.model_id,
                confidence=patch.confidence,
                accepted_by=accepted_by,
                accepted_at=now,
            )
        elif isinstance(patch, ExclusionPatternPatch):
            return ExclusionPatternPatch(
                patch_id=patch.patch_id,
                column=patch.column,
                pattern=patch.pattern,
                coded_value=patch.coded_value,
                context=patch.context,
                auto_apply=patch.auto_apply,
                status=PatchStatus.ACCEPTED,
                created_at=patch.created_at,
                provenance=patch.provenance,
                model_id=patch.model_id,
                confidence=patch.confidence,
            )
        elif isinstance(patch, RelationshipPatch):
            return RelationshipPatch(
                patch_id=patch.patch_id,
                columns=patch.columns,
                relationship_type=patch.relationship_type,
                rule=patch.rule,
                inference=patch.inference,
                confidence=patch.confidence,
                status=PatchStatus.ACCEPTED,
                created_at=patch.created_at,
                provenance=patch.provenance,
                model_id=patch.model_id,
            )
        return patch

    def _create_rejected_patch(
        self,
        patch: MetadataPatch | ExclusionPatternPatch | RelationshipPatch,
        reason: str,
    ) -> MetadataPatch | ExclusionPatternPatch | RelationshipPatch:
        """Create rejected version of patch."""
        if isinstance(patch, MetadataPatch):
            return MetadataPatch(
                patch_id=patch.patch_id,
                operation=patch.operation,
                column=patch.column,
                value=patch.value,
                status=PatchStatus.REJECTED,
                created_at=patch.created_at,
                provenance=patch.provenance,
                model_id=patch.model_id,
                confidence=patch.confidence,
                rejected_reason=reason,
            )
        elif isinstance(patch, ExclusionPatternPatch):
            return ExclusionPatternPatch(
                patch_id=patch.patch_id,
                column=patch.column,
                pattern=patch.pattern,
                coded_value=patch.coded_value,
                context=patch.context,
                auto_apply=patch.auto_apply,
                status=PatchStatus.REJECTED,
                created_at=patch.created_at,
                provenance=patch.provenance,
                model_id=patch.model_id,
                confidence=patch.confidence,
            )
        elif isinstance(patch, RelationshipPatch):
            return RelationshipPatch(
                patch_id=patch.patch_id,
                columns=patch.columns,
                relationship_type=patch.relationship_type,
                rule=patch.rule,
                inference=patch.inference,
                confidence=patch.confidence,
                status=PatchStatus.REJECTED,
                created_at=patch.created_at,
                provenance=patch.provenance,
                model_id=patch.model_id,
            )
        return patch

    # =========================================================================
    # Resolved Metadata Cache Operations
    # =========================================================================

    def cache_resolved(
        self,
        upload_id: str,
        version: str,
        resolved: ResolvedDatasetMetadata,
    ) -> None:
        """
        Cache resolved metadata.

        Args:
            upload_id: Upload identifier
            version: Dataset version
            resolved: Resolved metadata to cache
        """
        overlay_path = self._ensure_overlay_dir(upload_id, version)
        cache_path = overlay_path / "resolved.json"

        data = self._serialize_resolved(resolved)

        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "resolved_cached",
            upload_id=upload_id,
            version=version,
            column_count=len(resolved.columns),
        )

    def load_cached_resolved(
        self,
        upload_id: str,
        version: str,
    ) -> ResolvedDatasetMetadata | None:
        """
        Load cached resolved metadata.

        Args:
            upload_id: Upload identifier
            version: Dataset version

        Returns:
            Cached resolved metadata or None if not cached
        """
        overlay_path = self.get_overlay_path(upload_id, version)
        cache_path = overlay_path / "resolved.json"

        if not cache_path.exists():
            return None

        with open(cache_path) as f:
            data = json.load(f)

        return self._deserialize_resolved(data)

    def invalidate_cache(self, upload_id: str, version: str) -> None:
        """
        Invalidate resolved metadata cache.

        Args:
            upload_id: Upload identifier
            version: Dataset version
        """
        overlay_path = self.get_overlay_path(upload_id, version)
        cache_path = overlay_path / "resolved.json"

        if cache_path.exists():
            cache_path.unlink()
            logger.info("cache_invalidated", upload_id=upload_id, version=version)

    def _serialize_resolved(self, resolved: ResolvedDatasetMetadata) -> dict[str, Any]:
        """Serialize resolved metadata to dictionary."""
        columns_data = {}
        for name, col in resolved.columns.items():
            col_data: dict[str, Any] = {"name": col.name}
            if col.label:
                col_data["label"] = col.label
            if col.description:
                col_data["description"] = col.description
            if col.semantic_type:
                col_data["semantic_type"] = col.semantic_type.value
            if col.unit:
                col_data["unit"] = col.unit
            if col.is_phi:
                col_data["is_phi"] = col.is_phi
            if col.aliases:
                col_data["aliases"] = col.aliases
            if col.codebook:
                col_data["codebook"] = col.codebook
            if col.exclusion_patterns:
                col_data["exclusion_patterns"] = [
                    {
                        "pattern": ep.pattern,
                        "coded_value": ep.coded_value,
                        "context": ep.context,
                        "auto_apply": ep.auto_apply,
                    }
                    for ep in col.exclusion_patterns
                ]
            columns_data[name] = col_data

        result: dict[str, Any] = {"columns": columns_data}

        if resolved.relationships:
            result["relationships"] = [
                {
                    "columns": r.columns,
                    "relationship_type": r.relationship_type,
                    "rule": r.rule,
                    "inference": r.inference,
                    "confidence": r.confidence,
                }
                for r in resolved.relationships
            ]

        return result

    def _deserialize_resolved(self, data: dict[str, Any]) -> ResolvedDatasetMetadata:
        """Deserialize resolved metadata from dictionary."""
        columns = {}
        for name, col_data in data.get("columns", {}).items():
            exclusion_patterns = None
            if "exclusion_patterns" in col_data:
                exclusion_patterns = [
                    ResolvedExclusionPattern(
                        pattern=ep["pattern"],
                        coded_value=ep["coded_value"],
                        context=ep["context"],
                        auto_apply=ep["auto_apply"],
                    )
                    for ep in col_data["exclusion_patterns"]
                ]

            semantic_type = None
            if "semantic_type" in col_data:
                semantic_type = SemanticType(col_data["semantic_type"])

            columns[name] = ResolvedColumnMetadata(
                name=col_data["name"],
                label=col_data.get("label"),
                description=col_data.get("description"),
                semantic_type=semantic_type,
                unit=col_data.get("unit"),
                is_phi=col_data.get("is_phi", False),
                aliases=col_data.get("aliases", []),
                codebook=col_data.get("codebook"),
                exclusion_patterns=exclusion_patterns,
            )

        relationships = None
        if "relationships" in data:
            relationships = [
                ResolvedRelationship(
                    columns=r["columns"],
                    relationship_type=r["relationship_type"],
                    rule=r["rule"],
                    inference=r["inference"],
                    confidence=r["confidence"],
                )
                for r in data["relationships"]
            ]

        return ResolvedDatasetMetadata(columns=columns, relationships=relationships)

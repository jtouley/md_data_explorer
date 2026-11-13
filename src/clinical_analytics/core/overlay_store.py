# SPDX-License-Identifier: MIT
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
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

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


class PatchNotFoundError(ValueError):
    """No patch with the requested patch_id exists in the patch log."""


class PatchStateError(ValueError):
    """The patch exists but its latest status does not allow the requested change."""


AnyPatch: TypeAlias = MetadataPatch | ExclusionPatternPatch | RelationshipPatch

# One lock per overlay directory, shared by every OverlayStore instance in the process (the API
# builds a store per request and runs sync routes in a threadpool). Decisions on one upload/version
# are serialized; the API is a single local process, so this does not guard against other processes.
_DECISION_LOCKS: dict[Path, threading.Lock] = {}
_DECISION_LOCKS_GUARD = threading.Lock()


def _decision_lock(overlay_path: Path) -> threading.Lock:
    with _DECISION_LOCKS_GUARD:
        return _DECISION_LOCKS.setdefault(overlay_path.resolve(), threading.Lock())


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
        """Deserialize patch from dictionary without mutating input."""
        patch_type = data.get("_patch_type")
        # Create a copy without _patch_type for deserialization
        patch_data = {k: v for k, v in data.items() if k != "_patch_type"}

        if patch_type == "ExclusionPatternPatch":
            return ExclusionPatternPatch.from_dict(patch_data)
        elif patch_type == "RelationshipPatch":
            return RelationshipPatch.from_dict(patch_data)
        elif patch_type == "MetadataPatch":
            return MetadataPatch.from_dict(patch_data)
        else:
            # Try to infer from operation
            op = patch_data.get("operation")
            if op == PatchOperation.SET_EXCLUSION_PATTERN.value:
                return ExclusionPatternPatch.from_dict(patch_data)
            elif op == PatchOperation.SET_RELATIONSHIP.value:
                return RelationshipPatch.from_dict(patch_data)
            else:
                return MetadataPatch.from_dict(patch_data)

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

        # Write-then-rename so a failed write leaves the previous pending.json intact. A unique temp
        # name per call keeps two writers from sharing (and tearing) one temp file.
        fd, tmp_name = tempfile.mkstemp(prefix="pending.", suffix=".tmp", dir=overlay_path)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, pending_path)
        finally:
            # Gone after a successful replace; otherwise (OSError, or a value json can't encode)
            # remove it so failures don't pile up temp files in the overlay directory.
            tmp_path.unlink(missing_ok=True)

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

        Raises:
            PatchNotFoundError: If patch_id is neither pending nor in the patch log.
            PatchStateError: If patch_id was already resolved (accepted, rejected or reverted).
        """
        self._decide_pending(
            upload_id, version, patch_id, lambda patch: self._create_accepted_patch(patch, accepted_by)
        )
        logger.info(
            "patch_accepted",
            upload_id=upload_id,
            version=version,
            patch_id=patch_id,
            accepted_by=accepted_by,
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

        Raises:
            PatchNotFoundError: If patch_id is neither pending nor in the patch log.
            PatchStateError: If patch_id was already resolved (accepted, rejected or reverted).
        """
        self._decide_pending(upload_id, version, patch_id, lambda patch: self._create_rejected_patch(patch, reason))
        logger.info(
            "patch_rejected",
            upload_id=upload_id,
            version=version,
            patch_id=patch_id,
            reason=reason,
        )

    @contextmanager
    def _locked(self, upload_id: str, version: str) -> Iterator[None]:
        """Serialize accept/reject/revert for one upload/version."""
        with _decision_lock(self.get_overlay_path(upload_id, version)):
            yield

    def _latest_by_id(self, upload_id: str, version: str) -> dict[str, AnyPatch]:
        latest: dict[str, AnyPatch] = {}
        for p in self.load_patches(upload_id, version):
            latest[p.patch_id] = p
        return latest

    def _decide_pending(
        self,
        upload_id: str,
        version: str,
        patch_id: str,
        make_row: Callable[[AnyPatch], AnyPatch],
    ) -> None:
        """Record one decision on a pending patch: log row appended and pending entry removed, or neither.

        Raises:
            PatchNotFoundError: patch_id is neither pending nor logged.
            PatchStateError: patch_id already has a decision in the log.
            OSError: a write failed; the log and pending.json are left as they were.
        """
        with self._locked(upload_id, version):
            decided = self._latest_by_id(upload_id, version).get(patch_id)
            pending = self.load_pending(upload_id, version)
            if decided is not None:
                if any(p.patch_id == patch_id for p in pending):
                    # Left pending by a write that failed before rollback existed; drop it so the
                    # UI stops offering a decision the log already holds.
                    self.save_pending(upload_id, version, [p for p in pending if p.patch_id != patch_id])
                raise PatchStateError(f"Patch {patch_id!r} is not pending (latest status={decided.status.name})")
            patch = next((p for p in pending if p.patch_id == patch_id), None)
            if patch is None:
                raise PatchNotFoundError(f"No pending patch found for patch_id={patch_id!r}")

            # resolved.json is derived from the log; drop it before writing so a failure here
            # leaves nothing decided.
            self.invalidate_cache(upload_id, version)
            remaining = [p for p in pending if p.patch_id != patch_id]

            def drop_from_pending() -> None:
                if remaining:
                    self.save_pending(upload_id, version, remaining)
                else:
                    self.clear_pending(upload_id, version)

            self._append_with_rollback(upload_id, version, make_row(patch), then=drop_from_pending)

    def _append_with_rollback(
        self,
        upload_id: str,
        version: str,
        row: AnyPatch,
        then: Callable[[], None],
    ) -> None:
        """Append row, run then(); if either raises OSError, truncate the log back and re-raise."""
        jsonl_path = self._ensure_overlay_dir(upload_id, version) / "patches.jsonl"
        size_before = jsonl_path.stat().st_size if jsonl_path.exists() else 0
        try:
            self.append_patch(upload_id, version, row)
            then()
        except OSError:
            try:
                with open(jsonl_path, "r+b") as f:
                    f.truncate(size_before)
            except OSError as rollback_error:
                logger.error(
                    "patch_log_rollback_failed",
                    upload_id=upload_id,
                    version=version,
                    patch_id=row.patch_id,
                    error_type=type(rollback_error).__name__,
                )
            raise

    def revert_accepted_patch(
        self,
        upload_id: str,
        version: str,
        patch_id: str,
        reverted_by: str,
    ) -> None:
        """
        Append a REVERTED row for an accepted patch (same patch_id, append-only log).

        Raises:
            PatchNotFoundError: If patch_id is not in the patch log.
            PatchStateError: If the latest row for patch_id is not ACCEPTED.
        """
        with self._locked(upload_id, version):
            current = self._latest_by_id(upload_id, version).get(patch_id)
            if current is None:
                raise PatchNotFoundError(f"No patch found for patch_id={patch_id!r}")
            if current.status != PatchStatus.ACCEPTED:
                raise PatchStateError(f"Patch {patch_id!r} is not accepted (latest status={current.status.name})")

            self.invalidate_cache(upload_id, version)
            self._append_with_rollback(
                upload_id, version, self._create_reverted_patch(current, reverted_by), then=lambda: None
            )

        logger.info(
            "patch_reverted",
            upload_id=upload_id,
            version=version,
            patch_id=patch_id,
            reverted_by=reverted_by,
        )

    def _create_reverted_patch(
        self,
        patch: MetadataPatch | ExclusionPatternPatch | RelationshipPatch,
        reverted_by: str,
    ) -> MetadataPatch | ExclusionPatternPatch | RelationshipPatch:
        """Create REVERTED version of an accepted patch (append-only audit)."""
        now = datetime.now(UTC)

        if isinstance(patch, MetadataPatch):
            return MetadataPatch(
                patch_id=patch.patch_id,
                operation=patch.operation,
                column=patch.column,
                value=patch.value,
                status=PatchStatus.REVERTED,
                created_at=patch.created_at,
                provenance=patch.provenance,
                model_id=patch.model_id,
                confidence=patch.confidence,
                accepted_by=patch.accepted_by,
                accepted_at=patch.accepted_at,
                rejected_reason=None,
                reverted_by=reverted_by,
                reverted_at=now,
            )
        if isinstance(patch, ExclusionPatternPatch):
            return ExclusionPatternPatch(
                patch_id=patch.patch_id,
                column=patch.column,
                pattern=patch.pattern,
                coded_value=patch.coded_value,
                context=patch.context,
                auto_apply=patch.auto_apply,
                status=PatchStatus.REVERTED,
                created_at=patch.created_at,
                provenance=patch.provenance,
                model_id=patch.model_id,
                confidence=patch.confidence,
            )
        if isinstance(patch, RelationshipPatch):
            return RelationshipPatch(
                patch_id=patch.patch_id,
                columns=patch.columns,
                relationship_type=patch.relationship_type,
                rule=patch.rule,
                inference=patch.inference,
                confidence=patch.confidence,
                status=PatchStatus.REVERTED,
                created_at=patch.created_at,
                provenance=patch.provenance,
                model_id=patch.model_id,
            )
        return patch

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

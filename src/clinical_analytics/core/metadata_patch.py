"""
Metadata Patch Model for ADR011 - LLM-Assisted Metadata Enrichment.

This module provides frozen dataclasses for metadata patches with provenance tracking.
Patches represent atomic metadata operations that can be applied to column metadata.

Key Design Decisions:
- Frozen dataclasses ensure immutability for audit integrity
- Provenance fields track origin (llm, user, dictionary)
- Status tracks lifecycle (pending, accepted, rejected)
- Serialization methods for JSONL storage

Usage:
    from clinical_analytics.core.metadata_patch import (
        MetadataPatch,
        PatchOperation,
        PatchStatus,
    )

    patch = MetadataPatch(
        patch_id=str(uuid4()),
        operation=PatchOperation.SET_DESCRIPTION,
        column="hba1c_pct",
        value="Hemoglobin A1c percentage",
        status=PatchStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        provenance="llm",
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class PatchOperation(Enum):
    """Operations that can be performed on column metadata."""

    SET_LABEL = "set_label"
    ADD_ALIAS = "add_alias"
    SET_DESCRIPTION = "set_description"
    SET_SEMANTIC_TYPE = "set_semantic_type"
    MARK_PHI = "mark_phi"
    SET_UNIT = "set_unit"
    SET_CODEBOOK_ENTRY = "set_codebook_entry"
    SET_RELATIONSHIP = "set_relationship"
    SET_EXCLUSION_PATTERN = "set_exclusion_pattern"


class SemanticType(Enum):
    """Semantic types for clinical data columns."""

    IDENTIFIER = "identifier"
    DEMOGRAPHIC = "demographic"
    CLINICAL = "clinical"
    TEMPORAL = "temporal"
    OUTCOME = "outcome"
    MEASUREMENT = "measurement"
    CODED = "coded"


class PatchStatus(Enum):
    """Lifecycle status of a metadata patch."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class MetadataPatch:
    """
    Immutable metadata patch with provenance tracking.

    Represents a single atomic operation on column metadata.
    Frozen to ensure audit integrity.

    Attributes:
        patch_id: Unique identifier for this patch
        operation: Type of metadata operation
        column: Target column name
        value: New value (type depends on operation)
        status: Current lifecycle status
        created_at: Timestamp when patch was created
        provenance: Origin of patch (llm, user, dictionary)
        model_id: LLM model that generated this patch (if llm provenance)
        confidence: Confidence score (0-1) for LLM-generated patches
        accepted_by: User who accepted the patch (if accepted)
        accepted_at: Timestamp when patch was accepted
        rejected_reason: Reason for rejection (if rejected)
    """

    patch_id: str
    operation: PatchOperation
    column: str
    value: Any
    status: PatchStatus
    created_at: datetime
    provenance: Literal["llm", "user", "dictionary"]
    model_id: str | None = None
    confidence: float | None = None
    accepted_by: str | None = None
    accepted_at: datetime | None = None
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize patch to dictionary for JSON storage."""
        result = {
            "patch_id": self.patch_id,
            "operation": self.operation.value,
            "column": self.column,
            "value": self.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "provenance": self.provenance,
        }

        if self.model_id is not None:
            result["model_id"] = self.model_id
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.accepted_by is not None:
            result["accepted_by"] = self.accepted_by
        if self.accepted_at is not None:
            result["accepted_at"] = self.accepted_at.isoformat()
        if self.rejected_reason is not None:
            result["rejected_reason"] = self.rejected_reason

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetadataPatch":
        """Deserialize patch from dictionary (JSON deserialization)."""
        created_at = datetime.fromisoformat(data["created_at"])
        accepted_at = None
        if data.get("accepted_at"):
            accepted_at = datetime.fromisoformat(data["accepted_at"])

        return cls(
            patch_id=data["patch_id"],
            operation=PatchOperation(data["operation"]),
            column=data["column"],
            value=data["value"],
            status=PatchStatus(data["status"]),
            created_at=created_at,
            provenance=data["provenance"],
            model_id=data.get("model_id"),
            confidence=data.get("confidence"),
            accepted_by=data.get("accepted_by"),
            accepted_at=accepted_at,
            rejected_reason=data.get("rejected_reason"),
        )


@dataclass(frozen=True)
class ExclusionPatternPatch:
    """
    Patch for exclusion patterns on coded columns.

    Allows marking specific coded values as "exclude" patterns.
    For example, "n/a" → exclude value 0 in queries.

    Attributes:
        patch_id: Unique identifier
        column: Target column name
        pattern: Human-readable pattern name (e.g., "n/a")
        coded_value: Actual value to exclude (e.g., 0)
        context: Usage context for NL query engine
        auto_apply: Whether to auto-exclude in queries
        status: Lifecycle status
        created_at: Creation timestamp
        provenance: Origin of patch
    """

    patch_id: str
    column: str
    pattern: str
    coded_value: int | str
    context: str
    auto_apply: bool
    status: PatchStatus
    created_at: datetime
    provenance: Literal["llm", "user", "dictionary"]
    model_id: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "patch_id": self.patch_id,
            "operation": PatchOperation.SET_EXCLUSION_PATTERN.value,
            "column": self.column,
            "pattern": self.pattern,
            "coded_value": self.coded_value,
            "context": self.context,
            "auto_apply": self.auto_apply,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "provenance": self.provenance,
        }

        if self.model_id is not None:
            result["model_id"] = self.model_id
        if self.confidence is not None:
            result["confidence"] = self.confidence

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExclusionPatternPatch":
        """Deserialize from dictionary."""
        return cls(
            patch_id=data["patch_id"],
            column=data["column"],
            pattern=data["pattern"],
            coded_value=data["coded_value"],
            context=data["context"],
            auto_apply=data.get("auto_apply", False),
            status=PatchStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            provenance=data["provenance"],
            model_id=data.get("model_id"),
            confidence=data.get("confidence"),
        )


@dataclass(frozen=True)
class RelationshipPatch:
    """
    Patch for cross-column relationships.

    Captures relationships between columns for query refinement.
    For example: "Statin Used = 0 ↔ Statin Prescribed = No"

    Attributes:
        patch_id: Unique identifier
        columns: List of related column names
        relationship_type: Type of relationship (coded_exclusion, correlation, hierarchical)
        rule: Human-readable rule description
        inference: How NL engine should use this relationship
        confidence: Confidence score
        status: Lifecycle status
        created_at: Creation timestamp
        provenance: Origin of patch
    """

    patch_id: str
    columns: list[str]
    relationship_type: Literal["coded_exclusion", "correlation", "hierarchical"]
    rule: str
    inference: str
    confidence: float
    status: PatchStatus
    created_at: datetime
    provenance: Literal["llm", "user", "dictionary"]
    model_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "patch_id": self.patch_id,
            "operation": PatchOperation.SET_RELATIONSHIP.value,
            "columns": self.columns,
            "relationship_type": self.relationship_type,
            "rule": self.rule,
            "inference": self.inference,
            "confidence": self.confidence,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "provenance": self.provenance,
        }

        if self.model_id is not None:
            result["model_id"] = self.model_id

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationshipPatch":
        """Deserialize from dictionary."""
        return cls(
            patch_id=data["patch_id"],
            columns=data["columns"],
            relationship_type=data["relationship_type"],
            rule=data["rule"],
            inference=data["inference"],
            confidence=data["confidence"],
            status=PatchStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            provenance=data["provenance"],
            model_id=data.get("model_id"),
        )


@dataclass
class ResolvedExclusionPattern:
    """Resolved exclusion pattern for column metadata."""

    pattern: str
    coded_value: int | str
    context: str
    auto_apply: bool


@dataclass
class ResolvedRelationship:
    """Resolved cross-column relationship."""

    columns: list[str]
    relationship_type: Literal["coded_exclusion", "correlation", "hierarchical"]
    rule: str
    inference: str
    confidence: float


@dataclass
class ResolvedColumnMetadata:
    """
    Resolved metadata for a single column after merge.

    This represents the final merged state of column metadata
    after applying all accepted patches with proper precedence.
    """

    name: str
    label: str | None = None
    description: str | None = None
    semantic_type: SemanticType | None = None
    unit: str | None = None
    is_phi: bool = False
    aliases: list[str] = field(default_factory=list)
    codebook: dict[str, str] | None = None
    exclusion_patterns: list[ResolvedExclusionPattern] | None = None


@dataclass
class ResolvedDatasetMetadata:
    """
    Resolved metadata for entire dataset after merge.

    Contains resolved column metadata and dataset-level relationships.
    """

    columns: dict[str, ResolvedColumnMetadata] = field(default_factory=dict)
    relationships: list[ResolvedRelationship] | None = None

"""
Tests for MetadataPatch dataclasses and validation.

Phase 0: ADR011 Metadata Enrichment
Tests the frozen dataclasses for patch operations with provenance tracking.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest


class TestPatchOperation:
    """Tests for PatchOperation enum."""

    def test_patch_operation_rejects_unknown_value(self):
        """Wire values outside the enum are rejected."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        with pytest.raises(ValueError):
            PatchOperation("not_a_real_operation")

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            ("SET_LABEL", "set_label"),
            ("ADD_ALIAS", "add_alias"),
            ("SET_DESCRIPTION", "set_description"),
            ("SET_SEMANTIC_TYPE", "set_semantic_type"),
            ("MARK_PHI", "mark_phi"),
            ("SET_UNIT", "set_unit"),
            ("SET_CODEBOOK_ENTRY", "set_codebook_entry"),
            ("SET_RELATIONSHIP", "set_relationship"),
            ("SET_EXCLUSION_PATTERN", "set_exclusion_pattern"),
        ],
    )
    def test_patch_operation_member_value(self, member: str, value: str):
        """Each PatchOperation member stores its wire value."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        assert getattr(PatchOperation, member).value == value


class TestSemanticType:
    """Tests for SemanticType enum."""

    def test_semantic_type_rejects_unknown_value(self):
        """Wire values outside the enum are rejected."""
        from clinical_analytics.core.metadata_patch import SemanticType

        with pytest.raises(ValueError):
            SemanticType("not_a_semantic_type")

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            ("IDENTIFIER", "identifier"),
            ("DEMOGRAPHIC", "demographic"),
            ("CLINICAL", "clinical"),
            ("TEMPORAL", "temporal"),
            ("OUTCOME", "outcome"),
            ("MEASUREMENT", "measurement"),
            ("CODED", "coded"),
        ],
    )
    def test_semantic_type_member_value(self, member: str, value: str):
        """Each SemanticType member stores its wire value."""
        from clinical_analytics.core.metadata_patch import SemanticType

        assert getattr(SemanticType, member).value == value


class TestPatchStatus:
    """Tests for PatchStatus enum."""

    def test_patch_status_rejects_unknown_value(self):
        """Wire values outside the enum are rejected."""
        from clinical_analytics.core.metadata_patch import PatchStatus

        with pytest.raises(ValueError):
            PatchStatus("not_a_status")

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            ("PENDING", "pending"),
            ("ACCEPTED", "accepted"),
            ("REJECTED", "rejected"),
            ("REVERTED", "reverted"),
        ],
    )
    def test_patch_status_member_value(self, member: str, value: str):
        """Each PatchStatus member stores its wire value."""
        from clinical_analytics.core.metadata_patch import PatchStatus

        assert getattr(PatchStatus, member).value == value


class TestMetadataPatch:
    """Tests for MetadataPatch frozen dataclass."""

    def test_metadata_patch_creation_minimal(self):
        """Test creating MetadataPatch with minimal required fields."""
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
            created_at=datetime.now(UTC),
            provenance="llm",
        )

        assert patch.column == "hba1c_pct"
        assert patch.operation == PatchOperation.SET_DESCRIPTION
        assert patch.value == "Hemoglobin A1c percentage"
        assert patch.status == PatchStatus.PENDING
        assert patch.provenance == "llm"

    def test_metadata_patch_is_frozen(self):
        """Test that MetadataPatch is immutable (frozen)."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )

        patch = MetadataPatch(
            patch_id=str(uuid4()),
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age in years",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="user",
        )

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            patch.value = "Modified description"

    def test_metadata_patch_with_provenance_fields(self):
        """Test MetadataPatch with full provenance tracking."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )

        patch = MetadataPatch(
            patch_id="test-uuid-123",
            operation=PatchOperation.SET_SEMANTIC_TYPE,
            column="mortality",
            value="outcome",
            status=PatchStatus.ACCEPTED,
            created_at=datetime.now(UTC),
            provenance="llm",
            model_id="llama3.1:8b",
            confidence=0.95,
            accepted_by="user_jane",
            accepted_at=datetime.now(UTC),
        )

        assert patch.model_id == "llama3.1:8b"
        assert patch.confidence == 0.95
        assert patch.accepted_by == "user_jane"
        assert patch.accepted_at is not None

    def test_metadata_patch_codebook_entry(self):
        """Test MetadataPatch for codebook entry with code:label mapping."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )

        # Codebook entry stores code and label
        patch = MetadataPatch(
            patch_id=str(uuid4()),
            operation=PatchOperation.SET_CODEBOOK_ENTRY,
            column="statin_used",
            value={"code": "0", "label": "n/a"},
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )

        assert patch.operation == PatchOperation.SET_CODEBOOK_ENTRY
        assert patch.value == {"code": "0", "label": "n/a"}

    def test_metadata_patch_serialization_to_dict(self):
        """Test MetadataPatch can be serialized to dict for JSON storage."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )

        created_at = datetime.now(UTC)
        patch = MetadataPatch(
            patch_id="test-123",
            operation=PatchOperation.SET_DESCRIPTION,
            column="age",
            value="Patient age",
            status=PatchStatus.PENDING,
            created_at=created_at,
            provenance="user",
        )

        patch_dict = patch.to_dict()

        assert patch_dict["patch_id"] == "test-123"
        assert patch_dict["operation"] == "set_description"
        assert patch_dict["column"] == "age"
        assert patch_dict["status"] == "pending"
        assert "created_at" in patch_dict

    def test_metadata_patch_deserialization_from_dict(self):
        """Test MetadataPatch can be created from dict (JSON deserialization)."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )

        patch_dict = {
            "patch_id": "test-456",
            "operation": "set_description",
            "column": "bmi",
            "value": "Body Mass Index",
            "status": "accepted",
            "created_at": "2024-01-15T10:30:00+00:00",
            "provenance": "llm",
            "model_id": "llama3.1:8b",
        }

        patch = MetadataPatch.from_dict(patch_dict)

        assert patch.patch_id == "test-456"
        assert patch.operation == PatchOperation.SET_DESCRIPTION
        assert patch.column == "bmi"
        assert patch.status == PatchStatus.ACCEPTED
        assert patch.model_id == "llama3.1:8b"


class TestExclusionPatternPatch:
    """Tests for exclusion pattern patches."""

    def test_exclusion_pattern_patch_rejects_missing_fields(self):
        """Required fields are not optional."""
        from clinical_analytics.core.metadata_patch import ExclusionPatternPatch

        with pytest.raises(TypeError):
            ExclusionPatternPatch()

    def test_exclusion_pattern_creation(self):
        """Test creating an exclusion pattern patch."""
        from clinical_analytics.core.metadata_patch import (
            ExclusionPatternPatch,
            PatchStatus,
        )

        patch = ExclusionPatternPatch(
            patch_id=str(uuid4()),
            column="statin_used",
            pattern="n/a",
            coded_value=0,
            context="Use != 0 to exclude patients not on statins",
            auto_apply=False,
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )

        assert patch.column == "statin_used"
        assert patch.pattern == "n/a"
        assert patch.coded_value == 0
        assert patch.auto_apply is False

    def test_exclusion_pattern_serialization(self):
        """Test ExclusionPatternPatch serialization."""
        from clinical_analytics.core.metadata_patch import (
            ExclusionPatternPatch,
            PatchStatus,
        )

        patch = ExclusionPatternPatch(
            patch_id="excl-123",
            column="treatment_group",
            pattern="unknown",
            coded_value="UNK",
            context="Exclude unknown treatment assignments",
            auto_apply=True,
            status=PatchStatus.ACCEPTED,
            created_at=datetime.now(UTC),
            provenance="user",
        )

        patch_dict = patch.to_dict()

        assert patch_dict["pattern"] == "unknown"
        assert patch_dict["coded_value"] == "UNK"
        assert patch_dict["auto_apply"] is True


class TestRelationshipPatch:
    """Tests for cross-column relationship patches."""

    def test_relationship_patch_rejects_missing_fields(self):
        """Required fields are not optional."""
        from clinical_analytics.core.metadata_patch import RelationshipPatch

        with pytest.raises(TypeError):
            RelationshipPatch()

    def test_relationship_patch_creation(self):
        """Test creating a relationship patch between columns."""
        from clinical_analytics.core.metadata_patch import (
            PatchStatus,
            RelationshipPatch,
        )

        patch = RelationshipPatch(
            patch_id=str(uuid4()),
            columns=["Statin Used", "Statin Prescribed"],
            relationship_type="coded_exclusion",
            rule="Statin Used = 0 means patient was not prescribed statins",
            inference="When filtering by Statin Prescribed, also consider Statin Used = 0",
            confidence=0.85,
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )

        assert patch.columns == ["Statin Used", "Statin Prescribed"]
        assert patch.relationship_type == "coded_exclusion"
        assert patch.confidence == 0.85

    def test_relationship_patch_types(self):
        """Test different relationship types are supported."""
        from clinical_analytics.core.metadata_patch import (
            PatchStatus,
            RelationshipPatch,
        )

        # Correlation relationship
        patch = RelationshipPatch(
            patch_id=str(uuid4()),
            columns=["LDL", "Total Cholesterol"],
            relationship_type="correlation",
            rule="LDL is a component of Total Cholesterol",
            inference="High LDL correlates with high Total Cholesterol",
            confidence=0.9,
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )

        assert patch.relationship_type == "correlation"

        # Hierarchical relationship
        patch2 = RelationshipPatch(
            patch_id=str(uuid4()),
            columns=["Drug Class", "Drug Name"],
            relationship_type="hierarchical",
            rule="Drug Name is a child of Drug Class",
            inference="Filtering by Drug Class includes all associated Drug Names",
            confidence=0.95,
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )

        assert patch2.relationship_type == "hierarchical"

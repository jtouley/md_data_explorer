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

    def test_patch_operation_set_label_exists(self):
        """Verify SET_LABEL operation exists."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        assert PatchOperation.SET_LABEL is not None
        assert PatchOperation.SET_LABEL.value == "set_label"

    def test_patch_operation_add_alias_exists(self):
        """Verify ADD_ALIAS operation exists."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        assert PatchOperation.ADD_ALIAS is not None
        assert PatchOperation.ADD_ALIAS.value == "add_alias"

    def test_patch_operation_set_description_exists(self):
        """Verify SET_DESCRIPTION operation exists."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        assert PatchOperation.SET_DESCRIPTION is not None
        assert PatchOperation.SET_DESCRIPTION.value == "set_description"

    def test_patch_operation_set_semantic_type_exists(self):
        """Verify SET_SEMANTIC_TYPE operation exists."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        assert PatchOperation.SET_SEMANTIC_TYPE is not None
        assert PatchOperation.SET_SEMANTIC_TYPE.value == "set_semantic_type"

    def test_patch_operation_mark_phi_exists(self):
        """Verify MARK_PHI operation exists."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        assert PatchOperation.MARK_PHI is not None
        assert PatchOperation.MARK_PHI.value == "mark_phi"

    def test_patch_operation_set_unit_exists(self):
        """Verify SET_UNIT operation exists."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        assert PatchOperation.SET_UNIT is not None
        assert PatchOperation.SET_UNIT.value == "set_unit"

    def test_patch_operation_set_codebook_entry_exists(self):
        """Verify SET_CODEBOOK_ENTRY operation exists."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        assert PatchOperation.SET_CODEBOOK_ENTRY is not None
        assert PatchOperation.SET_CODEBOOK_ENTRY.value == "set_codebook_entry"

    def test_patch_operation_set_relationship_exists(self):
        """Verify SET_RELATIONSHIP operation exists."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        assert PatchOperation.SET_RELATIONSHIP is not None
        assert PatchOperation.SET_RELATIONSHIP.value == "set_relationship"

    def test_patch_operation_set_exclusion_pattern_exists(self):
        """Verify SET_EXCLUSION_PATTERN operation exists."""
        from clinical_analytics.core.metadata_patch import PatchOperation

        assert PatchOperation.SET_EXCLUSION_PATTERN is not None
        assert PatchOperation.SET_EXCLUSION_PATTERN.value == "set_exclusion_pattern"


class TestSemanticType:
    """Tests for SemanticType enum."""

    def test_semantic_type_identifier_exists(self):
        """Verify IDENTIFIER type exists."""
        from clinical_analytics.core.metadata_patch import SemanticType

        assert SemanticType.IDENTIFIER is not None
        assert SemanticType.IDENTIFIER.value == "identifier"

    def test_semantic_type_demographic_exists(self):
        """Verify DEMOGRAPHIC type exists."""
        from clinical_analytics.core.metadata_patch import SemanticType

        assert SemanticType.DEMOGRAPHIC is not None
        assert SemanticType.DEMOGRAPHIC.value == "demographic"

    def test_semantic_type_clinical_exists(self):
        """Verify CLINICAL type exists."""
        from clinical_analytics.core.metadata_patch import SemanticType

        assert SemanticType.CLINICAL is not None
        assert SemanticType.CLINICAL.value == "clinical"

    def test_semantic_type_temporal_exists(self):
        """Verify TEMPORAL type exists."""
        from clinical_analytics.core.metadata_patch import SemanticType

        assert SemanticType.TEMPORAL is not None
        assert SemanticType.TEMPORAL.value == "temporal"

    def test_semantic_type_outcome_exists(self):
        """Verify OUTCOME type exists."""
        from clinical_analytics.core.metadata_patch import SemanticType

        assert SemanticType.OUTCOME is not None
        assert SemanticType.OUTCOME.value == "outcome"

    def test_semantic_type_measurement_exists(self):
        """Verify MEASUREMENT type exists."""
        from clinical_analytics.core.metadata_patch import SemanticType

        assert SemanticType.MEASUREMENT is not None
        assert SemanticType.MEASUREMENT.value == "measurement"

    def test_semantic_type_coded_exists(self):
        """Verify CODED type exists."""
        from clinical_analytics.core.metadata_patch import SemanticType

        assert SemanticType.CODED is not None
        assert SemanticType.CODED.value == "coded"


class TestPatchStatus:
    """Tests for PatchStatus enum."""

    def test_patch_status_pending_exists(self):
        """Verify PENDING status exists."""
        from clinical_analytics.core.metadata_patch import PatchStatus

        assert PatchStatus.PENDING is not None
        assert PatchStatus.PENDING.value == "pending"

    def test_patch_status_accepted_exists(self):
        """Verify ACCEPTED status exists."""
        from clinical_analytics.core.metadata_patch import PatchStatus

        assert PatchStatus.ACCEPTED is not None
        assert PatchStatus.ACCEPTED.value == "accepted"

    def test_patch_status_rejected_exists(self):
        """Verify REJECTED status exists."""
        from clinical_analytics.core.metadata_patch import PatchStatus

        assert PatchStatus.REJECTED is not None
        assert PatchStatus.REJECTED.value == "rejected"


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

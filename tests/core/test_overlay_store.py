"""
Tests for OverlayStore - JSONL patch persistence.

Phase 1: ADR011 Metadata Enrichment
Tests for append-only patch log and overlay storage operations.
"""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest


@pytest.fixture
def overlay_store(tmp_path):
    """Create OverlayStore with temp directory."""
    from clinical_analytics.core.overlay_store import OverlayStore

    return OverlayStore(base_dir=tmp_path)


@pytest.fixture
def sample_patch():
    """Create sample MetadataPatch for testing."""
    from clinical_analytics.core.metadata_patch import (
        MetadataPatch,
        PatchOperation,
        PatchStatus,
    )

    return MetadataPatch(
        patch_id=str(uuid4()),
        operation=PatchOperation.SET_DESCRIPTION,
        column="age",
        value="Patient age in years",
        status=PatchStatus.ACCEPTED,
        created_at=datetime.now(UTC),
        provenance="user",
    )


@pytest.fixture
def sample_exclusion_patch():
    """Create sample ExclusionPatternPatch for testing."""
    from clinical_analytics.core.metadata_patch import (
        ExclusionPatternPatch,
        PatchStatus,
    )

    return ExclusionPatternPatch(
        patch_id=str(uuid4()),
        column="statin_used",
        pattern="n/a",
        coded_value=0,
        context="Exclude patients not on statins",
        auto_apply=False,
        status=PatchStatus.PENDING,
        created_at=datetime.now(UTC),
        provenance="llm",
    )


class TestOverlayStoreAppend:
    """Tests for append_patch operations."""

    def test_append_patch_creates_directory(self, overlay_store, sample_patch):
        """Test that append_patch creates the overlay directory if needed."""
        upload_id = "upload_12345"
        version = "v1"

        overlay_store.append_patch(upload_id, version, sample_patch)

        expected_dir = overlay_store.base_dir / "overlays" / upload_id / version
        assert expected_dir.exists()

    def test_append_patch_creates_jsonl_file(self, overlay_store, sample_patch):
        """Test that append_patch creates patches.jsonl file."""
        upload_id = "upload_12345"
        version = "v1"

        overlay_store.append_patch(upload_id, version, sample_patch)

        jsonl_path = overlay_store.base_dir / "overlays" / upload_id / version / "patches.jsonl"
        assert jsonl_path.exists()

    def test_append_patch_writes_json_line(self, overlay_store, sample_patch):
        """Test that patch is written as valid JSON line."""
        upload_id = "upload_12345"
        version = "v1"

        overlay_store.append_patch(upload_id, version, sample_patch)

        jsonl_path = overlay_store.base_dir / "overlays" / upload_id / version / "patches.jsonl"
        with open(jsonl_path) as f:
            line = f.readline()
            parsed = json.loads(line)

        assert parsed["patch_id"] == sample_patch.patch_id
        assert parsed["column"] == "age"
        assert parsed["operation"] == "set_description"

    def test_append_patch_is_append_only(self, overlay_store, sample_patch):
        """Test that multiple appends add to file (don't overwrite)."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )

        upload_id = "upload_12345"
        version = "v1"

        # Append first patch
        overlay_store.append_patch(upload_id, version, sample_patch)

        # Append second patch
        second_patch = MetadataPatch(
            patch_id=str(uuid4()),
            operation=PatchOperation.SET_LABEL,
            column="bmi",
            value="BMI",
            status=PatchStatus.ACCEPTED,
            created_at=datetime.now(UTC),
            provenance="user",
        )
        overlay_store.append_patch(upload_id, version, second_patch)

        # Verify both lines exist
        jsonl_path = overlay_store.base_dir / "overlays" / upload_id / version / "patches.jsonl"
        with open(jsonl_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_append_exclusion_pattern_patch(self, overlay_store, sample_exclusion_patch):
        """Test appending ExclusionPatternPatch."""
        upload_id = "upload_12345"
        version = "v1"

        overlay_store.append_patch(upload_id, version, sample_exclusion_patch)

        jsonl_path = overlay_store.base_dir / "overlays" / upload_id / version / "patches.jsonl"
        with open(jsonl_path) as f:
            line = f.readline()
            parsed = json.loads(line)

        assert parsed["pattern"] == "n/a"
        assert parsed["coded_value"] == 0


class TestOverlayStoreLoad:
    """Tests for load_patches operations."""

    def test_load_patches_empty_returns_empty_list(self, overlay_store):
        """Test that load_patches returns empty list for non-existent overlay."""
        patches = overlay_store.load_patches("nonexistent", "v1")
        assert patches == []

    def test_load_patches_returns_patches(self, overlay_store, sample_patch):
        """Test that load_patches returns appended patches."""
        upload_id = "upload_12345"
        version = "v1"

        overlay_store.append_patch(upload_id, version, sample_patch)
        patches = overlay_store.load_patches(upload_id, version)

        assert len(patches) == 1
        assert patches[0].patch_id == sample_patch.patch_id
        assert patches[0].column == "age"

    def test_load_patches_preserves_order(self, overlay_store):
        """Test that patches are returned in chronological order."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )

        upload_id = "upload_12345"
        version = "v1"

        # Create patches with different timestamps
        patches_to_add = []
        for i in range(3):
            patch = MetadataPatch(
                patch_id=f"patch-{i}",
                operation=PatchOperation.SET_DESCRIPTION,
                column=f"col_{i}",
                value=f"Description {i}",
                status=PatchStatus.ACCEPTED,
                created_at=datetime.now(UTC),
                provenance="user",
            )
            patches_to_add.append(patch)
            overlay_store.append_patch(upload_id, version, patch)

        loaded = overlay_store.load_patches(upload_id, version)

        assert len(loaded) == 3
        for i, patch in enumerate(loaded):
            assert patch.patch_id == f"patch-{i}"


class TestPendingSuggestions:
    """Tests for pending LLM suggestions storage."""

    def test_save_pending_creates_file(self, overlay_store, sample_patch):
        """Test that save_pending creates pending.json file."""
        upload_id = "upload_12345"
        version = "v1"
        suggestions = [sample_patch]

        overlay_store.save_pending(upload_id, version, suggestions)

        pending_path = overlay_store.base_dir / "overlays" / upload_id / version / "pending.json"
        assert pending_path.exists()

    def test_load_pending_empty_returns_empty_list(self, overlay_store):
        """Test that load_pending returns empty list for non-existent file."""
        suggestions = overlay_store.load_pending("nonexistent", "v1")
        assert suggestions == []

    def test_load_pending_returns_saved_suggestions(self, overlay_store, sample_patch):
        """Test that load_pending returns previously saved suggestions."""
        upload_id = "upload_12345"
        version = "v1"
        suggestions = [sample_patch]

        overlay_store.save_pending(upload_id, version, suggestions)
        loaded = overlay_store.load_pending(upload_id, version)

        assert len(loaded) == 1
        assert loaded[0].patch_id == sample_patch.patch_id

    def test_save_pending_overwrites(self, overlay_store):
        """Test that save_pending overwrites existing pending suggestions."""
        from clinical_analytics.core.metadata_patch import (
            MetadataPatch,
            PatchOperation,
            PatchStatus,
        )

        upload_id = "upload_12345"
        version = "v1"

        # Save first batch
        patch1 = MetadataPatch(
            patch_id="patch-1",
            operation=PatchOperation.SET_DESCRIPTION,
            column="col1",
            value="First",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )
        overlay_store.save_pending(upload_id, version, [patch1])

        # Save second batch (should overwrite)
        patch2 = MetadataPatch(
            patch_id="patch-2",
            operation=PatchOperation.SET_DESCRIPTION,
            column="col2",
            value="Second",
            status=PatchStatus.PENDING,
            created_at=datetime.now(UTC),
            provenance="llm",
        )
        overlay_store.save_pending(upload_id, version, [patch2])

        loaded = overlay_store.load_pending(upload_id, version)
        assert len(loaded) == 1
        assert loaded[0].patch_id == "patch-2"

    def test_clear_pending(self, overlay_store, sample_patch):
        """Test clearing pending suggestions."""
        upload_id = "upload_12345"
        version = "v1"

        overlay_store.save_pending(upload_id, version, [sample_patch])
        overlay_store.clear_pending(upload_id, version)

        loaded = overlay_store.load_pending(upload_id, version)
        assert loaded == []


class TestResolvedCache:
    """Tests for resolved metadata caching."""

    def test_cache_resolved_creates_file(self, overlay_store):
        """Test that cache_resolved creates resolved.json file."""
        from clinical_analytics.core.metadata_patch import (
            ResolvedColumnMetadata,
            ResolvedDatasetMetadata,
        )

        upload_id = "upload_12345"
        version = "v1"
        resolved = ResolvedDatasetMetadata(
            columns={"age": ResolvedColumnMetadata(name="age", description="Patient age")}
        )

        overlay_store.cache_resolved(upload_id, version, resolved)

        cache_path = overlay_store.base_dir / "overlays" / upload_id / version / "resolved.json"
        assert cache_path.exists()

    def test_load_cached_resolved_returns_cached(self, overlay_store):
        """Test that load_cached_resolved returns previously cached metadata."""
        from clinical_analytics.core.metadata_patch import (
            ResolvedColumnMetadata,
            ResolvedDatasetMetadata,
        )

        upload_id = "upload_12345"
        version = "v1"
        resolved = ResolvedDatasetMetadata(
            columns={"age": ResolvedColumnMetadata(name="age", description="Patient age")}
        )

        overlay_store.cache_resolved(upload_id, version, resolved)
        loaded = overlay_store.load_cached_resolved(upload_id, version)

        assert loaded is not None
        assert "age" in loaded.columns
        assert loaded.columns["age"].description == "Patient age"

    def test_load_cached_resolved_returns_none_if_missing(self, overlay_store):
        """Test that load_cached_resolved returns None if no cache exists."""
        loaded = overlay_store.load_cached_resolved("nonexistent", "v1")
        assert loaded is None

    def test_invalidate_cache(self, overlay_store):
        """Test that invalidate_cache removes the cached resolved metadata."""
        from clinical_analytics.core.metadata_patch import (
            ResolvedColumnMetadata,
            ResolvedDatasetMetadata,
        )

        upload_id = "upload_12345"
        version = "v1"
        resolved = ResolvedDatasetMetadata(columns={"age": ResolvedColumnMetadata(name="age")})

        overlay_store.cache_resolved(upload_id, version, resolved)
        overlay_store.invalidate_cache(upload_id, version)

        loaded = overlay_store.load_cached_resolved(upload_id, version)
        assert loaded is None


class TestOverlayStorePath:
    """Tests for overlay path management."""

    def test_get_overlay_path(self, overlay_store):
        """Test get_overlay_path returns correct path."""
        upload_id = "upload_12345"
        version = "v1"

        path = overlay_store.get_overlay_path(upload_id, version)

        expected = overlay_store.base_dir / "overlays" / upload_id / version
        assert path == expected

    def test_overlay_exists_false_initially(self, overlay_store):
        """Test overlay_exists returns False for non-existent overlay."""
        assert not overlay_store.overlay_exists("nonexistent", "v1")

    def test_overlay_exists_true_after_append(self, overlay_store, sample_patch):
        """Test overlay_exists returns True after appending a patch."""
        upload_id = "upload_12345"
        version = "v1"

        overlay_store.append_patch(upload_id, version, sample_patch)

        assert overlay_store.overlay_exists(upload_id, version)


class TestPatchAcceptReject:
    """Tests for accepting and rejecting patches."""

    def test_accept_patch_updates_status(self, overlay_store, sample_patch):
        """Test that accept_patch creates a new accepted patch entry."""
        from clinical_analytics.core.metadata_patch import PatchStatus

        upload_id = "upload_12345"
        version = "v1"

        # Save as pending first
        overlay_store.save_pending(upload_id, version, [sample_patch])

        # Accept the patch
        overlay_store.accept_patch(upload_id, version, sample_patch.patch_id, accepted_by="user_jane")

        # Load patches and verify
        patches = overlay_store.load_patches(upload_id, version)
        assert len(patches) == 1
        assert patches[0].status == PatchStatus.ACCEPTED
        assert patches[0].accepted_by == "user_jane"

    def test_reject_patch_records_rejection(self, overlay_store, sample_patch):
        """Test that reject_patch records the rejection."""
        from clinical_analytics.core.metadata_patch import PatchStatus

        upload_id = "upload_12345"
        version = "v1"

        # Save as pending first
        overlay_store.save_pending(upload_id, version, [sample_patch])

        # Reject the patch
        overlay_store.reject_patch(upload_id, version, sample_patch.patch_id, reason="Not accurate")

        # Verify rejection is recorded (in patches.jsonl)
        patches = overlay_store.load_patches(upload_id, version)
        assert len(patches) == 1
        assert patches[0].status == PatchStatus.REJECTED
        assert patches[0].rejected_reason == "Not accurate"


class TestDeserializePatchImmutability:
    """Tests for _deserialize_patch not mutating input."""

    def test_deserialize_patch_does_not_mutate_input_dict(self, overlay_store):
        """Test that _deserialize_patch does not modify the input dictionary."""

        original_data = {
            "_patch_type": "MetadataPatch",
            "patch_id": "test_123",
            "operation": "set_description",
            "column": "age",
            "value": "Patient age",
            "status": "accepted",
            "created_at": datetime.now(UTC).isoformat(),
            "provenance": "user",
        }

        # Make a copy to compare
        original_copy = dict(original_data)

        # Call _deserialize_patch
        overlay_store._deserialize_patch(original_data)

        # Original dict should not be modified
        assert original_data == original_copy
        assert "_patch_type" in original_data

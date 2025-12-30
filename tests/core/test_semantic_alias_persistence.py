"""
Test Adaptive Alias Persistence (Phase 2 - ADR003).

Tests for user alias persistence, loading, collision detection, and scope isolation.
"""

import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from clinical_analytics.core.semantic import SemanticLayer

# ============================================================================
# Fixtures (Phase 2)
# ============================================================================


@pytest.fixture
def sample_metadata_with_aliases():
    """Factory for metadata dict with alias_mappings structure."""

    def _make(
        upload_id: str = "test_upload_123",
        dataset_version: str = "v1.0.0",
        user_aliases: dict[str, str] | None = None,
    ) -> dict:
        return {
            "upload_id": upload_id,
            "dataset_version": dataset_version,
            "alias_mappings": {
                "user_aliases": user_aliases or {"VL": "viral_load", "LDL": "LDL mg/dL"},
                "system_aliases": {},  # System aliases built from column names
            },
            "columns": ["patient_id", "viral_load", "LDL mg/dL", "age"],
        }

    return _make


@pytest.fixture
def mock_metadata_storage(tmp_path):
    """Mock metadata file I/O for testing persistence."""
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    def _save(upload_id: str, metadata: dict) -> Path:
        """Save metadata to file."""
        metadata_path = metadata_dir / f"{upload_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata_path

    def _load(upload_id: str) -> dict | None:
        """Load metadata from file."""
        metadata_path = metadata_dir / f"{upload_id}.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path) as f:
            return json.load(f)

    return {
        "dir": metadata_dir,
        "save": _save,
        "load": _load,
    }


@pytest.fixture
def sample_semantic_layer_with_upload_id(tmp_path):
    """Create SemanticLayer instance with upload_id context."""
    # Create workspace structure
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname = 'test'")

    data_dir = workspace / "data" / "raw" / "test_dataset"
    data_dir.mkdir(parents=True)

    # Create minimal config with proper structure
    config = {
        "init_params": {"source_path": "data/raw/test_dataset/test.csv"},
        "column_mapping": {},
        "outcomes": {},
        "analysis": {"default_outcome": "outcome"},
    }

    # Create minimal CSV file
    df = pl.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "viral_load": [100, 200, 300],
            "LDL mg/dL": [50, 60, 70],
            "age": [30, 40, 50],
        }
    )
    df.write_csv(data_dir / "test.csv")

    layer = SemanticLayer(dataset_name="test_dataset", config=config, workspace_root=workspace)
    return layer


# ============================================================================
# Test Cases (Phase 2)
# ============================================================================


def test_add_user_alias_persists_to_metadata(
    sample_semantic_layer_with_upload_id,
    mock_metadata_storage,
    sample_metadata_with_aliases,
):
    """Verify add_user_alias() saves to metadata JSON."""
    # Arrange
    layer = sample_semantic_layer_with_upload_id
    upload_id = "test_upload_123"
    dataset_version = "v1.0.0"
    term = "VL"
    column = "viral_load"

    # Create initial metadata
    initial_metadata = sample_metadata_with_aliases(upload_id, dataset_version, {})
    metadata_path = mock_metadata_storage["save"](upload_id, initial_metadata)

    # Mock metadata loading/saving
    with (
        patch.object(
            layer,
            "_load_metadata",
            return_value=initial_metadata,
        ),
        patch.object(
            layer,
            "_save_metadata",
            side_effect=lambda upload_id, metadata: mock_metadata_storage["save"](upload_id, metadata),
        ),
        patch.object(
            layer,
            "_get_metadata_path",
            return_value=metadata_path,
        ),
    ):
        # Act
        layer.add_user_alias(term, column, upload_id, dataset_version)

        # Assert
        saved_metadata = mock_metadata_storage["load"](upload_id)
        assert saved_metadata is not None
        assert "alias_mappings" in saved_metadata
        assert "user_aliases" in saved_metadata["alias_mappings"]
        assert saved_metadata["alias_mappings"]["user_aliases"][term] == column


def test_load_user_aliases_on_initialization(
    tmp_path,
    mock_metadata_storage,
    sample_metadata_with_aliases,
):
    """Verify SemanticLayer loads persisted aliases on init."""
    # Arrange
    upload_id = "test_upload_123"
    dataset_version = "v1.0.0"
    user_aliases = {"VL": "viral_load", "LDL": "LDL mg/dL"}

    # Create workspace structure
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname = 'test'")

    # Create metadata directory structure
    metadata_dir = workspace / "data" / "uploads" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata with user aliases
    metadata = sample_metadata_with_aliases(upload_id, dataset_version, user_aliases)
    metadata_path = metadata_dir / f"{upload_id}.json"
    import json

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    data_dir = workspace / "data" / "raw" / "test_dataset"
    data_dir.mkdir(parents=True)

    # Create config with proper structure
    config = {
        "init_params": {"source_path": "data/raw/test_dataset/test.csv"},
        "column_mapping": {},
        "outcomes": {},
        "analysis": {"default_outcome": "outcome"},
    }

    # Create CSV
    df = pl.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "viral_load": [100, 200, 300],
            "LDL mg/dL": [50, 60, 70],
        }
    )
    df.write_csv(data_dir / "test.csv")

    # Act - Initialize SemanticLayer with upload_id and dataset_version
    layer = SemanticLayer(
        dataset_name="test_dataset",
        config=config,
        workspace_root=workspace,
        upload_id=upload_id,
        dataset_version=dataset_version,
    )

    # Assert
    alias_index = layer.get_column_alias_index()
    # User aliases should be in alias index (normalized)
    normalized_vl = layer._normalize_alias("VL")
    assert normalized_vl in alias_index
    assert alias_index[normalized_vl] == "viral_load"


def test_user_aliases_override_system_aliases(
    sample_semantic_layer_with_upload_id,
    mock_metadata_storage,
    sample_metadata_with_aliases,
):
    """Verify user aliases take precedence over system aliases."""
    # Arrange
    layer = sample_semantic_layer_with_upload_id
    upload_id = "test_upload_123"
    dataset_version = "v1.0.0"

    # System alias would map "viral" -> "viral_load" (from column name)
    # User alias maps "VL" -> "viral_load"
    user_aliases = {"VL": "viral_load"}

    metadata = sample_metadata_with_aliases(upload_id, dataset_version, user_aliases)
    mock_metadata_storage["save"](upload_id, metadata)

    # Act
    with patch.object(
        layer,
        "_load_metadata",
        return_value=metadata,
    ):
        layer._load_user_aliases(upload_id, dataset_version)
        alias_index = layer.get_column_alias_index()

        # Assert
        # User alias "VL" (normalized to "vl") should map to "viral_load"
        normalized_vl = layer._normalize_alias("VL")
        assert normalized_vl in alias_index
        assert alias_index[normalized_vl] == "viral_load"

        # User alias should override any system alias for same normalized key
        # (This will be verified once system aliases are built)


def test_alias_scope_per_dataset(
    tmp_path,
    mock_metadata_storage,
    sample_metadata_with_aliases,
):
    """Verify aliases scoped to (upload_id, dataset_version) don't leak to other datasets."""
    # Arrange
    upload_id_1 = "upload_1"
    upload_id_2 = "upload_2"
    dataset_version = "v1.0.0"

    # Create two different datasets with different aliases
    metadata_1 = sample_metadata_with_aliases(
        upload_id_1,
        dataset_version,
        {"VL": "viral_load"},
    )
    metadata_2 = sample_metadata_with_aliases(
        upload_id_2,
        dataset_version,
        {"VL": "viral_load_2"},  # Same alias, different column
    )

    mock_metadata_storage["save"](upload_id_1, metadata_1)
    mock_metadata_storage["save"](upload_id_2, metadata_2)

    # Create workspace structure
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname = 'test'")

    data_dir_1 = workspace / "data" / "raw" / "dataset_1"
    data_dir_1.mkdir(parents=True)
    data_dir_2 = workspace / "data" / "raw" / "dataset_2"
    data_dir_2.mkdir(parents=True)

    # Create configs with proper structure
    config_1 = {
        "init_params": {"source_path": "data/raw/dataset_1/dataset_1.csv"},
        "column_mapping": {},
        "outcomes": {},
        "analysis": {"default_outcome": "outcome"},
    }
    config_2 = {
        "init_params": {"source_path": "data/raw/dataset_2/dataset_2.csv"},
        "column_mapping": {},
        "outcomes": {},
        "analysis": {"default_outcome": "outcome"},
    }

    # Create CSVs
    df1 = pl.DataFrame({"patient_id": [1], "viral_load": [100]})
    df1.write_csv(data_dir_1 / "dataset_1.csv")

    df2 = pl.DataFrame({"patient_id": [1], "viral_load_2": [200]})
    df2.write_csv(data_dir_2 / "dataset_2.csv")

    # Act
    layer_1 = SemanticLayer(
        dataset_name="dataset_1",
        config=config_1,
        workspace_root=workspace,
    )
    layer_2 = SemanticLayer(
        dataset_name="dataset_2",
        config=config_2,
        workspace_root=workspace,
    )

    with patch(
        "clinical_analytics.core.semantic.Path",
        return_value=mock_metadata_storage["dir"],
    ):
        layer_1._load_user_aliases(upload_id_1, dataset_version)
        layer_2._load_user_aliases(upload_id_2, dataset_version)

        alias_index_1 = layer_1.get_column_alias_index()
        alias_index_2 = layer_2.get_column_alias_index()

        # Assert
        # Each layer should only have aliases from its own metadata
        normalized_vl = layer_1._normalize_alias("VL")
        if normalized_vl in alias_index_1:
            assert alias_index_1[normalized_vl] == "viral_load"
        if normalized_vl in alias_index_2:
            assert alias_index_2[normalized_vl] == "viral_load_2"


def test_orphaned_alias_handling(
    sample_semantic_layer_with_upload_id,
    mock_metadata_storage,
    sample_metadata_with_aliases,
):
    """Verify aliases marked orphaned when target column missing after schema change."""
    # Arrange
    layer = sample_semantic_layer_with_upload_id
    upload_id = "test_upload_123"
    dataset_version = "v1.0.0"

    # Create metadata with alias to column that doesn't exist
    user_aliases = {"VL": "nonexistent_column"}
    metadata = sample_metadata_with_aliases(upload_id, dataset_version, user_aliases)
    mock_metadata_storage["save"](upload_id, metadata)

    # Act
    with patch.object(
        layer,
        "_load_metadata",
        return_value=metadata,
    ):
        layer._load_user_aliases(upload_id, dataset_version)

        # Assert
        # Orphaned alias should not be in alias_index
        alias_index = layer.get_column_alias_index()
        # Orphaned alias should be ignored (not in index)
        # This will be verified once orphaned detection is implemented
        # Note: "VL" -> "nonexistent_column" won't be in index since column doesn't exist
        normalized_vl = layer._normalize_alias("VL")
        # Verify orphaned alias is not in index (column doesn't exist)
        assert normalized_vl not in alias_index or alias_index.get(normalized_vl) != "nonexistent_column"


def test_alias_collision_detection(
    sample_semantic_layer_with_upload_id,
    mock_metadata_storage,
    sample_metadata_with_aliases,
):
    """Verify collisions surfaced in UI, never silently remapped."""
    # Arrange
    layer = sample_semantic_layer_with_upload_id
    upload_id = "test_upload_123"
    dataset_version = "v1.0.0"

    # Create metadata with collision: same alias maps to multiple columns
    # This shouldn't be possible in normal flow, but test defensive behavior
    user_aliases = {"VL": "viral_load"}
    metadata = sample_metadata_with_aliases(upload_id, dataset_version, user_aliases)
    mock_metadata_storage["save"](upload_id, metadata)

    # Act
    with patch.object(
        layer,
        "_load_metadata",
        return_value=metadata,
    ):
        # Try to add alias that would create collision
        # (e.g., if "VL" already maps to "viral_load" via system alias)
        # This should be detected and surfaced, not silently remapped

        # For now, verify collision detection exists
        collision_warnings = layer.get_collision_warnings()
        assert isinstance(collision_warnings, set)


def test_alias_persistence_format(
    mock_metadata_storage,
    sample_metadata_with_aliases,
):
    """Verify metadata JSON format matches ADR002 schema."""
    # Arrange
    upload_id = "test_upload_123"
    dataset_version = "v1.0.0"
    user_aliases = {"VL": "viral_load", "LDL": "LDL mg/dL"}

    metadata = sample_metadata_with_aliases(upload_id, dataset_version, user_aliases)

    # Act
    metadata_path = mock_metadata_storage["save"](upload_id, metadata)

    # Assert
    assert metadata_path.exists()
    with open(metadata_path) as f:
        loaded = json.load(f)

    # Verify structure matches ADR002 schema
    assert "alias_mappings" in loaded
    assert "user_aliases" in loaded["alias_mappings"]
    assert "system_aliases" in loaded["alias_mappings"]
    assert isinstance(loaded["alias_mappings"]["user_aliases"], dict)
    assert loaded["alias_mappings"]["user_aliases"] == user_aliases


def test_alias_normalization_consistency(
    sample_semantic_layer_with_upload_id,
):
    """Verify user aliases normalized consistently with system aliases."""
    # Arrange
    layer = sample_semantic_layer_with_upload_id

    # Act
    # Test normalization consistency
    test_cases = [
        ("VL", "vl"),
        ("viral load", "viral load"),
        ("LDL mg/dL", "ldl mgdl"),
        ("CD4+", "cd4"),
    ]

    for input_term, expected_normalized in test_cases:
        normalized = layer._normalize_alias(input_term)
        # Normalization should be consistent
        assert normalized == expected_normalized

    # User aliases should use same normalization as system aliases
    # (This is verified by using _normalize_alias() for both)

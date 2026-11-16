# SPDX-License-Identifier: MIT
"""Upload metadata is normalized once at load: container fields are never None or the wrong type."""

import copy
import json
from typing import Any

import pytest
from structlog.testing import capture_logs

from clinical_analytics.storage.upload_metadata import normalize_upload_metadata

# (path to the field, its empty container). A nested field's parent is present and non-empty.
_CONTAINER_FIELDS = [
    pytest.param(("variable_types",), {}, id="variable_types"),
    pytest.param(("tables",), [], id="tables"),
    pytest.param(("parquet_paths",), {}, id="parquet_paths"),
    pytest.param(("version_history",), [], id="version_history"),
    pytest.param(("variable_mapping", "time_variables"), {}, id="variable_mapping.time_variables"),
    pytest.param(("variable_mapping", "predictors"), [], id="variable_mapping.predictors"),
    pytest.param(("inferred_schema", "column_mapping"), {}, id="inferred_schema.column_mapping"),
    pytest.param(("inferred_schema", "outcomes"), {}, id="inferred_schema.outcomes"),
    pytest.param(("inferred_schema", "time_zero"), {}, id="inferred_schema.time_zero"),
    pytest.param(("inferred_schema", "predictors"), [], id="inferred_schema.predictors"),
    pytest.param(("inferred_schema", "categorical_columns"), [], id="inferred_schema.categorical_columns"),
]


def _metadata_with(path: tuple[str, ...], present: bool, value: Any) -> dict[str, Any]:
    """Metadata whose field at ``path`` is ``value`` (or absent when ``present`` is False)."""
    metadata: dict[str, Any] = {
        "upload_id": "u1",
        "variable_mapping": {"patient_id": "pid"},
        "inferred_schema": {"granularities": ["patient_level"]},
    }
    parent = metadata
    for key in path[:-1]:
        parent = parent[key]
    if present:
        parent[path[-1]] = value
    return metadata


def _field(metadata: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path[:-1]:
        metadata = metadata[key]
    return metadata.get(path[-1], "ABSENT")


class TestNormalizeUploadMetadata:
    @pytest.mark.parametrize(("path", "empty"), _CONTAINER_FIELDS)
    def test_normalize_fieldNone_becomesEmptyContainerSilently(self, path, empty):
        # Arrange
        metadata = _metadata_with(path, present=True, value=None)

        # Act
        with capture_logs() as logs:
            result = normalize_upload_metadata(metadata)

        # Assert: how older uploads say "none", so no warning
        assert _field(result, path) == empty
        assert logs == []

    @pytest.mark.parametrize(("path", "empty"), _CONTAINER_FIELDS)
    def test_normalize_fieldWrongType_becomesEmptyContainerAndWarns(self, path, empty):
        # Arrange
        metadata = _metadata_with(path, present=True, value="not-a-container")

        # Act
        with capture_logs() as logs:
            result = normalize_upload_metadata(metadata, upload_id="u1")

        # Assert
        assert _field(result, path) == empty
        assert [(e["event"], e["field"], e["found_type"], e["upload_id"]) for e in logs] == [
            ("upload_metadata_field_wrong_type", ".".join(path), "str", "u1")
        ]

    @pytest.mark.parametrize(("path", "empty"), _CONTAINER_FIELDS)
    def test_normalize_fieldAbsent_staysAbsentExceptVariableTypes(self, path, empty):
        # Arrange
        metadata = _metadata_with(path, present=False, value=None)

        # Act
        result = normalize_upload_metadata(metadata)

        # Assert: config builders copy variable_types, so it is always there; others keep absence
        assert _field(result, path) == (empty if path == ("variable_types",) else "ABSENT")

    @pytest.mark.parametrize("container", [None, "not-a-dict"], ids=["none", "wrong_type"])
    def test_normalize_variableMappingOrSchemaNotDict_isEmptyDictAndFalsy(self, container):
        # Arrange
        metadata = {"variable_mapping": container, "inferred_schema": container}

        # Act
        result = normalize_upload_metadata(metadata)

        # Assert: empty (falsy), so the builders' "has a mapping/schema" checks are unchanged
        assert result["variable_mapping"] == {}
        assert result["inferred_schema"] == {}

    @pytest.mark.parametrize(
        ("path", "value", "kept", "dropped"),
        [
            pytest.param(
                ("variable_types",),
                {"age": {"type": "numeric"}, "sex": "categorical", "bmi": None},
                {"age": {"type": "numeric"}},
                2,
                id="variable_types_non_dict_entries",
            ),
            pytest.param(
                ("inferred_schema", "outcomes"),
                {"mx": {"source_column": "mx"}, "bad": "mx"},
                {"mx": {"source_column": "mx"}},
                1,
                id="outcomes_non_dict_entry",
            ),
            pytest.param(
                ("inferred_schema", "column_mapping"),
                {"pid": "patient_id", "x": 3},
                {"pid": "patient_id"},
                1,
                id="column_mapping_non_str_role",
            ),
            pytest.param(
                ("variable_mapping", "predictors"), ["age", None, 3], ["age"], 2, id="predictors_non_str_items"
            ),
            pytest.param(
                ("variable_mapping", "time_variables"),
                {"time_zero": None, "event": 1},
                {"time_zero": None},
                1,
                id="time_variables_none_kept_int_dropped",
            ),
        ],
    )
    def test_normalize_malformedEntries_droppedWithWarning(self, path, value, kept, dropped):
        # Arrange
        metadata = _metadata_with(path, present=True, value=value)

        # Act
        with capture_logs() as logs:
            result = normalize_upload_metadata(metadata)

        # Assert
        assert _field(result, path) == kept
        assert [e["event"] for e in logs] == ["upload_metadata_entry_dropped"] * dropped

    def test_normalize_validMetadata_isUnchangedIdempotentAndInputUntouched(self):
        # Arrange
        metadata = {
            "upload_id": "u1",
            "row_count": 3,
            "variable_types": {"age": {"type": "numeric"}},
            "variable_mapping": {"patient_id": "pid", "outcome": None, "time_variables": {}, "predictors": ["age"]},
            "inferred_schema": {"column_mapping": {"pid": "patient_id"}, "outcomes": {}},
        }
        original = copy.deepcopy(metadata)

        # Act
        once = normalize_upload_metadata(metadata)
        twice = normalize_upload_metadata(once)

        # Assert
        assert once == original
        assert twice == once
        assert metadata == original


class TestGetUploadMetadataNormalizes:
    @pytest.mark.parametrize(
        ("present", "value"),
        [
            pytest.param(True, None, id="none"),
            pytest.param(False, None, id="absent"),
            pytest.param(True, "x", id="str"),
        ],
    )
    def test_getUploadMetadata_variableTypesMissingOrInvalid_isEmptyDict(self, upload_storage, present, value):
        # Arrange: a metadata file as an older upload (or a damaged file) wrote it
        metadata: dict[str, Any] = {"upload_id": "u1", "upload_timestamp": "2024-01-01T00:00:00"}
        if present:
            metadata["variable_types"] = value
        (upload_storage.metadata_dir / "u1.json").write_text(json.dumps(metadata))

        # Act
        loaded = upload_storage.get_upload_metadata("u1")

        # Assert
        assert loaded is not None
        assert loaded["variable_types"] == {}

# SPDX-License-Identifier: MIT
"""Uploaded-dataset metadata, normalized once where it is loaded.

The metadata JSON is external data: older versions and the upload probe wrote keys with value
``None`` (``"variable_types": null``), and a hand-edited or damaged file can hold the wrong type.
A ``.get(key, {})`` default covers only a missing key, so every config builder and reader that
iterates such a field crashed on a present ``None``.

``normalize_upload_metadata`` is the one boundary: after it, every container field below is
either absent or of its declared container type, never ``None``, and every entry has the
declared entry type. A present ``None`` becomes the empty container (silently: it is how older
uploads say "none"); a field of the wrong type becomes the empty container and a malformed entry
is dropped, each with a structlog warning. ``variable_types`` is always present, since both
config builders copy it into the semantic-layer config. Absent fields other than that stay
absent, because some writers test ``key in metadata``.

Readers then use plain ``metadata.get(key, {})`` / ``[key]`` access.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

import structlog

logger = structlog.get_logger()

ContainerKind: TypeAlias = Literal["dict", "list"]
EntryCheck: TypeAlias = Callable[[Any], bool]


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _is_optional_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


@dataclass(frozen=True)
class _FieldSpec:
    """One container field: its type, and the type each entry (dict value / list item) must have."""

    kind: ContainerKind
    entry_ok: EntryCheck


# Top-level metadata fields that config builders and readers iterate or index.
_TOP_LEVEL_FIELDS: Final[dict[str, _FieldSpec]] = {
    "variable_types": _FieldSpec("dict", _is_dict),  # column -> {"type": ..., "codebook": ...}
    "variable_mapping": _FieldSpec("dict", lambda _v: True),  # nested fields checked below
    "inferred_schema": _FieldSpec("dict", lambda _v: True),  # nested fields checked below
    "tables": _FieldSpec("list", _is_str),
    "parquet_paths": _FieldSpec("dict", _is_str),
    "version_history": _FieldSpec("list", _is_dict),
}

# Fields inside variable_mapping (single-table uploads).
_VARIABLE_MAPPING_FIELDS: Final[dict[str, _FieldSpec]] = {
    "time_variables": _FieldSpec("dict", _is_optional_str),  # {"time_zero": None} is valid
    "predictors": _FieldSpec("list", _is_str),
}

# Fields inside inferred_schema (multi-table uploads and converted single-table ones).
_INFERRED_SCHEMA_FIELDS: Final[dict[str, _FieldSpec]] = {
    "column_mapping": _FieldSpec("dict", _is_str),  # source column -> role
    "outcomes": _FieldSpec("dict", _is_dict),  # name -> {"source_column": ..., "type": ...}
    "time_zero": _FieldSpec("dict", lambda _v: True),
    "predictors": _FieldSpec("list", _is_str),
    "categorical_columns": _FieldSpec("list", _is_str),
    "granularities": _FieldSpec("list", _is_str),
}

# Fields that are set to their empty container even when absent.
_ALWAYS_PRESENT: Final[frozenset[str]] = frozenset({"variable_types"})


def _normalize_container(value: Any, spec: _FieldSpec, field_path: str, log: Any) -> dict[str, Any] | list[Any]:
    """The field as a fresh container of ``spec.kind``, with malformed entries dropped (and logged)."""
    if value is None:
        return {} if spec.kind == "dict" else []
    expected = dict if spec.kind == "dict" else list
    if not isinstance(value, expected):
        log.warning(
            "upload_metadata_field_wrong_type",
            field=field_path,
            expected=spec.kind,
            found_type=type(value).__name__,
        )
        return {} if spec.kind == "dict" else []
    if isinstance(value, dict):
        kept_dict: dict[str, Any] = {}
        for key, entry in value.items():
            if spec.entry_ok(entry):
                kept_dict[key] = entry
            else:
                # The key is a column or table name (schema, not cell data).
                log.warning(
                    "upload_metadata_entry_dropped",
                    field=field_path,
                    entry_key=str(key),
                    found_type=type(entry).__name__,
                )
        return kept_dict
    kept_list: list[Any] = []
    for index, entry in enumerate(value):
        if spec.entry_ok(entry):
            kept_list.append(entry)
        else:
            log.warning(
                "upload_metadata_entry_dropped",
                field=field_path,
                entry_index=index,
                found_type=type(entry).__name__,
            )
    return kept_list


def _normalize_fields(
    source: Mapping[str, Any],
    specs: Mapping[str, _FieldSpec],
    prefix: str,
    log: Any,
    always_present: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    result = dict(source)
    for name, spec in specs.items():
        if name in source:
            result[name] = _normalize_container(source[name], spec, f"{prefix}{name}", log)
        elif name in always_present:
            result[name] = {} if spec.kind == "dict" else []
    return result


def normalize_upload_metadata(metadata: Mapping[str, Any], *, upload_id: str | None = None) -> dict[str, Any]:
    """Return a copy of upload metadata whose container fields are never ``None`` or the wrong type.

    Idempotent; the input is not changed. Fields this module does not declare are copied as-is.

    Args:
        metadata: The metadata as read from the JSON file (or the upload probe's dict).
        upload_id: Bound to the warnings, for tracing which upload was malformed.

    Returns:
        The normalized metadata.
    """
    log = logger.bind(upload_id=upload_id) if upload_id is not None else logger
    result = _normalize_fields(metadata, _TOP_LEVEL_FIELDS, "", log, _ALWAYS_PRESENT)
    if "variable_mapping" in result:
        result["variable_mapping"] = _normalize_fields(
            result["variable_mapping"], _VARIABLE_MAPPING_FIELDS, "variable_mapping.", log
        )
    if "inferred_schema" in result:
        result["inferred_schema"] = _normalize_fields(
            result["inferred_schema"], _INFERRED_SCHEMA_FIELDS, "inferred_schema.", log
        )
    return result

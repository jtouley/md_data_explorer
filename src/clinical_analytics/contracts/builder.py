"""Build an ODCS v3.2.0 data contract for one uploaded dataset version.

Inputs are the stored (post-normalization) frame, the identifier column from schema inference,
codebooks decoded from headers (``analysis.variable_types.code_labels``) and resolved ADR011
metadata (accepted patches and dictionary codebooks). The contract records expectations that do
not come from the values themselves, so checking the data against it at upload is not circular:

* the identifier column must be present and unique,
* coded columns must hold only codes their codebook documents,
* clinical variables must lie inside plausibility bounds (``contracts.config``),
* missingness per column is measured and flagged above a threshold,
* the row count must clear the small-cell threshold.

Observed values (levels, min/max) are never written into the contract: they would make the
checks pass by construction, and the contract should not repeat patient values.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pydantic
from open_data_contract_standard.model import OpenDataContractStandard

from clinical_analytics.analysis.variable_types import classify_variable, code_labels, short_name
from clinical_analytics.contracts import config
from clinical_analytics.contracts.errors import ContractValidationError
from clinical_analytics.core.metadata_patch import ResolvedColumnMetadata, ResolvedDatasetMetadata

_SLUG_STRIP = re.compile(r"[^A-Za-z0-9]+")
_INT_CODE = re.compile(r"^-?\d+$")
_FLOAT_CODE = re.compile(r"^-?\d+\.\d+$")


@dataclass(frozen=True)
class ContractInputs:
    """Everything except the frame that goes into a contract."""

    upload_id: str
    dataset_version: str
    dataset_name: str
    id_column: str | None
    source_file_sha256: str | None = None
    resolved: ResolvedDatasetMetadata | None = None
    accepted_patch_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    created_at: datetime | None = None
    extra_custom_properties: dict[str, Any] = field(default_factory=dict)


def logical_type(dtype: pl.DataType) -> str:
    """ODCS ``logicalType`` for a Polars dtype (enum in ODCS v3.2.0 SchemaBaseProperty)."""
    if dtype == pl.Boolean:
        return "boolean"
    if dtype.is_integer():
        return "integer"
    if dtype.is_float() or isinstance(dtype, pl.Decimal):
        return "number"
    if dtype == pl.Date:
        return "date"
    if isinstance(dtype, pl.Datetime):
        return "timestamp"
    if dtype == pl.Time:
        return "time"
    if isinstance(dtype, pl.List | pl.Array):
        return "array"
    if isinstance(dtype, pl.Struct):
        return "object"
    return "string"


def plausibility_key(column: str) -> str:
    """'Age (years)' and 'age_years' -> 'ageyears'; codebooks in headers are dropped first."""
    return re.sub(r"[^a-z0-9]", "", short_name(column).lower())


def find_plausibility_range(column: str) -> config.PlausibilityRange | None:
    key = plausibility_key(column)
    return next((r for r in config.PLAUSIBILITY_RANGES if key in r.aliases), None)


def _slug(column: str, index: int) -> str:
    slug = _SLUG_STRIP.sub("_", short_name(column)).strip("_").lower()
    return slug or f"column_{index + 1}"


def unique_names(columns: Iterable[str]) -> dict[str, str]:
    """Stable, unique ODCS names (``[a-z0-9_]``) for physical column names, in column order."""
    names: dict[str, str] = {}
    used: set[str] = set()
    for i, column in enumerate(columns):
        base = _slug(column, i)
        name, n = base, 2
        while name in used:
            name, n = f"{base}_{n}", n + 1
        used.add(name)
        names[column] = name
    return names


def _typed_code(code: str, dtype: pl.DataType) -> int | float | str | None:
    """A codebook key typed like the column, or None when it cannot be (e.g. 'n/a' on Int64)."""
    if not dtype.is_numeric():
        return code
    if _INT_CODE.match(code):
        return int(code)
    if _FLOAT_CODE.match(code):
        value = float(code)
        return int(value) if value.is_integer() else value
    return None


def column_codebook(column: str, meta: ResolvedColumnMetadata | None) -> dict[str, str]:
    """Header codebook, overridden per code by dictionary/accepted-patch codebook entries."""
    codebook = dict(code_labels(column))
    if meta is not None and meta.codebook:
        codebook.update({str(k): v for k, v in meta.codebook.items()})
    return codebook


def _custom(prop: str, value: Any) -> dict[str, Any]:
    return {"property": prop, "value": value}


def _missing_rule(name: str) -> dict[str, Any]:
    return {
        "id": f"{name}_missing",
        "type": "library",
        "metric": "nullValues",
        "unit": "percent",
        "mustBeLessOrEqualTo": config.MISSINGNESS_WARN_PERCENT,
        "dimension": "completeness",
        "severity": config.SEVERITY_MISSING,
        "description": "Share of empty values (Jakobsen et al. 2017: above ~5% complete-case analysis is doubtful)",
    }


def _id_rules(name: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{name}_not_null",
            "type": "library",
            "metric": "nullValues",
            "mustBe": 0,
            "dimension": "completeness",
            "severity": config.SEVERITY_ID,
            "description": "Every row needs an identifier",
        },
        {
            "id": f"{name}_no_duplicates",
            "type": "library",
            "metric": "duplicateValues",
            "mustBe": 0,
            "dimension": "uniqueness",
            "severity": config.SEVERITY_ID,
            "description": "One row per identifier",
        },
    ]


def _unidentified_id_rules() -> list[dict[str, Any]]:
    """Table-level ID rules naming no column: reported as not run instead of disappearing."""
    return [
        {
            "id": config.ID_NOT_NULL_RULE_ID,
            "type": "library",
            "metric": "nullValues",
            "arguments": {"properties": []},
            "mustBe": 0,
            "dimension": "completeness",
            "severity": config.SEVERITY_ID,
            "description": "Every row needs an identifier; none was identified",
        },
        {
            "id": config.ID_UNIQUE_RULE_ID,
            "type": "library",
            "metric": "duplicateValues",
            "arguments": {"properties": []},
            "mustBe": 0,
            "dimension": "uniqueness",
            "severity": config.SEVERITY_ID,
            "description": "One row per identifier; none was identified",
        },
    ]


def _property(column: str, series: pl.Series, name: str, inputs: ContractInputs) -> dict[str, Any]:
    meta = inputs.resolved.columns.get(column) if inputs.resolved else None
    dtype = series.dtype
    is_id = column == inputs.id_column
    prop: dict[str, Any] = {
        "name": name,
        "physicalName": column,
        "logicalType": logical_type(dtype),
        "physicalType": str(dtype),
        "businessName": (meta.label if meta and meta.label else short_name(column)),
    }
    if meta and meta.description:
        prop["description"] = meta.description
    if meta and meta.aliases:
        prop["synonyms"] = [{"synonym": alias} for alias in meta.aliases]

    custom: list[dict[str, Any]] = []
    if not is_id and series.null_count() < series.len():
        custom.append(_custom("variableKind", classify_variable(series)))
    if meta and meta.unit:
        custom.append(_custom("unit", meta.unit))
    if meta and meta.semantic_type:
        custom.append(_custom("clinicalSemanticType", meta.semantic_type.value))

    quality: list[dict[str, Any]] = []
    if is_id:
        prop.update({"primaryKey": True, "primaryKeyPosition": 1, "required": True, "unique": True})
        prop["criticalDataElement"] = True
        quality.extend(_id_rules(name))
    else:
        quality.append(_missing_rule(name))
    if is_id or (meta and meta.is_phi):
        prop["classification"] = "restricted"
        prop["tags"] = ["phi"]

    codebook = column_codebook(column, meta)
    typed = [(_typed_code(code, dtype), label) for code, label in codebook.items()]
    enum = [{"value": value, "label": label} for value, label in typed if value is not None]
    if enum and not is_id and dtype != pl.Boolean:
        prop["enum"] = enum
        if not any(cp["property"] == "variableKind" for cp in custom):
            custom.append(_custom("variableKind", "categorical"))
        quality.append(
            {
                "id": f"{name}_invalid",
                "type": "library",
                "metric": "invalidValues",
                "arguments": {"validValues": [e["value"] for e in enum]},
                "mustBe": 0,
                "dimension": "conformity",
                "severity": config.SEVERITY_CODEBOOK,
                "description": "Values must be codes the codebook documents",
            }
        )
        if code_labels(column):
            prop["transformDescription"] = "Codebook decoded from the column header"

    bounds = find_plausibility_range(column)
    if bounds is not None and not is_id:
        limits = {"minimum": bounds.minimum, "maximum": bounds.maximum}
        if dtype.is_numeric():
            prop["logicalTypeOptions"] = limits
        else:
            # Kept (not dropped) so the check shows as not run: "column is not numeric".
            custom.append(_custom(config.PLAUSIBILITY_RANGE_PROPERTY, limits))
        custom.append(_custom("plausibilitySource", bounds.source))
        if not (meta and meta.unit):
            custom.append(_custom("assumedUnit", bounds.unit))

    if custom:
        prop["customProperties"] = custom
    prop["quality"] = quality
    return prop


def build_contract(df: pl.DataFrame, inputs: ContractInputs) -> OpenDataContractStandard:
    """Contract for the unified (analysed) table of one upload version.

    Args:
        df: The frame as stored (ID columns already cast to text).
        inputs: Identity, identifier column and resolved metadata.

    Raises:
        ValueError: ``id_column`` is not a column of ``df``.
        ContractValidationError: the result does not fit the ODCS model (a bug here).
    """
    if inputs.id_column is not None and inputs.id_column not in df.columns:
        raise ValueError(f"Identifier column '{inputs.id_column}' not found")
    names = unique_names(df.columns)
    created = (inputs.created_at or datetime.now(UTC)).astimezone(UTC)
    custom = [
        _custom("uploadId", inputs.upload_id),
        _custom("datasetVersion", inputs.dataset_version),
        _custom("metadataPatchIds", list(inputs.accepted_patch_ids)),
        _custom("minCellSize", config.MIN_CELL_SIZE),
        _custom("generator", config.GENERATOR),
    ]
    if inputs.source_file_sha256:
        custom.append(_custom("sourceFileSha256", inputs.source_file_sha256))
    custom.extend(_custom(k, v) for k, v in inputs.extra_custom_properties.items())

    document: dict[str, Any] = {
        "apiVersion": config.ODCS_API_VERSION,
        "kind": config.ODCS_KIND,
        "id": inputs.upload_id,
        "name": inputs.dataset_name,
        "version": config.INITIAL_CONTRACT_VERSION,
        "status": "active",
        "tags": list(inputs.tags),
        "contractCreatedTs": created.isoformat().replace("+00:00", "Z"),
        "description": {
            "purpose": "Expectations for an uploaded dataset, checked locally at upload.",
            "usage": "Local exploration only. Generated by the app; never leaves the data home.",
            "limitations": "Plausibility bounds catch entry errors; they are not clinical reference ranges.",
        },
        "schema": [
            {
                "name": config.UNIFIED_OBJECT_NAME,
                "physicalName": config.UNIFIED_OBJECT_NAME,
                "logicalType": "object",
                "dataGranularityDescription": "One row per record as uploaded",
                "quality": [
                    *([] if inputs.id_column else _unidentified_id_rules()),
                    {
                        "id": "row_count_min",
                        "type": "library",
                        "metric": "rowCount",
                        "mustBeGreaterOrEqualTo": config.MIN_ROW_COUNT,
                        "dimension": "completeness",
                        "severity": config.SEVERITY_ROW_COUNT,
                        "description": "Below the small-cell threshold every count is suppressed",
                    },
                ],
                "properties": [_property(col, df.get_column(col), names[col], inputs) for col in df.columns],
            }
        ],
        "customProperties": custom,
    }
    if not document["tags"]:
        del document["tags"]
    try:
        return OpenDataContractStandard.model_validate(document)
    except pydantic.ValidationError as e:
        raise ContractValidationError(f"Generated contract is not valid ODCS: {e.error_count()} errors") from e

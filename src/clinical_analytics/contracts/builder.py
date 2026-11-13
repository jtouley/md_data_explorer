# SPDX-License-Identifier: MIT
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

import dataclasses
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pydantic
from open_data_contract_standard.model import OpenDataContractStandard

from clinical_analytics.analysis.numeric_text import is_numeric_text
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
    # Columns the upload names as analysis variables: only their missingness is flagged (L1).
    analysis_columns: tuple[str, ...] = ()
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
    """'Age (years)' and 'age_years' -> 'ageyears'; codebooks in headers are dropped first. The
    micro sign (either code point) reads as "u", so 'Creatinine (µmol/L)' -> 'creatinineumoll'."""
    lowered = short_name(column).lower().replace("\u00b5", "u").replace("\u03bc", "u")
    return re.sub(r"[^a-z0-9]", "", lowered)


def _unit_hints(variable: str) -> tuple[str, ...]:
    """The lab unit suffixes, plus those of the variable's unit pair ('tempf', 'temperaturec')."""
    pair = next((p for p in config.UNIT_PAIRS if p.variable == variable), None)
    return config.UNIT_HINTS if pair is None else (*config.UNIT_HINTS, *pair.low_hints, *pair.high_hints)


def _hinted(key: str, aliases: frozenset[str], hints: tuple[str, ...]) -> bool:
    """An alias followed by a unit in the header: 'Na (mmol/L)' -> 'nammoll', 'Glucose (mg/dL)'
    -> 'glucosemgdl', 'Temp (°F)' -> 'tempf'."""
    return any(key == alias + unit for alias in aliases for unit in hints)


def find_plausibility_range(column: str) -> config.PlausibilityRange | None:
    """The configured range for ``column`` by its normalised short name, or None.

    Any alias also matches with a unit after it, as lab exports write headers ("Glucose (mg/dL)",
    "creatinine umol/L", "Temp (°F)"). A short alias ("na", "k", "tc") matches only with a unit
    in the header: a column literally named "na" or "tc" (not applicable, treatment code) gets no
    lab range.
    """
    key = plausibility_key(column)
    return next(
        (
            r
            for r in config.PLAUSIBILITY_RANGES
            if key in r.aliases or _hinted(key, r.aliases | r.hinted_aliases, _unit_hints(r.variable))
        ),
        None,
    )


def range_for(column: str, median: float | None) -> config.PlausibilityRange | None:
    """``find_plausibility_range`` with the unit read from the median where the range has
    per-unit bounds (temperature: °C or °F). Median unknown: the wide bound, its unit text
    saying so ("unit unknown, wide bound")."""
    bounds = find_plausibility_range(column)
    if bounds is None or not bounds.by_median:
        return bounds
    if median is None:
        return dataclasses.replace(bounds, unit=f"{bounds.unit} ({config.WIDE_BOUND_NOTE})", by_median=())
    unit = next(u for u in bounds.by_median if u.median_below is None or median < u.median_below)
    return dataclasses.replace(bounds, minimum=unit.minimum, maximum=unit.maximum, unit=unit.unit, by_median=())


def range_text(bounds: config.PlausibilityRange) -> str:
    """'0–122 years', or 'below 0 concentration (unit unknown)' with no upper bound."""
    if bounds.maximum is None:
        return f"below {bounds.minimum:g} {bounds.unit}"
    return f"{bounds.minimum:g}–{bounds.maximum:g} {bounds.unit}"


@dataclass(frozen=True)
class Implausible:
    """Values of a column outside its plausibility range: the one result every path reports.

    ``present``: values the check read (non-missing, finite numbers), the line the small-cell
    rule applies to.
    """

    column: str
    count: int
    present: int
    bounds: str

    def disclosure(self) -> dict[str, Any]:
        """The analysis-result shape the presenter reads (``implausible``)."""
        return {"column": self.column, "count": self.count, "line_total": self.present, "bounds": self.bounds}


def _weighted_median(values: pl.Series, weights: pl.Series) -> float | None:
    frame = pl.DataFrame({"v": values, "w": weights}).drop_nulls().sort("v")
    if frame.height == 0:
        return None
    half = frame.get_column("w").sum() / 2
    return float(frame.filter(pl.col("w").cum_sum() >= half).get_column("v")[0])


def numbers_with_counts(series: pl.Series, weights: pl.Series | None = None) -> pl.DataFrame | None:
    """``series`` as finite numbers ``v`` with their row counts ``w`` (nulls dropped), or None when
    it is not numbers. Numbers stored as text are read with ``parse_numeric_text`` first, so text
    glucose "-5" is read as the float -5 is. ``weights``: the count of each value when ``series``
    holds distinct values (a DuckDB value count), so no full column is copied.
    """
    from clinical_analytics.analysis.numeric_text import AmbiguousNumberFormatError, parse_numeric_text

    values = series
    if series.dtype == pl.String:
        try:
            # Weighted like statistic_refusal: the near-numeric share counts rows, not distinct values.
            parsed = parse_numeric_text(series, weights)
        except AmbiguousNumberFormatError:
            return None  # the answer refuses on the format; there are no numbers to check
        if parsed is None:
            return None
        values = parsed.values
    elif not series.dtype.is_numeric() or series.dtype == pl.Boolean:
        return None
    counts = weights if weights is not None else pl.Series([1] * values.len(), dtype=pl.Int64)
    frame = pl.DataFrame({"v": values.cast(pl.Float64), "w": counts.cast(pl.Int64)}).drop_nulls()
    return frame.filter(pl.col("v").is_finite())


def implausible_values(series: pl.Series, weights: pl.Series | None = None) -> Implausible | None:
    """Values of ``series`` outside the column's range, or None (no range, none outside, or not
    numbers). Read with ``numbers_with_counts`` (numbers stored as text included).
    """
    if find_plausibility_range(series.name) is None:
        return None
    frame = numbers_with_counts(series, weights)
    if frame is None:
        return None
    bounds = range_for(series.name, _weighted_median(frame.get_column("v"), frame.get_column("w")))
    if bounds is None:
        return None
    outside = pl.col("v") < bounds.minimum
    if bounds.maximum is not None:
        outside = outside | (pl.col("v") > bounds.maximum)
    count = int(frame.filter(outside).get_column("w").sum())
    if count == 0:
        return None
    return Implausible(
        column=series.name, count=count, present=int(frame.get_column("w").sum()), bounds=range_text(bounds)
    )


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


def _missing_rule(name: str, flagged: bool) -> dict[str, Any]:
    """Missingness of a column: flagged above 5% for an analysis variable, else measured only."""
    if flagged:
        operator, limit, description = (
            "mustBeLessOrEqualTo",
            config.MISSINGNESS_WARN_PERCENT,
            "Share of empty values (Jakobsen et al. 2017: above ~5% complete-case analysis is doubtful)",
        )
    else:
        operator, limit, description = (
            "mustBeLessThan",
            config.MISSINGNESS_MEASURE_ONLY_PERCENT,
            "Share of empty values, listed; flagged only when every value is empty or for analysis variables",
        )
    return {
        "id": f"{name}_missing",
        "type": "library",
        "metric": "nullValues",
        "unit": "percent",
        operator: limit,
        "dimension": "completeness",
        "severity": config.SEVERITY_MISSING,
        "description": description,
    }


def _censored_rule(name: str) -> dict[str, Any]:
    """Values recorded as a limit ("<5") in a column of numbers stored as text."""
    return {
        "id": f"{name}{config.CENSORED_RULE_SUFFIX}",
        "type": "library",
        "metric": "censoredValues",
        "mustBe": 0,
        "dimension": "accuracy",
        "severity": config.SEVERITY_RANGE,
        "description": "Values recorded as a limit (like '<5') are not numbers; answers refuse or disclose them",
    }


def _id_rules(name: str) -> list[dict[str, Any]]:
    """Identifier rules. The exact duplicate check always runs; a repeat is reported, and answers
    ask whether rows are visits before counting patients. Wording and severity do not depend on
    the data, so a tiny table's report (every count hidden) does not tell whether IDs repeat."""
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
            "description": "One row per identifier; repeats mean rows may be visits, confirmed before counting",
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
        present = series.drop_nulls()
        repeated = present.n_unique() < present.len()  # exact, every row
        # A repeated identifier is not a key: the contract says one row per identifier only
        # when that is true of the data (patient grain), never for visit-level rows. A tiny
        # table states neither (its report hides every count, so the contract must not tell).
        prop["required"] = True
        if config.is_tiny_table(series.len()):
            pass
        elif repeated:
            prop["unique"] = False
        else:
            prop.update({"primaryKey": True, "primaryKeyPosition": 1, "unique": True})
        prop["criticalDataElement"] = True
        quality.extend(_id_rules(name))
    else:
        quality.append(_missing_rule(name, column in inputs.analysis_columns))
        if dtype == pl.String and is_numeric_text(series):
            quality.append(_censored_rule(name))
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

    median = series.median() if dtype.is_numeric() and dtype != pl.Boolean else None
    bounds = range_for(column, float(median) if isinstance(median, int | float) else None)
    # Codes are not measurements ("Age 1: young 2: old"): a coded column gets no range check.
    if bounds is None and not is_id and "enum" not in prop and dtype.is_numeric() and dtype != pl.Boolean:
        if series.null_count() < series.len() and classify_variable(series) == "continuous":
            custom.append(_custom(config.PLAUSIBILITY_RANGE_PROPERTY, config.NO_RANGE_KNOWN))
    if bounds is not None and not is_id and "enum" not in prop:
        limits = {"minimum": bounds.minimum}
        if bounds.maximum is not None:
            limits["maximum"] = bounds.maximum
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


def _granularity(df: pl.DataFrame, id_column: str | None) -> str:
    """The grain the data shows: one row per identifier, or several (rows may be visits)."""
    if id_column is None:
        return "One row per record as uploaded; no identifier column"
    if config.is_tiny_table(df.height):
        return "One row per record as uploaded; too few rows to state the grain"
    present = df.get_column(id_column).drop_nulls()
    if present.n_unique() < present.len():
        return f"Several rows per {id_column} (rows may be visits); not one row per patient"
    return f"One row per {id_column}"


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
                "dataGranularityDescription": _granularity(df, inputs.id_column),
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

# SPDX-License-Identifier: MIT
"""Compile a data contract's rules to Polars expressions and run them in one collect.

Supported (ODCS v3.2.0 ``library`` metrics plus type options):

* ``nullValues``      -> ``null_count()`` (count, or percent of rows with ``unit: percent``)
* ``duplicateValues`` -> surplus rows: non-null rows minus distinct non-null values (property
  level), or rows minus distinct combinations of ``arguments.properties`` (object level)
* ``invalidValues``   -> non-null values not in ``arguments.validValues``
* ``rowCount``        -> ``pl.len()``
* ``intervalOrder``   -> rows where ``arguments.properties`` = [lower, value, upper] are all
  present and ``lower <= value <= upper`` does not hold (table level)
* ``logicalTypeOptions.minimum/maximum`` on a numeric column -> non-null values outside the bounds

Anything else (``sql``, ``text``, ``custom`` rules, ``missingValues``) is reported as ``not_run``
with the reason, never skipped silently. NaN in a float column counts as missing, never as out of
range.

Every count is reported as measured, small ones included (ADR015): the report is shown on this
machine, to the person who holds the rows. Reports written before ADR015 may still carry a
"hidden" count with status "review"; they are read as they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, TypeAlias

import polars as pl
import structlog
from open_data_contract_standard.model import DataQuality, OpenDataContractStandard, SchemaProperty
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.analysis.numeric_text import censored_expr, missing_token_expr
from clinical_analytics.contracts import config
from clinical_analytics.contracts.errors import QualityCheckError

logger = structlog.get_logger()

CheckStatus: TypeAlias = Literal["passed", "warning", "failed", "review", "not_run"]
ROWS_KEY = "__rows"
RANGE_METRIC = "range"
CENSORED_METRIC = "censoredValues"
TYPE_METRIC = "logicalType"
LEVEL_VARIANTS_METRIC = "levelVariants"
INTERVAL_METRIC = "intervalOrder"
_OPERATORS = (
    "mustBe",
    "mustNotBe",
    "mustBeGreaterThan",
    "mustBeGreaterOrEqualTo",
    "mustBeLessThan",
    "mustBeLessOrEqualTo",
    "mustBeBetween",
    "mustNotBeBetween",
)


class QualityCheckResult(BaseModel):
    """One check, ready to show, with its count as measured."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., description="Rule id from the contract")
    column: str | None = Field(None, description="Display name of the column (None for table-level checks)")
    metric: str = Field(..., description="ODCS metric, or 'range' for plausibility bounds")
    dimension: str | None = Field(None, description="ODCS data-quality dimension")
    severity: str = Field(..., description="error or warning")
    status: CheckStatus
    count: int | str | None = Field(
        None, description="Rows affected; None if not run ('hidden' only in reports written before ADR015)"
    )
    percent: float | None = Field(None, description="Share of rows (1 decimal)")
    message: str = Field(..., description="Plain-language result")


class QualityReport(BaseModel):
    """Result of checking one dataset version against its contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ran"] = "ran"
    upload_id: str
    dataset_version: str
    contract_version: str
    run_ts: str = Field(..., description="UTC run time, ISO 8601")
    row_count: int | str = Field(..., description="Rows checked ('hidden' only in reports written before ADR015)")
    passed: int
    warnings: int
    failed: int
    review: int = Field(0, description="Checks with status 'review' (only in reports written before ADR015)")
    not_run: int
    summary: list[str] = Field(default_factory=list, description="Short plain-language lines for the UI")
    checks: list[QualityCheckResult] = Field(default_factory=list)


class QualityNotRun(BaseModel):
    """The check could not run for this upload; shown instead of a report (never silently absent)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["not_run"] = "not_run"
    reason: str = Field(..., description="Plain-language reason, without file paths or values")
    summary: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class CompiledCheck:
    """A contract rule bound to a Polars expression (or the reason it cannot run)."""

    id: str
    column: str | None
    display: str | None
    metric: str
    dimension: str | None
    severity: str
    rule: dict[str, Any]
    expr: pl.Expr | None
    not_run_reason: str | None = None
    percent: bool = False
    detail: str = ""
    columns: tuple[str, ...] = ()  # physical columns a table-level check reads


def _display(prop: SchemaProperty) -> str:
    return prop.businessName or prop.name or prop.physicalName or "?"


def _rule_dict(rule: DataQuality) -> dict[str, Any]:
    return rule.model_dump(exclude_none=True)


def _coerce_values(values: list[Any], dtype: pl.DataType) -> pl.Series | None:
    """Valid values typed like the column, or None when they cannot be (a contract/data mismatch)."""
    try:
        if dtype == pl.String:
            return pl.Series([str(v) for v in values], dtype=pl.String)
        return pl.Series(values).cast(dtype, strict=True)
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError, TypeError, OverflowError):
        return None


def _not_run(base: dict[str, Any], reason: str) -> CompiledCheck:
    return CompiledCheck(**base, expr=None, not_run_reason=reason)


def _is_numeric_text_property(prop: SchemaProperty) -> bool:
    """The builder marks a column of numbers stored as text with a censored-values rule."""
    return any((q.metric or "") == CENSORED_METRIC for q in prop.quality or [])


def _compile_property_rule(rule: DataQuality, prop: SchemaProperty, schema: pl.Schema, index: int) -> CompiledCheck:
    column = prop.physicalName or prop.name
    metric = rule.metric or rule.rule or rule.type or "unknown"
    base: dict[str, Any] = {
        "id": rule.id or f"{prop.name}_{metric}_{index}",
        "column": column,
        "display": _display(prop),
        "metric": metric,
        "dimension": rule.dimension,
        "severity": rule.severity or "error",
        "rule": _rule_dict(rule),
    }
    if (rule.type or "library") != "library":
        return _not_run(base, f"'{rule.type}' rules are not run locally")
    if column is None or column not in schema:
        return _not_run(base, "column not in the data")
    col = pl.col(column)
    if metric == "nullValues":
        # In a column of numbers stored as text (the contract gave it a censored rule), the
        # missing-value tokens the analysis treats as missing ("n/a", ".") count as missing, so
        # the report and the answers agree. In a categorical column "None" is a real level.
        if schema[column].is_float():
            missing = (col.is_null() | col.is_nan()).sum()
        elif schema[column] == pl.String and _is_numeric_text_property(prop):
            missing = (col.is_null() | missing_token_expr(column)).sum()
        else:
            missing = col.null_count()
        return CompiledCheck(**base, expr=missing, percent=rule.unit == "percent")
    if metric == CENSORED_METRIC:
        if schema[column] != pl.String:
            return _not_run(base, "column is not stored as text")
        return CompiledCheck(**base, expr=censored_expr(column).fill_null(False).sum())
    if metric == "duplicateValues":
        return CompiledCheck(**base, expr=col.drop_nulls().len() - col.drop_nulls().n_unique())
    if metric == "invalidValues":
        valid = (rule.arguments or {}).get("validValues")
        if not isinstance(valid, list) or not valid:
            return _not_run(base, "no validValues given")
        typed = _coerce_values(valid, schema[column])
        if typed is None:
            return _not_run(base, "codebook values do not match the column type")
        detail = ", ".join(str(v) for v in valid)
        present = col.is_not_null() & col.is_not_nan() if schema[column].is_float() else col.is_not_null()
        return CompiledCheck(**base, expr=(present & ~col.is_in(typed.implode())).sum(), detail=detail)
    return _not_run(base, f"metric '{metric}' is not supported")


def _compile_range(prop: SchemaProperty, schema: pl.Schema) -> CompiledCheck | None:
    options = prop.logicalTypeOptions or {}
    custom = next(
        (c.value for c in prop.customProperties or [] if c.property == config.PLAUSIBILITY_RANGE_PROPERTY), None
    )
    if "minimum" not in options and "maximum" not in options and isinstance(custom, dict):
        options = custom
    lo, hi = options.get("minimum"), options.get("maximum")
    if lo is None and hi is None:
        return None
    column = prop.physicalName or prop.name
    unit = next((c.value for c in prop.customProperties or [] if c.property in ("unit", "assumedUnit")), "")
    base: dict[str, Any] = {
        "id": f"{prop.name}_range",
        "column": column,
        "display": _display(prop),
        "metric": RANGE_METRIC,
        "dimension": "accuracy",
        "severity": config.SEVERITY_RANGE,
        "rule": {"mustBe": 0, "minimum": lo, "maximum": hi},
        "detail": f"{lo if lo is not None else '-inf'}–{hi if hi is not None else 'inf'} {unit}".strip(),
    }
    if column is None or column not in schema:
        return _not_run(base, "column not in the data")
    if not schema[column].is_numeric():
        return _not_run(base, "column is not numeric")
    col = pl.col(column)
    outside = (col < lo) if hi is None else (col > hi) if lo is None else ~col.is_between(lo, hi)
    present = col.is_not_null() & col.is_not_nan() if schema[column].is_float() else col.is_not_null()
    return CompiledCheck(**base, expr=(present & outside).sum())


def _present(column: str, schema: pl.Schema) -> pl.Expr:
    """A value the check reads: not null, and not NaN in a float column."""
    col = pl.col(column)
    return col.is_not_null() & col.is_not_nan() if schema[column].is_float() else col.is_not_null()


def _compile_interval(
    base: dict[str, Any], names: list[str], props: dict[str, SchemaProperty], schema: pl.Schema
) -> CompiledCheck:
    """Rows whose value lies outside its own interval: lower <= value <= upper, all three present."""
    missing = [n for n in names if n not in props]
    if len(names) != 3 or missing:
        return _not_run(base, "interval check names no known columns")
    lower, value, upper = (props[n].physicalName or n for n in names)
    shown = {**base, "display": _display(props[names[1]]), "detail": f"{lower} ≤ {value} ≤ {upper}"}
    if any(c not in schema for c in (lower, value, upper)):
        return _not_run(shown, "column not in the data")
    if not all(schema[c].is_numeric() for c in (lower, value, upper)):
        return _not_run(shown, "column is not numeric")
    present = _present(lower, schema) & _present(value, schema) & _present(upper, schema)
    outside = (pl.col(lower) > pl.col(value)) | (pl.col(value) > pl.col(upper))
    return CompiledCheck(**shown, expr=(present & outside).sum(), columns=(lower, value, upper))


def _compile_object_rule(
    rule: DataQuality, props: dict[str, SchemaProperty], index: int, schema: pl.Schema
) -> CompiledCheck:
    metric = rule.metric or rule.rule or rule.type or "unknown"
    base: dict[str, Any] = {
        "id": rule.id or f"table_{metric}_{index}",
        "column": None,
        "display": None,
        "metric": metric,
        "dimension": rule.dimension,
        "severity": rule.severity or "error",
        "rule": _rule_dict(rule),
    }
    if (rule.type or "library") != "library":
        return _not_run(base, f"'{rule.type}' rules are not run locally")
    if base["id"] in (config.ID_NOT_NULL_RULE_ID, config.ID_UNIQUE_RULE_ID) and not (rule.arguments or {}).get(
        "properties"
    ):
        return _not_run({**base, "display": config.ID_RULE_DISPLAY}, config.NO_ID_COLUMN_REASON)
    if metric == "rowCount":
        return CompiledCheck(**base, expr=pl.len())
    if metric == "duplicateValues":
        names = (rule.arguments or {}).get("properties") or []
        missing = [n for n in names if n not in props]
        if not names or missing:
            return _not_run(base, "duplicate check names no known columns")
        cols = [props[n].physicalName or n for n in names]
        return CompiledCheck(**base, expr=pl.len() - pl.struct(cols).n_unique(), columns=tuple(cols))
    if metric == INTERVAL_METRIC:
        return _compile_interval(base, list((rule.arguments or {}).get("properties") or []), props, schema)
    return _not_run(base, f"metric '{metric}' is not supported at table level")


def compile_contract(contract: OpenDataContractStandard, schema: pl.Schema) -> list[CompiledCheck]:
    """Every rule in the contract's first schema object, compiled against the data's schema.

    Raises:
        QualityCheckError: the contract has no schema object, or rule ids repeat.
    """
    if not contract.schema_:
        raise QualityCheckError("Contract has no schema object")
    obj = contract.schema_[0]
    props = {p.name: p for p in obj.properties or [] if p.name}
    checks: list[CompiledCheck] = [_compile_object_rule(r, props, i, schema) for i, r in enumerate(obj.quality or [])]
    for prop in obj.properties or []:
        checks.extend(_compile_property_rule(r, prop, schema, i) for i, r in enumerate(prop.quality or []))
        range_check = _compile_range(prop, schema)
        if range_check is not None:
            checks.append(range_check)
    ids = [c.id for c in checks]
    if len(ids) != len(set(ids)) or ROWS_KEY in ids:
        raise QualityCheckError("Contract rule ids are not unique")
    return checks


def measure(lf: pl.LazyFrame, checks: list[CompiledCheck]) -> dict[str, int]:
    """Raw counts for every runnable check plus the row count, from a single collect."""
    exprs = [c.expr.alias(c.id) for c in checks if c.expr is not None]
    row = lf.select(*exprs, pl.len().alias(ROWS_KEY)).collect().row(0, named=True)
    return {k: int(v) for k, v in row.items()}


def _passes(rule: dict[str, Any], observed: float) -> bool:
    op = next((o for o in _OPERATORS if o in rule), None)
    if op is None:
        raise QualityCheckError(f"Rule '{rule.get('id', '?')}' has no comparison operator")
    target: Any = rule[op]
    result: bool
    if op == "mustBe":
        result = observed == target
    elif op == "mustNotBe":
        result = observed != target
    elif op == "mustBeGreaterThan":
        result = observed > target
    elif op == "mustBeGreaterOrEqualTo":
        result = observed >= target
    elif op == "mustBeLessThan":
        result = observed < target
    elif op == "mustBeLessOrEqualTo":
        result = observed <= target
    else:
        lo, hi = target
        inside = bool(lo <= observed <= hi)
        result = inside if op == "mustBeBetween" else not inside
    return bool(result)


def _message(check: CompiledCheck, count: int, percent: float | None) -> str:
    name = check.display or "Table"
    if check.metric == "rowCount":
        return f"{count} rows"
    if count == 0:
        return f"{name}: no problems found"
    if check.metric == "nullValues":
        if check.percent and percent is not None:
            return f"{name}: {percent:.1f}% missing"
        return f"{name}: {count} values missing" if check.percent else f"{name}: {count} rows have no value"
    if check.metric == "duplicateValues":
        return f"{name}: {count} repeated values"
    if check.metric == "invalidValues":
        return f"{name}: {count} values are not in the codebook ({check.detail})"
    if check.metric == RANGE_METRIC:
        return f"{name}: {count} values outside {check.detail}"
    if check.metric == CENSORED_METRIC:
        return f"{name}: {count} values recorded as a limit (like '<5'), not as numbers"
    if check.metric == INTERVAL_METRIC:
        return f"{name}: {count} rows where {check.detail} does not hold"
    return f"{name}: {count} rows flagged"


def _result(check: CompiledCheck, raw: dict[str, int]) -> QualityCheckResult:
    common: dict[str, Any] = {
        "id": check.id,
        "column": check.display,
        "metric": check.metric,
        "dimension": check.dimension,
        "severity": check.severity,
    }
    if check.expr is None:
        return QualityCheckResult(
            **common, status="not_run", message=f"{check.display or 'Table'}: not checked ({check.not_run_reason})"
        )
    count, rows = raw[check.id], raw[ROWS_KEY]
    observed: float = (100.0 * count / rows if rows else 0.0) if check.percent else count
    passed = _passes(check.rule, observed)
    percent = None if check.metric == "rowCount" or not rows else round(100.0 * count / rows, 1)
    status: CheckStatus = "passed" if passed else ("failed" if check.severity == "error" else "warning")
    return QualityCheckResult(
        **common, status=status, count=count, percent=percent, message=_message(check, count, percent)
    )


def _summary(results: list[QualityCheckResult]) -> list[str]:
    """Failed, warnings, then any other missing values; capped for the upload notice."""
    order = {"failed": 0, "warning": 1, "review": 2, "passed": 3, "not_run": 4}
    notable = [
        r
        for r in results
        if not _deferred(r)
        and (
            r.status in ("failed", "warning", "review", "not_run")
            or (r.metric == "nullValues" and r.count not in (0, None))
        )
    ]
    notable.sort(key=lambda r: order[r.status])
    lines = [r.message for r in notable]
    if len(lines) > config.SUMMARY_MAX_LINES:
        extra = len(lines) - config.SUMMARY_MAX_LINES + 1
        lines = lines[: config.SUMMARY_MAX_LINES - 1] + [f"…and {extra} more"]
    return lines


def _custom_value(contract: OpenDataContractStandard, prop: str) -> Any:
    return next((c.value for c in contract.customProperties or [] if c.property == prop), None)


def type_results(
    contract: OpenDataContractStandard, schema: pl.Schema, data_version: str | None = None
) -> list[QualityCheckResult]:
    """Type conformance: each property's ``logicalType`` against the data's type (schema only,
    no counts). A column the contract expects as a number but the data holds as text is how a
    range check stops running without anyone noticing (DQ M5).

    The check has evidence only when the data is another version than the one the contract was
    built from (version N+1 against N's contract). Checked against its own version, the contract's
    types were read off this data, so the check cannot fail: it is reported as not run, never as
    a pass. ``data_version`` None: the data is the contract's own version.
    """
    from clinical_analytics.contracts.builder import logical_type

    if not contract.schema_:
        return []
    built_on = str(_custom_value(contract, "datasetVersion") or "")
    if data_version is None or data_version == built_on:
        # One line, not one per column: nothing was compared, so nothing passed.
        return [
            QualityCheckResult(
                id=f"table{config.TYPE_RULE_SUFFIX}",
                column=None,
                metric=TYPE_METRIC,
                dimension="conformity",
                severity="error",
                status="not_run",
                count=None,
                percent=None,
                message=(
                    "Column types: not checked (the contract was built from this version; they are "
                    "checked when a new version is uploaded)"
                ),
            )
        ]
    results = []
    for prop in contract.schema_[0].properties or []:
        column = prop.physicalName or prop.name
        if column is None or prop.logicalType is None:
            continue
        common: dict[str, Any] = {
            "id": f"{prop.name}{config.TYPE_RULE_SUFFIX}",
            "column": _display(prop),
            "metric": TYPE_METRIC,
            "dimension": "conformity",
            "severity": "error",
            "count": None,
            "percent": None,
        }
        if column not in schema:
            results.append(
                QualityCheckResult(
                    **common, status="not_run", message=f"{_display(prop)}: not checked (column not in the data)"
                )
            )
            continue
        actual = logical_type(schema[column])
        if actual == prop.logicalType:
            results.append(
                QualityCheckResult(**common, status="passed", message=f"{_display(prop)}: stored as {actual}")
            )
        else:
            results.append(
                QualityCheckResult(
                    **common,
                    status="failed",
                    message=f"{_display(prop)}: stored as {actual}, but the contract expects {prop.logicalType}",
                )
            )
    return results


def is_id_property(prop: SchemaProperty) -> bool:
    """The identifier column (also governed SQL's, ``sql/catalog.py``): the builder marks it
    critical and required (also for tiny tables, where it states neither uniqueness nor a key)."""
    return bool(prop.primaryKey or prop.criticalDataElement or prop.required)


def is_restricted_property(prop: SchemaProperty) -> bool:
    """PHI or otherwise restricted (also governed SQL's rule for coarsen-only columns)."""
    return prop.classification == "restricted" or "phi" in (prop.tags or [])


def level_variant_results(lf: pl.LazyFrame, contract: OpenDataContractStandard) -> list[QualityCheckResult]:
    """Text columns whose levels include spellings of one category (Male / male / M).

    A warning at upload; a question that groups or limits on such a level is held until the
    doctor confirms the spellings are one. The message names up to four of the spellings.
    """
    from clinical_analytics.analysis.grouping import level_variants

    if not contract.schema_:
        return []
    schema = lf.collect_schema()
    props = {p.physicalName or p.name: p for p in contract.schema_[0].properties or []}
    # Never scanned: the ID column (marked by role, whatever its uniqueness) and restricted/PHI
    # columns, whose values the report must not echo (a clinician or ward name).
    skipped = {column for column, p in props.items() if is_id_property(p) or is_restricted_property(p)}
    text = [c for c, dtype in schema.items() if dtype == pl.String and c not in skipped and c in props]
    if not text:
        return []
    # Fixed field names ("value", "__n"), so no column name ("count", "__n") collides with them.
    levels = lf.select(
        [pl.col(c).drop_nulls().alias("value").value_counts(name="__n").implode().alias(c) for c in text]
    ).collect()
    results = []
    for column in text:
        counts = {row["value"]: row["__n"] for row in levels.get_column(column)[0].to_list()}
        groups = level_variants(list(counts))
        if not groups:
            continue
        prop = props[column]
        named = [s for group in groups for s in group]
        example = f" (for example {', '.join(named[:4])})"
        results.append(
            QualityCheckResult(
                id=f"{prop.name}{config.LEVEL_VARIANTS_RULE_SUFFIX}",
                column=_display(prop),
                metric=LEVEL_VARIANTS_METRIC,
                dimension="consistency",
                severity="warning",
                status="warning",
                count=None,
                percent=None,
                message=(
                    f"{_display(prop)}: some values may be one category written differently{example}; "
                    "questions on them are held until you confirm"
                ),
            )
        )
    return results


def run_quality(
    lf: pl.LazyFrame,
    contract: OpenDataContractStandard,
    run_at: datetime | None = None,
    data_version: str | None = None,
    previous_contract: OpenDataContractStandard | None = None,
) -> QualityReport:
    """Check ``lf`` against ``contract`` and build the report: the counted checks in one
    collect, then type conformance (schema only) and category spellings (one more collect).

    ``data_version``: the version ``lf`` holds. ``previous_contract``: the contract of the
    dataset's previous version, which the column types are checked against (version N+1 against
    N's contract); without one, the types are checked against ``contract`` only when it is
    another version's, else reported as not run (see ``type_results``).
    """
    schema = lf.collect_schema()
    checks = compile_contract(contract, schema)
    typed_by = previous_contract if previous_contract is not None else contract
    extra = [*type_results(typed_by, schema, data_version), *level_variant_results(lf, contract)]
    report = report_from_counts(contract, checks, measure(lf, checks), run_at=run_at, extra=extra)
    logger.info(
        "quality_run_completed",
        upload_id=report.upload_id,
        checks=len(report.checks),
        failed=report.failed,
        warnings=report.warnings,
        not_run=report.not_run,
    )
    return report


def report_from_counts(
    contract: OpenDataContractStandard,
    checks: list[CompiledCheck],
    raw: dict[str, int],
    run_at: datetime | None = None,
    extra: list[QualityCheckResult] | None = None,
) -> QualityReport:
    """The report for raw counts (``measure`` output). Pure: no data is read.

    ``extra``: results that carry no count (type conformance, category spellings), added as they are.
    """
    rows = raw[ROWS_KEY]
    results = [_result(c, raw) for c in checks] + list(extra or [])
    ts = (run_at or datetime.now(UTC)).astimezone(UTC)
    report = QualityReport(
        upload_id=str(_custom_value(contract, "uploadId") or contract.id),
        dataset_version=str(_custom_value(contract, "datasetVersion") or ""),
        contract_version=contract.version or "",
        run_ts=ts.isoformat().replace("+00:00", "Z"),
        row_count=rows,
        passed=sum(r.status == "passed" for r in results),
        warnings=sum(r.status == "warning" for r in results),
        failed=sum(r.status == "failed" for r in results),
        review=sum(r.status == "review" for r in results),
        not_run=sum(r.status == "not_run" for r in results),
        summary=_summary(results),
        checks=results,
    )
    return report


def _deferred(result: QualityCheckResult) -> bool:
    """The type check of a contract's own version: not run (and counted so), but not a finding
    for the upload notice, since it can only run on the next version."""
    return result.metric == TYPE_METRIC and result.status == "not_run" and result.column is None

"""Compile a data contract's rules to Polars expressions and run them in one collect.

Supported (ODCS v3.2.0 ``library`` metrics plus type options):

* ``nullValues``      -> ``null_count()`` (count, or percent of rows with ``unit: percent``)
* ``duplicateValues`` -> surplus rows: non-null rows minus distinct non-null values (property
  level), or rows minus distinct combinations of ``arguments.properties`` (object level)
* ``invalidValues``   -> non-null values not in ``arguments.validValues``
* ``rowCount``        -> ``pl.len()``
* ``logicalTypeOptions.minimum/maximum`` on a numeric column -> non-null values outside the bounds

Anything else (``sql``, ``text``, ``custom`` rules, ``missingValues``) is reported as ``not_run``
with the reason, never skipped silently. Reports carry counts already suppressed (1..10 -> "<11"),
so the file on disk and the API response are safe to show as they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, TypeAlias

import polars as pl
import structlog
from open_data_contract_standard.model import DataQuality, OpenDataContractStandard, SchemaProperty
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.contracts import config
from clinical_analytics.contracts.errors import QualityCheckError
from clinical_analytics.core.result_presentation import suppress_count

logger = structlog.get_logger()

CheckStatus: TypeAlias = Literal["passed", "warning", "failed", "not_run"]
ROWS_KEY = "__rows"
RANGE_METRIC = "range"
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
    """One check, ready to show: counts are small-cell suppressed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., description="Rule id from the contract")
    column: str | None = Field(None, description="Display name of the column (None for table-level checks)")
    metric: str = Field(..., description="ODCS metric, or 'range' for plausibility bounds")
    dimension: str | None = Field(None, description="ODCS data-quality dimension")
    severity: str = Field(..., description="error or warning")
    status: CheckStatus
    count: int | str | None = Field(None, description="Rows affected; '<11' when 1..10; None if not run")
    percent: float | None = Field(None, description="Share of rows (1 decimal); None when suppressed")
    message: str = Field(..., description="Plain-language result")


class QualityReport(BaseModel):
    """Result of checking one dataset version against its contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    upload_id: str
    dataset_version: str
    contract_version: str
    run_ts: str = Field(..., description="UTC run time, ISO 8601")
    row_count: int | str = Field(..., description="Rows checked (suppressed when 1..10)")
    passed: int
    warnings: int
    failed: int
    not_run: int
    summary: list[str] = Field(default_factory=list, description="Short plain-language lines for the UI")
    checks: list[QualityCheckResult] = Field(default_factory=list)


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
        return CompiledCheck(**base, expr=col.null_count(), percent=rule.unit == "percent")
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
        return CompiledCheck(**base, expr=(col.is_not_null() & ~col.is_in(typed.implode())).sum(), detail=detail)
    return _not_run(base, f"metric '{metric}' is not supported")


def _compile_range(prop: SchemaProperty, schema: pl.Schema) -> CompiledCheck | None:
    options = prop.logicalTypeOptions or {}
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
    return CompiledCheck(**base, expr=(col.is_not_null() & outside).sum())


def _compile_object_rule(rule: DataQuality, props: dict[str, SchemaProperty], index: int) -> CompiledCheck:
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
    if metric == "rowCount":
        return CompiledCheck(**base, expr=pl.len())
    if metric == "duplicateValues":
        names = (rule.arguments or {}).get("properties") or []
        missing = [n for n in names if n not in props]
        if not names or missing:
            return _not_run(base, "duplicate check names no known columns")
        cols = [props[n].physicalName or n for n in names]
        return CompiledCheck(**base, expr=pl.len() - pl.struct(cols).n_unique())
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
    checks: list[CompiledCheck] = [_compile_object_rule(r, props, i) for i, r in enumerate(obj.quality or [])]
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


def _message(check: CompiledCheck, count: int, shown: int | str, percent: float | None) -> str:
    name = check.display or "Table"
    if check.metric == "rowCount":
        return f"{shown} rows" if count >= config.MIN_ROW_COUNT else f"{shown} rows: most counts will be hidden"
    if count == 0:
        return f"{name}: no problems found"
    if check.metric == "nullValues":
        if check.percent:
            return f"{name}: {percent:.1f}% missing" if percent is not None else f"{name}: {shown} values missing"
        return f"{name}: {shown} rows have no value"
    if check.metric == "duplicateValues":
        return f"{name}: {shown} repeated values"
    if check.metric == "invalidValues":
        return f"{name}: {shown} values are not in the codebook ({check.detail})"
    if check.metric == RANGE_METRIC:
        return f"{name}: {shown} values outside {check.detail}"
    return f"{name}: {shown} rows flagged"


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
    shown = suppress_count(count, config.MIN_CELL_SIZE)
    # A percent next to a known row count reveals a small count, so it is hidden with the count.
    percent = round(100.0 * count / rows, 1) if rows and shown == count and check.metric != "rowCount" else None
    status: CheckStatus = "passed" if passed else ("failed" if check.severity == "error" else "warning")
    return QualityCheckResult(
        **common, status=status, count=shown, percent=percent, message=_message(check, count, shown, percent)
    )


def _summary(results: list[QualityCheckResult]) -> list[str]:
    """Failed, then warnings, then any other missing values; capped for a small upload notice."""
    order = {"failed": 0, "warning": 1, "passed": 2, "not_run": 3}
    notable = [
        r
        for r in results
        if r.status in ("failed", "warning", "not_run") or (r.metric == "nullValues" and r.count not in (0, None))
    ]
    notable.sort(key=lambda r: order[r.status])
    lines = [r.message for r in notable]
    if len(lines) > config.SUMMARY_MAX_LINES:
        extra = len(lines) - config.SUMMARY_MAX_LINES + 1
        lines = lines[: config.SUMMARY_MAX_LINES - 1] + [f"…and {extra} more"]
    return lines


def _custom_value(contract: OpenDataContractStandard, prop: str) -> Any:
    return next((c.value for c in contract.customProperties or [] if c.property == prop), None)


def run_quality(lf: pl.LazyFrame, contract: OpenDataContractStandard, run_at: datetime | None = None) -> QualityReport:
    """Check ``lf`` against ``contract`` (one collect) and build the suppressed report."""
    checks = compile_contract(contract, lf.collect_schema())
    raw = measure(lf, checks)
    results = [_result(c, raw) for c in checks]
    ts = (run_at or datetime.now(UTC)).astimezone(UTC)
    report = QualityReport(
        upload_id=str(_custom_value(contract, "uploadId") or contract.id),
        dataset_version=str(_custom_value(contract, "datasetVersion") or ""),
        contract_version=contract.version or "",
        run_ts=ts.isoformat().replace("+00:00", "Z"),
        row_count=suppress_count(raw[ROWS_KEY], config.MIN_CELL_SIZE),
        passed=sum(r.status == "passed" for r in results),
        warnings=sum(r.status == "warning" for r in results),
        failed=sum(r.status == "failed" for r in results),
        not_run=sum(r.status == "not_run" for r in results),
        summary=_summary(results),
        checks=results,
    )
    logger.info(
        "quality_run_completed",
        upload_id=report.upload_id,
        checks=len(results),
        failed=report.failed,
        warnings=report.warnings,
        not_run=report.not_run,
    )
    return report

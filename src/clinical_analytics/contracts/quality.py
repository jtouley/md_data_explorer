"""Compile a data contract's rules to Polars expressions and run them in one collect.

Supported (ODCS v3.2.0 ``library`` metrics plus type options):

* ``nullValues``      -> ``null_count()`` (count, or percent of rows with ``unit: percent``)
* ``duplicateValues`` -> surplus rows: non-null rows minus distinct non-null values (property
  level), or rows minus distinct combinations of ``arguments.properties`` (object level)
* ``invalidValues``   -> non-null values not in ``arguments.validValues``
* ``rowCount``        -> ``pl.len()``
* ``logicalTypeOptions.minimum/maximum`` on a numeric column -> non-null values outside the bounds

Anything else (``sql``, ``text``, ``custom`` rules, ``missingValues``) is reported as ``not_run``
with the reason, never skipped silently. NaN in a float column counts as missing, never as out of
range.

Reports carry counts already suppressed, so the file on disk and the API response are safe to show
as they are. The model:

* The row count is always shown (the app shows it in the overview and /preview anyway).
* A column runs at most one bad-value check (codebook, range or duplicates), so it is one exact
  partition of the rows: ``[missing, bad, rest]`` or ``[missing, rest]``. ``rest`` is never
  shown but can be worked out when the others are.
* When any count in a column's line is 1..10 (``rest`` included), every non-zero shown count of
  that column is hidden. Choosing only some (the smaller, or by a fixed order) would let a reader
  rule out the worlds where the other choice was made. From 13 rows up this leaves every small
  count able to be 11 or more for a reader who knows the shown numbers, the row count, that a
  hidden count is at least 1 and that repeated values leave at least one distinct value.
* Tables of 1..12 rows (``config.MAX_ROWS_NO_COUNTS``, a fixed cut-off) hide every column count,
  zeros included, and say "too few rows to report counts".
* A suppressed count shows the one label "hidden" (never "<11"), loses its percent, and its
  check's status is always "review" (``config.HIDDEN_STATUS``), whatever the value: a pass/warning
  would say which side of a threshold it is on (1 of 20 missing "passed" at 5% meant exactly 1).
  Totals, messages and the summary therefore depend only on shown numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, TypeAlias

import polars as pl
import structlog
from open_data_contract_standard.model import DataQuality, OpenDataContractStandard, SchemaProperty
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.contracts import config
from clinical_analytics.contracts.errors import QualityCheckError

logger = structlog.get_logger()

CheckStatus: TypeAlias = Literal["passed", "warning", "failed", "review", "not_run"]
HIDDEN_NOTE = "count hidden to protect privacy"
ROWS_KEY = "__rows"
RANGE_METRIC = "range"
_BAD_VALUE_METRICS = frozenset({"invalidValues", "duplicateValues", RANGE_METRIC})
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
    count: int | str | None = Field(
        None, description="Rows affected; 'hidden' (at least 1) when suppressed; None if not run"
    )
    percent: float | None = Field(None, description="Share of rows (1 decimal); None when suppressed")
    message: str = Field(..., description="Plain-language result")


class QualityReport(BaseModel):
    """Result of checking one dataset version against its contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ran"] = "ran"
    upload_id: str
    dataset_version: str
    contract_version: str
    run_ts: str = Field(..., description="UTC run time, ISO 8601")
    row_count: int | str = Field(..., description="Rows checked; 'hidden' when suppressed")
    passed: int
    warnings: int
    failed: int
    review: int = Field(0, description="Checks whose count is hidden (status 'review')")
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
        missing = (col.is_null() | col.is_nan()).sum() if schema[column].is_float() else col.null_count()
        return CompiledCheck(**base, expr=missing, percent=rule.unit == "percent")
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
    return _one_bad_value_check_per_column(checks)


def _one_bad_value_check_per_column(checks: list[CompiledCheck]) -> list[CompiledCheck]:
    """Run at most one bad-value check (codebook, range, duplicates) per column, table-level ones included.

    The disclosure model treats each column as one partition ``[missing, bad, rest]``. Two
    overlapping bad-value counts on one column (out-of-range values are never codes) would let a
    reader subtract them; later ones are reported as not run instead.
    """
    counted: set[str] = set()
    out: list[CompiledCheck] = []
    for c in checks:  # column checks claim their column first
        column = c.column
        if c.expr is not None and column is not None and c.metric in _BAD_VALUE_METRICS:
            if column in counted:
                c = replace(c, expr=None, not_run_reason=config.OVERLAPPING_CHECK_REASON)
            else:
                counted.add(column)
        out.append(c)
    # A table-level check (e.g. duplicates over a composite key) counts toward every column it
    # names: next to "patient_id: 12 repeated", "Table: 11 repeated" would give 1 by subtraction.
    for i, c in enumerate(out):
        if c.expr is not None and c.column is None and c.metric in _BAD_VALUE_METRICS:
            if counted & set(c.columns):
                out[i] = replace(c, expr=None, not_run_reason=config.OVERLAPPING_CHECK_REASON)
            else:
                counted |= set(c.columns)
    return out


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


def _message(check: CompiledCheck, count: int, shown: int | str, percent: float | None, tiny: bool = False) -> str:
    name = check.display or "Table"
    if check.metric == "rowCount":
        return f"{shown} rows"
    if tiny:
        return f"{name}: {config.TOO_FEW_ROWS_NOTE}"
    if count == 0:
        return f"{name}: no problems found"
    hidden = shown == config.COUNT_HIDDEN
    amount = "some" if hidden else shown
    note = f"; {HIDDEN_NOTE}" if hidden else ""
    if check.metric == "nullValues":
        if check.percent and percent is not None:
            return f"{name}: {percent:.1f}% missing"
        return (
            f"{name}: {amount} values missing{note}" if check.percent else f"{name}: {amount} rows have no value{note}"
        )
    if check.metric == "duplicateValues":
        return f"{name}: {amount} repeated values{note}"
    if check.metric == "invalidValues":
        return f"{name}: {amount} values are not in the codebook ({check.detail}){note}"
    if check.metric == RANGE_METRIC:
        return f"{name}: {amount} values outside {check.detail}{note}"
    return f"{name}: {amount} rows flagged{note}"


Line: TypeAlias = tuple[str | None, str | None, str]  # (missing, bad, rest); rest is never shown


def _lines(checks: list[CompiledCheck], raw: dict[str, int]) -> tuple[dict[str, str], dict[str, int], list[Line]]:
    """Cells and lines of the disclosure model (see the module docstring).

    Returns: check id -> cell, cell -> count, and the lines (each sums to the row count).
    """
    rows = raw[ROWS_KEY]
    runnable = [c for c in checks if c.expr is not None and c.metric != "rowCount"]
    cell_of: dict[str, str] = {}
    value: dict[str, int] = {}
    missing_cell: dict[str, str] = {}
    for c in runnable:
        if c.metric == "nullValues" and c.column is not None:
            canonical = missing_cell.setdefault(c.column, c.id)
            same = canonical != c.id and raw[canonical] == raw[c.id]
            cell_of[c.id] = canonical if same else c.id
        else:
            cell_of[c.id] = c.id
        value[cell_of[c.id]] = raw[c.id]
    lines: list[Line] = []
    for c in runnable:
        if c.metric == "nullValues":
            continue
        m = missing_cell.get(c.column) if c.column is not None else None
        rest = f"{c.id}::rest"
        value[rest] = rows - (value[m] if m else 0) - value[c.id]
        lines.append((m, c.id, rest))
    in_line = {cell for line in lines for cell in line if cell}
    for c in runnable:
        cell = cell_of[c.id]
        if c.metric == "nullValues" and cell not in in_line:
            rest = f"{cell}::rest"
            value[rest] = rows - value[cell]
            lines.append((cell, None, rest))
            in_line |= {cell, rest}
    return cell_of, value, lines


def _max_feasible(
    cell: str, rows: int, value: dict[str, int], hidden: set[str], lines: list[Line], distinct_rests: set[str]
) -> int:
    """Largest value a reader could not rule out for ``cell``: hidden counts >= 1, rest >= 0, and
    for a duplicate check rest (the distinct values) >= 1 whenever any value repeats."""

    def low(c: str | None) -> int:
        return 0 if c is None else (1 if c in hidden else value[c])

    def rest_floor(bad: str | None, rest: str) -> int:
        return 1 if rest in distinct_rests and bad is not None and low(bad) > 0 else 0

    if cell.endswith("::rest"):
        m, bad, _ = next(line for line in lines if line[2] == cell)
        is_open = (m in hidden) or (bad in hidden)
        return rows - low(m) - low(bad) if is_open else value[cell]
    if cell not in hidden:
        return value[cell]
    as_missing = [line for line in lines if line[0] == cell]
    if as_missing:
        return min(rows - low(bad) - rest_floor(bad, rest) for _, bad, rest in as_missing)
    m, bad, rest = next(line for line in lines if line[1] == cell)
    return rows - low(m) - rest_floor(bad, rest)


def _is_tiny(rows: int) -> bool:
    """Tables of 1..MAX_ROWS_NO_COUNTS rows report no column counts (a fixed cut-off)."""
    return 0 < rows <= config.MAX_ROWS_NO_COUNTS


def _hidden_cells(rows: int, value: dict[str, int], lines: list[Line], distinct_rests: set[str]) -> set[str]:
    """Shown cells to hide (see module docstring). The row count itself is always shown."""
    limit = config.MIN_CELL_SIZE
    shown_cells = {c for line in lines for c in line[:2] if c is not None}
    if _is_tiny(rows):
        return shown_cells  # zeros too: a lone hidden count could otherwise equal the row count
    # A line with any count of 1..10 (rest included) hides all its non-zero shown counts, closed
    # over lines that share a cell. Hiding only some, chosen by size or by a fixed order, lets a
    # reader rule out the worlds where the other choice would have been made.
    hidden: set[str] = set()
    changed = True
    while changed:
        changed = False
        for line in lines:
            if any(c is not None and (c in hidden or 0 < value[c] < limit) for c in line):
                extra = {c for c in line[:2] if c is not None and value[c] > 0} - hidden
                if extra:
                    hidden |= extra
                    changed = True
    small = [c for c, v in value.items() if 0 < v < limit]
    if any(_max_feasible(c, rows, value, hidden, lines, distinct_rests) < limit for c in small):
        # Unreachable above MAX_ROWS_NO_COUNTS (see config): rows - 2 >= 11.
        raise QualityCheckError("Small counts cannot be protected for this table")
    return hidden


def _shown_counts(checks: list[CompiledCheck], raw: dict[str, int]) -> dict[str, tuple[int | str, float | None]]:
    """Count and percent as shown for every runnable check (the row count is always shown)."""
    rows = raw[ROWS_KEY]
    cell_of, value, lines = _lines(checks, raw)
    distinct_rests = {f"{c.id}::rest" for c in checks if c.metric == "duplicateValues"}
    hidden = _hidden_cells(rows, value, lines, distinct_rests)
    shown: dict[str, tuple[int | str, float | None]] = {}
    for c in checks:
        if c.expr is None:
            continue
        count = raw[c.id]
        if c.metric == "rowCount":
            shown[c.id] = (rows, None)
        elif cell_of[c.id] in hidden:
            shown[c.id] = (config.COUNT_HIDDEN, None)
        else:
            shown[c.id] = (count, round(100.0 * count / rows, 1) if rows else None)
    return shown


def _result(
    check: CompiledCheck,
    raw: dict[str, int],
    shown_counts: dict[str, tuple[int | str, float | None]],
    tiny: bool = False,
) -> QualityCheckResult:
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
    shown, percent = shown_counts[check.id]
    status: CheckStatus = "passed" if passed else ("failed" if check.severity == "error" else "warning")
    if shown == config.COUNT_HIDDEN:
        status = config.HIDDEN_STATUS  # never depends on the hidden value
    return QualityCheckResult(
        **common, status=status, count=shown, percent=percent, message=_message(check, count, shown, percent, tiny)
    )


def _summary(results: list[QualityCheckResult]) -> list[str]:
    """Failed, warnings, hidden ("review"), then any other missing values; capped for the upload notice."""
    order = {"failed": 0, "warning": 1, "review": 2, "passed": 3, "not_run": 4}
    notable = [
        r
        for r in results
        if r.status in ("failed", "warning", "review", "not_run")
        or (r.metric == "nullValues" and r.count not in (0, None))
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
    report = report_from_counts(contract, checks, measure(lf, checks), run_at=run_at)
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
) -> QualityReport:
    """The suppressed report for raw counts (``measure`` output). Pure: no data is read."""
    shown_counts = _shown_counts(checks, raw)
    rows = raw[ROWS_KEY]
    tiny = _is_tiny(rows)
    results = [_result(c, raw, shown_counts, tiny) for c in checks]
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
        summary=_tiny_summary(rows, results) if tiny else _summary(results),
        checks=results,
    )
    return report


def _tiny_summary(rows: int, results: list[QualityCheckResult]) -> list[str]:
    """One fixed line for a tiny table, plus checks that could not run (schema, not values)."""
    return [f"{rows} rows: {config.TOO_FEW_ROWS_NOTE}", *(r.message for r in results if r.status == "not_run")]

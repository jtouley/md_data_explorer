"""Engine-level sandbox: even SQL that skipped the validator cannot read, write or reach out."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import duckdb
import polars as pl
import pytest
from open_data_contract_standard.model import OpenDataContractStandard

from clinical_analytics.contracts.config import UNIFIED_OBJECT_NAME
from clinical_analytics.core import nl_query_config
from clinical_analytics.sql.catalog import ColumnSpec, SqlCatalog
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.sandbox import BUSY, TIMED_OUT, UNREADABLE, SandboxConnection, SqlBusyError
from tests.sql.conftest import synthetic_cohort


@pytest.fixture
def catalog(make_sql_catalog: Callable[..., SqlCatalog]) -> SqlCatalog:
    return make_sql_catalog()


@pytest.fixture
def secret_file(tmp_path: Path) -> Path:
    path = tmp_path / "secret.txt"
    path.write_text("not for SQL")
    return path


@pytest.mark.parametrize(
    "template",
    [
        "SELECT * FROM read_text('{secret}')",
        "SELECT * FROM read_csv('{secret}')",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM glob('{dir}/*')",
        "SELECT * FROM read_parquet('https://example.com/data.parquet')",
        "SELECT * FROM duckdb_secrets()",
    ],
)
def test_sandbox_fileOrNetworkRead_failsAtTheEngine(catalog: SqlCatalog, secret_file: Path, template: str) -> None:
    # Arrange: a SELECT the validator would refuse, sent straight to the engine
    sql = template.format(secret=secret_file, dir=secret_file.parent)

    # Act / Assert
    with SandboxConnection(catalog) as sandbox, pytest.raises(SqlRefusedError):
        sandbox.run_unsuppressed(sql)


@pytest.mark.parametrize(
    "template",
    [
        "COPY unified TO '{dir}/out.csv'",
        "COPY (SELECT 1) TO '{parquet}'",
        "ATTACH '{dir}/x.db' AS x",
        "EXPORT DATABASE '{dir}/export'",
        "INSTALL httpfs",
        "LOAD httpfs",
        "SET enable_external_access = true",
        "SET allowed_paths = ['/']",
        "RESET lock_configuration",
    ],
)
def test_sandbox_writeAttachOrReconfigure_failsAtTheEngine(
    catalog: SqlCatalog, secret_file: Path, template: str
) -> None:
    # Arrange: statements run on the raw connection, past the package's own SELECT check
    sql = template.format(dir=secret_file.parent, parquet=catalog.parquet_path)

    # Act / Assert
    with SandboxConnection(catalog) as sandbox, pytest.raises(duckdb.Error):
        sandbox._con.execute(sql)
    assert not (secret_file.parent / "out.csv").exists()
    assert not (secret_file.parent / "x.db").exists()


@pytest.mark.parametrize(
    "sql",
    ["COPY unified TO 'out.csv'", "SELECT 1; SELECT 2", "CREATE TABLE t AS SELECT 1", "PRAGMA database_list"],
)
def test_sandbox_runNonSelect_refused(catalog: SqlCatalog, sql: str) -> None:
    # Arrange: the sandbox's own check on what it is asked to run

    # Act / Assert
    with SandboxConnection(catalog) as sandbox, pytest.raises(SqlRefusedError, match="read-only"):
        sandbox.run_unsuppressed(sql)


def test_sandbox_slowQuery_interruptedAtTimeout(catalog: SqlCatalog) -> None:
    # Arrange: a query that runs far longer than the limit
    sql = "SELECT count(*) FROM range(100000000) a, range(100000000) b"

    # Act / Assert
    with SandboxConnection(catalog, timeout_s=0.2) as sandbox, pytest.raises(SqlRefusedError, match=TIMED_OUT):
        sandbox.run_unsuppressed(sql)


@pytest.mark.parametrize("timeout_s", [0, -1.0, float("nan")])
def test_sandbox_nonPositiveTimeout_rejected(catalog: SqlCatalog, timeout_s: float) -> None:
    # Arrange: a timeout of 0 must not mean "no limit"

    # Act / Assert
    with pytest.raises(ValueError, match="positive"):
        SandboxConnection(catalog, timeout_s=timeout_s)


def test_sandbox_failedQuery_reasonNeverQuotesAValue(catalog: SqlCatalog) -> None:
    # Arrange: a conversion error, whose DuckDB message quotes the cell ("Could not convert 'F'")
    sql = "SELECT CAST(sex AS INTEGER) AS x FROM unified"

    # Act
    with SandboxConnection(catalog) as sandbox, pytest.raises(SqlRefusedError) as refused:
        sandbox.run_unsuppressed(sql)

    # Assert
    assert "'F'" not in refused.value.reason
    assert "TRY_CAST" in refused.value.reason


def test_sandbox_missingParquet_refused(catalog: SqlCatalog) -> None:
    # Arrange
    catalog.parquet_path.unlink()

    # Act / Assert
    with pytest.raises(SqlRefusedError, match="missing"):
        SandboxConnection(catalog)


@pytest.mark.parametrize("damage", ["stale_contract", "truncated_file", "not_parquet"])
def test_sandbox_unreadableStoredData_refusedWithoutPathOrColumn(catalog: SqlCatalog, damage: str) -> None:
    # Arrange: a contract column missing from the Parquet, or a damaged file
    import dataclasses

    if damage == "stale_contract":
        catalog = dataclasses.replace(
            catalog, columns=(*catalog.columns, ColumnSpec("gone_column", "string", is_id=False, phi=False))
        )
    elif damage == "truncated_file":
        catalog.parquet_path.write_bytes(catalog.parquet_path.read_bytes()[:200])
    else:
        catalog.parquet_path.write_text("patient_id,age\nP1,40\n")

    # Act
    with pytest.raises(SqlRefusedError) as refused:
        SandboxConnection(catalog)

    # Assert: a refusal (not a 500), naming neither the file nor the column
    assert refused.value.reason == UNREADABLE
    assert str(catalog.parquet_path) not in refused.value.reason
    assert "gone_column" not in refused.value.reason


def test_sandbox_allSlotsTaken_busyAfterTheWaitAndSlotFreedOnClose(catalog: SqlCatalog) -> None:
    # Arrange: the one sandbox slot taken
    first = SandboxConnection(catalog)

    # Act / Assert
    with pytest.raises(SqlBusyError, match=BUSY):
        SandboxConnection(catalog, timeout_s=0.05)
    first.close()
    first.close()  # twice is safe, and frees the slot once
    with SandboxConnection(catalog, timeout_s=0.05) as second:
        assert second.run_unsuppressed("SELECT count(*) AS n FROM unified").item() == 300


def test_sandbox_failedOpen_freesItsSlot(catalog: SqlCatalog) -> None:
    # Arrange: a view that cannot bind
    import dataclasses

    broken = dataclasses.replace(
        catalog, columns=(*catalog.columns, ColumnSpec("gone_column", "string", is_id=False, phi=False))
    )
    with pytest.raises(SqlRefusedError):
        SandboxConnection(broken)

    # Act / Assert: the slot is free again
    with SandboxConnection(catalog, timeout_s=0.05):
        pass


def test_sandbox_memoryLimit_isTheSandboxShare(catalog: SqlCatalog) -> None:
    # Arrange
    expected = duckdb.connect()
    expected.execute(f"SET memory_limit = '{nl_query_config.SQL_SANDBOX_MEMORY_LIMIT.sql}'")
    want = expected.execute("SELECT current_setting('memory_limit')").fetchone()

    # Act
    with SandboxConnection(catalog) as sandbox:
        got = sandbox.run_unsuppressed("SELECT current_setting('memory_limit') AS m").item()

    # Assert
    assert want is not None
    assert got == want[0]


def test_sandbox_viewHasContractColumnsOnly(make_sql_catalog: Callable[..., SqlCatalog]) -> None:
    # Arrange: a contract that lists fewer columns than the Parquet holds
    import dataclasses

    full = make_sql_catalog()
    narrowed = dataclasses.replace(full, columns=tuple(c for c in full.columns if c.name != "ldl"))

    # Act
    with SandboxConnection(narrowed) as sandbox:
        columns = sandbox.run_unsuppressed("SELECT * FROM unified LIMIT 0").columns

    # Assert
    assert "ldl" not in columns
    assert "sex" in columns


def test_catalog_columnsDifferingOnlyInCase_refused(make_sql_catalog: Callable[..., SqlCatalog]) -> None:
    # Arrange: DuckDB would read "age" as "age_1" and bind the view's "age" to "Age"
    df = pl.DataFrame({"patient_id": ["P1", "P2"], "Age": [40, 50], "age": [41, 51]})

    # Act / Assert
    with pytest.raises(SqlRefusedError, match="capitalisation"):
        make_sql_catalog(df)


@pytest.mark.parametrize(
    ("column", "marks", "expected_id", "expected_phi"),
    [
        ("ward", {"classification": "restricted", "tags": None}, False, True),
        ("ward", {"classification": None, "tags": ["phi"]}, False, True),
        ("ward", {"classification": None, "tags": None}, False, False),
        ("patient_id", {"criticalDataElement": False, "required": False, "primaryKey": True}, True, True),
    ],
)
def test_catalog_idAndPhiMarks_matchQualityReportRules(
    tmp_path: Path,
    make_contract: Callable[..., OpenDataContractStandard],
    column: str,
    marks: dict[str, object],
    expected_id: bool,
    expected_phi: bool,
) -> None:
    # Arrange: a contract whose column carries only one of the marks the quality report honours
    df = synthetic_cohort()
    path = tmp_path / "cohort.parquet"
    df.write_parquet(path)
    contract = make_contract(df)
    unified = next(o for o in contract.schema_ or [] if o.name == UNIFIED_OBJECT_NAME)
    prop = next(p for p in unified.properties or [] if p.name == column)
    for field, value in marks.items():
        setattr(prop, field, value)

    # Act
    spec = SqlCatalog.from_contract(contract, path).column(column)

    # Assert
    assert spec is not None
    assert (spec.is_id, spec.phi) == (expected_id, expected_phi)

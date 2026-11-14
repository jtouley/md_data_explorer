"""Engine-level sandbox: even SQL that skipped the validator cannot read, write or reach out."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import duckdb
import pytest

from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.sandbox import TIMED_OUT, SandboxConnection


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

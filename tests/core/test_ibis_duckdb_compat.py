"""Ibis 12 on DuckDB 2.0: keep the app off the Ibis calls that are broken on this pairing.

Ibis 12.0.0 reads some results with ``DuckDBPyConnection.fetch_arrow_table`` /
``fetch_record_batch``, which DuckDB 2.0 removed from DB-API results
(https://github.com/duckdb/duckdb-python/releases/tag/v2.0.0). Affected backend calls:
``sql``, ``raw_sql(...).fetch_*``, ``list_tables`` (and ``.tables``), ``list_catalogs``,
``list_databases``, ``to_pyarrow_batches`` and the exports built on it (``to_delta``,
``to_parquet_dir``) (https://github.com/ibis-project/ibis/issues/12196). The fix is open
upstream (https://github.com/ibis-project/ibis/pull/12206); once a release carries it, raise
the ``ibis-framework`` pin and delete this module.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import ibis
import polars as pl
import pytest
from packaging.version import Version
from polars.testing import assert_frame_equal

SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "clinical_analytics"
IBIS_GAP_ACTIVE = Version(duckdb.__version__) >= Version("2.0.0") and Version(ibis.__version__) < Version("13")

# Ibis backend/expression methods that end in a removed DuckDB result fetch, on any receiver,
# called or not (``.tables`` and ``to_delta`` / ``to_parquet_dir`` go through list_tables /
# to_pyarrow_batches), looked up with getattr, plus the removed fetches themselves
# (``raw_sql(...).fetch_arrow_table()`` and the like). ``.sql`` only counts when called, since
# plain ``.sql`` attributes are common.
_BROKEN_NAMES = r"list_tables|list_catalogs|list_databases|to_pyarrow_batches|to_delta|to_parquet_dir|tables"
BROKEN_IBIS_CALL = re.compile(
    r"\.\s*sql\s*\("
    rf"|\.\s*(?:{_BROKEN_NAMES})\b"
    rf"|getattr\(\s*[^,()]+,\s*[\"'](?:sql|{_BROKEN_NAMES})[\"']"
    r"|\.\s*(?:fetch_arrow_table|fetch_record_batch)\s*\("
)
# Lines that match by name but are not Ibis: raw DuckDB connections, or the app's own attributes.
# Keyed by path relative to src/clinical_analytics ("*" = any file); values are the exact text.
NOT_IBIS: dict[str, tuple[str, ...]] = {
    "*": ("duckdb_con.sql(", "duckdb.sql("),
    "storage/datastore.py": ("self.list_tables(",),  # DataStore.list_tables: raw SHOW TABLES
    "core/multi_table_handler.py": ("self.tables",),  # dict of Polars frames
}


def broken_ibis_calls(relpath: str, line: str) -> list[str]:
    """The broken-Ibis call texts on one source line, minus the NOT_IBIS allowlist."""
    if line.lstrip().startswith("#"):
        return []
    allowed = NOT_IBIS["*"] + NOT_IBIS.get(relpath, ())
    covered = [(m.start(), m.start() + len(text)) for text in allowed for m in re.finditer(re.escape(text), line)]
    return [
        m.group(0)
        for m in BROKEN_IBIS_CALL.finditer(line)
        if not any(start <= m.start() and m.end() <= end for start, end in covered)
    ]


@pytest.mark.parametrize(
    ("relpath", "line", "n_hits"),
    [
        ("core/x.py", "names = backend.list_tables()", 1),
        ("core/x.py", "t = ic.sql('SELECT 1')", 1),
        ("core/x.py", "t = ibis.duckdb.connect().sql('SELECT 1')", 1),
        ("core/x.py", "rows = con.raw_sql('SELECT 1').fetch_arrow_table()", 1),
        ("core/x.py", "rows = con.con.execute(q).fetch_record_batch()", 1),
        ("core/x.py", "for name in con.tables:", 1),
        ("core/x.py", "expr.to_delta(path)", 1),
        ("core/x.py", "expr.to_parquet_dir(path)", 1),
        ("core/x.py", "for b in expr.to_pyarrow_batches():", 1),
        ("core/x.py", "names = con.list_catalogs() + con.list_databases()", 2),
        ("core/x.py", "rel = duckdb_con.sql('SELECT 1')", 0),
        ("core/x.py", "cur = self.con.raw_sql('SELECT 1')", 0),
        ("core/x.py", "expr.to_parquet(path)", 0),
        ("core/x.py", "    # con.list_tables() is broken on DuckDB 2.0", 0),
        ("core/x.py", "self.tables_by_name = {}", 0),
        ("storage/datastore.py", "tables = self.list_tables()", 0),
        ("core/x.py", "tables = self.list_tables()", 1),
        ("core/multi_table_handler.py", "for t in self.tables.items():", 0),
        ("core/semantic.py", "for t in self.tables.items():", 1),
        ("core/x.py", "t = con.sql ('SELECT 1')", 1),
        ("core/x.py", "t = con . sql('SELECT 1')", 1),
        ("core/x.py", "lister = backend.list_tables", 1),
        ("core/x.py", "run = getattr(con, 'sql')", 1),
        ("core/x.py", 'names = getattr(backend, "list_tables")()', 1),
        ("core/x.py", "q = plan.sql_text", 0),
    ],
)
def test_brokenIbisCalls_line_matchesIbisMethodsOnAnyReceiver(relpath: str, line: str, n_hits: int) -> None:
    # Act
    hits = broken_ibis_calls(relpath, line)

    # Assert
    assert len(hits) == n_hits, hits


def test_src_brokenIbisBackendCalls_noneUsed() -> None:
    # Arrange
    sources = sorted(SRC_DIR.rglob("*.py"))

    # Act
    hits = [
        f"{rel}:{lineno}: {line.strip()}"
        for path in sources
        for rel in [path.relative_to(SRC_DIR).as_posix()]
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if broken_ibis_calls(rel, line)
    ]

    # Assert
    assert hits == [], "Ibis calls broken on DuckDB 2.0 + Ibis 12 (see module docstring):\n" + "\n".join(hits)


@pytest.mark.skipif(not IBIS_GAP_ACTIVE, reason="gap only exists on DuckDB >= 2.0 with Ibis < 13")
@pytest.mark.xfail(strict=True, raises=AttributeError, reason="ibis#12196: fetch_arrow_table removed in DuckDB 2.0")
def test_ibisListTables_duckdb2WithIbis12_stillBroken() -> None:
    # Arrange
    con = ibis.duckdb.connect()

    # Act: when this stops raising, Ibis fixed the gap; drop the guard above.
    con.list_tables()


def test_ibisAppPaths_duckdb2_executeTableAndMedianWork(tmp_path: Path) -> None:
    # Arrange: the Ibis paths the semantic layer does use.
    csv = tmp_path / "c.csv"
    csv.write_text("grp,age\nx,10\nx,20\ny,30\ny,41\n")
    con = ibis.duckdb.connect()
    con.con.execute('CREATE OR REPLACE TABLE "t" AS SELECT * FROM read_csv_auto(?)', [str(csv)])
    table = con.table("t")

    # Act
    out = table.group_by("grp").aggregate(median=table.age.median(), n=table.age.count()).order_by("grp").to_polars()

    # Assert
    expected = pl.DataFrame({"grp": ["x", "y"], "median": [15.0, 35.5], "n": [2, 2]})
    assert_frame_equal(out, expected, check_dtypes=False)

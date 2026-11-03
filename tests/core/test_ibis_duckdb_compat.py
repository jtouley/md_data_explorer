"""Ibis 12 on DuckDB 2.0: keep the app off the Ibis calls that are broken on this pairing.

Ibis 12.0.0 reads some results with ``DuckDBPyConnection.fetch_arrow_table`` /
``fetch_record_batch``, which DuckDB 2.0 removed from DB-API results
(https://github.com/duckdb/duckdb-python/releases/tag/v2.0.0). Affected backend calls:
``sql``, ``raw_sql(...).fetch_*``, ``list_tables``, ``list_catalogs``, ``list_databases`` and
``to_pyarrow_batches`` (https://github.com/ibis-project/ibis/issues/12196). The fix is open
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
BROKEN_IBIS_CALL = re.compile(
    r"(?:con|connection|_ibis_connection)\.(?:sql|raw_sql|list_tables|list_catalogs|list_databases)\("
    r"|\.to_pyarrow_batches\("
)
IBIS_GAP_ACTIVE = Version(duckdb.__version__) >= Version("2.0.0") and Version(ibis.__version__) < Version("13")


def test_src_brokenIbisBackendCalls_noneUsed() -> None:
    # Arrange
    sources = sorted(SRC_DIR.rglob("*.py"))

    # Act
    hits = [
        f"{path.relative_to(SRC_DIR)}:{lineno}: {line.strip()}"
        for path in sources
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if BROKEN_IBIS_CALL.search(line) and not line.lstrip().startswith("#")
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

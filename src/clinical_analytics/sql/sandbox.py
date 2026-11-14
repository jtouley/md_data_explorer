# SPDX-License-Identifier: MIT
"""The DuckDB connection governed SQL runs on: in memory, one per question, never the layer's.

Read-only and sandboxed at the engine, whatever the validator lets through:

* the only file it can open is the dataset version's stored Parquet (``allowed_paths``), read
  through one view; external access is then switched off and the configuration locked, so no
  query can read another file, write one (COPY, EXPORT), ATTACH a database, reach the network,
  or install or load an extension;
* the catalog it could write to is its own in-memory one, dropped when the question ends, so
  nothing persists and nothing reaches the semantic layer, the store or another question;
* a timer interrupts a query that runs longer than the limit; memory and threads are capped,
  and spilling to disk is off;
* at most ``SQL_SANDBOX_SLOTS`` sandboxes are open at once, each with the share of
  ``DUCKDB_MEMORY_BUDGET`` reserved for it (``nl_query_config``), so the cached layers and the
  sandboxes together stay within the budget.
"""

from __future__ import annotations

import threading
from types import TracebackType

import duckdb
import polars as pl
import structlog

from clinical_analytics.core import nl_query_config
from clinical_analytics.sql import config
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.errors import SqlRefusedError

logger = structlog.get_logger()

TIMED_OUT = "The query took too long and was stopped."
BUSY = "Another SQL question is running. Try again in a moment."
UNREADABLE = (
    "This dataset's stored data can't be read as its data contract describes it. "
    "Upload it again, or open its quality report to refresh the contract."
)
_SLOTS = threading.BoundedSemaphore(nl_query_config.SQL_SANDBOX_SLOTS)


class SqlBusyError(SqlRefusedError):
    """Every sandbox slot stayed taken for the wait limit: try again (HTTP 503)."""


_COULD_NOT_RUN = {
    duckdb.ConversionException: (
        "A value could not be converted to the type the query needs (TRY_CAST returns NULL instead)."
    ),
    duckdb.BinderException: "The query does not fit the columns' types.",
    duckdb.OutOfMemoryException: "The query needs more memory than a question may use.",
}


class SandboxConnection:
    """One sandboxed connection over ``catalog``. Use as a context manager."""

    def __init__(self, catalog: SqlCatalog, *, timeout_s: float | None = None) -> None:
        """Open the sandbox, waiting up to ``timeout_s`` (default ``TIMEOUT_SECONDS``) for a slot.

        Raises:
            ValueError: ``timeout_s`` is not positive (0 must never mean "no limit").
            SqlBusyError: no slot came free in time.
            SqlRefusedError: the stored data is missing or can't be read as the contract says.
        """
        timeout_s = config.TIMEOUT_SECONDS if timeout_s is None else timeout_s
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        if not catalog.parquet_path.is_file():
            raise SqlRefusedError("This dataset's stored data is missing. Upload it again.")
        self._timeout_s = timeout_s
        if not _SLOTS.acquire(timeout=timeout_s):
            raise SqlBusyError(BUSY)
        self._slot = True
        try:
            self._con = self._open(catalog)
        except BaseException:
            self._release()
            raise

    @staticmethod
    def _open(catalog: SqlCatalog) -> duckdb.DuckDBPyConnection:
        """The locked-down connection with the view over the stored Parquet.

        Raises:
            SqlRefusedError: the Parquet can't be bound as the contract describes it (a column the
                contract lists is missing, or the file is unreadable). The engine's message, which
                names columns and the path, is logged by type only.
        """
        con = duckdb.connect(
            ":memory:",
            config={
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
                "allow_community_extensions": False,
                "allow_unsigned_extensions": False,
                "threads": nl_query_config.DUCKDB_THREADS,
            },
        )
        try:
            path = str(catalog.parquet_path.resolve())
            con.execute("SET allowed_paths = ?", [[path]])
            con.execute("SET enable_external_access = false")
            con.execute("SET max_temp_directory_size = '0B'")
            con.execute(f"SET memory_limit = '{nl_query_config.SQL_SANDBOX_MEMORY_LIMIT.sql}'")
            try:
                con.execute(catalog.view_sql())
            except duckdb.Error as e:
                logger.warning("governed_sql_view_failed", error_type=type(e).__name__)
                raise SqlRefusedError(UNREADABLE) from e
            con.execute("SET lock_configuration = true")
        except BaseException:
            con.close()
            raise
        return con

    def _release(self) -> None:
        if self._slot:
            self._slot = False
            _SLOTS.release()

    def run_unsuppressed(self, sql: str) -> pl.DataFrame:
        """Run one SELECT this package generated, within the time limit.

        The rows come back as computed, small cells included: only ``sql.privacy`` calls this,
        and nothing it returns is shown before suppression.

        Raises:
            SqlRefusedError: not one SELECT, timed out, or failed (the reason never quotes a value).
        """
        statements = duckdb.extract_statements(sql)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise SqlRefusedError("Only SELECT queries can run: the data is read-only.")
        timer = threading.Timer(self._timeout_s, self._con.interrupt)
        timer.start()
        try:
            frame = pl.from_arrow(self._con.query(sql).to_arrow_table())
            assert isinstance(frame, pl.DataFrame)  # a table, never a series
            return frame
        except duckdb.InterruptException as e:
            raise SqlRefusedError(TIMED_OUT) from e
        except duckdb.Error as e:
            # The message can quote a cell value ("Could not convert 'x' to INT"): type only.
            logger.info("governed_sql_failed", error_type=type(e).__name__)
            reason = next((text for kind, text in _COULD_NOT_RUN.items() if isinstance(e, kind)), None)
            raise SqlRefusedError(reason or "The query could not run on this dataset.") from e
        finally:
            timer.cancel()

    def close(self) -> None:
        """Close the connection and free its slot. Safe to call twice."""
        try:
            self._con.close()
        finally:
            self._release()

    def __enter__(self) -> SandboxConnection:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

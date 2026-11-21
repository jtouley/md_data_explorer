# SPDX-License-Identifier: MIT
"""The DuckDB connection governed SQL runs on: in memory, one per question, never the layer's.

Read-only and sandboxed at the engine, whatever the validator lets through:

* the only file it can open is the dataset version's stored Parquet (``allowed_paths``), read
  through one view; external access is then switched off and the configuration locked, so no
  query can read another file, write one (COPY, EXPORT), ATTACH a database, reach the network,
  or install or load an extension;
* the catalog it could write to is its own in-memory one, dropped when the question ends, so
  nothing persists and nothing reaches the semantic layer, the store or another question;
* one deadline per sandbox (the limit, from when it opens): a timer interrupts the statement
  running when it passes, so a whole answer, however many statements it runs, holds its slot
  for at most the limit; memory and threads are capped, and spilling to disk is off;
* at most ``SQL_SANDBOX_SLOTS`` sandboxes are open at once, each with the share of
  ``DUCKDB_MEMORY_BUDGET`` reserved for it (``nl_query_config``), so the cached layers and the
  sandboxes together stay within the budget.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
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
# The clock the deadline is read from; tests swap it to step time without sleeping.
_clock: Callable[[], float] = time.monotonic
_SLOTS = threading.BoundedSemaphore(nl_query_config.SQL_SANDBOX_SLOTS)


class SqlBusyError(SqlRefusedError):
    """Every sandbox slot stayed taken for the wait limit: try again (HTTP 503)."""


class SqlRunError(SqlRefusedError):
    """A model-written statement failed in the engine. ``engine_message`` is DuckDB's own text:
    feedback for the local model only, never logged or shown (it can quote a cell value)."""

    def __init__(self, reason: str, engine_message: str) -> None:
        super().__init__(reason)
        self.engine_message = engine_message


_COULD_NOT_RUN = {
    duckdb.ConversionException: (
        "A value could not be converted to the type the query needs (TRY_CAST returns NULL instead)."
    ),
    duckdb.BinderException: "The query does not fit the columns' types.",
    duckdb.OutOfMemoryException: "The query needs more memory than a question may use.",
}


def sandbox_memory_limit() -> nl_query_config.DuckdbMemoryLimit:
    """The sandbox's share of ``DUCKDB_MEMORY_BUDGET`` when governed SQL is on. With it off, the
    route answers 404 and no sandbox takes a share; one opened anyway (tests, tools) gets the
    standalone per-layer limit, as every DuckDB outside the API's cache does."""
    return nl_query_config.SQL_SANDBOX_MEMORY_LIMIT or nl_query_config.DUCKDB_MEMORY_LIMIT


class SandboxConnection:
    """One sandboxed connection over ``catalog``. Use as a context manager."""

    def __init__(self, catalog: SqlCatalog, *, timeout_s: float | None = None) -> None:
        """Open the sandbox, waiting up to ``timeout_s`` (default ``TIMEOUT_SECONDS``) for a slot;
        every statement it then runs shares one deadline, ``timeout_s`` after it opened.

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
        if not _SLOTS.acquire(timeout=timeout_s):
            raise SqlBusyError(BUSY)
        self._slot = True
        opened = False
        try:
            self._con = self._open(catalog)
            opened = True
        finally:
            if not opened:  # whatever stopped it (a refusal, an interrupt): the slot is freed
                self._release()
        self._deadline = _clock() + timeout_s

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
        configured = False
        try:
            path = str(catalog.parquet_path.resolve())
            con.execute("SET allowed_paths = ?", [[path]])
            con.execute("SET enable_external_access = false")
            con.execute("SET max_temp_directory_size = '0B'")
            con.execute(f"SET memory_limit = '{sandbox_memory_limit().sql}'")
            # Timestamps with a time zone read in UTC, as on the question box's layer.
            con.execute(f"SET TimeZone = '{nl_query_config.SESSION_TIME_ZONE}'")
            try:
                con.execute(catalog.view_sql())
            except duckdb.Error as e:
                logger.warning("governed_sql_view_failed", error_type=type(e).__name__)
                raise SqlRefusedError(UNREADABLE) from e
            con.execute("SET lock_configuration = true")
            configured = True
        finally:
            if not configured:
                con.close()
        return con

    def _release(self) -> None:
        if self._slot:
            self._slot = False
            _SLOTS.release()

    def run_generated(self, sql: str) -> pl.DataFrame:
        """Run one SELECT this package generated, within the time limit.

        Only SQL that ``sql.answer`` generated from a checked query runs here, never the user's text.

        Raises:
            SqlRefusedError: not one SELECT, timed out, or failed (the reason never quotes a value).
        """
        return self._run(sql, keep_engine_message=False)

    def run_model_sql(self, sql: str) -> pl.DataFrame:
        """Run one SELECT written (or compiled from a plan written) by the local model.

        The sandbox is the guard: read-only, one Parquet file, no external access. A failure
        raises ``SqlRunError`` carrying the engine's own message, which goes back to the local
        model as feedback for its next attempt and nowhere else (it can quote a cell value, so it
        is never logged or shown).

        Raises:
            SqlRunError: the statement failed in the engine (``engine_message`` says how).
            SqlRefusedError: not one SELECT, or timed out.
        """
        return self._run(sql, keep_engine_message=True)

    def _run(self, sql: str, *, keep_engine_message: bool) -> pl.DataFrame:
        try:
            statements = duckdb.extract_statements(sql)
        except duckdb.Error as e:
            if keep_engine_message:
                raise SqlRunError("The query is not valid SQL.", str(e)) from e
            raise SqlRefusedError("The query is not valid SQL.") from e
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise SqlRefusedError("Only SELECT queries can run: the data is read-only.")
        remaining = self._deadline - _clock()
        if remaining <= 0:
            raise SqlRefusedError(TIMED_OUT)
        timer = threading.Timer(remaining, self._con.interrupt)
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
            reason = reason or "The query could not run on this dataset."
            if keep_engine_message:
                raise SqlRunError(reason, str(e)) from e
            raise SqlRefusedError(reason) from e
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

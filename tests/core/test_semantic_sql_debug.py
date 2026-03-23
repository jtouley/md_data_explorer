"""Tests for MDDE_SQL_DEBUG Ibis compile + optional EXPLAIN logging (staff review plan)."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from clinical_analytics.core import semantic as semantic_mod


class TestMddeSqlDebugHelper:
    """Unit tests for _log_ibis_sql_and_optional_explain (no full SemanticLayer)."""

    def test_mdde_sql_debug_off_no_compile_no_explain(self) -> None:
        """When MDDE_SQL_DEBUG unset or 0, helper returns without compile or DuckDB calls."""
        mock_expr = MagicMock()
        mock_con = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            semantic_mod._log_ibis_sql_and_optional_explain(mock_expr, mock_con)

        mock_expr.compile.assert_not_called()
        mock_con.execute.assert_not_called()

        with patch.dict(os.environ, {"MDDE_SQL_DEBUG": "0"}):
            semantic_mod._log_ibis_sql_and_optional_explain(mock_expr, mock_con)

        mock_expr.compile.assert_not_called()

    def test_mdde_sql_debug_1_logs_compile_no_explain(self, caplog: pytest.LogCaptureFixture) -> None:
        """MDDE_SQL_DEBUG=1 compiles SQL and logs sql_chars; does not run EXPLAIN ANALYZE."""
        mock_expr = MagicMock()
        mock_expr.compile.return_value = "SELECT 1 AS x"
        mock_con = MagicMock()

        with patch.dict(os.environ, {"MDDE_SQL_DEBUG": "1"}):
            with caplog.at_level(logging.INFO):
                semantic_mod._log_ibis_sql_and_optional_explain(mock_expr, mock_con)

        mock_expr.compile.assert_called_once()
        mock_con.execute.assert_not_called()
        assert "mdde_sql_debug" in caplog.text
        assert "sql_chars=13" in caplog.text

    def test_mdde_sql_debug_analyze_runs_explain(self) -> None:
        """MDDE_SQL_DEBUG=analyze runs DuckDB EXPLAIN ANALYZE on compiled SQL."""
        mock_expr = MagicMock()
        mock_expr.compile.return_value = "SELECT 1"
        mock_con = MagicMock()

        with patch.dict(os.environ, {"MDDE_SQL_DEBUG": "analyze"}):
            semantic_mod._log_ibis_sql_and_optional_explain(mock_expr, mock_con)

        mock_expr.compile.assert_called_once()
        mock_con.execute.assert_called_once()
        call_sql = mock_con.execute.call_args[0][0]
        assert "EXPLAIN ANALYZE" in call_sql
        assert "SELECT 1" in call_sql

    def test_mdde_sql_log_max_chars_truncates(self, caplog: pytest.LogCaptureFixture) -> None:
        """MDDE_SQL_LOG_MAX_CHARS truncates logged SQL preview."""
        long_sql = "x" * 100
        mock_expr = MagicMock()
        mock_expr.compile.return_value = long_sql
        mock_con = MagicMock()

        with patch.dict(os.environ, {"MDDE_SQL_DEBUG": "1", "MDDE_SQL_LOG_MAX_CHARS": "20"}):
            with caplog.at_level(logging.INFO):
                semantic_mod._log_ibis_sql_and_optional_explain(mock_expr, mock_con)

        assert "truncated=1" in caplog.text
        assert "sql_chars=100" in caplog.text

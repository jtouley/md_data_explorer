"""
Tests for MIMIC-III dataset loader.
"""

import duckdb
import polars as pl
import pytest

from clinical_analytics.datasets.mimic3.loader import MIMIC3Loader, load_mimic3_from_duckdb


class TestMIMIC3Loader:
    """Test suite for MIMIC-III loader."""

    def test_loader_initialization(self, tmp_path):
        """Test loader initialization."""
        db_path = tmp_path / "test.db"
        loader = MIMIC3Loader(db_path=db_path)

        assert loader.db_path == db_path
        assert loader.db_connection is None
        assert loader.conn is None

    def test_loader_initialization_with_connection(self):
        """Test loader initialization with existing connection."""
        conn = duckdb.connect(":memory:")
        loader = MIMIC3Loader(db_connection=conn)

        assert loader.db_connection == conn
        assert loader.db_path is None

    def test_connect_file(self, tmp_path):
        """Test connecting to DuckDB file."""
        db_path = tmp_path / "test.db"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        loader = MIMIC3Loader(db_path=db_path)
        loader.connect()

        assert loader.conn is not None
        loader.disconnect()

    def test_connect_nonexistent_file(self, tmp_path):
        """Test connecting to nonexistent file raises error."""
        db_path = tmp_path / "nonexistent.db"
        loader = MIMIC3Loader(db_path=db_path)

        with pytest.raises(FileNotFoundError):
            loader.connect()

    def test_connect_no_params(self):
        """Test connecting without params raises error."""
        loader = MIMIC3Loader()

        with pytest.raises(ValueError):
            loader.connect()

    def test_disconnect(self, tmp_path):
        """Test disconnecting from database."""
        db_path = tmp_path / "test.db"
        conn = duckdb.connect(str(db_path))
        conn.close()

        loader = MIMIC3Loader(db_path=db_path)
        loader.connect()
        loader.disconnect()

        assert loader.conn is None

    def test_execute_query(self, tmp_path):
        """Test executing SQL query."""
        db_path = tmp_path / "test.db"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
        conn.close()

        loader = MIMIC3Loader(db_path=db_path)
        result = loader.execute_query("SELECT * FROM test")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "id" in result.columns
        assert "name" in result.columns
        loader.disconnect()

    def test_load_cohort(self, tmp_path):
        """Test loading cohort data."""
        db_path = tmp_path / "test.db"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE patients (subject_id INTEGER, age INTEGER)")
        conn.execute("INSERT INTO patients VALUES (1, 45), (2, 62)")
        conn.close()

        loader = MIMIC3Loader(db_path=db_path)
        result = loader.load_cohort("SELECT * FROM patients")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        loader.disconnect()

    def test_load_cohort_empty_query(self, tmp_path):
        """Test loading cohort with empty query raises error."""
        db_path = tmp_path / "test.db"
        conn = duckdb.connect(str(db_path))
        conn.close()

        loader = MIMIC3Loader(db_path=db_path)
        loader.connect()

        with pytest.raises(ValueError):
            loader.load_cohort("")

        with pytest.raises(ValueError):
            loader.load_cohort(None)

        loader.disconnect()

    def test_check_tables_exist(self, tmp_path):
        """Test checking table existence."""
        db_path = tmp_path / "test.db"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE patients (id INTEGER)")
        conn.execute("CREATE TABLE admissions (id INTEGER)")
        conn.close()

        loader = MIMIC3Loader(db_path=db_path)
        table_status = loader.check_tables_exist()

        assert isinstance(table_status, dict)
        assert table_status["patients"] is True
        assert table_status["admissions"] is False  # Not created
        loader.disconnect()

    def test_context_manager(self, tmp_path):
        """Test using loader as context manager."""
        db_path = tmp_path / "test.db"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        with MIMIC3Loader(db_path=db_path) as loader:
            assert loader.conn is not None
            result = loader.execute_query("SELECT * FROM test")
            assert isinstance(result, pl.DataFrame)

        # Connection should be closed
        assert loader.conn is None

    def test_load_mimic3_from_duckdb(self, tmp_path):
        """Test convenience function for loading from DuckDB."""
        db_path = tmp_path / "test.db"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER, value INTEGER)")
        conn.execute("INSERT INTO test VALUES (1, 10), (2, 20)")
        conn.close()

        result = load_mimic3_from_duckdb(db_path, "SELECT * FROM test")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

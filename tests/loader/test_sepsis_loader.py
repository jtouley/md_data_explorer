"""
Tests for Sepsis dataset loader.
"""

import polars as pl
import pytest

from clinical_analytics.core.mapper import ColumnMapper
from clinical_analytics.datasets.sepsis.loader import (
    find_psv_files,
    load_and_aggregate,
    load_patient_file,
)


class TestSepsisLoader:
    """Test suite for Sepsis loader."""

    def test_find_psv_files(self, tmp_path):
        """Test finding PSV files recursively."""
        # Create test directory structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # Create test PSV files
        (tmp_path / "p00001.psv").write_text("col1|col2\n1|2\n")
        (subdir / "p00002.psv").write_text("col1|col2\n3|4\n")

        files = list(find_psv_files(tmp_path))

        assert len(files) == 2
        assert all(f.suffix == ".psv" for f in files)

    def test_find_psv_files_none(self, tmp_path):
        """Test finding PSV files when none exist."""
        files = list(find_psv_files(tmp_path))
        assert len(files) == 0

    def test_load_patient_file(self, tmp_path):
        """Test loading a single PSV file."""
        test_file = tmp_path / "test.psv"
        test_file.write_text("Age|Gender|SepsisLabel\n45|M|0\n")

        df = load_patient_file(test_file)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        assert "Age" in df.columns
        assert "Gender" in df.columns
        assert "SepsisLabel" in df.columns

    def test_load_and_aggregate(self, tmp_path):
        """Test loading and aggregating multiple PSV files."""
        # Create test PSV files
        (tmp_path / "p00001.psv").write_text("Age|Gender|SepsisLabel\n45|M|0\n45|M|1\n")
        (tmp_path / "p00002.psv").write_text("Age|Gender|SepsisLabel\n62|F|0\n62|F|0\n")

        config = {
            "aggregation": {
                "static_features": [
                    {"column": "Age", "method": "first", "target": "age"},
                    {"column": "Gender", "method": "first", "target": "gender"},
                ],
                "outcome": {"column": "SepsisLabel", "method": "max", "target": "sepsis_label"},
            }
        }
        mapper = ColumnMapper(config)

        result = load_and_aggregate(tmp_path, mapper=mapper)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2  # Two patients
        assert "patient_id" in result.columns
        assert "age" in result.columns
        assert "gender" in result.columns
        assert "sepsis_label" in result.columns

    def test_load_and_aggregate_no_files(self, tmp_path):
        """Test loading when no PSV files exist."""
        with pytest.raises(FileNotFoundError):
            load_and_aggregate(tmp_path)

    def test_load_and_aggregate_with_limit(self, tmp_path):
        """Test loading with file limit."""
        # Create multiple test files
        for i in range(5):
            (tmp_path / f"p{i:05d}.psv").write_text("Age|Gender|SepsisLabel\n45|M|0\n")

        result = load_and_aggregate(tmp_path, limit=2)

        # Should only process 2 files
        assert len(result) == 2

    def test_load_and_aggregate_without_mapper(self, tmp_path):
        """Test loading without mapper (fallback aggregation)."""
        (tmp_path / "p00001.psv").write_text("Age|Gender|SepsisLabel\n45|M|0\n45|M|1\n")

        result = load_and_aggregate(tmp_path, mapper=None)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert "patient_id" in result.columns

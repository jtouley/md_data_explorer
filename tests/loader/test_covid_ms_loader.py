"""
Tests for COVID-MS dataset loader.
"""

import pytest
import polars as pl
from pathlib import Path
from clinical_analytics.datasets.covid_ms.loader import load_raw_data, clean_data
from clinical_analytics.core.mapper import ColumnMapper


class TestCOVIDMSLoader:
    """Test suite for COVID-MS loader."""

    def test_load_raw_data(self, tmp_path):
        """Test loading raw CSV data."""
        # Create test CSV file
        test_file = tmp_path / "test_data.csv"
        test_data = pl.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'age': [45, 62, 38],
            'sex': ['M', 'F', 'M']
        })
        test_data.write_csv(test_file)

        df = load_raw_data(test_file)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert 'patient_id' in df.columns

    def test_load_raw_data_nonexistent(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_raw_data(Path('/nonexistent/file.csv'))

    def test_clean_data_basic(self):
        """Test basic data cleaning."""
        df = pl.DataFrame({
            'sex': ['M', 'F', None, 'M'],
            'age': [45, 62, 38, 50]
        })

        result = clean_data(df)

        assert isinstance(result, pl.DataFrame)
        # Nulls should be filled
        assert result['sex'].null_count() == 0
        assert 'Unknown' in result['sex'].to_list()

    def test_clean_data_with_mapper(self):
        """Test data cleaning with mapper for outcome transformations."""
        df = pl.DataFrame({
            'sex': ['M', 'F', 'M'],
            'outcome_source': ['yes', 'no', 'yes']
        })

        config = {
            'outcomes': {
                'outcome': {
                    'source_column': 'outcome_source',
                    'type': 'binary',
                    'mapping': {'yes': 1, 'no': 0}
                }
            }
        }
        mapper = ColumnMapper(config)

        result = clean_data(df, mapper=mapper)

        assert 'outcome' in result.columns
        assert result['outcome'].to_list() == [1, 0, 1]

    def test_clean_data_without_mapper(self):
        """Test data cleaning without mapper."""
        df = pl.DataFrame({
            'sex': ['M', 'F', None],
            'age': [45, 62, 38]
        })

        result = clean_data(df, mapper=None)

        assert isinstance(result, pl.DataFrame)
        assert result['sex'].null_count() == 0


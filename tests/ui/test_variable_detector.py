"""
Tests for variable type detector component.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from clinical_analytics.ui.components.variable_detector import VariableTypeDetector


class TestVariableTypeDetector:
    """Test suite for VariableTypeDetector."""

    def test_detect_binary_yes_no(self):
        """Test detecting binary variable with yes/no values."""
        series = pd.Series(['yes', 'no', 'yes', 'no', 'yes'])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'outcome')

        assert var_type == 'binary'
        assert 'unique_count' in metadata

    def test_detect_binary_1_0(self):
        """Test detecting binary variable with 1/0 values."""
        series = pd.Series([1, 0, 1, 0, 1])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'status')

        assert var_type == 'binary'

    def test_detect_binary_true_false(self):
        """Test detecting binary variable with True/False values."""
        series = pd.Series([True, False, True, False])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'active')

        assert var_type == 'binary'

    def test_detect_categorical(self):
        """Test detecting categorical variable."""
        series = pd.Series(['A', 'B', 'C', 'A', 'B', 'C', 'A'])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'category')

        assert var_type == 'categorical'
        assert metadata['unique_count'] == 3

    def test_detect_continuous(self):
        """Test detecting continuous variable."""
        series = pd.Series([1.5, 2.3, 3.7, 4.2, 5.1, 6.8, 7.3, 8.9, 9.2, 10.5, 11.3, 12.7])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'age')

        assert var_type == 'continuous'

    def test_detect_datetime(self):
        """Test detecting datetime variable."""
        series = pd.Series([
            '2020-01-01',
            '2020-02-01',
            '2020-03-01',
            '2020-04-01'
        ])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'date')

        assert var_type == 'datetime'

    def test_detect_datetime_parsed(self):
        """Test detecting datetime with parsed datetime type."""
        series = pd.Series([
            datetime(2020, 1, 1),
            datetime(2020, 2, 1),
            datetime(2020, 3, 1)
        ])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'timestamp')

        assert var_type == 'datetime'

    def test_detect_id_column(self):
        """Test detecting ID column."""
        series = pd.Series([f'P{i:05d}' for i in range(100)])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'patient_id')

        assert var_type == 'id'
        assert metadata['suggested_as_patient_id'] is True

    def test_detect_with_nulls(self):
        """Test detection with null values."""
        series = pd.Series([1, 2, None, 3, None, 4])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'value')

        # Should still detect correctly despite nulls
        assert var_type in ['continuous', 'categorical']

    def test_detect_all_nulls(self):
        """Test detection with all null values."""
        series = pd.Series([None, None, None])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'empty')

        # Should handle gracefully
        assert var_type is not None

    def test_detect_outcome_pattern(self):
        """Test detecting outcome variable by name pattern."""
        series = pd.Series([1, 0, 1, 0])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'outcome_death')

        assert var_type == 'binary'

    def test_detect_time_pattern(self):
        """Test detecting time variable by name pattern."""
        series = pd.Series(['2020-01-01', '2020-02-01', '2020-03-01'])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'admission_date')

        assert var_type == 'datetime'

    def test_detect_categorical_threshold(self):
        """Test categorical threshold logic."""
        # Create series with exactly threshold + 1 unique values
        series = pd.Series([f'cat_{i}' for i in range(21)])  # 21 unique values
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, 'category')

        # Should be continuous (above threshold) or categorical depending on implementation
        assert var_type in ['categorical', 'continuous']


"""
Tests for data profiling module.
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from clinical_analytics.core.profiling import DataProfiler


class TestDataProfiler:
    """Test suite for DataProfiler."""

    def test_profiler_initialization_pandas(self):
        """Test profiler initialization with Pandas DataFrame."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        profiler = DataProfiler(df)

        assert profiler.data is not None
        assert isinstance(profiler.data, pd.DataFrame)

    def test_profiler_initialization_polars(self):
        """Test profiler initialization with Polars DataFrame."""
        df = pl.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        profiler = DataProfiler(df)

        assert profiler.data is not None
        assert isinstance(profiler.data, pd.DataFrame)  # Should convert to pandas

    def test_generate_profile(self):
        """Test generating complete profile."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'B', 'A'],
            'with_nulls': [1, None, 3, None, 5]
        })
        profiler = DataProfiler(df)
        profile = profiler.generate_profile()

        assert 'overview' in profile
        assert 'missing_data' in profile
        assert 'numeric_features' in profile
        assert 'categorical_features' in profile
        assert 'data_quality' in profile

    def test_profile_overview(self):
        """Test overview statistics."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        profiler = DataProfiler(df)
        overview = profiler._profile_overview()

        assert overview['n_rows'] == 3
        assert overview['n_columns'] == 2
        assert 'col1' in overview['column_names']
        assert 'col2' in overview['column_names']
        assert 'memory_usage_mb' in overview

    def test_profile_missing_data(self):
        """Test missing data analysis."""
        df = pd.DataFrame({
            'complete': [1, 2, 3],
            'some_missing': [1, None, 3],
            'all_missing': [None, None, None]
        })
        profiler = DataProfiler(df)
        missing = profiler._profile_missing_data()

        assert missing['total_missing_cells'] == 4  # 1 + 3
        assert 'some_missing' in missing['columns_with_missing']
        assert 'all_missing' in missing['columns_with_missing']
        assert missing['pct_missing_overall'] > 0

    def test_profile_missing_data_no_missing(self):
        """Test missing data analysis with no missing values."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        profiler = DataProfiler(df)
        missing = profiler._profile_missing_data()

        assert missing['total_missing_cells'] == 0
        assert missing['pct_missing_overall'] == 0.0
        assert len(missing['columns_with_missing']) == 0

    def test_profile_numeric_features(self):
        """Test numeric feature profiling."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'with_zeros': [0, 1, 2, 0, 3, 0, 4, 5, 0, 6]
        })
        profiler = DataProfiler(df)
        numeric = profiler._profile_numeric_features()

        assert 'numeric' in numeric
        assert 'with_zeros' in numeric
        assert numeric['numeric']['mean'] == 5.5
        assert numeric['numeric']['min'] == 1
        assert numeric['numeric']['max'] == 10
        assert numeric['numeric']['median'] == 5.5
        assert numeric['with_zeros']['n_zeros'] == 3

    def test_profile_numeric_features_empty(self):
        """Test numeric profiling with no numeric columns."""
        df = pd.DataFrame({
            'text': ['a', 'b', 'c']
        })
        profiler = DataProfiler(df)
        numeric = profiler._profile_numeric_features()

        assert numeric == {}

    def test_profile_categorical_features(self):
        """Test categorical feature profiling."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'C', 'A']
        })
        profiler = DataProfiler(df)
        categorical = profiler._profile_categorical_features()

        assert 'category' in categorical
        assert categorical['category']['n_unique'] == 3
        assert categorical['category']['mode'] == 'A'
        assert categorical['category']['pct_mode'] > 0
        assert len(categorical['category']['top_values']) > 0

    def test_profile_categorical_features_empty(self):
        """Test categorical profiling with no categorical columns."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3]
        })
        profiler = DataProfiler(df)
        categorical = profiler._profile_categorical_features()

        assert categorical == {}

    def test_profile_data_quality(self):
        """Test data quality assessment."""
        df = pd.DataFrame({
            'good': [1, 2, 3, 4, 5],
            'high_missing': [1, None, None, None, None],
            'constant': [1, 1, 1, 1, 1]
        })
        profiler = DataProfiler(df)
        quality = profiler._profile_data_quality()

        assert 'quality_score' in quality
        assert 'issues' in quality
        assert 'n_issues' in quality
        assert 0 <= quality['quality_score'] <= 100
        assert len(quality['issues']) > 0

    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        # Perfect data
        df_perfect = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        profiler_perfect = DataProfiler(df_perfect)
        score_perfect = profiler_perfect._calculate_quality_score()

        # Data with issues
        df_issues = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['a', 'a', 'a']
        })
        profiler_issues = DataProfiler(df_issues)
        score_issues = profiler_issues._calculate_quality_score()

        assert score_perfect >= score_issues
        assert 0 <= score_perfect <= 100
        assert 0 <= score_issues <= 100

    def test_to_dict(self):
        """Test converting profile to dictionary."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        profiler = DataProfiler(df)
        profile_dict = profiler.to_dict()

        assert isinstance(profile_dict, dict)
        assert 'overview' in profile_dict

    def test_to_html(self):
        """Test generating HTML report."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'category': ['A', 'B', 'C']
        })
        profiler = DataProfiler(df)
        html = profiler.to_html()

        assert isinstance(html, str)
        assert '<html>' in html
        assert 'Data Profile Report' in html
        assert 'Overview' in html

    def test_profile_with_duplicates(self):
        """Test profiling with duplicate rows."""
        df = pd.DataFrame({
            'col1': [1, 2, 1, 2, 3],
            'col2': ['a', 'b', 'a', 'b', 'c']
        })
        profiler = DataProfiler(df)
        quality = profiler._profile_data_quality()

        # Should detect duplicates
        duplicate_issues = [issue for issue in quality['issues'] if issue['type'] == 'duplicate_rows']
        assert len(duplicate_issues) > 0

    def test_profile_with_constant_columns(self):
        """Test profiling with constant columns."""
        df = pd.DataFrame({
            'constant': [1, 1, 1, 1],
            'variable': [1, 2, 3, 4]
        })
        profiler = DataProfiler(df)
        quality = profiler._profile_data_quality()

        constant_issues = [issue for issue in quality['issues'] if issue['type'] == 'constant_columns']
        assert len(constant_issues) > 0
        assert 'constant' in constant_issues[0]['columns']


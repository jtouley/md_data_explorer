"""
Data Profiling Module - Generate dataset statistics and quality metrics.

This module provides utilities for profiling clinical datasets including
missing data analysis, distribution summaries, and data quality metrics.
"""

import pandas as pd
import polars as pl
from typing import Dict, Any, Union, List
from pathlib import Path


class DataProfiler:
    """
    Generates comprehensive data quality and statistical profiles for datasets.
    """

    def __init__(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Initialize profiler with dataset.

        Args:
            data: Pandas or Polars DataFrame to profile
        """
        # Convert Polars to Pandas for consistent profiling
        if isinstance(data, pl.DataFrame):
            self.data = data.to_pandas()
        else:
            self.data = data

    def generate_profile(self) -> Dict[str, Any]:
        """
        Generate comprehensive data profile.

        Returns:
            Dictionary containing all profile metrics
        """
        profile = {
            'overview': self._profile_overview(),
            'missing_data': self._profile_missing_data(),
            'numeric_features': self._profile_numeric_features(),
            'categorical_features': self._profile_categorical_features(),
            'data_quality': self._profile_data_quality(),
        }

        return profile

    def _profile_overview(self) -> Dict[str, Any]:
        """Generate overview statistics."""
        return {
            'n_rows': len(self.data),
            'n_columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'dtypes': self.data.dtypes.astype(str).to_dict(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / (1024 * 1024),
        }

    def _profile_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = self.data.isnull().sum()
        missing_pct = (missing_counts / len(self.data) * 100).round(2)

        missing_summary = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_pct': missing_pct
        })

        # Filter to only columns with missing data
        missing_summary = missing_summary[missing_summary['missing_count'] > 0]
        missing_summary = missing_summary.sort_values('missing_pct', ascending=False)

        return {
            'total_missing_cells': int(self.data.isnull().sum().sum()),
            'pct_missing_overall': round(
                (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns)) * 100),
                2
            ),
            'columns_with_missing': missing_summary.to_dict('index'),
            'complete_rows': int((~self.data.isnull().any(axis=1)).sum()),
            'pct_complete_rows': round(
                ((~self.data.isnull().any(axis=1)).sum() / len(self.data) * 100),
                2
            ),
        }

    def _profile_numeric_features(self) -> Dict[str, Any]:
        """Profile numeric columns."""
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()

        if not numeric_cols:
            return {}

        numeric_profile = {}

        for col in numeric_cols:
            col_data = self.data[col].dropna()

            if len(col_data) == 0:
                continue

            numeric_profile[col] = {
                'count': int(len(col_data)),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'q25': float(col_data.quantile(0.25)),
                'median': float(col_data.median()),
                'q75': float(col_data.quantile(0.75)),
                'max': float(col_data.max()),
                'n_zeros': int((col_data == 0).sum()),
                'n_unique': int(col_data.nunique()),
            }

        return numeric_profile

    def _profile_categorical_features(self) -> Dict[str, Any]:
        """Profile categorical columns."""
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            return {}

        categorical_profile = {}

        for col in categorical_cols:
            col_data = self.data[col].dropna()

            if len(col_data) == 0:
                continue

            value_counts = col_data.value_counts()

            categorical_profile[col] = {
                'n_unique': int(col_data.nunique()),
                'mode': str(col_data.mode()[0]) if len(col_data.mode()) > 0 else None,
                'top_values': value_counts.head(10).to_dict(),
                'pct_mode': round((value_counts.iloc[0] / len(col_data) * 100), 2) if len(value_counts) > 0 else 0,
            }

        return categorical_profile

    def _profile_data_quality(self) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality_issues = []

        # Check for columns with high missingness
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100)
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            quality_issues.append({
                'type': 'high_missingness',
                'severity': 'warning',
                'columns': high_missing,
                'message': f'{len(high_missing)} columns with >50% missing data'
            })

        # Check for constant columns
        constant_cols = []
        for col in self.data.columns:
            if self.data[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            quality_issues.append({
                'type': 'constant_columns',
                'severity': 'info',
                'columns': constant_cols,
                'message': f'{len(constant_cols)} columns with constant values'
            })

        # Check for duplicate rows
        n_duplicates = self.data.duplicated().sum()
        if n_duplicates > 0:
            quality_issues.append({
                'type': 'duplicate_rows',
                'severity': 'warning',
                'count': int(n_duplicates),
                'message': f'{n_duplicates} duplicate rows found'
            })

        return {
            'quality_score': self._calculate_quality_score(),
            'issues': quality_issues,
            'n_issues': len(quality_issues),
        }

    def _calculate_quality_score(self) -> float:
        """
        Calculate overall data quality score (0-100).

        Higher is better. Based on:
        - Completeness (no missing data)
        - Uniqueness (no duplicates)
        - Consistency (appropriate data types)
        """
        # Completeness score (0-40 points)
        completeness = (1 - (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns)))) * 40

        # Uniqueness score (0-30 points)
        uniqueness = (1 - (self.data.duplicated().sum() / len(self.data))) * 30

        # Consistency score (0-30 points) - simplified for now
        consistency = 30  # Placeholder - could check for data type consistency

        total_score = completeness + uniqueness + consistency

        return round(float(total_score), 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return self.generate_profile()

    def to_html(self) -> str:
        """Generate HTML report of data profile."""
        profile = self.generate_profile()

        html = "<html><head><style>"
        html += "body { font-family: Arial, sans-serif; margin: 20px; }"
        html += "h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }"
        html += "h3 { color: #555; }"
        html += "table { border-collapse: collapse; width: 100%; margin: 10px 0; }"
        html += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }"
        html += "th { background-color: #007bff; color: white; }"
        html += ".warning { color: #ff6b6b; }"
        html += ".info { color: #4ecdc4; }"
        html += "</style></head><body>"

        html += f"<h1>Data Profile Report</h1>"

        # Overview
        html += "<h2>Overview</h2>"
        html += f"<p>Rows: {profile['overview']['n_rows']:,}</p>"
        html += f"<p>Columns: {profile['overview']['n_columns']}</p>"
        html += f"<p>Memory: {profile['overview']['memory_usage_mb']:.2f} MB</p>"

        # Data Quality
        html += "<h2>Data Quality</h2>"
        html += f"<p>Quality Score: {profile['data_quality']['quality_score']}/100</p>"
        if profile['data_quality']['issues']:
            html += "<h3>Issues Found:</h3><ul>"
            for issue in profile['data_quality']['issues']:
                html += f"<li class='{issue['severity']}'>{issue['message']}</li>"
            html += "</ul>"

        # Missing Data
        html += "<h2>Missing Data</h2>"
        html += f"<p>Overall Missing: {profile['missing_data']['pct_missing_overall']:.2f}%</p>"
        html += f"<p>Complete Rows: {profile['missing_data']['pct_complete_rows']:.2f}%</p>"

        html += "</body></html>"

        return html


def profile_dataset(dataset, save_path: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Profile a clinical dataset and optionally save report.

    Args:
        dataset: ClinicalDataset instance
        save_path: Optional path to save HTML report

    Returns:
        Profile dictionary
    """
    # Load data if not already loaded
    if dataset._data is None:
        dataset.load()

    # Create profiler
    profiler = DataProfiler(dataset._data)
    profile = profiler.generate_profile()

    # Save HTML report if requested
    if save_path:
        save_path = Path(save_path)
        with open(save_path, 'w') as f:
            f.write(profiler.to_html())
        print(f"Profile saved to {save_path}")

    return profile

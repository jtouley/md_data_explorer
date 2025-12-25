#!/usr/bin/env python3
"""
Clinical Analytics Platform - Validation Suite

This script validates the complete platform including:
- Polars-based dataset loading and processing
- UnifiedCohort schema compliance
- Statistical analysis functionality
- Data quality and integrity
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import polars as pl
import pandas as pd
from typing import List, Tuple

from clinical_analytics.datasets.covid_ms.definition import CovidMSDataset
from clinical_analytics.datasets.sepsis.definition import SepsisDataset
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.analysis.stats import run_logistic_regression


class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass


class PlatformValidator:
    """Validates the clinical analytics platform"""

    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0

    def run_all_tests(self):
        """Run all validation tests"""
        print("=" * 70)
        print("Clinical Analytics Platform - Validation Suite")
        print("=" * 70)
        print()

        # Test 1: COVID-MS Dataset
        self.test_covid_ms_loading()
        self.test_covid_ms_polars_backend()
        self.test_covid_ms_schema_compliance()
        self.test_covid_ms_data_quality()

        # Test 2: Sepsis Dataset
        self.test_sepsis_loading()
        self.test_sepsis_schema_compliance()

        # Test 3: Statistical Analysis
        self.test_logistic_regression()

        # Test 4: UnifiedCohort Schema
        self.test_unified_cohort_schema()

        # Summary
        print()
        print("=" * 70)
        print("Validation Summary")
        print("=" * 70)
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"⚠️  Warnings: {self.warnings}")
        print()

        if self.failed_tests > 0:
            print("❌ VALIDATION FAILED - Please review failures above")
            sys.exit(1)
        elif self.warnings > 0:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            sys.exit(0)
        else:
            print("✅ ALL VALIDATIONS PASSED")
            sys.exit(0)

    def log_test(self, test_name: str, status: str, message: str = ""):
        """Log test results"""
        symbols = {
            'pass': '✅',
            'fail': '❌',
            'warn': '⚠️ '
        }
        print(f"{symbols[status]} {test_name}")
        if message:
            print(f"   {message}")

        if status == 'pass':
            self.passed_tests += 1
        elif status == 'fail':
            self.failed_tests += 1
        elif status == 'warn':
            self.warnings += 1

    # COVID-MS Dataset Tests
    def test_covid_ms_loading(self):
        """Test COVID-MS dataset loading"""
        print("\n[1] COVID-MS Dataset Loading")
        print("-" * 70)

        try:
            dataset = CovidMSDataset()

            # Validation
            if not dataset.validate():
                self.log_test(
                    "COVID-MS data file exists",
                    'fail',
                    "GDSI_OpenDataset_Final.csv not found"
                )
                return

            self.log_test("COVID-MS data file exists", 'pass')

            # Load
            dataset.load()
            self.log_test("COVID-MS dataset loads successfully", 'pass')

            # Check record count
            if hasattr(dataset, '_data') and dataset._data is not None:
                record_count = len(dataset._data)
                self.log_test(
                    f"COVID-MS loaded {record_count} records",
                    'pass' if record_count > 0 else 'fail'
                )
            else:
                self.log_test("COVID-MS data loaded", 'fail', "No data found")

        except Exception as e:
            self.log_test("COVID-MS dataset loading", 'fail', str(e))

    def test_covid_ms_polars_backend(self):
        """Verify COVID-MS uses Polars backend"""
        print("\n[2] COVID-MS Polars Backend Verification")
        print("-" * 70)

        try:
            dataset = CovidMSDataset()

            if not dataset.validate():
                self.log_test("COVID-MS Polars backend", 'warn', "Data not available")
                return

            dataset.load()

            # Check if internal data is Polars DataFrame
            if hasattr(dataset, '_data') and isinstance(dataset._data, pl.DataFrame):
                self.log_test("COVID-MS uses Polars DataFrame internally", 'pass')

                # Check Polars-specific operations
                if 'outcome_hospitalized' in dataset._data.columns:
                    self.log_test("Polars data cleaning applied", 'pass')
                else:
                    self.log_test("Polars data cleaning", 'fail', "Missing expected columns")
            else:
                self.log_test(
                    "COVID-MS uses Polars DataFrame",
                    'fail',
                    f"Expected pl.DataFrame, got {type(dataset._data)}"
                )

        except Exception as e:
            self.log_test("COVID-MS Polars backend", 'fail', str(e))

    def test_covid_ms_schema_compliance(self):
        """Test COVID-MS UnifiedCohort schema compliance"""
        print("\n[3] COVID-MS UnifiedCohort Schema Compliance")
        print("-" * 70)

        try:
            dataset = CovidMSDataset()

            if not dataset.validate():
                self.log_test("COVID-MS schema compliance", 'warn', "Data not available")
                return

            cohort = dataset.get_cohort()

            # Check return type (should be Pandas for statsmodels)
            if isinstance(cohort, pd.DataFrame):
                self.log_test("get_cohort() returns Pandas DataFrame", 'pass')
            else:
                self.log_test(
                    "get_cohort() returns Pandas DataFrame",
                    'fail',
                    f"Got {type(cohort)}"
                )
                return

            # Check required columns
            missing_cols = []
            for col in UnifiedCohort.REQUIRED_COLUMNS:
                if col not in cohort.columns:
                    missing_cols.append(col)

            if not missing_cols:
                self.log_test("All required UnifiedCohort columns present", 'pass')
            else:
                self.log_test(
                    "Required columns check",
                    'fail',
                    f"Missing: {missing_cols}"
                )

            # Check data types
            if UnifiedCohort.PATIENT_ID in cohort.columns:
                if cohort[UnifiedCohort.PATIENT_ID].dtype == object:
                    self.log_test("patient_id is string type", 'pass')
                else:
                    self.log_test(
                        "patient_id type",
                        'warn',
                        f"Expected string, got {cohort[UnifiedCohort.PATIENT_ID].dtype}"
                    )

            if UnifiedCohort.TIME_ZERO in cohort.columns:
                if pd.api.types.is_datetime64_any_dtype(cohort[UnifiedCohort.TIME_ZERO]):
                    self.log_test("time_zero is datetime type", 'pass')
                else:
                    self.log_test(
                        "time_zero type",
                        'fail',
                        f"Expected datetime, got {cohort[UnifiedCohort.TIME_ZERO].dtype}"
                    )

            if UnifiedCohort.OUTCOME in cohort.columns:
                if pd.api.types.is_numeric_dtype(cohort[UnifiedCohort.OUTCOME]):
                    self.log_test("outcome is numeric type", 'pass')
                else:
                    self.log_test(
                        "outcome type",
                        'fail',
                        f"Expected numeric, got {cohort[UnifiedCohort.OUTCOME].dtype}"
                    )

        except Exception as e:
            self.log_test("COVID-MS schema compliance", 'fail', str(e))

    def test_covid_ms_data_quality(self):
        """Test COVID-MS data quality"""
        print("\n[4] COVID-MS Data Quality Checks")
        print("-" * 70)

        try:
            dataset = CovidMSDataset()

            if not dataset.validate():
                self.log_test("COVID-MS data quality", 'warn', "Data not available")
                return

            cohort = dataset.get_cohort()

            # Check for duplicates
            if cohort[UnifiedCohort.PATIENT_ID].duplicated().any():
                self.log_test("No duplicate patient IDs", 'warn', "Duplicates found")
            else:
                self.log_test("No duplicate patient IDs", 'pass')

            # Check outcome values
            outcome_values = cohort[UnifiedCohort.OUTCOME].unique()
            if all(v in [0, 1] for v in outcome_values if pd.notna(v)):
                self.log_test("Outcome values are binary (0/1)", 'pass')
            else:
                self.log_test(
                    "Binary outcome values",
                    'warn',
                    f"Found values: {outcome_values}"
                )

            # Check for missing data
            missing_pct = (cohort.isnull().sum() / len(cohort) * 100)
            critical_cols = UnifiedCohort.REQUIRED_COLUMNS
            critical_missing = missing_pct[critical_cols]

            if (critical_missing > 0).any():
                self.log_test(
                    "No missing values in critical columns",
                    'warn',
                    f"Missing data found"
                )
            else:
                self.log_test("No missing values in critical columns", 'pass')

        except Exception as e:
            self.log_test("COVID-MS data quality", 'fail', str(e))

    # Sepsis Dataset Tests
    def test_sepsis_loading(self):
        """Test Sepsis dataset loading"""
        print("\n[5] Sepsis Dataset Loading")
        print("-" * 70)

        try:
            dataset = SepsisDataset()

            # Validation
            if not dataset.validate():
                self.log_test(
                    "Sepsis data loading",
                    'warn',
                    "No PSV files found (expected for demo)"
                )
                return

            # If data exists, load it
            dataset.load()
            self.log_test("Sepsis dataset loads successfully", 'pass')

            cohort = dataset.get_cohort()
            if len(cohort) > 0:
                self.log_test(f"Sepsis loaded {len(cohort)} patients", 'pass')
            else:
                self.log_test("Sepsis data", 'warn', "No patients loaded")

        except Exception as e:
            self.log_test("Sepsis dataset loading", 'fail', str(e))

    def test_sepsis_schema_compliance(self):
        """Test Sepsis UnifiedCohort schema compliance"""
        print("\n[6] Sepsis UnifiedCohort Schema Compliance")
        print("-" * 70)

        try:
            dataset = SepsisDataset()

            if not dataset.validate():
                self.log_test("Sepsis schema compliance", 'warn', "Data not available")
                return

            cohort = dataset.get_cohort()

            # Check required columns
            missing_cols = []
            for col in UnifiedCohort.REQUIRED_COLUMNS:
                if col not in cohort.columns:
                    missing_cols.append(col)

            if not missing_cols:
                self.log_test("Sepsis schema compliance", 'pass')
            else:
                self.log_test(
                    "Sepsis required columns",
                    'fail',
                    f"Missing: {missing_cols}"
                )

        except Exception as e:
            self.log_test("Sepsis schema compliance", 'fail', str(e))

    # Statistical Analysis Tests
    def test_logistic_regression(self):
        """Test logistic regression functionality"""
        print("\n[7] Logistic Regression Analysis")
        print("-" * 70)

        try:
            # Use COVID-MS data for testing
            dataset = CovidMSDataset()

            if not dataset.validate():
                self.log_test("Logistic regression test", 'warn', "No data for testing")
                return

            cohort = dataset.get_cohort()

            # Prepare data
            predictors = ['age_group', 'sex']
            analysis_data = cohort[[UnifiedCohort.OUTCOME] + predictors].copy()

            # Convert categorical to dummies
            analysis_data = pd.get_dummies(
                analysis_data,
                columns=['age_group', 'sex'],
                drop_first=True
            )
            analysis_data = analysis_data.dropna()

            if len(analysis_data) < 10:
                self.log_test(
                    "Logistic regression",
                    'warn',
                    "Insufficient data after cleaning"
                )
                return

            # Get updated predictor names after dummy encoding
            predictor_cols = [c for c in analysis_data.columns if c != UnifiedCohort.OUTCOME]

            # Run regression
            model, summary_df = run_logistic_regression(
                analysis_data,
                UnifiedCohort.OUTCOME,
                predictor_cols
            )

            self.log_test("Logistic regression executes successfully", 'pass')

            # Check model output
            if hasattr(model, 'prsquared'):
                self.log_test(
                    f"Model Pseudo R² = {model.prsquared:.4f}",
                    'pass'
                )

            # Check summary DataFrame
            required_summary_cols = ['Odds Ratio', 'CI Lower', 'CI Upper', 'P-Value']
            if all(col in summary_df.columns for col in required_summary_cols):
                self.log_test("Summary DataFrame has required columns", 'pass')
            else:
                self.log_test(
                    "Summary DataFrame columns",
                    'fail',
                    "Missing required columns"
                )

        except Exception as e:
            self.log_test("Logistic regression", 'fail', str(e))

    # Schema Tests
    def test_unified_cohort_schema(self):
        """Test UnifiedCohort schema definition"""
        print("\n[8] UnifiedCohort Schema Definition")
        print("-" * 70)

        try:
            # Check required constants exist
            required_attrs = [
                'PATIENT_ID',
                'TIME_ZERO',
                'OUTCOME',
                'OUTCOME_LABEL',
                'REQUIRED_COLUMNS'
            ]

            missing_attrs = []
            for attr in required_attrs:
                if not hasattr(UnifiedCohort, attr):
                    missing_attrs.append(attr)

            if not missing_attrs:
                self.log_test("UnifiedCohort schema attributes defined", 'pass')
            else:
                self.log_test(
                    "UnifiedCohort schema",
                    'fail',
                    f"Missing: {missing_attrs}"
                )

            # Check REQUIRED_COLUMNS is a list
            if isinstance(UnifiedCohort.REQUIRED_COLUMNS, list):
                self.log_test("REQUIRED_COLUMNS is a list", 'pass')
                self.log_test(
                    f"Required columns: {', '.join(UnifiedCohort.REQUIRED_COLUMNS)}",
                    'pass'
                )
            else:
                self.log_test(
                    "REQUIRED_COLUMNS type",
                    'fail',
                    f"Expected list, got {type(UnifiedCohort.REQUIRED_COLUMNS)}"
                )

        except Exception as e:
            self.log_test("UnifiedCohort schema", 'fail', str(e))


def main():
    """Main validation entry point"""
    validator = PlatformValidator()
    validator.run_all_tests()


if __name__ == "__main__":
    main()

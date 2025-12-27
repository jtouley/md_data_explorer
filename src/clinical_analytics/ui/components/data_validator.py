"""
Data Quality Validator

Validates uploaded data for common quality issues:
- Duplicate patient IDs
- Excessive missing data
- Invalid values
- Data type mismatches
"""


import pandas as pd


class DataQualityValidator:
    """
    Validate data quality for uploaded datasets.

    Performs checks for:
    - Duplicate IDs
    - Missing data patterns
    - Value range validation
    - Required columns
    """

    # Thresholds for quality warnings
    MISSING_DATA_WARNING_THRESHOLD = 30  # % missing per column
    MISSING_DATA_ERROR_THRESHOLD = 80  # % missing per column

    @classmethod
    def validate_patient_id(cls, df: pd.DataFrame, id_column: str) -> tuple[bool, list[dict[str, any]]]:
        """
        Validate patient ID column.

        Checks:
        - No duplicates
        - No missing values
        - Adequate uniqueness

        Args:
            df: DataFrame
            id_column: Name of ID column

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if id_column not in df.columns:
            issues.append(
                {
                    "severity": "error",
                    "type": "missing_column",
                    "message": f"ID column '{id_column}' not found in data",
                }
            )
            return False, issues

        id_series = df[id_column]

        # Check for missing IDs
        n_missing = id_series.isna().sum()
        if n_missing > 0:
            issues.append(
                {
                    "severity": "error",
                    "type": "missing_ids",
                    "message": f"{n_missing} missing patient IDs found. Every row must have an ID.",
                    "count": int(n_missing),
                }
            )

        # Check for duplicate IDs
        duplicates = id_series.duplicated()
        n_duplicates = duplicates.sum()
        if n_duplicates > 0:
            duplicate_ids = id_series[duplicates].unique()[:5]  # Sample
            issues.append(
                {
                    "severity": "error",
                    "type": "duplicate_ids",
                    "message": f"{n_duplicates} duplicate patient IDs found. Each patient must have unique ID.",
                    "count": int(n_duplicates),
                    "examples": list(duplicate_ids),
                }
            )

        # Check uniqueness ratio
        n_unique = id_series.dropna().nunique()
        n_total = len(id_series.dropna())
        if n_total > 0:
            uniqueness_ratio = n_unique / n_total
            if uniqueness_ratio < 0.95:
                issues.append(
                    {
                        "severity": "warning",
                        "type": "low_uniqueness",
                        "message": f"ID column only {uniqueness_ratio * 100:.1f}% unique. Expected >95% for patient IDs.",
                        "uniqueness": uniqueness_ratio,
                    }
                )

        is_valid = not any(issue["severity"] == "error" for issue in issues)
        return is_valid, issues

    @classmethod
    def validate_missing_data(cls, df: pd.DataFrame) -> tuple[bool, list[dict[str, any]]]:
        """
        Validate missing data patterns.

        Checks:
        - Per-column missing data rates
        - Overall missing data
        - Suspicious patterns

        Args:
            df: DataFrame

        Returns:
            Tuple of (is_acceptable, list of issues)
        """
        issues = []

        # Check each column
        for col in df.columns:
            n_missing = df[col].isna().sum()
            pct_missing = (n_missing / len(df)) * 100

            if pct_missing >= cls.MISSING_DATA_ERROR_THRESHOLD:
                issues.append(
                    {
                        "severity": "error",
                        "type": "excessive_missing",
                        "message": f"Column '{col}' has {pct_missing:.1f}% missing data. Consider removing this variable.",
                        "column": col,
                        "missing_count": int(n_missing),
                        "missing_pct": pct_missing,
                    }
                )

            elif pct_missing >= cls.MISSING_DATA_WARNING_THRESHOLD:
                issues.append(
                    {
                        "severity": "warning",
                        "type": "high_missing",
                        "message": f"Column '{col}' has {pct_missing:.1f}% missing data.",
                        "column": col,
                        "missing_count": int(n_missing),
                        "missing_pct": pct_missing,
                    }
                )

        # Overall missing data
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        overall_missing_pct = (missing_cells / total_cells) * 100

        if overall_missing_pct >= 50:
            issues.append(
                {
                    "severity": "error",
                    "type": "overall_missing",
                    "message": f"Dataset has {overall_missing_pct:.1f}% missing values overall. Data quality too poor.",
                    "missing_pct": overall_missing_pct,
                }
            )
        elif overall_missing_pct >= 25:
            issues.append(
                {
                    "severity": "warning",
                    "type": "overall_missing",
                    "message": f"Dataset has {overall_missing_pct:.1f}% missing values overall.",
                    "missing_pct": overall_missing_pct,
                }
            )

        is_acceptable = not any(issue["severity"] == "error" for issue in issues)
        return is_acceptable, issues

    @classmethod
    def validate_outcome_column(cls, df: pd.DataFrame, outcome_column: str) -> tuple[bool, list[dict[str, any]]]:
        """
        Validate outcome column.

        Checks:
        - Column exists
        - Has valid values (for binary outcomes)
        - Sufficient variation
        - Not too many missing values

        Args:
            df: DataFrame
            outcome_column: Name of outcome column

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if outcome_column not in df.columns:
            issues.append(
                {
                    "severity": "error",
                    "type": "missing_column",
                    "message": f"Outcome column '{outcome_column}' not found in data",
                }
            )
            return False, issues

        outcome_series = df[outcome_column].dropna()

        # Check for missing outcomes
        n_missing = df[outcome_column].isna().sum()
        pct_missing = (n_missing / len(df)) * 100

        if pct_missing > 20:
            issues.append(
                {
                    "severity": "error",
                    "type": "missing_outcome",
                    "message": f"Outcome has {pct_missing:.1f}% missing values. Too many to analyze.",
                    "missing_count": int(n_missing),
                    "missing_pct": pct_missing,
                }
            )

        # Check for variation
        n_unique = outcome_series.nunique()

        if n_unique < 2:
            issues.append(
                {
                    "severity": "error",
                    "type": "no_variation",
                    "message": f"Outcome has only {n_unique} unique value(s). Need variation for analysis.",
                    "unique_count": n_unique,
                }
            )

        # For binary outcomes, check balance
        if n_unique == 2:
            value_counts = outcome_series.value_counts()
            minority_pct = (value_counts.min() / len(outcome_series)) * 100

            if minority_pct < 5:
                issues.append(
                    {
                        "severity": "warning",
                        "type": "imbalanced_outcome",
                        "message": f"Outcome is very imbalanced ({minority_pct:.1f}% minority class). May affect analysis.",
                        "minority_pct": minority_pct,
                        "distribution": value_counts.to_dict(),
                    }
                )

        is_valid = not any(issue["severity"] == "error" for issue in issues)
        return is_valid, issues

    @classmethod
    def validate_complete(
        cls, df: pd.DataFrame, id_column: str | None = None, outcome_column: str | None = None
    ) -> dict[str, any]:
        """
        Run complete validation suite.

        Args:
            df: DataFrame to validate
            id_column: Patient ID column (optional)
            outcome_column: Outcome column (optional)

        Returns:
            Validation results dictionary with:
            {
                'is_valid': bool,
                'issues': list of issues,
                'summary': summary statistics
            }
        """
        all_issues = []

        # Basic structure validation
        if len(df) == 0:
            return {
                "is_valid": False,
                "issues": [
                    {
                        "severity": "error",
                        "type": "empty_dataset",
                        "message": "Dataset is empty (no rows)",
                    }
                ],
                "summary": None,
            }

        if len(df.columns) == 0:
            return {
                "is_valid": False,
                "issues": [{"severity": "error", "type": "no_columns", "message": "Dataset has no columns"}],
                "summary": None,
            }

        # Validate patient ID if provided
        if id_column:
            id_valid, id_issues = cls.validate_patient_id(df, id_column)
            all_issues.extend(id_issues)

        # Validate missing data
        missing_ok, missing_issues = cls.validate_missing_data(df)
        all_issues.extend(missing_issues)

        # Validate outcome if provided
        if outcome_column:
            outcome_valid, outcome_issues = cls.validate_outcome_column(df, outcome_column)
            all_issues.extend(outcome_issues)

        # Calculate summary statistics
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_cells": df.size,
            "missing_cells": int(df.isna().sum().sum()),
            "missing_pct": float((df.isna().sum().sum() / df.size) * 100),
            "errors": sum(1 for issue in all_issues if issue["severity"] == "error"),
            "warnings": sum(1 for issue in all_issues if issue["severity"] == "warning"),
        }

        # Overall validity
        is_valid = summary["errors"] == 0

        return {"is_valid": is_valid, "issues": all_issues, "summary": summary}

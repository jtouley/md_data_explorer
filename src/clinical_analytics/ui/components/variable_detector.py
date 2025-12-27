"""
Variable Type Detector

Automatically detects variable types from uploaded data:
- Continuous (numeric)
- Categorical (string/object with limited unique values)
- Binary (2 unique values)
- Date/Time
"""

import pandas as pd


class VariableTypeDetector:
    """
    Automatically detect variable types from uploaded data.

    Uses heuristics based on:
    - Data type
    - Number of unique values
    - Value patterns
    - Column names
    """

    # Threshold for categorical vs continuous
    CATEGORICAL_THRESHOLD = 20  # If unique values <= this, likely categorical

    # Common patterns for specific variable types
    ID_PATTERNS = ["id", "patient", "subject", "record", "mrn"]
    OUTCOME_PATTERNS = ["outcome", "result", "death", "died", "survived", "status", "event"]
    TIME_PATTERNS = ["date", "time", "timestamp", "admission", "discharge", "dob", "visit"]
    BINARY_VALUES = {
        "yes/no": (["yes", "no"], ["Yes", "No"]),
        "1/0": ([1, 0], ["1", "0"]),
        "true/false": (["true", "false"], ["True", "False"], [True, False]),
        "male/female": (["male", "female"], ["Male", "Female"], ["M", "F"]),
        "alive/dead": (["alive", "dead"], ["Alive", "Dead"]),
    }

    @classmethod
    def detect_variable_type(
        cls, series: pd.Series, column_name: str
    ) -> tuple[str, dict[str, any]]:
        """
        Detect variable type for a single column.

        Args:
            series: Pandas Series
            column_name: Column name (used for pattern matching)

        Returns:
            Tuple of (variable_type, metadata_dict)
            where variable_type is one of: 'binary', 'categorical', 'continuous', 'datetime', 'id'
        """
        # Clean column name for pattern matching
        col_lower = column_name.lower().strip()

        # Count unique values (excluding NaN)
        unique_values = series.dropna().unique()
        n_unique = len(unique_values)
        n_total = len(series.dropna())

        # Check for ID column first (high cardinality, often string)
        if any(pattern in col_lower for pattern in cls.ID_PATTERNS):
            if n_unique / n_total > 0.95:  # 95%+ unique values
                return "id", {"unique_count": n_unique, "suggested_as_patient_id": True}

        # Check for datetime
        if any(pattern in col_lower for pattern in cls.TIME_PATTERNS):
            # Try parsing as datetime
            try:
                pd.to_datetime(series.dropna(), errors="coerce")
                return "datetime", {"format": "auto-detected", "requires_parsing": True}
            except:
                pass

        # Explicit datetime type
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime", {"format": "datetime", "requires_parsing": False}

        # Check for binary
        if n_unique == 2:
            values_list = [str(v).lower() for v in unique_values]

            # Check if matches known binary patterns
            binary_type = None
            for pattern_name, pattern_lists in cls.BINARY_VALUES.items():
                for pattern in pattern_lists:
                    pattern_lower = [str(p).lower() for p in pattern]
                    if set(values_list) == set(pattern_lower):
                        binary_type = pattern_name
                        break
                if binary_type:
                    break

            # Check if this might be an outcome
            is_outcome = any(pattern in col_lower for pattern in cls.OUTCOME_PATTERNS)

            return "binary", {
                "values": list(unique_values),
                "pattern": binary_type or "custom",
                "suggested_as_outcome": is_outcome,
            }

        # Check for categorical (limited unique values)
        if n_unique <= cls.CATEGORICAL_THRESHOLD:
            # Numeric with few unique values
            if pd.api.types.is_numeric_dtype(series):
                # Could be ordinal (0,1,2,3) or categorical
                return "categorical", {
                    "unique_count": n_unique,
                    "values": sorted(unique_values.tolist()),
                    "numeric_categorical": True,
                }

            # String/object categorical
            return "categorical", {
                "unique_count": n_unique,
                "values": sorted(unique_values.tolist()),
                "numeric_categorical": False,
            }

        # Check for continuous (numeric)
        if pd.api.types.is_numeric_dtype(series):
            return "continuous", {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
            }

        # Default to categorical for high-cardinality string columns
        return "categorical", {
            "unique_count": n_unique,
            "high_cardinality": True,
            "values": sorted(unique_values[:10].tolist()),  # Sample values
        }

    @classmethod
    def detect_all_variables(cls, df: pd.DataFrame) -> dict[str, dict[str, any]]:
        """
        Detect variable types for all columns in a DataFrame.

        Args:
            df: Pandas DataFrame

        Returns:
            Dictionary mapping column names to type info:
            {
                'column_name': {
                    'type': 'continuous',
                    'metadata': {...},
                    'suggested_role': 'predictor|outcome|id|time'
                }
            }
        """
        results = {}

        for col in df.columns:
            var_type, metadata = cls.detect_variable_type(df[col], col)

            # Suggest role based on type and metadata
            suggested_role = "predictor"  # Default

            if var_type == "id":
                suggested_role = "patient_id"
            elif var_type == "datetime":
                suggested_role = "time_variable"
            elif metadata.get("suggested_as_outcome", False):
                suggested_role = "outcome"

            results[col] = {
                "type": var_type,
                "metadata": metadata,
                "suggested_role": suggested_role,
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": float(df[col].isna().sum() / len(df) * 100),
            }

        return results

    @classmethod
    def suggest_schema_mapping(cls, df: pd.DataFrame) -> dict[str, str | None]:
        """
        Suggest mapping to UnifiedCohort schema.

        Args:
            df: Pandas DataFrame

        Returns:
            Dictionary with suggested mappings:
            {
                'patient_id': 'suggested_column' or None,
                'outcome': 'suggested_column' or None,
                'time_zero': 'suggested_column' or None
            }
        """
        variable_info = cls.detect_all_variables(df)

        suggestions = {"patient_id": None, "outcome": None, "time_zero": None}

        # Find patient ID candidate
        id_candidates = [
            col for col, info in variable_info.items() if info["suggested_role"] == "patient_id"
        ]
        if id_candidates:
            suggestions["patient_id"] = id_candidates[0]

        # Find outcome candidate
        outcome_candidates = [
            col for col, info in variable_info.items() if info["suggested_role"] == "outcome"
        ]
        if outcome_candidates:
            suggestions["outcome"] = outcome_candidates[0]

        # Find time variable candidate
        time_candidates = [
            col for col, info in variable_info.items() if info["suggested_role"] == "time_variable"
        ]
        if time_candidates:
            suggestions["time_zero"] = time_candidates[0]

        return suggestions

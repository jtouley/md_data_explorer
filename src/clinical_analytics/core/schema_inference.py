"""
Automatic Schema Inference Engine - Zero Manual Configuration

This module implements intelligent schema detection for clinical datasets using
only Polars and DuckDB operations (no pandas). Automatically detects:
- Patient ID columns (unique identifiers)
- Outcome variables (binary clinical endpoints)
- Time variables (dates, survival time)
- Event indicators (censoring status)
- Categorical vs continuous variables

Key Principles:
- Use Polars for all DataFrame operations
- Use DuckDB for SQL-based analysis
- Privacy-preserving (all local, no API calls)
- Fail gracefully with sensible defaults
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import polars as pl
import duckdb
import re


@dataclass
class DictionaryMetadata:
    """
    Metadata extracted from data dictionary PDF.

    This provides ground truth context about column meanings,
    valid values, and clinical significance.

    Attributes:
        column_descriptions: Dict mapping column names to descriptions
        column_types: Dict mapping column names to expected data types
        valid_values: Dict mapping column names to valid value sets
        source_file: Path to the PDF dictionary
    """
    column_descriptions: Dict[str, str] = field(default_factory=dict)
    column_types: Dict[str, str] = field(default_factory=dict)
    valid_values: Dict[str, List[str]] = field(default_factory=dict)
    source_file: Optional[Path] = None

    def get_description(self, column: str) -> Optional[str]:
        """Get description for a column, case-insensitive."""
        col_lower = column.lower()
        for key, value in self.column_descriptions.items():
            if key.lower() == col_lower:
                return value
        return None


@dataclass
class InferredSchema:
    """
    Complete inferred schema for a clinical dataset.

    This represents the automatically detected structure of a dataset,
    eliminating the need for manual YAML configuration files.

    Attributes:
        patient_id_column: Unique identifier column (e.g., 'patient_id', 'subject_id')
        time_zero: Reference time column or static value
        outcome_columns: Binary outcome variables (mortality, ICU, etc.)
        time_columns: Time-related columns (dates, survival time)
        event_columns: Event indicator columns (censoring status)
        categorical_columns: Categorical variables
        continuous_columns: Continuous numeric variables
        confidence_scores: Detection confidence for each field (0-1)
    """
    patient_id_column: Optional[str] = None
    time_zero: Optional[str] = None
    outcome_columns: List[str] = field(default_factory=list)
    time_columns: List[str] = field(default_factory=list)
    event_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    continuous_columns: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    dictionary_metadata: Optional[DictionaryMetadata] = None

    def to_dataset_config(self) -> Dict[str, Any]:
        """
        Convert inferred schema to dataset config format.

        This generates the config dict that would have been manually written
        in YAML files, enabling seamless integration with existing systems.

        Returns:
            Config dictionary compatible with ClinicalDataset
        """
        config = {
            'column_mapping': {},
            'outcomes': {},
            'time_zero': {}
        }

        # Map patient ID
        if self.patient_id_column:
            config['column_mapping'][self.patient_id_column] = 'patient_id'

        # Map outcomes with descriptions from dictionary
        for outcome_col in self.outcome_columns:
            outcome_config = {
                'source_column': outcome_col,
                'type': 'binary',
                'confidence': self.confidence_scores.get(f'outcome_{outcome_col}', 0.5)
            }

            # Add description from dictionary if available
            if self.dictionary_metadata:
                desc = self.dictionary_metadata.get_description(outcome_col)
                if desc:
                    outcome_config['description'] = desc

            config['outcomes'][outcome_col] = outcome_config

        # Time zero
        if self.time_zero:
            if self.time_zero in self.time_columns:
                config['time_zero'] = {'source_column': self.time_zero}
            else:
                config['time_zero'] = {'value': self.time_zero}

        return config

    def summary(self) -> str:
        """Generate human-readable summary of inferred schema."""
        lines = ["=== Inferred Schema Summary ==="]

        if self.patient_id_column:
            conf = self.confidence_scores.get('patient_id', 0.0)
            lines.append(f"Patient ID: {self.patient_id_column} (confidence: {conf:.2f})")

        if self.outcome_columns:
            lines.append(f"Outcomes ({len(self.outcome_columns)}): {', '.join(self.outcome_columns)}")

        if self.time_columns:
            lines.append(f"Time columns ({len(self.time_columns)}): {', '.join(self.time_columns)}")

        if self.event_columns:
            lines.append(f"Event columns ({len(self.event_columns)}): {', '.join(self.event_columns)}")

        lines.append(f"Categorical: {len(self.categorical_columns)} columns")
        lines.append(f"Continuous: {len(self.continuous_columns)} columns")

        return "\n".join(lines)


class SchemaInferenceEngine:
    """
    Automatic schema inference for any tabular clinical dataset.

    Replaces hardcoded YAML configs with intelligent pattern recognition
    using Polars and DuckDB operations only.

    Detection Strategies:
    1. Pattern matching on column names
    2. Statistical analysis of value distributions
    3. Cardinality-based classification
    4. Data type analysis

    Example:
        >>> engine = SchemaInferenceEngine()
        >>> df = pl.read_csv("patient_data.csv")
        >>> schema = engine.infer_schema(df)
        >>> print(schema.summary())
    """

    # Pattern keywords for different column types
    PATIENT_ID_PATTERNS: Set[str] = {
        'patient_id', 'patientid', 'patient', 'id', 'subject_id',
        'subjectid', 'subject', 'mrn', 'study_id', 'studyid', 'participant_id'
    }

    OUTCOME_PATTERNS: Set[str] = {
        'outcome', 'death', 'mortality', 'died', 'deceased', 'expired',
        'icu', 'hospitalized', 'hospitalisation', 'readmit', 'complication',
        'adverse', 'event', 'endpoint', 'status', 'relapse'
    }

    TIME_PATTERNS: Set[str] = {
        'time', 'date', 'day', 'month', 'year', 'duration',
        'survival', 'followup', 'follow_up', 'days_to', 'time_to'
    }

    EVENT_PATTERNS: Set[str] = {
        'event', 'status', 'censor', 'indicator', 'flag', 'occurred'
    }

    def __init__(self):
        """Initialize schema inference engine."""
        pass

    def infer_schema(self, df: pl.DataFrame) -> InferredSchema:
        """
        Infer complete schema from Polars DataFrame.

        Args:
            df: Raw Polars DataFrame to analyze

        Returns:
            InferredSchema with all detected columns and confidence scores

        Example:
            >>> df = pl.read_csv("data.csv")
            >>> schema = engine.infer_schema(df)
            >>> config = schema.to_dataset_config()
        """
        schema = InferredSchema()

        # 1. Detect patient ID column (highest priority)
        schema.patient_id_column, conf = self._detect_patient_id(df)
        if schema.patient_id_column:
            schema.confidence_scores['patient_id'] = conf

        # 2. Detect outcome columns (binary variables with clinical names)
        schema.outcome_columns = self._detect_outcomes(df)
        for col in schema.outcome_columns:
            schema.confidence_scores[f'outcome_{col}'] = 0.8  # Default confidence

        # 3. Detect time columns (dates, numeric time values)
        schema.time_columns = self._detect_time_columns(df)

        # 4. Detect event columns (binary, often paired with time columns)
        schema.event_columns = self._detect_event_columns(df)

        # 5. Classify remaining columns as categorical or continuous
        classified = {schema.patient_id_column} | set(schema.outcome_columns) | \
                     set(schema.time_columns) | set(schema.event_columns)

        for col in df.columns:
            if col in classified:
                continue

            if self._is_categorical(df, col):
                schema.categorical_columns.append(col)
            else:
                schema.continuous_columns.append(col)

        # 6. Infer time_zero (earliest date column or static)
        if schema.time_columns:
            # Use earliest date as time_zero
            schema.time_zero = schema.time_columns[0]
        else:
            # No time columns - use static time_zero
            schema.time_zero = "2020-01-01"

        return schema

    def _detect_patient_id(self, df: pl.DataFrame) -> tuple[Optional[str], float]:
        """
        Detect patient ID column using Polars operations.

        Strategy:
        1. Check name patterns (patient_id, subject_id, etc.)
        2. Verify high uniqueness (>95% unique values)
        3. Check for null values (ID should have none)

        Args:
            df: Polars DataFrame

        Returns:
            Tuple of (column_name, confidence_score)
        """
        for col in df.columns:
            col_lower = col.lower()

            # Check name patterns
            if any(pattern in col_lower for pattern in self.PATIENT_ID_PATTERNS):
                # Verify high uniqueness
                unique_count = df[col].n_unique()
                total_count = df.height
                uniqueness_ratio = unique_count / total_count if total_count > 0 else 0

                # Check for nulls
                null_count = df[col].null_count()

                if uniqueness_ratio > 0.95 and null_count == 0:
                    return col, 0.95
                elif uniqueness_ratio > 0.9:
                    return col, 0.85

        # Fallback: highest cardinality column with no nulls
        best_col = None
        best_ratio = 0.0

        for col in df.columns:
            unique_count = df[col].n_unique()
            total_count = df.height
            null_count = df[col].null_count()

            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0

            if null_count == 0 and uniqueness_ratio > best_ratio:
                best_ratio = uniqueness_ratio
                best_col = col

        if best_ratio > 0.95:
            return best_col, 0.7  # Lower confidence for fallback

        return None, 0.0

    def _detect_outcomes(self, df: pl.DataFrame) -> List[str]:
        """
        Detect outcome columns (binary with clinical names).

        Strategy:
        1. Must be binary (2 unique values)
        2. Check name patterns (mortality, icu, etc.)
        3. Prefer 0/1 or True/False encoding

        Args:
            df: Polars DataFrame

        Returns:
            List of outcome column names
        """
        outcome_cols = []

        for col in df.columns:
            col_lower = col.lower()

            # Must be binary (2 unique non-null values)
            unique_count = df[col].n_unique()

            if unique_count != 2:
                continue

            # Check name patterns
            if any(pattern in col_lower for pattern in self.OUTCOME_PATTERNS):
                outcome_cols.append(col)

        return outcome_cols

    def _detect_time_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Detect time-related columns using Polars dtype checking.

        Strategy:
        1. Check for datetime dtypes
        2. Check name patterns (time, date, duration, etc.)
        3. For numeric columns, check if values look like time (positive integers)

        Args:
            df: Polars DataFrame

        Returns:
            List of time column names
        """
        time_cols = []

        for col in df.columns:
            col_lower = col.lower()
            dtype = df[col].dtype

            # Check if datetime type
            if dtype in [pl.Datetime, pl.Date]:
                time_cols.append(col)
                continue

            # Check name patterns
            if any(pattern in col_lower for pattern in self.TIME_PATTERNS):
                time_cols.append(col)

        return time_cols

    def _detect_event_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Detect event indicator columns (binary, often paired with time).

        Strategy:
        1. Must be binary
        2. Check name patterns (event, status, censor, etc.)
        3. Often appears near time columns

        Args:
            df: Polars DataFrame

        Returns:
            List of event column names
        """
        event_cols = []

        for col in df.columns:
            col_lower = col.lower()

            # Must be binary
            unique_count = df[col].n_unique()

            if unique_count != 2:
                continue

            # Check name patterns
            if any(pattern in col_lower for pattern in self.EVENT_PATTERNS):
                event_cols.append(col)

        return event_cols

    def _is_categorical(self, df: pl.DataFrame, col: str) -> bool:
        """
        Classify column as categorical or continuous using Polars.

        Strategy:
        1. String columns are categorical
        2. Numeric columns with ≤20 unique values are categorical
        3. Boolean columns are categorical

        Args:
            df: Polars DataFrame
            col: Column name

        Returns:
            True if categorical, False if continuous
        """
        dtype = df[col].dtype

        # String columns are categorical
        if dtype == pl.Utf8 or dtype == pl.Categorical:
            return True

        # Boolean columns are categorical
        if dtype == pl.Boolean:
            return True

        # Numeric columns: check cardinality
        if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64]:
            unique_count = df[col].n_unique()
            return unique_count <= 20

        # Default to categorical for unknown types
        return True

    def parse_dictionary_pdf(self, pdf_path: Path) -> Optional[DictionaryMetadata]:
        """
        Parse data dictionary PDF to extract column metadata using LangChain.

        Uses LangChain's PyPDFLoader for better text extraction and pattern matching to find:
        - Column names
        - Descriptions
        - Data types
        - Valid values

        Args:
            pdf_path: Path to data dictionary PDF

        Returns:
            DictionaryMetadata with extracted information, or None if parsing fails

        Note:
            This is a best-effort extraction. PDF structure varies widely,
            so we use heuristics to identify column documentation.
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            print("Warning: LangChain not installed. Install with: uv add langchain langchain-community")
            return None

        if not pdf_path.exists():
            return None

        metadata = DictionaryMetadata(source_file=pdf_path)

        try:
            # Load PDF using LangChain
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Combine text from all pages
            full_text = "\n".join([page.page_content for page in pages])

            # Pattern 1: "ColumnName: Description" or "ColumnName - Description"
            pattern1 = r'^([a-z_][a-z0-9_]*)\s*[:\-]\s*(.+)$'

            # Pattern 2: "Variable Name: Description" format
            pattern2 = r'(?:variable|column|field)\s+(?:name|id)?\s*[:\-]?\s*([a-z_][a-z0-9_]*)\s*(?:description|meaning)?[:\-]\s*(.+)$'

            for line in full_text.split('\n'):
                line = line.strip()

                if not line:
                    continue

                # Try pattern 1
                match = re.match(pattern1, line, re.IGNORECASE)
                if match:
                    col_name = match.group(1).lower()
                    description = match.group(2).strip()

                    # Skip if description is too short (likely not a real description)
                    if len(description) > 10:
                        metadata.column_descriptions[col_name] = description
                    continue

                # Try pattern 2
                match = re.match(pattern2, line, re.IGNORECASE)
                if match:
                    col_name = match.group(1).lower()
                    description = match.group(2).strip()

                    if len(description) > 10:
                        metadata.column_descriptions[col_name] = description

            # Extract data types if mentioned
            type_patterns = {
                'integer': ['integer', 'int', 'numeric', 'number'],
                'float': ['float', 'decimal', 'real', 'double'],
                'string': ['string', 'text', 'varchar', 'char'],
                'date': ['date', 'datetime', 'timestamp'],
                'boolean': ['boolean', 'bool', 'binary', 'yes/no']
            }

            for col_name, desc in metadata.column_descriptions.items():
                desc_lower = desc.lower()

                for dtype, keywords in type_patterns.items():
                    if any(kw in desc_lower for kw in keywords):
                        metadata.column_types[col_name] = dtype
                        break

            return metadata if metadata.column_descriptions else None

        except Exception as e:
            print(f"Error parsing PDF dictionary: {e}")
            return None

    def infer_schema_with_dictionary(
        self,
        df: pl.DataFrame,
        dictionary_path: Optional[Path] = None
    ) -> InferredSchema:
        """
        Infer schema from DataFrame and merge with PDF dictionary metadata.

        This is the enhanced version that provides richer context by combining:
        1. DataFrame analysis (data-driven inference)
        2. PDF dictionary (documentation-driven context)

        Args:
            df: Polars DataFrame to analyze
            dictionary_path: Optional path to data dictionary PDF

        Returns:
            InferredSchema enriched with dictionary metadata

        Example:
            >>> df = pl.read_csv("patient_data.csv")
            >>> dict_path = Path("data/dictionaries/patient_data_dict.pdf")
            >>> schema = engine.infer_schema_with_dictionary(df, dict_path)
            >>> print(schema.summary())  # Shows DataFrame + dictionary context
        """
        # First, do standard DataFrame-based inference
        schema = self.infer_schema(df)

        # Then, try to merge with dictionary metadata
        if dictionary_path:
            dict_metadata = self.parse_dictionary_pdf(dictionary_path)

            if dict_metadata:
                schema.dictionary_metadata = dict_metadata

                # Use dictionary hints to improve detection confidence
                for col in df.columns:
                    col_desc = dict_metadata.get_description(col)

                    if not col_desc:
                        continue

                    desc_lower = col_desc.lower()

                    # Check if dictionary suggests this is an outcome
                    outcome_keywords = ['outcome', 'mortality', 'death', 'died', 'icu', 'event']
                    if any(kw in desc_lower for kw in outcome_keywords):
                        if col not in schema.outcome_columns:
                            # Dictionary suggests this is an outcome
                            if df[col].n_unique() == 2:  # Verify it's binary
                                schema.outcome_columns.append(col)
                                schema.confidence_scores[f'outcome_{col}'] = 0.95  # High confidence from dictionary

                    # Check if dictionary suggests this is a time variable
                    time_keywords = ['time', 'date', 'duration', 'days', 'months', 'years']
                    if any(kw in desc_lower for kw in time_keywords):
                        if col not in schema.time_columns:
                            schema.time_columns.append(col)

        return schema

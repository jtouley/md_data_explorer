"""
MIMIC-III Dataset Definition - DuckDB/Postgres-based implementation.

Config-driven implementation for MIMIC-III Clinical Database.
"""

from pathlib import Path
import pandas as pd
import polars as pl
from typing import Optional

from clinical_analytics.core.dataset import ClinicalDataset
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.core.mapper import ColumnMapper, load_dataset_config
from clinical_analytics.datasets.mimic3.loader import MIMIC3Loader


class Mimic3Dataset(ClinicalDataset):
    """
    Implementation for MIMIC-III Clinical Database.

    Uses DuckDB for SQL-based data extraction.
    Config-driven via datasets.yaml.
    """

    def __init__(self, db_path: Optional[str] = None, db_connection=None):
        # Load config FIRST - config-driven!
        self.config = load_dataset_config('mimic3')

        # Use config default if not provided
        if db_path is None and db_connection is None:
            db_path = self.config['init_params'].get('db_path')

        super().__init__(name="mimic3", source_path=None, db_connection=db_connection)

        self.db_path = Path(db_path) if db_path else None
        self._data: Optional[pl.DataFrame] = None

        # Initialize config-driven mapper
        self.mapper = ColumnMapper(self.config)

        # Initialize loader
        self.loader = MIMIC3Loader(db_path=self.db_path, db_connection=db_connection)

    def validate(self) -> bool:
        """
        Validate database connection and required tables.

        Returns:
            True if database accessible and has required tables
        """
        try:
            self.loader.connect()

            # Check if required tables exist
            table_status = self.loader.check_tables_exist()

            # At minimum, need patients and admissions tables
            required = ['patients', 'admissions']
            has_required = all(table_status.get(t, False) for t in required)

            self.loader.disconnect()

            return has_required

        except Exception as e:
            print(f"Validation failed: {e}")
            return False

    def load(self) -> None:
        """
        Load data from database using SQL query from config.

        Query must be specified in config (sql_queries.cohort_extraction).
        """
        if not self.validate():
            print(f"WARNING: Database validation failed. Dataset will be empty.")
            self._data = pl.DataFrame()
            return

        # Get query from config (required, no fallback)
        query = self.config.get('sql_queries', {}).get('cohort_extraction')
        
        if not query:
            raise ValueError(
                "SQL query not found in config. "
                "Please define sql_queries.cohort_extraction in datasets.yaml for mimic3"
            )

        # Load data
        self._data = self.loader.load_cohort(query=query)

        print(f"Loaded {len(self._data)} records from {self.name}")

    def get_cohort(self, **filters) -> pd.DataFrame:
        """
        Return analysis cohort in Pandas format for statsmodels compatibility.

        NOW CONFIG-DRIVEN: All mappings and defaults come from datasets.yaml
        """
        if self._data is None:
            self.load()

        if len(self._data) == 0:
            return pd.DataFrame(columns=UnifiedCohort.REQUIRED_COLUMNS)

        df = self._data.clone()

        # Apply filters using mapper (config-driven)
        # Remove target_outcome from filters (it's not a data filter)
        filter_only = {k: v for k, v in filters.items() if k != "target_outcome"}
        df = self.mapper.apply_filters(df, filter_only)

        # Determine which outcome to use (from config or override)
        outcome_col = filters.get("target_outcome", self.mapper.get_default_outcome())

        # Get config-driven defaults
        time_zero_value = self.mapper.get_time_zero_value()
        outcome_label = self.mapper.get_default_outcome_label(outcome_col)

        # Use mapper to transform to UnifiedCohort schema
        # Since MIMIC-III already has proper column names from SQL,
        # we may need less mapping than file-based datasets
        cohort = self.mapper.map_to_unified_cohort(
            df,
            time_zero_value=time_zero_value,
            outcome_col=outcome_col,
            outcome_label=outcome_label
        )

        # Convert to Pandas for statsmodels compatibility
        return cohort.to_pandas()

    def __del__(self):
        """Cleanup: disconnect loader if still connected."""
        if hasattr(self, 'loader') and self.loader:
            try:
                self.loader.disconnect()
            except:
                pass

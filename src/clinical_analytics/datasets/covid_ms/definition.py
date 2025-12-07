from pathlib import Path
import pandas as pd
import polars as pl
from typing import Optional

from clinical_analytics.core.dataset import ClinicalDataset
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.core.mapper import ColumnMapper, load_dataset_config
from clinical_analytics.datasets.covid_ms.loader import load_raw_data, clean_data

class CovidMSDataset(ClinicalDataset):
    """
    Implementation for the Global Data Sharing Initiative (GDSI)
    COVID-19 and Multiple Sclerosis dataset.

    Uses Polars for efficient ETL, returns Pandas for stats compatibility.
    NOW CONFIG-DRIVEN: Uses datasets.yaml for all mappings and settings.
    """

    def __init__(self, source_path: Optional[str] = None):
        # Load config FIRST - no more hardcoded defaults!
        self.config = load_dataset_config('covid_ms')

        # Use config default if not provided
        if source_path is None:
            source_path = self.config['init_params']['source_path']

        super().__init__(name="covid_ms", source_path=source_path)
        self._data: Optional[pl.DataFrame] = None

        # Initialize config-driven mapper
        self.mapper = ColumnMapper(self.config)

    def validate(self) -> bool:
        if not self.source_path or not self.source_path.exists():
            return False
        # Could add header validation here
        return True

    def load(self) -> None:
        """Load and clean the dataset into memory."""
        if not self.validate():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")
            
        raw_df = load_raw_data(self.source_path)
        # Pass mapper to clean_data for config-driven transformations
        self._data = clean_data(raw_df, mapper=self.mapper)
        print(f"Loaded {len(self._data)} records from {self.name}")

    def get_cohort(self, **filters) -> pd.DataFrame:
        """
        Return analysis cohort in Pandas format for statsmodels compatibility.

        NOW CONFIG-DRIVEN: All mappings, filters, and defaults come from datasets.yaml
        """
        if self._data is None:
            self.load()

        df = self._data.clone()

        # Apply filters from config (merged with overrides)
        default_filters = self.mapper.get_default_filters()
        all_filters = {**default_filters, **filters}

        # Remove target_outcome from filters (it's not a data filter)
        filter_only = {k: v for k, v in all_filters.items() if k != "target_outcome"}

        # Apply filters using mapper (config-driven)
        df = self.mapper.apply_filters(df, filter_only)

        # Determine which outcome to use (from config or override)
        outcome_col = filters.get("target_outcome", self.mapper.get_default_outcome())

        # Get config-driven defaults
        time_zero_value = self.mapper.get_time_zero_value()
        outcome_label = self.mapper.get_default_outcome_label(outcome_col)

        # Use mapper to transform to UnifiedCohort schema
        # This uses datasets.yaml column_mapping instead of hardcoded logic
        cohort = self.mapper.map_to_unified_cohort(
            df,
            time_zero_value=time_zero_value,
            outcome_col=outcome_col,
            outcome_label=outcome_label
        )

        # Convert to Pandas for statsmodels compatibility
        return cohort.to_pandas()


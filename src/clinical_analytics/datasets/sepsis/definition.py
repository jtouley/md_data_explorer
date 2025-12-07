from pathlib import Path
import pandas as pd
import polars as pl
from typing import Optional

from clinical_analytics.core.dataset import ClinicalDataset
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.core.mapper import ColumnMapper, load_dataset_config
from clinical_analytics.datasets.sepsis.loader import load_and_aggregate, find_psv_files

class SepsisDataset(ClinicalDataset):
    """
    Implementation for PhysioNet Challenge 2019 Sepsis Dataset.
    Source: https://physionet.org/content/challenge-2019/

    Uses Polars for efficient PSV processing, returns Pandas for stats compatibility.
    NOW CONFIG-DRIVEN: Uses datasets.yaml for all mappings and settings.
    """

    def __init__(self, source_path: Optional[str] = None):
        # Load config FIRST - no more hardcoded defaults!
        self.config = load_dataset_config('sepsis')

        # Use config default if not provided
        if source_path is None:
            source_path = self.config['init_params']['source_path']

        super().__init__(name="sepsis", source_path=source_path)
        self._data: Optional[pl.DataFrame] = None

        # Initialize config-driven mapper
        self.mapper = ColumnMapper(self.config)

    def validate(self) -> bool:
        if not self.source_path or not self.source_path.exists():
            return False
        # Check if any .psv files exist
        try:
            next(find_psv_files(self.source_path))
            return True
        except StopIteration:
            return False

    def load(self) -> None:
        """Load and aggregate PSV files using Polars."""
        if not self.validate():
            # For the purpose of the scaffold, if validation fails (no data),
            # we don't crash, but warn.
            print(f"WARNING: No PSV files found in {self.source_path}. Sepsis dataset will be empty.")
            self._data = pl.DataFrame()
            return

        # Load with mapper for config-driven aggregation
        self._data = load_and_aggregate(self.source_path, mapper=self.mapper)

    def get_cohort(self, **filters) -> pd.DataFrame:
        """
        Return analysis cohort in Pandas format for statsmodels compatibility.

        NOW CONFIG-DRIVEN: All mappings, filters, and defaults come from datasets.yaml
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
        # This uses datasets.yaml column_mapping instead of hardcoded logic
        cohort = self.mapper.map_to_unified_cohort(
            df,
            time_zero_value=time_zero_value,
            outcome_col=outcome_col,
            outcome_label=outcome_label
        )

        # Convert to Pandas for statsmodels compatibility
        return cohort.to_pandas()


from pathlib import Path
import pandas as pd
import polars as pl
from typing import Optional

from clinical_analytics.core.dataset import ClinicalDataset
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.datasets.sepsis.loader import load_and_aggregate, find_psv_files

class SepsisDataset(ClinicalDataset):
    """
    Implementation for PhysioNet Challenge 2019 Sepsis Dataset.
    Source: https://physionet.org/content/challenge-2019/

    Uses Polars for efficient PSV processing, returns Pandas for stats compatibility.
    """

    def __init__(self, source_path: str = "data/raw/sepsis/"):
        super().__init__(name="sepsis", source_path=source_path)
        self._data: Optional[pl.DataFrame] = None

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

        # Load with a limit for dev/testing if needed, or full
        self._data = load_and_aggregate(self.source_path)

    def get_cohort(self, **filters) -> pd.DataFrame:
        """Return analysis cohort in Pandas format for statsmodels compatibility."""
        if self._data is None:
            self.load()

        if len(self._data) == 0:
            return pd.DataFrame(columns=UnifiedCohort.REQUIRED_COLUMNS)

        df = self._data.clone()

        # Mapping to Unified Schema using Polars
        cohort = df.select([
            pl.col('patient_id').alias(UnifiedCohort.PATIENT_ID),
            pl.lit("2019-01-01").str.strptime(pl.Datetime, "%Y-%m-%d").alias(UnifiedCohort.TIME_ZERO),
            pl.col('sepsis_label').alias(UnifiedCohort.OUTCOME),
            pl.lit("sepsis_occurrence").alias(UnifiedCohort.OUTCOME_LABEL),
            # Features
            pl.col('age'),
            pl.col('gender'),
            pl.col('num_hours')
        ])

        # Convert to Pandas for statsmodels compatibility
        return cohort.to_pandas()


from pathlib import Path
import pandas as pd
import polars as pl
from typing import Optional

from clinical_analytics.core.dataset import ClinicalDataset
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.datasets.covid_ms.loader import load_raw_data, clean_data

class CovidMSDataset(ClinicalDataset):
    """
    Implementation for the Global Data Sharing Initiative (GDSI)
    COVID-19 and Multiple Sclerosis dataset.

    Uses Polars for efficient ETL, returns Pandas for stats compatibility.
    """

    def __init__(self, source_path: str = "data/raw/covid_ms/GDSI_OpenDataset_Final.csv"):
        super().__init__(name="covid_ms", source_path=source_path)
        self._data: Optional[pl.DataFrame] = None

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
        self._data = clean_data(raw_df)
        print(f"Loaded {len(self._data)} records from {self.name}")

    def get_cohort(self, **filters) -> pd.DataFrame:
        """
        Return analysis cohort in Pandas format for statsmodels compatibility.

        Outcome default: Hospitalization (binary).
        Time zero: Set to a dummy date (2020-01-01) since this is a registry without explicit dates.
        """
        if self._data is None:
            self.load()

        df = self._data.clone()

        # Apply filters using Polars expressions
        if filters.get("confirmed_only", True):
            df = df.filter(pl.col('covid19_confirmed_case').str.to_lowercase() == 'yes')

        if "age_group" in filters:
            df = df.filter(pl.col('age_in_cat') == filters['age_group'])

        # Determine outcome column
        outcome_col = filters.get("target_outcome", "outcome_hospitalized")
        if outcome_col not in df.columns:
            if outcome_col == "hospitalization":
                outcome_col = "outcome_hospitalized"
            elif outcome_col == "icu":
                outcome_col = "outcome_icu"

        # Build cohort using Polars select and rename
        cohort = df.select([
            pl.col('secret_name').alias(UnifiedCohort.PATIENT_ID),
            pl.lit("2020-01-01").str.strptime(pl.Datetime, "%Y-%m-%d").alias(UnifiedCohort.TIME_ZERO),
            pl.col(outcome_col).alias(UnifiedCohort.OUTCOME),
            pl.lit(outcome_col).alias(UnifiedCohort.OUTCOME_LABEL),
            # Include relevant covariates with renaming
            pl.col('age_in_cat').alias('age_group'),
            pl.col('sex'),
            pl.col('dmt_type_overall').alias('dmt'),
            pl.col('ms_type2').alias('ms_type'),
            pl.col('has_comorbidities')
        ])

        # Convert to Pandas for statsmodels compatibility
        return cohort.to_pandas()


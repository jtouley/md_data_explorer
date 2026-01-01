import logging

import pandas as pd

from clinical_analytics.core.dataset import ClinicalDataset, Granularity
from clinical_analytics.core.mapper import load_dataset_config
from clinical_analytics.core.semantic import SemanticLayer

logger = logging.getLogger(__name__)


class CovidMSDataset(ClinicalDataset):
    """
    Implementation for the Global Data Sharing Initiative (GDSI)
    COVID-19 and Multiple Sclerosis dataset.

    NOW USES IBIS SEMANTIC LAYER: SQL generated behind the scenes from config.
    No more custom Python mapping logic - just define in YAML and execute.
    """

    def __init__(self, source_path: str | None = None):
        # Load config
        self.config = load_dataset_config("covid_ms")

        # Use config default if not provided
        if source_path is None:
            source_path = self.config["init_params"]["source_path"]

        super().__init__(name="covid_ms", source_path=source_path)

        # Initialize semantic layer (this registers the data source with DuckDB)
        self.semantic = SemanticLayer("covid_ms", config=self.config)

    def validate(self) -> bool:
        """Validate that source data exists."""
        if not self.source_path or not self.source_path.exists():
            return False
        return True

    def load(self) -> None:
        """
        Load is now a no-op - data is queried on-demand via semantic layer.
        The semantic layer registers the CSV with DuckDB but doesn't load it into memory.
        """
        if not self.validate():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")
        # Semantic layer handles registration, nothing to load into memory
        logger.info(f"Semantic layer initialized for {self.name}")

    def get_cohort(self, granularity: Granularity = "patient_level", **filters) -> pd.DataFrame:
        """
        Return analysis cohort - SQL generated behind the scenes via Ibis.

        Single-table datasets only support patient_level granularity.

        Args:
            granularity: Grain level (must be "patient_level" for single-table datasets)
            **filters: Optional filters
        """
        # Validate: single-table datasets only support patient_level
        if granularity != "patient_level":
            raise ValueError(
                f"{self.__class__.__name__} only supports patient_level granularity. Requested: {granularity}"
            )

        # Delegate to base class implementation
        return super().get_cohort(granularity=granularity, **filters)

import logging

import ibis
import pandas as pd
import polars as pl

from clinical_analytics.core.dataset import ClinicalDataset, Granularity
from clinical_analytics.core.mapper import ColumnMapper, load_dataset_config
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.datasets.sepsis.loader import find_psv_files, load_and_aggregate

logger = logging.getLogger(__name__)


class SepsisDataset(ClinicalDataset):
    """
    Implementation for PhysioNet Challenge 2019 Sepsis Dataset.
    Source: https://physionet.org/content/challenge-2019/

    Hybrid approach: Uses Polars to aggregate PSV files (time-series -> patient-level),
    then registers with DuckDB and uses semantic layer for querying.
    """

    def __init__(self, source_path: str | None = None):
        # Load config
        self.config = load_dataset_config("sepsis")

        # Use config default if not provided
        if source_path is None:
            source_path = self.config["init_params"]["source_path"]

        super().__init__(name="sepsis", source_path=source_path)
        self._aggregated_data: pl.DataFrame | None = None
        self.semantic: SemanticLayer | None = None

    def validate(self) -> bool:
        """Validate that PSV files exist."""
        if not self.source_path or not self.source_path.exists():
            return False
        # Check if any .psv files exist
        try:
            next(find_psv_files(self.source_path))
            return True
        except StopIteration:
            return False

    def load(self) -> None:
        """
        Load and aggregate PSV files using Polars, then register with DuckDB.

        Since Sepsis requires time-series aggregation, we use Polars for that,
        then register the aggregated result with DuckDB for semantic layer querying.
        """
        if not self.validate():
            logger.warning(f"No PSV files found in {self.source_path}. Sepsis dataset will be empty.")
            self._aggregated_data = pl.DataFrame()
            return

        # Use existing Polars aggregation logic (it works well for this)
        mapper = ColumnMapper(self.config)
        self._aggregated_data = load_and_aggregate(self.source_path, mapper=mapper)

        # Register aggregated data with DuckDB for semantic layer
        if len(self._aggregated_data) > 0:
            con = ibis.duckdb.connect()
            # Convert Polars to Pandas (Ibis works with pandas)
            df_pandas = self._aggregated_data.to_pandas()

            # Register pandas DataFrame with DuckDB using the underlying connection
            # Ibis DuckDB connection wraps a DuckDB connection
            import duckdb

            # Get the underlying DuckDB connection from Ibis
            # Ibis stores it in _con or we can access via execute
            duckdb_con = con.con if hasattr(con, "con") else duckdb.connect()
            duckdb_con.register("sepsis_aggregated", df_pandas)

            # Create semantic layer with custom config that points to the registered table
            semantic_config = self.config.copy()
            semantic_config["init_params"] = {"db_table": "sepsis_aggregated"}

            self.semantic = SemanticLayer("sepsis", config=semantic_config)
            # Override the connection to use the one we just created
            self.semantic.con = con
            self.semantic.raw = con.table("sepsis_aggregated")
            self.semantic._base_view = None  # Force rebuild with new raw table

    def get_cohort(self, granularity: Granularity = "patient_level", **filters) -> pd.DataFrame:
        """
        Return analysis cohort - uses semantic layer for SQL generation.

        Aggregation is done once in load(), then semantic layer handles
        all filtering and transformation via SQL.

        Args:
            granularity: Grain level (must be "patient_level")
            **filters: Optional filters
        """
        # Validate: Sepsis dataset is patient-level only
        if granularity != "patient_level":
            raise ValueError(f"SepsisDataset only supports patient_level granularity. Requested: {granularity}")

        # Ensure data is loaded and aggregated
        if self._aggregated_data is None:
            self.load()

        if len(self._aggregated_data) == 0:
            return pd.DataFrame(columns=UnifiedCohort.REQUIRED_COLUMNS)

        # Delegate to base class implementation
        return super().get_cohort(granularity=granularity, **filters)

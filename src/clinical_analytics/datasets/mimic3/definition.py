"""
MIMIC-III Dataset Definition - Config-driven with Semantic Layer.

Follows the standard pattern documented in IBIS_SEMANTIC_LAYER.md.
"""

from pathlib import Path
import logging
import pandas as pd
from typing import Optional

from clinical_analytics.core.dataset import ClinicalDataset, Granularity
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.core.mapper import load_dataset_config
from clinical_analytics.core.semantic import SemanticLayer

logger = logging.getLogger(__name__)


class Mimic3Dataset(ClinicalDataset):
    """
    Implementation for MIMIC-III Clinical Database.

    Follows standard semantic layer pattern with special handling for DB sources.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.config = load_dataset_config('mimic3')

        # Use config default if not provided
        if db_path is None:
            db_path = self.config['init_params'].get('db_path')

        # Note: source_path will be None for DB-based datasets
        super().__init__(name="mimic3", source_path=Path(db_path) if db_path else None)

        # Initialize semantic layer (standard pattern)
        self.semantic = SemanticLayer('mimic3', config=self.config)

    def validate(self) -> bool:
        """
        Validate that database is accessible.

        Returns:
            True if database exists and has data, False otherwise
        """
        if not self.source_path or not self.source_path.exists():
            return False

        try:
            # SemanticLayer already registered the source
            # Try to query count to verify it works
            if hasattr(self.semantic, 'raw') and self.semantic.raw is not None:
                count = self.semantic.raw.count().execute()
                return count > 0
            return False
        except Exception as e:
            logger.warning(f"MIMIC-III validation failed: {e}")
            return False

    def load(self) -> None:
        """
        Load is a no-op - data is queried on-demand via semantic layer.

        The semantic layer registers the source with DuckDB but doesn't load into memory.
        """
        if not self.validate():
            logger.warning("MIMIC-III database not available. Dataset will be empty.")
        else:
            logger.info(f"Semantic layer initialized for {self.name}")

    def get_cohort(
        self,
        granularity: Granularity = "patient_level",
        **filters
    ) -> pd.DataFrame:
        """
        Return analysis cohort using semantic layer (standard pattern).

        All logic comes from datasets.yaml config.
        
        Args:
            granularity: Grain level (patient_level, admission_level, event_level)
                        MIMIC-III is patient-level only (for now)
            **filters: Optional filters
        """
        # Validate: MIMIC-III is patient-level only (for now)
        if granularity != "patient_level":
            raise ValueError(
                f"Mimic3Dataset only supports patient_level granularity. "
                f"Requested: {granularity}"
            )
        
        if not self.validate():
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=UnifiedCohort.REQUIRED_COLUMNS)

        # Delegate to semantic layer (standard pattern)
        outcome_col = filters.get("target_outcome")
        filter_only = {k: v for k, v in filters.items() if k != "target_outcome"}

        return self.semantic.get_cohort(
            outcome_col=outcome_col,
            filters=filter_only
        )

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias

import pandas as pd

if TYPE_CHECKING:
    from clinical_analytics.core.semantic import SemanticLayer

Granularity: TypeAlias = Literal["patient_level", "admission_level", "event_level"]
Grain: TypeAlias = Literal["patient", "admission", "event"]


class ClinicalDataset(ABC):
    """
    Abstract Base Class for all clinical datasets.

    Designed to be extensible for both file-based (CSV, PSV) and
    SQL-based (DuckDB, Postgres) data sources.
    """

    GRANULARITY_TO_GRAIN: ClassVar[dict[str, Grain]] = {
        "patient_level": "patient",
        "admission_level": "admission",
        "event_level": "event",
    }
    VALID_GRANULARITIES: ClassVar[frozenset[str]] = frozenset(GRANULARITY_TO_GRAIN.keys())

    def __init__(self, name: str, source_path: str | Path | None = None, db_connection: Any = None):
        """
        Initialize the clinical dataset.

        Args:
            name: Unique identifier for the dataset (e.g., 'covid_ms', 'mimic3')
            source_path: Path to raw files (for file-based datasets)
            db_connection: Database connection object (for SQL-based datasets)
        """
        self.name = name
        self.source_path = Path(source_path) if source_path else None
        self.db_connection = db_connection
        # Instance attribute for semantic layer - NOT shared across instances
        self._semantic: SemanticLayer | None = None

    @property
    def semantic(self) -> SemanticLayer:
        """
        Lazy-initialized semantic layer for this dataset.

        Returns:
            SemanticLayer instance

        Raises:
            ValueError: If semantic layer not initialized (call load() first)
        """
        if self._semantic is None:
            raise ValueError(
                f"Semantic layer not initialized for dataset '{self.name}'. "
                "Call load() first to initialize the semantic layer."
            )
        return self._semantic

    @semantic.setter
    def semantic(self, value: SemanticLayer | None) -> None:
        """Set the semantic layer (used during load())."""
        self._semantic = value

    @abstractmethod
    def validate(self) -> bool:
        """
        Check if source data exists and meets minimum requirements.

        Returns:
            bool: True if data is valid/accessible, False otherwise.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Ingest data into an internal staging format.

        For file-based sources: Reads files into memory or a local DuckDB.
        For SQL sources: Establishes/verifies connection and checks schema.
        """
        pass

    def get_cohort(
        self,
        granularity: Granularity = "patient_level",
        **filters: Any,
    ) -> pd.DataFrame:
        """
        Return a standardized analysis dataframe conformant to UnifiedCohort schema.

        Default implementation using semantic layer. Most datasets can use this
        implementation. Override only if custom logic needed (e.g., granularity
        validation, special preprocessing).

        Args:
            granularity:
                - "patient_level": One row per patient (default)
                - "admission_level": One row per admission/encounter
                - "event_level": One row per event (e.g., lab result, medication)
            **filters: Dataset-specific filters (e.g., age_min=18, specific_diagnosis=True)

        Returns:
            pd.DataFrame: A DataFrame containing at least the required UnifiedCohort columns.
        """
        # Extract outcome override if provided
        outcome_col = filters.pop("target_outcome", None)

        # Delegate to semantic layer - it generates SQL and executes
        if self._semantic is None:
            raise ValueError(
                f"Dataset '{self.name}' does not have semantic layer initialized. "
                "Call load() first, or override get_cohort() for custom behavior."
            )

        return self.semantic.get_cohort(
            granularity=granularity,
            outcome_col=outcome_col,
            filters=filters,
            show_sql=False,
        )

    @classmethod
    def _map_granularity_to_grain(cls, granularity: Granularity) -> Grain:
        """
        Map API granularity values to internal grain values.

        Args:
            granularity: API granularity value (patient_level, admission_level, event_level)

        Returns:
            Internal grain value for multi-table handler (patient, admission, event)

        Raises:
            ValueError: If granularity is not one of the valid values
        """
        # Defensive runtime validation. Literal does not enforce this at runtime.
        # Use GRANULARITY_TO_GRAIN as single source of truth.
        if granularity not in cls.GRANULARITY_TO_GRAIN:
            raise ValueError(
                f"Invalid granularity: {granularity!r}. Must be one of: {sorted(cls.GRANULARITY_TO_GRAIN.keys())}"
            )

        return cls.GRANULARITY_TO_GRAIN[granularity]

    def get_semantic_layer(self) -> SemanticLayer:
        """
        Get semantic layer for this dataset.

        Returns:
            SemanticLayer instance

        Raises:
            ValueError: If semantic layer is not available for this dataset
        """
        if self._semantic is None:
            raise ValueError(
                f"Dataset '{self.name}' does not support semantic layer features "
                "(natural language queries, SQL preview, query builder). "
                "This is typically only available for registry datasets with config-driven semantic layers."
            )
        return self._semantic

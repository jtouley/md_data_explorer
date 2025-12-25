from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

class ClinicalDataset(ABC):
    """
    Abstract Base Class for all clinical datasets.
    
    Designed to be extensible for both file-based (CSV, PSV) and 
    SQL-based (DuckDB, Postgres) data sources.
    """

    def __init__(self, name: str, source_path: Union[str, Path, None] = None, db_connection: Any = None):
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

    @abstractmethod
    def get_cohort(self, **filters) -> pd.DataFrame:
        """
        Return a standardized analysis dataframe conformant to UnifiedCohort schema.
        
        Args:
            **filters: Dataset-specific filters (e.g., age_min=18, specific_diagnosis=True)
            
        Returns:
            pd.DataFrame: A DataFrame containing at least the required UnifiedCohort columns.
        """
        pass


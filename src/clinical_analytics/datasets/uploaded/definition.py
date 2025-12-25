"""
User-Uploaded Dataset

Dynamic dataset implementation for user-uploaded files.
Integrates uploaded data with the existing registry system.
"""

from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any

from clinical_analytics.core.dataset import ClinicalDataset
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage


class UploadedDataset(ClinicalDataset):
    """
    Dynamic dataset implementation for user uploads.

    This class allows uploaded datasets to work seamlessly with
    the existing registry system and analysis infrastructure.
    """

    def __init__(self, upload_id: str, storage: Optional[UserDatasetStorage] = None):
        """
        Initialize uploaded dataset.

        Args:
            upload_id: Unique upload identifier
            storage: Storage manager (optional, will create if not provided)
        """
        if storage is None:
            storage = UserDatasetStorage()

        self.storage = storage
        self.upload_id = upload_id
        self.metadata = None
        self.data = None

        # Load metadata
        self.metadata = storage.get_upload_metadata(upload_id)
        if not self.metadata:
            raise ValueError(f"Upload {upload_id} not found")

        # Initialize with upload info
        dataset_name = self.metadata.get('dataset_name', upload_id)
        super().__init__(name=dataset_name, source_path=None)

    def validate(self) -> bool:
        """
        Check if upload data exists and is valid.

        Returns:
            True if data is accessible
        """
        csv_path = self.storage.raw_dir / f"{self.upload_id}.csv"
        return csv_path.exists()

    def load(self) -> None:
        """
        Load uploaded data into memory.
        """
        if not self.validate():
            raise FileNotFoundError(f"Upload data not found: {self.upload_id}")

        self.data = self.storage.get_upload_data(self.upload_id)

        if self.data is None:
            raise ValueError(f"Failed to load upload data: {self.upload_id}")

    def get_cohort(self, **filters) -> pd.DataFrame:
        """
        Return analysis cohort mapped to UnifiedCohort schema.

        Maps user columns to UnifiedCohort schema based on
        the variable mapping from upload wizard.

        Args:
            **filters: Optional filters (not yet implemented)

        Returns:
            DataFrame conforming to UnifiedCohort schema
        """
        if self.data is None:
            self.load()

        # Get variable mapping from metadata
        variable_mapping = self.metadata.get('variable_mapping', {})

        if not variable_mapping:
            raise ValueError("Variable mapping not found in upload metadata")

        # Extract mapping fields
        patient_id_col = variable_mapping.get('patient_id')
        outcome_col = variable_mapping.get('outcome')
        predictors = variable_mapping.get('predictors', [])
        time_vars = variable_mapping.get('time_variables', {})

        # Build cohort dataframe
        cohort_data = {}

        # Map patient ID
        if patient_id_col:
            cohort_data[UnifiedCohort.PATIENT_ID] = self.data[patient_id_col]
        else:
            # Generate sequential IDs if not provided
            cohort_data[UnifiedCohort.PATIENT_ID] = [f"patient_{i}" for i in range(len(self.data))]

        # Map outcome
        if outcome_col:
            cohort_data[UnifiedCohort.OUTCOME] = self.data[outcome_col]
        else:
            raise ValueError("Outcome column not specified in mapping")

        # Map time zero (use upload date if not provided)
        if time_vars and time_vars.get('time_zero'):
            time_col = time_vars['time_zero']
            if time_col in self.data.columns:
                cohort_data[UnifiedCohort.TIME_ZERO] = pd.to_datetime(self.data[time_col])
            else:
                cohort_data[UnifiedCohort.TIME_ZERO] = pd.Timestamp(self.metadata['upload_timestamp'])
        else:
            # Use upload timestamp as time zero
            cohort_data[UnifiedCohort.TIME_ZERO] = pd.Timestamp(self.metadata['upload_timestamp'])

        # Add predictor variables (keep original names)
        for pred in predictors:
            if pred in self.data.columns:
                cohort_data[pred] = self.data[pred]

        # Create cohort dataframe
        cohort = pd.DataFrame(cohort_data)

        # Apply any filters
        if filters:
            # Basic filter support (can be extended)
            for key, value in filters.items():
                if key in cohort.columns:
                    cohort = cohort[cohort[key] == value]

        return cohort

    def get_info(self) -> Dict[str, Any]:
        """
        Get dataset information.

        Returns:
            Dictionary with dataset metadata
        """
        return {
            'upload_id': self.upload_id,
            'name': self.name,
            'uploaded_at': self.metadata.get('upload_timestamp'),
            'original_filename': self.metadata.get('original_filename'),
            'row_count': self.metadata.get('row_count'),
            'column_count': self.metadata.get('column_count'),
            'columns': self.metadata.get('columns'),
            'variable_mapping': self.metadata.get('variable_mapping')
        }


class UploadedDatasetFactory:
    """
    Factory for creating UploadedDataset instances.

    Provides methods to list and create uploaded datasets
    for integration with the registry system.
    """

    @staticmethod
    def list_available_uploads() -> list[Dict[str, Any]]:
        """
        List all available uploaded datasets.

        Returns:
            List of upload metadata dictionaries
        """
        storage = UserDatasetStorage()
        return storage.list_uploads()

    @staticmethod
    def create_dataset(upload_id: str) -> UploadedDataset:
        """
        Create dataset instance for an upload.

        Args:
            upload_id: Upload identifier

        Returns:
            UploadedDataset instance
        """
        return UploadedDataset(upload_id=upload_id)

    @staticmethod
    def register_all_uploads() -> Dict[str, UploadedDataset]:
        """
        Create dataset instances for all uploads.

        Returns:
            Dictionary mapping upload_id to dataset instances
        """
        uploads = UploadedDatasetFactory.list_available_uploads()
        datasets = {}

        for upload in uploads:
            upload_id = upload['upload_id']
            try:
                datasets[upload_id] = UploadedDataset(upload_id=upload_id)
            except Exception as e:
                print(f"Warning: Failed to load upload {upload_id}: {e}")

        return datasets

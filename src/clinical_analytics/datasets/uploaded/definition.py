"""
User-Uploaded Dataset

Dynamic dataset implementation for user-uploaded files.
Integrates uploaded data with the existing registry system.
"""

from pathlib import Path
import logging
import pandas as pd
from typing import Optional, Dict, Any

from clinical_analytics.core.dataset import ClinicalDataset, Granularity
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

logger = logging.getLogger(__name__)


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
        self._semantic_initialized = False  # Track lazy init

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

    def get_cohort(
        self,
        granularity: Granularity = "patient_level",
        **filters
    ) -> pd.DataFrame:
        """
        Return analysis cohort mapped to UnifiedCohort schema.

        Maps user columns to UnifiedCohort schema based on either:
        - variable_mapping (from single-table upload wizard)
        - inferred_schema (from multi-table ZIP upload)

        Args:
            granularity: Grain level (patient_level, admission_level, event_level)
                        Single-table uploads only support patient_level
            **filters: Optional filters (not yet implemented)

        Returns:
            DataFrame conforming to UnifiedCohort schema (outcome column optional)
        """
        # Validate: single-table uploads only support patient_level
        if granularity != "patient_level":
            raise ValueError(
                f"UploadedDataset (single-table) only supports patient_level granularity. "
                f"Requested: {granularity}. Multi-table ZIP uploads support all granularities."
            )
        
        if self.data is None:
            self.load()

        # Get variable mapping from metadata (single-table uploads)
        variable_mapping = self.metadata.get('variable_mapping', {})

        # If no variable mapping, try to build it from inferred schema (ZIP uploads)
        if not variable_mapping:
            inferred_schema = self.metadata.get('inferred_schema', {})

            if inferred_schema:
                # Convert inferred_schema to variable_mapping format
                variable_mapping = self._convert_inferred_schema_to_mapping(inferred_schema)
            else:
                raise ValueError("Neither variable_mapping nor inferred_schema found in upload metadata")

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

        # Map outcome (optional - semantic layer pattern)
        # Some analyses (Descriptive Stats, Correlations) don't require outcomes
        if outcome_col:
            cohort_data[UnifiedCohort.OUTCOME] = self.data[outcome_col]
            # Add outcome_label if available
            outcome_label = variable_mapping.get('outcome_label', 'outcome')
            cohort_data[UnifiedCohort.OUTCOME_LABEL] = outcome_label
        # If no outcome specified, skip it - semantic layer handles this gracefully
        # Downstream code must check for OUTCOME/OUTCOME_LABEL existence before using

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

    def _convert_inferred_schema_to_mapping(self, inferred_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert inferred_schema format (from ZIP uploads) to variable_mapping format.

        Args:
            inferred_schema: Schema from schema inference engine

        Returns:
            variable_mapping dictionary compatible with get_cohort()
        """
        variable_mapping = {
            'patient_id': None,
            'outcome': None,
            'time_variables': {},
            'predictors': []
        }

        # Extract patient ID from column_mapping
        column_mapping = inferred_schema.get('column_mapping', {})
        for col, role in column_mapping.items():
            if role == 'patient_id':
                variable_mapping['patient_id'] = col
                break

        # Extract first outcome
        outcomes = inferred_schema.get('outcomes', {})
        if outcomes:
            # Use first outcome as primary
            first_outcome = list(outcomes.keys())[0]
            variable_mapping['outcome'] = first_outcome

        # Extract time_zero
        time_zero_config = inferred_schema.get('time_zero', {})
        if 'source_column' in time_zero_config:
            variable_mapping['time_variables']['time_zero'] = time_zero_config['source_column']

        # Add all other columns as predictors (exclude patient_id and outcome)
        if self.data is not None:
            all_cols = set(self.data.columns)
            excluded = {variable_mapping['patient_id'], variable_mapping['outcome']}
            variable_mapping['predictors'] = [
                col for col in all_cols
                if col not in excluded and col not in {None}
            ]

        return variable_mapping
    
    def get_semantic_layer(self):
        """
        Get semantic layer, lazy-initializing from unified cohort if available.
        
        Overrides base class to support multi-table uploads.
        """
        # Lazy initialize if not already done
        if not self._semantic_initialized:
            self._maybe_init_semantic()
            self._semantic_initialized = True
        
        # Call parent to get semantic (or raise if None)
        return super().get_semantic_layer()
    
    def _maybe_init_semantic(self) -> None:
        """Lazy initialization of semantic layer for multi-table uploads."""
        # Only for multi-table uploads with inferred_schema
        inferred_schema = self.metadata.get('inferred_schema')
        if not inferred_schema:
            return
        
        csv_path = self.storage.raw_dir / f"{self.upload_id}.csv"
        if not csv_path.exists():
            logger.warning(f"Unified cohort CSV not found for upload {self.upload_id}")
            return
        
        try:
            from clinical_analytics.core.semantic import SemanticLayer, _safe_identifier
            
            config = self._build_config_from_inferred_schema(inferred_schema)
            workspace_root = self.storage.upload_dir.parent.parent
            
            # Use absolute path - SemanticLayer handles absolute paths correctly
            config['init_params'] = {'source_path': str(csv_path.resolve())}
            
            self.semantic = SemanticLayer(
                dataset_name=self.name,
                config=config,
                workspace_root=workspace_root
            )
            
            # Register all individual tables from the upload
            tables_dir = self.storage.raw_dir / f"{self.upload_id}_tables"
            if tables_dir.exists():
                table_names = self.metadata.get('tables', [])
                logger.info(f"Registering {len(table_names)} individual tables from {tables_dir}")
                
                duckdb_con = self.semantic.con.con
                safe_dataset_name = _safe_identifier(self.name)
                
                for table_name in table_names:
                    table_path = tables_dir / f"{table_name}.csv"
                    if table_path.exists():
                        safe_table_name = _safe_identifier(f"{safe_dataset_name}_{table_name}")
                        abs_path = str(table_path.resolve())
                        
                        duckdb_con.execute(
                            f"CREATE OR REPLACE TABLE {safe_table_name} AS SELECT * FROM read_csv_auto(?)",
                            [abs_path]
                        )
                        logger.info(f"Registered table '{safe_table_name}' from {abs_path}")
                    else:
                        logger.warning(f"Table file not found: {table_path}")
                
                logger.info(f"Created semantic layer for uploaded dataset '{self.name}' with {len(table_names)} tables")
            else:
                logger.info(f"Created semantic layer for uploaded dataset '{self.name}'")
        except Exception as e:
            logger.warning(f"Failed to create semantic layer: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Leave self.semantic as None (class attribute default)
    
    def _build_config_from_inferred_schema(self, inferred_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Build semantic layer config from inferred schema."""
        outcomes = inferred_schema.get('outcomes', {})
        default_outcome = list(outcomes.keys())[0] if outcomes else None
        
        config = {
            'name': self.name,
            'display_name': self.metadata.get('original_filename', self.name),
            'status': 'available',
            'init_params': {},  # Will be set to absolute CSV path
            'column_mapping': inferred_schema.get('column_mapping', {}),
            'outcomes': outcomes,
            'time_zero': inferred_schema.get('time_zero', {}),
            'analysis': {
                'default_outcome': default_outcome,
                'default_predictors': inferred_schema.get('predictors', []),
                'categorical_variables': inferred_schema.get('categorical_columns', [])
            }
        }
        return config

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
                logger.warning(f"Failed to load upload {upload_id}: {e}")

        return datasets

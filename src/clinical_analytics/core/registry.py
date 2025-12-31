"""
Dataset Registry - Auto-discovery and factory for clinical datasets.

This module provides a registry pattern for discovering and instantiating
ClinicalDataset implementations without hardcoded if/else chains.
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from clinical_analytics.core.dataset import ClinicalDataset
from clinical_analytics.core.schema_inference import SchemaInferenceEngine

logger = logging.getLogger(__name__)


def _filter_kwargs_for_ctor(cls, kwargs: dict) -> dict:
    """
    Filter kwargs to only include parameters accepted by the class constructor.

    Prevents "unexpected keyword argument" errors when configs contain params
    that a dataset class doesn't accept (e.g., db_connection for Mimic3Dataset).

    Args:
        cls: Dataset class to instantiate
        kwargs: Dictionary of parameters to filter

    Returns:
        Filtered dictionary with only accepted parameters
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # If constructor accepts **kwargs, pass everything through
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        return kwargs

    # Filter to only accepted parameters
    allowed = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(allowed))

    if dropped:
        logger.info(
            f"Dropping unsupported init params for {cls.__name__}: {dropped}. "
            f"These parameters were ignored. Check dataset constructor signature."
        )

    return allowed


class DatasetRegistry:
    """
    Registry for auto-discovering and managing ClinicalDataset implementations.

    This eliminates hardcoded dataset lists and enables true extensibility.
    """

    _datasets: dict[str, type[ClinicalDataset]] = {}
    _configs: dict[str, dict] = {}
    _config_loaded: bool = False
    _auto_inferred: dict[str, pl.DataFrame] = {}  # Store DataFrames for auto-inferred datasets

    @classmethod
    def discover_datasets(cls) -> dict[str, type[ClinicalDataset]]:
        """
        Auto-discover all ClinicalDataset implementations in the datasets package.

        Excludes built-in datasets (covid_ms, mimic3, sepsis) - only user uploads are supported.

        Returns:
            Dict mapping dataset names to their classes
        """
        import clinical_analytics.datasets as datasets_pkg

        datasets_path = Path(datasets_pkg.__file__).parent

        # Built-in datasets to exclude (only user uploads are supported)
        builtin_datasets = {"covid_ms", "mimic3", "sepsis"}

        # Iterate through all subdirectories in datasets/
        for module_info in pkgutil.iter_modules([str(datasets_path)]):
            if module_info.ispkg:
                module_name = module_info.name

                # Skip built-in datasets
                if module_name in builtin_datasets:
                    continue

                try:
                    # Import the definition module
                    definition_module = importlib.import_module(f"clinical_analytics.datasets.{module_name}.definition")

                    # Find all ClinicalDataset subclasses in this module
                    for name, obj in inspect.getmembers(definition_module, inspect.isclass):
                        if (
                            issubclass(obj, ClinicalDataset)
                            and obj is not ClinicalDataset
                            and obj.__module__ == definition_module.__name__
                        ):
                            # Register using module name as key
                            cls._datasets[module_name] = obj

                except (ImportError, AttributeError) as e:
                    # Skip modules that don't have definition.py or proper structure
                    print(f"Skipping {module_name}: {e}")
                    continue

        return cls._datasets

    @classmethod
    def load_config(cls, config_path: Path | None = None) -> None:
        """
        Load dataset configurations from YAML file.

        Args:
            config_path: Path to datasets.yaml config file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "data" / "configs" / "datasets.yaml"

        if not config_path.exists():
            print(f"Warning: Config file not found at {config_path}")
            cls._configs = {}
            cls._config_loaded = True
            return

        with open(config_path) as f:
            cls._configs = yaml.safe_load(f) or {}

        cls._config_loaded = True

    @classmethod
    def get_dataset(cls, name: str, **override_params) -> ClinicalDataset:
        """
        Factory method to instantiate a dataset by name.

        Args:
            name: Dataset identifier (e.g., 'covid_ms', 'sepsis')
            **override_params: Parameters to override config values

        Returns:
            Instantiated ClinicalDataset

        Raises:
            KeyError: If dataset name not found
        """
        # Ensure datasets are discovered
        if not cls._datasets:
            cls.discover_datasets()

        # Ensure config is loaded
        if not cls._config_loaded:
            cls.load_config()

        if name not in cls._datasets:
            available = ", ".join(cls._datasets.keys())
            raise KeyError(f"Dataset '{name}' not found in registry. Available datasets: {available}")

        dataset_class = cls._datasets[name]

        # Get config for this dataset if available
        config = cls._configs.get(name, {})

        # Special handling for UploadedDataset: requires upload_id, not from config
        if dataset_class.__name__ == "UploadedDataset":
            # UploadedDataset requires upload_id as positional argument
            # The name parameter IS the upload_id for uploaded datasets
            if "upload_id" in override_params:
                upload_id = override_params["upload_id"]
            elif "upload_id" in config.get("init_params", {}):
                upload_id = config["init_params"]["upload_id"]
            else:
                # Use the name as upload_id (registry key for uploaded datasets is upload_id)
                upload_id = name

            storage = override_params.get("storage") or config.get("init_params", {}).get("storage")
            return dataset_class(upload_id=upload_id, storage=storage)

        # Merge config with override params
        params = {**config.get("init_params", {}), **override_params}

        # Filter params by constructor signature
        params = _filter_kwargs_for_ctor(dataset_class, params)

        # Instantiate and return
        return dataset_class(**params)

    @classmethod
    def list_datasets(cls) -> list[str]:
        """
        Get list of available dataset names.

        Returns:
            List of dataset identifiers
        """
        if not cls._datasets:
            cls.discover_datasets()

        return list(cls._datasets.keys())

    @classmethod
    def get_dataset_info(cls, name: str) -> dict:
        """
        Get configuration and metadata for a dataset.

        Args:
            name: Dataset identifier

        Returns:
            Dictionary with dataset info
        """
        if not cls._config_loaded:
            cls.load_config()

        if not cls._datasets:
            cls.discover_datasets()

        info = {
            "name": name,
            "available": name in cls._datasets,
            "config": cls._configs.get(name, {}),
        }

        if name in cls._datasets:
            dataset_class = cls._datasets[name]
            info["class"] = dataset_class.__name__
            info["module"] = dataset_class.__module__
            info["doc"] = dataset_class.__doc__

        return info

    @classmethod
    def get_all_dataset_info(cls) -> dict[str, dict]:
        """
        Get info for all available datasets.

        Returns:
            Dictionary mapping dataset names to their info
        """
        if not cls._datasets:
            cls.discover_datasets()

        return {name: cls.get_dataset_info(name) for name in cls._datasets.keys()}

    @classmethod
    def register_from_dataframe(
        cls,
        dataset_name: str,
        df: pl.DataFrame,
        display_name: str | None = None,
        infer_schema: bool = True,
    ) -> dict[str, Any]:
        """
        Register dataset from Polars DataFrame with automatic schema inference.

        This eliminates the need for manual YAML configuration files.
        Schema is automatically detected using SchemaInferenceEngine.

        Args:
            dataset_name: Unique identifier for this dataset
            df: Polars DataFrame with raw data
            display_name: Human-readable name (defaults to dataset_name)
            infer_schema: Whether to auto-infer schema (default: True)

        Returns:
            Generated config dictionary

        Example:
            >>> df = pl.read_csv("patient_data.csv")
            >>> config = DatasetRegistry.register_from_dataframe(
            ...     "my_study",
            ...     df,
            ...     display_name="My Clinical Study"
            ... )
            >>> print(f"Detected patient ID: {config['column_mapping']}")

        Note:
            For uploaded datasets, this is called automatically by UserDatasetStorage.
            For built-in datasets, YAML configs can be replaced with this method.
        """
        # Infer schema from DataFrame
        if infer_schema:
            engine = SchemaInferenceEngine()
            schema = engine.infer_schema(df)
            config = schema.to_dataset_config()
        else:
            config = {"column_mapping": {}, "outcomes": {}, "time_zero": {}}

        # Add metadata
        config["name"] = dataset_name
        config["display_name"] = display_name or dataset_name
        config["status"] = "auto-inferred" if infer_schema else "manual"
        config["row_count"] = df.height
        config["column_count"] = df.width

        # Store config
        cls._configs[dataset_name] = config

        # Store DataFrame for later retrieval
        cls._auto_inferred[dataset_name] = df

        # Mark config as loaded
        cls._config_loaded = True

        return config

    @classmethod
    def get_auto_inferred_dataframe(cls, dataset_name: str) -> pl.DataFrame | None:
        """
        Retrieve Polars DataFrame for an auto-inferred dataset.

        Args:
            dataset_name: Dataset identifier

        Returns:
            Polars DataFrame or None if not found
        """
        return cls._auto_inferred.get(dataset_name)

    @classmethod
    def reset(cls) -> None:
        """Reset registry (mainly for testing)."""
        cls._datasets = {}
        cls._configs = {}
        cls._config_loaded = False
        cls._auto_inferred = {}

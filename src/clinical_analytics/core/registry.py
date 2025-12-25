"""
Dataset Registry - Auto-discovery and factory for clinical datasets.

This module provides a registry pattern for discovering and instantiating
ClinicalDataset implementations without hardcoded if/else chains.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, Type, Optional, List
import yaml

from clinical_analytics.core.dataset import ClinicalDataset


class DatasetRegistry:
    """
    Registry for auto-discovering and managing ClinicalDataset implementations.

    This eliminates hardcoded dataset lists and enables true extensibility.
    """

    _datasets: Dict[str, Type[ClinicalDataset]] = {}
    _configs: Dict[str, dict] = {}
    _config_loaded: bool = False

    @classmethod
    def discover_datasets(cls) -> Dict[str, Type[ClinicalDataset]]:
        """
        Auto-discover all ClinicalDataset implementations in the datasets package.

        Returns:
            Dict mapping dataset names to their classes
        """
        import clinical_analytics.datasets as datasets_pkg

        datasets_path = Path(datasets_pkg.__file__).parent

        # Iterate through all subdirectories in datasets/
        for module_info in pkgutil.iter_modules([str(datasets_path)]):
            if module_info.ispkg:
                module_name = module_info.name

                try:
                    # Import the definition module
                    definition_module = importlib.import_module(
                        f"clinical_analytics.datasets.{module_name}.definition"
                    )

                    # Find all ClinicalDataset subclasses in this module
                    for name, obj in inspect.getmembers(definition_module, inspect.isclass):
                        if (issubclass(obj, ClinicalDataset) and
                            obj is not ClinicalDataset and
                            obj.__module__ == definition_module.__name__):

                            # Register using module name as key
                            cls._datasets[module_name] = obj

                except (ImportError, AttributeError) as e:
                    # Skip modules that don't have definition.py or proper structure
                    print(f"Skipping {module_name}: {e}")
                    continue

        return cls._datasets

    @classmethod
    def load_config(cls, config_path: Optional[Path] = None) -> None:
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

        with open(config_path, 'r') as f:
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
            available = ', '.join(cls._datasets.keys())
            raise KeyError(
                f"Dataset '{name}' not found in registry. "
                f"Available datasets: {available}"
            )

        dataset_class = cls._datasets[name]

        # Get config for this dataset if available
        config = cls._configs.get(name, {})

        # Merge config with override params
        params = {**config.get('init_params', {}), **override_params}

        # Instantiate and return
        return dataset_class(**params)

    @classmethod
    def list_datasets(cls) -> List[str]:
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
            'name': name,
            'available': name in cls._datasets,
            'config': cls._configs.get(name, {}),
        }

        if name in cls._datasets:
            dataset_class = cls._datasets[name]
            info['class'] = dataset_class.__name__
            info['module'] = dataset_class.__module__
            info['doc'] = dataset_class.__doc__

        return info

    @classmethod
    def get_all_dataset_info(cls) -> Dict[str, dict]:
        """
        Get info for all available datasets.

        Returns:
            Dictionary mapping dataset names to their info
        """
        if not cls._datasets:
            cls.discover_datasets()

        return {name: cls.get_dataset_info(name) for name in cls._datasets.keys()}

    @classmethod
    def reset(cls) -> None:
        """Reset registry (mainly for testing)."""
        cls._datasets = {}
        cls._configs = {}
        cls._config_loaded = False

"""
Tests for ClinicalDataset instance isolation.

Validates:
- Semantic layer is an instance attribute (not class attribute)
- Each dataset instance has independent semantic layer
- Property accessor works correctly
"""

import pytest
from clinical_analytics.core.dataset import ClinicalDataset


class _ConcreteDataset(ClinicalDataset):
    """Concrete implementation for testing abstract base class."""

    def validate(self) -> bool:
        return True

    def load(self) -> None:
        # Simulate loading by setting semantic layer
        pass

    def get_cohort(self, granularity: str = "patient_level", **filters):
        import pandas as pd

        return pd.DataFrame({"patient_id": ["P001"]})


class TestDatasetInstanceIsolation:
    """Tests for semantic layer instance isolation."""

    def test_semantic_is_instance_attribute(self) -> None:
        """Test that semantic is per-instance, not shared."""
        ds1 = _ConcreteDataset("dataset1")
        ds2 = _ConcreteDataset("dataset2")

        # Both should start with None
        assert ds1._semantic is None
        assert ds2._semantic is None

        # Setting on one should not affect the other
        ds1._semantic = "semantic1"  # type: ignore
        assert ds1._semantic == "semantic1"
        assert ds2._semantic is None

    def test_semantic_property_raises_before_load(self) -> None:
        """Test that accessing semantic before load() raises ValueError."""
        ds = _ConcreteDataset("test_dataset")

        with pytest.raises(ValueError, match="not initialized"):
            _ = ds.semantic

    def test_semantic_property_returns_after_set(self) -> None:
        """Test that semantic property returns value after setting."""
        ds = _ConcreteDataset("test_dataset")

        # Simulate what load() would do
        ds._semantic = "mock_semantic_layer"  # type: ignore

        assert ds.semantic == "mock_semantic_layer"

    def test_semantic_setter_works(self) -> None:
        """Test that semantic setter works correctly."""
        ds = _ConcreteDataset("test_dataset")

        ds.semantic = "new_semantic"  # type: ignore
        assert ds._semantic == "new_semantic"

        ds.semantic = None
        assert ds._semantic is None

    def test_multiple_instances_independent(self) -> None:
        """Test that multiple dataset instances have independent semantic layers."""
        datasets = [_ConcreteDataset(f"ds{i}") for i in range(5)]

        # Set different values on each
        for i, ds in enumerate(datasets):
            ds._semantic = f"semantic_{i}"  # type: ignore

        # Verify each has its own value
        for i, ds in enumerate(datasets):
            assert ds._semantic == f"semantic_{i}"

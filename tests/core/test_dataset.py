"""Unit tests for ClinicalDataset base class."""

import pytest

from clinical_analytics.core.dataset import ClinicalDataset


def test_map_granularity_to_grain_valid():
    """Test valid granularity values map correctly."""
    assert ClinicalDataset._map_granularity_to_grain("patient_level") == "patient"
    assert ClinicalDataset._map_granularity_to_grain("admission_level") == "admission"
    assert ClinicalDataset._map_granularity_to_grain("event_level") == "event"


def test_map_granularity_to_grain_invalid_raises():
    """Test invalid granularity raises ValueError with clear message."""
    with pytest.raises(ValueError, match="Invalid granularity"):
        # type: ignore[arg-type]  # runtime validation test
        ClinicalDataset._map_granularity_to_grain("invalid_level")

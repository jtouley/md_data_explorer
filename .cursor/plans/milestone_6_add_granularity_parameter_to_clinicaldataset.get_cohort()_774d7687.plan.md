---
name: "Milestone 6: Add Granularity Parameter to ClinicalDataset.get_cohort()"
overview: "Contract-only change: Add granularity parameter to base class signature with runtime validation in mapping helper. No behavior changes in concrete implementations until M7."
todos:
  - id: update_base_signature
    content: Update ClinicalDataset.get_cohort() abstract method signature to include granularity parameter with default 'patient_level' and TypeAlias for type safety
    status: pending
  - id: add_mapping_helper
    content: Add _map_granularity_to_grain() classmethod that validates input and raises ValueError (not KeyError) with clear message
    status: pending
  - id: add_valid_granularities_constant
    content: Add VALID_GRANULARITIES ClassVar constant (frozenset) as single source of truth, used by mapping helper for validation
    status: pending
  - id: update_docstring
    content: Update docstring to document granularity parameter and all three valid values
    status: pending
    dependencies:
      - update_base_signature
  - id: create_unit_tests
    content: Create unit tests ONLY for mapping helper validation behavior (valid inputs and invalid input raises ValueError)
    status: pending
    dependencies:
      - add_mapping_helper
---

# Milestone 6: Add Granularity Parameter to ClinicalDataset.get_cohort()

## Overview

**Contract-only change**: Update the abstract `get_cohort()` method in the `ClinicalDataset` base class to include a `granularity` parameter. Add a mapping helper with **runtime validation** that maps API-level granularity values (`patient_level`, `admission_level`, `event_level`) to internal grain values (`patient`, `admission`, `event`) used by the multi-table handler.**Scope**: This milestone establishes the interface contract. Concrete implementations will handle the parameter in milestone 7. No behavior changes in existing dataset classes.

## Context

The multi-table handler's `materialize_mart()` method uses internal grain values (`patient`, `admission`, `event`), but the API should expose user-friendly granularity values (`patient_level`, `admission_level`, `event_level`). This milestone establishes the interface contract that all dataset implementations will follow in milestone 7.

## Changes Required

### 1. Update Base Class Signature

**File**: [`src/clinical_analytics/core/dataset.py`](src/clinical_analytics/core/dataset.py)Update imports and abstract method signature:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Literal, TypeAlias

import pandas as pd

Granularity: TypeAlias = Literal["patient_level", "admission_level", "event_level"]
Grain: TypeAlias = Literal["patient", "admission", "event"]


class ClinicalDataset(ABC):
    """
    Abstract Base Class for all clinical datasets.
    
    Designed to be extensible for both file-based (CSV, PSV) and 
    SQL-based (DuckDB, Postgres) data sources.
    """
    
    VALID_GRANULARITIES: ClassVar[frozenset[str]] = frozenset({
        "patient_level",
        "admission_level",
        "event_level",
    })

    # ... existing __init__, validate(), load() methods ...

    @abstractmethod
    def get_cohort(
        self,
        granularity: Granularity = "patient_level",
        **filters: Any,
    ) -> pd.DataFrame:
        """
        Return a standardized analysis dataframe conformant to UnifiedCohort schema.
        
        Args:
            granularity:
        - "patient_level": One row per patient (default)
        - "admission_level": One row per admission/encounter
        - "event_level": One row per event (e.g., lab result, medication)
            **filters: Dataset-specific filters (e.g., age_min=18, specific_diagnosis=True)
            
        Returns:
            pd.DataFrame: A DataFrame containing at least the required UnifiedCohort columns.
        """
        raise NotImplementedError  # Alternative: use ... (ellipsis) for pure style
```



### 2. Add Granularity Mapping Helper with Runtime Validation

**File**: [`src/clinical_analytics/core/dataset.py`](src/clinical_analytics/core/dataset.py)Add classmethod that validates and maps:

```python
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
        # Use VALID_GRANULARITIES as single source of truth.
        if granularity not in cls.VALID_GRANULARITIES:
            raise ValueError(
                f"Invalid granularity: {granularity!r}. "
                f"Must be one of: {sorted(cls.VALID_GRANULARITIES)}"
            )

        mapping: Dict[str, Grain] = {
            "patient_level": "patient",
            "admission_level": "admission",
            "event_level": "event",
        }

        return mapping[granularity]
```



## Implementation Details

### Type Safety vs Runtime Validation

- **`Literal` types**: Enforce valid values at **static type-checking time** (mypy, pyright)
- **Runtime validation**: The mapping helper validates and raises `ValueError` (not `KeyError`) with a clear message
- **Why both**: Type checkers catch errors during development; runtime validation catches errors from dynamic input (e.g., user input, API calls)

### Backward Compatibility

- Default value `"patient_level"` ensures existing code continues to work
- All existing calls to `get_cohort(**filters)` will implicitly use `granularity="patient_level"`
- No breaking changes: abstract method signature change doesn't affect existing implementations until they're updated in M7

### Design Decisions

1. **`@classmethod` instead of `@staticmethod`**: Allows subclasses to override mapping if needed (future-proof)
2. **`VALID_GRANULARITIES` ClassVar (frozenset)**: Single source of truth for valid values; frozenset prevents accidental mutation
3. **`VALID_GRANULARITIES` used in mapping helper**: Mapping function validates against `VALID_GRANULARITIES` (not hardcoded keys) to maintain single source of truth
4. **`TypeAlias` for Granularity/Grain**: Improves readability and allows easy extension
5. **`raise NotImplementedError` in abstract method**: More explicit than `pass` for abstract methods (alternative: `...` ellipsis is also valid)

## Testing Strategy

### Unit Tests (ONLY test what we implement in M6)

**File**: [`tests/core/test_dataset.py`](tests/core/test_dataset.py) (create if needed)Test **only** the mapping helper's validation behavior:

```python
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
```

**What we do NOT test in M6**:

- ❌ Testing that abstract `get_cohort()` validates granularity (we're not implementing validation there)
- ❌ Testing default granularity via mock implementations (not implementing behavior in M6)
- ❌ Testing concrete dataset implementations (that's M7)

## Dependencies

- **Milestone 5**: Must be completed (multi-table handler with `materialize_mart()` that accepts `grain` parameter)
- **Milestone 7**: Will implement granularity support in all dataset classes (will use `_map_granularity_to_grain()` and `VALID_GRANULARITIES`)
- **Milestone 8**: Will update `SemanticLayer.get_cohort()` to handle granularity

## Success Criteria

- [x] `ClinicalDataset.get_cohort()` signature includes `granularity` parameter with default `"patient_level"`
- [x] `Granularity` and `Grain` TypeAlias defined for type safety
- [x] `VALID_GRANULARITIES` ClassVar constant (frozenset) added as single source of truth, used by mapping helper
- [x] `_map_granularity_to_grain()` classmethod validates input and raises `ValueError` (not `KeyError`)
- [x] Docstring documents all three granularity options
- [x] Unit tests verify mapping function works for valid inputs
- [x] Unit tests verify mapping function raises `ValueError` for invalid input
- [x] No breaking changes to existing code (backward compatible)

## Notes

- This is a **contract change only** - concrete implementations will handle the parameter in milestone 7
- The mapping helper includes **runtime validation** because `Literal` types don't enforce at runtime
- We raise `ValueError` (not `KeyError`) because this is API-facing and should have a clear error message
- Tests only cover what we actually implement in M6 (mapping helper validation), not aspirational behavior
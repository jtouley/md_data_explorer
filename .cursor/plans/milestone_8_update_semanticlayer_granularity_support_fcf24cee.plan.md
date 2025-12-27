---
name: "Milestone 8: Update SemanticLayer Granularity Support"
overview: Update SemanticLayer.get_cohort() to accept and handle granularity parameter, enabling granularity-aware queries for both single-table and multi-table datasets. This completes the granularity parameter chain from ClinicalDataset → dataset implementations → SemanticLayer.
todos: []
---

# Milestone 8: Update Se

manticLayer Granularity Support

## Overview

Update `SemanticLayer.get_cohort()` to accept a `granularity` parameter and handle granularity mapping. This completes the granularity parameter chain established in Milestones 6 and 7, enabling consistent granularity support across the entire dataset API.**Context**:

- Milestone 6: Added `granularity` parameter to `ClinicalDataset.get_cohort()` base class
- Milestone 7: Updated all dataset implementations to support `granularity` parameter
- Milestone 8: Update `SemanticLayer.get_cohort()` to accept and handle `granularity` parameter

## Current State

### SemanticLayer.get_cohort() Signature

Currently, `SemanticLayer.get_cohort()` does not accept a `granularity` parameter:

```295:330:src/clinical_analytics/core/semantic.py
    def get_cohort(
        self,
        outcome_col: Optional[str] = None,
        outcome_label: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        show_sql: bool = False
    ) -> pd.DataFrame:
```



### Dataset Implementations

Single-table datasets (covid_ms, sepsis, mimic3) already:

1. Accept `granularity` parameter in their `get_cohort()` methods
2. Validate granularity (raise `ValueError` if not `patient_level`)
3. Call `self.semantic.get_cohort()` without passing granularity

Example from `CovidMSDataset`:

```52:86:src/clinical_analytics/datasets/covid_ms/definition.py
    def get_cohort(
        self,
        granularity: Granularity = "patient_level",
        **filters
    ) -> pd.DataFrame:
        # Validate: single-table datasets only support patient_level
        if granularity != "patient_level":
            raise ValueError(...)
        
        # Delegate to semantic layer - it generates SQL and executes
        return self.semantic.get_cohort(
            outcome_col=outcome_col,
            filters=filter_only,
            show_sql=False
        )
```



## Problems

1. **API Inconsistency**: `SemanticLayer.get_cohort()` doesn't match the `ClinicalDataset.get_cohort()` signature, breaking the abstraction
2. **Future Multi-Table Support**: When multi-table datasets use SemanticLayer (via materialized marts), they'll need to pass granularity
3. **Missing Parameter**: Even though single-table datasets validate granularity before calling SemanticLayer, the parameter should be part of the SemanticLayer API for consistency

## Solution Design

### 1. Update SemanticLayer.get_cohort() Signature

Add `granularity` parameter with default `"patient_level"` for backward compatibility:

```python
def get_cohort(
    self,
    granularity: Granularity = "patient_level",
    outcome_col: Optional[str] = None,
    outcome_label: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    show_sql: bool = False
) -> pd.DataFrame:
```

**Parameter Order**: Place `granularity` first (after `self`) to match `ClinicalDataset.get_cohort()` signature pattern. All internal calls must use keyword arguments to avoid positional argument churn.

### 2. Import Granularity Type

Import `Granularity` type alias from `clinical_analytics.core.dataset`:

```python
from clinical_analytics.core.dataset import Granularity
```



### 3. Granularity Handling Strategy

**Decision: SemanticLayer is Permissive**

- SemanticLayer accepts any valid `Granularity` value without validation
- Validation remains in dataset classes (current pattern: single-table datasets raise `ValueError` for non-patient_level)
- For single-table datasets: granularity is accepted but ignored (always patient-level queries)
- For multi-table datasets (future): granularity will be used to select appropriate materialized mart

**Rationale**: Keeps validation at the dataset boundary where it belongs. SemanticLayer is a query builder, not a validator.

### 4. Update build_cohort_query() Method

Add `granularity` parameter to `build_cohort_query()` for consistency:

```python
def build_cohort_query(
    self,
    granularity: Granularity = "patient_level",
    outcome_col: Optional[str] = None,
    outcome_label: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> ibis.Table:
```

**Note**: For single-table datasets, granularity doesn't affect the query (always patient-level). For multi-table datasets (future), this would select the appropriate materialized mart.

### 5. Update Dataset Implementations

Update single-table dataset implementations to pass `granularity` to SemanticLayer:**Files to update**:

- `src/clinical_analytics/datasets/covid_ms/definition.py`
- `src/clinical_analytics/datasets/sepsis/definition.py`
- `src/clinical_analytics/datasets/mimic3/definition.py`

Change from:

```python
return self.semantic.get_cohort(
    outcome_col=outcome_col,
    filters=filter_only,
    show_sql=False
)
```

To:

```python
return self.semantic.get_cohort(
    granularity=granularity,  # Pass through granularity parameter
    outcome_col=outcome_col,
    filters=filter_only,
    show_sql=False
)
```



## Implementation

### File: `src/clinical_analytics/core/semantic.py`

#### 1. Add Imports

```python
import logging
from clinical_analytics.core.dataset import Granularity

logger = logging.getLogger(__name__)
```



#### 2. Update get_cohort() Method

```python
def get_cohort(
    self,
    granularity: Granularity = "patient_level",
    outcome_col: Optional[str] = None,
    outcome_label: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    show_sql: bool = False
) -> pd.DataFrame:
    """
    Execute the cohort query and return Pandas DataFrame.
    
    This is the main entry point - generates SQL behind the scenes and executes it.
    
    Args:
        granularity: Grain level (patient_level, admission_level, event_level)
                    Accepted but not validated (validation done by dataset classes).
                    For single-table datasets, ignored (always patient-level).
                    For multi-table datasets (future), selects appropriate materialized mart.
        outcome_col: Which outcome to use
        outcome_label: Label for outcome
        filters: Optional filters
        show_sql: If True, log the generated SQL (for debugging)
        
    Returns:
        Pandas DataFrame conforming to UnifiedCohort schema
    """
    query = self.build_cohort_query(
        granularity=granularity,
        outcome_col=outcome_col,
        outcome_label=outcome_label,
        filters=filters
    )
    
    if show_sql:
        sql = query.compile()
        logger.info(f"Generated SQL for {self.dataset_name}:\n{sql}")
    
    result = query.execute()
    return result
```



#### 3. Update build_cohort_query() Method

```python
def build_cohort_query(
    self,
    granularity: Granularity = "patient_level",
    outcome_col: Optional[str] = None,
    outcome_label: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> ibis.Table:
    """
    Build a query that returns UnifiedCohort-compliant data.
    
    Args:
        granularity: Grain level (patient_level, admission_level, event_level)
                    Accepted but not validated (validation done by dataset classes).
                    For single-table datasets, ignored (always patient-level queries).
                    For multi-table datasets (future), selects appropriate materialized mart.
        outcome_col: Which outcome column to use (defaults to config)
        outcome_label: Label for outcome (defaults to config)
        filters: Optional filters to apply
        
    Returns:
        Ibis table expression (lazy - SQL not executed yet)
    """
    # Note: granularity is accepted but not used for single-table datasets.
    # For multi-table datasets (future), this would select the appropriate
    # materialized mart based on granularity.
    
    view = self.get_base_view()
    
    # Apply default filters from config
    default_filters = self.config.get('default_filters', {})
    all_filters = {**default_filters, **(filters or {})}
    
    # Remove target_outcome from filters (it's not a data filter)
    filter_only = {k: v for k, v in all_filters.items() if k != "target_outcome"}
    view = self.apply_filters(view, filter_only)
    
    # ... rest of existing implementation
```



### Files: Dataset Implementations

Update all three single-table dataset implementations to pass `granularity` parameter:

#### `src/clinical_analytics/datasets/covid_ms/definition.py`

```python
# Delegate to semantic layer - it generates SQL and executes
return self.semantic.get_cohort(
    granularity=granularity,  # Pass through granularity
    outcome_col=outcome_col,
    filters=filter_only,
    show_sql=False
)
```



#### `src/clinical_analytics/datasets/sepsis/definition.py`

```python
# Delegate to semantic layer
return self.semantic.get_cohort(
    granularity=granularity,  # Pass through granularity
    outcome_col=outcome_col,
    filters=filter_only,
    show_sql=False
)
```



#### `src/clinical_analytics/datasets/mimic3/definition.py`

```python
return self.semantic.get_cohort(
    granularity=granularity,  # Pass through granularity
    outcome_col=outcome_col,
    filters=filter_only
)
```



## Testing Strategy

### Unit Tests

1. **Granularity parameter wiring**
   - Test that `get_cohort(granularity="patient_level")` works without error
   - Test that `build_cohort_query(granularity=...)` receives and accepts granularity parameter
   - Test that all three granularity values are accepted without error (permissive behavior)

2. **Backward compatibility**
   - Test that calling `get_cohort()` without `granularity` parameter still works (uses default)
   - Test that existing dataset implementations continue to work
   - Test that `show_sql=True` logs SQL instead of printing

### Integration Tests

4. **Dataset implementations pass granularity correctly**

- Test that `CovidMSDataset.get_cohort(granularity="patient_level")` works
- Test that `SepsisDataset.get_cohort(granularity="patient_level")` works
- Test that `Mimic3Dataset.get_cohort(granularity="patient_level")` works
- Verify that non-patient_level granularity still raises `ValueError` at dataset level (not SemanticLayer level)

## Dependencies

- **Milestone 6**: Must be completed (granularity parameter in base class)
- **Milestone 7**: Must be completed (granularity support in dataset implementations)

## Success Criteria

- [ ] `SemanticLayer.get_cohort()` signature includes `granularity` parameter with default `"patient_level"`
- [ ] `SemanticLayer.build_cohort_query()` signature includes `granularity` parameter
- [ ] All single-table dataset implementations pass `granularity` to SemanticLayer
- [ ] Existing functionality unchanged (backward compatible)
- [ ] Unit tests verify granularity parameter wiring and permissive behavior
- [ ] `show_sql=True` uses `logger.info()` instead of `print()`
- [ ] Integration tests verify dataset implementations work correctly
- [ ] No breaking changes to existing code

## Migration Notes

- **Backward Compatibility**: Default `granularity="patient_level"` ensures existing code continues to work
- **No Behavior Change**: For single-table datasets, granularity is accepted but ignored (always patient-level queries)
- **Permissive Validation**: SemanticLayer accepts any valid Granularity without validation (validation stays in dataset classes)
- **Logging**: `show_sql=True` uses `logger.info()` instead of `print()` (consistent with M7 logging standards)
- **Future-Proof**: Parameter is in place for multi-table dataset support (when SemanticLayer works with materialized marts)

## Future Work (Out of Scope for M8)
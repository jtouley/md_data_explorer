---
name: "Address Staff Engineer Feedback: State Machine Documentation and UI DRY Refactoring"
overview: Address staff engineer feedback by documenting the session state machine, extracting duplicated dataset loading logic into a reusable component, and documenting technical debt for future refactoring. This plan prioritizes high-ROI fixes (UI duplication) while properly documenting architectural debt (state machine, god objects) without premature refactoring.
todos: []
---

# Address Staff Engineer Feedback: State Machine Documentation and UI DRY Refactoring

## Overview

This plan addresses the staff engineer review feedback by:

1. **Documenting the session state machine** (per staff recommendation - don't refactor yet)
2. **Extracting duplicated dataset loading** (high ROI - eliminates 350-560 lines of duplication)
3. **Documenting technical debt** (god objects, persistence gap) for future refactoring
4. **Moving common get_cohort logic** to base class (medium priority)

## Context

Staff engineer identified:

- Session state acting as fragile mini state machine (`analysis_context`, `intent_signal`, `use_nl_query`)
- NLQueryEngine drifting toward "god object" (acceptable for MVP, but watch boundary)
- UI pages have massive duplication (dataset loading pattern repeated 7+ times)
- Persistence gap affecting NL quality (upstream issue, not NL problem)

**Staff guidance**: Document state machine, don't refactor yet. Fix UI duplication (high ROI). Defer god object splits (acceptable MVP debt).

## Implementation Plan

### Phase 1: Document Session State Machine (Low Effort, High Value)

**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`Add comprehensive state machine documentation at the top of `main()` function, after imports but before logic:

```python
# STATE MACHINE DOCUMENTATION (Temporary - see ADR008 for future refactor)
#
# The Ask Questions page uses Streamlit session state as a mini state machine
# to manage analysis context and execution flow. This is fragile but acceptable for MVP.
#
# State Keys:
#   - analysis_context: AnalysisContext | None
#       Stores the current analysis configuration (variables, intent, filters)
#       Set when NL query is parsed or user answers clarifying questions
#   - intent_signal: "nl_parsed" | None
#       Signals that NL parsing completed and context is ready
#       Used to trigger analysis execution flow
#   - use_nl_query: bool (legacy, may be removed in future)
#       Legacy flag for NL query mode (kept for backward compatibility)
#
# Allowed Transitions:
#   1. None -> "nl_parsed"
#      Trigger: User submits NL query, parsing succeeds
#      Action: Set analysis_context, set intent_signal="nl_parsed"
#      Location: ~line 1519-1520
#
#   2. "nl_parsed" -> None
#      Trigger: User changes dataset, clears conversation, or resets
#      Action: Clear analysis_context, set intent_signal=None
#      Location: ~line 1088-1089, 1157-1158
#
#   3. "nl_parsed" -> "executed" (implicit)
#      Trigger: Context is complete, analysis executes
#      Action: Execute analysis, render results, keep context for follow-ups
#      Location: ~line 1274-1299
#
# Fragility Notes:
#   - Order matters: Must check intent_signal before accessing analysis_context
#   - Reruns can invalidate assumptions if state is modified mid-cycle
#   - Future contributors will break this accidentally without explicit docs
#   - State is not validated - invalid states can cause silent failures
#
# Future Refactor (Post-MVP):
#   - Extract to StateManager class with explicit transition methods
#   - Add state validation and transition guards
#   - Use state machine library (e.g., transitions) for robustness
#   - See ADR008 for detailed refactoring plan
```

**Location**: Insert after line 1163 (after session state initialization), before line 1164.

### Phase 2: Extract Dataset Loading Component (High ROI)

**New File**: `src/clinical_analytics/ui/components/dataset_loader.py`Create reusable component that encapsulates all dataset loading logic:

```python
"""
Dataset Loader Component - Unified dataset selection and loading.

Eliminates duplication across UI pages by providing a single component
for dataset selection, loading, and error handling.
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clinical_analytics.core.dataset import ClinicalDataset
    import pandas as pd

from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory


@st.cache_data(show_spinner=False)
def _get_dataset_display_names() -> tuple[dict[str, str], dict[str, dict]]:
    """
    Get dataset display names mapping (cached for performance).
    
    Returns:
        Tuple of (display_names dict, uploaded_datasets dict)
    """
    available_datasets = DatasetRegistry.list_datasets()
    dataset_info = DatasetRegistry.get_all_dataset_info()
    
    dataset_display_names = {}
    for ds_name in available_datasets:
        info = dataset_info[ds_name]
        display_name = info["config"].get("display_name", ds_name.replace("_", "-").upper())
        dataset_display_names[display_name] = ds_name
    
    uploaded_datasets = {}
    uploaded_ids = set()
    try:
        uploads = UploadedDatasetFactory.list_available_uploads()
        for upload in uploads:
            upload_id = upload["upload_id"]
            dataset_name = upload.get("dataset_name", upload_id)
            display_name = f"📤 {dataset_name}"
            dataset_display_names[display_name] = upload_id
            uploaded_datasets[upload_id] = upload
            uploaded_ids.add(upload_id)
    except Exception:
        pass
    
    return dataset_display_names, uploaded_datasets


def render_dataset_selector(sidebar: bool = True) -> tuple[str, str, bool] | None:
    """
    Render dataset selector in sidebar or main area.
    
    Args:
        sidebar: If True, render in sidebar. If False, render in main area.
    
    Returns:
        Tuple of (dataset_choice, dataset_choice_display, is_uploaded) or None if no datasets
    """
    dataset_display_names, uploaded_datasets = _get_dataset_display_names()
    
    if not dataset_display_names:
        container = st.sidebar if sidebar else st
        container.error("No datasets available. Please upload data first.")
        if not sidebar:
            container.info("👈 Go to **Add Your Data** to upload your first dataset")
        return None
    
    container = st.sidebar if sidebar else st
    container.header("Data Selection")
    
    dataset_choice_display = container.selectbox(
        "Choose Dataset",
        list(dataset_display_names.keys()),
        key="dataset_selector" if sidebar else None
    )
    dataset_choice = dataset_display_names[dataset_choice_display]
    
    # Check if uploaded (multiple checks for robustness)
    is_uploaded = (
        dataset_choice in uploaded_datasets
        or dataset_choice_display.startswith("📤")
        or dataset_choice not in DatasetRegistry.list_datasets()
    )
    
    return dataset_choice, dataset_choice_display, is_uploaded


def load_dataset_with_cohort(
    dataset_choice: str,
    dataset_choice_display: str,
    is_uploaded: bool,
    show_spinner: bool = True,
) -> tuple["ClinicalDataset", "pd.DataFrame"] | None:
    """
    Load dataset and return dataset instance and cohort DataFrame.
    
    Args:
        dataset_choice: Internal dataset identifier
        dataset_choice_display: Display name for user feedback
        is_uploaded: Whether this is an uploaded dataset
        show_spinner: Whether to show loading spinner
    
    Returns:
        Tuple of (dataset, cohort) or None if loading failed
    """
    spinner_context = st.spinner(f"Loading {dataset_choice_display}...") if show_spinner else st.empty()
    
    with spinner_context:
        try:
            if is_uploaded:
                dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
                dataset.load()
            else:
                dataset = DatasetRegistry.get_dataset(dataset_choice)
                if not dataset.validate():
                    st.error("Dataset validation failed")
                    return None
                dataset.load()
            
            cohort = dataset.get_cohort()
            
            if cohort.empty:
                st.error("No data in dataset")
                return None
            
            return dataset, cohort
            
        except KeyError:
            st.error(f"Dataset '{dataset_choice}' not found in registry.")
            st.info("💡 If this is an uploaded dataset, please refresh the page or check the upload status.")
            return None
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            if show_spinner:
                st.exception(e)
            return None


def load_selected_dataset(sidebar: bool = True) -> tuple["ClinicalDataset", "pd.DataFrame"] | None:
    """
    Complete workflow: render selector, load dataset, return results.
    
    This is the main entry point for most pages.
    
    Args:
        sidebar: Whether to render selector in sidebar
    
    Returns:
        Tuple of (dataset, cohort) or None if selection/loading failed
    """
    selector_result = render_dataset_selector(sidebar=sidebar)
    if selector_result is None:
        return None
    
    dataset_choice, dataset_choice_display, is_uploaded = selector_result
    return load_dataset_with_cohort(dataset_choice, dataset_choice_display, is_uploaded)
```

**Refactor Target Files** (replace duplicated code with component):

1. `src/clinical_analytics/ui/pages/20_📊_Descriptive_Stats.py` (lines 148-228)
2. `src/clinical_analytics/ui/pages/21_📈_Compare_Groups.py` (lines 169-223)
3. `src/clinical_analytics/ui/pages/22_🎯_Risk_Factors.py` (lines 48-101)
4. `src/clinical_analytics/ui/pages/23_⏱️_Survival_Analysis.py` (lines 79-142)
5. `src/clinical_analytics/ui/pages/24_🔗_Correlations.py` (lines 143-207)
6. `src/clinical_analytics/ui/pages/2_📊_Your_Dataset.py` (lines 30-74)
7. `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (lines 1049-1092, 1092-1110)

**Refactoring Pattern** (for each file):**Before**:

```python
# Load available datasets
available_datasets = DatasetRegistry.list_datasets()
dataset_info = DatasetRegistry.get_all_dataset_info()
# ... 50+ lines of duplicated code ...
cohort = dataset.get_cohort()
```

**After**:

```python
from clinical_analytics.ui.components.dataset_loader import load_selected_dataset

dataset, cohort = load_selected_dataset()
if dataset is None or cohort is None:
    return
```



### Phase 3: Move Common get_cohort Logic to Base Class

**File**: `src/clinical_analytics/core/dataset.py`Add default implementation to `ClinicalDataset` base class:

```python
def get_cohort(
    self,
    granularity: Granularity = "patient_level",
    **filters: Any,
) -> pd.DataFrame:
    """
    Default implementation using semantic layer.
    
    Most datasets can use this implementation. Override only if custom logic needed.
    """
    # Extract outcome override if provided
    outcome_col = filters.pop("target_outcome", None)
    
    # Delegate to semantic layer
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
```

**Update Dataset Classes** to remove duplication:

1. `src/clinical_analytics/datasets/covid_ms/definition.py` (lines 50-80) - Can remove entire method
2. `src/clinical_analytics/datasets/mimic3/definition.py` (lines 72-95) - Can remove entire method
3. `src/clinical_analytics/datasets/sepsis/definition.py` (lines 89-120) - Keep override (has custom validation)

### Phase 4: Document Technical Debt (ADR008)

**New File**: `docs/implementation/ADR/ADR008.md`Create ADR documenting architectural debt identified by staff engineer:

```markdown
# ADR008: Technical Debt Documentation (Post-MVP Refactoring)

## Status
**DEFERRED** - Documented for post-MVP refactoring

## Context

Staff engineer review identified several architectural concerns that are acceptable for MVP but should be addressed post-MVP:

1. **Session State Machine Fragility** (Ask Questions page)
2. **God Object Pattern** (NLQueryEngine, SemanticLayer, ColumnMapper)
3. **Persistence Gap** (single-table uploads have weaker metadata)

## Decisions

### 1. Session State Machine (Deferred)

**Current State**: Streamlit session state acts as implicit state machine
**Risk**: Fragile, order-dependent, breaks on reruns
**Decision**: Document transitions (see Phase 1), defer refactor to post-MVP

**Future Refactor**:
- Extract to `StateManager` class
- Use state machine library (e.g., `transitions`)
- Add transition validation and guards

### 2. God Object Pattern (Deferred)

**Current State**: 
- `NLQueryEngine`: Intent classification, variable extraction, collision handling, suggestions
- `SemanticLayer`: SQL generation, registration, execution, alias management
- `ColumnMapper`: Mapping, filtering, aggregation, outcomes

**Risk**: Hard to test, violates SRP, difficult to extend
**Decision**: Acceptable for MVP, watch boundary (don't add UI decisions)

**Future Refactor**:
- Split `NLQueryEngine` into: `IntentClassifier`, `VariableExtractor`, `CollisionHandler`
- Split `SemanticLayer` into: `DataSourceRegistry`, `SQLGenerator`, `QueryExecutor`
- Split `ColumnMapper` into: `ColumnMapper`, `FilterEngine`, `AggregationEngine`

### 3. Persistence Gap (Deferred)

**Current State**: Single-table uploads have weaker semantic metadata than multi-table
**Impact**: NL accuracy depends on metadata quality
**Decision**: Document as upstream issue, don't over-optimize NL heuristics

**Future Fix**: Ensure single-table uploads generate equivalent metadata (see ADR007)

## Consequences

- **Positive**: MVP can ship without blocking refactors
- **Negative**: Technical debt accumulates, harder to maintain long-term
- **Mitigation**: Document clearly, prioritize post-MVP

## References

- Staff Engineer Review (2025-01-XX)
- ADR001: Query Plan Producer
- ADR007: Feature Parity Architecture
```



## Testing Strategy

### Phase 1 (Documentation)

- No code changes, no tests needed
- Verify documentation is clear and accurate

### Phase 2 (Dataset Loader)

- **Unit Tests**: `tests/ui/test_dataset_loader.py`
- Test `_get_dataset_display_names()` caching
- Test `render_dataset_selector()` with/without datasets
- Test `load_dataset_with_cohort()` error handling
- Test `load_selected_dataset()` complete workflow
- **Integration Tests**: Update existing page tests
- Verify pages still work after refactoring
- Test error handling (no datasets, invalid dataset, etc.)

### Phase 3 (Base Class)

- **Unit Tests**: Update `tests/core/test_dataset.py`
- Test default `get_cohort()` implementation
- Test that datasets can override if needed
- Test error when semantic layer not initialized

### Phase 4 (Documentation)

- No code changes, no tests needed

## Success Criteria

1. ✅ Session state machine fully documented with transitions
2. ✅ Dataset loading duplication eliminated (7 files refactored)
3. ✅ Common `get_cohort()` logic moved to base class (2-3 classes simplified)
4. ✅ Technical debt documented in ADR008
5. ✅ All existing tests pass
6. ✅ No regression in UI functionality

## Risk Mitigation

- **Risk**: Refactoring breaks existing pages
- **Mitigation**: Comprehensive tests, incremental rollout (one page at a time)
- **Risk**: Caching in dataset loader causes stale data
- **Mitigation**: Use `@st.cache_data` with appropriate TTL, clear cache on upload
- **Risk**: Base class change breaks custom implementations
- **Mitigation**: Keep override capability, test all dataset types

## Timeline Estimate

- **Phase 1**: 30 minutes (documentation only)
- **Phase 2**: 4-6 hours (component + 7 file refactors + tests)
- **Phase 3**: 1-2 hours (base class + 2-3 file updates + tests)
- **Phase 4**: 1 hour (ADR writing)

**Total**: ~7-10 hours

## Dependencies

- None (all phases are independent and can be done in parallel or sequence)

## Related Work

- ADR001: Query Plan Producer (state machine context)
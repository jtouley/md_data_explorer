---
name: "Address Staff Engineer Feedback: State Machine Documentation and UI DRY Refactoring"
overview: Address staff engineer feedback by documenting the session state machine, extracting duplicated dataset loading logic into a reusable component, and documenting technical debt for future refactoring. This plan prioritizes high-ROI fixes (UI duplication) while properly documenting architectural debt (state machine, god objects) without premature refactoring. Also includes PR20 fixes: cache primitive correction, empty query rejection, scope canonicalization, runtime guardrails, transcript schema enhancements, pending lifecycle validation, and integration tests.
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

### Initial Staff Engineer Review

Staff engineer identified:

- Session state acting as fragile mini state machine (`analysis_context`, `intent_signal`, `use_nl_query`)
- NLQueryEngine drifting toward "god object" (acceptable for MVP, but watch boundary)
- UI pages have massive duplication (dataset loading pattern repeated 7+ times)
- Persistence gap affecting NL quality (upstream issue, not NL problem)

**Staff guidance**: Document state machine, don't refactor yet. Fix UI duplication (high ROI). Defer god object splits (acceptable MVP debt).

### PR20 Staff Engineer Review (Ask Questions chat transcript + run_key)

Additional feedback on PR20 identified critical fixes needed:

**P0 (Must-fix)**:
- `st.cache_data` wrong for semantic layer (should be `st.cache_resource`)
- Empty query rejection at ingestion (normalize returns `""` for `None`, but should reject)
- Scope canonicalization needs recursive determinism

**P1 (Should-fix)**:
- Runtime guardrails for normalized-only contract (assertions, not just comments)
- Transcript message schema needs enhancement (query_text, assistant_text, intent, confidence)
- Pending lifecycle must be proven safe (no infinite rerun loops)
- Integration tests missing for rerun artifacts

**P2 (Nits)**:
- Type improvements (Literal types, QueryPlan Protocol, Pandas/Polars consistency)

**Verdict**: Approve with changes - architecture shift is correct, but fix cache primitive and empty-query ingestion behavior.

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

### Phase 5: Fix Cache Primitive for Semantic Layer (P0 - Must Fix)

**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`

**Issue**: `st.cache_data` is wrong for semantic layer objects. If `get_cached_semantic_layer()` returns non-trivially picklable objects (DB/ibis connections, lazy backends), `st.cache_data` will fail at runtime.

**Fix**: Change from `@st.cache_data` to `@st.cache_resource` for long-lived objects/resources.

**Location**: Find `get_cached_semantic_layer()` function and change decorator:

```python
# BEFORE
@st.cache_data
def get_cached_semantic_layer(...):
    ...

# AFTER
@st.cache_resource
def get_cached_semantic_layer(...):
    """
    Cache semantic layer as a resource (not data).
    
    Semantic layers contain connections/backends that are not picklable.
    Use st.cache_resource for long-lived objects, st.cache_data for pure data.
    """
    ...
```

**Testing**:
- Verify semantic layer caching works correctly
- Test that connections/backends are preserved across reruns
- Ensure no pickling errors occur

### Phase 6: Reject Empty Queries at Ingestion (P0 - Must Fix)

**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`

**Issue**: `normalize_query()` returns `""` for `None`, but the ingestion contract should reject empty queries. Currently, empty queries can:
- Generate a run_key for `""`
- Append transcript messages for `""`
- Attempt computation for `""`

**Fix**: Add early return/no-op for empty normalized queries in the main flow.

**Location**: After query normalization, before run_key generation:

```python
# After normalize_query() call
normalized_query = normalize_query(user_query)

# Add guard: reject empty queries at ingestion
if not normalized_query or normalized_query.strip() == "":
    # No-op: don't generate run_key, don't append message, don't compute
    return

# Only proceed if query is non-empty
run_key = generate_run_key(normalized_query, scope)
```

**Testing**:
- Test that `None` input is rejected (no run_key, no message, no compute)
- Test that `""` input is rejected
- Test that whitespace-only queries are rejected
- Verify valid queries still work normally

### Phase 7: Recursive Deterministic Scope Canonicalization (P0 - Must Fix)

**File**: `src/clinical_analytics/core/nl_query.py` (or wherever `canonicalize_scope()` lives)

**Issue**: Current `canonicalize_scope()` only handles shallow cases (key order, list order). Needs to handle:
- Nested dict/list structures
- Non-JSON-native values (enums, dataclasses)
- Semantically equivalent scopes producing stable output

**Fix**: Implement recursive canonicalization:

```python
def canonicalize_scope(scope: dict | None) -> dict:
    """
    Recursively canonicalize scope for deterministic hashing.
    
    Handles:
    - Nested dicts/lists
    - Key sorting at all levels
    - List sorting (if order doesn't matter)
    - Enum/dataclass normalization (convert to primitives)
    - Non-JSON-native value handling
    """
    if scope is None:
        return {}
    
    if isinstance(scope, dict):
        # Sort keys and recursively canonicalize values
        return {
            k: canonicalize_scope(v)
            for k, v in sorted(scope.items())
        }
    elif isinstance(scope, list):
        # Sort list items if they're comparable, otherwise preserve order
        # For deterministic hashing, we may need to sort if order doesn't matter
        # OR preserve order if it does matter semantically
        canonicalized = [canonicalize_scope(item) for item in scope]
        # If all items are comparable, sort for stability
        try:
            return sorted(canonicalized)
        except TypeError:
            # Mixed types or non-comparable - preserve order but canonicalize items
            return canonicalized
    elif hasattr(scope, '__dict__'):  # Dataclass or object
        # Convert to dict and recurse
        return canonicalize_scope(scope.__dict__)
    elif hasattr(scope, 'value'):  # Enum
        # Use enum value
        return scope.value
    else:
        # Primitive type - return as-is
        return scope
```

**Testing**:
- Test nested dicts (2+ levels deep)
- Test nested lists
- Test mixed dict/list structures
- Test enum values
- Test dataclass objects
- Test that semantically equivalent scopes produce identical output
- Test non-JSON-native values (fail loudly or normalize appropriately)

### Phase 8: Runtime Guardrails for Normalized-Only Contract (P1 - Should Fix)

**File**: `src/clinical_analytics/core/nl_query.py` (or wherever `generate_run_key()` lives)

**Issue**: Comments don't enforce contracts. `generate_run_key()` should assert that input is already normalized.

**Fix**: Add cheap validation assertions:

```python
def generate_run_key(
    normalized_query: str,  # Must be pre-normalized
    scope: dict | None = None,
) -> str:
    """
    Generate deterministic run key from normalized query and scope.
    
    Args:
        normalized_query: Query text that has already been normalized
            (lowercase, trimmed, no double spaces). Must not be None or empty.
        scope: Optional semantic scope constraints
    
    Raises:
        ValueError: If query is not normalized (has leading/trailing spaces,
            double spaces, or is not lowercase)
    """
    # Runtime guardrails: enforce normalized-only contract
    if not normalized_query:
        raise ValueError("normalized_query cannot be empty or None")
    
    # Check for normalization violations (cheap assertions)
    if normalized_query != normalized_query.strip():
        raise ValueError(
            f"Query not normalized: has leading/trailing spaces. "
            f"Got: {repr(normalized_query)}"
        )
    if "  " in normalized_query:
        raise ValueError(
            f"Query not normalized: contains double spaces. "
            f"Got: {repr(normalized_query)}"
        )
    if normalized_query != normalized_query.lower():
        raise ValueError(
            f"Query not normalized: not lowercase. "
            f"Got: {repr(normalized_query)}"
        )
    
    # Proceed with key generation
    ...
```

**Testing**:
- Test that non-normalized queries raise ValueError
- Test that normalized queries work correctly
- Test error messages are clear

### Phase 9: Enhance Transcript Message Schema (P1 - Should Fix)

**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`

**Issue**: Current `ChatMessage` schema only stores `text` and `run_key`. Future needs:
- `query_text`: Original user query that produced this result
- `assistant_text`: What was displayed to user
- `intent` / `confidence`: For history UX without recompute

**Fix**: Extend `ChatMessage` TypedDict:

```python
from typing import TypedDict, Literal

class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    text: str  # For user: query text; for assistant: displayed text
    # New fields for assistant messages
    query_text: str | None  # Original user query that produced this result
    assistant_text: str | None  # What was displayed (may differ from text)
    intent: str | None  # Intent classification (e.g., "correlation", "comparison")
    confidence: float | None  # Confidence score (0.0-1.0)
    run_key: str | None  # Run key for this message (assistant only)
    status: Literal["pending", "completed", "error"] | None  # Message status
```

**Update** `render_chat()` and message creation to populate new fields:

```python
# When creating assistant message
assistant_message: ChatMessage = {
    "role": "assistant",
    "text": displayed_text,
    "query_text": original_user_query,  # Store original
    "assistant_text": displayed_text,  # Store what we showed
    "intent": analysis_context.intent if analysis_context else None,
    "confidence": analysis_context.confidence if analysis_context else None,
    "run_key": run_key,
    "status": "completed",
}
```

**Testing**:
- Test that new fields are populated correctly
- Test backward compatibility (old messages without new fields)
- Test that history UX can use intent/confidence without recompute

### Phase 10: Prove Pending Lifecycle (No Infinite Reruns) (P1 - Should Fix)

**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`

**Issue**: Pending lifecycle must be proven safe. Flow must be:
1. Set pending -> `st.rerun()`
2. On next run, detect pending -> compute -> clear pending -> append assistant message -> `st.rerun()` (optional)

If pending is not cleared on all code paths (error/exception), infinite rerun loops occur.

**Fix**: Add comprehensive error handling and state machine validation:

```python
# Pending state machine
if "pending_query" in st.session_state:
    pending = st.session_state.pending_query
    
    try:
        # Compute analysis
        result = execute_analysis(pending.query_plan, ...)
        
        # Success: clear pending, append message, optional rerun
        del st.session_state.pending_query
        append_assistant_message(result, ...)
        # Optional: st.rerun() if needed for UI update
        
    except Exception as e:
        # CRITICAL: Clear pending even on error to prevent infinite loop
        del st.session_state.pending_query
        
        # Append error message
        append_error_message(str(e), ...)
        
        # Log error for debugging
        logger.error("Analysis failed", exc_info=e)
        
        # Don't rerun - show error to user
```

**Add validation** to detect stuck pending states:

```python
# At start of main(), check for stuck pending
if "pending_query" in st.session_state:
    pending = st.session_state.pending_query
    # Add timestamp check - if pending > 30 seconds, clear it
    if hasattr(pending, "created_at"):
        if (time.time() - pending.created_at) > 30:
            logger.warning("Clearing stuck pending query")
            del st.session_state.pending_query
```

**Testing**:
- Test normal flow: pending -> compute -> clear -> message
- Test error flow: pending -> exception -> clear -> error message (no infinite loop)
- Test stuck pending detection and cleanup
- Test that reruns don't create infinite loops

### Phase 11: Integration Tests for Rerun Artifacts (P1 - Should Fix)

**New File**: `tests/ui/test_ask_questions_rerun.py`

**Issue**: Unit tests cover hashing, but integration tests are missing for the original regression:
- "rerun does not create empty emoji containers"
- "two different queries do not collide run_key (BMI vs LDL style case)"

**Fix**: Create integration tests:

```python
"""
Integration tests for Ask Questions page rerun behavior.

Tests the original regression cases:
- Reruns don't create empty containers
- Query collisions don't occur
"""

import pytest
import streamlit as st
from clinical_analytics.ui.pages.Ask_Questions import (
    normalize_query,
    generate_run_key,
    render_chat,
)

@pytest.mark.integration
@pytest.mark.slow
def test_rerun_does_not_create_empty_containers(mock_session_state):
    """
    Regression test: Reruns should not create empty emoji containers.
    
    Original bug: Reruns would create empty containers when state was inconsistent.
    """
    # Setup: Create transcript with messages
    transcript = [
        {"role": "user", "text": "show me correlations"},
        {"role": "assistant", "text": "Here are correlations...", "run_key": "abc123"},
    ]
    mock_session_state["chat_transcript"] = transcript
    
    # Act: Render chat (simulating rerun)
    render_chat(transcript)
    
    # Assert: No empty containers created
    # (This test may need to inspect Streamlit's rendered output)
    # For now, verify transcript is unchanged
    assert len(mock_session_state["chat_transcript"]) == 2


@pytest.mark.integration
@pytest.mark.slow
def test_different_queries_do_not_collide_run_key():
    """
    Regression test: Two different queries should not collide run_key.
    
    Original bug: "BMI" vs "LDL" queries would collide if normalization
    was too aggressive or scope wasn't included.
    """
    query1 = "What is the correlation between BMI and age?"
    query2 = "What is the correlation between LDL and age?"
    
    normalized1 = normalize_query(query1)
    normalized2 = normalize_query(query2)
    
    # Different queries should normalize differently
    assert normalized1 != normalized2
    
    # Different run keys even with same scope
    scope = {"dataset": "mimic"}
    key1 = generate_run_key(normalized1, scope)
    key2 = generate_run_key(normalized2, scope)
    
    assert key1 != key2, "Different queries should produce different run keys"


@pytest.mark.integration
@pytest.mark.slow
def test_same_query_same_scope_same_key():
    """Same query + same scope = same run key (deterministic)."""
    query = "show correlations"
    scope = {"dataset": "mimic", "filters": {"age": ">50"}}
    
    normalized = normalize_query(query)
    key1 = generate_run_key(normalized, scope)
    key2 = generate_run_key(normalized, scope)
    
    assert key1 == key2, "Same inputs should produce same key"


@pytest.mark.integration
@pytest.mark.slow
def test_same_query_different_scope_different_key():
    """Same query + different scope = different run key."""
    query = "show correlations"
    normalized = normalize_query(query)
    
    scope1 = {"dataset": "mimic"}
    scope2 = {"dataset": "sepsis"}
    
    key1 = generate_run_key(normalized, scope1)
    key2 = generate_run_key(normalized, scope2)
    
    assert key1 != key2, "Different scopes should produce different keys"
```

**Testing**:
- Run integration tests with `make test-integration`
- Verify all regression cases are covered
- Ensure tests catch the original bugs if reintroduced

### Phase 12: Type Improvements (P2 - Nits)

**Files**: 
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (ChatMessage, Pending)
- Any files using Pandas/Polars interchangeably

**Issue**: 
1. `role` and `status` fields should be `Literal` types, not `str`
2. `Pending.query_plan: object | None` is a smell - should use proper type or Protocol
3. Avoid passing Pandas/Polars interchangeably unless necessary

**Fix 1**: Type `role` and `status` as Literal:

```python
from typing import Literal, TypedDict

class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    status: Literal["pending", "completed", "error"] | None
    ...
```

**Fix 2**: Type `query_plan` properly:

```python
from typing import Protocol

class QueryPlan(Protocol):
    """Protocol for query plans."""
    intent: str
    variables: list[str]
    # Add other expected attributes

class Pending(TypedDict):
    query_text: str
    query_plan: QueryPlan | None  # Use Protocol instead of object
    scope: dict | None
    created_at: float
```

**Fix 3**: Standardize on Polars internally, convert to Pandas only at render boundary:

```python
# In compute paths: use Polars
def compute_analysis(...) -> pl.DataFrame:
    # All internal processing in Polars
    result = pl.DataFrame(...)
    return result

# At render boundary: convert to Pandas if needed
# PANDAS EXCEPTION: Required for Streamlit st.dataframe display
# TODO: Remove when Streamlit supports Polars natively
import pandas as pd
display_df = result.to_pandas()
st.dataframe(display_df)
```

**Testing**:
- Run `make type-check` to verify type improvements
- Ensure no type errors introduced
- Verify Polars-first pattern is followed

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

### Phase 5 (Cache Primitive Fix)

- **Integration Tests**: Verify semantic layer caching works with `st.cache_resource`
- Test that connections/backends are preserved across reruns
- Test that no pickling errors occur

### Phase 6 (Empty Query Rejection)

- **Unit Tests**: `tests/ui/test_ask_questions.py`
- Test that `None` input is rejected (no run_key, no message, no compute)
- Test that `""` input is rejected
- Test that whitespace-only queries are rejected
- Verify valid queries still work normally

### Phase 7 (Recursive Scope Canonicalization)

- **Unit Tests**: `tests/core/test_nl_query.py` (or appropriate test file)
- Test nested dicts (2+ levels deep)
- Test nested lists
- Test mixed dict/list structures
- Test enum values
- Test dataclass objects
- Test that semantically equivalent scopes produce identical output
- Test non-JSON-native values (fail loudly or normalize appropriately)

### Phase 8 (Runtime Guardrails)

- **Unit Tests**: `tests/core/test_nl_query.py`
- Test that non-normalized queries raise ValueError
- Test that normalized queries work correctly
- Test error messages are clear

### Phase 9 (Transcript Schema Enhancement)

- **Unit Tests**: `tests/ui/test_ask_questions.py`
- Test that new fields are populated correctly
- Test backward compatibility (old messages without new fields)
- Test that history UX can use intent/confidence without recompute

### Phase 10 (Pending Lifecycle)

- **Unit Tests**: `tests/ui/test_ask_questions.py`
- Test normal flow: pending -> compute -> clear -> message
- Test error flow: pending -> exception -> clear -> error message (no infinite loop)
- Test stuck pending detection and cleanup
- Test that reruns don't create infinite loops

### Phase 11 (Integration Tests)

- **Integration Tests**: `tests/ui/test_ask_questions_rerun.py`
- Test rerun does not create empty containers
- Test different queries do not collide run_key
- Test same query + same scope = same key
- Test same query + different scope = different key

### Phase 12 (Type Improvements)

- **Type Checking**: Run `make type-check` to verify type improvements
- Ensure no type errors introduced
- Verify Polars-first pattern is followed

## Success Criteria

1. ✅ Session state machine fully documented with transitions
2. ✅ Dataset loading duplication eliminated (7 files refactored)
3. ✅ Common `get_cohort()` logic moved to base class (2-3 classes simplified)
4. ✅ Technical debt documented in ADR008
5. ✅ Semantic layer uses `st.cache_resource` (not `st.cache_data`)
6. ✅ Empty queries rejected at ingestion (no run_key, no message, no compute)
7. ✅ Scope canonicalization handles nested structures recursively
8. ✅ Runtime guardrails enforce normalized-only contract
9. ✅ Transcript message schema includes query_text, assistant_text, intent, confidence
10. ✅ Pending lifecycle proven safe (no infinite rerun loops)
11. ✅ Integration tests cover rerun artifacts and query collisions
12. ✅ Type improvements: Literal types, QueryPlan Protocol, Polars-first
13. ✅ All existing tests pass
14. ✅ No regression in UI functionality
15. ✅ All quality gates pass (`make check`)

## Risk Mitigation

- **Risk**: Refactoring breaks existing pages
- **Mitigation**: Comprehensive tests, incremental rollout (one page at a time)
- **Risk**: Caching in dataset loader causes stale data
- **Mitigation**: Use `@st.cache_data` with appropriate TTL, clear cache on upload
- **Risk**: Base class change breaks custom implementations
- **Mitigation**: Keep override capability, test all dataset types
- **Risk**: Cache primitive change breaks semantic layer behavior
- **Mitigation**: Test thoroughly with real semantic layers, verify connections preserved
- **Risk**: Empty query rejection breaks existing workflows
- **Mitigation**: Test edge cases, ensure valid queries still work
- **Risk**: Recursive canonicalization too aggressive (sorts when order matters)
- **Mitigation**: Test with real scopes, verify semantic equivalence preserved
- **Risk**: Runtime guardrails too strict (false positives)
- **Mitigation**: Test with real queries, ensure normalization contract is clear
- **Risk**: Transcript schema changes break backward compatibility
- **Mitigation**: Handle missing fields gracefully, test with old transcripts
- **Risk**: Pending lifecycle fixes introduce new bugs
- **Mitigation**: Comprehensive error handling tests, stuck state detection
- **Risk**: Integration tests miss edge cases
- **Mitigation**: Test original regression cases, add more as discovered

## Timeline Estimate

- **Phase 1**: 30 minutes (documentation only)
- **Phase 2**: 4-6 hours (component + 7 file refactors + tests)
- **Phase 3**: 1-2 hours (base class + 2-3 file updates + tests)
- **Phase 4**: 1 hour (ADR writing)
- **Phase 5**: 30 minutes (cache primitive fix + tests)
- **Phase 6**: 1 hour (empty query rejection + tests)
- **Phase 7**: 2-3 hours (recursive canonicalization + comprehensive tests)
- **Phase 8**: 1 hour (runtime guardrails + tests)
- **Phase 9**: 2 hours (transcript schema enhancement + tests)
- **Phase 10**: 2-3 hours (pending lifecycle validation + tests)
- **Phase 11**: 2-3 hours (integration tests for rerun artifacts)
- **Phase 12**: 1-2 hours (type improvements + type-check verification)

**Total**: ~19-25 hours

## Dependencies

- None (all phases are independent and can be done in parallel or sequence)

## Related Work

- ADR001: Query Plan Producer (state machine context)
- PR20: Ask Questions chat transcript + run_key fixes (Phases 5-12 address staff feedback)
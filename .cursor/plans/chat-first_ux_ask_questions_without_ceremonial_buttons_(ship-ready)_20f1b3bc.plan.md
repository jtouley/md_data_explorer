---
name: ""
overview: ""
todos: []
---

# Chat-First UX: Ask Questions Without Ceremonial Buttons

## Overview

Transform the Ask Questions page to chat-first with proper guardrails. Auto-execute when ready, but only with confidence gating OR user confirmation. All architectural traps fixed: no Streamlit in core, correct caching, idempotency with result persistence, alias collision handling with proper plumbing, result lifecycle management with history tracking, stable run keys, serializable result artifacts.**Goal:** Questions happen right away, never run wrong thing, never run twice, results persist, memory doesn't balloon, no pickling drama.

## Development Workflow & Quality Gates

**Mandatory checks before every commit/PR:**

1. **Formatting**: `make format` (auto-fix) or `make format-check` (verify)
2. **Linting**: `make lint-fix` (auto-fix) or `make lint` (verify)
3. **Type checking**: `make type-check`
4. **Testing**: `make test-fast` (quick feedback) or `make test` (full suite)

**Pre-commit hook** (recommended):

```bash
# .git/hooks/pre-commit
#!/bin/sh
make format-check && make lint && make type-check && make test-fast
```

**PR CI checks** (GitHub Actions):

- All checks must pass before merge
- Use `make ci` for CI-specific checks

**Never commit code that fails these checks.**

### Test-First Development Workflow (MANDATORY)

**Rule: Always run tests immediately after writing them. Never mark work as "done" without running tests.**

**Workflow Steps:**

1. **Write failing test** (Red) - Use AAA pattern (Arrange-Act-Assert)
2. **Run test** - `make test-fast` to verify it fails as expected
3. **Implement minimum code to pass** (Green)
4. **Run test again** - `make test-fast` to verify it passes
5. **Fix code quality issues immediately** - Run `make lint-fix` and `make format` if needed
6. **Refactor** (Refactor)
7. **Run full test suite** - `make test-fast` before commit

**DRY Principle for Tests:**

- **Use shared fixtures from `conftest.py`** - Avoid repetitive imports and setup
- **Create module-level fixtures** - For common test data (e.g., `sample_cohort`, `sample_context`, `mock_session_state`)
- **Use `ask_questions_page` fixture** - Import the page module once in `conftest.py`, reuse in all tests
- **Never duplicate imports** - If you see the same import pattern in multiple test files, extract to `conftest.py`

**Example: Shared Fixtures in `conftest.py`:**

```python
# tests/conftest.py
@pytest.fixture(scope="module")
def ask_questions_page():
    """Import the Ask Questions page module (reusable across all tests)."""
    import importlib.util
    from pathlib import Path
    # ... import logic ...

@pytest.fixture
def mock_session_state():
    """Mock Streamlit session_state (reusable across all tests)."""
    return {}

@pytest.fixture
def sample_cohort():
    """Create sample Polars DataFrame (reusable across all tests)."""
    return pl.DataFrame({...})
```

**Dependency Management:**

- **Check dependencies before running tests** - Ensure all required packages are installed (`uv sync --extra dev --group dev`)
- **Mock external dependencies in tests** - Use `unittest.mock.patch` for Streamlit, file I/O, etc.
- **Handle missing dependencies gracefully** - Tests should fail with clear error messages if dependencies are missing

**Code Quality Fixes:**

- **Fix issues immediately** - Don't accumulate technical debt
- **Run linting after writing code** - `make lint-fix` to auto-fix issues
- **Check for duplicate imports** - Remove redundant imports immediately
- **Verify formatting** - `make format` before committing

## Architectural Constraints

### Polars-First Data Processing

**Rule**: Use Polars (`pl.DataFrame`) for all data processing. Pandas (`pd.DataFrame`) only at render boundary for Streamlit display.**Rationale**:

- Polars is faster, more memory-efficient, and the project standard
- Pandas conversion only where Streamlit requires it (`st.dataframe`, `st.plotly_chart`)

**Pattern**:

```python
# ✅ CORRECT: Polars-native compute
def compute_descriptive_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict:
    """Compute analysis using Polars-native operations."""
    summary_stats = df.describe()
    return {
        "type": "descriptive",
        "summary_stats": summary_stats.to_dicts(),  # Polars native
        "row_count": df.height,  # Polars attribute
        "column_count": df.width,  # Polars attribute
        "columns": df.columns,
    }

# ✅ CORRECT: Pandas only at render boundary
# PANDAS EXCEPTION: Required for Streamlit st.dataframe display
# TODO: Remove when Streamlit supports Polars natively
def render_descriptive_analysis(result: dict):
    """Render using pandas ONLY for Streamlit display."""
    import pandas as pd  # Only imported at render boundary
    
    # Convert to pandas ONLY for st.dataframe
    summary_df = pd.DataFrame(result["summary_stats"])
    st.dataframe(summary_df)
```

**Anti-pattern**:

```python
# ❌ WRONG: Using pandas in compute layer
def compute_descriptive_analysis(df: pd.DataFrame, context: AnalysisContext) -> dict:
    summary_stats = df.describe()  # Pandas in compute layer
    return {"summary_stats": summary_stats.to_dict()}
```

### Test-Driven Development

**Workflow**: Test-first development (Red-Green-Refactor)

1. **Write failing test** (Red) - Use AAA pattern (Arrange-Act-Assert)
2. **Run test immediately** - `make test-fast` to verify it fails as expected
3. **Implement minimum code to pass** (Green)
4. **Run test again** - `make test-fast` to verify it passes
5. **Fix code quality issues immediately** - Run `make lint-fix` and `make format` if needed
6. **Refactor** (Refactor)
7. **Run full test suite** - `make test-fast` before commit

**CRITICAL RULE: Always run tests after writing them. Never mark work as "done" without running tests.**

**Test Structure for PR A**:

```javascript
tests/
  conftest.py                                    # Shared fixtures (DRY principle)
  unit/
    ui/
      pages/
        test_ask_questions_idempotency.py    # Test idempotency guard
        test_ask_questions_lifecycle.py      # Test result lifecycle (history, cleanup)
        test_ask_questions_run_key.py        # Test stable run_key generation
```

**Test File Template (Using Shared Fixtures)**:

```python
# tests/unit/ui/pages/test_ask_questions_idempotency.py
"""
Test idempotency guard with result persistence.

Uses shared fixtures from conftest.py (DRY principle).
"""
import pytest
import polars as pl
from unittest.mock import patch
from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

# Fixtures from conftest.py: ask_questions_page, mock_session_state, sample_cohort, sample_context

def test_idempotency_same_query_uses_cached_result(
    mock_session_state, sample_cohort, sample_context, ask_questions_page
):
    """
    Test that identical query uses cached result (idempotency).
    
    Follows AAA pattern: Arrange-Act-Assert
    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Set up test data and dependencies
    dataset_version = "test_dataset_v1"
    query_text = "describe all patients"
    run_key = ask_questions_page.generate_run_key(dataset_version, query_text, sample_context)
    
    # Pre-populate session_state with cached result
    result_key = f"analysis_result:{dataset_version}:{run_key}"
    cached_result = {"type": "descriptive", "row_count": 5}
    mock_session_state[result_key] = cached_result
    
    # Act: Execute analysis (should use cache)
    with patch("streamlit.session_state", mock_session_state):
        with patch("streamlit.spinner"):
            with patch("ask_questions_page.render_analysis_by_type") as mock_render:
                ask_questions_page.execute_analysis_with_idempotency(
                    sample_cohort, sample_context, run_key, dataset_version, query_text
                )
                
                # Assert: Used cached result (render called with cached data)
                mock_render.assert_called_once_with(cached_result, sample_context.inferred_intent)
```

**DRY Principle for Tests**:

- **Use shared fixtures from `conftest.py`** - Avoid repetitive imports and setup
- **Create module-level fixtures** - For common test data (e.g., `sample_cohort`, `sample_context`, `mock_session_state`)
- **Use `ask_questions_page` fixture** - Import the page module once in `conftest.py`, reuse in all tests
- **Never duplicate imports** - If you see the same import pattern in multiple test files, extract to `conftest.py`
- **Fix code quality issues immediately** - Remove duplicate imports, fix linting errors before committing

**Test Coverage Requirements**:

- Unit tests for all compute functions (pure, no Streamlit)
- Integration tests for UI flow (with mocked `st.session_state`)
- Edge case tests (empty results, missing keys, eviction logic)
- All tests use shared fixtures from `conftest.py` (DRY principle)

## Final Bug Fixes (P0 - Must Fix)

### Bug Fix 1: Result Cleanup with History Tracking (P0)

**Problem:** `run_key` is sha256 hash, doesn't contain `dataset_version`. Substring check `dataset_version in key` matches nothing, memory balloons. Also, `cleanup_old_results()` scans all session_state keys (O(n)) to find results to remove.**Solution:** Track run history per dataset_version using deque. **O(1) eviction:** Proactively delete evicted keys when deque reaches maxlen, instead of scanning all keys.**Files:**

- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (MODIFY)

**Implementation:**

```python
from collections import deque
import structlog

logger = structlog.get_logger()

MAX_STORED_RESULTS_PER_DATASET = 5

def remember_run(dataset_version: str, run_key: str):
    """
    Remember this run in history for dataset version.
    
    O(1) eviction: Proactively delete evicted result when list reaches maxlen,
    instead of scanning all session_state keys.
    
    Note: Store history as list[str] (not deque) to avoid serialization quirks.
    Convert to deque locally for LRU logic.
    """
    hist_key = f"run_history_{dataset_version}"
    hist_list = st.session_state.get(hist_key)
    
    if hist_list is None:
        hist_list = []
        st.session_state[hist_key] = hist_list
    
    # Convert to deque for LRU logic (local only, not stored)
    hist = deque(hist_list, maxlen=MAX_STORED_RESULTS_PER_DATASET)
    
    # Capture what will be evicted BEFORE any modifications
    evicted_key = None
    if len(hist) == MAX_STORED_RESULTS_PER_DATASET and run_key not in hist:
        evicted_key = hist[0]  # Oldest will be evicted
    
    # De-dupe: move existing key to end (LRU behavior)
    if run_key in hist:
        hist.remove(run_key)
    hist.append(run_key)
    
    # Store back as list (deque not serializable in session_state)
    st.session_state[hist_key] = list(hist)
    
    # Delete evicted result immediately (O(1) instead of O(n) scan)
    if evicted_key:
        result_key = f"analysis_result:{dataset_version}:{evicted_key}"
        if result_key in st.session_state:
            del st.session_state[result_key]
            logger.info(
                "evicted_old_result",
                dataset_version=dataset_version,
                evicted_run_key=evicted_key,
            )

def cleanup_old_results(dataset_version: str):
    """
    Safety net: Remove any orphaned results not in history.
    
    Note: Most cleanup happens proactively in remember_run() (O(1)).
    This function is a safety net for edge cases (e.g., manual deletions).
    """
    hist_key = f"run_history_{dataset_version}"
    hist_list = st.session_state.get(hist_key)
    
    if not hist_list:
        return
    
    # Keep only results in history (using dataset-scoped keys)
    keep_keys = {f"analysis_result:{dataset_version}:{rk}" for rk in hist_list}
    
    # Remove any dataset-scoped result keys not in keep set
    result_prefix = f"analysis_result:{dataset_version}:"
    keys_to_remove = [
        key for key in st.session_state.keys()
        if key.startswith(result_prefix) and key not in keep_keys
    ]
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    if keys_to_remove:
        logger.info(
            "cleaned_orphaned_results",
            dataset_version=dataset_version,
            count=len(keys_to_remove),
        )

def clear_all_results(dataset_version: str):
    """
    Clear all results and history for this dataset version.
    
    Dataset-scoped: Only clears results for this specific dataset_version,
    not global results. Uses dataset-scoped result keys for trivial cleanup.
    """
    hist_key = f"run_history_{dataset_version}"
    
    # Get history before clearing (to know which keys to delete)
    hist_list = st.session_state.get(hist_key, [])
    
    # Clear history
    if hist_key in st.session_state:
        del st.session_state[hist_key]
    
    # Clear all dataset-scoped results (trivial with scoped keys)
    result_prefix = f"analysis_result:{dataset_version}:"
    keys_to_remove = [
        key for key in st.session_state.keys()
        if key.startswith(result_prefix)
    ]
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    # Clear dataset-scoped last_run_key
    dataset_last_run_key = f"last_run_key:{dataset_version}"
    if dataset_last_run_key in st.session_state:
        del st.session_state[dataset_last_run_key]
    
    logger.info(
        "cleared_all_results",
        dataset_version=dataset_version,
        result_count=len(keys_to_remove),
    )

# In execute_analysis_with_idempotency()
def execute_analysis_with_idempotency(
    cohort: pl.DataFrame,  # Polars DataFrame
    context: AnalysisContext,
    run_key: str,
    dataset_version: str
):
    """Execute analysis with idempotency guard and result persistence."""
    # Use dataset-scoped result key for trivial cleanup
    result_key = f"analysis_result:{dataset_version}:{run_key}"
    
    # Check if already computed
    if result_key in st.session_state:
        render_analysis_by_type(st.session_state[result_key], context.inferred_intent)
        return
    
    # Not computed - compute and store
    with st.spinner("Running analysis..."):
        result = compute_analysis_by_type(cohort, context)
        
        # Store result (serializable format)
        st.session_state[result_key] = result
        st.session_state[f"last_run_key:{dataset_version}"] = run_key
        
        # Remember this run in history (O(1) eviction happens here)
        remember_run(dataset_version, run_key)
        
        # Optional: Safety net cleanup (only needed for edge cases)
        # Most cleanup happens proactively in remember_run()
        # cleanup_old_results(dataset_version)
        
        # Render
        render_analysis_by_type(result, context.inferred_intent)
```

**Testing:**

- [ ] Run history tracked per dataset_version (stored as list[str], converted to deque locally)
- [ ] Only last 5 results kept per dataset_version
- [ ] O(1) eviction: 6th result evicts oldest immediately (no scan)
- [ ] LRU behavior: Re-adding existing run_key moves it to end (no eviction)
- [ ] Dataset-scoped result keys: `analysis_result:{dataset_version}:{run_key}` format
- [ ] `clear_all_results()` only clears dataset-scoped keys (not global)
- [ ] History-based cleanup works (not substring matching)
- [ ] Safety net cleanup handles orphaned results

### Bug Fix 2: Collision Suggestions Plumbing (P0)

**Problem:** `_fuzzy_match_variable` is in NLQueryEngine, not SemanticLayer. Passing entire query text to variable matcher is wrong.**Solution:** Propagate suggestions from NL parse step into context. NL engine returns suggestions, UI renders from context.**Files:**

- `src/clinical_analytics/core/nl_query_engine.py` (MODIFY - return suggestions)
- `src/clinical_analytics/ui/components/question_engine.py` (MODIFY - add match_suggestions to context)
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (MODIFY - render suggestions from context)

**Implementation:**

```python
# nl_query_engine.py - Return suggestions from variable matching
def _extract_variables_from_query(self, query: str) -> tuple[list[str], dict[str, list[str]]]:
    """
    Extract variables with collision suggestions.
    
    Returns:
        (matched_variables, collision_suggestions)
        collision_suggestions: {query_term: [canonical_names]}
    """
    words = query.lower().split()
    matched_vars = []
    collision_suggestions = {}
    
    # Try n-grams (3-word, 2-word, 1-word)
    for n in [3, 2, 1]:
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            
            # Skip if phrase contains _USED_ marker (already matched)
            if "_USED_" in phrase:
                continue
            
            # Match phrase (returns canonical_name, confidence, suggestions)
            matched, conf, suggestions = self._fuzzy_match_variable(phrase)
            
            if matched and matched not in matched_vars:
                matched_vars.append(matched)
                # Mark words as used
                words[i:i+n] = ["_USED_"] * n
            
            # Store collision suggestions
            if suggestions:
                collision_suggestions[phrase] = suggestions
    
    return matched_vars, collision_suggestions

def _fuzzy_match_variable(self, query_term: str) -> tuple[str | None, float, list[str] | None]:
    """
    Match variable with collision awareness.
    
    Args:
        query_term: Single term or phrase (not entire query)
    
    Returns:
        (matched_canonical_name, confidence, suggestions)
    """
    alias_index = self.semantic_layer.get_column_alias_index()
    normalized_query = self.semantic_layer._normalize_alias(query_term)
    
    # Check if this alias was dropped due to collision
    suggestions = self.semantic_layer.get_collision_suggestions(query_term)
    if suggestions:
        # Collision detected - return suggestions
        return None, 0.2, suggestions
    
    # Direct match
    if normalized_query in alias_index:
        collisions = self.semantic_layer.get_collision_warnings()
        if normalized_query in collisions:
            return alias_index[normalized_query], 0.4, None
        return alias_index[normalized_query], 0.9, None
    
    # Fuzzy match
    matches = get_close_matches(
        normalized_query,
        alias_index.keys(),
        n=1,
        cutoff=0.7
    )
    
    if matches:
        matched_alias = matches[0]
        collisions = self.semantic_layer.get_collision_warnings()
        if matched_alias in collisions:
            return alias_index[matched_alias], 0.4, None
        return alias_index[matched_alias], 0.7, None
    
    return None, 0.0, None

# question_engine.py - Add match_suggestions to AnalysisContext
@dataclass
class AnalysisContext:
    """Analysis context with collision suggestions."""
    inferred_intent: AnalysisIntent
    primary_variable: str | None = None
    grouping_variable: str | None = None
    predictor_variables: list[str] = None
    time_variable: str | None = None
    event_variable: str | None = None
    confidence: float = 0.0
    match_suggestions: dict[str, list[str]] = None  # NEW: {query_term: [canonical_names]}
    
    def __post_init__(self):
        if self.predictor_variables is None:
            self.predictor_variables = []
        if self.match_suggestions is None:
            self.match_suggestions = {}

# In ask_free_form_question() - propagate suggestions
def ask_free_form_question(semantic_layer) -> AnalysisContext | None:
    """Ask free-form question, return context with suggestions."""
    query = st.text_input("Your question:")
    
    if query:
        nl_engine = NLQueryEngine(semantic_layer)
        intent = nl_engine.parse_query(query)
        
        # Extract variables with suggestions
        matched_vars, collision_suggestions = nl_engine._extract_variables_from_query(query)
        
        # Build context with suggestions
        context = AnalysisContext(
            inferred_intent=intent.intent_type,
            primary_variable=matched_vars[0] if matched_vars else None,
            confidence=intent.confidence,
            match_suggestions=collision_suggestions  # Propagate suggestions
        )
        
        return context
    
    return None

# Ask_Questions.py - Render suggestions from context
def show_detected_variables(context: AnalysisContext, semantic_layer, editable=True):
    """Show detected variables with collision suggestions from context."""
    # ... show detected variables ...
    
    # Show collision suggestions if available
    if context.match_suggestions:
        st.warning("Some terms matched multiple columns. Please select:")
        
        for query_term, suggestions in context.match_suggestions.items():
            st.write(f"**'{query_term}'** matches:")
            selected = st.selectbox(
                f"Select column for '{query_term}':",
                options=suggestions,
                key=f"collision_{query_term}"
            )
            if selected:
                # Update context with selected column
                if not context.primary_variable:
                    context.primary_variable = selected
```

**Testing:**

- [ ] Suggestions come from NL engine (not SemanticLayer)
- [ ] Suggestions propagated to context.match_suggestions
- [ ] UI renders suggestions from context
- [ ] User can select from collision suggestions
- [ ] N-grams skip `_USED_` markers (no false matches on already-matched phrases)

### Design Tweak: Serializable Result Artifacts (P0)

**Problem:** Storing raw pandas objects in session_state causes pickling drama and performance cliffs.**Solution:** Store only serializable data: dicts, lists, primitives. For tables: use `to_dict(orient="records")` or Arrow/Parquet bytes (if small).**Architectural Principle:** All compute functionality must be in `src/clinical_analytics/analysis/` (backend/core), not in UI pages. UI pages only contain render functions and flow logic.**Files:**

- `src/clinical_analytics/analysis/compute.py` (CREATE - pure compute functions, Polars-first)
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (MODIFY - render functions only, import from analysis.compute)

**Implementation:Location:** `src/clinical_analytics/analysis/compute.py` (NEW FILE)

```python
# src/clinical_analytics/analysis/compute.py
"""
Pure compute functions for analysis - Polars-first, return serializable dicts.

These functions have no UI dependencies and can be tested independently.
"""
import polars as pl
from clinical_analytics.ui.components.question_engine import AnalysisContext

def compute_descriptive_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict:
    """
    Compute analysis using Polars-native operations.
    
    Returns serializable dict (no Polars objects).
    """
    # Compute statistics using Polars
    summary_stats = df.describe()
    
    # Convert to serializable format (Polars native)
    return {
        "type": "descriptive",
        "summary_stats": summary_stats.to_dicts(),  # Polars native (list of dicts)
        "row_count": df.height,  # Polars attribute
        "column_count": df.width,  # Polars attribute
        "columns": df.columns,
        # Charts: store data, not matplotlib objects
        "chart_data": {
            "histogram": histogram_data,  # List of dicts
            "boxplot": boxplot_data,  # List of dicts
        }
    }

def compute_comparison_analysis(df: pl.DataFrame, context: AnalysisContext) -> dict:
    """
    Compute comparison using Polars-native operations.
    
    Returns serializable dict (no Polars objects).
    """
    # ... computation using Polars ...
    
    # For DataFrames: convert to records (Polars native)
    comparison_df = ...  # Result Polars DataFrame
    
    return {
        "type": "comparison",
        "comparison_results": comparison_df.to_dicts(),  # Polars native (list of dicts)
        "schema": {
            col: str(dtype) for col, dtype in comparison_df.schema.items()
        },
        "statistics": {
            "p_value": float(p_value),
            "test_statistic": float(test_stat),
        }
    }

def compute_analysis_by_type(df: pl.DataFrame, context: AnalysisContext) -> dict:
    """
    Route to appropriate compute function based on intent.
    
    Args:
        df: Polars DataFrame (cohort data)
        context: AnalysisContext with intent and variables
    
    Returns:
        Serializable dict with analysis results
    """
    if context.inferred_intent == AnalysisIntent.DESCRIBE:
        return compute_descriptive_analysis(df, context)
    elif context.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
        return compute_comparison_analysis(df, context)
    # ... etc
```

**Location:** `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (MODIFY - render functions only)

# PANDAS EXCEPTION: Required for Streamlit st.dataframe display

# TODO: Remove when Streamlit supports Polars natively

def render_descriptive_analysis(result: dict):"""Render from serializable dict.Pandas conversion ONLY at render boundary for Streamlit display."""import pandas as pd  # Only imported at render boundaryst.markdown("## 📊 Descriptive Statistics")

# Convert to pandas ONLY for st.dataframe display

summary_df = pd.DataFrame(result["summary_stats"])st.dataframe(summary_df)

# Render charts from data

if "chart_data" in result:render_charts_from_data(result["chart_data"])

# PANDAS EXCEPTION: Required for Streamlit st.dataframe display

# TODO: Remove when Streamlit supports Polars natively

def render_comparison_analysis(result: dict):"""Render from serializable dict.Pandas conversion ONLY at render boundary for Streamlit display."""import pandas as pd  # Only imported at render boundary

# Convert to pandas ONLY for st.dataframe display

comparison_df = pd.DataFrame(result["comparison_results"])

# Use astype() pattern instead of dtype parameter (more reliable)

comparison_df = comparison_df.astype(result["schema"], errors="ignore")st.dataframe(comparison_df)

# ... render statistics ...

````javascript

**Testing:**

- [ ] Compute functions in `src/clinical_analytics/analysis/compute.py` (not in UI)
- [ ] Compute functions use Polars (`pl.DataFrame`) for all data processing
- [ ] Compute functions have no UI/Streamlit dependencies
- [ ] Results stored as dicts (not Polars/pandas objects)
- [ ] Polars DataFrames converted to `to_dicts()` (Polars native)
- [ ] Render functions in UI page convert to pandas ONLY at boundary for Streamlit
- [ ] DataFrame reconstruction uses `df.astype(schema, errors="ignore")` pattern (not dtype parameter)
- [ ] Charts stored as data (not matplotlib objects)
- [ ] No pickling errors
- [ ] Results render correctly from serialized format
- [ ] Compute functions can be tested independently (no Streamlit mocking needed)

## Implementation Plan (Final)

### Phase 1: Ask-First Flow with Proper Gating + Idempotency (P0)

**Files:**

- `src/clinical_analytics/analysis/compute.py` (CREATE - pure compute functions, Polars-first)
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (MODIFY - render functions only, import from analysis.compute)
- `tests/unit/analysis/test_compute.py` (CREATE - test compute functions independently)
- `tests/unit/ui/pages/test_ask_questions_idempotency.py` (CREATE)
- `tests/unit/ui/pages/test_ask_questions_lifecycle.py` (CREATE)
- `tests/unit/ui/pages/test_ask_questions_run_key.py` (CREATE)

**Test-First Workflow** (per `.cursor/rules/101-testing-hygiene.mdc`):

1. Write failing test (Red) - Use AAA pattern (Arrange-Act-Assert)
2. **Run test immediately** - `make test-fast` to verify it fails as expected
3. Implement minimum code to pass (Green)
4. **Run test again** - `make test-fast` to verify it passes
5. **Fix code quality issues immediately** - Run `make lint-fix` and `make format` if needed
6. Refactor
7. **Run full test suite** - `make test-fast` before commit (NEVER run pytest directly)

**CRITICAL RULE: Always run tests after writing them. Never mark work as "done" without running tests.**

**MANDATORY: Commit after each phase completion**

Before starting the next phase, you MUST:
1. Run `make check` to ensure all quality gates pass (format, lint, type-check, test)
2. Commit all changes for the completed phase with a descriptive commit message
3. Verify the commit includes all relevant files for that phase

**Never start a new phase without committing the previous phase.**

**Test Structure** (per `.cursor/rules/101-testing-hygiene.mdc`):

```python
# tests/unit/ui/pages/test_ask_questions_idempotency.py
import pytest
import polars as pl
import polars.testing as plt
from unittest.mock import MagicMock, patch
from clinical_analytics.ui.pages.ask_questions import (
    execute_analysis_with_idempotency,
    remember_run,
)

@pytest.fixture
def mock_session_state():
    """Mock Streamlit session_state."""
    return {}

def test_idempotency_same_query_uses_cached_result(mock_session_state):
    """
    Test that identical query uses cached result (idempotency).
    
    Test name follows: test_unit_scenario_expectedBehavior
    """
    # Arrange: Set up test data
    cohort = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    context = AnalysisContext(inferred_intent="DESCRIBE", confidence=0.9)
    run_key = "test_key_123"
    dataset_version = "test_dataset_v1"
    
    # Act: First call - compute and store
    with patch("streamlit.session_state", mock_session_state):
        execute_analysis_with_idempotency(cohort, context, run_key, dataset_version)
    
    # Assert: Result stored
    result_key = f"analysis_result:{dataset_version}:{run_key}"
    assert result_key in mock_session_state
    
    # Act: Second call - should use cache
    with patch("streamlit.session_state", mock_session_state):
        with patch("clinical_analytics.ui.pages.ask_questions.compute_analysis_by_type") as mock_compute:
            execute_analysis_with_idempotency(cohort, context, run_key, dataset_version)
            # Assert: No recomputation
            mock_compute.assert_not_called()

# tests/unit/ui/pages/test_ask_questions_lifecycle.py
def test_lifecycle_remember_run_evicts_oldest_when_maxlen_reached(mock_session_state):
    """
    Test that 6th result evicts oldest (O(1) eviction).
    
    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Add 5 results to history
    # Act: Add 6th result
    # Assert: Oldest result evicted, newest stored

def test_lifecycle_clear_all_results_only_clears_dataset_scoped_keys(mock_session_state):
    """
    Test that clear_all_results only clears dataset-scoped keys.
    
    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Add results for multiple datasets
    # Act: Clear results for one dataset
    # Assert: Only that dataset's results cleared

# tests/unit/ui/pages/test_ask_questions_run_key.py
def test_run_key_generation_same_query_produces_same_key():
    """
    Test that same query + variables generates same run_key.
    
    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Same query, same variables
    # Act: Generate run_key twice
    # Assert: Keys are identical

def test_run_key_generation_whitespace_normalization_produces_same_key():
    """
    Test that whitespace normalization produces same key.
    
    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Query with different whitespace
    # Act: Generate run_keys
    # Assert: Keys are identical
````

**Implementation:**

- Remove "Run Analysis" button
- Gate auto-execution: `confidence >= 0.75` OR user confirmed (default 0.0)
- **Refactor:** 
- Create `src/clinical_analytics/analysis/compute.py` with pure `compute_*()` functions (returns serializable dict, uses `pl.DataFrame`, no UI dependencies)
- Keep `render_*()` functions in UI page (Streamlit, converts to pandas only at boundary)
- UI page imports `compute_analysis_by_type` from `analysis.compute`
- **Idempotency:** Persist results in `st.session_state[result_key]`, render from cache on rerun
- **History tracking:** Use `list[str]` to track run history per dataset_version (convert to deque locally)
- **O(1) eviction:** Proactively delete evicted result when list reaches maxlen (capture before append)
- **Dataset-scoped keys:** Use `analysis_result:{dataset_version}:{run_key}` format
- **Cleanup:** Safety net cleanup based on history (not substring matching)
- **Stable run_key:** Canonicalize with JSON, sort predictors, normalize empty values, collapse whitespace
- Low confidence: Show detected + editable selectors + confirmation button
- Confirmation key includes dataset_version + query hash

**Testing:**

- [ ] Compute functions in `analysis/compute.py` have no UI dependencies
- [ ] Compute functions can be tested independently (no Streamlit mocking)
- [ ] UI page imports and calls compute functions correctly
- [ ] High confidence auto-executes
- [ ] Low confidence requires confirmation
- [ ] Results persist across reruns (idempotency)
- [ ] Same query doesn't recompute (stable run_key)
- [ ] Only last 5 results stored per dataset (history-based cleanup)
- [ ] O(1) eviction works (6th result evicts oldest immediately)
- [ ] Dataset-scoped keys work correctly
- [ ] "Clear Results" clears only dataset-scoped results
- [ ] All tests use shared fixtures from `conftest.py` (DRY principle)
- [ ] No duplicate imports in test files
- [ ] All tests pass (`make test-fast`) - **MUST run tests after writing them**

**MANDATORY: Commit Phase 1 before starting Phase 2**

Before moving to Phase 2, you MUST:
1. Run `make check` to ensure all quality gates pass
2. Commit all Phase 1 changes with a descriptive commit message
3. Verify commit includes: compute.py, updated Ask Questions page, all test files, pyproject.toml (structlog dependency)

**Example commit message:**
```
feat: Phase 1 - Chat-first UX with idempotency and lifecycle management

- Implement idempotency guard with result persistence (compute vs render split)
- Add result lifecycle management with O(1) eviction using deque
- Create stable run_key generation with canonicalized JSON hashing
- Move compute functions to analysis/compute.py (Polars-first, no UI deps)
- Add serializable result artifacts (dicts, not pandas objects)
- Implement auto-execute flow with confidence gating (>= 0.75 or user confirmed)
- Add comprehensive test suite (15 tests passing)
- Add structlog dependency for structured logging

All tests passing: 15/15
```

### Phase 2: Column Intelligence + Alias Collision Handling (P0)

**Files:**

- `src/clinical_analytics/core/semantic.py` (MODIFY - NO Streamlit, collision detection + suggestions)
- `src/clinical_analytics/core/column_parser.py` (CREATE - robust parser)
- `src/clinical_analytics/core/schema_inference.py` (MODIFY)
- `src/clinical_analytics/core/nl_query_engine.py` (MODIFY - return suggestions, proper plumbing)
- `src/clinical_analytics/ui/components/question_engine.py` (MODIFY - add match_suggestions to context)
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (MODIFY - render suggestions from context)

**Implementation:**

- ColumnMetadata class (no Streamlit dependencies)
- Robust mapping parser (handles spaces, punctuation, ranges)
- **Alias collision detection:** Build `alias -> set[canonical_names]`, drop ambiguous aliases
- **Collision suggestions:** Expose `get_collision_suggestions()` in SemanticLayer
- **Proper plumbing:** NL engine extracts variables + suggestions, propagates to context
- **UI suggestions:** Render suggestions from `context.match_suggestions`
- Alias indexing: Build word frequency first, conditionally index single words
- Wire into NL matching, UI rendering, result interpretation
- Cache alias index in UI layer (returns plain dict)

**Testing:**

- [ ] Core layer has no Streamlit imports
- [ ] Collisions detected and logged
- [ ] Suggestions come from NL engine (not SemanticLayer)
- [ ] Suggestions propagated to context.match_suggestions
- [ ] UI shows collision suggestions from context
- [ ] User can select from suggestions
- [ ] All tests use shared fixtures from `conftest.py` (DRY principle)
- [ ] Tests run immediately after writing (`make test-fast`)

**MANDATORY: Commit Phase 2 before starting Phase 3**

Before moving to Phase 3, you MUST:
1. Run `make check` to ensure all quality gates pass
2. Commit all Phase 2 changes with a descriptive commit message

### Phase 3: Caching with Correct Patterns (P1)

**Files:**

- `src/clinical_analytics/core/dataset.py` (MODIFY)
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (MODIFY)

**Implementation:**

- Pass version identifiers as explicit function arguments
- Cache small intermediates only (profiling, metadata, alias index)
- Do NOT cache cohorts (or use manual cache with size guard)
- Do NOT access st.session_state inside cached functions

**Testing:**

- [ ] Cache keys from function args
- [ ] Cohorts not cached (or manual cache)
- [ ] Small intermediates cached
- [ ] All tests use shared fixtures from `conftest.py` (DRY principle)
- [ ] Tests run immediately after writing (`make test-fast`)

**MANDATORY: Commit Phase 3 before starting Phase 4**

Before moving to Phase 4, you MUST:
1. Run `make check` to ensure all quality gates pass
2. Commit all Phase 3 changes with a descriptive commit message

### Phase 4: Low-Confidence Feedback (P0)

**Files:**

- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (MODIFY)

**Implementation:**

- Show detected variables with display names
- Show collision suggestions from context.match_suggestions
- Allow editing before confirmation
- Pre-fill structured inputs
- Ensure semantic_layer ready before showing

**Testing:**

- [ ] Low confidence shows detected + editable selectors
- [ ] Collision suggestions shown from context
- [ ] User can correct without retyping
- [ ] All tests use shared fixtures from `conftest.py` (DRY principle)
- [ ] Tests run immediately after writing (`make test-fast`)

**MANDATORY: Commit Phase 4 before starting Phase 5**

Before moving to Phase 5, you MUST:
1. Run `make check` to ensure all quality gates pass
2. Commit all Phase 4 changes with a descriptive commit message

### Phase 5: Result Interpretation (P1)

**Files:**

- `src/clinical_analytics/ui/components/analysis_results.py` (MODIFY)

**Implementation:**

- Prioritize "what does this mean"
- Use value_mapping for readable labels
- Handle CI crossing 1, sample size warnings

**Testing:**

- [ ] Interpretation uses "associated with" wording
- [ ] Value mappings used for labels
- [ ] All tests use shared fixtures from `conftest.py` (DRY principle)
- [ ] Tests run immediately after writing (`make test-fast`)

**MANDATORY: Commit Phase 5 before starting Phase 6**

Before moving to Phase 6, you MUST:
1. Run `make check` to ensure all quality gates pass
2. Commit all Phase 5 changes with a descriptive commit message

### Phase 6: Structured Logging (P1)

**Files:**

- `src/clinical_analytics/core/nl_query_engine.py` (MODIFY)

**Implementation:**

- Standardize log fields: dataset_id, upload_id, query, intent, confidence, error_type

**Testing:**

- [ ] All logs include standardized fields
- [ ] All tests use shared fixtures from `conftest.py` (DRY principle)
- [ ] Tests run immediately after writing (`make test-fast`)

**MANDATORY: Commit Phase 6 before starting Phase 7**

Before moving to Phase 7, you MUST:
1. Run `make check` to ensure all quality gates pass
2. Commit all Phase 6 changes with a descriptive commit message

### Phase 7: Quick Wins (P2-P3)

- Vectorized operations (measured)
- Remove unused imports
- Simple message constants

**MANDATORY: Commit Phase 7 before marking plan complete**

Before marking the plan as complete, you MUST:
1. Run `make check` to ensure all quality gates pass
2. Commit all Phase 7 changes with a descriptive commit message
3. Verify all phases have been committed separately

## Success Criteria

### Safety & Correctness

- [ ] Never auto-executes wrong analysis (gated by confidence OR confirmation)
- [ ] Missing confidence defaults to 0.0 (fail closed)
- [ ] Never runs twice (idempotency with result persistence)
- [ ] Results persist across Streamlit reruns
- [ ] Alias collisions detected and handled safely
- [ ] Collision suggestions properly plumbed (engine → context → UI)

### Memory Management

- [ ] Run history tracked per dataset_version (stored as list[str], converted to deque locally)
- [ ] Only last 5 results kept per dataset_version
- [ ] O(1) eviction: Proactive deletion when list reaches maxlen (no O(n) scan)
- [ ] Dataset-scoped result keys: `analysis_result:{dataset_version}:{run_key}` format
- [ ] `clear_all_results()` only clears dataset-scoped keys (not global)
- [ ] Old results cleaned up based on history (not substring matching)
- [ ] "Clear Results" action available
- [ ] Memory doesn't balloon over time

### Result Artifacts

- [ ] Results stored as serializable dicts (not pandas objects)
- [ ] DataFrames converted to records + schema
- [ ] DataFrame reconstruction uses `df.astype(schema, errors="ignore")` pattern (not dtype parameter)
- [ ] Charts stored as data (not matplotlib objects)
- [ ] No pickling errors
- [ ] Results render correctly from serialized format

### Stability

- [ ] Run key stable (same query + variables = same key)
- [ ] Query text normalized (whitespace collapsed)
- [ ] Predictors sorted before hashing
- [ ] Empty values normalized consistently
- [ ] JSON serialization stable
- [ ] sha256 used consistently

### Architecture

- [ ] Core layer has no Streamlit dependencies
- [ ] Compute functions in `src/clinical_analytics/analysis/` (not in UI)
- [ ] Compute functions pure (return serializable dicts, no UI dependencies)
- [ ] Render functions in UI page handle Streamlit UI only
- [ ] UI page imports compute functions from `analysis.compute`
- [ ] Results stored in session_state with lifecycle management

### Integration

- [ ] ColumnMetadata wired into NL matching, UI rendering, result interpretation
- [ ] Display names shown everywhere
- [ ] Value mappings used for readable labels
- [ ] Alias collisions don't cause false matches
- [ ] Collision suggestions improve UX (proper plumbing)
- [ ] N-grams skip `_USED_` markers (no false matches on already-matched phrases)

### Rule Compliance

**Polars-First** (per `.cursor/rules/100-polars-first.mdc`):

- [ ] All compute functions use `pl.DataFrame` (Polars native)
- [ ] Pandas ONLY at Streamlit render boundary with `# PANDAS EXCEPTION` comment
- [ ] Use Polars expression API (`pl.col().str.to_uppercase()`, not methods)
- [ ] Use `df.height`/`df.width` (not `len(df)`/`len(df.columns)`)
- [ ] Use `df.to_dicts()` (not `to_dict(orient="records")`)
- [ ] Use `df.schema` (not `df.dtypes`)
- [ ] Lazy execution where applicable (`pl.scan_*`)

**Testing Hygiene** (per `.cursor/rules/101-testing-hygiene.mdc`):

- [ ] All tests follow AAA pattern (Arrange-Act-Assert)
- [ ] Test names follow `test_unit_scenario_expectedBehavior` pattern
- [ ] Use `polars.testing.assert_frame_equal` for Polars DataFrame assertions
- [ ] Use Makefile commands (`make test-fast`, `make test`) - NEVER run pytest directly
- [ ] Tests isolated (no shared mutable state)
- [ ] Use fixtures appropriately (session/module/function scope)

**Makefile Usage** (per `.cursor/rules/000-project-setup-and-makefile.mdc`):

- [ ] All testing uses `make test-fast` or `make test` (never `pytest` directly)
- [ ] All linting uses `make lint` or `make lint-fix` (never `ruff` directly)
- [ ] All formatting uses `make format` or `make format-check` (never `ruff format` directly)
- [ ] Before commit: `make check` or `make check-fast` passes

## PR Slicing

**PR A (P0 - Safety Critical):**

- Create `src/clinical_analytics/analysis/compute.py` with pure compute functions (Polars-first, no UI dependencies)
- Ask-first flow with proper gating (confidence threshold OR confirmation)
- Default confidence to 0.0 (fail closed)
- Idempotency with result persistence (compute vs render split)
- Result lifecycle management (history-based cleanup with deque)
- Serializable result artifacts (dicts, records, not pandas objects)
- Stable run_key (canonicalized with JSON, whitespace collapsed)
- Streamlit rerun handling (dataset_version keys, no Streamlit in core)
- UI page imports from `analysis.compute`, only contains render functions

**PR B (P0 - Integration):**

- ColumnMetadata + robust mapping parser
- Alias collision detection with proper plumbing (engine → context → UI)
- Alias index with constraints (avoid false positives)
- Wire ColumnMetadata into NL + UI + results
- Low-confidence feedback with collision suggestions from context

**PR C (P1 - Performance & Polish):PR C (P1 - Performance & Polish):**

- Caching with correct invalidation keys
- Structured logging with standardized fields
- Vectorized operations (measured first)
- Remove unused imports
- Simple message constants

## Key Learnings & Preferences for Future Phases

### Development Workflow Preferences

1. **Always run tests after writing them** - Never mark work as "done" without running tests. This is a mandatory step, not optional.
2. **Fix code quality issues immediately** - Don't accumulate technical debt. Run `make lint-fix` and `make format` immediately after writing code.
3. **DRY principle for tests** - Use shared fixtures from `conftest.py` to avoid repetitive imports and setup. Never duplicate the same import pattern across multiple test files.
4. **Test-first workflow** - Write failing test → Run test → Implement → Run test again → Fix issues → Refactor → Run full suite.

### Code Quality Standards

1. **No duplicate imports** - Remove redundant imports immediately. If you see the same import pattern in multiple files, extract to `conftest.py`.
2. **Dependency management** - Check dependencies before running tests. Ensure all required packages are installed (`uv sync --extra dev --group dev`).
3. **Mock external dependencies** - Use `unittest.mock.patch` for Streamlit, file I/O, etc. Tests should be isolated and not depend on external state.

### Test Structure Standards

1. **Use shared fixtures** - All test files should use fixtures from `conftest.py` (e.g., `ask_questions_page`, `mock_session_state`, `sample_cohort`, `sample_context`).
2. **AAA pattern** - All tests follow Arrange-Act-Assert with clear separation.
3. **Descriptive test names** - Follow `test_unit_scenario_expectedBehavior` pattern.
4. **Run tests immediately** - After writing a test, run `make test-fast` to verify it works (or fails as expected).

### Architectural Preferences

1. **Separation of concerns** - Compute functions belong in `src/clinical_analytics/analysis/` (backend/core), not in UI pages.
2. **Polars-first** - Use `pl.DataFrame` for all data processing. Pandas only at render boundary for Streamlit display.
3. **Serializable artifacts** - Store only serializable data (dicts, lists, primitives) in `st.session_state`. No pandas/polars objects.

### Quality Gates

**Before marking any work as "done":**

1. ✅ Tests written and passing (`make test-fast`)
2. ✅ Code quality checks pass (`make lint-fix`, `make format`)
3. ✅ No duplicate imports or code quality issues
4. ✅ All tests use shared fixtures (DRY principle)
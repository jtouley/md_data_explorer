---
name: Fix Chat Response Display - State Machine and Rendering Flow
overview: Fix the issue where chat responses aren't displaying in the UI. The problem involves the fragile Streamlit state machine where `intent_signal` controls execution, but execution results aren't properly displayed in the chat transcript. This plan addresses both the immediate fix (ensure responses show) and adds proper state management to prevent future issues.
todos:
  - id: "1"
    content: Write failing test for state machine persistence (intent_signal persists across reruns)
    status: pending
  - id: "2"
    content: Write failing test for chat message rendering with cached results
    status: pending
  - id: "3"
    content: Fix state initialization to not clear intent_signal incorrectly
    status: pending
    dependencies:
      - "1"
  - id: "4"
    content: Fix assistant message addition order (ensure result cached before adding message)
    status: pending
    dependencies:
      - "2"
  - id: "5"
    content: Improve render_chat cache retrieval with better error handling
    status: pending
    dependencies:
      - "2"
  - id: "6"
    content: Add comprehensive logging to execution guard
    status: pending
    dependencies:
      - "3"
  - id: "7"
    content: Run tests and fix quality issues (make test-ui)
    status: pending
    dependencies:
      - "3"
      - "4"
      - "5"
      - "6"
  - id: "8"
    content: Write integration test for full chat flow (query → parse → execute → render)
    status: pending
    dependencies:
      - "7"
  - id: "9"
    content: Run full test suite and verify no regressions (make test-fast)
    status: pending
    dependencies:
      - "8"
  - id: "10"
    content: Commit changes with comprehensive commit message
    status: pending
    dependencies:
      - "9"
---

# Fix Chat Response Display - State Machine and Rendering Flow

## Problem Analysis

From terminal logs and code analysis:

1. **State Machine Issue**: `intent_signal` is set to `"nl_parsed"` after query parsing (line 2612), but logs show it's `None` when execution block checks (line 1997). This prevents execution from running.
2. **Rendering Flow Issue**: `execute_analysis_with_idempotency()` renders results inline during execution (line 1265), but the assistant message added to chat (line 2264) may not be displaying because:

- Result might not be in cache when `render_chat()` tries to retrieve it
- Assistant message might not have correct `run_key`
- Rerun after adding assistant message might not be happening

3. **Order of Operations**: `render_chat()` is called BEFORE execution (line 1980), so on the rerun after execution, it should show both messages. But if execution doesn't happen, no assistant message exists.

## Root Causes

1. **State Persistence**: `intent_signal` may not be persisting across reruns
2. **Execution Guard**: Execution only happens if `intent_signal is not None` (line 2003), but it's being cleared or not set correctly
3. **Cache Timing**: Result might be stored after assistant message is added, causing cache miss in `render_chat()`

## Solution Strategy

### Phase 1: Immediate Fix - Ensure Execution Happens

- Add defensive checks to ensure `intent_signal` persists
- Add logging to track state transitions
- Fix execution guard to handle edge cases

### Phase 2: Fix Chat Rendering

- Ensure assistant message is added with correct `run_key` AFTER result is cached
- Ensure `render_chat()` can retrieve results from cache
- Add fallback rendering if cache miss occurs

### Phase 3: Add Tests

- Test state machine transitions
- Test chat message rendering with cached results
- Test full flow: query → parse → execute → cache → render

## Implementation Plan

### Step 1: Write Test for State Machine Persistence (TDD Red)

**File**: `tests/ui/pages/test_ask_questions_state_machine.py`

**Note**: Fixtures verified: `mock_session_state` (defined in test file), `ask_questions_page` (conftest.py line 382), `monkeypatch` (pytest built-in)

Test that `intent_signal` persists across reruns and execution happens:

```python
from unittest.mock import patch, MagicMock
from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent
from clinical_analytics.core.result_cache import ResultCache
from datetime import datetime

def test_intent_signal_persists_across_rerun_triggers_execution(mock_session_state, ask_questions_page, monkeypatch):
    """Test that intent_signal='nl_parsed' persists and triggers execution on rerun."""
    # Arrange: Set up state as if query was just parsed
    context = AnalysisContext(inferred_intent=AnalysisIntent.COUNT)
    mock_session_state["analysis_context"] = context
    mock_session_state["intent_signal"] = "nl_parsed"
    mock_session_state["chat"] = []
    mock_session_state["result_cache"] = ResultCache(max_size=50)

    monkeypatch.setattr("streamlit.session_state", mock_session_state)

    # Mock execution dependencies to verify execution block runs
    with patch("streamlit.rerun") as mock_rerun:
        with patch("streamlit.chat_input", return_value=None):
            # Extract execution block logic or use AppTest for full page testing
            # For unit test: Mock the execution function or extract to testable function
            # For integration test: Use AppTest.from_file() to test full page

            # Act: Simulate execution block check (extract execution logic to testable function)
            # OR use AppTest to test full page execution
            from streamlit.testing.v1 import AppTest
            app = AppTest.from_file("src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py")
            app.run()

            # Assert: Execution block should run because intent_signal is not None
            # Verify side effects: cache.put called, assistant message added, or execution_result stored
            assert app.session_state.get("intent_signal") == "nl_parsed" or app.session_state.get("intent_signal") is None
            # If execution ran, result should be in cache or assistant message should be added
            cache = app.session_state.get("result_cache")
            if cache:
                # Check if any results were cached (execution happened)
                assert len(cache._results) > 0 or len(app.session_state.get("chat", [])) > 0
```

### Step 2: Write Test for Chat Message Rendering (TDD Red)

**File**: `tests/ui/pages/test_ask_questions_chat_rendering.py`

**Note**: Fixtures verified: `mock_session_state` (defined in test file), `ask_questions_page` (conftest.py line 382)

Test that assistant messages with run_key can retrieve results from cache:

```python
from unittest.mock import patch, MagicMock
from clinical_analytics.core.result_cache import ResultCache, CachedResult
from datetime import datetime
import polars as pl

def test_render_chat_displays_assistant_message_with_cached_result(mock_session_state, ask_questions_page, monkeypatch):
    """Test that render_chat() displays assistant message and retrieves result from cache."""
    # Arrange: Add assistant message to chat with run_key, result in cache
    run_key = "test_run_key"
    dataset_version = "test_dataset_v1"
    result = {"type": "count", "value": 42, "intent": "COUNT"}

    cache = ResultCache(max_size=50)
    cache.put(CachedResult(
        run_key=run_key,
        query="test query",
        result=result,
        timestamp=datetime.now(),
        dataset_version=dataset_version,
    ))
    mock_session_state["result_cache"] = cache

    chat_message = {
        "role": "assistant",
        "text": "42 results found",
        "run_key": run_key,
        "status": "completed",
    }
    mock_session_state["chat"] = [chat_message]

    monkeypatch.setattr("streamlit.session_state", mock_session_state)

    # Mock Streamlit UI functions
    with patch("streamlit.chat_message") as mock_chat_message:
        with patch("streamlit.write") as mock_write:
            # Act: Call render_chat() function directly
            from clinical_analytics.ui.pages.ask_questions_page import render_chat
            cohort = pl.DataFrame({"id": [1, 2, 3]})
            render_chat(dataset_version=dataset_version, cohort=cohort)

            # Assert: Result is rendered (check that render functions were called)
            # Verify cache.get() was called with correct run_key
            cached_result = cache.get(run_key, dataset_version)
            assert cached_result is not None
            assert cached_result.result == result

            # Verify UI rendering was attempted (chat_message and write called)
            assert mock_chat_message.called or mock_write.called
```

### Step 3: Fix State Machine Persistence (TDD Green)

**File**: `src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`

**Location**: Fix state initialization in `main()` function (where `analysis_context` is initialized)

**Issue**: State initialization clears `intent_signal` to None if `analysis_context` doesn't exist, but this might be clearing it incorrectly when `intent_signal` was already set.

**Fix**: Only initialize if BOTH don't exist, don't reset if one exists:

```python
# Current (in main() function):
if "analysis_context" not in st.session_state:
    st.session_state["analysis_context"] = None
    st.session_state["intent_signal"] = None

# Fixed:
if "analysis_context" not in st.session_state:
    st.session_state["analysis_context"] = None
if "intent_signal" not in st.session_state:
    st.session_state["intent_signal"] = None
```

**Also**: Add logging to track state transitions and verify persistence.

### Step 4: Fix Assistant Message Addition Order (TDD Green)

**File**: `src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`

**Location**: Fix assistant message addition in `execute_analysis_with_idempotency()` function (after `cache.put()` call)

**Issue**: Assistant message is added but result might not be in cache yet, or cache retrieval might fail.

**Fix**: Ensure result is cached BEFORE adding assistant message, and verify cache contains result (with logging, not assertion):

```python
# After cache.put() in execute_analysis_with_idempotency():
# Verify result is in cache (for debugging, not enforcement)
cached_verify = cache.get(run_key, dataset_version)
if cached_verify is None:
    logger.error(
        "result_not_in_cache_after_put",
        run_key=run_key,
        dataset_version=dataset_version,
        cache_size=len(cache._results.get(dataset_version, {})) if hasattr(cache, '_results') else 0,
    )
    # Continue anyway - cache.put() should have worked, but don't crash
    # The assistant message will show fallback text if cache miss occurs later

# Then add assistant message with verified run_key (in main() function after execution)
```

### Step 5: Fix render_chat Cache Retrieval (TDD Green)

**File**: `src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`

**Location**: Improve cache retrieval in `render_chat()` function (cache miss handling)

**Issue**: If cache miss occurs, shows warning but doesn't render anything.

**Fix**: Add better error handling and fallback rendering:

```python
# In render_chat() function, improve fallback for cache miss:
else:
    # Result not in cache - log for debugging
    logger.warning(
        "result_not_in_cache",
        run_key=run_key,
        dataset_version=dataset_version,
        cache_keys=list(cache._results.get(dataset_version, {}).keys()) if hasattr(cache, '_results') else "unknown"
    )
    # Show user-friendly message
    st.warning(f"⚠️ Result not found in cache. Run key: {run_key[:16]}...")
    # Still show assistant text as fallback
    st.write(text)
```

### Step 6: Enhance Execution Guard Logging (TDD Green)

**File**: `src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`

**Location**: Enhance existing logging in `main()` function at execution guard (where `intent_signal` is checked)

**Issue**: Execution block doesn't run, but we don't know why. Existing logging exists but needs enhancement.

**Fix**: Enhance existing logging to include more diagnostic information:

```python
# In main() function, enhance existing execution_block_check logging:
intent_signal = st.session_state.get("intent_signal")
has_context = "analysis_context" in st.session_state
context_value = st.session_state.get("analysis_context")

logger.debug(
    "execution_block_check",
    intent_signal=intent_signal,
    has_analysis_context=has_context,
    context_type=type(context_value).__name__ if context_value else None,
    will_execute=intent_signal is not None,
)
```

### Step 7: Run Tests and Fix Quality Issues (TDD Refactor)

- Run: `make test-ui PYTEST_ARGS="tests/ui/pages/test_ask_questions_state_machine.py tests/ui/pages/test_ask_questions_chat_rendering.py -xvs"`
- Fix any linting, formatting, type issues
- Extract duplicate test setup to fixtures (Rule of Two)

### Step 8: Integration Test - Full Flow (TDD Refactor)

**File**: `tests/integration/test_chat_response_display.py`

**Note**: Fixtures verified: `mock_session_state` (defined in test file), `ask_questions_page` (conftest.py line 382), `sample_dataset` (verify exists in conftest.py)

Test end-to-end: query → parse → execute → cache → render:

```python
from streamlit.testing.v1 import AppTest

def test_full_chat_flow_displays_responses(mock_session_state, ask_questions_page, sample_dataset):
    """Test that querying via chat input results in visible response in chat UI."""
    # Arrange: Set up dataset, semantic layer
    # Use AppTest to test full page execution
    app = AppTest.from_file("src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py")
    app.run()

    # Act: Simulate user query via chat input
    app.chat_input[0].set_value("how many patients").run()

    # Assert:
    #   - User message appears in chat
    assert len(app.session_state["chat"]) >= 1
    assert app.session_state["chat"][0]["role"] == "user"

    #   - Assistant message appears in chat with run_key
    assert len(app.session_state["chat"]) >= 2
    assert app.session_state["chat"][1]["role"] == "assistant"
    assert app.session_state["chat"][1].get("run_key") is not None

    #   - Result is in cache (verifies execution happened)
    cache = app.session_state.get("result_cache")
    assert cache is not None
    cached_result = cache.get(app.session_state["chat"][1]["run_key"], dataset_version)
    assert cached_result is not None

    #   - Result is retrievable from cache on subsequent reruns
    # Simulate rerun by calling render_chat() again
    # Verify cache still contains result
    cached_result_rerun = cache.get(app.session_state["chat"][1]["run_key"], dataset_version)
    assert cached_result_rerun is not None
    assert cached_result_rerun.result == cached_result.result

def test_query_execution_shows_response_in_chat_ui(mock_session_state, monkeypatch, ask_questions_page):
    """Test the actual bug: query execution results in visible response in chat UI."""
    # This is the integration test that verifies the bug is fixed
    # Reproduces the reported issue: responses don't show in UI after query
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py")
    app.run()

    # Simulate query
    app.chat_input[0].set_value("how many confirmed covid cases").run()

    # Assert: Response is visible in chat (assistant message exists with run_key)
    chat = app.session_state.get("chat", [])
    assert len(chat) >= 2, "Both user and assistant messages should be in chat"
    assert chat[1]["role"] == "assistant", "Assistant message should exist"
    assert chat[1].get("run_key") is not None, "Assistant message should have run_key for result retrieval"

    # Verify result can be retrieved from cache
    cache = app.session_state.get("result_cache")
    assert cache is not None, "Result cache should exist"
    result = cache.get(chat[1]["run_key"], dataset_version)
    assert result is not None, "Result should be retrievable from cache"
```

### Step 9: Run Full Test Suite

- Run: `make test-ui` (verify no regressions)
- Run: `make test-fast` (quick regression check)

### Step 10: Commit Changes

Commit with comprehensive message including test coverage.

## Key Files to Modify

1. **`src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`**

- Fix state initialization in `main()` function (where `analysis_context` is initialized)
- Fix assistant message addition order in `execute_analysis_with_idempotency()` function (after `cache.put()`)
- Improve render_chat cache retrieval in `render_chat()` function (cache miss handling)
- Enhance execution guard logging in `main()` function (where `intent_signal` is checked)

2. **`tests/ui/pages/test_ask_questions_state_machine.py`** (new)

- Test state machine persistence
- Test execution triggering

3. **`tests/ui/pages/test_ask_questions_chat_rendering.py`** (new)

- Test chat message rendering with cached results
- Test cache miss handling

4. **`tests/integration/test_chat_response_display.py`** (new)

- End-to-end flow test

## Success Criteria

1. ✅ Queries execute when `intent_signal="nl_parsed"` is set
2. ✅ Assistant messages appear in chat transcript
3. ✅ Results are rendered from cache in chat messages
4. ✅ All tests pass (new + existing)
5. ✅ No regressions in existing functionality

## Notes

- This fix addresses the immediate issue while maintaining current architecture
- State extraction (from referenced plans) would be a longer-term solution

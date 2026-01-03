---
name: Streamlit State Extraction MVP (UI-Agnostic Core)
overview: |
  Extract state management from Streamlit into pure Python classes, making Streamlit
  a "dumb view layer." This enables future UI ports (FastAPI, CLI, Jupyter) without
  rewriting core logic.

  **MVP Goal**: Prove UI-agnostic execution by extracting core state logic into
  reusable classes. Streamlit becomes an adapter, not the architecture.

  Focus: Ports and Adapters pattern - extract domain logic, keep Streamlit as adapter.

todos:
  # MVP Phase: Core State Extraction (Prove UI-Agnostic Execution)
  - id: "mvp-1"
    content: Write failing tests for ConversationManager (transcript, messages, state) (TDD Red)
    status: pending
    activeForm: Writing tests for ConversationManager
    notes: |
      Test interface:
      - add_message(role: str, content: str) -> message_id: str
      - get_transcript() -> list[Message]
      - get_current_dataset() -> str | None
      - set_dataset(dataset_id: str) -> None
      - get_active_query() -> str | None
      - set_active_query(query: str) -> None
      - get_follow_ups() -> list[str]
      - clear() -> None
      
      Message dataclass:
      - id: str
      - role: Literal["user", "assistant"]
      - content: str
      - timestamp: datetime
      - run_key: str | None (for result association)

  - id: "mvp-2"
    content: Create ConversationManager class (extract from session_state logic) (TDD Green)
    status: pending
    activeForm: Creating ConversationManager
    dependencies:
      - "mvp-1"
    notes: |
      Extract from Ask_Questions.py:
      - normalize_query() (line 141-158)
      - canonicalize_scope() (line 160-200)
      - remember_run() (lifecycle management, lines 231-296)
      - State machine logic (lines 1537-1702)
      
      Requirements:
      - Zero Streamlit imports
      - Pure Python dataclasses
      - Serializable (for persistence later)
      - Location: src/clinical_analytics/core/conversation_manager.py

  - id: "mvp-3"
    content: Run tests for ConversationManager and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring ConversationManager
    dependencies:
      - "mvp-2"
    notes: |
      Run: make test-core PYTEST_ARGS="tests/core/test_conversation_manager.py -xvs"
      Fix: All linting, formatting, type issues

  - id: "mvp-4"
    content: Write failing tests for ResultCache (LRU eviction, serialization) (TDD Red)
    status: pending
    activeForm: Writing tests for ResultCache
    notes: |
      Test interface:
      - get(run_key: str) -> CachedResult | None
      - put(run_key: str, result: CachedResult) -> None
      - evict_oldest() -> None
      - clear() -> None
      - serialize() -> dict (for persistence)
      - deserialize(data: dict) -> ResultCache
      
      CachedResult dataclass:
      - run_key: str
      - query: str
      - result: dict (serializable analysis result)
      - timestamp: datetime
      - ttl: int | None (optional time-to-live)

  - id: "mvp-5"
    content: Create ResultCache class (extract from session_state caching) (TDD Green)
    status: pending
    activeForm: Creating ResultCache
    dependencies:
      - "mvp-4"
    notes: |
      Extract from Ask_Questions.py:
      - remember_run() (caching logic)
      - cleanup_old_results() (LRU eviction)
      
      Requirements:
      - Zero Streamlit imports
      - Deterministic run keys
      - LRU eviction (configurable max size, default 50)
      - Serializable
      - Location: src/clinical_analytics/core/result_cache.py

  - id: "mvp-6"
    content: Run tests for ResultCache and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring ResultCache
    dependencies:
      - "mvp-5"
    notes: |
      Run: make test-core PYTEST_ARGS="tests/core/test_result_cache.py -xvs"
      Fix: All linting, formatting, type issues

  - id: "mvp-7"
    content: Write failing tests for QueryService (wraps QuestionEngine) (TDD Red)
    status: pending
    activeForm: Writing tests for QueryService
    notes: |
      Test interface:
      - ask(question: str, dataset_id: str, context: ConversationContext) -> QueryResult
      
      QueryResult dataclass:
      - plan: QueryPlan
      - issues: list[Issue]
      - result: dict | None (analysis result if executed)
      - confidence: float
      - run_key: str
      
      Requirements:
      - Zero Streamlit dependencies
      - Accept ConversationManager for context
      - Return structured QueryResult

  - id: "mvp-8"
    content: Create QueryService class (wraps existing question_engine.py logic) (TDD Green)
    status: pending
    activeForm: Creating QueryService
    dependencies:
      - "mvp-7"
    notes: |
      Wrap src/clinical_analytics/ui/components/question_engine.py:
      - QuestionEngine.parse_query() - intent extraction
      - QuestionEngine.execute_with_timeout() - analysis execution
      - Reuse AnalysisContext, AnalysisIntent types
      
      Requirements:
      - Zero Streamlit imports
      - Accept ConversationManager for context
      - Return structured QueryResult
      - Location: src/clinical_analytics/core/query_service.py

  - id: "mvp-9"
    content: Run tests for QueryService and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring QueryService
    dependencies:
      - "mvp-8"
    notes: |
      Run: make test-core PYTEST_ARGS="tests/core/test_query_service.py -xvs"
      Fix: All linting, formatting, type issues

  # MVP Phase: Streamlit Adapter (Refactor UI to Use Core)
  - id: "mvp-10"
    content: Refactor Ask_Questions.py to use ConversationManager (TDD Red)
    status: pending
    activeForm: Refactoring Ask_Questions.py
    dependencies:
      - "mvp-3"
    notes: |
      Replace st.session_state usage with ConversationManager:
      - Initialize: manager = ConversationManager()
      - Store in session_state only for Streamlit lifecycle: st.session_state["conversation_manager"] = manager
      - All state logic goes through manager
      - Search for all st.session_state["messages"], st.session_state["current_dataset"], etc.
      - Replace with manager.get_transcript(), manager.get_current_dataset(), etc.

  - id: "mvp-11"
    content: Refactor Ask_Questions.py to use ResultCache (TDD Green)
    status: pending
    activeForm: Refactoring Ask_Questions.py caching
    dependencies:
      - "mvp-6"
      - "mvp-10"
    notes: |
      Replace session_state caching with ResultCache:
      - Initialize: cache = ResultCache(max_size=50)
      - Store in session_state only for Streamlit lifecycle: st.session_state["result_cache"] = cache
      - All cache operations go through cache object
      - Replace remember_run() calls with cache.put()
      - Replace cleanup_old_results() with cache.evict_oldest()

  - id: "mvp-12"
    content: Refactor Ask_Questions.py to use QueryService (TDD Green)
    status: pending
    activeForm: Refactoring Ask_Questions.py query execution
    dependencies:
      - "mvp-9"
      - "mvp-11"
    notes: |
      Replace direct QuestionEngine calls with QueryService:
      - Initialize: service = QueryService()
      - Call: result = service.ask(question, dataset_id, manager.get_context())
      - Render result using existing UI code
      - Update manager with new message and query

  - id: "mvp-13"
    content: Run tests and verify st.session_state isolation (TDD Refactor)
    status: pending
    activeForm: Verifying Streamlit isolation
    dependencies:
      - "mvp-12"
    notes: |
      Acceptance criteria:
      - Search repo for st.session_state - only in UI layer (Ask_Questions.py, Add_Your_Data.py)
      - Core test suite runs without Streamlit installed
      - All existing UI tests still pass
      - Run: make test-ui PYTEST_ARGS="tests/ui/test_ask_questions.py -xvs"
      - Verify: grep -r "st.session_state" src/clinical_analytics/core/ (should return nothing)

  # Phase 1: Minimal Persistence Boundary (Optional for MVP)
  - id: "1"
    content: Write failing tests for StateStore interface (save/load) (TDD Red)
    status: pending
    activeForm: Writing tests for StateStore
    dependencies:
      - "mvp-13"
    notes: |
      Test interface:
      - save(upload_id: str, dataset_version: str, state: ConversationState) -> None
      - load(upload_id: str, dataset_version: str) -> ConversationState | None
      - Pluggable backend (file/DuckDB)
      
      ConversationState dataclass:
      - conversation_manager: ConversationManager (serialized)
      - result_cache: ResultCache (serialized)
      - dataset_id: str
      - last_updated: datetime

  - id: "2"
    content: Create StateStore interface and FileBackend implementation (TDD Green)
    status: pending
    activeForm: Creating StateStore
    dependencies:
      - "1"
    notes: |
      Minimal persistence:
      - StateStore abstract base class
      - FileStateStore (JSON files in data/sessions/{upload_id}_{dataset_version}.json)
      - ConversationState = ConversationManager + ResultCache serialized
      - No UI logic changes required
      - Location: src/clinical_analytics/core/state_store.py

  - id: "3"
    content: Run tests for StateStore and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring StateStore
    dependencies:
      - "2"
    notes: |
      Run: make test-core PYTEST_ARGS="tests/core/test_state_store.py -xvs"
      Fix: All linting, formatting, type issues

  - id: "4"
    content: Integrate StateStore into Streamlit adapter (load on init, save on change)
    status: pending
    activeForm: Integrating persistence
    dependencies:
      - "3"
    notes: |
      In Ask_Questions.py:
      - On page load: Load state from StateStore if exists
      - On state change: Save state to StateStore
      - Acceptance criteria:
        - Restart app, conversation + cache persists (for single dataset)
        - No UI logic changes required

  # Phase 2: Quality Gates & Verification
  - id: "5"
    content: Run make format && make lint-fix on all new code
    status: pending
    activeForm: Running format and lint
    dependencies:
      - "mvp-13"
      - "4"
    notes: |
      Run on all new files:
      - src/clinical_analytics/core/conversation_manager.py
      - src/clinical_analytics/core/result_cache.py
      - src/clinical_analytics/core/query_service.py
      - src/clinical_analytics/core/state_store.py (if Phase 1 done)
      - Modified: src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py

  - id: "6"
    content: Verify all tests passing, run type-check, verify core runs without Streamlit
    status: pending
    activeForm: Verifying all quality gates
    dependencies:
      - "5"
    notes: |
      - Run: make test-core (all core tests pass)
      - Run: make test-ui (all UI tests pass)
      - Run: make type-check (no new type errors)
      - Verify: Core classes import without streamlit
        - python -c "import sys; sys.path.insert(0, 'src'); from clinical_analytics.core.conversation_manager import ConversationManager; print('OK')"
      - All quality gates must pass

  - id: "7"
    content: Commit changes with comprehensive commit message
    status: pending
    activeForm: Committing changes
    dependencies:
      - "6"
    notes: |
      Format:
      feat: Extract state management from Streamlit (UI-agnostic core MVP)

      - Create ConversationManager (pure Python, zero Streamlit)
      - Create ResultCache (LRU eviction, serializable)
      - Create QueryService (wraps QuestionEngine, zero Streamlit)
      - Refactor Streamlit UI to use core classes (adapter pattern)
      - Add minimal StateStore persistence boundary (optional)
      - Add comprehensive test suite (X tests passing)

      All tests passing: X/Y
      Following TDD: Red-Green-Refactor
      Core runs without Streamlit installed
      MVP: Proves UI-agnostic execution

---
# Streamlit State Extraction MVP Plan

## Overview

This plan extracts state management from Streamlit into pure Python classes, making Streamlit a "dumb view layer." This enables future UI ports (FastAPI, CLI, Jupyter) without rewriting core logic.

**MVP Goal**: Prove UI-agnostic execution by extracting core state logic into reusable classes. Streamlit becomes an adapter, not the architecture.

## Current State Analysis

### Existing Streamlit Dependencies
- `st.session_state` - Mini state machine for workflow
- State scattered across `Ask_Questions.py`:
  - `normalize_query()` (line 141-158)
  - `canonicalize_scope()` (line 160-200)
  - `remember_run()` (lifecycle management, lines 231-296)
  - `cleanup_old_results()` (LRU eviction)
  - State machine logic (lines 1537-1702)

### Core Business Logic (Reusable)
✅ **Can be extracted and reused**:
- `components/question_engine.py` - Query intent inference
- `components/result_interpreter.py` - Statistical interpretation
- All `src/clinical_analytics/core/` modules (semantic layer, NL engine, etc.)
- All `src/clinical_analytics/analysis/` modules (stats, survival, compute)

## Target Architecture (MVP)

### Core (UI-Agnostic)
```
src/clinical_analytics/core/
├── conversation_manager.py    # Conversation history, dataset context, follow-ups
├── result_cache.py            # LRU cache for analysis results
├── query_service.py           # Wraps QuestionEngine, zero Streamlit
└── state_store.py             # Optional: Persistence boundary
```

### Adapter (Streamlit)
```
src/clinical_analytics/ui/pages/
└── 03_💬_Ask_Questions.py    # Dumb view layer, calls core classes
```

**Key Principle**: Streamlit becomes a thin adapter that:
- Takes user input
- Calls QueryService
- Renders PlanCard, Issues, ResultPanels
- Stores core objects in session_state ONLY for Streamlit lifecycle

## MVP Success Criteria

1. ✅ **Core classes exist** (ConversationManager, ResultCache, QueryService)
2. ✅ **Zero Streamlit imports in core** (verified by grep)
3. ✅ **Core test suite runs without Streamlit** (imports work)
4. ✅ **Streamlit UI refactored** to use core classes
5. ✅ **All existing tests pass** (no regressions)
6. ✅ **st.session_state isolated** to UI layer only

## What We're NOT Building (Deferred)

- ❌ FastAPI backend (deferred - not needed for MVP)
- ❌ Next.js frontend (deferred - not needed for MVP)
- ❌ SSE streaming (deferred - solve with progressive rendering)
- ❌ Multi-user sessions (deferred - single-user MVP)
- ❌ Database migrations (deferred - file-based persistence is fine)

## Migration Strategy

### MVP Phase: Core Extraction
1. Extract ConversationManager (pure Python)
2. Extract ResultCache (pure Python)
3. Extract QueryService (pure Python)
4. Refactor Streamlit to use core classes
5. Verify isolation (core runs without Streamlit)

### Phase 1: Minimal Persistence (Optional)
1. Create StateStore interface
2. Implement FileBackend
3. Integrate into Streamlit adapter
4. Verify persistence works

## Testing Strategy

### Core Tests (No Streamlit)
- Unit tests for ConversationManager
- Unit tests for ResultCache
- Unit tests for QueryService
- Integration tests (core classes work together)

### UI Tests (With Streamlit)
- Refactored Ask_Questions.py tests
- Verify UI still works
- Verify no regressions

### TDD Workflow
1. Write failing test (Red)
2. Implement feature (Green)
3. Refactor and fix quality (Refactor)
4. Run module test suite
5. Commit with tests

## Dependencies

### No New Dependencies Required
- Use existing Python stdlib (dataclasses, collections, datetime)
- Use existing project dependencies (pydantic for validation if needed)

## Risk Mitigation

### Risk 1: Breaking Existing UI
- **Mitigation**: TDD workflow, comprehensive tests, incremental refactoring
- **Rollback**: Git revert if tests fail

### Risk 2: State Migration Issues
- **Mitigation**: Keep session_state for Streamlit lifecycle, core classes are stateless
- **Benefit**: No migration needed, clean separation

## Success Criteria

1. ✅ Core classes exist and are tested
2. ✅ Core runs without Streamlit installed
3. ✅ Streamlit UI refactored to use core
4. ✅ All tests pass (no regressions)
5. ✅ st.session_state isolated to UI layer
6. ✅ Ready for future UI ports (FastAPI, CLI, Jupyter)

## Timeline Estimate

- **MVP Phase**: ~2-3 days
- **Phase 1 (Persistence)**: ~1 day (optional)
- **Total**: ~3-4 days

## Next Steps After MVP

Once MVP is complete:
1. ✅ Core is UI-agnostic (proven)
2. ✅ Can port to FastAPI/CLI/Jupyter without rewriting core
3. ✅ Focus shifts to ADR004 (the actual product bet)

---

## Notes

- This plan follows strict TDD discipline
- All Makefile commands must be used (never `pytest` directly)
- Fixtures from `tests/conftest.py` must be reused
- Rule of Three: Don't abstract until third instance
- All tests must pass before commit


---
name: Electron UI Migration
overview: Replace Streamlit UI with an Electron.js desktop app featuring a Claude-style chat interface. The Electron app connects to the existing FastAPI backend (with new endpoints) for all analytics functionality. Includes surgical extraction of Streamlit dependencies from core layer. ADR011 enrichment components (enrichment_panel.py, patch_history.py, enrichment_integration.py) are pure Python - they are KEPT and wrapped by API routes, with Electron UI built on top.
todos:
  - id: phase0-core-cleanup
    content: "Prerequisite: Extract Streamlit + UI imports from core layer (clarifying_questions.py, query_service.py)"
    status: done
  - id: phase1-dataset-routes
    content: "Backend: Implement /api/datasets routes (use existing make test-api)"
    status: done
    dependencies:
      - phase0-core-cleanup
  - id: phase2-query-service
    content: "Backend: Extend existing core/query_service.py for async API usage"
    status: done
    dependencies:
      - phase1-dataset-routes
  - id: phase3-query-routes-sse
    content: "Backend: Implement /api/queries routes with SSE streaming"
    status: done
    dependencies:
      - phase2-query-service
  - id: phase3b-enrichment-api
    content: "Backend: Add /api/enrichments routes (wrap existing pure Python logic)"
    status: done
    dependencies:
      - phase1-dataset-routes
  - id: phase4-electron-skeleton
    content: "Electron: Create app skeleton with SSE + CORS config update"
    status: done
  - id: phase5-chat-interface
    content: "Electron: Build chat interface with dataset selector"
    status: done
    dependencies:
      - phase4-electron-skeleton
      - phase1-dataset-routes
  - id: phase6-result-rendering
    content: "Electron: Render analysis results inline in chat (typed intent summaries + table previews for P20–P24 parity paths)"
    status: done
    dependencies:
      - phase5-chat-interface
      - phase3-query-routes-sse
  - id: phase6b-enrichment-panel
    content: "Electron: Build enrichment diff panel UI (consumes Phase 3b API)"
    status: done
    dependencies:
      - phase5-chat-interface
      - phase3b-enrichment-api
  - id: phase6c-patch-history
    content: "Electron: Build patch history viewer UI (consumes Phase 3b API)"
    status: done
    dependencies:
      - phase6b-enrichment-panel
  - id: phase7-session-sidebar
    content: "Electron: Add session history sidebar"
    status: done
    dependencies:
      - phase5-chat-interface
  - id: phase8-dataset-upload
    content: "Electron: Implement dataset upload flow"
    status: done
    dependencies:
      - phase6-result-rendering
  - id: phase9-e2e-testing
    content: "Testing: E2E — Playwright Chromium renderer suite + native Electron-binary harness enforced in quality gate"
    status: done
    dependencies:
      - phase8-dataset-upload
  - id: phase10a-delete-pages
    content: "Cutover: Delete Streamlit pages (git is rollback)"
    status: pending
    dependencies:
      - phase9-e2e-testing
  - id: phase10b-delete-components
    content: "Cutover: Delete Streamlit-only UI components"
    status: pending
    dependencies:
      - phase10a-delete-pages
  - id: phase10c-remove-deps
    content: "Cutover: Remove Streamlit from pyproject.toml"
    status: pending
    dependencies:
      - phase10b-delete-components
  - id: phase10d-delete-tests
    content: "Cutover: Delete Streamlit UI tests"
    status: pending
    dependencies:
      - phase10c-remove-deps
  - id: phase10e-final-commit
    content: "Cutover: Final commit with clear rollback instructions"
    status: pending
    dependencies:
      - phase10d-delete-tests
---

# Streamlit to Electron.js Chat UI Migration

**Status canon:** Phase and parity truth for gating live in [`docs/implementation/MASTER_PLAN.md`](../../docs/implementation/MASTER_PLAN.md) and [`docs/specs/omnibus_electron_streamlit_cutover.md`](../../docs/specs/omnibus_electron_streamlit_cutover.md). Update this file’s YAML todos when you close a phase so Task Master / agents do not drift.

**Repo context diagnostic (`repo-context` handoff):** [`.context/diagnostics/electron_ui_migration_context.md`](../../.context/diagnostics/electron_ui_migration_context.md) — use with **`/staff-consult`** and **plan-to-pr** (or your team’s ship skill) step 1 (path is gitignored by default; copy excerpts into PR/plan if needed).

## Problem Statement

The current Streamlit UI has 8 pages with complex state management that is fragile and hard to maintain (see `Ask_Questions.py` lines 1755-1830). The goal is to replace it with a lightweight Electron.js desktop app featuring a chat-only interface where all analytics are driven through natural language conversation.

**Critical constraint**: The core layer (`src/clinical_analytics/core/`) currently has a Streamlit dependency in `clarifying_questions.py` that must be extracted before QueryService can wrap NLQueryEngine without pulling in UI dependencies.

## Plan Dependencies (Must Execute First)

| Plan | Reason | Status | Estimated Time |
|------|--------|--------|----------------|
| `fix_variable_name_resolution_a7c3e1f2.plan.md` | QueryService uses NLQueryEngine which has known variable resolution bug | ✅ **Completed** (in Archive) | Done |
| `ask_questions_solid_refactor_a57bf5ed.plan.md` | Refactors Ask_Questions.py | ⚠️ **Pending** - DECOUPLED | 12 hours |
| `adr011_metadata_enrichment_bb84ac23.plan.md` | LLM-assisted metadata enrichment | ✅ **Completed** (merged PR45) | Done |

**Execution order:**
1. ~~`fix_variable_name_resolution`~~ ✅ Completed (archived)
2. ~~`adr011_metadata_enrichment`~~ ✅ Completed (PR45 merged)
3. **This plan Phase 0** (extracts Streamlit + UI imports from core)
4. This plan Phases 1-10 (Electron migration)

**ADR011 Integration:** PR45 merged ADR011. The "UI components" (`enrichment_panel.py`, `patch_history.py`, `enrichment_integration.py`) are **pure Python with no Streamlit imports** (verified 2026-01-20). This plan:
- Phase 3b: API routes wrap existing pure Python logic (no reimplementation)
- Phase 6b: Electron UI consumes API (enrichment panel)
- Phase 6c: Electron UI consumes API (patch history viewer)
- Phase 10b: **KEEPS** ADR011 components (pure Python, used by API)

**Decoupling from `ask_questions_solid_refactor`:** Phase 2 extends the existing `core/query_service.py` directly instead of using the planned `AnalysisExecutor`. This removes the blocking dependency. The SOLID refactor can proceed independently after this plan completes.

**Note**: Phase 0 cleans up both Streamlit imports AND UI layer imports from core (discovered: `core/query_service.py` imports from `ui/components/question_engine`).

## TDD Workflow (All Phases)

**Mandatory for backend phases (0-3):**

1. **Red**: Write failing test first
2. **Verify Red**: Run `make test PYTEST_ARGS="tests/api/test_<module>.py -xvs"` - confirm test fails
3. **Green**: Implement minimum code to pass test
4. **Verify Green**: Run same test command - confirm test passes
5. **Quality Gate**: Run `make check` (format, lint, type-check, test)
6. **Commit**: Include implementation + tests in same commit

**For Electron phases (4-9):**

1. **Red**: Write failing unit test (Vitest) or E2E test (Playwright)
2. **Verify Red**: Run `cd electron && npm test` or `npm run test:e2e`
3. **Green**: Implement minimum code
4. **Verify Green**: Run same test command
5. **Quality Gate**: Run `cd electron && npm run lint && npm run typecheck`
6. **Commit**: Include implementation + tests

## Development Commands

**Backend (Phases 0-3):**

```bash
# Start FastAPI dev server
make run-api
# OR: uvicorn clinical_analytics.api.main:app --reload --port 8000

# Run API tests (uses existing test infrastructure)
make test PYTEST_ARGS="tests/api/ -xvs"

# Run specific test file
make test PYTEST_ARGS="tests/api/test_datasets.py -xvs"

# Run core tests (for Phase 0)
make test-core PYTEST_ARGS="tests/core/test_clarifying_questions.py -xvs"

# Full quality gate
make check
```

**Electron (Phases 4-9):**

```bash
# Install dependencies
cd electron && npm install

# Start Electron dev mode
npm run dev

# Run unit tests
npm test

# Run E2E tests (requires backend running)
npm run test:e2e

# Lint and format
npm run lint
npm run format
```

## Observability Requirements

All new API endpoints must use structured logging with `structlog`:

```python
import structlog
logger = structlog.get_logger()

# Required log events per endpoint:
logger.info("query_submitted", query_id=query_id, dataset_id=dataset_id)
logger.info("query_completed", query_id=query_id, duration_ms=elapsed, intent_type=intent)
logger.warning("query_failed", query_id=query_id, error=str(e))
```

## Architecture

```mermaid
graph TB
    subgraph ElectronApp [Electron App]
        MainProcess[main.js]
        Preload[preload.js]
        Renderer[renderer/]
    end

    subgraph RendererComponents [Renderer Components]
        ChatInterface[ChatInterface]
        DatasetSelector[DatasetSelector]
        SessionList[SessionList]
        ResultRenderer[ResultRenderers]
    end

    subgraph FastAPIBackend [FastAPI Backend]
        SessionRoutes[/api/sessions]
        DatasetRoutes[/api/datasets]
        QueryRoutes[/api/queries]
        QueryService[QueryService]
        NLQueryEngine[NLQueryEngine]
        SemanticLayer[SemanticLayer]
    end

    MainProcess --> Preload
    Preload --> Renderer
    Renderer --> RendererComponents
    ChatInterface -->|HTTP + SSE| QueryRoutes
    DatasetSelector -->|HTTP| DatasetRoutes
    SessionList -->|HTTP| SessionRoutes
    QueryRoutes --> QueryService
    QueryService --> NLQueryEngine
    NLQueryEngine --> SemanticLayer
```

## API Contracts

### QueryService Protocol (Phase 2)

```python
from typing import Protocol, AsyncIterator

class QueryServiceProtocol(Protocol):
    """Service layer interface for query execution."""

    async def submit_query(self, request: QueryRequest) -> str:
        """Submit query, returns query_id."""
        ...

    async def get_result(self, query_id: str) -> QueryResult:
        """Get completed result by query_id."""
        ...

    async def stream_events(self, query_id: str) -> AsyncIterator[SSEEvent]:
        """Stream SSE events for query progress."""
        ...
```

### Dataset Endpoints (Phase 1)

```python
# GET /api/datasets
class DatasetSummary(BaseModel):
    dataset_id: str          # e.g., "upload_abc123" or "mimic"
    name: str                # Display name
    source: Literal["uploaded", "builtin"]
    table_count: int
    row_count: int
    created_at: datetime | None

class DatasetListResponse(BaseModel):
    datasets: list[DatasetSummary]
    total: int

# GET /api/datasets/{dataset_id}
class DatasetDetail(BaseModel):
    dataset_id: str
    name: str
    source: Literal["uploaded", "builtin"]
    tables: list[TableInfo]
    schema: dict[str, str]   # column -> dtype
    created_at: datetime | None
    metadata: dict[str, Any] | None

class TableInfo(BaseModel):
    name: str
    row_count: int
    columns: list[str]

# GET /api/datasets/{dataset_id}/preview?limit=10
class DatasetPreview(BaseModel):
    dataset_id: str
    rows: list[dict[str, Any]]  # First N rows as dicts
    columns: list[str]
    total_rows: int
```

### Query Endpoints (Phase 3)

```python
# POST /api/queries
class QueryRequest(BaseModel):
    query: str               # Natural language query
    dataset_id: str
    session_id: str | None   # Optional session for context

class QueryResponse(BaseModel):
    query_id: str            # e.g., "qry_abc123"
    status: Literal["pending", "processing", "completed", "failed"]
    stream_url: str          # SSE endpoint: /api/queries/{query_id}/stream

# GET /api/queries/{query_id}
class QueryResult(BaseModel):
    query_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    intent_type: str | None  # DESCRIBE, COMPARE_GROUPS, etc.
    result: AnalysisResult | None
    interpretation: str | None
    follow_ups: list[str]
    error: str | None
    created_at: datetime
    completed_at: datetime | None

class AnalysisResult(BaseModel):
    data: list[dict[str, Any]]  # Result rows
    columns: list[str]
    summary: dict[str, Any]     # Statistics summary
```

### SSE Event Mapping (Phase 3)

Aligns with existing `SSEEvent` schema in `schemas.py`:

| Phase | Existing Event | Payload |
|-------|----------------|---------|
| Query submitted | `query_started` | `{query_id, dataset_id}` |
| Parsing query | `query_progress` | `{stage: "parsing", message: "Analyzing query..."}` |
| Executing analysis | `query_progress` | `{stage: "executing", message: "Running analysis..."}` |
| Results ready | `query_completed` | `{query_id, intent_type, result_preview}` |
| Interpretation ready | `interpretation_ready` | `{interpretation, follow_ups}` |
| Error occurred | `query_failed` | `{error, details}` |

## Key Files

**Backend (extend existing):**

- [`src/clinical_analytics/api/routes/sessions.py`](src/clinical_analytics/api/routes/sessions.py) - Already complete
- `src/clinical_analytics/api/routes/datasets.py` - NEW: Dataset CRUD + upload
- `src/clinical_analytics/api/routes/queries.py` - NEW: Query execution with SSE
- `src/clinical_analytics/api/routes/enrichments.py` - NEW: ADR011 enrichment API (Phase 3b)
- `src/clinical_analytics/api/services/query_service.py` - NEW: Async wrapper around core service
- [`src/clinical_analytics/core/query_service.py`](src/clinical_analytics/core/query_service.py) - EXISTING: Core sync service (refactored in Phase 0)
- `src/clinical_analytics/core/analysis_types.py` - NEW: Extracted from UI layer (Phase 0)

**ADR011 Core (already merged in PR45, reused by API):**

- [`src/clinical_analytics/core/overlay_store.py`](src/clinical_analytics/core/overlay_store.py) - EXISTING: JSONL patch storage
- [`src/clinical_analytics/core/metadata_resolver.py`](src/clinical_analytics/core/metadata_resolver.py) - EXISTING: Deterministic merge
- [`src/clinical_analytics/core/llm_enrichment.py`](src/clinical_analytics/core/llm_enrichment.py) - EXISTING: LLM suggestions

**Electron (new directory):**

- `electron/main.js` - Main process (window management)
- `electron/preload.js` - IPC bridge (security boundary)
- `electron/renderer/index.html` - App shell
- `electron/renderer/renderer.js` - Chat logic, SSE handling
- `electron/renderer/style.css` - Claude-inspired styling
- `electron/package.json` - Electron dependencies

## Implementation Plan

### Phase 0: Extract Streamlit + UI Imports from Core Layer

**Goal:** Ensure core layer has zero Streamlit imports AND zero UI layer imports

**Problems Identified:**

1. `src/clinical_analytics/core/clarifying_questions.py` contains:
```python
import streamlit as st
# Uses st.subheader(), st.selectbox(), st.warning() directly
```

2. `src/clinical_analytics/core/query_service.py` line 16 imports from UI layer:
```python
from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent
```

These architectural violations prevent clean API usage without pulling in UI dependencies.

**TDD Steps:**

1. Write test (`tests/core/test_core_layer_no_ui.py`):
   - `test_core_layer_has_no_streamlit_imports` (grep check)
   - `test_core_layer_has_no_ui_imports` (grep check for `from clinical_analytics.ui`)
   - `test_clarifying_questions_returns_data_not_renders`
   - `test_query_service_has_no_ui_imports`
2. Run `make test-core PYTEST_ARGS="tests/core/test_core_layer_no_ui.py -xvs"` (verify RED)
3. Refactor `clarifying_questions.py`:
   - Extract UI rendering to `ui/components/clarifying_ui.py`
   - Keep pure logic in core, return data structures (dicts/dataclasses)
   - Create `ClarificationRequest` dataclass for UI layer to render
4. Refactor `query_service.py`:
   - Extract `AnalysisContext`/`AnalysisIntent` to `core/analysis_types.py` (pure dataclasses)
   - Update UI layer to import from core instead
5. Run tests again (verify GREEN)
6. Run `make check` (quality gate)

**Implementation:**

```python
# NEW: src/clinical_analytics/core/analysis_types.py (extracted from UI)
from dataclasses import dataclass
from enum import Enum

class AnalysisIntent(Enum):
    """Analysis intent types - moved from ui/components/question_engine.py."""
    DESCRIBE = "describe"
    COMPARE_GROUPS = "compare_groups"
    FIND_PREDICTORS = "find_predictors"
    EXAMINE_SURVIVAL = "examine_survival"
    EXPLORE_RELATIONSHIPS = "explore_relationships"
    COUNT = "count"

@dataclass
class AnalysisContext:
    """Analysis context - moved from ui/components/question_engine.py."""
    inferred_intent: AnalysisIntent | None = None
    primary_variable: str | None = None
    grouping_variable: str | None = None
    predictor_variables: list[str] | None = None
    time_variable: str | None = None
    event_variable: str | None = None
    filters: dict | None = None
    query_plan: Any = None

# REFACTORED: src/clinical_analytics/core/clarifying_questions.py
@dataclass
class ClarificationRequest:
    """Data structure for UI layer to render clarifying questions."""
    question_type: Literal["intent", "variable", "grouping", "collision"]
    prompt: str
    options: list[str]
    current_value: str | None = None

def generate_clarifications(
    intent: QueryIntent,
    semantic_layer: SemanticLayer,
) -> list[ClarificationRequest]:
    """Generate list of clarifications needed (no UI rendering)."""
    # ... pure logic, returns data structures only

# REFACTORED: src/clinical_analytics/core/query_service.py
# Change: from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent
# To:     from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent

# UPDATE: src/clinical_analytics/ui/components/question_engine.py
# Remove AnalysisContext/AnalysisIntent definitions
# Add:    from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent

# NEW: src/clinical_analytics/ui/components/clarifying_ui.py
import streamlit as st
from clinical_analytics.core.clarifying_questions import ClarificationRequest

def render_clarifications(clarifications: list[ClarificationRequest]) -> dict[str, Any]:
    """Render clarifications using Streamlit, return user selections."""
    # ... Streamlit-specific rendering
```

**Quality Gate:**

- `grep "^import streamlit" src/clinical_analytics/core/` returns 0 matches
- `grep "^from streamlit" src/clinical_analytics/core/` returns 0 matches
- `grep "from clinical_analytics.ui" src/clinical_analytics/core/` returns 0 matches
- All existing tests pass
- Tests: 4 new tests

**Commit:** `refactor: Phase 0 - Extract Streamlit + UI imports from core layer`

- `src/clinical_analytics/core/analysis_types.py` (new)
- `src/clinical_analytics/core/clarifying_questions.py` (refactored)
- `src/clinical_analytics/core/query_service.py` (refactored imports)
- `src/clinical_analytics/ui/components/clarifying_ui.py` (new)
- `src/clinical_analytics/ui/components/question_engine.py` (updated imports)
- `tests/core/test_core_layer_no_ui.py` (new)
- All tests passing: X/Y

---

### Phase 1: Backend - Dataset Routes

**Goal:** Expose dataset listing and metadata via API

**Note:** `make test-api` command already exists (Makefile line 154). No need to add it.

**TDD Steps:**

1. Write failing tests (`tests/api/test_datasets.py`):
   - `test_datasets_list_empty_returns_empty_list`
   - `test_datasets_list_with_uploads_returns_datasets`
   - `test_datasets_get_existing_returns_detail`
   - `test_datasets_get_missing_returns_404`
   - `test_datasets_preview_returns_rows`
   - `test_datasets_preview_respects_limit`
2. Run `make test-api` or `make test PYTEST_ARGS="tests/api/test_datasets.py -xvs"` (verify RED)
3. Implement routes in `src/clinical_analytics/api/routes/datasets.py`
4. Run tests again (verify GREEN)
5. Run `make check` (quality gate)

**Implementation:**

- `GET /api/datasets` - List available datasets
- `GET /api/datasets/{dataset_id}` - Get dataset metadata
- `GET /api/datasets/{dataset_id}/preview` - Get sample rows

**Key integration:** Reuse `UploadedDatasetFactory.list_available_uploads()` from [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py)

**Quality Gate:**

- Tests: 6 new tests
- Coverage: Maintain 67% minimum
- Lint: Pass `make lint`
- Format: Pass `make format-check`

**Commit:** `feat: Phase 1 - Dataset API routes`

- `src/clinical_analytics/api/routes/datasets.py`
- `src/clinical_analytics/api/models/schemas.py` (add DatasetSummary, etc.)
- `src/clinical_analytics/api/main.py` (register router)
- `tests/api/test_datasets.py`
- All tests passing: X/Y

---

### Phase 2: Backend - Query Service (Extend Existing)

**Goal:** Extend existing `core/query_service.py` for async API usage

**Existing Code:** `src/clinical_analytics/core/query_service.py` already exists with:
- `QueryService` class wrapping `NLQueryEngine`
- `QueryResult` dataclass with plan, issues, result, confidence
- `ask()` method for synchronous query execution

**Prerequisite:** Phase 0 must complete first (removes UI imports from `core/query_service.py`)

**Decision:** Extend existing service instead of creating duplicate. Add async wrapper in `api/services/`.

**TDD Steps:**

1. Write failing tests (`tests/api/test_query_service_async.py`):
   - `test_async_query_service_delegates_to_core`
   - `test_async_query_service_returns_query_id`
   - `test_async_query_service_streams_progress_events`
   - `test_async_query_service_invalid_dataset_raises_error`
   - `test_async_query_service_empty_query_raises_error`
2. Run `make test-api` or `make test PYTEST_ARGS="tests/api/test_query_service_async.py -xvs"` (verify RED)
3. Implement `AsyncQueryService` class in `api/services/query_service.py`
4. Run tests again (verify GREEN)
5. Run `make check` (quality gate)

**Implementation:**

Create async wrapper that delegates to existing `core/query_service.py`:

```python
# src/clinical_analytics/api/services/query_service.py
from clinical_analytics.core.query_service import QueryService as CoreQueryService

class AsyncQueryService:
    """Async wrapper around core QueryService for API usage."""

    def __init__(self, semantic_layer: SemanticLayer):
        self._core = CoreQueryService(semantic_layer)

    async def submit_query(self, request: QueryRequest) -> str:
        """Submit query, returns query_id."""
        query_id = generate_query_id()
        # Run synchronous core service in thread pool
        await asyncio.to_thread(self._core.ask, request.query, ...)
        return query_id

    async def stream_events(self, query_id: str) -> AsyncIterator[SSEEvent]:
        """Stream SSE events for query progress."""
        yield SSEEvent(event="query_started", data={"query_id": query_id})
        # ... delegate to core, emit progress events ...
```

**Key integration:**
- Wraps existing `core/query_service.py` (no duplication)
- Uses `NLQueryEngine` via core service
- No dependency on `AnalysisExecutor` (decoupled from SOLID refactor plan)

**Quality Gate:**

- Tests: 5 new tests
- Coverage: Maintain 67% minimum
- Lint: Pass `make lint`
- Format: Pass `make format-check`

**Commit:** `feat: Phase 2 - AsyncQueryService wrapping core service`

- `src/clinical_analytics/api/services/query_service.py` (new async wrapper)
- `tests/api/test_query_service_async.py`
- All tests passing: X/Y

---

### Phase 3: Backend - Query Routes with SSE

**Goal:** Expose query execution with streaming progress

**TDD Steps:**

1. Write failing tests (`tests/api/test_queries.py`):
   - `test_queries_post_valid_returns_query_id`
   - `test_queries_post_invalid_dataset_returns_404`
   - `test_queries_get_pending_returns_status`
   - `test_queries_get_completed_returns_result`
   - `test_queries_stream_emits_events` (async test)
2. Run `make test PYTEST_ARGS="tests/api/test_queries.py -xvs"` (verify RED)
3. Implement routes
4. Run tests again (verify GREEN)
5. Run `make check` (quality gate)

**Implementation:**

- `POST /api/queries` - Submit query (returns query_id)
- `GET /api/queries/{query_id}/stream` - SSE stream for results
- `GET /api/queries/{query_id}` - Get completed result

**SSE Implementation:**

```python
from fastapi.responses import StreamingResponse
import structlog

logger = structlog.get_logger()

async def stream_query_events(query_id: str):
    logger.info("sse_stream_started", query_id=query_id)
    yield f"event: query_started\ndata: {json.dumps({'query_id': query_id})}\n\n"
    yield f"event: query_progress\ndata: {json.dumps({'stage': 'parsing'})}\n\n"
    # ... execute query ...
    yield f"event: query_completed\ndata: {json.dumps({'result': result})}\n\n"
    logger.info("sse_stream_completed", query_id=query_id)
```

**Quality Gate:**

- Tests: 5 new tests
- Coverage: Maintain 67% minimum
- Lint: Pass `make lint`
- Format: Pass `make format-check`

**Commit:** `feat: Phase 3 - Query routes with SSE streaming`

- `src/clinical_analytics/api/routes/queries.py`
- `src/clinical_analytics/api/models/schemas.py` (add QueryRequest, etc.)
- `src/clinical_analytics/api/main.py` (register router)
- `tests/api/test_queries.py`
- All tests passing: X/Y

---

### Phase 3b: Backend - Enrichment API Routes (ADR011 Migration)

**Goal:** Expose ADR011 metadata enrichment functionality via API for Electron

**Key Discovery (Verified 2026-01-20):** The "UI components" from PR45 are actually **pure Python logic** with no Streamlit imports. They can be directly reused by API routes:

**Pure Python Logic (REUSE - no Streamlit):**
- `ui/components/enrichment_panel.py` - `prepare_diff_view_data()`, `handle_accept()`, `handle_reject()`
- `ui/components/patch_history.py` - `load_patch_history()`, `filter_patch_history()`, `export_to_csv/json()`
- `ui/components/enrichment_integration.py` - `EnrichmentService` class with `trigger_enrichment()`, `accept_suggestion()`, etc.

**Core Logic (already API-ready):**
- `core/overlay_store.py` - JSONL patch storage
- `core/metadata_resolver.py` - Deterministic merge resolver
- `core/llm_enrichment.py` - LLM suggestion generation

**TDD Steps:**

1. Write failing tests (`tests/api/test_enrichments.py`):
   - `test_enrichments_get_pending_returns_suggestions`
   - `test_enrichments_post_accept_applies_patch`
   - `test_enrichments_post_reject_marks_rejected`
   - `test_enrichments_get_history_returns_patches`
   - `test_enrichments_post_generate_creates_suggestions`
2. Run `make test-api` (verify RED)
3. Implement routes in `src/clinical_analytics/api/routes/enrichments.py`
4. Run tests again (verify GREEN)
5. Run `make check` (quality gate)

**API Endpoints:**

```python
# GET /api/datasets/{dataset_id}/enrichments/pending
class PendingSuggestion(BaseModel):
    patch_id: str
    operation: str  # SET_LABEL, SET_DESCRIPTION, etc.
    column: str
    suggested_value: str
    current_value: str | None
    confidence: float
    model_id: str

class PendingResponse(BaseModel):
    suggestions: list[PendingSuggestion]
    total: int

# POST /api/datasets/{dataset_id}/enrichments/{patch_id}/accept
class AcceptRequest(BaseModel):
    edited_value: str | None = None  # Optional edit before accept

# POST /api/datasets/{dataset_id}/enrichments/{patch_id}/reject
class RejectRequest(BaseModel):
    reason: str | None = None

# GET /api/datasets/{dataset_id}/enrichments/history
class PatchHistoryResponse(BaseModel):
    patches: list[MetadataPatch]
    total: int

# POST /api/datasets/{dataset_id}/enrichments/generate
class GenerateRequest(BaseModel):
    force_regenerate: bool = False
```

**Key Integration:**
- **Import existing pure Python logic** - API routes wrap existing functions (no reimplementation)
- Reuses `EnrichmentService` from `ui/components/enrichment_integration.py`
- Reuses `prepare_diff_view_data()` from `ui/components/enrichment_panel.py`
- Reuses `load_patch_history()`, `export_to_json()` from `ui/components/patch_history.py`
- Reuses `OverlayStore` for persistence
- Reuses `generate_enrichment_suggestions()` from `llm_enrichment.py`
- No Streamlit dependencies (verified - all imports are pure Python)

**Quality Gate:**

- Tests: 5 new tests
- Coverage: Maintain 67% minimum
- Lint: Pass `make lint`
- Format: Pass `make format-check`

**Commit:** `feat: Phase 3b - Enrichment API routes for ADR011`

- `src/clinical_analytics/api/routes/enrichments.py` (new)
- `src/clinical_analytics/api/models/schemas.py` (add enrichment models)
- `src/clinical_analytics/api/main.py` (register router)
- `tests/api/test_enrichments.py` (new)
- All tests passing: X/Y

---

### Phase 4: Electron App Skeleton

**Goal:** Create minimal Electron app that connects to FastAPI

**Tooling Stack:**

- **Node.js**: 20 LTS
- **Electron**: 28.x (latest stable)
- **Build**: None (vanilla JS, no bundler for simplicity)
- **Linting**: ESLint with `eslint-plugin-electron`
- **Testing**: Vitest (unit), Playwright (E2E)
- **CSS**: Plain CSS with CSS variables
- **Package manager**: npm

**CORS Configuration (REQUIRED BEFORE ELECTRON TESTING):**

Current `src/clinical_analytics/api/main.py` only allows Next.js ports:
```python
# CURRENT (incomplete):
"http://localhost:3000,http://127.0.0.1:3000"
```

**Must update to include Electron dev server ports:**
```python
# UPDATED (add this in Phase 4):
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://localhost:5173",
).split(",")
# Note: Electron in dev mode uses localhost, not file://
```

**TDD Step:** Add test `test_cors_allows_electron_origin` before updating CORS.

**SSE Production Configuration:**

SSE requires HTTP connection. When Electron is packaged:

- **Dev mode**: Uses `http://localhost:8000` (works with current CORS)
- **Production**: Must use one of:
  1. **Embedded local server** (recommended): Electron spawns FastAPI as subprocess
  2. **Custom protocol**: Register `app://` scheme that proxies to local server

```javascript
// main.js - Production SSE handling
const { spawn } = require('child_process');

// Option 1: Spawn FastAPI on startup
const backend = spawn('uvicorn', [
  'clinical_analytics.api.main:app',
  '--port', '8000',
  '--host', '127.0.0.1'
]);

// Renderer always connects to localhost:8000
// Works in both dev and prod
```

**Security Configuration:**

```javascript
// main.js - Electron security best practices
const mainWindow = new BrowserWindow({
  webPreferences: {
    nodeIntegration: false,        // Disable Node in renderer
    contextIsolation: true,        // Enable context isolation
    preload: path.join(__dirname, 'preload.js'),
    sandbox: true,                 // Enable sandbox
  }
});
```

**TDD Steps:**

1. Create `electron/package.json` with dependencies
2. Write test: `test_electron_app_launches` (Playwright)
3. Run `npm run test:e2e` (verify RED)
4. Implement `main.js`, `preload.js`, `renderer/index.html`
5. Run test (verify GREEN)

**Playwright Electron Setup:**

```javascript
// electron.spec.js
const { _electron: electron } = require('@playwright/test');
const { test, expect } = require('@playwright/test');

test('app launches', async () => {
  const app = await electron.launch({ args: ['.'] });
  const window = await app.firstWindow();
  await expect(window).toHaveTitle(/Clinical Analytics/);
  await app.close();
});
```

**Implementation:**

1. Create `electron/` directory structure
2. Set up `package.json` with Electron dependencies
3. Create `main.js` - Window creation, dev tools, backend spawn
4. Create `preload.js` - Expose `clinicalAPI` via contextBridge
5. Create basic `renderer/index.html` with chat layout
6. Verify connection to `http://localhost:8000` (FastAPI)

**Quality Gate:**

- Tests: 1 E2E test (app launches)
- Lint: Pass `npm run lint`
- App connects to FastAPI health endpoint

**Commit:** `feat: Phase 4 - Electron app skeleton with SSE config`

- `electron/package.json`
- `electron/main.js`
- `electron/preload.js`
- `electron/renderer/index.html`
- `electron/renderer/renderer.js`
- `electron/renderer/style.css`
- `electron/tests/app.spec.js`
- `electron/playwright.config.js`
- All tests passing

---

### Phase 5: Chat Interface

**Goal:** Build chat UI component

**TDD Steps:**

1. Write unit tests for chat state management
2. Write E2E test: `test_user_can_send_message`
3. Run tests (verify RED)
4. Implement chat interface
5. Run tests (verify GREEN)

**Implementation:**

1. Create `renderer/renderer.js`:

- Session state management (current chat, messages)
- Dataset selector dropdown
- Chat message rendering (user + assistant)
- Query input with submit handling
- SSE subscription for streaming responses

2. Create `renderer/style.css`:

- Light cream/white theme (similar to Claude)
- User message bubbles (right-aligned)
- Assistant messages (left-aligned, markdown rendered)
- Result cards within messages

**Quality Gate:**

- Tests: 5 total (3 unit tests for state management, 2 E2E tests for UI)
- Lint: Pass `npm run lint`
- Dataset selector populates from API
- Coverage: Electron unit tests at 80%+ for new code

**Commit:** `feat: Phase 5 - Chat interface with dataset selector`

- `electron/renderer/renderer.js` (updated)
- `electron/renderer/style.css` (updated)
- `electron/tests/chat.spec.js`
- All tests passing

---

### Phase 6: Result Rendering in Chat

**Goal:** Render analysis results inline in chat messages

**TDD Steps:**

1. Write unit tests for each renderer function
2. Write E2E test: `test_describe_query_shows_stats_table`
3. Run tests (verify RED)
4. Implement renderers
5. Run tests (verify GREEN)

**Implementation:**

1. Create result renderer functions for each intent type:

- `renderDescriptive()` - Stats table
- `renderComparison()` - Group comparison with significance
- `renderPredictor()` - Odds ratios table
- `renderSurvival()` - Kaplan-Meier description (charts later)
- `renderRelationship()` - Correlation summary

2. Add markdown rendering (marked.js, same as reference)
3. Add collapsible sections for detailed results

**Quality Gate:**

- Tests: 6 total (5 unit tests for renderers, 1 E2E test for integration)
- Lint: Pass `npm run lint`
- All 5 intent types render correctly
- Coverage: Renderer functions at 90%+

**Commit:** `feat: Phase 6 - Result rendering in chat`

- `electron/renderer/renderers.js` (new)
- `electron/renderer/renderer.js` (updated)
- `electron/tests/renderers.spec.js`
- All tests passing

---

### Phase 6b: Enrichment Panel (Electron UI for ADR011 API)

**Goal:** Build Electron UI that consumes `/api/enrichments` routes (which wrap existing pure Python logic)

**Note:** `ui/components/enrichment_panel.py` is pure Python (no Streamlit) and is KEPT - the API wraps it in Phase 3b. This phase builds the Electron frontend.

**Features (matching existing logic):**
- Side-by-side diff view (current vs suggested)
- Batch accept/reject controls
- Confidence indicator per suggestion
- Edit before accept capability

**TDD Steps:**

1. Write unit tests for enrichment state management
2. Write E2E test: `test_user_can_accept_suggestion`
3. Write E2E test: `test_user_can_reject_suggestion`
4. Run tests (verify RED)
5. Implement enrichment panel
6. Run tests (verify GREEN)

**Implementation:**

1. Create `electron/renderer/enrichment-panel.js`:

- Fetch pending suggestions from `/api/datasets/{id}/enrichments/pending`
- Render diff view with before/after comparison
- Accept button → POST to `/api/datasets/{id}/enrichments/{patch_id}/accept`
- Reject button → POST to `/api/datasets/{id}/enrichments/{patch_id}/reject`
- Edit-before-accept modal
- Batch operations (accept all / reject all)

2. Create `electron/renderer/enrichment-panel.css`:

- Diff view styling (green for additions, red for removals)
- Confidence badges
- Accept/reject button styles

**Quality Gate:**

- Tests: 5 total (2 unit tests, 3 E2E tests)
- Lint: Pass `npm run lint`
- All suggestion operations work via API
- Coverage: 80%+ for new code

**Commit:** `feat: Phase 6b - Enrichment panel for Electron`

- `electron/renderer/enrichment-panel.js` (new)
- `electron/renderer/enrichment-panel.css` (new)
- `electron/tests/enrichment.spec.js` (new)
- All tests passing

---

### Phase 6c: Patch History Viewer (Electron UI for ADR011 API)

**Goal:** Build Electron UI that consumes `/api/enrichments/history` routes (which wrap existing pure Python logic)

**Note:** `ui/components/patch_history.py` is pure Python (no Streamlit) and is KEPT - the API wraps it in Phase 3b. This phase builds the Electron frontend.

**Features (matching existing logic):**
- Chronological list of applied patches
- Actor, timestamp, model_id for each patch
- Revert capability per patch
- Export patch history as JSON

**TDD Steps:**

1. Write E2E test: `test_history_shows_applied_patches`
2. Write E2E test: `test_user_can_revert_patch`
3. Run tests (verify RED)
4. Implement history viewer
5. Run tests (verify GREEN)

**Implementation:**

1. Create `electron/renderer/patch-history.js`:

- Fetch history from `/api/datasets/{id}/enrichments/history`
- Render chronological list with timestamps
- Revert button per patch
- Export as JSON button

**Quality Gate:**

- Tests: 3 total E2E tests
- Lint: Pass `npm run lint`
- History displays correctly and revert works

**Commit:** `feat: Phase 6c - Patch history viewer for Electron`

- `electron/renderer/patch-history.js` (new)
- `electron/tests/patch-history.spec.js` (new)
- All tests passing

---

### Phase 7: Session Management

**Goal:** Chat history sidebar

**TDD Steps:**

1. Write E2E test: `test_session_list_shows_history`
2. Write E2E test: `test_click_session_restores_messages`
3. Run tests (verify RED)
4. Implement sidebar
5. Run tests (verify GREEN)

**Implementation:**

1. Create left sidebar component (like reference project)
2. List previous sessions with dataset name
3. Click to load session and restore messages
4. New chat button
5. Delete session button

**Quality Gate:**

- Tests: 3 total E2E tests (list, restore, delete)
- Lint: Pass `npm run lint`
- Sessions persist and restore correctly

**Commit:** `feat: Phase 7 - Session history sidebar`

- `electron/renderer/sidebar.js` (new)
- `electron/renderer/renderer.js` (updated)
- `electron/renderer/style.css` (updated)
- `electron/tests/sessions.spec.js`
- All tests passing

---

### Phase 8: Dataset Upload

**Goal:** Support file upload from Electron

**TDD Steps:**

1. Write backend test: `test_datasets_upload_csv_creates_dataset`
2. Write E2E test: `test_user_can_upload_file`
3. Run tests (verify RED)
4. Implement upload flow
5. Run tests (verify GREEN)

**Implementation:**

1. Add file upload input to UI
2. Implement `POST /api/datasets/upload` backend route
3. Create upload progress indicator (SSE)
4. Auto-select uploaded dataset after success

**Quality Gate:**

- Tests: 4 total (2 backend API tests, 2 E2E tests for upload flow)
- Lint: Pass all linters (`make check` + `npm run lint`)
- CSV and Excel uploads work
- Coverage: Maintain 67%+ backend

**Commit:** `feat: Phase 8 - Dataset upload flow`

- `src/clinical_analytics/api/routes/datasets.py` (updated)
- `electron/renderer/upload.js` (new)
- `tests/api/test_datasets.py` (updated)
- `electron/tests/upload.spec.js`
- All tests passing

---

### Phase 9: Integration Testing

**Goal:** End-to-end verification

**Implementation:**

1. Write E2E tests with Playwright for Electron
2. Test query flow: type question -> see results
3. Test session persistence: refresh -> restore
4. Test dataset selection: switch -> context changes
5. Test upload flow: drop file -> analyze

**Playwright Electron Test Setup:**

```javascript
// playwright.config.js
const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/e2e',
  timeout: 30000,
  use: {
    trace: 'on-first-retry',
  },
});

// tests/e2e/fixtures.js
const { test: base, _electron: electron } = require('@playwright/test');

exports.test = base.extend({
  electronApp: async ({}, use) => {
    const app = await electron.launch({ args: ['.'] });
    await use(app);
    await app.close();
  },
  window: async ({ electronApp }, use) => {
    const window = await electronApp.firstWindow();
    await use(window);
  },
});
```

**Quality Gate:**

- Tests: 5 E2E tests (one per user flow)
- All tests passing with backend running
- No flaky tests

**Commit:** `test: Phase 9 - E2E test suite`

- `electron/tests/e2e/query-flow.spec.js`
- `electron/tests/e2e/session-flow.spec.js`
- `electron/tests/e2e/upload-flow.spec.js`
- `electron/tests/e2e/fixtures.js`
- `electron/playwright.config.js`
- All tests passing

---

### Phase 10a: Delete Streamlit Pages

**Goal:** Remove Streamlit pages from codebase

**Git is the archive.** If rollback needed: `git revert <commit>` or `git checkout HEAD~1 -- path/to/file`

**Files to delete:**

```bash
git rm src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py
git rm src/clinical_analytics/ui/pages/02_📊_Your_Dataset.py
git rm src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py
git rm src/clinical_analytics/ui/pages/20_📊_Descriptive_Stats.py
git rm src/clinical_analytics/ui/pages/21_📈_Compare_Groups.py
git rm src/clinical_analytics/ui/pages/22_🎯_Risk_Factors.py
git rm src/clinical_analytics/ui/pages/23_⏱️_Survival_Analysis.py
git rm src/clinical_analytics/ui/pages/24_🔗_Correlations.py
git rm src/clinical_analytics/ui/app.py
```

**Keep (shared with API):**

- `src/clinical_analytics/ui/storage/` - Used by API
- `src/clinical_analytics/ui/__init__.py`

**Commit:** `chore: Phase 10a - Remove Streamlit pages`

**Smoke Test (before proceeding):**
```bash
make test-fast  # Catch import errors early
```

---

### Phase 10b: Delete Streamlit-only Components

**Goal:** Remove Streamlit UI components (keep pure logic)

| Component | Uses st.* | Action | Reason |
|-----------|-----------|--------|--------|
| `analysis_wizard.py` | Yes | DELETE | Streamlit-specific UI |
| `data_validator.py` | No | KEEP | Pure data validation |
| `dataset_loader.py` | Yes | DELETE | Streamlit-specific UI |
| `question_engine.py` | Yes | DELETE | Streamlit-specific UI |
| `result_interpreter.py` | Yes | DELETE | Has Streamlit imports (verified) |
| `renderers.py` | Yes | DELETE | Has Streamlit imports (missing from original plan) |
| `trust_ui.py` | Yes | DELETE | Streamlit-specific UI |
| `variable_detector.py` | No | KEEP | Pure logic |
| `variable_mapper.py` | Yes | DELETE | Streamlit-specific UI |
| `clarifying_ui.py` | Yes | DELETE | Created in Phase 0 |
| `enrichment_panel.py` | **No** | **KEEP** | Pure Python - reused by API routes (Phase 3b) |
| `patch_history.py` | **No** | **KEEP** | Pure Python - reused by API routes (Phase 3b) |
| `enrichment_integration.py` | **No** | **KEEP** | Pure Python service layer - reused by API routes |

```bash
git rm src/clinical_analytics/ui/components/analysis_wizard.py
git rm src/clinical_analytics/ui/components/dataset_loader.py
git rm src/clinical_analytics/ui/components/question_engine.py
git rm src/clinical_analytics/ui/components/result_interpreter.py  # Has Streamlit imports
git rm src/clinical_analytics/ui/components/renderers.py           # Has Streamlit imports
git rm src/clinical_analytics/ui/components/trust_ui.py
git rm src/clinical_analytics/ui/components/variable_mapper.py
git rm src/clinical_analytics/ui/components/clarifying_ui.py
# KEEP: ADR011 components are pure Python - reused by API routes
# - enrichment_panel.py (used by Phase 3b API)
# - patch_history.py (used by Phase 3b API)
# - enrichment_integration.py (used by Phase 3b API)
git rm src/clinical_analytics/ui/app_utils.py
git rm src/clinical_analytics/ui/helpers.py
git rm src/clinical_analytics/ui/messages.py
git rm src/clinical_analytics/ui/ollama_init.py
```

**Commit:** `chore: Phase 10b - Remove Streamlit-only components`

**Smoke Test (before proceeding):**
```bash
make test-fast  # Catch import errors from component removal
```

---

### Phase 10c: Remove Streamlit Dependency

**Goal:** Remove Streamlit from pyproject.toml entirely

**Changes to pyproject.toml:**

```toml
[project]
dependencies = [
    # DELETE: "streamlit>=1.28.0",
    # ... keep other deps
]
```

**Verification:**

```bash
uv sync
uv run python -c "from clinical_analytics.api.main import app"
```

**Commit:** `chore: Phase 10c - Remove Streamlit dependency`

---

### Phase 10d: Delete Streamlit Tests

**Goal:** Remove Streamlit-specific tests

```bash
git rm -rf tests/ui/pages/
# Review and delete Streamlit-specific component tests
# Keep tests for pure logic components (data_validator, variable_detector)
```

**Verification:**

```bash
make test-fast
make test PYTEST_ARGS="tests/api/ -xvs"
```

**Commit:** `chore: Phase 10d - Remove Streamlit tests`

---

### Phase 10e: Final Cleanup

**Goal:** Update Makefile, README

**Makefile Updates:**

```makefile
# Add:
dev-electron:
	cd electron && npm run dev

build-electron:
	cd electron && npm run build

dev-full:
	make run-api & cd electron && npm run dev

# Remove Streamlit-specific targets (if any)
```

**Verification:**

```bash
# No Streamlit imports remain
grep -r "import streamlit" src/ && echo "FAIL" || echo "OK"

# All tests pass
make test-fast

# Electron launches
cd electron && npm run dev
```

**Commit:** `chore: Phase 10e - Update Makefile for Electron`

- `Makefile` (updated)
- `README.md` (updated)

**Rollback (if needed):**

```bash
# Revert entire cutover
git revert <phase-10a-commit>..<phase-10e-commit>

# Or restore specific files
git checkout HEAD~5 -- src/clinical_analytics/ui/pages/
```

---

## Success Criteria

1. Chat-only interface for all analytics
2. All 6 analysis types work via natural language
3. Dataset selection and upload work
4. Session history persists across restarts
5. SSE streaming shows real-time query progress
6. Zero Streamlit dependencies in production flow
7. Core layer (`src/clinical_analytics/core/`) has zero Streamlit imports
8. Core layer has zero UI layer imports (`from clinical_analytics.ui` = 0 matches)
9. Clean git history with atomic commits per phase (enables `git revert` rollback)
10. Coverage maintained at 67% minimum (`make test-cov-check` passes)

## Out of Scope (Defer)

- Charting/visualization (text-based results first)
- Multi-user authentication
- Cloud deployment (desktop-only for now)
- Export to Word/PNG (keep CSV only initially)

## Rollback Plan

Git is the rollback mechanism.

**If Electron app fails after Phase 10 cutover:**

```bash
# Option 1: Revert all Phase 10 commits
git revert --no-commit <phase-10a-sha>..<phase-10e-sha>
git commit -m "revert: Restore Streamlit UI"

# Option 2: Restore specific files
git checkout <pre-cutover-sha> -- src/clinical_analytics/ui/pages/
git checkout <pre-cutover-sha> -- src/clinical_analytics/ui/components/
git checkout <pre-cutover-sha> -- pyproject.toml

# Then reinstall deps
uv sync
```

**Before cutover (Phases 0-9):** Both UIs can run in parallel:

- FastAPI: `make run-api` (port 8000)
- Streamlit: `make run` (port 8501)
- Electron: `cd electron && npm run dev`

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SSE not working through Electron | Use fetch + ReadableStream; spawn backend as subprocess in prod |
| Complex result rendering | Start text-only, add formatting incrementally |
| CORS issues with Electron | Configure localhost origins, test early in Phase 4 |
| Test flakiness in E2E | Use Playwright's auto-waiting, avoid arbitrary sleeps |
| Core layer still has Streamlit | Phase 0 explicitly cleans this; grep verification in CI |
| Blocking plan not complete | Phase 2 has explicit prerequisite check |
| Cutover breaks something | Atomic commits per phase; rollback via `git revert` |

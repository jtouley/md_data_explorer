---
name: Streamlit to Lightweight UI Refactor
overview: |
  Refactor the clinical analytics platform UI away from Streamlit to a lightweight,
  modern web interface similar to claude-run. Replace Streamlit's state management with
  a proper backend API and a React-based frontend with SSE streaming for real-time updates.

  This migration will provide:
  - Better performance and scalability
  - Modern, responsive UI with dark mode
  - Session persistence and history browsing
  - Collapsible results and better conversation UX
  - Separation of concerns (API backend + web frontend)

  Architecture: FastAPI backend (Python) + Next.js frontend (TypeScript/React)

todos:
  # Phase 1: Architecture Design & Setup
  - id: "1"
    content: Create architecture design document (API endpoints, state management, data flow)
    status: pending
    activeForm: Creating architecture design document

  - id: "1.5"
    content: Document API contracts (request/response schemas) in architecture doc
    status: pending
    activeForm: Documenting API contracts
    dependencies:
      - "1"
    notes: |
      Pydantic models for: SessionCreate, SessionResponse, QueryRequest, QueryResponse,
      DatasetUpload, AnalysisResult, etc.

  - id: "2"
    content: Set up Next.js web frontend structure (Follow claude-run web/ structure: App Router, src/components/, src/lib/)
    status: pending
    activeForm: Setting up Next.js frontend
    dependencies:
      - "1.5"

  - id: "3"
    content: Set up FastAPI backend structure (API routes, models, services)
    status: pending
    activeForm: Setting up FastAPI backend
    dependencies:
      - "1"

  - id: "4"
    content: Add dependencies to pyproject.toml (fastapi, uvicorn, pydantic v2, sse-starlette)
    status: pending
    activeForm: Adding backend dependencies
    dependencies:
      - "3"

  - id: "5"
    content: Create package.json with Next.js 15+, TypeScript, TailwindCSS dependencies
    status: pending
    activeForm: Creating package.json
    dependencies:
      - "2"

  # Phase 1.5: Semantic Layer FastAPI Compatibility (CRITICAL)
  - id: "5.1"
    content: Write failing test for semantic layer in FastAPI route handler (TDD Red)
    status: pending
    activeForm: Writing semantic layer FastAPI test
    dependencies:
      - "4"
    notes: |
      Test that SemanticLayer works with FastAPI Depends() dependency injection

  - id: "5.2"
    content: Verify DuckDB connection pooling works with async context (TDD Green)
    status: pending
    activeForm: Verifying DuckDB async compatibility
    dependencies:
      - "5.1"
    notes: |
      Test async operations don't block, connection reuse works

  - id: "5.3"
    content: Test @st.cache_resource pattern translates to FastAPI Depends() (TDD Green)
    status: pending
    activeForm: Testing FastAPI dependency caching
    dependencies:
      - "5.2"
    notes: |
      Verify semantic layer instance reuse across requests

  - id: "5.4"
    content: Document adapter patterns and run compatibility tests (TDD Refactor)
    status: pending
    activeForm: Documenting semantic layer adapter patterns
    dependencies:
      - "5.3"
    notes: |
      Document any workarounds or patterns needed for FastAPI integration

  # Phase 2: Core Backend API (TDD)
  - id: "6"
    content: Write failing tests for session management API endpoints (TDD Red)
    status: pending
    activeForm: Writing tests for session management
    dependencies:
      - "5.4"

  - id: "7"
    content: Implement session management (create, list, get, delete sessions) (TDD Green)
    status: pending
    activeForm: Implementing session management
    dependencies:
      - "6"

  - id: "8"
    content: Run tests for session management and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring session management
    dependencies:
      - "7"

  - id: "9"
    content: Write failing tests for dataset management API endpoints (TDD Red)
    status: pending
    activeForm: Writing tests for dataset management
    dependencies:
      - "4"

  - id: "10"
    content: Implement dataset management (list, upload, get metadata) (TDD Green)
    status: pending
    activeForm: Implementing dataset management
    dependencies:
      - "9"

  - id: "11"
    content: Run tests for dataset management and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring dataset management
    dependencies:
      - "10"

  - id: "12"
    content: Write failing tests for query/analysis API endpoints (TDD Red)
    status: pending
    activeForm: Writing tests for query API
    dependencies:
      - "4"
      - "7"
      - "10"

  - id: "13"
    content: Implement query API (parse NL query, execute analysis, stream results via SSE) (TDD Green)
    status: pending
    activeForm: Implementing query API
    dependencies:
      - "12"

  - id: "14"
    content: Run tests for query API and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring query API
    dependencies:
      - "13"

  - id: "14.5"
    content: Freeze API contracts - generate TypeScript types from OpenAPI schema, version lock
    status: pending
    activeForm: Freezing API contracts
    dependencies:
      - "14"
    notes: |
      Generate TypeScript types using openapi-typescript or similar
      Prevents frontend/backend drift during parallel development

  # Phase 3: State Adapter Layer (Extract from Streamlit)
  - id: "15"
    content: Write failing tests for ConversationManager (transcript, messages, state) (TDD Red)
    status: pending
    activeForm: Writing tests for ConversationManager
    dependencies:
      - "7"

  - id: "16"
    content: Create ConversationManager class (extract from session_state logic) (TDD Green)
    status: pending
    activeForm: Creating ConversationManager
    dependencies:
      - "15"
    notes: |
      Extract from Ask_Questions.py:
      - normalize_query() (line 141-158)
      - canonicalize_scope() (line 160-200)
      - remember_run() (lifecycle management, lines 231-296)
      - cleanup_old_results() (LRU eviction)
      - State machine logic (lines 1537-1702)

  - id: "17"
    content: Run tests for ConversationManager and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring ConversationManager
    dependencies:
      - "16"

  - id: "18"
    content: Write failing tests for ResultCache (LRU eviction, serialization) (TDD Red)
    status: pending
    activeForm: Writing tests for ResultCache
    dependencies:
      - "7"

  - id: "19"
    content: Create ResultCache class (extract from session_state caching) (TDD Green)
    status: pending
    activeForm: Creating ResultCache
    dependencies:
      - "18"
    notes: |
      Extract from Ask_Questions.py remember_run(), cleanup_old_results()

  - id: "20"
    content: Run tests for ResultCache and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring ResultCache
    dependencies:
      - "19"

  # Phase 4: API Services (Connect Core Logic)
  - id: "21"
    content: Write failing tests for QueryService (wraps QuestionEngine) (TDD Red)
    status: pending
    activeForm: Writing tests for QueryService
    dependencies:
      - "14"
      - "16"

  - id: "22"
    content: Create QueryService class (wraps existing question_engine.py logic) (TDD Green)
    status: pending
    activeForm: Creating QueryService
    dependencies:
      - "21"
    notes: |
      Wrap src/clinical_analytics/ui/components/question_engine.py:
      - QuestionEngine.parse_query() - intent extraction
      - QuestionEngine.execute_with_timeout() - analysis execution
      - Reuse AnalysisContext, AnalysisIntent types

  - id: "23"
    content: Run tests for QueryService and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring QueryService
    dependencies:
      - "22"

  - id: "24"
    content: Write failing tests for InterpretationService (wraps result_interpreter) (TDD Red)
    status: pending
    activeForm: Writing tests for InterpretationService
    dependencies:
      - "14"

  - id: "25"
    content: Create InterpretationService (wraps existing result_interpreter.py logic) (TDD Green)
    status: pending
    activeForm: Creating InterpretationService
    dependencies:
      - "24"
    notes: |
      Reuse src/clinical_analytics/ui/components/result_interpreter.py

  - id: "26"
    content: Run tests for InterpretationService and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring InterpretationService
    dependencies:
      - "25"

  # Phase 5: Frontend Core Components
  - id: "27"
    content: Create conversation list component (Model after claude-run SessionList: search, filter, sort)
    status: pending
    activeForm: Creating conversation list component
    dependencies:
      - "5"
      - "14.5"
    notes: |
      Follow claude-run patterns for session browsing

  - id: "27a"
    content: Write Jest tests for ConversationList component (TDD)
    status: pending
    activeForm: Writing ConversationList tests
    dependencies:
      - "27"
    notes: |
      Test: rendering, search, filter, sort, session selection

  - id: "28"
    content: Create chat interface component (message rendering, input, SSE streaming)
    status: pending
    activeForm: Creating chat interface component
    dependencies:
      - "5"
      - "14.5"
    notes: |
      Reuse rendering logic from Ask_Questions.py render_chat(), render_result()

  - id: "28a"
    content: Write Jest tests for ChatInterface component (TDD)
    status: pending
    activeForm: Writing ChatInterface tests
    dependencies:
      - "28"
    notes: |
      Test: message rendering, user input, SSE streaming, error states

  - id: "29"
    content: Create dataset selector component (dropdown with upload option)
    status: pending
    activeForm: Creating dataset selector component
    dependencies:
      - "5"
      - "14.5"
    notes: |
      Similar to existing dataset_loader.py but in React

  - id: "29a"
    content: Write Jest tests for DatasetSelector component (TDD)
    status: pending
    activeForm: Writing DatasetSelector tests
    dependencies:
      - "29"
    notes: |
      Test: dropdown rendering, dataset selection, upload trigger

  - id: "30"
    content: Create result renderers (descriptive, comparison, predictor, survival, relationship, count)
    status: pending
    activeForm: Creating result renderers
    dependencies:
      - "28a"
    notes: |
      Port from Ask_Questions.py with Recharts for charts:
      - DescriptiveResults: stats summary
      - ComparisonResults: t-test/ANOVA/chi-square
      - PredictorResults: logistic regression, odds ratios
      - SurvivalResults: Kaplan-Meier curves (Recharts LineChart)
      - RelationshipResults: correlation heatmap (custom component)
      - CountResults: grouped counts

  - id: "30a"
    content: Write Jest tests for DescriptiveResults renderer (TDD)
    status: pending
    activeForm: Writing DescriptiveResults tests
    dependencies:
      - "30"

  - id: "30b"
    content: Write Jest tests for ComparisonResults renderer (TDD)
    status: pending
    activeForm: Writing ComparisonResults tests
    dependencies:
      - "30"

  - id: "30c"
    content: Write Jest tests for PredictorResults renderer (TDD)
    status: pending
    activeForm: Writing PredictorResults tests
    dependencies:
      - "30"

  - id: "30d"
    content: Write Jest tests for SurvivalResults renderer (TDD)
    status: pending
    activeForm: Writing SurvivalResults tests
    dependencies:
      - "30"

  - id: "30e"
    content: Write Jest tests for RelationshipResults renderer (TDD)
    status: pending
    activeForm: Writing RelationshipResults tests
    dependencies:
      - "30"

  - id: "30f"
    content: Write Jest tests for CountResults renderer (TDD)
    status: pending
    activeForm: Writing CountResults tests
    dependencies:
      - "30"

  - id: "31"
    content: Create collapsible sections component (for Trust UI, follow-ups, interpretations)
    status: pending
    activeForm: Creating collapsible sections
    dependencies:
      - "28a"
    notes: |
      Similar to claude-run collapsible tool calls

  - id: "31a"
    content: Write Jest tests for CollapsibleSection component (TDD)
    status: pending
    activeForm: Writing CollapsibleSection tests
    dependencies:
      - "31"
    notes: |
      Test: expand/collapse, nested sections, accessibility

  - id: "32"
    content: Create variable selection UI component (for low-confidence queries)
    status: pending
    activeForm: Creating variable selection UI
    dependencies:
      - "28a"
    notes: |
      Port from Ask_Questions.py variable selection logic

  - id: "32a"
    content: Write Jest tests for VariableSelection component (TDD)
    status: pending
    activeForm: Writing VariableSelection tests
    dependencies:
      - "32"
    notes: |
      Test: dropdown population, selection handling, auto-execution

  # Phase 6: Frontend Features & Polish
  - id: "33"
    content: Implement SSE streaming for live query execution updates
    status: pending
    activeForm: Implementing SSE streaming
    dependencies:
      - "32a"
      - "13"

  - id: "33.5"
    content: Write tests for SSE error scenarios (connection drop, timeout, malformed events)
    status: pending
    activeForm: Writing SSE error tests
    dependencies:
      - "33"
    notes: |
      Test: network disconnect, server timeout, partial results, reconnection

  - id: "34"
    content: Implement dark mode toggle and theme persistence
    status: pending
    activeForm: Implementing dark mode
    dependencies:
      - "5"
    notes: |
      Using TailwindCSS dark mode utilities

  - id: "35"
    content: Implement session search and filtering
    status: pending
    activeForm: Implementing session search
    dependencies:
      - "27"

  - id: "36"
    content: Implement result export (CSV, JSON download buttons)
    status: pending
    activeForm: Implementing result export
    dependencies:
      - "30"
    notes: |
      Port from existing download_button logic in Ask_Questions.py

  - id: "37"
    content: Add loading states and error boundaries
    status: pending
    activeForm: Adding loading states
    dependencies:
      - "28"

  # Phase 7: Data Upload & Validation Flow
  - id: "38"
    content: Write failing tests for file upload API endpoint (TDD Red)
    status: pending
    activeForm: Writing tests for file upload
    dependencies:
      - "10"

  - id: "39"
    content: Implement file upload endpoint with validation (TDD Green)
    status: pending
    activeForm: Implementing file upload
    dependencies:
      - "38"
    notes: |
      Reuse logic from pages/01_📤_Add_Your_Data.py

  - id: "40"
    content: Run tests for file upload and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring file upload
    dependencies:
      - "39"

  - id: "41"
    content: Create upload flow component (drag-drop, progress, validation feedback)
    status: pending
    activeForm: Creating upload flow component
    dependencies:
      - "39"

  - id: "42"
    content: Create variable mapping wizard component (interactive schema mapping)
    status: pending
    activeForm: Creating variable mapping wizard
    dependencies:
      - "41"
    notes: |
      Port from components/variable_mapper.py

  # Phase 8: Integration & Migration
  - id: "43a"
    content: Define database contracts (Pydantic schemas for Session, Message, CachedResult)
    status: pending
    activeForm: Defining database contracts
    dependencies:
      - "7"
    notes: |
      Pydantic models defining exact schema for:
      - Session (id, user_id, dataset_id, created_at, updated_at)
      - Message (id, session_id, role, content, run_key, status, created_at)
      - CachedResult (id, session_id, query_hash, result_data, created_at, ttl)

  - id: "43b"
    content: Write Alembic migration with schema validation tests (TDD Red)
    status: pending
    activeForm: Writing Alembic migration
    dependencies:
      - "43a"
    notes: |
      Create initial migration, test schema creation

  - id: "43c"
    content: Verify contract compliance (serialize/deserialize all types) (TDD Green)
    status: pending
    activeForm: Verifying contract compliance
    dependencies:
      - "43b"
    notes: |
      Test that all Pydantic models can be stored and retrieved from DB

  - id: "44"
    content: Write failing tests for persistence layer (TDD Red)
    status: pending
    activeForm: Writing tests for persistence
    dependencies:
      - "43c"

  - id: "45"
    content: Implement persistence layer (session CRUD with SQLAlchemy) (TDD Green)
    status: pending
    activeForm: Implementing persistence layer
    dependencies:
      - "44"

  - id: "46"
    content: Run tests for persistence and fix quality issues (TDD Refactor)
    status: pending
    activeForm: Refactoring persistence layer
    dependencies:
      - "45"

  - id: "47"
    content: Integrate ConversationManager with persistence layer
    status: pending
    activeForm: Integrating ConversationManager with persistence
    dependencies:
      - "17"
      - "45"

  - id: "48"
    content: Document session migration strategy - clean slate launch, Streamlit sessions archived read-only
    status: pending
    activeForm: Documenting session migration
    dependencies:
      - "47"
    notes: |
      Explicitly document: "Migration launches with empty history, existing Streamlit
      sessions archived read-only in archive/streamlit_sessions/"
      Alternative: Build migration script if user feedback demands it

  # Phase 9: API Documentation & Testing
  - id: "49"
    content: Add OpenAPI/Swagger documentation to FastAPI routes
    status: pending
    activeForm: Adding OpenAPI documentation
    dependencies:
      - "14"

  - id: "50"
    content: Write integration tests for end-to-end query flow (upload → query → results)
    status: pending
    activeForm: Writing integration tests
    dependencies:
      - "40"
      - "23"
      - "26"

  - id: "50.5"
    content: Write performance tests for query API (P95 latency <500ms for simple queries)
    status: pending
    activeForm: Writing performance tests
    dependencies:
      - "50"
    notes: |
      Test query execution latency with realistic data volumes
      Verify API response times meet SLA

  - id: "51"
    content: Run full test suite and fix quality issues (make check-fast)
    status: pending
    activeForm: Running full test suite
    dependencies:
      - "50.5"

  # Phase 10: Deployment & Documentation
  - id: "52"
    content: Create Makefile targets for dev server (make dev-web, make dev-api)
    status: pending
    activeForm: Creating Makefile targets
    dependencies:
      - "3"
      - "5"

  - id: "52.5"
    content: Add frontend test targets to Makefile (make test-web, make test-e2e, make test-web-watch)
    status: pending
    activeForm: Adding frontend test targets
    dependencies:
      - "52"
    notes: |
      - make test-web: Run Jest unit tests
      - make test-e2e: Run Playwright E2E tests
      - make test-web-watch: Jest watch mode
      - Integration with make check-fast (run both backend + frontend fast tests)

  - id: "53"
    content: Create Docker Compose setup for local development
    status: pending
    activeForm: Creating Docker Compose setup
    dependencies:
      - "52.5"

  - id: "54"
    content: Update README.md with new setup instructions and architecture diagram
    status: pending
    activeForm: Updating README
    dependencies:
      - "53"

  - id: "55"
    content: Create ADR documenting the Streamlit → Lightweight UI migration
    status: pending
    activeForm: Creating ADR
    dependencies:
      - "54"
    notes: |
      Document rationale, trade-offs, and migration strategy

  - id: "56"
    content: Archive old Streamlit UI code to archive/ directory
    status: pending
    activeForm: Archiving Streamlit code
    dependencies:
      - "51"
    notes: |
      Keep for reference but remove from main codebase

  # Phase 11: Quality Gates & Commit
  - id: "57"
    content: Run make format && make lint-fix on all new code
    status: pending
    activeForm: Running format and lint
    dependencies:
      - "51"

  - id: "58"
    content: Verify all tests passing, run type-check, verify no new security vulnerabilities
    status: pending
    activeForm: Verifying all quality gates
    dependencies:
      - "57"
    notes: |
      - Run: make test-core, make test-ui, make test-analysis, make test-web
      - Run: make type-check (or mypy) and resolve all errors
      - Run: dependency audit (pip-audit or safety)
      - All quality gates must pass before commit

  - id: "59"
    content: Commit changes with comprehensive commit message
    status: pending
    activeForm: Committing changes
    dependencies:
      - "58"
    notes: |
      Format:
      feat: Streamlit to Lightweight UI Refactor

      - Migrate from Streamlit to FastAPI + Next.js architecture
      - Add session persistence with SQLite
      - Implement SSE streaming for real-time updates
      - Create modern React-based conversation UI
      - Port all analysis types and result renderers
      - Add comprehensive test suite (X tests passing)

      All tests passing: X/Y
      Following TDD: Red-Green-Refactor

  - id: "60"
    content: Push to branch and create pull request
    status: pending
    activeForm: Pushing to branch
    dependencies:
      - "59"

---

# Streamlit to Lightweight UI Refactor Plan

## Overview

This plan outlines the migration from Streamlit to a modern, lightweight web interface similar to [claude-run](https://github.com/kamranahmedse/claude-run). The goal is to improve performance, scalability, and user experience while maintaining all existing functionality.

## Current State Analysis

### Existing Streamlit Architecture
- **Main App**: `src/clinical_analytics/ui/app.py` - Dataset selector, legacy mode
- **Key Pages**:
  - `01_📤_Add_Your_Data.py` - Upload and variable mapping
  - `03_💬_Ask_Questions.py` - Main conversational interface (2290 lines)
  - `02_📊_Your_Dataset.py` - Dataset overview
  - `20-24_*.py` - Legacy analysis pages (gated in V1 MVP)

### Heavy Streamlit Dependencies
- `st.session_state` - Mini state machine for workflow
- `st.chat_message()` / `st.chat_input()` - Chat interface
- `@st.cache_resource` - Semantic layer caching (non-picklable DuckDB/Ibis)
- `st.rerun()` - Trigger re-execution
- Numerous widgets (tabs, expanders, columns, metrics, dataframes, etc.)

### Core Business Logic (Reusable)
✅ **Can be extracted and reused**:
- `components/question_engine.py` (786 lines) - Query intent inference
- `components/result_interpreter.py` - Statistical interpretation
- `components/trust_ui.py` - Verification and export
- `components/dataset_loader.py` - Dataset selection logic
- `components/variable_detector.py` - Type detection
- `components/variable_mapper.py` - Schema mapping
- All `src/clinical_analytics/core/` modules (semantic layer, NL engine, etc.)
- All `src/clinical_analytics/analysis/` modules (stats, survival, compute)

## Target Architecture

### Backend: FastAPI (Python)
```
src/clinical_analytics/api/
├── __init__.py
├── main.py                    # FastAPI app entry point
├── routes/
│   ├── sessions.py            # Session management
│   ├── datasets.py            # Dataset upload/list
│   ├── queries.py             # NL query execution (SSE streaming)
│   └── analysis.py            # Analysis endpoints
├── services/
│   ├── conversation_manager.py  # Extract from session_state logic
│   ├── result_cache.py          # LRU caching
│   ├── query_service.py         # Wraps QuestionEngine
│   └── interpretation_service.py # Wraps ResultInterpreter
├── models/
│   ├── session.py             # Pydantic models
│   ├── message.py
│   ├── query.py
│   └── result.py
└── db/
    ├── database.py            # SQLAlchemy setup
    ├── models.py              # ORM models
    └── migrations/
```

### Frontend: Next.js (TypeScript/React)
```
web/
├── package.json
├── tsconfig.json
├── next.config.js
├── tailwind.config.js
├── src/
│   ├── app/
│   │   ├── page.tsx           # Main conversation view
│   │   ├── layout.tsx         # Root layout with theme
│   │   └── api/               # API proxy (optional)
│   ├── components/
│   │   ├── ConversationList.tsx    # Session browser (like claude-run)
│   │   ├── ChatInterface.tsx       # Message rendering + input
│   │   ├── DatasetSelector.tsx     # Dataset dropdown
│   │   ├── VariableMapper.tsx      # Interactive schema mapping
│   │   ├── ResultRenderers/
│   │   │   ├── DescriptiveResults.tsx
│   │   │   ├── ComparisonResults.tsx
│   │   │   ├── PredictorResults.tsx
│   │   │   ├── SurvivalResults.tsx
│   │   │   ├── RelationshipResults.tsx
│   │   │   └── CountResults.tsx
│   │   ├── CollapsibleSection.tsx  # For Trust UI, follow-ups
│   │   └── VariableSelection.tsx   # Low-confidence UI
│   ├── lib/
│   │   ├── api.ts             # API client
│   │   ├── sse.ts             # SSE handling
│   │   └── types.ts           # TypeScript types
│   └── styles/
│       └── globals.css
└── public/
```

## Key Design Decisions

### 1. **Backend Framework: FastAPI**
- **Why**: Python-based (reuse existing logic), async support, OpenAPI docs, SSE support
- **Alternative considered**: Flask (simpler but lacks async), Django (too heavy)

### 2. **Frontend Framework: Next.js**
- **Why**: React-based, excellent developer experience, TypeScript support, SSE-friendly
- **Alternative considered**: Vanilla React (more setup), SvelteKit (different ecosystem)

### 3. **State Management: Server-Side Sessions**
- **Why**: Avoid Streamlit's fragile `session_state`, enable multi-tab browsing
- **Implementation**: SQLite for MVP (easy migration to Postgres later)

### 4. **Real-Time Updates: SSE (Server-Sent Events)**
- **Why**: One-way streaming (server → client), simpler than WebSockets for this use case
- **Usage**: Stream query execution progress, intermediate results

### 5. **Styling: TailwindCSS**
- **Why**: Utility-first, dark mode support, responsive out of box
- **Alternative considered**: Styled Components (more verbose)

### 6. **Caching Strategy**
- **Backend**: Redis (production) or in-memory dict (dev) for result cache
- **Frontend**: React Query for API caching and optimistic updates

### 7. **Plotting Strategy**
- **Why**: Separation of concerns, responsive rendering, no image bandwidth overhead
- **Implementation**:
  - Backend generates plot data as JSON (e.g., `{time: [], survival: [], ci_lower: [], ci_upper: []}`)
  - Frontend renders with Recharts components
  - Survival curves: Backend returns time-series data, frontend renders with LineChart
  - Heatmaps: Backend returns `{matrix: [][], labels: []}`, frontend renders with custom heatmap component
  - Correlation plots: Backend returns correlation coefficients, frontend creates interactive visualizations
- **Alternative considered**: Backend PNG generation (poor UX, not responsive), matplotlib in browser (not possible)
- **Trade-offs**: More frontend code for rendering, but gains interactivity, responsiveness, and customization

## Migration Strategy

### Phase 1: Parallel Development
1. Build new API alongside existing Streamlit UI
2. Extract business logic into reusable services
3. Create frontend components incrementally

### Phase 2: Feature Parity
1. Port all analysis types (DESCRIBE, COMPARE_GROUPS, etc.)
2. Implement all result renderers
3. Add Trust UI and follow-up suggestions
4. Complete upload and variable mapping flow

### Phase 3: Cutover
1. Deploy with feature flag `ENABLE_NEW_UI=true` (default: false for safety)
2. Both UIs available during transition (parallel deployment)
3. Monitor for critical bugs (rollback if >3 critical bugs in 48h)
4. Archive Streamlit code to `archive/streamlit_ui/` after stabilization
5. Update documentation and setup instructions

**Rollback Plan**:
- **Trigger**: >3 critical bugs within 48 hours of cutover
- **Process**:
  1. Set `ENABLE_NEW_UI=false` environment variable
  2. Restart application (reverts to Streamlit UI)
  3. Investigate and fix issues in new UI
  4. Re-deploy new UI when ready
- **Data Safety**: Sessions stored in SQLite during new UI usage, Streamlit session files remain read-only

### Phase 4: Cleanup
1. Remove Streamlit dependencies from pyproject.toml
2. Archive unused Streamlit-specific tests
3. Simplify Makefile (remove streamlit run commands)
4. Remove feature flag after 2 weeks of stable operation

## Testing Strategy

### Backend Tests
- **Unit tests**: All services, managers, and utilities
- **Integration tests**: API endpoints with test database
- **Fixture reuse**: Leverage existing `conftest.py` fixtures (make_semantic_layer, etc.)

### Frontend Tests
- **Component tests**: Jest + React Testing Library
- **E2E tests**: Playwright for critical user flows
- **Visual regression**: Chromatic or Percy (optional)

### TDD Workflow (Per spec-driven.md)
1. Write failing test (Red)
2. Implement feature (Green)
3. Refactor and fix quality (Refactor)
4. Run module test suite
5. Commit with tests

## Dependencies

### Backend (Add to pyproject.toml)
```toml
dependencies = [
  # ... existing deps ...
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.32.0",
  "pydantic>=2.10.0",
  "sqlalchemy>=2.0.0",
  "alembic>=1.14.0",         # DB migrations
  "sse-starlette>=2.0.0",    # SSE support
  "python-multipart>=0.0.9", # File uploads
  "aiosqlite>=0.20.0",       # Async SQLite
]
```

### Frontend (New package.json)
```json
{
  "dependencies": {
    "next": "^15.1.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "typescript": "^5.7.2",
    "tailwindcss": "^3.4.0",
    "@tanstack/react-query": "^5.62.0",
    "recharts": "^2.15.0",      // For charts
    "eventsource-parser": "^1.1.2" // SSE parsing
  },
  "devDependencies": {
    "@types/node": "^22.10.0",
    "@types/react": "^19.0.0",
    "eslint": "^9.17.0",
    "prettier": "^3.4.0"
  }
}
```

## Risk Mitigation

### Risk 1: Loss of Streamlit's Rapid Prototyping
- **Mitigation**: Keep Streamlit code in archive for reference
- **Benefit**: Gain maintainability, testability, and scalability

### Risk 2: Learning Curve for Frontend Stack
- **Mitigation**: Follow claude-run patterns closely, use well-documented tools
- **Benefit**: Modern UI/UX, better performance

### Risk 3: Non-Picklable Objects (DuckDB/Ibis)
- **Mitigation**: Use FastAPI dependency injection for semantic layer instances
- **Current**: Already handled in Streamlit with `@st.cache_resource`

### Risk 4: Breaking Changes During Migration
- **Mitigation**: TDD workflow, comprehensive integration tests
- **Quality gate**: All existing tests must pass before cutover

## Success Criteria

1. ✅ All 8 Streamlit pages ported to new UI
2. ✅ All analysis types working (DESCRIBE, COMPARE_GROUPS, FIND_PREDICTORS, etc.)
3. ✅ Session persistence and browsing
4. ✅ Real-time SSE streaming for query execution
5. ✅ Dark mode support
6. ✅ All existing tests passing + new tests for API/frontend
7. ✅ Documentation updated (README, ADR)
8. ✅ Performance >= Streamlit (faster page loads, no reruns)

## Timeline Estimate

- **Phase 1-2 (Setup)**: ~2-3 days
- **Phase 3-6 (Backend)**: ~5-7 days
- **Phase 7-9 (Frontend Core)**: ~7-10 days
- **Phase 10-11 (Integration)**: ~3-5 days
- **Phase 12-14 (Polish & Deploy)**: ~2-3 days

**Total**: ~19-28 days (depends on scope adjustments)

## References

- [claude-run GitHub](https://github.com/kamranahmedse/claude-run) - UI inspiration
- [FastAPI Docs](https://fastapi.tiangolo.com/) - Backend framework
- [Next.js Docs](https://nextjs.org/docs) - Frontend framework
- Current codebase analysis (from exploration agent above)

## Notes

- This plan follows strict TDD discipline per `spec-driven.md`
- All Makefile commands must be used (never `pytest` directly)
- Fixtures from `tests/conftest.py` must be reused
- Rule of Three: Don't abstract until third instance
- All tests must pass before commit

---

# Staff Engineer Plan Review

**Date**: 2026-01-03
**Reviewer**: Staff Engineer AI
**Plan File**: `.cursor/plans/streamlit_to_lightweight_ui_refactor_4iMKv.plan.md`

## Execution Readiness Decision

**READY WITH CHANGES**

The plan is comprehensive and well-structured, but requires critical updates before execution to prevent rework. Several blocking issues around test infrastructure, semantic layer compatibility, and frontend testing must be addressed first.

## Plan Summary

- Migrates from Streamlit to FastAPI + Next.js architecture with 60 tracked todos across 11 phases
- Scope is appropriate but sequencing has gaps: frontend tests are underspecified, and compatibility verification with existing test infrastructure is missing
- Timeline estimate (19-28 days) is optimistic given the complexity; realistic estimate is 25-35 days with proper testing

## Blocking Issues

### 1. Missing Test Infrastructure Compatibility Phase

**Problem**: Plan assumes existing `conftest.py` fixtures (make_semantic_layer, mock_semantic_layer) will work seamlessly with FastAPI dependency injection, but provides no verification phase.

**Impact**: Will discover incompatibilities during Phase 4 (API Services) causing rework in Phases 2-3.

**Required fix**: Add new phase BEFORE Phase 2:
- **Phase 1.5**: Verify semantic layer compatibility with FastAPI
  - Write failing test for semantic layer in FastAPI route handler
  - Confirm DuckDB connection pooling works with async context
  - Test that `@st.cache_resource` pattern translates to FastAPI `Depends()` pattern
  - Document any required adapter patterns

### 2. Frontend Test Strategy Underspecified

**Problem**: Phase 5-6 create React components but lack corresponding test todos. "Component tests: Jest + React Testing Library" is mentioned in Testing Strategy section but never executed.

**Impact**: Frontend components shipped without tests, violating TDD workflow and project standards.

**Required fix**: Add test todos for EVERY frontend component:
- After todo #27 (ConversationList): Add todo for ConversationList component tests
- After todo #28 (ChatInterface): Add todo for ChatInterface tests
- After todo #29 (DatasetSelector): Add todo for DatasetSelector tests
- After todo #30 (ResultRenderers): Add todo for each renderer's tests (6 renderers = 6 test todos)
- After todo #31 (CollapsibleSection): Add todo for CollapsibleSection tests
- After todo #32 (VariableSelection): Add todo for VariableSelection tests

Format: Follow TDD pattern (Red-Green-Refactor) with explicit test todos.

### 3. Missing Makefile Frontend Test Integration

**Problem**: Plan mentions Jest/Playwright for frontend but doesn't specify Makefile targets. Project standards require `make test-*` commands, not direct tool invocation.

**Impact**: CI/CD breakage, violation of Makefile enforcement rule from `.claude/CLAUDE.md`.

**Required fix**: Add todo in Phase 10 (Deployment):
- **Todo #52.5**: Add frontend test targets to Makefile
  - `make test-web` - Run Jest unit tests
  - `make test-e2e` - Run Playwright E2E tests
  - `make test-web-watch` - Jest watch mode
  - Integration with existing `make check-fast` (run both backend + frontend fast tests)

### 4. Database Schema Not Contract-Validated

**Problem**: Todo #43 creates SQLite schema but lacks contract definition. No explicit validation that schema supports required operations (session CRUD, result caching with LRU eviction, conversation history).

**Impact**: Schema rework after discovering missing columns/indexes in Phase 8.

**Required fix**: Split todo #43 into:
- **#43a**: Define database contracts (Pydantic schemas for Session, Message, CachedResult)
- **#43b**: Write Alembic migration with schema validation tests
- **#43c**: Verify contract compliance (can serialize/deserialize all required types)

### 5. Unclear Matplotlib/Seaborn → Recharts Migration

**Problem**: Current Streamlit UI uses matplotlib/seaborn for plots (e.g., survival curves, correlation heatmaps). Plan mentions "recharts" for frontend but doesn't specify:
- How survival analysis Kaplan-Meier curves will be rendered (matplotlib is Python-based)
- Migration path for existing plot generation logic
- Who owns plotting: backend (generate PNG/SVG) or frontend (send data, render in JS)?

**Impact**: Discovery phase during Phase 6 (Frontend Features), causing architectural rework.

**Required fix**: Add architectural decision to "Key Design Decisions" section:
- **#7: Plotting Strategy**: Backend generates plot data (JSON), frontend renders with Recharts
  - Survival curves: Backend returns `{time: [], survival: [], ci_lower: [], ci_upper: []}`, frontend renders with LineChart
  - Heatmaps: Backend returns `{matrix: [][], labels: []}`, frontend renders with custom heatmap component
  - Rationale: Separation of concerns, responsive rendering, no image bandwidth overhead

## Non-Blocking Feedback

### Phase Boundaries and Validation Points

1. **Phase 2-3 boundary unclear**: When does "API routes exist" vs "API routes are production-ready"? Add explicit validation point after Phase 3 (API tests pass, OpenAPI schema validated).

2. **Phase 6-7 boundary missing UX validation**: After todo #37 (loading states), add validation checkpoint: "UX review with sample data - verify responsiveness, accessibility, error states".

3. **Phase 8 (Integration) should precede Phase 9 (Documentation)**: Documenting APIs before integration is tested risks outdated docs. Swap order of Phase 8 and Phase 9.

### Rollback and Migration Safety

1. **No rollback plan specified**: If migration fails mid-cutover (Phase 3 of Migration Strategy), what's the rollback path? Add:
   - Feature flag: `ENABLE_NEW_UI` env var (default: false)
   - Parallel deployment: Both UIs available during transition
   - Rollback criteria: If >3 critical bugs in 48h, revert to Streamlit

2. **Session migration is marked "Optional" (todo #48)**: This is optimistic. Users will expect conversation history to persist. Either:
   - Make it required and add tests
   - OR document explicitly: "Migration launches with empty history, existing Streamlit sessions archived read-only"

### Observability and Testing Gaps

1. **No API performance testing**: Plan validates functionality but not latency. Add todo after #50 (integration tests):
   - **#50.5**: Write performance tests for query API (P95 latency <500ms for simple queries)

2. **Missing SSE error handling tests**: Todo #33 implements SSE streaming but doesn't test failure modes (network disconnect, timeout, partial results). Add after #33:
   - **#33.5**: Test SSE error scenarios (connection drop, server timeout, malformed events)

3. **No observability for semantic layer in FastAPI**: Streamlit logs are configured, but plan doesn't specify how structured logging (structlog) integrates with FastAPI. Add note to todo #3:
   - FastAPI middleware for request logging with correlation IDs
   - Bind context: `session_id`, `dataset_id`, `query_hash`

### Quality Gates

1. **Phase completion criteria not explicit**: Each phase should have clear "Done" criteria. Add to plan structure:
   - Phase 2 Done: All API routes return 200/201, OpenAPI docs generated, unit tests pass
   - Phase 5 Done: All components render without errors, Storybook deployed (optional), basic interaction tests pass
   - Phase 8 Done: End-to-end flow (upload → query → results → export) works, integration tests pass

2. **Missing pre-merge checklist**: Todo #58 verifies tests pass but doesn't check:
   - No new type errors (`make type-check` or mypy)
   - No new security vulnerabilities (dependency audit)
   - Add to todo #58: "Run `make type-check` and resolve all errors"

## Spec-Driven Execution Check

### Input/Output Clarity
**Assessment**: Mostly clear, but gaps exist.

- ✅ Backend API contracts: Implied by FastAPI patterns but not explicitly documented
- ⚠️ Frontend component props: Not specified; will cause ad-hoc decisions during implementation
- ❌ Database schema: Underspecified (see Blocking Issue #4)

**Fix**: Add todo #1.5: "Document API contracts (request/response schemas) in architecture doc"

### Success Criteria and Quality Gates
**Assessment**: Defined at plan level but missing per-phase validation.

- ✅ Overall success criteria clear (8 criteria listed)
- ⚠️ Phase-level gates missing (see Non-Blocking Feedback: Quality Gates)

**Fix**: Add "Done Criteria" field to each phase in todos YAML

### Incremental Execution
**Assessment**: Can proceed incrementally with one exception.

- ✅ Backend phases independent (2-4 can run in parallel with adjustments)
- ❌ Frontend phases tightly coupled: Components (Phase 5) depend on exact API contracts from Phase 2-3, but contracts not frozen

**Fix**: Add todo #14.5: "Freeze API contracts - generate TypeScript types from OpenAPI schema, version lock"

### Test Requirements
**Assessment**: TDD workflow specified but inconsistently applied.

- ✅ Backend follows Red-Green-Refactor (every feature has test todo)
- ❌ Frontend lacks test todos (see Blocking Issue #2)

**Fix**: Apply TDD pattern to all frontend todos

### Makefile Command Usage
**Assessment**: Backend compliant, frontend unclear.

- ✅ Backend: `make test-core`, `make test-ui` specified
- ❌ Frontend: No Makefile targets defined (see Blocking Issue #3)

**Fix**: Add frontend Makefile targets (todo #52.5)

### Ambiguity Flags

1. **"Reuse logic from Ask_Questions.py"** (multiple todos): Not specific enough. Which functions? Which state machine logic? Add explicit function references:
   - Todo #16: Extract `normalize_query()`, `canonicalize_scope()`, `remember_run()`, `cleanup_old_results()`
   - Todo #22: Wrap `QuestionEngine.parse_query()`, `QuestionEngine.execute_with_timeout()`

2. **"Port from Ask_Questions.py"** (todo #30): 6 render functions with different dependencies (matplotlib, pandas, interpretation logic). Specify for each renderer:
   - Which backend API provides data?
   - Which Recharts component to use?
   - How to handle edge cases (empty data, missing columns)?

3. **"Following claude-run patterns"** (todo #2, #27): Vague. Add reference to specific claude-run files:
   - Todo #2: "Follow claude-run web/ structure (App Router, src/components/, src/lib/)"
   - Todo #27: "Model after claude-run SessionList component (search, filter, sort)"

## Update Instructions

### Critical (Must Fix Before Execution):

1. **Add Phase 1.5**: Semantic layer FastAPI compatibility verification (insert between todos #1 and #2)
2. **Add frontend test todos**: Apply TDD pattern to Phase 5-6 (add 10+ test todos after component implementation todos)
3. **Add Makefile frontend targets**: Todo #52.5 after todo #52
4. **Split todo #43**: Database schema into contracts (#43a), migration (#43b), validation (#43c)
5. **Add Plotting Strategy decision**: Document backend-generates-data / frontend-renders-charts pattern in "Key Design Decisions"

### Important (Fix Before Phase Execution):

6. **Add todo #1.5**: Document API contracts with request/response schemas
7. **Add todo #14.5**: Freeze API contracts and generate TypeScript types
8. **Add todo #50.5**: API performance tests (P95 latency)
9. **Add todo #33.5**: SSE error scenario tests
10. **Add rollback plan**: Feature flag and parallel deployment strategy in Migration Strategy section

### Nice-to-Have (Improves Execution Confidence):

11. **Add "Done Criteria"** to each phase in todos YAML (format: `done_criteria: "All tests pass, docs updated"`)
12. **Make todo #48 explicit**: Session migration either required or documented as "clean slate"
13. **Swap Phase 8 and 9**: Integration before documentation
14. **Add correlation IDs**: Structured logging with session/dataset/query context in todo #3
15. **Clarify "reuse/port" todos**: Add explicit function references and target components

## Plan Structure Analysis

- **Total phases**: 11 phases
- **Total todos**: 60 todos
- **Dependencies mapped**: ✅ Yes (all todos have dependencies field)
- **Test coverage specified**: ⚠️ Partial (backend complete, frontend missing)
- **Quality gates defined**: ⚠️ Overall yes, per-phase no

## Recommended Next Steps

1. Apply critical fixes (#1-5 above)
2. Re-run /plan-review to verify fixes
3. Execute Phase 1 (Architecture design doc) - this will surface remaining ambiguities
4. Run /spec-driven only after architecture doc is reviewed and approved

## Overall Assessment

This is a well-researched, ambitious plan with clear understanding of the problem space. The current state analysis is excellent (comprehensive Streamlit dependency audit). However, execution readiness is blocked by test infrastructure gaps and frontend test omissions.

The plan author clearly knows the codebase and has identified the right extraction points (ConversationManager, ResultCache, QuestionEngine). The TDD discipline is strong for backend but needs to be applied consistently to frontend.

With the critical fixes above, this plan will execute cleanly following spec-driven workflow.

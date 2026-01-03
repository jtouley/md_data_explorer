# Lightweight UI Architecture Design

**Status**: Design Phase
**Date**: 2026-01-03
**Related**: ADR010 (Streamlit to Lightweight UI Migration)

## Overview

This document defines the architecture for the new lightweight web UI that replaces Streamlit. The architecture follows API-first principles with a FastAPI backend and Next.js frontend, maintaining all existing functionality while improving performance, scalability, and user experience.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            Browser (Next.js App)                          │
│                                                                            │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐ │
│  │  ConversationList  │  │   ChatInterface    │  │  DatasetSelector   │ │
│  │  (Session Browser) │  │   (Main View)      │  │   (Dropdown)       │ │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘ │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │             Result Renderers (6 analysis types)                      │ │
│  │  Descriptive │ Comparison │ Predictor │ Survival │ Relationship      │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐ │
│  │  UploadFlow        │  │  VariableMapper    │  │  CollapsibleSection│ │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/REST + SSE
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend (Python)                           │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐│
│  │                         API Routes                                     ││
│  │  /api/sessions  │  /api/datasets  │  /api/queries  │  /api/analysis   ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                                    │                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐│
│  │                         Services Layer                                 ││
│  │  ConversationManager │ ResultCache │ QueryService │ InterpretationSvc ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                                    │                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐│
│  │                      Persistence Layer                                 ││
│  │  SQLAlchemy │ Alembic Migrations │ SQLite (dev) / Postgres (prod)    ││
│  └──────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Direct imports (no changes)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Existing Core Business Logic                           │
│                                                                            │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐│
│  │   QuestionEngine     │  │  ResultInterpreter   │  │  SemanticLayer  ││
│  │  (NL query parsing)  │  │  (Plain English)     │  │  (DuckDB/Ibis)  ││
│  └──────────────────────┘  └──────────────────────┘  └─────────────────┘│
│                                                                            │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐│
│  │  Stats/Survival      │  │  Dataset Loaders     │  │  Trust UI Logic ││
│  │  (analysis modules)  │  │  (uploaded, MIMIC)   │  │  (patient export│
│  └──────────────────────┘  └──────────────────────┘  └─────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Session Management

#### `POST /api/sessions`
Create a new conversation session.

**Request**:
```json
{
  "dataset_id": "upload_abc123",
  "metadata": {
    "browser": "Chrome 120",
    "device": "desktop"
  }
}
```

**Response** (201 Created):
```json
{
  "session_id": "sess_xyz789",
  "dataset_id": "upload_abc123",
  "created_at": "2026-01-03T10:30:00Z",
  "updated_at": "2026-01-03T10:30:00Z",
  "message_count": 0
}
```

#### `GET /api/sessions`
List all sessions, optionally filtered.

**Query params**:
- `dataset_id`: Filter by dataset
- `search`: Search in conversation history
- `limit`: Max sessions to return (default: 50)
- `offset`: Pagination offset

**Response** (200 OK):
```json
{
  "sessions": [
    {
      "session_id": "sess_xyz789",
      "dataset_id": "upload_abc123",
      "dataset_name": "Patient Cohort 2025",
      "created_at": "2026-01-03T10:30:00Z",
      "updated_at": "2026-01-03T11:45:00Z",
      "message_count": 12,
      "last_message_preview": "What is the average age?"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

#### `GET /api/sessions/{session_id}`
Get session details with full conversation history.

**Response** (200 OK):
```json
{
  "session_id": "sess_xyz789",
  "dataset_id": "upload_abc123",
  "created_at": "2026-01-03T10:30:00Z",
  "updated_at": "2026-01-03T11:45:00Z",
  "messages": [
    {
      "message_id": "msg_001",
      "role": "user",
      "content": "What is the average age?",
      "created_at": "2026-01-03T10:31:00Z"
    },
    {
      "message_id": "msg_002",
      "role": "assistant",
      "content": "The average age is 45.2 years.",
      "run_key": "run_abc123",
      "status": "completed",
      "confidence": 0.92,
      "intent": "DESCRIBE",
      "created_at": "2026-01-03T10:31:02Z"
    }
  ]
}
```

#### `DELETE /api/sessions/{session_id}`
Delete a session and all its messages.

**Response** (204 No Content)

### Dataset Management

#### `GET /api/datasets`
List available datasets.

**Response** (200 OK):
```json
{
  "datasets": [
    {
      "dataset_id": "upload_abc123",
      "dataset_name": "Patient Cohort 2025",
      "dataset_type": "uploaded",
      "upload_timestamp": "2026-01-02T14:20:00Z",
      "row_count": 1250,
      "column_count": 18,
      "has_outcome": true
    }
  ]
}
```

#### `POST /api/datasets/upload`
Upload a new dataset (CSV, Excel, SPSS).

**Request** (multipart/form-data):
- `file`: File upload
- `dataset_name`: User-provided name

**Response** (201 Created):
```json
{
  "dataset_id": "upload_abc123",
  "dataset_name": "Patient Cohort 2025",
  "status": "processing",
  "upload_timestamp": "2026-01-03T12:00:00Z"
}
```

**SSE Stream** (`/api/datasets/upload/{upload_id}/progress`):
```
event: progress
data: {"status": "validating", "progress": 0.2, "message": "Validating file format"}

event: progress
data: {"status": "detecting_variables", "progress": 0.5, "message": "Detecting variable types"}

event: complete
data: {"status": "ready", "progress": 1.0, "dataset_id": "upload_abc123"}
```

#### `GET /api/datasets/{dataset_id}`
Get dataset metadata and schema.

**Response** (200 OK):
```json
{
  "dataset_id": "upload_abc123",
  "dataset_name": "Patient Cohort 2025",
  "row_count": 1250,
  "columns": [
    {
      "name": "age",
      "type": "numeric",
      "nullable": false,
      "stats": {"min": 18, "max": 95, "mean": 45.2}
    },
    {
      "name": "gender",
      "type": "categorical",
      "nullable": false,
      "values": ["M", "F"],
      "distribution": {"M": 620, "F": 630}
    }
  ],
  "semantic_layer_available": true
}
```

### Query/Analysis API

#### `POST /api/queries`
Execute a natural language query.

**Request**:
```json
{
  "session_id": "sess_xyz789",
  "dataset_id": "upload_abc123",
  "query_text": "compare blood pressure between males and females",
  "context": {
    "previous_intent": "DESCRIBE",
    "previous_variables": ["age", "blood_pressure"]
  }
}
```

**Response** (202 Accepted):
```json
{
  "query_id": "query_def456",
  "status": "processing",
  "stream_url": "/api/queries/query_def456/stream"
}
```

**SSE Stream** (`/api/queries/{query_id}/stream`):
```
event: parsing
data: {"status": "parsing", "progress": 0.1, "message": "Parsing natural language query"}

event: intent_detected
data: {"intent": "COMPARE_GROUPS", "confidence": 0.89, "variables": {"metric": "blood_pressure", "grouping": "gender"}}

event: executing
data: {"status": "executing", "progress": 0.5, "message": "Running statistical analysis"}

event: result
data: {
  "status": "completed",
  "result": {
    "type": "comparison",
    "test_type": "t_test",
    "groups": {
      "M": {"mean": 128.5, "std": 15.2, "n": 620},
      "F": {"mean": 122.3, "std": 14.8, "n": 630}
    },
    "p_value": 0.001,
    "significant": true,
    "interpretation": "Males have significantly higher blood pressure than females (p<0.001)."
  },
  "run_key": "run_ghi789",
  "confidence": 0.89
}

event: follow_ups
data: {
  "suggestions": [
    "Does age affect this difference?",
    "What about medication use between groups?"
  ]
}
```

#### `GET /api/queries/{query_id}`
Get query result (for completed queries).

**Response** (200 OK):
```json
{
  "query_id": "query_def456",
  "session_id": "sess_xyz789",
  "query_text": "compare blood pressure between males and females",
  "status": "completed",
  "result": { /* same as SSE event: result data */ },
  "created_at": "2026-01-03T10:31:00Z",
  "completed_at": "2026-01-03T10:31:02Z"
}
```

### Analysis Results API

#### `GET /api/analysis/results/{run_key}`
Get detailed analysis result by run_key.

**Response** (200 OK):
```json
{
  "run_key": "run_ghi789",
  "result_type": "comparison",
  "data": { /* full result data with charts */ },
  "interpretation": "Males have significantly higher blood pressure...",
  "trust_info": {
    "query_plan": { /* raw QueryPlan object */ },
    "alias_resolved_plan": { /* resolved aliases */ },
    "patient_count": 1250,
    "export_available": true
  }
}
```

#### `POST /api/analysis/export/{run_key}`
Export patient-level data for a result.

**Request**:
```json
{
  "format": "csv",
  "max_rows": 100
}
```

**Response** (200 OK, text/csv):
```csv
patient_id,gender,blood_pressure,age
001,M,130,45
002,F,118,52
...
```

## Pydantic Models (API Contracts)

### Request Models

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class SessionCreate(BaseModel):
    """Request to create a new session."""
    dataset_id: str = Field(..., description="Dataset ID to use for this session")
    metadata: Optional[dict] = Field(None, description="Optional client metadata")

class QueryRequest(BaseModel):
    """Request to execute a natural language query."""
    session_id: str = Field(..., description="Session ID for context")
    dataset_id: str = Field(..., description="Dataset to query")
    query_text: str = Field(..., min_length=1, description="Natural language query")
    context: Optional[dict] = Field(None, description="Previous conversation context")

class DatasetUpload(BaseModel):
    """Metadata for dataset upload."""
    dataset_name: str = Field(..., min_length=1, max_length=200)
    # File handled separately via multipart/form-data

class AnalysisExportRequest(BaseModel):
    """Request to export patient-level data."""
    format: Literal["csv", "json"] = "csv"
    max_rows: int = Field(100, ge=1, le=1000, description="Max rows to export")
```

### Response Models

```python
class SessionResponse(BaseModel):
    """Response containing session details."""
    session_id: str
    dataset_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int

class MessageResponse(BaseModel):
    """Response for a single message."""
    message_id: str
    role: Literal["user", "assistant"]
    content: str
    run_key: Optional[str] = None
    status: Literal["pending", "completed", "error"]
    confidence: Optional[float] = None
    intent: Optional[str] = None
    created_at: datetime

class QueryResponse(BaseModel):
    """Response for query execution."""
    query_id: str
    status: Literal["processing", "completed", "error"]
    stream_url: str

class AnalysisResult(BaseModel):
    """Generic analysis result container."""
    result_type: Literal["descriptive", "comparison", "predictor", "survival", "relationship", "count"]
    data: dict  # Type-specific result data
    interpretation: Optional[str] = None
    confidence: float
    run_key: str

class DatasetResponse(BaseModel):
    """Response containing dataset metadata."""
    dataset_id: str
    dataset_name: str
    dataset_type: Literal["uploaded", "mimic3", "sepsis", "covid_ms"]
    row_count: int
    column_count: int
    has_outcome: bool
    upload_timestamp: Optional[datetime] = None
```

### Internal Models (Database)

```python
from sqlalchemy import Column, String, Integer, DateTime, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Session(Base):
    """Database model for conversation sessions."""
    __tablename__ = "sessions"

    session_id = Column(String(50), primary_key=True)
    dataset_id = Column(String(100), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    metadata = Column(Text, nullable=True)  # JSON

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    """Database model for conversation messages."""
    __tablename__ = "messages"

    message_id = Column(String(50), primary_key=True)
    session_id = Column(String(50), ForeignKey("sessions.session_id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    run_key = Column(String(100), nullable=True)
    status = Column(String(20), nullable=False)  # "pending", "completed", "error"
    confidence = Column(Float, nullable=True)
    intent = Column(String(50), nullable=True)
    created_at = Column(DateTime, nullable=False)

    session = relationship("Session", back_populates="messages")

class CachedResult(Base):
    """Database model for cached query results."""
    __tablename__ = "cached_results"

    cache_id = Column(String(50), primary_key=True)
    session_id = Column(String(50), ForeignKey("sessions.session_id"), nullable=False)
    query_hash = Column(String(64), nullable=False, index=True)  # SHA256 hash
    result_data = Column(Text, nullable=False)  # JSON
    created_at = Column(DateTime, nullable=False)
    ttl_seconds = Column(Integer, nullable=False, default=3600)  # 1 hour default
```

## State Management

### Server-Side Sessions

All conversation state is stored server-side in the database:

1. **Session Creation**: When user starts conversation, `POST /api/sessions` creates session record
2. **Message Persistence**: Each query/response is stored as a Message record
3. **Result Caching**: Analysis results cached by query_hash for fast retrieval
4. **LRU Eviction**: ResultCache service manages LRU eviction (max 5 results per session)
5. **Session Cleanup**: Inactive sessions deleted after 30 days

**State Lifecycle**:
```
User → Create Session → Execute Query → Store Message → Cache Result
                                        ↓
                              Update Session.updated_at
                                        ↓
                              [If cache full] → LRU Evict Oldest
```

### Frontend State (React Query)

Frontend uses React Query for API caching and state management:

```typescript
// Example: Session list query
const { data: sessions } = useQuery({
  queryKey: ['sessions'],
  queryFn: () => fetch('/api/sessions').then(r => r.json()),
  staleTime: 5 * 60 * 1000, // 5 minutes
});

// Example: Query execution with SSE
const executeMutation = useMutation({
  mutationFn: (queryRequest: QueryRequest) =>
    fetch('/api/queries', {
      method: 'POST',
      body: JSON.stringify(queryRequest)
    }).then(r => r.json()),
  onSuccess: (data) => {
    // Open SSE stream for real-time updates
    const eventSource = new EventSource(data.stream_url);
    eventSource.onmessage = (event) => {
      // Update UI with streaming progress
    };
  }
});
```

## Data Flow

### Query Execution Flow

```
1. User enters query in ChatInterface
   ↓
2. Frontend sends POST /api/queries
   ↓
3. FastAPI route validates request (Pydantic)
   ↓
4. QueryService.execute_query() called
   ↓
5. QuestionEngine.parse_query() extracts intent
   ↓
6. Check ResultCache for cached result (by query_hash)
   ↓
7a. [Cache hit] → Return cached result immediately
7b. [Cache miss] → Execute via SemanticLayer
   ↓
8. SemanticLayer generates SQL via Ibis
   ↓
9. DuckDB executes query
   ↓
10. ResultInterpreter generates plain-English explanation
   ↓
11. Store result in ResultCache (LRU eviction if needed)
   ↓
12. Stream result back to frontend via SSE
   ↓
13. Frontend renders result with appropriate renderer
```

### Upload Flow

```
1. User drags file to UploadFlow component
   ↓
2. Frontend sends POST /api/datasets/upload (multipart)
   ↓
3. FastAPI route validates file (size, format, security)
   ↓
4. DatasetLoader validates schema
   ↓
5. VariableDetector infers column types
   ↓
6. Stream progress via SSE to frontend
   ↓
7. User maps variables in VariableMapper component
   ↓
8. Dataset persisted to storage (DuckDB + Delta Lake)
   ↓
9. SemanticLayer initialized for dataset
   ↓
10. Frontend redirects to ChatInterface with new dataset selected
```

## Component Architecture

### Backend Services

#### ConversationManager
**Purpose**: Manages conversation lifecycle and state.

**Responsibilities**:
- Create/retrieve sessions
- Store messages
- Normalize queries (lowercase, whitespace collapse)
- Canonicalize semantic scope for hashing
- Track conversation history for context

**Extracted from**: `Ask_Questions.py` lines 141-296, 1537-1702

#### ResultCache
**Purpose**: LRU cache for query results.

**Responsibilities**:
- Store results by query_hash
- LRU eviction (max 5 per session)
- TTL management (default 1 hour)
- Cleanup orphaned results

**Extracted from**: `Ask_Questions.py` `remember_run()`, `cleanup_old_results()`

#### QueryService
**Purpose**: Wraps QuestionEngine for API usage.

**Responsibilities**:
- Parse natural language queries
- Execute analyses via SemanticLayer
- Stream progress via SSE
- Generate follow-up suggestions

**Wraps**: `question_engine.py` - `QuestionEngine.parse_query()`, `execute_with_timeout()`

#### InterpretationService
**Purpose**: Wraps ResultInterpreter for API usage.

**Responsibilities**:
- Generate plain-English interpretations
- Explain statistical significance
- Provide context-aware explanations

**Wraps**: `result_interpreter.py` - `ResultInterpreter.interpret()`

### Frontend Components

#### ConversationList
**Purpose**: Browse all conversation sessions.

**Features**:
- List sessions sorted by recency
- Search by query text or dataset name
- Filter by dataset
- Click to open session

**Similar to**: claude-run SessionList component

#### ChatInterface
**Purpose**: Main conversation view.

**Features**:
- Display message history
- Text input for queries
- SSE streaming for live updates
- Render results with type-specific renderers

**Renders**: User messages, assistant messages with results

#### ResultRenderers (6 types)
**Purpose**: Render each analysis type with appropriate visualization.

1. **DescriptiveResults**: Stats summary (mean, median, std, quartiles)
2. **ComparisonResults**: T-test/ANOVA/Chi-square with interpretation
3. **PredictorResults**: Logistic regression with odds ratios
4. **SurvivalResults**: Kaplan-Meier curves (Recharts LineChart)
5. **RelationshipResults**: Correlation heatmap (custom component)
6. **CountResults**: Grouped counts with optional "most" extraction

**Data flow**: Backend sends JSON plot data → Frontend renders with Recharts

#### DatasetSelector
**Purpose**: Dropdown to select active dataset.

**Features**:
- List available datasets
- Show dataset metadata (row count, upload date)
- Trigger upload flow

#### UploadFlow
**Purpose**: Drag-drop file upload with progress.

**Features**:
- File validation (size, format)
- Progress bar (SSE-driven)
- Error handling with clear messages

#### VariableMapper
**Purpose**: Interactive schema mapping wizard.

**Features**:
- Select patient ID column
- Select outcome variable
- Map time variables (optional)
- Validate mappings before save

## Integration with Existing Code

### No Changes Required

The following modules are used **directly** without modification:

- `src/clinical_analytics/core/` - All modules (semantic layer, NL engine, etc.)
- `src/clinical_analytics/analysis/` - Stats, survival, compute modules
- `src/clinical_analytics/datasets/` - Dataset loaders and definitions
- `tests/conftest.py` - All fixtures (reused for API tests)

### Wrapped for API Usage

The following components are **wrapped** by services:

- `ui/components/question_engine.py` → `api/services/query_service.py`
- `ui/components/result_interpreter.py` → `api/services/interpretation_service.py`
- `ui/components/trust_ui.py` → Trust data included in `AnalysisResult` responses

### Extracted and Refactored

The following logic is **extracted** from Streamlit code:

- Session state machine → `ConversationManager` class
- Result caching → `ResultCache` class
- Variable detection logic → Reused in upload endpoint
- Rendering logic → Ported to React components with Recharts

## Security Considerations

### Input Validation

- All API inputs validated with Pydantic schemas
- File uploads: size limits (100MB), allowed formats (CSV, XLSX, SAV)
- SQL injection: prevented via Ibis query builder (no raw SQL)
- Path traversal: prevented via validated dataset IDs

### Authentication & Authorization

**Phase 1 (MVP)**: No authentication (local deployment only)

**Future**: JWT-based auth with user/session isolation

### Rate Limiting

- Per-IP rate limits: 100 req/min (query endpoints)
- Upload rate limit: 10 uploads/hour per IP

### Data Privacy

- Patient data never leaves server
- Exports capped at 100 rows
- Session data auto-deleted after 30 days

## Performance Targets

### API Latency (P95)

- `GET /api/sessions`: <100ms
- `POST /api/queries` (simple): <500ms
- `POST /api/queries` (complex): <2000ms
- `POST /api/datasets/upload`: <5s for 10MB file

### Frontend Performance

- First Contentful Paint: <1.5s
- Time to Interactive: <3s
- SSE event latency: <100ms

## Development Workflow

### Running Locally

```bash
# Terminal 1: Backend
make dev-api
# FastAPI runs on http://localhost:8000

# Terminal 2: Frontend
make dev-web
# Next.js runs on http://localhost:3000
```

### Testing

```bash
# Backend tests
make test-core
make test-ui
make test-analysis

# Frontend tests
make test-web          # Jest unit tests
make test-e2e          # Playwright E2E tests
make test-web-watch    # Jest watch mode

# All tests
make check-fast        # Backend + frontend fast tests
```

### Type Checking

```bash
# Backend
make type-check        # mypy

# Frontend
cd web && npm run type-check  # TypeScript compiler
```

## Deployment

### Development

- SQLite database (file-based)
- In-memory result cache
- No Redis required

### Production

- PostgreSQL database
- Redis for result cache
- Docker Compose orchestration

## Next Steps

1. ✅ **Phase 1 Done**: Architecture design documented
2. **Phase 1.5**: Verify semantic layer FastAPI compatibility
3. **Phase 2**: Implement backend API with TDD
4. **Phase 3**: Extract state management from Streamlit
5. **Phase 4**: Wrap business logic in services
6. **Phase 5-6**: Build frontend components with tests
7. **Phase 7**: Upload flow and validation
8. **Phase 8**: Integration and persistence
9. **Phase 9**: Documentation and testing
10. **Phase 10**: Deployment and cutover

## References

- [ADR010: Streamlit to Lightweight UI Migration](../implementation/ADR/ADR010.md)
- [Plan: Streamlit UI Refactor](../../.cursor/plans/streamlit_to_lightweight_ui_refactor_4iMKv.plan.md)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Next.js Docs](https://nextjs.org/docs)
- [Recharts Docs](https://recharts.org/en-US/)

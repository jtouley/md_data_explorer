"""Pydantic models for API request/response schemas.

These models define the API contracts between frontend and backend.
All models use Pydantic v2 with strict validation.

Reference: docs/architecture/LIGHTWEIGHT_UI_ARCHITECTURE.md
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Session Management Schemas
# ============================================================================


class SessionCreate(BaseModel):
    """Request to create a new conversation session."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="Dataset ID to use for this session")
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Optional client metadata (e.g., user_agent, theme)"
    )


class SessionResponse(BaseModel):
    """Response after creating or retrieving a session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Unique session identifier")
    dataset_id: str = Field(..., description="Dataset associated with this session")
    created_at: datetime = Field(..., description="Session creation timestamp (UTC)")
    updated_at: datetime = Field(..., description="Last update timestamp (UTC)")
    message_count: int = Field(0, description="Number of messages in this session")


class SessionListResponse(BaseModel):
    """Response containing list of sessions."""

    model_config = ConfigDict(extra="forbid")

    sessions: list[SessionResponse] = Field(..., description="List of sessions")
    total: int = Field(..., description="Total number of sessions matching filters")


# ============================================================================
# Query Execution Schemas
# ============================================================================


class QueryRequest(BaseModel):
    """Request to execute a natural language query."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Session ID for conversation context")
    dataset_id: str = Field(..., description="Dataset to query")
    query_text: str = Field(..., min_length=1, description="Natural language query")
    context: Optional[dict[str, Any]] = Field(
        None,
        description="Previous conversation context (e.g., previous_intent, previous_variables)",
    )


class QueryResponse(BaseModel):
    """Response after submitting a query (async processing)."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(..., description="Unique query identifier")
    status: Literal["processing", "completed", "failed"] = Field(
        ..., description="Current query status"
    )
    stream_url: Optional[str] = Field(
        None, description="SSE stream URL for real-time updates"
    )


class QueryResult(BaseModel):
    """Complete query result with data and metadata."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(..., description="Query identifier")
    intent: str = Field(
        ...,
        description="Detected intent (DESCRIBE, COMPARE_GROUPS, FIND_PREDICTORS, etc.)",
    )
    status: Literal["completed", "failed"] = Field(..., description="Query status")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score for intent detection"
    )
    result_data: Optional[dict[str, Any]] = Field(
        None, description="Query result data (structure varies by intent)"
    )
    interpretation: Optional[str] = Field(
        None, description="LLM-generated interpretation of results"
    )
    follow_up_suggestions: list[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )
    error: Optional[str] = Field(None, description="Error message if query failed")
    execution_time_ms: Optional[int] = Field(
        None, description="Query execution time in milliseconds"
    )


# ============================================================================
# Message/Conversation Schemas
# ============================================================================


class Message(BaseModel):
    """Single message in a conversation."""

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(..., description="Unique message identifier")
    session_id: str = Field(..., description="Parent session ID")
    role: Literal["user", "assistant"] = Field(..., description="Message sender role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp (UTC)")
    query_id: Optional[str] = Field(
        None, description="Associated query ID (for assistant messages)"
    )
    status: Literal["pending", "completed", "failed"] = Field(
        ..., description="Message status"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score (for assistant messages)"
    )


class ConversationHistory(BaseModel):
    """Full conversation history for a session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Session identifier")
    dataset_id: str = Field(..., description="Dataset associated with session")
    messages: list[Message] = Field(..., description="Ordered list of messages")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# ============================================================================
# Dataset Management Schemas
# ============================================================================


class DatasetInfo(BaseModel):
    """Information about a loaded dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., description="Human-readable dataset name")
    cohort: str = Field(..., description="Cohort name (patient group)")
    row_count: int = Field(..., ge=0, description="Number of rows in dataset")
    column_count: int = Field(..., ge=0, description="Number of columns in dataset")
    categorical_columns: list[str] = Field(
        default_factory=list, description="List of categorical column names"
    )
    numeric_columns: list[str] = Field(
        default_factory=list, description="List of numeric column names"
    )
    datetime_columns: list[str] = Field(
        default_factory=list, description="List of datetime column names"
    )
    sample_preview: Optional[dict[str, Any]] = Field(
        None, description="Sample data preview (first 5 rows as dict)"
    )


class DatasetListResponse(BaseModel):
    """Response containing list of available datasets."""

    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetInfo] = Field(..., description="List of available datasets")
    total: int = Field(..., description="Total number of datasets")


# ============================================================================
# Upload Flow Schemas (Phase 7)
# ============================================================================


class UploadMetadata(BaseModel):
    """Metadata for dataset upload."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    file_format: Literal["csv", "xlsx", "sav"] = Field(
        ..., description="Detected file format"
    )
    dataset_name: Optional[str] = Field(
        None, description="User-provided dataset name"
    )
    cohort_name: Optional[str] = Field(None, description="User-provided cohort name")


class UploadResponse(BaseModel):
    """Response after successful dataset upload."""

    model_config = ConfigDict(extra="forbid")

    upload_id: str = Field(..., description="Unique upload identifier")
    dataset_id: str = Field(..., description="Created dataset identifier")
    status: Literal["uploaded", "processing", "ready", "failed"] = Field(
        ..., description="Upload processing status"
    )
    preview_url: Optional[str] = Field(
        None, description="URL to preview uploaded data"
    )


class VariableMapping(BaseModel):
    """Variable type mapping for uploaded dataset."""

    model_config = ConfigDict(extra="forbid")

    column_name: str = Field(..., description="Column name in dataset")
    detected_type: Literal["categorical", "numeric", "datetime", "unknown"] = Field(
        ..., description="Auto-detected variable type"
    )
    user_override: Optional[
        Literal["categorical", "numeric", "datetime", "exclude"]
    ] = Field(None, description="User-specified type override")


class VariableMappingRequest(BaseModel):
    """Request to update variable mappings for a dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="Dataset identifier")
    mappings: list[VariableMapping] = Field(
        ..., description="List of variable mappings"
    )


# ============================================================================
# Server-Sent Events (SSE) Schemas
# ============================================================================


class SSEEvent(BaseModel):
    """Server-sent event payload."""

    model_config = ConfigDict(extra="forbid")

    event: Literal[
        "query_started",
        "query_progress",
        "query_completed",
        "query_failed",
        "interpretation_ready",
    ] = Field(..., description="Event type")
    data: dict[str, Any] = Field(..., description="Event-specific data payload")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Event timestamp (UTC)"
    )


# ============================================================================
# Error Response Schema
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response format."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(
        None, description="Additional error context"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp (UTC)"
    )

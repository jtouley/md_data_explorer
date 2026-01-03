"""FastAPI backend for clinical analytics platform.

This module provides a REST API and Server-Sent Events (SSE) interface
to replace the Streamlit UI while reusing all core business logic.

Architecture:
- API routes: Session, dataset, query, conversation endpoints
- Services: Conversation manager, result cache, query service
- Models: Pydantic schemas (API contracts) + SQLAlchemy (DB persistence)
- Dependencies: Semantic layer injection, authentication (future)
"""

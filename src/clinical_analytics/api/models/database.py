"""SQLAlchemy ORM models for database persistence.

These models define the database schema for sessions and messages.
Uses SQLAlchemy 2.0 declarative mapping with type annotations.

Reference: docs/architecture/LIGHTWEIGHT_UI_ARCHITECTURE.md
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Session(Base):
    """Conversation session table.

    Each session represents one conversation thread with a specific dataset.
    Sessions persist across browser refreshes and can be browsed/searched.
    """

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationship: one session has many messages
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.timestamp",
    )

    def __repr__(self) -> str:
        return f"<Session(session_id={self.session_id!r}, dataset_id={self.dataset_id!r})>"


class Message(Base):
    """Message in a conversation session.

    Stores both user queries and assistant responses with full metadata.
    """

    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("sessions.session_id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    # Query execution metadata (for assistant messages)
    query_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # "pending", "completed", "failed"
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Result data stored as JSON in separate cache (not in DB)
    # Use run_key to look up cached result via ResultCache

    # Relationship: many messages belong to one session
    session: Mapped["Session"] = relationship("Session", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message(message_id={self.message_id!r}, role={self.role!r}, session_id={self.session_id!r})>"

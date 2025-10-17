"""Memory system models for the LLM agent."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlmodel import Field, SQLModel, JSON, Column, Text
from pgvector.sqlalchemy import Vector


class ChatEvent(SQLModel, table=True):
    """Raw chat message events."""
    
    __tablename__ = "chat_events"
    __table_args__ = {"schema": "app"}
    
    event_id: int = Field(default=None, primary_key=True)
    session_id: UUID
    role: str = Field(regex="^(user|assistant|system)$")
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Entity(SQLModel, table=True):
    """Extracted entities from messages and database linking."""
    
    __tablename__ = "entities"
    __table_args__ = {"schema": "app"}
    
    entity_id: int = Field(default=None, primary_key=True)
    session_id: UUID
    name: str
    type: str  # e.g., 'customer', 'order', 'invoice', 'person', 'topic'
    source: str = Field(regex="^(message|db)$")
    external_ref: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Memory(SQLModel, table=True):
    """Memory chunks with vector embeddings."""
    
    __tablename__ = "memories"
    __table_args__ = {"schema": "app"}
    
    memory_id: int = Field(default=None, primary_key=True)
    session_id: UUID
    kind: str = Field(regex="^(episodic|semantic|profile|commitment|todo)$")
    text: str
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(1536)))
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_days: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MemorySummary(SQLModel, table=True):
    """Cross-session memory consolidation summaries."""
    
    __tablename__ = "memory_summaries"
    __table_args__ = {"schema": "app"}
    
    summary_id: int = Field(default=None, primary_key=True)
    user_id: str
    session_window: int
    summary: str
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(1536)))
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Pydantic models for API requests/responses
class ChatRequest(SQLModel):
    """Request model for chat endpoint."""
    user_id: str
    session_id: Optional[UUID] = None
    message: str


class ChatResponse(SQLModel):
    """Response model for chat endpoint."""
    reply: str
    used_memories: List[Dict[str, Any]] = Field(default_factory=list)
    used_domain_facts: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: UUID


class MemoryRequest(SQLModel):
    """Request model for memory endpoint."""
    user_id: str
    k: int = Field(default=10, ge=1, le=100)


class MemoryResponse(SQLModel):
    """Response model for memory endpoint."""
    memories: List[Dict[str, Any]] = Field(default_factory=list)
    summaries: List[Dict[str, Any]] = Field(default_factory=list)


class ConsolidateRequest(SQLModel):
    """Request model for consolidate endpoint."""
    user_id: str


class ConsolidateResponse(SQLModel):
    """Response model for consolidate endpoint."""
    summary_id: int
    message: str


class EntitiesRequest(SQLModel):
    """Request model for entities endpoint."""
    session_id: UUID


class EntitiesResponse(SQLModel):
    """Response model for entities endpoint."""
    entities: List[Dict[str, Any]] = Field(default_factory=list)


class ExplainRequest(SQLModel):
    """Request model for explain endpoint."""
    session_id: UUID
    memory_id: Optional[int] = None


class ExplainResponse(SQLModel):
    """Response model for explain endpoint."""
    explanation: str
    memory_sources: List[Dict[str, Any]] = Field(default_factory=list)
    domain_sources: List[Dict[str, Any]] = Field(default_factory=list)

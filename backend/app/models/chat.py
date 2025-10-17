"""Chat-related models and utilities."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlmodel import Field, SQLModel


class ChatMessage(SQLModel):
    """Individual chat message."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatSession(SQLModel):
    """Chat session with messages."""
    session_id: UUID
    user_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryRetrievalResult(SQLModel):
    """Result of memory retrieval operation."""
    memory_id: int
    text: str
    kind: str
    similarity: float
    importance: float
    created_at: datetime


class DomainFact(SQLModel):
    """Domain fact retrieved from database."""
    table: str
    id: str
    data: Dict[str, Any]
    relevance_score: float


class RetrievalContext(SQLModel):
    """Context for retrieval operations."""
    memories: List[MemoryRetrievalResult] = Field(default_factory=list)
    domain_facts: List[DomainFact] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)


class PromptContext(SQLModel):
    """Context for LLM prompt construction."""
    system_prompt: str
    user_message: str
    memories: List[MemoryRetrievalResult] = Field(default_factory=list)
    domain_facts: List[DomainFact] = Field(default_factory=list)
    conversation_history: List[ChatMessage] = Field(default_factory=list)


class LLMResponse(SQLModel):
    """Response from LLM service."""
    content: str
    usage: Dict[str, Any] = Field(default_factory=dict)
    model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

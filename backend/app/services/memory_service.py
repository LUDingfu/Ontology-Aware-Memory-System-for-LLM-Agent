"""Memory management service."""

import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlmodel import Session, select, func

from app.models.memory import Memory, MemorySummary
from app.models.chat import MemoryRetrievalResult


class MemoryService:
    """Service for managing LLM agent memories."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_memory(
        self,
        session_id: UUID,
        kind: str,
        text: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        ttl_days: Optional[int] = None
    ) -> Memory:
        """Create a new memory."""
        # Check for duplicate content
        content_hash = hashlib.md5(text.encode()).hexdigest()
        existing = self.session.exec(
            select(Memory).where(
                Memory.text == text,
                Memory.session_id == session_id
            )
        ).first()
        
        if existing:
            return existing
        
        memory = Memory(
            session_id=session_id,
            kind=kind,
            text=text,
            embedding=embedding,
            importance=importance,
            ttl_days=ttl_days
        )
        
        self.session.add(memory)
        self.session.commit()
        self.session.refresh(memory)
        
        return memory
    
    def retrieve_memories(
        self,
        query_embedding: List[float],
        user_id: str,
        session_id: Optional[UUID] = None,
        kind: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryRetrievalResult]:
        """Retrieve relevant memories using vector similarity."""
        # Build query
        query = select(Memory)
        
        if session_id:
            query = query.where(Memory.session_id == session_id)
        
        if kind:
            query = query.where(Memory.kind == kind)
        
        # Filter expired memories (simplified - TTL functionality can be added later)
        # For now, we'll skip TTL filtering to avoid SQL complexity
        # query = query.where(
        #     (Memory.ttl_days.is_(None)) |
        #     (Memory.created_at + func.make_interval(days=Memory.ttl_days) > now)
        # )
        
        # Execute query and calculate similarities
        memories = self.session.exec(query).all()
        
        results = []
        for memory in memories:
            if memory.embedding is not None:
                try:
                    # Convert pgvector to list if needed
                    embedding_list = list(memory.embedding) if hasattr(memory.embedding, '__iter__') else memory.embedding
                    if len(embedding_list) > 0:
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(query_embedding, embedding_list)
                        
                        # Weight by importance and recency
                        recency_weight = self._calculate_recency_weight(memory.created_at)
                        final_score = similarity * memory.importance * recency_weight
                        
                        results.append(MemoryRetrievalResult(
                            memory_id=memory.memory_id,
                            text=memory.text,
                            kind=memory.kind,
                            similarity=final_score,
                            importance=memory.importance,
                            created_at=memory.created_at
                        ))
                except Exception as e:
                    print(f"Error processing memory {memory.memory_id}: {e}")
                    continue
        
        # Sort by similarity score and return top results
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]
    
    def consolidate_memories(
        self,
        user_id: str,
        session_window: int = 3
    ) -> MemorySummary:
        """Consolidate memories from recent sessions into a summary."""
        # Get recent memories
        cutoff_date = datetime.utcnow() - timedelta(days=30)  # Last 30 days
        memories = self.session.exec(
            select(Memory).where(Memory.created_at >= cutoff_date)
        ).all()
        
        if not memories:
            return None
        
        # Group by kind and create summary
        summary_parts = []
        for kind in ['episodic', 'semantic', 'profile', 'commitment']:
            kind_memories = [m for m in memories if m.kind == kind]
            if kind_memories:
                summary_parts.append(f"{kind.title()} memories: {len(kind_memories)} items")
        
        summary_text = f"Memory consolidation for user {user_id} covering {session_window} sessions. " + "; ".join(summary_parts)
        
        # Create or update summary
        existing_summary = self.session.exec(
            select(MemorySummary).where(
                MemorySummary.user_id == user_id,
                MemorySummary.session_window == session_window
            )
        ).first()
        
        if existing_summary:
            existing_summary.summary = summary_text
            existing_summary.created_at = datetime.utcnow()
            self.session.commit()
            self.session.refresh(existing_summary)
            return existing_summary
        else:
            summary = MemorySummary(
                user_id=user_id,
                session_window=session_window,
                summary=summary_text
            )
            self.session.add(summary)
            self.session.commit()
            self.session.refresh(summary)
            return summary
    
    def get_user_memories(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Memory]:
        """Get all memories for a user."""
        # This would need to be implemented based on how user_id relates to sessions
        # For now, return recent memories
        memories = self.session.exec(
            select(Memory).order_by(Memory.created_at.desc()).limit(limit)
        ).all()
        
        return memories
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)
    
    def _calculate_recency_weight(self, created_at: datetime) -> float:
        """Calculate recency weight for memory."""
        days_old = (datetime.utcnow() - created_at).days
        return max(0.1, 1.0 - (days_old / 365.0))  # Decay over a year

"""Explain API endpoints (bonus feature)."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.api.deps import get_db
from app.models.memory import ExplainRequest, ExplainResponse, Memory, Entity
from app.services import MemoryService, EntityService

router = APIRouter(prefix="/explain", tags=["explain"])


@router.get("/", response_model=ExplainResponse)
def explain_response(
    session_id: UUID,
    memory_id: Optional[int] = None,
    session: Session = Depends(get_db)
) -> ExplainResponse:
    """
    Explain the sources and reasoning behind a response.
    """
    try:
        memory_service = MemoryService(session)
        entity_service = EntityService(session)
        
        # Get memories used in the session
        memories = session.exec(
            select(Memory).where(Memory.session_id == session_id)
        ).all()
        
        # Get entities detected in the session
        entities = entity_service.get_entities_for_session(session_id)
        
        # Format memory sources
        memory_sources = [
            {
                "memory_id": memory.memory_id,
                "kind": memory.kind,
                "text": memory.text,
                "importance": memory.importance,
                "created_at": memory.created_at.isoformat()
            }
            for memory in memories
        ]
        
        # Format domain sources
        domain_sources = []
        for entity in entities:
            if entity.external_ref:
                domain_sources.append({
                    "entity_name": entity.name,
                    "entity_type": entity.type,
                    "table": entity.external_ref.get("table"),
                    "id": entity.external_ref.get("id"),
                    "source": entity.source
                })
        
        # Generate explanation
        explanation = f"""
        This response was generated using:
        - {len(memory_sources)} memory sources from the knowledge base
        - {len(domain_sources)} domain entities linked to database records
        - Session ID: {session_id}
        
        Memory sources include {len([m for m in memories if m.kind == 'semantic'])} semantic memories, 
        {len([m for m in memories if m.kind == 'episodic'])} episodic memories, and 
        {len([m for m in memories if m.kind == 'profile'])} profile memories.
        """
        
        return ExplainResponse(
            explanation=explanation.strip(),
            memory_sources=memory_sources,
            domain_sources=domain_sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

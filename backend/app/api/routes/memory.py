"""Memory API endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.api.deps import get_db
from app.models.memory import MemoryRequest, MemoryResponse, Memory, MemorySummary
from app.services import MemoryService

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/", response_model=MemoryResponse)
def get_memories(
    user_id: str,
    k: int = 10,
    session: Session = Depends(get_db)
) -> MemoryResponse:
    """
    Get memories and summaries for a user.
    """
    try:
        memory_service = MemoryService(session)
        
        # Get user memories
        memories = memory_service.get_user_memories(user_id, k)
        
        # Get memory summaries
        summaries = session.exec(
            select(MemorySummary).where(MemorySummary.user_id == user_id)
        ).all()
        
        # Format memories
        formatted_memories = [
            {
                "memory_id": memory.memory_id,
                "kind": memory.kind,
                "text": memory.text,
                "importance": memory.importance,
                "created_at": memory.created_at.isoformat()
            }
            for memory in memories
        ]
        
        # Format summaries
        formatted_summaries = [
            {
                "summary_id": summary.summary_id,
                "session_window": summary.session_window,
                "summary": summary.summary,
                "created_at": summary.created_at.isoformat()
            }
            for summary in summaries
        ]
        
        return MemoryResponse(
            memories=formatted_memories,
            summaries=formatted_summaries
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

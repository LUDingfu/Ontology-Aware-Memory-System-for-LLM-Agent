"""Consolidation API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.api.deps import get_db
from app.models.memory import ConsolidateRequest, ConsolidateResponse
from app.services import MemoryService

router = APIRouter(prefix="/consolidate", tags=["consolidate"])


@router.post("/", response_model=ConsolidateResponse)
def consolidate_memories(
    request: ConsolidateRequest,
    session: Session = Depends(get_db)
) -> ConsolidateResponse:
    """
    Consolidate memories from recent sessions into summaries.
    """
    try:
        memory_service = MemoryService(session)
        
        # Consolidate memories
        summary = memory_service.consolidate_memories(
            user_id=request.user_id,
            session_window=3
        )
        
        if not summary:
            raise HTTPException(status_code=404, detail="No memories found to consolidate")
        
        return ConsolidateResponse(
            summary_id=summary.summary_id,
            message=f"Successfully consolidated memories for user {request.user_id}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

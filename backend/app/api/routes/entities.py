"""Entities API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.api.deps import get_db
from app.models.memory import EntitiesRequest, EntitiesResponse, Entity
from app.services import EntityService

router = APIRouter(prefix="/entities", tags=["entities"])


@router.get("/", response_model=EntitiesResponse)
def get_entities(
    session_id: UUID,
    session: Session = Depends(get_db)
) -> EntitiesResponse:
    """
    Get entities detected in a session.
    """
    try:
        entity_service = EntityService(session)
        
        # Get entities for session
        entities = entity_service.get_entities_for_session(session_id)
        
        # Format entities
        formatted_entities = [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "type": entity.type,
                "source": entity.source,
                "external_ref": entity.external_ref,
                "created_at": entity.created_at.isoformat()
            }
            for entity in entities
        ]
        
        return EntitiesResponse(entities=formatted_entities)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

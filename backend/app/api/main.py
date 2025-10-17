from fastapi import APIRouter

from app.api.routes import chat, memory, consolidate, entities, explain
from app.core.config import settings

api_router = APIRouter()

# Health check endpoint
@api_router.get("/health-check/", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ontology-aware-memory-system"}

# Include all API routes
api_router.include_router(chat.router)
api_router.include_router(memory.router)
api_router.include_router(consolidate.router)
api_router.include_router(entities.router)
api_router.include_router(explain.router)

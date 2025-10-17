"""Chat API endpoints."""

import uuid
from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.api.deps import get_db
from app.models.memory import ChatRequest, ChatResponse
from app.models.chat import PromptContext, ChatMessage
from app.services import EmbeddingService, LLMService, RetrievalService, EntityService, MemoryService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    session: Session = Depends(get_db)
) -> ChatResponse:
    """
    Process a chat message and return a response with memory and domain context.
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or uuid.uuid4()
        
        # Initialize services
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        retrieval_service = RetrievalService(session)
        entity_service = EntityService(session)
        memory_service = MemoryService(session)
        
        # Generate embedding for the query
        query_embedding = embedding_service.generate_embedding(request.message)
        if not query_embedding or len(query_embedding) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
        
        # Retrieve relevant context
        context = retrieval_service.retrieve_context(
            query=request.message,
            query_embedding=query_embedding,
            user_id=request.user_id,
            session_id=session_id
        )
        
        # Extract entities from the message
        entities = entity_service.extract_entities(request.message, session_id)
        linked_entities = entity_service.link_entities_to_domain(entities)
        
        # Store entities in database
        for entity in linked_entities:
            session.add(entity)
        session.commit()
        
        # Build prompt context
        prompt_context = PromptContext(
            system_prompt="You are an intelligent business assistant with access to customer data, orders, invoices, and memory.",
            user_message=request.message,
            memories=context.memories,
            domain_facts=context.domain_facts,
            conversation_history=[]  # TODO: Load from database
        )
        
        # Generate LLM response
        llm_response = llm_service.generate_response(prompt_context)
        
        # Extract potential memories from the response
        potential_memories = llm_service.extract_memories_from_response(llm_response.content)
        
        # Store memories
        for memory_text in potential_memories:
            memory_embedding = embedding_service.generate_embedding(memory_text)
            memory_service.create_memory(
                session_id=session_id,
                kind="semantic",
                text=memory_text,
                embedding=memory_embedding,
                importance=0.7
            )
        
        # Store chat event
        from app.models.memory import ChatEvent
        chat_event = ChatEvent(
            session_id=session_id,
            role="user",
            content=request.message
        )
        session.add(chat_event)
        
        assistant_event = ChatEvent(
            session_id=session_id,
            role="assistant",
            content=llm_response.content
        )
        session.add(assistant_event)
        session.commit()
        
        # Format response
        used_memories = [
            {
                "memory_id": memory.memory_id,
                "text": memory.text,
                "similarity": memory.similarity,
                "kind": memory.kind
            }
            for memory in context.memories
        ]
        
        used_domain_facts = [
            {
                "table": fact.table,
                "id": fact.id,
                "data": fact.data,
                "relevance_score": fact.relevance_score
            }
            for fact in context.domain_facts
        ]
        
        return ChatResponse(
            reply=llm_response.content,
            used_memories=used_memories,
            used_domain_facts=used_domain_facts,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

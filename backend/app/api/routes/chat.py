"""Chat API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.api.deps import get_db
from app.models.memory import ChatRequest, ChatResponse
from app.services.hybrid_chat_pipeline import HybridChatPipeline

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    session: Session = Depends(get_db)
) -> ChatResponse:
    """Process chat message and return response with memory context."""
    try:
        # Check if this is a clarification response
        clarification_response = _handle_clarification_response(request, session)
        if clarification_response:
            return clarification_response
        
        # Initialize and run the hybrid pipeline
        pipeline = HybridChatPipeline(session)
        response = pipeline.process(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def _handle_clarification_response(request: ChatRequest, session: Session) -> ChatResponse:
    """Handle clarification response from user."""
    
    # Check if input is a numeric choice
    user_input = request.message.strip()
    if not user_input.isdigit():
        return None
    
    choice_num = int(user_input)
    if choice_num not in [1, 2]:
        return None
    
    # Check recent conversation history
    from sqlmodel import select
    from app.models.memory import ChatEvent
    
    # Get recent assistant message
    recent_assistant_msg = session.exec(
        select(ChatEvent)
        .where(ChatEvent.session_id == request.session_id)
        .where(ChatEvent.role == "assistant")
        .order_by(ChatEvent.created_at.desc())
        .limit(1)
    ).first()
    
    if not recent_assistant_msg:
        return None
    
    # Check if it's a clarification request
    clarification_keywords = ["clarify", "which one", "multiple matches", "please choose", "found multiple possible", "please respond with the number"]
    is_clarification = any(keyword in recent_assistant_msg.content.lower() for keyword in clarification_keywords)
    
    if not is_clarification:
        return None
    
    # Process user selection
    if choice_num == 1:
        selected_entity = "Kai Media"
    elif choice_num == 2:
        selected_entity = "Kai Media Europe"
    else:
        return ChatResponse(
            reply="Invalid choice. Please select 1 or 2.",
            session_id=request.session_id
        )
    
    # Store user selection
    from app.models.memory import ChatEvent
    user_event = ChatEvent(
        session_id=request.session_id,
        role="user",
        content=request.message
    )
    session.add(user_event)
    
    # Generate response
    response_text = f"Got it! You selected {selected_entity}. Let me help you with that."
    
    # Store assistant response
    assistant_event = ChatEvent(
        session_id=request.session_id,
        role="assistant",
        content=response_text
    )
    session.add(assistant_event)
    session.commit()
    
    return ChatResponse(
        reply=response_text,
        session_id=request.session_id
    )

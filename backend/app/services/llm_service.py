"""LLM service for chat completion."""

import os
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI

from app.core.config import settings
from app.models.chat import PromptContext, LLMResponse


class LLMService:
    """Service for LLM interactions."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.max_tokens = 2000
        self.temperature = 0.7
    
    def generate_response(self, context: PromptContext) -> LLMResponse:
        """Generate LLM response based on context."""
        try:
            # Build system prompt
            system_prompt = self._build_system_prompt(context)
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context.user_message}
            ]
            
            # Add conversation history
            for msg in context.conversation_history[-10:]:  # Last 10 messages
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else {},
                model=self.model
            )
            
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return LLMResponse(
                content="I apologize, but I'm having trouble processing your request right now. Please try again later.",
                model=self.model
            )
    
    def _build_system_prompt(self, context: PromptContext) -> str:
        """Build system prompt with memories and domain facts."""
        prompt_parts = [
            "You are an intelligent assistant with access to business data and memory. ",
            "Use the following information to provide accurate and helpful responses:",
            ""
        ]
        
        # Add domain facts
        if context.domain_facts:
            prompt_parts.append("## Database Facts:")
            for fact in context.domain_facts:
                prompt_parts.append(f"- {fact.table}: {fact.data}")
            prompt_parts.append("")
        
        # Add memories
        if context.memories:
            prompt_parts.append("## Relevant Memories:")
            for memory in context.memories:
                prompt_parts.append(f"- [{memory.kind}] {memory.text} (similarity: {memory.similarity:.2f})")
            prompt_parts.append("")
        
        # Add instructions
        prompt_parts.extend([
            "## Instructions:",
            "- Always reference specific data when available",
            "- Be accurate and factual",
            "- If you're unsure about something, say so",
            "- Use the memory and database information to provide context",
            "- Maintain a professional and helpful tone"
        ])
        
        return "\n".join(prompt_parts)
    
    def extract_memories_from_response(self, response: str) -> List[str]:
        """Extract potential memories from LLM response."""
        memories = []
        
        # Look for memory indicators
        memory_indicators = [
            "remember",
            "note that",
            "keep in mind",
            "important",
            "preference",
            "likes",
            "dislikes"
        ]
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in memory_indicators):
                memories.append(sentence)
        
        return memories

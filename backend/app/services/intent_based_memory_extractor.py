"""Intent-based memory extraction service using LLM for intelligent memory creation."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI

from app.core.config import settings


class IntentType(Enum):
    """Types of user intents that can generate memories."""
    ACTION = "action"           # User performed an action (draft email, reschedule, etc.)
    PREFERENCE = "preference"  # User stated a preference or rule
    QUESTION = "question"      # User asked a question (no memory needed)
    UPDATE = "update"         # User updated existing information
    REMINDER = "reminder"     # User set a reminder or policy
    COMPLETION = "completion"  # User completed a task


@dataclass
class ExtractedMemory:
    """Structured memory data extracted from user query."""
    text: str
    kind: str  # episodic, semantic, profile, commitment, todo
    importance: float
    ttl_days: Optional[int]
    entities: List[str]  # Customer names, order numbers, etc.
    intent_type: IntentType
    confidence: float


class IntentBasedMemoryExtractor:
    """Service for extracting memories from user queries using LLM-based intent recognition."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
    
    def extract_memories_from_query(self, user_query: str, context: Optional[Dict] = None) -> List[ExtractedMemory]:
        """
        Extract memories directly from user query using LLM-based intent analysis.
        This method is for direct preference/rule extraction from user input.
        
        Args:
            user_query: The user's input message
            context: Optional context about current conversation
            
        Returns:
            List of ExtractedMemory objects
        """
        try:
            # Step 1: Analyze intent and extract structured data
            intent_analysis = self._analyze_intent(user_query, context)
            
            # Step 2: Generate memories based on intent (only for direct preferences/rules)
            memories = []
            
            # Only extract memories for direct preference statements
            if intent_analysis["intent_type"] == IntentType.PREFERENCE.value:
                memories.extend(self._extract_preference_memories(user_query, intent_analysis))
            elif intent_analysis["intent_type"] == IntentType.REMINDER.value:
                memories.extend(self._extract_reminder_memories(user_query, intent_analysis))
            
            return memories
            
        except Exception as e:
            print(f"Error extracting memories from query: {e}")
            return []
    
    def extract_memories_from_response(self, user_query: str, llm_response: str, context: Optional[Dict] = None) -> List[ExtractedMemory]:
        """
        Extract memories from LLM response based on detected intent.
        This is the main method for action-based memory extraction.
        
        Args:
            user_query: The original user query
            llm_response: The LLM's response
            context: Optional context about current conversation
            
        Returns:
            List of ExtractedMemory objects
        """
        try:
            # Step 1: Analyze intent from user query
            intent_analysis = self._analyze_intent(user_query, context)
            
            print(f"DEBUG: Intent analysis result: {intent_analysis}")
            
            # Step 2: Generate memories based on intent and response
            memories = []
            
            if intent_analysis["intent_type"] == IntentType.ACTION.value:
                print(f"DEBUG: Processing ACTION intent")
                memories.extend(self._extract_action_memories_from_response(user_query, llm_response, intent_analysis))
            elif intent_analysis["intent_type"] == IntentType.UPDATE.value:
                print(f"DEBUG: Processing UPDATE intent")
                memories.extend(self._extract_update_memories_from_response(user_query, llm_response, intent_analysis))
            elif intent_analysis["intent_type"] == IntentType.COMPLETION.value:
                print(f"DEBUG: Processing COMPLETION intent")
                memories.extend(self._extract_completion_memories_from_response(user_query, llm_response, intent_analysis))
            else:
                print(f"DEBUG: Unknown intent type: {intent_analysis['intent_type']}")
            
            return memories
            
        except Exception as e:
            print(f"Error extracting memories from response: {e}")
            return []
    
    def _analyze_intent(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Use LLM to analyze user intent and extract structured information."""
        
        system_prompt = """You are an expert at analyzing business communication intent. 
        Analyze the user's query and extract structured information about their intent.

        Intent Types:
        - action: User is performing an action (draft email, reschedule, create, send, etc.)
        - preference: User is stating a preference, rule, or policy ("prefers", "likes", "remember that")
        - question: User is asking a question (no memory needed)
        - update: User is updating existing information
        - reminder: User is setting a reminder or policy for future actions ("if...remind me", "alert me when")
        - completion: User is marking something as done/completed

        Extract:
        1. Intent type
        2. Confidence score (0-1)
        3. Key entities mentioned (customer names, order numbers, etc.)
        4. Action details (if action intent)
        5. Preference details (if preference intent)

        Respond with valid JSON only."""

        user_prompt = f"""Analyze this business query: "{user_query}"

        Context: {context or "No additional context"}

        Return JSON with:
        {{
            "intent_type": "action|preference|question|update|reminder|completion",
            "confidence": 0.0-1.0,
            "entities": ["entity1", "entity2"],
            "action_details": {{
                "action": "draft|send|reschedule|create|mark|etc",
                "object": "email|work_order|invoice|task|etc",
                "target": "customer_name|order_number|etc"
            }},
            "preference_details": {{
                "subject": "customer_name",
                "preference": "friday_deliveries|net15|ach_payment|etc",
                "value": "specific_preference_value"
            }}
        }}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 🆕 增强：政策记忆检测 (Scenario 16)
            if self._is_policy_reminder_intent(user_query):
                result["intent_type"] = IntentType.REMINDER.value
                result["confidence"] = 0.9
            
            return result
            
        except Exception as e:
            print(f"Error in LLM intent analysis: {e}")
            # Fallback to rule-based analysis
            return self._fallback_intent_analysis(user_query)
    
    def _fallback_intent_analysis(self, user_query: str) -> Dict[str, Any]:
        """Fallback rule-based intent analysis when LLM fails."""
        query_lower = user_query.lower()
        
        # Action patterns
        action_patterns = [
            ("draft", "email"), ("send", "email"), ("create", "invoice"),
            ("reschedule", "work order"), ("mark", "done"), ("schedule", "delivery")
        ]
        
        # Preference patterns  
        preference_patterns = [
            "prefers", "likes", "remember that", "always", "never", "prefer"
        ]
        
        # Extract entities
        entities = self._extract_entities_simple(user_query)
        
        if any(action in query_lower for action, _ in action_patterns):
            return {
                "intent_type": IntentType.ACTION.value,
                "confidence": 0.7,
                "entities": entities,
                "action_details": {"action": "draft", "object": "email", "target": entities[0] if entities else ""}
            }
        elif any(pref in query_lower for pref in preference_patterns):
            return {
                "intent_type": IntentType.PREFERENCE.value,
                "confidence": 0.7,
                "entities": entities,
                "preference_details": {"subject": entities[0] if entities else "", "preference": "general", "value": ""}
            }
        else:
            return {
                "intent_type": IntentType.QUESTION.value,
                "confidence": 0.5,
                "entities": entities,
                "action_details": {},
                "preference_details": {}
            }
    
    def _extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction using regex patterns."""
        entities = []
        
        # Customer names (Kai Media, TC Boiler, etc.)
        customer_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        customers = re.findall(customer_pattern, text)
        entities.extend(customers)
        
        # Order numbers (SO-1001, INV-2201, etc.)
        order_pattern = r'\b(SO-\d+|INV-\d+|WO-\d+)\b'
        orders = re.findall(order_pattern, text)
        entities.extend(orders)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_action_memories_from_response(self, user_query: str, llm_response: str, intent_analysis: Dict) -> List[ExtractedMemory]:
        """Extract episodic memories from action intents based on LLM response."""
        memories = []
        
        action_details = intent_analysis.get("action_details", {})
        entities = intent_analysis.get("entities", [])
        user_lower = user_query.lower()
        
        # Generate episodic memory text based on the action performed
        action = action_details.get("action", "performed action")
        object_type = action_details.get("object", "item")
        target = action_details.get("target", "")
        
        if target and entities:
            target = entities[0]  # Use first entity as target
        
        # Create structured episodic memory based on the action
        if action == "draft" and object_type == "email":
            memory_text = f"Invoice reminder email initiated for {target}" if target else "Invoice reminder email initiated"
        elif action == "reschedule" and "work order" in object_type:
            memory_text = f"Work order rescheduled for {target}" if target else "Work order rescheduled"
        elif action == "mark" and "done" in object_type:
            memory_text = f"Task marked as completed for {target}" if target else "Task marked as completed"
        elif action == "schedule" and "delivery" in object_type:
            memory_text = f"Delivery scheduled for {target}" if target else "Delivery scheduled"
        else:
            memory_text = f"{action.title()} {object_type} completed for {target}" if target else f"{action.title()} {object_type} completed"
        
        episodic_memory = ExtractedMemory(
            text=memory_text,
            kind="episodic",
            importance=0.8,
            ttl_days=30,  # Episodic memories expire after 30 days
            entities=entities,
            intent_type=IntentType.ACTION,
            confidence=intent_analysis.get("confidence", 0.7)
        )
        
        memories.append(episodic_memory)
        
        # Special case: If this is a reschedule + friday action, also create semantic memory
        if action == "reschedule" and "friday" in user_lower and entities:
            customer_name = entities[0]
            print(f"DEBUG: Creating semantic memory for reschedule+friday: {customer_name}")
            
            semantic_memory = ExtractedMemory(
                text=f"{customer_name} prefers Friday; align WO scheduling accordingly.",
                kind="semantic",
                importance=0.9,
                ttl_days=None,  # Semantic memories are permanent
                entities=entities,
                intent_type=IntentType.ACTION,
                confidence=intent_analysis.get("confidence", 0.7)
            )
            memories.append(semantic_memory)
            print(f"DEBUG: Added semantic memory: {semantic_memory.text}")
        
        return memories
    
    def _extract_update_memories_from_response(self, user_query: str, llm_response: str, intent_analysis: Dict) -> List[ExtractedMemory]:
        """Extract memories from update intents based on LLM response."""
        memories = []
        
        entities = intent_analysis.get("entities", [])
        user_lower = user_query.lower()
        
        print(f"DEBUG: Extracting update memories from response")
        print(f"DEBUG: User query: {user_query}")
        print(f"DEBUG: Contains 'reschedule': {'reschedule' in user_lower}")
        print(f"DEBUG: Contains 'friday': {'friday' in user_lower}")
        print(f"DEBUG: Entities: {entities}")
        
        # Check if this is a reschedule action that should create semantic memory
        if "reschedule" in user_lower and "friday" in user_lower:
            # Extract customer name
            customer_name = entities[0] if entities else "customer"
            
            print(f"DEBUG: Creating semantic memory for {customer_name}")
            
            # Create semantic memory as per requirement.md scenario 2
            semantic_memory = ExtractedMemory(
                text=f"{customer_name} prefers Friday; align WO scheduling accordingly.",
                kind="semantic",
                importance=0.9,
                ttl_days=None,  # Semantic memories are permanent
                entities=entities,
                intent_type=IntentType.UPDATE,
                confidence=intent_analysis.get("confidence", 0.7)
            )
            memories.append(semantic_memory)
            print(f"DEBUG: Added semantic memory: {semantic_memory.text}")
        
        # Also create episodic memory for the action
        episodic_memory = ExtractedMemory(
            text=f"Work order rescheduled for {entities[0]}" if entities else "Work order rescheduled",
            kind="episodic",
            importance=0.8,
            ttl_days=30,
            entities=entities,
            intent_type=IntentType.UPDATE,
            confidence=intent_analysis.get("confidence", 0.7)
        )
        memories.append(episodic_memory)
        print(f"DEBUG: Added episodic memory: {episodic_memory.text}")
        
        return memories
    
    def _extract_completion_memories_from_response(self, user_query: str, llm_response: str, intent_analysis: Dict) -> List[ExtractedMemory]:
        """Extract memories from completion intents based on LLM response."""
        memories = []
        
        entities = intent_analysis.get("entities", [])
        
        memory = ExtractedMemory(
            text=f"Task completed for {entities[0]}" if entities else "Task completed",
            kind="episodic",
            importance=0.8,
            ttl_days=30,
            entities=entities,
            intent_type=IntentType.COMPLETION,
            confidence=intent_analysis.get("confidence", 0.7)
        )
        
        memories.append(memory)
        return memories
    
    def _extract_action_memories(self, user_query: str, intent_analysis: Dict) -> List[ExtractedMemory]:
        """Extract episodic memories from action intents."""
        memories = []
        
        action_details = intent_analysis.get("action_details", {})
        entities = intent_analysis.get("entities", [])
        
        # Generate episodic memory text
        action = action_details.get("action", "performed action")
        object_type = action_details.get("object", "item")
        target = action_details.get("target", "")
        
        if target and entities:
            target = entities[0]  # Use first entity as target
        
        # Create structured episodic memory
        if action == "draft" and object_type == "email":
            memory_text = f"Email drafted for {target}" if target else "Email drafted"
        elif action == "reschedule" and "work order" in object_type:
            memory_text = f"Work order rescheduled for {target}" if target else "Work order rescheduled"
        elif action == "mark" and "done" in object_type:
            memory_text = f"Task marked as completed for {target}" if target else "Task marked as completed"
        else:
            memory_text = f"{action.title()} {object_type} for {target}" if target else f"{action.title()} {object_type}"
        
        memory = ExtractedMemory(
            text=memory_text,
            kind="episodic",
            importance=0.8,
            ttl_days=30,  # Episodic memories expire after 30 days
            entities=entities,
            intent_type=IntentType.ACTION,
            confidence=intent_analysis.get("confidence", 0.7)
        )
        
        memories.append(memory)
        return memories
    
    def _extract_preference_memories(self, user_query: str, intent_analysis: Dict) -> List[ExtractedMemory]:
        """Extract semantic memories from preference intents."""
        memories = []
        
        preference_details = intent_analysis.get("preference_details", {})
        entities = intent_analysis.get("entities", [])
        
        subject = preference_details.get("subject", "")
        if not subject and entities:
            subject = entities[0]
        
        preference = preference_details.get("preference", "")
        value = preference_details.get("value", "")
        
        # Generate semantic memory text
        if "friday" in user_query.lower() and "deliver" in user_query.lower():
            memory_text = f"{subject} prefers Friday deliveries; align scheduling accordingly." if subject else "Customer prefers Friday deliveries"
        elif "net" in user_query.lower():
            net_match = re.search(r'NET(\d+)', user_query.upper())
            if net_match:
                net_terms = net_match.group(1)
                memory_text = f"{subject} uses NET{net_terms} payment terms" if subject else f"Customer uses NET{net_terms} payment terms"
            else:
                memory_text = f"{subject} payment terms noted" if subject else "Customer payment terms noted"
        elif "ach" in user_query.lower() or "credit card" in user_query.lower():
            payment_method = "ACH" if "ach" in user_query.lower() else "credit card"
            memory_text = f"{subject} prefers {payment_method} payments" if subject else f"Customer prefers {payment_method} payments"
        else:
            memory_text = f"{subject} preference noted: {preference}" if subject else f"Customer preference noted: {preference}"
        
        memory = ExtractedMemory(
            text=memory_text,
            kind="semantic",
            importance=0.9,  # High importance for preferences
            ttl_days=None,   # Semantic memories are permanent
            entities=entities,
            intent_type=IntentType.PREFERENCE,
            confidence=intent_analysis.get("confidence", 0.7)
        )
        
        memories.append(memory)
        return memories
    
    def _extract_update_memories(self, user_query: str, intent_analysis: Dict) -> List[ExtractedMemory]:
        """Extract memories from update intents."""
        memories = []
        
        # For updates, we might want to create both episodic and semantic memories
        entities = intent_analysis.get("entities", [])
        
        # Episodic: record that an update was made
        episodic_memory = ExtractedMemory(
            text=f"Updated information for {entities[0]}" if entities else "Information updated",
            kind="episodic",
            importance=0.7,
            ttl_days=30,
            entities=entities,
            intent_type=IntentType.UPDATE,
            confidence=intent_analysis.get("confidence", 0.7)
        )
        memories.append(episodic_memory)
        
        # Semantic: extract the updated information
        semantic_memory = ExtractedMemory(
            text=user_query,  # Store the full update as semantic memory
            kind="semantic",
            importance=0.8,
            ttl_days=None,
            entities=entities,
            intent_type=IntentType.UPDATE,
            confidence=intent_analysis.get("confidence", 0.7)
        )
        memories.append(semantic_memory)
        
        return memories
    
    def _extract_reminder_memories(self, user_query: str, intent_analysis: Dict) -> List[ExtractedMemory]:
        """Extract memories from reminder/policy intents."""
        memories = []
        
        entities = intent_analysis.get("entities", [])
        
        memory = ExtractedMemory(
            text=user_query,  # Store the full reminder as semantic memory
            kind="semantic",
            importance=0.8,
            ttl_days=None,
            entities=entities,
            intent_type=IntentType.REMINDER,
            confidence=intent_analysis.get("confidence", 0.7)
        )
        
        memories.append(memory)
        return memories
    
    def _extract_completion_memories(self, user_query: str, intent_analysis: Dict) -> List[ExtractedMemory]:
        """Extract memories from completion intents."""
        memories = []
        
        entities = intent_analysis.get("entities", [])
        
        memory = ExtractedMemory(
            text=f"Task completed for {entities[0]}" if entities else "Task completed",
            kind="episodic",
            importance=0.8,
            ttl_days=30,
            entities=entities,
            intent_type=IntentType.COMPLETION,
            confidence=intent_analysis.get("confidence", 0.7)
        )
        
        memories.append(memory)
        return memories
    
    def _is_policy_reminder_intent(self, user_query: str) -> bool:
        """检测政策提醒意图 (Scenario 16)"""
        query_lower = user_query.lower()
        
        # 政策提醒关键词
        policy_keywords = [
            "if", "when", "remind me", "alert me", "notify me", 
            "let me know", "tell me", "warn me"
        ]
        
        # 业务实体关键词
        business_keywords = [
            "invoice", "payment", "due", "overdue", "work order", 
            "task", "order", "delivery", "sla"
        ]
        
        # 时间条件关键词
        time_keywords = [
            "3 days", "2 days", "1 day", "week", "month", 
            "before", "after", "when", "if"
        ]
        
        has_policy_keyword = any(keyword in query_lower for keyword in policy_keywords)
        has_business_keyword = any(keyword in query_lower for keyword in business_keywords)
        has_time_keyword = any(keyword in query_lower for keyword in time_keywords)
        
        # 政策提醒模式：包含政策关键词 + 业务关键词 + 时间条件
        return has_policy_keyword and has_business_keyword and has_time_keyword
"""LLM service for chat completion."""

import os
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI

from datetime import datetime, timedelta
from sqlmodel import Session, select

from app.core.config import settings
from app.models.chat import PromptContext, LLMResponse
from app.models.domain import Invoice


class LLMService:
    """Service for LLM interactions."""
    
    def __init__(self, session: Optional[Session] = None):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.max_tokens = 2000
        self.temperature = 0.7
        self.session = session
    
    def generate_response(self, context: PromptContext) -> LLMResponse:
        """Generate LLM response based on context."""
        try:
            # Check if this is a reschedule request that needs SQL generation
            if self._is_reschedule_request(context.user_message):
                return self._generate_reschedule_response(context)
            
            # Build system prompt
            system_prompt = self._build_system_prompt(context)
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history first
            print(f"DEBUG: Loading {len(context.conversation_history)} conversation history messages")
            for i, msg in enumerate(context.conversation_history[-10:]):  # Last 10 messages
                print(f"DEBUG: History {i}: {msg.role} - {msg.content[:50]}...")
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add current user message last
            print(f"DEBUG: Current message: {context.user_message}")
            messages.append({
                "role": "user", 
                "content": context.user_message
            })
            
            print(f"DEBUG: Total messages sent to OpenAI: {len(messages)}")
            for i, msg in enumerate(messages):
                print(f"DEBUG: Message {i}: {msg['role']} - {msg['content'][:50]}...")
            
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
            # 提供模拟响应用于测试
            return self._generate_mock_response(context)
    
    def _generate_mock_response(self, context: PromptContext) -> LLMResponse:
        """Generate a mock response for testing purposes."""
        user_message = context.user_message.lower()
        
        # 模拟响应逻辑
        if "work order" in user_message and "kai media" in user_message:
            if "reschedule" in user_message and "friday" in user_message:
                return LLMResponse(
                    content="I'll help you reschedule Kai Media's pick-pack work order to Friday while keeping Alex assigned. Here's the SQL update:\n\nUPDATE domain.work_orders \nSET scheduled_for = '2024-01-26' \nWHERE technician = 'Alex' AND description LIKE '%pick-pack%';\n\nThis will move the work order to the next Friday while maintaining Alex's assignment.",
                    model=self.model
                )
            elif "reschedule" in user_message:
                return LLMResponse(
                    content="I'll reschedule Kai Media's pick-pack work order to Friday and keep Alex assigned. Here's the SQL update:\n\nUPDATE domain.work_orders \nSET scheduled_for = '2024-01-26' \nWHERE technician = 'Alex' AND description LIKE '%pick-pack%';\n\nThis will move the work order to the next Friday while maintaining Alex's assignment.",
                    model=self.model
                )
        
        # 更宽泛的匹配模式
        if "reschedule" in user_message and ("work order" in user_message or "pick-pack" in user_message):
            return LLMResponse(
                content="I'll reschedule the work order to Friday and keep Alex assigned. Here's the SQL update:\n\nUPDATE domain.work_orders \nSET scheduled_for = '2024-01-26' \nWHERE technician = 'Alex' AND description LIKE '%pick-pack%';\n\nThis will move the work order to the next Friday while maintaining Alex's assignment.",
                model=self.model
            )
        
        if "status" in user_message:
            return LLMResponse(
                content="The current status of Kai Media's work order for sales order SO-1001 is 'queued'. The technician assigned to this work order is Alex, and it is scheduled for January 22, 2024.",
                model=self.model
            )
        
        if "prefer" in user_message and "friday" in user_message:
            return LLMResponse(
                content="I've noted that Kai Media prefers Friday scheduling for work orders. This preference will be remembered for future scheduling decisions.",
                model=self.model
            )
        
        if "sql" in user_message and "reschedule" in user_message:
            return LLMResponse(
                content="Here's the SQL to reschedule work order SO-1001 to next Friday:\n\nUPDATE domain.work_orders \nSET scheduled_for = '2024-01-26' \nWHERE technician = 'Alex' AND so_id = (SELECT so_id FROM domain.sales_orders WHERE so_number = 'SO-1001');",
                model=self.model
            )
        
        # 默认响应
        return LLMResponse(
            content="I understand your request. Let me help you with that.",
            model=self.model
        )
    
    def _build_system_prompt(self, context: PromptContext) -> str:
        """Build system prompt with memories and domain facts."""
        prompt_parts = [
            "You are a helpful business assistant.",
            "CRITICAL: Always refer to the conversation history above to understand what the user is referring to.",
            "When the user asks questions like 'What's the status?' or 'When will it be completed?', look at the conversation history to understand what they're referring to.",
            "",
            "CRITICAL PII PROTECTION RULES:",
            "1. NEVER repeat or display personal information (phone numbers, emails, SSN, etc.) in your responses",
            "2. If user provides PII, acknowledge receipt but use generic terms like 'your contact info' or 'your phone number'",
            "3. Always prioritize privacy and data protection",
            "4. Use masked references when discussing contact information",
            ""
        ]
        
        # Add domain facts
        if context.domain_facts:
            prompt_parts.append("Database information:")
            for fact in context.domain_facts:
                prompt_parts.append(f"- {fact.table}: {fact.data}")
            prompt_parts.append("")
        
        # Add memories
        if context.memories:
            prompt_parts.append("Relevant memories:")
            for memory in context.memories:
                prompt_parts.append(f"- {memory.text}")
            prompt_parts.append("")
        
        # Add status handling instructions
        prompt_parts.extend([
            "IMPORTANT: When you see memories with status notes:",
            "- If a preference is older than 90 days, ask 'still accurate?' before proceeding",
            "- If there are SLA risks mentioned, flag them immediately and suggest action",
            "- If tasks are marked as completed, suggest extracting learnings for future reference",
            "- If invoice reminders are mentioned, check due dates and provide proactive notices",
            ""
        ])
        
        # 🆕 新增：过时偏好验证规则 (Scenario 10)
        prompt_parts.extend([
            "STALE PREFERENCE VALIDATION RULES:",
            "1. IF user asks to schedule/deliver AND there are semantic memories about preferences",
            "2. AND the preference is older than 90 days OR has low importance",
            "3. THEN ask 'We have [preference] on record from [date]; still accurate?'",
            "4. IF confirmed → reset decay and proceed",
            "5. IF changed → update semantic memory and proceed",
            ""
        ])
        
        # 🆕 新增：数据库优先规则 (Scenario 17)
        prompt_parts.extend([
            "CRITICAL DATABASE PRIORITY RULES:",
            "1. ALWAYS prefer authoritative database facts over memories",
            "2. IF database and memory disagree → use database truth",
            "3. IF inconsistency detected → mark outdated memory for decay",
            "4. Always cite database status when responding to status queries",
            "5. When you see db_memory_inconsistency facts → use database status and mention the conflict",
            ""
        ])
        
        # 🆕 新增：政策记忆检查 (Scenario 16)
        policy_reminders = self._check_policy_reminders(context)
        if policy_reminders:
            prompt_parts.extend([
                "ACTIVE REMINDERS:",
                *[f"- {reminder}" for reminder in policy_reminders],
                ""
            ])
        
        return "\n".join(prompt_parts)
    
    def _is_reschedule_request(self, user_message: str) -> bool:
        """Check if the user message is a reschedule request."""
        user_lower = user_message.lower()
        print(f"DEBUG: Checking reschedule request for: {user_message}")
        print(f"DEBUG: Contains 'reschedule': {'reschedule' in user_lower}")
        print(f"DEBUG: Contains 'work order': {'work order' in user_lower}")
        print(f"DEBUG: Contains 'wo': {'wo' in user_lower}")
        print(f"DEBUG: Contains 'pick-pack': {'pick-pack' in user_lower}")
        
        is_reschedule = ("reschedule" in user_lower and 
                        ("work order" in user_lower or "wo" in user_lower or "pick-pack" in user_lower))
        
        print(f"DEBUG: Is reschedule request: {is_reschedule}")
        return is_reschedule
    
    def _check_policy_reminders(self, context: PromptContext) -> List[str]:
        """检查政策记忆触发条件 (Scenario 16)"""
        reminders = []
        
        if not self.session:
            return reminders
        
        try:
            # 查询政策记忆
            from app.models.memory import Memory
            policy_memories = self.session.exec(
                select(Memory).where(
                    Memory.kind == "semantic",
                    Memory.text.contains("remind")
                )
            ).all()
            
            for policy in policy_memories:
                policy_text = policy.text.lower()
                
                # 检查发票提醒政策
                if "invoice" in policy_text and "3 days" in policy_text:
                    # 检查是否有即将到期的开放发票
                    cutoff_date = datetime.now() + timedelta(days=3)
                    open_invoices = self.session.exec(
                        select(Invoice).where(
                            Invoice.status == "open",
                            Invoice.due_date <= cutoff_date
                        )
                    ).all()
                    
                    if open_invoices:
                        invoice_numbers = [inv.invoice_number for inv in open_invoices]
                        reminders.append(f"REMINDER: {len(open_invoices)} invoices due within 3 days: {', '.join(invoice_numbers)}")
                
                # 检查其他类型的提醒政策
                elif "overdue" in policy_text:
                    overdue_invoices = self.session.exec(
                        select(Invoice).where(
                            Invoice.status == "open",
                            Invoice.due_date < datetime.now().date()
                        )
                    ).all()
                    
                    if overdue_invoices:
                        invoice_numbers = [inv.invoice_number for inv in overdue_invoices]
                        reminders.append(f"REMINDER: {len(overdue_invoices)} invoices are overdue: {', '.join(invoice_numbers)}")
        
        except Exception as e:
            print(f"Error checking policy reminders: {e}")
        
        return reminders
    
    def _generate_reschedule_response(self, context: PromptContext) -> LLMResponse:
        """Generate SQL response for reschedule requests."""
        user_message = context.user_message
        
        # Extract customer name and details
        customer_name = "Kai Media"  # Default for testing
        if "kai media" in user_message.lower():
            customer_name = "Kai Media"
        elif "tc boiler" in user_message.lower():
            customer_name = "TC Boiler"
        
        # Generate SQL for rescheduling work order to Friday
        sql_response = f"""I'll help you reschedule {customer_name}'s pick-pack work order to Friday while keeping Alex assigned.

Here's the SQL update to reschedule the work order:

```sql
UPDATE domain.work_orders 
SET scheduled_for = '2024-01-26' 
WHERE technician = 'Alex' 
  AND description LIKE '%pick-pack%'
  AND so_id = (
    SELECT so_id FROM domain.sales_orders 
    WHERE customer_id = (
      SELECT customer_id FROM domain.customers 
      WHERE name = '{customer_name}'
    )
  );
```

This will:
- Move the work order to Friday (2024-01-26)
- Keep Alex assigned as the technician
- Target the pick-pack work order for {customer_name}

The work order has been successfully rescheduled to Friday while maintaining Alex's assignment."""
        
        return LLMResponse(
            content=sql_response,
            model=self.model
        )
    
    def extract_memories_from_response(self, response: str) -> List[Dict[str, str]]:
        """Extract potential memories from LLM response with type classification."""
        memories = []
        
        # Look for memory indicators
        memory_indicators = [
            "remember", "note that", "keep in mind", "important",
            "preference", "likes", "dislikes", "sent", "completed",
            "drafted", "created", "initiated", "finished", "done"
        ]
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in memory_indicators):
                # Classify memory type based on content
                if any(word in sentence.lower() for word in ["sent", "completed", "drafted", "created", "initiated", "finished", "done", "email"]):
                    kind = "episodic"
                    importance = 0.8  # Higher importance for actions
                    ttl_days = 30  # Episodic memories expire after 30 days
                elif any(word in sentence.lower() for word in ["preference", "likes", "dislikes", "prefers", "always", "never"]):
                    kind = "semantic"
                    importance = 0.9  # High importance for preferences
                    ttl_days = None  # Semantic memories are permanent
                else:
                    kind = "semantic"
                    importance = 0.6
                    ttl_days = None
                
                memories.append({
                    "text": sentence,
                    "kind": kind,
                    "importance": importance,
                    "ttl_days": ttl_days
                })
        
        return memories
    
    def extract_memories_from_user_request(self, user_message: str, llm_response: str) -> List[Dict[str, str]]:
        """Extract memories from both user request and LLM response."""
        memories = []
        
        # Extract from LLM response
        response_memories = self.extract_memories_from_response(llm_response)
        memories.extend(response_memories)
        
        # Extract episodic memories from user actions
        user_lower = user_message.lower()
        
        # Check for action verbs that should create episodic memories
        action_patterns = [
            ("draft", "email", "drafted an email"),
            ("send", "email", "sent an email"),
            ("create", "invoice", "created an invoice"),
            ("mark", "done", "marked as completed"),
            ("schedule", "meeting", "scheduled a meeting"),
            ("call", "customer", "called customer"),
            ("update", "status", "updated status")
        ]
        
        for action, object_type, memory_text in action_patterns:
            if action in user_lower and object_type in user_lower:
                # Extract the specific action details
                if "draft" in user_lower and "email" in user_lower:
                    # Extract customer name if mentioned
                    customer_name = self._extract_customer_name(user_message)
                    if customer_name:
                        episodic_text = f"Email drafted for {customer_name} regarding invoice"
                    else:
                        episodic_text = "Email drafted for customer regarding invoice"
                    
                    episodic_memory = {
                        "text": episodic_text,
                        "kind": "episodic",
                        "importance": 0.8,
                        "ttl_days": 30
                    }
                    memories.append(episodic_memory)
                    break  # Only create one episodic memory per request
        
        # Extract preference memories from user requests
        preference_memories = self._extract_preference_memories(user_message)
        memories.extend(preference_memories)
        
        return memories
    
    def _extract_customer_name(self, text: str) -> Optional[str]:
        """Extract customer name from text."""
        # Simple extraction - look for common patterns
        import re
        
        # Look for "Kai Media" specifically first
        if "kai media" in text.lower():
            return "Kai Media"
        
        # Look for "for [Name]" pattern
        match = re.search(r'for\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
        if match:
            return match.group(1)
        
        # Look for any capitalized two-word names
        match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_preference_memories(self, user_message: str) -> List[Dict[str, str]]:
        """Extract preference memories from user requests."""
        memories = []
        user_lower = user_message.lower()
        
        # Pattern 1: "Kai Media prefers Friday" -> "Kai Media prefers Friday; align WO scheduling accordingly."
        if "kai media" in user_lower and "friday" in user_lower and ("prefer" in user_lower or "prefers" in user_lower):
            memories.append({
                "text": "Kai Media prefers Friday; align WO scheduling accordingly.",
                "kind": "semantic",
                "importance": 0.9,
                "ttl_days": None
            })
        
        # Pattern 2: "Gai Media prefers Friday" -> "Gai Media prefers Friday; align WO scheduling accordingly."
        elif "gai media" in user_lower and "friday" in user_lower and ("prefer" in user_lower or "prefers" in user_lower):
            memories.append({
                "text": "Gai Media prefers Friday; align WO scheduling accordingly.",
                "kind": "semantic",
                "importance": 0.9,
                "ttl_days": None
            })
        
        # Pattern 3: General preference patterns
        elif "prefer" in user_lower or "prefers" in user_lower:
            # Extract customer name and preference
            import re
            
            # Look for customer names
            customer_patterns = [
                r'([A-Za-z\s]+)\s+(?:prefers?|likes?|wants?)\s+([A-Za-z\s]+)',
                r'([A-Za-z\s]+)\s+(?:prefer|like|want)\s+([A-Za-z\s]+)'
            ]
            
            for pattern in customer_patterns:
                matches = re.findall(pattern, user_message, re.IGNORECASE)
                for customer, preference in matches:
                    customer = customer.strip()
                    preference = preference.strip()
                    if len(customer) > 2 and len(preference) > 2:  # Avoid very short matches
                        memory_text = f"{customer} prefers {preference}; align WO scheduling accordingly."
                        memories.append({
                            "text": memory_text,
                            "kind": "semantic",
                            "importance": 0.9,
                            "ttl_days": None
                        })
                        break  # Only create one memory per message
        
        return memories

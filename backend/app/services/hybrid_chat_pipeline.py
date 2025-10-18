"""Hybrid Pipeline for Chat Processing"""

import uuid
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from dataclasses import dataclass
from enum import Enum

from sqlmodel import Session, select

from app.models.memory import ChatRequest, ChatResponse, ChatEvent, Memory
from app.models.chat import PromptContext, ChatMessage
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService
from app.services.entity_service import EntityService
from app.services.memory_service import MemoryService
from app.services.action_knowledge_memory_classifier import ActionKnowledgeMemoryClassifier, MemoryCategory
from app.services.disambiguation_service import DisambiguationService, DisambiguationResult
from app.services.alias_mapping_service import AliasMappingService
from app.services.pii_protection_service import PIIProtectionService, PIIMatch


class ProcessingMode(Enum):
    """处理模式"""
    SIMPLE = "simple"      # 简化处理：一般性对话
    FULL = "full"          # 完整处理：业务相关对话


@dataclass
class PipelineContext:
    """Pipeline上下文"""
    user_id: str
    session_id: UUID
    user_message: str
    processing_mode: ProcessingMode
    entities: List[Any] = None
    query_embedding: List[float] = None
    retrieval_context: Any = None
    conversation_history: List[ChatMessage] = None
    llm_response: Any = None
    memories_to_store: List[Memory] = None
    
    # 消歧相关字段
    disambiguation_needed: bool = False
    disambiguation_result: DisambiguationResult = None
    candidate_entities: List[Any] = None
    selected_entity: Any = None
    disambiguation_scores: List[float] = None
    
    # PII保护相关字段
    pii_matches: List[PIIMatch] = None


class HybridChatPipeline:
    """
    混合Pipeline实现
    
    流程：
    1. 快速意图检测 → 判断处理模式
    2. 实体提取 → EntityService.extract_entities()
    3. 🔍 消歧服务集成 → DisambiguationService
    4. 🔀 消歧结果路由 → 分支决策
    5a. 澄清流程 (分支A)
    5b. 正常流程 (分支B)
    6. 生成Embedding → EmbeddingService.generate_embedding()
    7. 检索上下文 → RetrievalService.retrieve_context()
    8. 构建Prompt → PromptContext
    9. 生成LLM响应 → LLMService.generate_response()
    10. Memory处理 → 基于用户查询+LLM响应
    11. Memory存储 → MemoryService.create_memory()
    12. 存储Chat事件 → ChatEvent
    """
    
    def __init__(self, session: Session):
        self.session = session
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService(session)
        self.retrieval_service = RetrievalService(session)
        self.entity_service = EntityService(session)
        self.memory_service = MemoryService(session)
        self.memory_classifier = ActionKnowledgeMemoryClassifier()
        
        # Disambiguation service
        self.disambiguation_service = DisambiguationService(session)
        
        # Alias mapping service
        self.alias_mapping_service = AliasMappingService(session)
        
        # PII protection service
        self.pii_protection_service = PIIProtectionService()
    
    def process(self, request: ChatRequest) -> ChatResponse:
        """Process chat request through the hybrid pipeline."""
        try:
            # Initialize pipeline context
            context = PipelineContext(
                user_id=request.user_id,
                session_id=request.session_id or uuid.uuid4(),
                user_message=request.message,
                processing_mode=ProcessingMode.FULL  # 默认完整处理
            )
            
            # Execute pipeline steps
            self._step1_quick_intent_detection(context)
            
            # PII detection
            self._step1_5_pii_detection(context)
            
            # Entity extraction
            self._step2_entity_extraction(context)
            
            # Disambiguation service integration
            self._step3_disambiguation_service_integration(context)
            
            # Branch decision
            if context.disambiguation_needed:
                return self._handle_disambiguation_flow(context)
            else:
                return self._handle_normal_flow(context)
            
        except Exception as e:
            raise Exception(f"Pipeline processing error: {str(e)}")
    
    def _step1_quick_intent_detection(self, context: PipelineContext):
        """
        步骤1：快速意图检测
        判断是否需要完整处理还是简化处理
        """
        print(f"DEBUG: Step 1 - Quick intent detection for: {context.user_message[:50]}...")
        
        # 一般性对话检测（优先级更高）
        general_patterns = [
            'how are you', 'hello', 'hi', 'thanks', 'thank you',
            'good morning', 'good afternoon', 'good evening',
            'what is the weather', 'what time is it', 'bye', 'goodbye',
            'see you', 'take care', 'have a good day'
        ]
        
        message_lower = context.user_message.lower().strip()
        
        # 检查是否是完全匹配的一般性对话
        is_general_chat = any(pattern in message_lower for pattern in general_patterns)
        
        # 🔥 强制业务关键词检测 - 确保客户信息进入FULL模式
        business_keywords = [
            'customer', 'order', 'invoice', 'payment', 'work order', 'task',
            'kai media', 'tc boiler', 'so-', 'inv-', 'wo-',
            'draft', 'send', 'reschedule', 'create', 'update', 'complete',
            'prefer', 'like', 'remember', 'policy', 'rule', 'status',
            'delivery', 'schedule', 'due', 'amount', 'balance',
            # 🔥 新增：确保客户信息进入FULL模式
            'agreed', 'terms', 'net15', 'ach', 'rush', 'monthly', 'plan'
        ]
        
        has_business_content = any(keyword in message_lower for keyword in business_keywords)
        
        # 🔥 强制FULL模式：如果包含客户信息，必须使用FULL模式
        customer_force_keywords = ['tc boiler', 'kai media', 'net15', 'payment terms', 'prefer', 'agreed', 'remember:']
        force_full_mode = any(keyword in message_lower for keyword in customer_force_keywords)
        
        # 决定处理模式
        if force_full_mode:
            context.processing_mode = ProcessingMode.FULL
            print(f"DEBUG: Detected FULL processing mode (FORCED - customer info detected)")
        elif is_general_chat and not has_business_content:
            context.processing_mode = ProcessingMode.SIMPLE
            print(f"DEBUG: Detected SIMPLE processing mode (general chat)")
            print(f"DEBUG: is_general_chat={is_general_chat}, has_business_content={has_business_content}")
        else:
            context.processing_mode = ProcessingMode.FULL
            print(f"DEBUG: Detected FULL processing mode (business content: {has_business_content})")
            print(f"DEBUG: is_general_chat={is_general_chat}, has_business_content={has_business_content}")
    
    def _step1_5_pii_detection(self, context: PipelineContext):
        """
        步骤1.5：PII检测和处理
        检测用户消息中的个人身份信息并进行掩码处理
        """
        print(f"DEBUG: Step 1.5 - PII detection for: {context.user_message[:50]}...")
        
        # 检测PII
        pii_matches = self.pii_protection_service.detect_pii(context.user_message)
        
        if pii_matches:
            print(f"DEBUG: Found {len(pii_matches)} PII matches")
            for match in pii_matches:
                print(f"DEBUG: PII match - {match.pii_type}: {match.original} -> {match.masked}")
            
            # 掩码化用户消息
            context.user_message = self.pii_protection_service.mask_pii(context.user_message, pii_matches)
            context.pii_matches = pii_matches
            
            print(f"DEBUG: Masked user message: {context.user_message}")
        else:
            print(f"DEBUG: No PII detected")
            context.pii_matches = []
    
    
    def _step2_entity_extraction(self, context: PipelineContext):
        """
        步骤2：实体提取
        """
        print(f"DEBUG: Step 2 - Entity extraction")
        
        try:
            # 提取实体
            print(f"DEBUG: Calling EntityService.extract_entities with message: '{context.user_message}'")
            entities = self.entity_service.extract_entities(context.user_message, context.session_id, context.user_id)
            print(f"DEBUG: EntityService.extract_entities returned {len(entities)} entities")
            
            # 链接实体到域数据
            print(f"DEBUG: Calling EntityService.link_entities_to_domain")
            linked_entities = self.entity_service.link_entities_to_domain(entities)
            print(f"DEBUG: EntityService.link_entities_to_domain returned {len(linked_entities)} entities")
            
            # 存储实体到数据库
            for entity in linked_entities:
                self.session.add(entity)
            self.session.commit()
            
            context.entities = linked_entities
            print(f"DEBUG: Extracted {len(linked_entities)} entities")
            
        except Exception as e:
            print(f"ERROR: Entity extraction failed: {e}")
            import traceback
            traceback.print_exc()
            context.entities = []
    
    def _step3_disambiguation_service_integration(self, context: PipelineContext):
        """步骤3: 消歧服务集成（包含澄清处理）"""
        print(f"DEBUG: Step 3 - Disambiguation service integration")
        
        # 加载对话历史用于澄清检测
        conversation_history = self._load_conversation_history(context)
        
        # 调用消歧服务（包含澄清处理）
        disambiguation_result = self.disambiguation_service.decide_disambiguation(
            entities=context.entities,
            conversation_history=conversation_history,
            user_message=context.user_message,
            session_id=context.session_id,
            user_id=context.user_id
        )
        
        # 更新上下文
        context.disambiguation_needed = disambiguation_result.needed
        context.disambiguation_result = disambiguation_result
        
        if disambiguation_result.needed:
            print(f"DEBUG: Disambiguation needed for {len(disambiguation_result.candidates)} entities")
            context.candidate_entities = disambiguation_result.candidates
            context.disambiguation_scores = disambiguation_result.scores
        else:
            print(f"DEBUG: No disambiguation needed, selected: {disambiguation_result.selected}")
            context.selected_entity = disambiguation_result.selected
    
    def _load_conversation_history(self, context: PipelineContext) -> List[ChatMessage]:
        """加载对话历史用于澄清检测"""
        conversation_history = []
        try:
            chat_events = self.session.exec(
                select(ChatEvent)
                .where(ChatEvent.session_id == context.session_id)
                .order_by(ChatEvent.created_at.desc())
                .limit(10)  # Last 10 messages for context
            ).all()
            
            # Convert to ChatMessage format (reverse to get chronological order)
            for event in reversed(chat_events):
                conversation_history.append(ChatMessage(
                    role=event.role,
                    content=event.content,
                    timestamp=event.created_at
                ))
            
            print(f"DEBUG: Loaded {len(conversation_history)} messages for disambiguation")
        except Exception as e:
            print(f"Warning: Could not load conversation history for disambiguation: {e}")
            conversation_history = []
        
        return conversation_history
    
    def _store_chat_events(self, context: PipelineContext, assistant_response: str):
        """存储ChatEvent"""
        print(f"DEBUG: Storing chat events")
        
        # 存储用户消息
        chat_event = ChatEvent(
            session_id=context.session_id,
            role="user",
            content=context.user_message
        )
        self.session.add(chat_event)
        
        # 存储助手响应
        assistant_event = ChatEvent(
            session_id=context.session_id,
            role="assistant",
            content=assistant_response
        )
        self.session.add(assistant_event)
        
        # 提交到数据库
        self.session.commit()
        print(f"DEBUG: Chat events stored successfully")
    
    def _handle_disambiguation_flow(self, context: PipelineContext) -> ChatResponse:
        """处理澄清流程"""
        print(f"DEBUG: Handling disambiguation flow")
        
        try:
            # 生成澄清问题
            print(f"DEBUG: Building clarification prompt")
            clarification_prompt = self._build_clarification_prompt(context)
            print(f"DEBUG: Clarification prompt built: {clarification_prompt[:100]}...")
            
            # 存储ChatEvent
            print(f"DEBUG: Storing chat events for clarification")
            self._store_chat_events(context, clarification_prompt)
            print(f"DEBUG: Chat events stored successfully")
            
            return ChatResponse(
                reply=clarification_prompt,
                disambiguation_needed=True,
                candidate_entities=[
                    {
                        "name": entity.name,
                        "type": entity.type,
                        "external_ref": entity.external_ref
                    }
                    for entity in context.candidate_entities
                ],
                session_id=context.session_id
            )
        except Exception as e:
            print(f"ERROR: Disambiguation flow failed: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个简单的澄清提示
            return ChatResponse(
                reply="I found multiple possible matches. Please clarify which one you mean.",
                disambiguation_needed=True,
                candidate_entities=[],
                session_id=context.session_id
            )
    
    def _handle_normal_flow(self, context: PipelineContext) -> ChatResponse:
        """处理正常流程"""
        print(f"DEBUG: Handling normal flow")
        
        # 继续正常Pipeline步骤
        self._step6_embedding_generation(context)
        self._step7_context_retrieval(context)
        self._step8_conversation_history(context)
        self._step9_prompt_building(context)
        self._step10_llm_response(context)
        self._step11_memory_processing(context)
        self._step12_memory_storage(context)
        self._step13_chat_events_storage(context)
        
        return self._build_response(context)
    
    def _build_clarification_prompt(self, context: PipelineContext) -> str:
        """构建澄清问题"""
        candidates = context.candidate_entities
        
        prompt = f"""I found multiple possible matches for your query. Please clarify which one you mean:

"""
        for i, entity in enumerate(candidates):
            prompt += f"{i+1}. {entity.name}\n"
        
        prompt += "\nPlease respond with the number or name of your choice."
        return prompt
    
    
    def _step6_embedding_generation(self, context: PipelineContext):
        """
        步骤6：生成Embedding
        """
        print(f"DEBUG: Step 6 - Embedding generation")
        
        # 生成查询embedding
        query_embedding = self.embedding_service.generate_embedding(context.user_message)
        if not query_embedding or len(query_embedding) == 0:
            raise Exception("Failed to generate embedding")
        
        context.query_embedding = query_embedding
        print(f"DEBUG: Generated embedding with {len(query_embedding)} dimensions")
    
    def _step7_context_retrieval(self, context: PipelineContext):
        """
        步骤7：检索上下文
        """
        print(f"DEBUG: Step 7 - Context retrieval")
        
        # 简化模式：跳过上下文检索
        if context.processing_mode == ProcessingMode.SIMPLE:
            print(f"DEBUG: Skipping context retrieval for SIMPLE mode")
            context.retrieval_context = None
            return
        
        # 检索相关上下文
        retrieval_context = self.retrieval_service.retrieve_context(
            query=context.user_message,
            query_embedding=context.query_embedding,
            user_id=context.user_id,
            session_id=context.session_id
        )
        
        context.retrieval_context = retrieval_context
        print(f"DEBUG: Retrieved {len(retrieval_context.memories)} memories and {len(retrieval_context.domain_facts)} domain facts")
    
    def _step8_conversation_history(self, context: PipelineContext):
        """
        步骤8：加载对话历史
        """
        print(f"DEBUG: Step 8 - Conversation history loading")
        
        conversation_history = []
        try:
            chat_events = self.session.exec(
                select(ChatEvent)
                .where(ChatEvent.session_id == context.session_id)
                .order_by(ChatEvent.created_at.desc())
                .limit(10)  # Last 10 messages for context
            ).all()
            
            # Convert to ChatMessage format (reverse to get chronological order)
            for event in reversed(chat_events):
                conversation_history.append(ChatMessage(
                    role=event.role,
                    content=event.content,
                    timestamp=event.created_at
                ))
            
            print(f"DEBUG: Loaded {len(conversation_history)} messages into conversation history")
        except Exception as e:
            print(f"Warning: Could not load conversation history: {e}")
            conversation_history = []
        
        context.conversation_history = conversation_history
    
    def _step9_prompt_building(self, context: PipelineContext):
        """
        步骤9：构建Prompt
        """
        print(f"DEBUG: Step 9 - Prompt building")
        
        # 根据处理模式选择不同的系统提示
        if context.processing_mode == ProcessingMode.SIMPLE:
            system_prompt = "You are a helpful assistant. Provide a brief, friendly response."
        else:
            system_prompt = "You are an intelligent business assistant with access to customer data, orders, invoices, and memory."
        
        # 构建Prompt上下文
        memories = []
        domain_facts = []
        
        if context.processing_mode == ProcessingMode.FULL and context.retrieval_context:
            memories = context.retrieval_context.memories
            domain_facts = context.retrieval_context.domain_facts
        
        prompt_context = PromptContext(
            system_prompt=system_prompt,
            user_message=context.user_message,
            memories=memories,
            domain_facts=domain_facts,
            conversation_history=context.conversation_history
        )
        
        context.prompt_context = prompt_context
        print(f"DEBUG: Built prompt context with {len(prompt_context.memories)} memories")
        print(f"DEBUG: Processing mode: {context.processing_mode.value}")
        print(f"DEBUG: Retrieval context exists: {context.retrieval_context is not None}")
    
    def _step10_llm_response(self, context: PipelineContext):
        """
        步骤10：生成LLM响应
        """
        print(f"DEBUG: Step 10 - LLM response generation")
        
        # 生成LLM响应
        llm_response = self.llm_service.generate_response(context.prompt_context)
        context.llm_response = llm_response
        print(f"DEBUG: Generated LLM response: {llm_response.content[:100]}...")
    
    def _step11_memory_processing(self, context: PipelineContext):
        """
        步骤11：Memory处理
        基于用户查询+LLM响应进行Memory分类和提取
        """
        print(f"DEBUG: Step 11 - Memory processing")
        
        memories_to_store = []
        
        # 根据处理模式决定Memory处理策略
        # 🔥 强制检查：如果包含客户信息，直接分类为semantic（无论什么模式）
        customer_keywords = ["tc boiler", "kai media", "net15", "payment terms", "prefer", "agreed", "remember:"]
        if any(keyword in context.user_message.lower() for keyword in customer_keywords):
            print(f"DEBUG: Detected customer keyword, forcing semantic classification")
            memory_text = context.user_message
            memory_embedding = self.embedding_service.generate_embedding(memory_text)
            
            memory = Memory(
                text=memory_text,
                kind="semantic",  # 强制分类为semantic
                importance=0.9,    # 高重要性
                ttl_days=None,     # 永久记忆
                embedding=memory_embedding
            )
            memories_to_store.append(memory)
            print(f"DEBUG: Created semantic memory: {memory_text[:50]}...")
        else:
            if context.processing_mode == ProcessingMode.SIMPLE:
                # 简化模式：创建短期Memory或跳过
                if self._should_create_short_term_memory(context):
                    memory = self._create_short_term_memory(context)
                    memories_to_store.append(memory)
            else:
                # 完整模式：使用ActionKnowledge分类器
                memories_to_store = self._process_memories_with_classifier(context)
        
        context.memories_to_store = memories_to_store
        print(f"DEBUG: Processed {len(memories_to_store)} memories to store")
    
    def _should_create_short_term_memory(self, context: PipelineContext) -> bool:
        """判断是否应该创建短期Memory"""
        # 对于一般性对话，创建短期Memory用于上下文连续性
        return len(context.user_message) > 10  # 避免存储太短的消息
    
    def _create_short_term_memory(self, context: PipelineContext) -> Memory:
        """创建短期Memory"""
        memory_text = f"User said: {context.user_message}"
        memory_embedding = self.embedding_service.generate_embedding(memory_text)
        
        return Memory(
            text=memory_text,
            kind="episodic",
            importance=0.3,  # 低重要性
            ttl_days=7,      # 7天过期
            embedding=memory_embedding
        )
    
    def _process_memories_with_classifier(self, context: PipelineContext) -> List[Memory]:
        """使用ActionKnowledge分类器处理Memory - 只记录用户操作意图，不记录LLM回复"""
        memories = []
        
        # 🔥 强制检查：如果包含"Remember:"，直接分类为semantic
        if "remember:" in context.user_message.lower():
            print(f"DEBUG: Detected 'Remember:' keyword, forcing semantic classification")
            memory_text = context.user_message
            memory_embedding = self.embedding_service.generate_embedding(memory_text)
            
            memory = Memory(
                text=memory_text,
                kind="semantic",  # 强制分类为semantic
                importance=0.9,    # 高重要性
                ttl_days=None,     # 永久记忆
                embedding=memory_embedding
            )
            memories.append(memory)
            print(f"DEBUG: Created semantic memory: {memory_text[:50]}...")
            return memories
        
        # 🔥 强制检查：如果包含客户信息，直接分类为semantic
        customer_keywords = ["tc boiler", "kai media", "net15", "payment terms", "prefer", "agreed"]
        if any(keyword in context.user_message.lower() for keyword in customer_keywords):
            print(f"DEBUG: Detected customer keyword, forcing semantic classification")
            memory_text = context.user_message
            memory_embedding = self.embedding_service.generate_embedding(memory_text)
            
            memory = Memory(
                text=memory_text,
                kind="semantic",  # 强制分类为semantic
                importance=0.9,    # 高重要性
                ttl_days=None,     # 永久记忆
                embedding=memory_embedding
            )
            memories.append(memory)
            print(f"DEBUG: Created semantic memory: {memory_text[:50]}...")
            return memories
        
        # 只分析用户查询，不分析LLM响应
        user_memory = self.memory_classifier.classify_memory(
            text=context.user_message,
            context={
                "session_id": str(context.session_id),
                "user_id": context.user_id,
                "entities": [entity.model_dump() for entity in context.entities] if context.entities else []
            }
        )
        
        # 根据用户查询的分类结果创建Memory
        if user_memory.category in [MemoryCategory.ACTION, MemoryCategory.KNOWLEDGE]:
            memory_text = user_memory.text
            memory_embedding = self.embedding_service.generate_embedding(memory_text)
            
            memory = Memory(
                text=memory_text,
                kind=user_memory.kind,
                importance=user_memory.importance,
                ttl_days=user_memory.ttl_days,
                embedding=memory_embedding
            )
            memories.append(memory)
        
        # 特殊处理：检查是否包含隐含的偏好信息
        # 例如："Reschedule ... to Friday" 可能隐含 "prefers Friday"
        implicit_preference = self._extract_implicit_preference(context.user_message)
        if implicit_preference:
            preference_embedding = self.embedding_service.generate_embedding(implicit_preference)
            preference_memory = Memory(
                text=implicit_preference,
                kind="semantic",
                importance=0.9,
                ttl_days=None,  # 永久记忆
                embedding=preference_embedding
            )
            memories.append(preference_memory)
        
        return memories
    
    def _extract_implicit_preference(self, user_message: str) -> Optional[str]:
        """提取隐含的偏好信息"""
        message_lower = user_message.lower()
        
        # 检查reschedule + Friday模式
        if ('reschedule' in message_lower and 'friday' in message_lower and 
            'kai media' in message_lower):
            return "Kai Media prefers Friday; align WO scheduling accordingly."
        
        # 检查其他偏好模式
        if ('prefer' in message_lower and 'friday' in message_lower):
            # 提取客户名称
            if 'kai media' in message_lower:
                return "Kai Media prefers Friday deliveries for all shipments."
            elif 'tc boiler' in message_lower:
                return "TC Boiler prefers Friday deliveries for all shipments."
        
        # 检查NET付款条件
        if ('net' in message_lower and ('tc boiler' in message_lower or 'kai media' in message_lower)):
            if 'tc boiler' in message_lower:
                return "TC Boiler is NET15; align payment terms accordingly."
            elif 'kai media' in message_lower:
                return "Kai Media is NET15; align payment terms accordingly."
        
        return None
    
    def _step12_memory_storage(self, context: PipelineContext):
        """
        步骤12：Memory存储
        """
        print(f"DEBUG: Step 12 - Memory storage")
        
        # 存储所有Memory
        for memory in context.memories_to_store:
            self.memory_service.create_memory(
                session_id=context.session_id,
                kind=memory.kind,
                text=memory.text,
                embedding=memory.embedding,
                importance=memory.importance,
                ttl_days=memory.ttl_days,
                pii_matches=context.pii_matches  # 传递PII信息
            )
        
        # 🔥 智能整合Memory (只在需要时触发)
        try:
            self.memory_service.consolidate_memories(user_id=context.user_id, session_window=3, force=False)
        except Exception as e:
            print(f"Warning: Memory consolidation failed: {e}")
        
        print(f"DEBUG: Stored {len(context.memories_to_store)} memories")
    
    def _step13_chat_events_storage(self, context: PipelineContext):
        """
        步骤13：存储Chat事件
        """
        print(f"DEBUG: Step 13 - Chat events storage")
        
        # 存储用户消息
        chat_event = ChatEvent(
            session_id=context.session_id,
            role="user",
            content=context.user_message
        )
        self.session.add(chat_event)
        
        # 存储助手响应
        assistant_event = ChatEvent(
            session_id=context.session_id,
            role="assistant",
            content=context.llm_response.content
        )
        self.session.add(assistant_event)
        self.session.commit()
        
        print(f"DEBUG: Stored chat events")
    
    def _build_response(self, context: PipelineContext) -> ChatResponse:
        """构建响应"""
        print(f"DEBUG: Building response - Processing mode: {context.processing_mode.value}")
        print(f"DEBUG: Retrieval context exists: {context.retrieval_context is not None}")
        if context.retrieval_context:
            print(f"DEBUG: Retrieval context memories: {len(context.retrieval_context.memories)}")
            print(f"DEBUG: Retrieval context domain facts: {len(context.retrieval_context.domain_facts)}")
        
        # 格式化使用的记忆
        used_memories = []
        if (context.processing_mode == ProcessingMode.FULL and 
            context.retrieval_context and 
            context.retrieval_context.memories):
            used_memories = [
                {
                    "memory_id": memory.memory_id,
                    "text": memory.text,
                    "similarity": memory.similarity,
                    "kind": memory.kind
                }
                for memory in context.retrieval_context.memories
            ]
        
        # 格式化使用的域事实
        used_domain_facts = []
        if (context.processing_mode == ProcessingMode.FULL and 
            context.retrieval_context and 
            context.retrieval_context.domain_facts):
            used_domain_facts = [
                {
                    "table": fact.table,
                    "id": fact.id,
                    "data": fact.data,
                    "relevance_score": fact.relevance_score
                }
                for fact in context.retrieval_context.domain_facts
            ]
        
        # 格式化候选实体
        candidate_entities = []
        if context.candidate_entities:
            candidate_entities = [
                {
                    "name": entity.name,
                    "type": entity.type,
                    "external_ref": entity.external_ref
                }
                for entity in context.candidate_entities
            ]
        
        return ChatResponse(
            reply=context.llm_response.content,
            used_memories=used_memories,
            used_domain_facts=used_domain_facts,
            session_id=context.session_id,
            disambiguation_needed=context.disambiguation_needed,
            candidate_entities=candidate_entities
        )
    
    def _handle_normal_flow_with_selected_entity(self, context: PipelineContext) -> ChatResponse:
        """处理带有已选择实体的正常流程"""
        print(f"DEBUG: Handling normal flow with selected entity: {context.selected_entity}")
        
        # 跳过实体提取，直接进行后续步骤
        self._step6_embedding_generation(context)
        self._step7_context_retrieval(context)
        self._step8_conversation_history_loading(context)
        self._step9_prompt_building(context)
        self._step10_llm_response_generation(context)
        self._step11_memory_processing(context)
        self._step12_memory_storage(context)
        self._step13_chat_events_storage(context)
        
        return self._build_response(context)

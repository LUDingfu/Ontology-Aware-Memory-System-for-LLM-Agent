"""Disambiguation Service for Entity Disambiguation"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from sqlmodel import Session

from app.models.memory import Entity, Memory
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import MemoryService
from app.services.alias_mapping_service import AliasMappingService


@dataclass
class DisambiguationResult:
    """消歧结果"""
    needed: bool
    selected: Optional[Entity] = None
    candidates: Optional[List[Entity]] = None
    scores: Optional[List[float]] = None


class DisambiguationService:
    """消歧服务"""
    
    def __init__(self, session: Session):
        self.session = session
        self.threshold = 0.1  # 分数差异阈值
        self.embedding_service = EmbeddingService()
        self.memory_service = MemoryService(session)
        self.alias_mapping_service = AliasMappingService(session)
    
    def decide_disambiguation(self, entities: List[Entity], conversation_history: List = None, user_message: str = None, session_id: str = None, user_id: str = None) -> DisambiguationResult:
        """
        消歧决策（包含澄清处理）
        
        Args:
            entities: 候选实体列表
            conversation_history: 对话历史
            user_message: 用户消息
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            DisambiguationResult: 消歧结果
        """
        print(f"DEBUG: DisambiguationService.decide_disambiguation() called with {len(entities)} entities")
        print(f"DEBUG: conversation_history: {conversation_history is not None}, length: {len(conversation_history) if conversation_history else 0}")
        print(f"DEBUG: user_message: {user_message}")
        
        # 首先检查是否是澄清回应
        if conversation_history and self._is_clarification_response(conversation_history):
            print(f"DEBUG: Detected clarification response, processing...")
            return self._process_clarification_from_history(user_message, entities, session_id, user_id)
        else:
            print(f"DEBUG: Not a clarification response or no conversation history")
        
        if not entities:
            print(f"DEBUG: No entities found, returning no disambiguation needed")
            return DisambiguationResult(needed=False, selected=None)
        
        if len(entities) == 1:
            print(f"DEBUG: Single entity found: {entities[0].name}, no disambiguation needed")
            return DisambiguationResult(needed=False, selected=entities[0])
        
        # 计算分数
        scores = [self._calculate_entity_score(entity) for entity in entities]
        print(f"DEBUG: Entity scores: {[(entity.name, score) for entity, score in zip(entities, scores)]}")
        
        max_score = max(scores)
        second_max = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0
        score_difference = max_score - second_max
        
        print(f"DEBUG: Max score: {max_score}, Second max: {second_max}, Difference: {score_difference}")
        
        # 消歧决策 - 降低阈值，更容易触发澄清
        if score_difference > 0.05:  # 降低阈值从0.1到0.05
            # 分数差异大，直接选择
            selected_entity = entities[scores.index(max_score)]
            print(f"DEBUG: Score difference large enough, selecting: {selected_entity.name}")
            return DisambiguationResult(needed=False, selected=selected_entity)
        else:
            # 分数接近，需要澄清
            print(f"DEBUG: Score difference too small, disambiguation needed")
            return DisambiguationResult(
                needed=True, 
                candidates=entities, 
                scores=scores
            )
    
    def process_clarification(self, user_response: str, candidates: List[Entity], session_id: str, user_id: str = "default") -> Entity:
        """
        处理用户澄清
        
        Args:
            user_response: 用户澄清响应
            candidates: 候选实体列表
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            Entity: 选定的实体
        """
        print(f"DEBUG: Processing clarification: '{user_response}' for {len(candidates)} candidates")
        
        # 解析用户选择
        selected_entity = self._parse_user_selection(user_response, candidates)
        print(f"DEBUG: Selected entity: {selected_entity.name}")
        
        # 创建exact match alias mapping
        self._create_exact_match_alias(user_response, selected_entity, user_id)
        
        return selected_entity
    
    def _is_clarification_response(self, conversation_history: List) -> bool:
        """检查最近的对话是否是澄清回应"""
        print(f"DEBUG: _is_clarification_response called with {len(conversation_history)} messages")
        
        if len(conversation_history) < 2:
            print(f"DEBUG: Not enough conversation history ({len(conversation_history)} messages)")
            return False
        
        # 检查助手是否刚问了澄清问题
        last_assistant_msg = None
        for msg in reversed(conversation_history):
            if msg.role == "assistant":
                last_assistant_msg = msg.content
                break
        
        if not last_assistant_msg:
            print(f"DEBUG: No assistant message found")
            return False
        
        print(f"DEBUG: Last assistant message: {last_assistant_msg[:100]}...")
        
        # 检查是否包含澄清关键词
        clarification_keywords = ["clarify", "which one", "multiple matches", "please choose", "found multiple possible", "please respond with the number"]
        is_clarification = any(keyword in last_assistant_msg.lower() for keyword in clarification_keywords)
        
        print(f"DEBUG: Is clarification response: {is_clarification}")
        return is_clarification
    
    def _process_clarification_from_history(self, user_message: str, entities: List[Entity], session_id: str, user_id: str) -> DisambiguationResult:
        """从对话历史处理澄清"""
        print(f"DEBUG: Processing clarification from history: '{user_message}'")
        
        # 解析用户选择
        selected_entity = self._parse_user_selection(user_message, entities)
        
        if selected_entity:
            print(f"DEBUG: Clarification successful, selected: {selected_entity.name}")
            # 创建exact match alias mapping
            self._create_exact_match_alias(user_message, selected_entity, user_id)
            return DisambiguationResult(needed=False, selected=selected_entity)
        else:
            print(f"DEBUG: Clarification failed, re-asking")
            return DisambiguationResult(
                needed=True, 
                candidates=entities, 
                scores=[self._calculate_entity_score(entity) for entity in entities]
            )
    
    def _calculate_entity_score(self, entity: Entity) -> float:
        """计算实体分数"""
        # 基于confidence和外部引用计算分数
        confidence = entity.external_ref.get('confidence', 'exact')
        
        if confidence == 'exact':
            return 1.0
        elif confidence == 'fuzzy':
            return 0.8
        else:
            return 0.5
    
    def _parse_user_selection(self, user_response: str, candidates: List[Entity]) -> Entity:
        """解析用户选择"""
        response_lower = user_response.lower()
        print(f"DEBUG: Parsing user selection from: '{user_response}'")
        
        # 尝试按数字选择
        for i, entity in enumerate(candidates):
            if str(i+1) in response_lower:
                print(f"DEBUG: Found number selection: {i+1} -> {entity.name}")
                return entity
        
        # 尝试按名称选择
        for entity in candidates:
            if entity.name.lower() in response_lower:
                print(f"DEBUG: Found name selection: {entity.name}")
                return entity
        
        # 尝试部分匹配
        for entity in candidates:
            entity_words = entity.name.lower().split()
            response_words = response_lower.split()
            
            # 检查是否有足够的词匹配
            matches = sum(1 for word in entity_words if word in response_words)
            if matches >= len(entity_words) * 0.5:  # 至少50%的词匹配
                print(f"DEBUG: Found partial match: {entity.name}")
                return entity
        
        # 默认选择第一个
        print(f"DEBUG: No clear selection found, defaulting to first candidate: {candidates[0].name}")
        return candidates[0]
    
    def _create_exact_match_alias(self, user_input: str, selected_entity: Entity, user_id: str):
        """创建exact match别名映射"""
        print(f"DEBUG: Creating exact match alias mapping: '{user_input}' -> '{selected_entity.name}'")
        
        # 使用AliasMappingService存储exact match
        success = self.alias_mapping_service.store_alias_mapping(
            user_id=user_id,
            alias_text=user_input,
            entity_name=selected_entity.name,
            entity_id=selected_entity.external_ref.get("id", "")
        )
        
        if success:
            print(f"DEBUG: Exact match alias mapping created successfully")
        else:
            print(f"ERROR: Failed to create exact match alias mapping")
        
        print(f"DEBUG: Alias mapping created successfully")

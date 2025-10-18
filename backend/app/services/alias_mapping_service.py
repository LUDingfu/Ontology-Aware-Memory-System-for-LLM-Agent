"""简化的别名映射服务，使用exact match和Memory存储"""

from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlmodel import Session, select

from app.models.memory import Memory, Entity
from app.services.embedding_service import EmbeddingService


class AliasMappingService:
    """别名映射服务 - 使用exact match和Memory存储"""
    
    def __init__(self, session: Session):
        self.session = session
        self.embedding_service = EmbeddingService()
    
    def store_alias_mapping(self, user_id: str, alias_text: str, entity_name: str, entity_id: str) -> bool:
        """
        存储别名映射到Memory表
        
        Args:
            user_id: 用户ID
            alias_text: 别名文本（如"Kai"）
            entity_name: 实体名称（如"Kai Media"）
            entity_id: 实体ID
            
        Returns:
            bool: 是否存储成功
        """
        try:
            print(f"DEBUG: Storing alias mapping: '{alias_text}' -> '{entity_name}' (ID: {entity_id})")
            
            # 创建semantic memory存储别名映射
            alias_memory = Memory(
                text=f"Alias mapping: '{alias_text}' refers to '{entity_name}' (ID: {entity_id})",
                kind="semantic",
                importance=0.8,
                ttl_days=None,  # 永久记忆
                embedding=self.embedding_service.generate_embedding(f"{alias_text} {entity_name}"),
                external_ref={
                    "type": "alias_mapping",
                    "alias_text": alias_text.lower(),
                    "entity_name": entity_name,
                    "entity_id": entity_id,
                    "user_id": user_id
                }
            )
            
            self.session.add(alias_memory)
            self.session.commit()
            
            print(f"DEBUG: Alias mapping stored successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to store alias mapping: {e}")
            self.session.rollback()
            return False
    
    def get_exact_match_entity(self, user_id: str, query_text: str) -> Optional[Dict[str, Any]]:
        """
        获取exact match的实体
        
        Args:
            user_id: 用户ID
            query_text: 查询文本
            
        Returns:
            Dict: 匹配的实体信息，如果没有匹配返回None
        """
        try:
            print(f"DEBUG: Looking for exact match for: '{query_text}'")
            
            # 查询exact match的别名映射
            alias_memory = self.session.exec(
                select(Memory).where(
                    Memory.kind == "semantic",
                    Memory.external_ref["type"] == "alias_mapping",
                    Memory.external_ref["alias_text"] == query_text.lower(),
                    Memory.external_ref["user_id"] == user_id
                )
            ).first()
            
            if alias_memory:
                external_ref = alias_memory.external_ref
                print(f"DEBUG: Found exact match: '{query_text}' -> '{external_ref['entity_name']}'")
                return {
                    "name": external_ref["entity_name"],
                    "id": external_ref["entity_id"],
                    "confidence": "exact"
                }
            
            print(f"DEBUG: No exact match found for: '{query_text}'")
            return None
            
        except Exception as e:
            print(f"ERROR: Failed to get exact match: {e}")
            return None
    
    def store_multilingual_mapping(self, user_id: str, foreign_text: str, english_text: str) -> bool:
        """
        存储多语种映射
        
        Args:
            user_id: 用户ID
            foreign_text: 外语文本
            english_text: 对应的英语文本
            
        Returns:
            bool: 是否存储成功
        """
        try:
            print(f"DEBUG: Storing multilingual mapping: '{foreign_text}' -> '{english_text}'")
            
            # 创建semantic memory存储多语种映射
            multilingual_memory = Memory(
                text=f"Multilingual mapping: '{foreign_text}' means '{english_text}'",
                kind="semantic",
                importance=0.7,
                ttl_days=None,  # 永久记忆
                embedding=self.embedding_service.generate_embedding(f"{foreign_text} {english_text}"),
                external_ref={
                    "type": "multilingual_mapping",
                    "foreign_text": foreign_text.lower(),
                    "english_text": english_text,
                    "user_id": user_id
                }
            )
            
            self.session.add(multilingual_memory)
            self.session.commit()
            
            print(f"DEBUG: Multilingual mapping stored successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to store multilingual mapping: {e}")
            self.session.rollback()
            return False
    
    def translate_to_english(self, user_id: str, foreign_text: str) -> str:
        """
        将外语翻译为英语
        
        Args:
            user_id: 用户ID
            foreign_text: 外语文本
            
        Returns:
            str: 英语文本，如果没有映射则返回原文本
        """
        try:
            print(f"DEBUG: Translating to English: '{foreign_text}'")
            
            # 查询多语种映射
            multilingual_memory = self.session.exec(
                select(Memory).where(
                    Memory.kind == "semantic",
                    Memory.external_ref["type"] == "multilingual_mapping",
                    Memory.external_ref["foreign_text"] == foreign_text.lower(),
                    Memory.external_ref["user_id"] == user_id
                )
            ).first()
            
            if multilingual_memory:
                english_text = multilingual_memory.external_ref["english_text"]
                print(f"DEBUG: Translation found: '{foreign_text}' -> '{english_text}'")
                return english_text
            
            print(f"DEBUG: No translation found for: '{foreign_text}', using original")
            return foreign_text
            
        except Exception as e:
            print(f"ERROR: Failed to translate: {e}")
            return foreign_text

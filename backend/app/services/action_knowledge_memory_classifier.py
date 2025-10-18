"""Action vs Knowledge Memory Classifier - 基于Action vs Knowledge的Memory分类器"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI

from app.core.config import settings


class MemoryCategory(Enum):
    """Memory分类：基于Action vs Knowledge"""
    ACTION = "action"           # 系统做了什么 (Episodic)
    KNOWLEDGE = "knowledge"     # 用户偏好/知识 (Semantic)
    STATUS = "status"           # 状态信息 (Episodic)
    PREFERENCE = "preference"   # 用户偏好 (Semantic)


@dataclass
class ClassifiedMemory:
    """分类后的Memory"""
    text: str
    category: MemoryCategory
    kind: str  # episodic, semantic, profile, commitment
    importance: float
    ttl_days: Optional[int]
    entities: List[str]
    confidence: float
    reasoning: str  # 分类原因


class ActionKnowledgeMemoryClassifier:
    """
    基于Action vs Knowledge的Memory分类器
    
    核心理念：
    - ACTION: 系统做了什么，LLM做了什么 → Episodic Memory
    - KNOWLEDGE: 用户偏好、规则、知识 → Semantic Memory
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
    
    def classify_memory(self, text: str, context: Optional[Dict] = None) -> ClassifiedMemory:
        """
        分类单个Memory文本
        
        Args:
            text: Memory文本
            context: 上下文信息（用户查询、LLM响应等）
            
        Returns:
            ClassifiedMemory对象
        """
        try:
            # 使用LLM进行智能分类
            classification_result = self._llm_classify(text, context)
            
            # 转换为标准格式
            return self._convert_to_classified_memory(text, classification_result)
            
        except Exception as e:
            print(f"Error in LLM classification: {e}")
            # Fallback到规则分类
            classification_result = self._rule_based_classify(text)
            return self._convert_to_classified_memory(text, classification_result)
    
    def _llm_classify(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """使用LLM进行智能分类"""
        
        system_prompt = """你是一个专业的Memory分类专家。你的任务是将Memory文本分类为ACTION或KNOWLEDGE。

分类标准：
1. ACTION (系统做了什么) → episodic:
   - 系统执行了某个操作 (发送邮件、创建订单、更新状态等)
   - LLM生成了某个内容 (草稿、报告、建议等)
   - 完成了某个任务 (处理、执行、操作等)
   - 状态变化 (订单状态、发票状态等)
   - 示例: "Email drafted for Kai Media", "Work order rescheduled", "Invoice sent"

2. KNOWLEDGE (用户偏好/知识) → semantic:
   - 用户表达的偏好 (喜欢、不喜欢、习惯等)
   - 业务规则和策略 (付款条件、交付偏好等)
   - 客户信息和特征 (行业、规模、特点等)
   - 长期知识 (政策、流程、标准等)
   - 包含"Remember"、"prefer"、"like"等关键词的语句
   - 示例: "Remember: Kai Media prefers Friday deliveries", "TC Boiler is NET15", "Customer prefers ACH"

🔥 强制分类规则：
- 任何包含"Remember:"的语句 → KNOWLEDGE/semantic
- 任何包含"prefer"、"like"、"always"、"never"的语句 → KNOWLEDGE/semantic
- 任何包含"is NET"、"terms"、"payment"的语句 → KNOWLEDGE/semantic
- 任何包含"TC Boiler"、"Kai Media"等客户名称的语句 → KNOWLEDGE/semantic
- 任何包含"agreed"、"terms"、"NET15"、"ACH"的语句 → KNOWLEDGE/semantic

请分析以下Memory文本，并返回JSON格式的分类结果：
{
    "category": "ACTION" 或 "KNOWLEDGE",
    "kind": "episodic" 或 "semantic",
    "importance": 0.0-1.0,
    "ttl_days": null 或 数字,
    "reasoning": "分类原因",
    "confidence": 0.0-1.0
}"""

        user_prompt = f"""Memory文本: "{text}"

上下文信息: {context or "无"}

请分析这个Memory文本属于ACTION还是KNOWLEDGE，并说明原因。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 尝试解析JSON
            try:
                # 提取JSON部分
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_text = result_text[json_start:json_end]
                    return json.loads(json_text)
            except json.JSONDecodeError:
                pass
            
            # 如果JSON解析失败，使用文本分析
            return self._parse_text_classification(result_text)
            
        except Exception as e:
            print(f"LLM classification error: {e}")
            return self._rule_based_classify(text)
    
    def _parse_text_classification(self, text: str) -> Dict[str, Any]:
        """解析LLM的文本分类结果"""
        text_lower = text.lower()
        
        # 判断分类
        if "action" in text_lower:
            category = "ACTION"
            kind = "episodic"
            importance = 0.8
            ttl_days = 30
        elif "knowledge" in text_lower:
            category = "KNOWLEDGE"
            kind = "semantic"
            importance = 0.9
            ttl_days = None
        else:
            # 默认分类
            category = "ACTION"
            kind = "episodic"
            importance = 0.7
            ttl_days = 30
        
        return {
            "category": category,
            "kind": kind,
            "importance": importance,
            "ttl_days": ttl_days,
            "reasoning": text[:100],
            "confidence": 0.7
        }
    
    def _rule_based_classify(self, text: str) -> Dict[str, Any]:
        """基于规则的分类（Fallback）"""
        text_lower = text.lower()
        
        # ACTION关键词
        action_keywords = [
            "drafted", "sent", "created", "completed", "finished", "done",
            "rescheduled", "updated", "processed", "executed", "performed",
            "email", "work order", "invoice", "order", "task"
        ]
        
        # KNOWLEDGE关键词
        knowledge_keywords = [
            "prefers", "likes", "dislikes", "always", "never", "usually",
            "policy", "rule", "standard", "preference", "habit", "custom",
            "net15", "net30", "ach", "credit card", "friday", "monday"
        ]
        
        action_score = sum(1 for keyword in action_keywords if keyword in text_lower)
        knowledge_score = sum(1 for keyword in knowledge_keywords if keyword in text_lower)
        
        if action_score > knowledge_score:
            return {
                "category": "ACTION",
                "kind": "episodic",
                "importance": 0.8,
                "ttl_days": 30,
                "reasoning": f"Rule-based: ACTION keywords found ({action_score})",
                "confidence": 0.6
            }
        elif knowledge_score > action_score:
            return {
                "category": "KNOWLEDGE",
                "kind": "semantic",
                "importance": 0.9,
                "ttl_days": None,
                "reasoning": f"Rule-based: KNOWLEDGE keywords found ({knowledge_score})",
                "confidence": 0.6
            }
        else:
            # 默认分类
            return {
                "category": "ACTION",
                "kind": "episodic",
                "importance": 0.7,
                "ttl_days": 30,
                "reasoning": "Rule-based: Default ACTION classification",
                "confidence": 0.5
            }
    
    def _convert_to_classified_memory(self, text: str, classification: Dict[str, Any]) -> ClassifiedMemory:
        """转换为ClassifiedMemory对象"""
        
        # 提取实体
        entities = self._extract_entities(text)
        
        # 安全地转换category
        category_value = classification["category"].upper()
        if category_value == "ACTION":
            category = MemoryCategory.ACTION
        elif category_value == "KNOWLEDGE":
            category = MemoryCategory.KNOWLEDGE
        elif category_value == "STATUS":
            category = MemoryCategory.STATUS
        elif category_value == "PREFERENCE":
            category = MemoryCategory.PREFERENCE
        else:
            category = MemoryCategory.ACTION  # 默认值
        
        return ClassifiedMemory(
            text=text,
            category=category,
            kind=classification["kind"],
            importance=classification["importance"],
            ttl_days=classification["ttl_days"],
            entities=entities,
            confidence=classification["confidence"],
            reasoning=classification["reasoning"]
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        entities = []
        
        # 客户名称
        customer_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        customers = re.findall(customer_pattern, text)
        entities.extend(customers)
        
        # 订单号
        order_pattern = r'\b(SO-\d+|INV-\d+|WO-\d+)\b'
        orders = re.findall(order_pattern, text)
        entities.extend(orders)
        
        return list(set(entities))  # 去重
    
    def classify_batch(self, texts: List[str], context: Optional[Dict] = None) -> List[ClassifiedMemory]:
        """批量分类"""
        results = []
        for text in texts:
            classified = self.classify_memory(text, context)
            results.append(classified)
        return results
    
    def get_classification_stats(self, memories: List[ClassifiedMemory]) -> Dict[str, Any]:
        """获取分类统计信息"""
        stats = {
            "total": len(memories),
            "action_count": 0,
            "knowledge_count": 0,
            "episodic_count": 0,
            "semantic_count": 0,
            "avg_confidence": 0.0
        }
        
        if not memories:
            return stats
        
        total_confidence = 0.0
        
        for memory in memories:
            if memory.category == MemoryCategory.ACTION:
                stats["action_count"] += 1
            elif memory.category == MemoryCategory.KNOWLEDGE:
                stats["knowledge_count"] += 1
            
            if memory.kind == "episodic":
                stats["episodic_count"] += 1
            elif memory.kind == "semantic":
                stats["semantic_count"] += 1
            
            total_confidence += memory.confidence
        
        stats["avg_confidence"] = total_confidence / len(memories)
        
        return stats


# 使用示例
if __name__ == "__main__":
    classifier = ActionKnowledgeMemoryClassifier()
    
    # 测试用例
    test_cases = [
        "Email drafted for Kai Media regarding invoice",
        "Kai Media prefers Friday deliveries",
        "Work order rescheduled for SO-1001",
        "TC Boiler is NET15 and prefers ACH",
        "Invoice sent to customer",
        "Customer prefers morning deliveries"
    ]
    
    print("=== Action vs Knowledge Memory Classification Test ===")
    
    for text in test_cases:
        result = classifier.classify_memory(text)
        print(f"\nText: {text}")
        print(f"Category: {result.category.value}")
        print(f"Kind: {result.kind}")
        print(f"Importance: {result.importance}")
        print(f"TTL: {result.ttl_days}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")
        print("-" * 50)

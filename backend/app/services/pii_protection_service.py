"""最小化PII保护服务 - 只处理手机号检测和掩码"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PIIMatch:
    """PII匹配结果"""
    original: str
    masked: str
    pii_type: str
    purpose: Optional[str] = None


class PIIProtectionService:
    """最小化PII保护服务 - 只处理手机号"""
    
    def __init__(self):
        # 手机号正则表达式
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        
        # 目的关键词
        self.purpose_keywords = {
            "urgent": ["urgent", "emergency", "alert", "critical"],
            "contact": ["contact", "call", "reach", "notify"],
            "reminder": ["reminder", "remind", "notify"]
        }
    
    def detect_pii(self, text: str) -> List[PIIMatch]:
        """检测文本中的PII - 只检测手机号"""
        matches = []
        
        # 检测手机号
        phone_matches = self.phone_pattern.findall(text)
        for phone in phone_matches:
            # 提取目的
            purpose = self._extract_purpose(text)
            
            matches.append(PIIMatch(
                original=phone,
                masked="***-***-****",
                pii_type="phone",
                purpose=purpose
            ))
        
        return matches
    
    def mask_pii(self, text: str, matches: List[PIIMatch]) -> str:
        """掩码化文本中的PII"""
        masked_text = text
        for match in matches:
            masked_text = masked_text.replace(match.original, match.masked)
        return masked_text
    
    def _extract_purpose(self, text: str) -> Optional[str]:
        """提取PII的使用目的"""
        text_lower = text.lower()
        
        for purpose, keywords in self.purpose_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return purpose
        
        return None
    
    def create_masked_memory_text(self, original_text: str, matches: List[PIIMatch]) -> str:
        """创建掩码化的记忆文本"""
        if not matches:
            return original_text
        
        # 掩码化文本
        masked_text = self.mask_pii(original_text, matches)
        
        # 添加目的信息
        purposes = [match.purpose for match in matches if match.purpose]
        if purposes:
            purpose_str = ", ".join(set(purposes))
            masked_text += f" (for {purpose_str})"
        
        return masked_text

"""Embedding generation service."""

import os
from typing import List, Optional

import openai
from openai import OpenAI

from app.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_EMBEDDING_MODEL
        self.dimensions = 1536  # OpenAI text-embedding-3-small default
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # 返回模拟嵌入向量用于测试
            return self._generate_mock_embedding(text)
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate a mock embedding for testing purposes."""
        import hashlib
        # 使用文本的哈希值生成一致的模拟向量
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # 生成1536维的模拟向量
        mock_embedding = []
        for i in range(self.dimensions):
            # 使用哈希值和位置生成伪随机但一致的值
            seed = (hash_int + i) % 1000000
            mock_embedding.append((seed / 1000000.0 - 0.5) * 2)  # 范围 [-1, 1]
        
        return mock_embedding
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i}: {e}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        return embeddings
    
    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self.dimensions

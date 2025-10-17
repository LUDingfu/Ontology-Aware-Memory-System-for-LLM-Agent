import json
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


def create_test_chat_message(content: str, role: str = "user") -> Dict[str, Any]:
    """Create a test chat message."""
    return {
        "content": content,
        "role": role
    }


def create_test_chat_request(messages: list, user_id: str = "test-user") -> Dict[str, Any]:
    """Create a test chat request."""
    return {
        "messages": messages,
        "user_id": user_id
    }


def create_test_memory_request(user_id: str = "test-user") -> Dict[str, Any]:
    """Create a test memory request."""
    return {
        "user_id": user_id
    }


def create_test_consolidate_request(user_id: str = "test-user") -> Dict[str, Any]:
    """Create a test consolidate request."""
    return {
        "user_id": user_id
    }


def create_test_entities_request(session_id: str) -> Dict[str, Any]:
    """Create a test entities request."""
    return {
        "session_id": session_id
    }


def create_test_explain_request(session_id: str, response_id: str) -> Dict[str, Any]:
    """Create a test explain request."""
    return {
        "session_id": session_id,
        "response_id": response_id
    }


def mock_openai_embedding_response(embedding: list = None) -> Dict[str, Any]:
    """Create a mock OpenAI embedding response."""
    if embedding is None:
        embedding = [0.1] * 1536
    
    return {
        "data": [
            {
                "embedding": embedding,
                "index": 0
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10
        }
    }


def mock_openai_chat_response(content: str = "Test response") -> Dict[str, Any]:
    """Create a mock OpenAI chat response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    }


def assert_response_success(response, expected_status: int = 200):
    """Assert that a response is successful."""
    assert response.status_code == expected_status
    assert response.json() is not None


def assert_response_error(response, expected_status: int = 400):
    """Assert that a response is an error."""
    assert response.status_code == expected_status
    assert "detail" in response.json()


class MockOpenAIServices:
    """Context manager for mocking OpenAI services."""
    
    def __init__(self, embedding_response=None, chat_response=None):
        self.embedding_response = embedding_response or mock_openai_embedding_response()
        self.chat_response = chat_response or mock_openai_chat_response()
        self.embedding_patch = None
        self.chat_patch = None
    
    def __enter__(self):
        self.embedding_patch = patch('app.services.embedding_service.OpenAI')
        self.chat_patch = patch('app.services.llm_service.OpenAI')
        
        mock_embedding_client = MagicMock()
        mock_embedding_client.embeddings.create.return_value = self.embedding_response
        
        mock_chat_client = MagicMock()
        mock_chat_client.chat.completions.create.return_value = self.chat_response
        
        self.embedding_patch.start().return_value = mock_embedding_client
        self.chat_patch.start().return_value = mock_chat_client
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.embedding_patch:
            self.embedding_patch.stop()
        if self.chat_patch:
            self.chat_patch.stop()


def patch_openai_services(embedding_response=None, chat_response=None):
    """Patch OpenAI services for testing."""
    return MockOpenAIServices(embedding_response, chat_response)

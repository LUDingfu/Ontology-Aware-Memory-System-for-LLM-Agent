import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from tests.utils.test_utils import (
    create_test_chat_request,
    create_test_chat_message,
    mock_openai_embedding_response,
    mock_openai_chat_response,
    assert_response_success,
    patch_openai_services
)


class TestChatAPI:
    """Test cases for the chat API endpoint."""

    def test_chat_endpoint_success(self, client: TestClient, session_id: str):
        """Test successful chat request."""
        with patch_openai_services():
            request_data = create_test_chat_request([
                create_test_chat_message("Hello, how are you?")
            ])
            
            response = client.post("/api/v1/chat/", json=request_data)
            assert_response_success(response)

    def test_chat_endpoint_empty_messages(self, client: TestClient):
        """Test chat request with empty messages."""
        request_data = create_test_chat_request([])
        
        response = client.post("/api/v1/chat/", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_chat_endpoint_missing_user_id(self, client: TestClient):
        """Test chat request without user_id."""
        request_data = {
            "messages": [create_test_chat_message("Hello")]
        }
        
        response = client.post("/api/v1/chat/", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_chat_endpoint_invalid_message_role(self, client: TestClient):
        """Test chat request with invalid message role."""
        request_data = create_test_chat_request([
            {
                "content": "Hello",
                "role": "invalid_role"
            }
        ])
        
        response = client.post("/api/v1/chat/", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_chat_endpoint_multiple_messages(self, client: TestClient):
        """Test chat request with multiple messages."""
        with patch_openai_services():
            request_data = create_test_chat_request([
                create_test_chat_message("Hello", "user"),
                create_test_chat_message("Hi there!", "assistant"),
                create_test_chat_message("How can I help you?", "user")
            ])
            
            response = client.post("/api/v1/chat/", json=request_data)
            assert_response_success(response)

    def test_chat_endpoint_openai_error(self, client: TestClient):
        """Test chat request when OpenAI API fails."""
        with patch('app.services.embedding_service.OpenAI') as mock_embedding, \
             patch('app.services.llm_service.OpenAI') as mock_chat:
            
            # Mock embedding service to raise an exception
            mock_embedding.return_value.embeddings.create.side_effect = Exception("OpenAI API error")
            
            request_data = create_test_chat_request([
                create_test_chat_message("Hello")
            ])
            
            response = client.post("/api/v1/chat/", json=request_data)
            assert response.status_code == 500

    def test_chat_endpoint_memory_storage(self, client: TestClient):
        """Test that chat messages are stored in memory."""
        with patch_openai_services():
            request_data = create_test_chat_request([
                create_test_chat_message("I need help with my order")
            ])
            
            response = client.post("/api/v1/chat/", json=request_data)
            assert_response_success(response)
            
            # Verify response contains expected fields
            response_data = response.json()
            assert "response" in response_data
            assert "session_id" in response_data
            assert "memory_ids" in response_data

    def test_chat_endpoint_entity_extraction(self, client: TestClient):
        """Test that entities are extracted from chat messages."""
        with patch_openai_services():
            request_data = create_test_chat_request([
                create_test_chat_message("I want to check the status of order #12345")
            ])
            
            response = client.post("/api/v1/chat/", json=request_data)
            assert_response_success(response)
            
            # Verify response contains entity information
            response_data = response.json()
            assert "entities" in response_data

    def test_chat_endpoint_context_retrieval(self, client: TestClient):
        """Test that relevant context is retrieved for chat."""
        with patch_openai_services():
            request_data = create_test_chat_request([
                create_test_chat_message("Tell me about customer John Smith")
            ])
            
            response = client.post("/api/v1/chat/", json=request_data)
            assert_response_success(response)
            
            # Verify response contains context information
            response_data = response.json()
            assert "context" in response_data

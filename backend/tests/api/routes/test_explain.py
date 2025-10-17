import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from tests.utils.test_utils import (
    create_test_explain_request,
    assert_response_success,
    assert_response_error
)


class TestExplainAPI:
    """Test cases for the explain API endpoint (bonus feature)."""

    def test_explain_endpoint_success(self, client: TestClient, session_id: str):
        """Test successful explain request."""
        request_data = create_test_explain_request(session_id, "response-123")
        
        response = client.get("/api/v1/explain/", params=request_data)
        assert_response_success(response)

    def test_explain_endpoint_missing_session_id(self, client: TestClient):
        """Test explain request without session_id."""
        request_data = {
            "response_id": "response-123"
        }
        
        response = client.get("/api/v1/explain/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_explain_endpoint_missing_response_id(self, client: TestClient, session_id: str):
        """Test explain request without response_id."""
        request_data = {
            "session_id": session_id
        }
        
        response = client.get("/api/v1/explain/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_explain_endpoint_invalid_session_id(self, client: TestClient):
        """Test explain request with invalid session_id format."""
        request_data = create_test_explain_request("invalid-uuid", "response-123")
        
        response = client.get("/api/v1/explain/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_explain_endpoint_nonexistent_session(self, client: TestClient, session_id: str):
        """Test explain request for non-existent session."""
        request_data = create_test_explain_request(session_id, "response-123")
        
        response = client.get("/api/v1/explain/", params=request_data)
        assert_response_success(response)
        
        # Verify empty result structure
        response_data = response.json()
        assert "explanation" in response_data
        assert "memory_sources" in response_data
        assert "domain_sources" in response_data
        assert len(response_data["memory_sources"]) == 0
        assert len(response_data["domain_sources"]) == 0

    def test_explain_endpoint_nonexistent_response(self, client: TestClient, session_id: str):
        """Test explain request for non-existent response."""
        request_data = create_test_explain_request(session_id, "nonexistent-response")
        
        response = client.get("/api/v1/explain/", params=request_data)
        assert_response_success(response)
        
        # Verify empty result structure
        response_data = response.json()
        assert "explanation" in response_data
        assert "memory_sources" in response_data
        assert "domain_sources" in response_data

    def test_explain_endpoint_with_memory_sources(self, client: TestClient, session_id: str):
        """Test explain request that returns memory sources."""
        with patch('app.services.memory_service.MemoryService.get_user_memories') as mock_memory, \
             patch('app.services.retrieval_service.RetrievalService._retrieve_domain_facts') as mock_domain:
            
            mock_memory.return_value = [
                {
                    "memory_id": 1,
                    "text": "User asked about order status",
                    "kind": "episodic",
                    "importance": 0.8
                }
            ]
            mock_domain.return_value = []
            
            request_data = create_test_explain_request(session_id, "response-123")
            
            response = client.get("/api/v1/explain/", params=request_data)
            assert_response_success(response)
            
            # Verify response contains memory sources
            response_data = response.json()
            assert len(response_data["memory_sources"]) == 1
            assert response_data["memory_sources"][0]["text"] == "User asked about order status"

    def test_explain_endpoint_with_domain_sources(self, client: TestClient, session_id: str):
        """Test explain request that returns domain sources."""
        with patch('app.services.memory_service.MemoryService.get_user_memories') as mock_memory, \
             patch('app.services.retrieval_service.RetrievalService._retrieve_domain_facts') as mock_domain:
            
            mock_memory.return_value = []
            mock_domain.return_value = [
                {
                    "table": "customers",
                    "record_id": 123,
                    "fields": {"name": "John Doe", "email": "john@example.com"}
                }
            ]
            
            request_data = create_test_explain_request(session_id, "response-123")
            
            response = client.get("/api/v1/explain/", params=request_data)
            assert_response_success(response)
            
            # Verify response contains domain sources
            response_data = response.json()
            assert len(response_data["domain_sources"]) == 1
            assert response_data["domain_sources"][0]["table"] == "customers"

    def test_explain_endpoint_with_both_sources(self, client: TestClient, session_id: str):
        """Test explain request that returns both memory and domain sources."""
        with patch('app.services.memory_service.MemoryService.get_user_memories') as mock_memory, \
             patch('app.services.retrieval_service.RetrievalService._retrieve_domain_facts') as mock_domain:
            
            mock_memory.return_value = [
                {
                    "memory_id": 1,
                    "text": "User asked about order status",
                    "kind": "episodic",
                    "importance": 0.8
                }
            ]
            mock_domain.return_value = [
                {
                    "table": "customers",
                    "record_id": 123,
                    "fields": {"name": "John Doe", "email": "john@example.com"}
                }
            ]
            
            request_data = create_test_explain_request(session_id, "response-123")
            
            response = client.get("/api/v1/explain/", params=request_data)
            assert_response_success(response)
            
            # Verify response contains both types of sources
            response_data = response.json()
            assert len(response_data["memory_sources"]) == 1
            assert len(response_data["domain_sources"]) == 1

    def test_explain_endpoint_database_error(self, client: TestClient, session_id: str):
        """Test explain request when database fails."""
        with patch('app.services.memory_service.MemoryService.get_memories') as mock_memory:
            mock_memory.side_effect = Exception("Database error")
            
            request_data = create_test_explain_request(session_id, "response-123")
            
            response = client.get("/api/v1/explain/", params=request_data)
            assert response.status_code == 500

    def test_explain_endpoint_service_error(self, client: TestClient, session_id: str):
        """Test explain request when retrieval service fails."""
        with patch('app.services.retrieval_service.RetrievalService.get_domain_facts') as mock_domain:
            mock_domain.side_effect = Exception("Retrieval service error")
            
            request_data = create_test_explain_request(session_id, "response-123")
            
            response = client.get("/api/v1/explain/", params=request_data)
            assert response.status_code == 500

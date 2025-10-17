import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from tests.utils.test_utils import (
    create_test_memory_request,
    assert_response_success,
    assert_response_error
)


class TestMemoryAPI:
    """Test cases for the memory API endpoint."""

    def test_memory_endpoint_success(self, client: TestClient):
        """Test successful memory retrieval."""
        request_data = create_test_memory_request()
        
        response = client.get("/api/v1/memory/", params=request_data)
        assert_response_success(response)

    def test_memory_endpoint_missing_user_id(self, client: TestClient):
        """Test memory request without user_id."""
        response = client.get("/api/v1/memory/")
        assert response.status_code == 422  # Validation error

    def test_memory_endpoint_with_limit(self, client: TestClient):
        """Test memory request with limit parameter."""
        request_data = create_test_memory_request()
        request_data["limit"] = 10
        
        response = client.get("/api/v1/memory/", params=request_data)
        assert_response_success(response)

    def test_memory_endpoint_with_kind_filter(self, client: TestClient):
        """Test memory request with kind filter."""
        request_data = create_test_memory_request()
        request_data["kind"] = "episodic"
        
        response = client.get("/api/v1/memory/", params=request_data)
        assert_response_success(response)

    def test_memory_endpoint_with_importance_threshold(self, client: TestClient):
        """Test memory request with importance threshold."""
        request_data = create_test_memory_request()
        request_data["importance_threshold"] = 0.7
        
        response = client.get("/api/v1/memory/", params=request_data)
        assert_response_success(response)

    def test_memory_endpoint_invalid_kind(self, client: TestClient):
        """Test memory request with invalid kind."""
        request_data = create_test_memory_request()
        request_data["kind"] = "invalid_kind"
        
        response = client.get("/api/v1/memory/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_memory_endpoint_invalid_limit(self, client: TestClient):
        """Test memory request with invalid limit."""
        request_data = create_test_memory_request()
        request_data["limit"] = -1
        
        response = client.get("/api/v1/memory/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_memory_endpoint_invalid_importance_threshold(self, client: TestClient):
        """Test memory request with invalid importance threshold."""
        request_data = create_test_memory_request()
        request_data["importance_threshold"] = 1.5  # Should be between 0 and 1
        
        response = client.get("/api/v1/memory/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_memory_endpoint_empty_result(self, client: TestClient):
        """Test memory request for user with no memories."""
        request_data = create_test_memory_request("non-existent-user")
        
        response = client.get("/api/v1/memory/", params=request_data)
        assert_response_success(response)
        
        # Verify empty result structure
        response_data = response.json()
        assert "memories" in response_data
        assert "summary" in response_data
        assert len(response_data["memories"]) == 0

    def test_memory_endpoint_multiple_filters(self, client: TestClient):
        """Test memory request with multiple filters."""
        request_data = create_test_memory_request()
        request_data["kind"] = "semantic"
        request_data["limit"] = 5
        request_data["importance_threshold"] = 0.6
        
        response = client.get("/api/v1/memory/", params=request_data)
        assert_response_success(response)

    def test_memory_endpoint_database_error(self, client: TestClient):
        """Test memory request when database fails."""
        with patch('app.services.memory_service.MemoryService.get_user_memories') as mock_service:
            mock_service.side_effect = Exception("Database error")
            
            request_data = create_test_memory_request()
            
            response = client.get("/api/v1/memory/", params=request_data)
            assert response.status_code == 500

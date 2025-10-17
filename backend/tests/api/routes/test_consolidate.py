import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from tests.utils.test_utils import (
    create_test_consolidate_request,
    assert_response_success,
    assert_response_error
)


class TestConsolidateAPI:
    """Test cases for the consolidate API endpoint."""

    def test_consolidate_endpoint_success(self, client: TestClient):
        """Test successful memory consolidation."""
        request_data = create_test_consolidate_request()
        
        response = client.post("/api/v1/consolidate/", json=request_data)
        assert_response_success(response)

    def test_consolidate_endpoint_missing_user_id(self, client: TestClient):
        """Test consolidate request without user_id."""
        request_data = {}
        
        response = client.post("/api/v1/consolidate/", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_consolidate_endpoint_with_force(self, client: TestClient):
        """Test consolidate request with force parameter."""
        request_data = create_test_consolidate_request()
        request_data["force"] = True
        
        response = client.post("/api/v1/consolidate/", json=request_data)
        assert_response_success(response)

    def test_consolidate_endpoint_with_threshold(self, client: TestClient):
        """Test consolidate request with custom threshold."""
        request_data = create_test_consolidate_request()
        request_data["threshold"] = 0.8
        
        response = client.post("/api/v1/consolidate/", json=request_data)
        assert_response_success(response)

    def test_consolidate_endpoint_invalid_threshold(self, client: TestClient):
        """Test consolidate request with invalid threshold."""
        request_data = create_test_consolidate_request()
        request_data["threshold"] = 1.5  # Should be between 0 and 1
        
        response = client.post("/api/v1/consolidate/", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_consolidate_endpoint_no_memories_to_consolidate(self, client: TestClient):
        """Test consolidate request for user with no memories to consolidate."""
        request_data = create_test_consolidate_request("user-with-no-memories")
        
        response = client.post("/api/v1/consolidate/", json=request_data)
        assert_response_success(response)
        
        # Verify response indicates no consolidation occurred
        response_data = response.json()
        assert "consolidated_count" in response_data
        assert response_data["consolidated_count"] == 0

    def test_consolidate_endpoint_multiple_parameters(self, client: TestClient):
        """Test consolidate request with multiple parameters."""
        request_data = create_test_consolidate_request()
        request_data["force"] = True
        request_data["threshold"] = 0.7
        
        response = client.post("/api/v1/consolidate/", json=request_data)
        assert_response_success(response)

    def test_consolidate_endpoint_database_error(self, client: TestClient):
        """Test consolidate request when database fails."""
        with patch('app.services.memory_service.MemoryService.consolidate_memories') as mock_service:
            mock_service.side_effect = Exception("Database error")
            
            request_data = create_test_consolidate_request()
            
            response = client.post("/api/v1/consolidate/", json=request_data)
            assert response.status_code == 500

    def test_consolidate_endpoint_service_error(self, client: TestClient):
        """Test consolidate request when consolidation service fails."""
        with patch('app.services.memory_service.MemoryService.consolidate_memories') as mock_service:
            mock_service.side_effect = Exception("Consolidation service error")
            
            request_data = create_test_consolidate_request()
            
            response = client.post("/api/v1/consolidate/", json=request_data)
            assert response.status_code == 500

    def test_consolidate_endpoint_successful_consolidation(self, client: TestClient):
        """Test consolidate request that successfully consolidates memories."""
        with patch('app.services.memory_service.MemoryService.consolidate_memories') as mock_service:
            mock_service.return_value = 5  # Simulate consolidating 5 memories
            
            request_data = create_test_consolidate_request()
            
            response = client.post("/api/v1/consolidate/", json=request_data)
            assert_response_success(response)
            
            # Verify response contains consolidation results
            response_data = response.json()
            assert "consolidated_count" in response_data
            assert response_data["consolidated_count"] == 5

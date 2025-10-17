import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from tests.utils.test_utils import (
    create_test_entities_request,
    assert_response_success,
    assert_response_error
)


class TestEntitiesAPI:
    """Test cases for the entities API endpoint."""

    def test_entities_endpoint_success(self, client: TestClient, session_id: str):
        """Test successful entities retrieval."""
        request_data = create_test_entities_request(session_id)
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert_response_success(response)

    def test_entities_endpoint_missing_session_id(self, client: TestClient):
        """Test entities request without session_id."""
        request_data = {}
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_entities_endpoint_invalid_session_id(self, client: TestClient):
        """Test entities request with invalid session_id format."""
        request_data = create_test_entities_request("invalid-uuid")
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_entities_endpoint_with_type_filter(self, client: TestClient, session_id: str):
        """Test entities request with type filter."""
        request_data = create_test_entities_request(session_id)
        request_data["type"] = "customer"
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert_response_success(response)

    def test_entities_endpoint_with_source_filter(self, client: TestClient, session_id: str):
        """Test entities request with source filter."""
        request_data = create_test_entities_request(session_id)
        request_data["source"] = "message"
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert_response_success(response)

    def test_entities_endpoint_with_limit(self, client: TestClient, session_id: str):
        """Test entities request with limit parameter."""
        request_data = create_test_entities_request(session_id)
        request_data["limit"] = 10
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert_response_success(response)

    def test_entities_endpoint_invalid_type(self, client: TestClient, session_id: str):
        """Test entities request with invalid type."""
        request_data = create_test_entities_request(session_id)
        request_data["type"] = "invalid_type"
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_entities_endpoint_invalid_source(self, client: TestClient, session_id: str):
        """Test entities request with invalid source."""
        request_data = create_test_entities_request(session_id)
        request_data["source"] = "invalid_source"
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_entities_endpoint_invalid_limit(self, client: TestClient, session_id: str):
        """Test entities request with invalid limit."""
        request_data = create_test_entities_request(session_id)
        request_data["limit"] = -1
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert response.status_code == 422  # Validation error

    def test_entities_endpoint_empty_result(self, client: TestClient, session_id: str):
        """Test entities request for session with no entities."""
        request_data = create_test_entities_request(session_id)
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert_response_success(response)
        
        # Verify empty result structure
        response_data = response.json()
        assert "entities" in response_data
        assert len(response_data["entities"]) == 0

    def test_entities_endpoint_multiple_filters(self, client: TestClient, session_id: str):
        """Test entities request with multiple filters."""
        request_data = create_test_entities_request(session_id)
        request_data["type"] = "customer"
        request_data["source"] = "message"
        request_data["limit"] = 5
        
        response = client.get("/api/v1/entities/", params=request_data)
        assert_response_success(response)

    def test_entities_endpoint_database_error(self, client: TestClient, session_id: str):
        """Test entities request when database fails."""
        with patch('app.services.entity_service.EntityService.get_entities_for_session') as mock_service:
            mock_service.side_effect = Exception("Database error")
            
            request_data = create_test_entities_request(session_id)
            
            response = client.get("/api/v1/entities/", params=request_data)
            assert response.status_code == 500

    def test_entities_endpoint_with_external_ref(self, client: TestClient, session_id: str):
        """Test entities request that returns entities with external references."""
        with patch('app.services.entity_service.EntityService.get_entities_for_session') as mock_service:
            mock_service.return_value = [
                {
                    "entity_id": 1,
                    "session_id": session_id,
                    "name": "John Doe",
                    "type": "customer",
                    "source": "db",
                    "external_ref": {"customer_id": 123},
                    "created_at": "2024-01-01T00:00:00Z"
                }
            ]
            
            request_data = create_test_entities_request(session_id)
            
            response = client.get("/api/v1/entities/", params=request_data)
            assert_response_success(response)
            
            # Verify response contains entity with external reference
            response_data = response.json()
            assert len(response_data["entities"]) == 1
            assert response_data["entities"][0]["external_ref"] is not None

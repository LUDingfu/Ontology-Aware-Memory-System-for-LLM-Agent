import pytest
from fastapi.testclient import TestClient

from tests.utils.test_utils import assert_response_success


class TestHealthCheckAPI:
    """Test cases for the health check API endpoint."""

    def test_health_check_endpoint_success(self, client: TestClient):
        """Test successful health check request."""
        response = client.get("/api/v1/health-check/")
        assert_response_success(response)
        
        # Verify response structure
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert response_data["service"] == "ontology-aware-memory-system"

    def test_health_check_endpoint_methods(self, client: TestClient):
        """Test that health check only accepts GET requests."""
        # Test POST method (should fail)
        response = client.post("/api/v1/health-check/")
        assert response.status_code == 405  # Method not allowed
        
        # Test PUT method (should fail)
        response = client.put("/api/v1/health-check/")
        assert response.status_code == 405  # Method not allowed
        
        # Test DELETE method (should fail)
        response = client.delete("/api/v1/health-check/")
        assert response.status_code == 405  # Method not allowed

    def test_health_check_endpoint_no_auth_required(self, client: TestClient):
        """Test that health check doesn't require authentication."""
        # This should work without any authentication headers
        response = client.get("/api/v1/health-check/")
        assert_response_success(response)

    def test_health_check_endpoint_response_time(self, client: TestClient):
        """Test that health check responds quickly."""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/health-check/")
        end_time = time.time()
        
        assert_response_success(response)
        assert (end_time - start_time) < 1.0  # Should respond within 1 second

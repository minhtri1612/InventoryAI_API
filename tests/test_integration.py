"""
Integration tests for the API service
These tests verify the end-to-end behavior of the system
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from app import create_app


class TestAPIIntegration:
    """Integration tests for the complete API workflow."""

    @pytest.fixture
    def app(self):
        """Create application for testing."""
        return create_app({
            "TESTING": True,
            "VLLM_URL": "http://localhost:8000/v1/completions",
            "VLLM_MODEL": "microsoft/phi-2",
            "QDRANT_HOST": "localhost",
            "QDRANT_PORT": 6333
        })

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()

    def test_full_query_workflow(self, client):
        """Test complete workflow from query to structured response."""
        with patch("app.routes.get_llm_service") as mock_llm:
            # Mock LLM to return structured response
            mock_llm_instance = MagicMock()
            mock_llm_instance.generate_structured_response.return_value = {
                "api_endpoint": "/api/inventory",
                "api_payload": None,
                "payload_instruction": "Send GET request to list all items"
            }
            mock_llm.return_value = mock_llm_instance

            # Send query
            response = client.post(
                "/api/query",
                json={"question": "How can I list all inventory items?"},
                content_type="application/json"
            )
            data = json.loads(response.data)

            # Verify response structure
            assert response.status_code == 200
            assert data["success"] is True
            assert data["data"]["api_endpoint"] == "/api/inventory"
            assert data["data"]["payload_instruction"] is not None

    def test_query_with_create_operation(self, client):
        """Test query for creating a new inventory item."""
        with patch("app.routes.get_llm_service") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.generate_structured_response.return_value = {
                "api_endpoint": "/api/inventory",
                "api_payload": {
                    "name": "string",
                    "quantity": "integer",
                    "price": "float"
                },
                "payload_instruction": "POST request with name, quantity, and price in body"
            }
            mock_llm.return_value = mock_llm_instance

            response = client.post(
                "/api/query",
                json={"question": "How do I create a new item with name, quantity, and price?"},
                content_type="application/json"
            )
            data = json.loads(response.data)

            assert response.status_code == 200
            assert data["success"] is True
            assert data["data"]["api_payload"] is not None
            assert "name" in data["data"]["api_payload"]

    def test_query_with_delete_operation(self, client):
        """Test query for deleting an inventory item."""
        with patch("app.routes.get_llm_service") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.generate_structured_response.return_value = {
                "api_endpoint": "/api/inventory/<id>",
                "api_payload": None,
                "payload_instruction": "DELETE request with item ID in URL"
            }
            mock_llm.return_value = mock_llm_instance

            response = client.post(
                "/api/query",
                json={"question": "How do I delete an item?"},
                content_type="application/json"
            )
            data = json.loads(response.data)

            assert response.status_code == 200
            assert data["success"] is True
            assert "/api/inventory" in data["data"]["api_endpoint"]

    def test_query_with_search_operation(self, client):
        """Test query for searching inventory items."""
        with patch("app.routes.get_llm_service") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.generate_structured_response.return_value = {
                "api_endpoint": "/api/search",
                "api_payload": {"q": "search_term"},
                "payload_instruction": "GET request with 'q' query parameter"
            }
            mock_llm.return_value = mock_llm_instance

            response = client.post(
                "/api/query",
                json={"question": "How do I search for items?"},
                content_type="application/json"
            )
            data = json.loads(response.data)

            assert response.status_code == 200
            assert data["success"] is True
            assert data["data"]["api_endpoint"] == "/api/search"

    def test_service_health_integration(self, client):
        """Test health check integration with all services."""
        with patch("app.routes.get_llm_service") as mock_llm, \
             patch("app.routes.get_qdrant_service") as mock_qdrant:
            
            mock_llm_instance = MagicMock()
            mock_llm_instance.check_health.return_value = True
            mock_llm.return_value = mock_llm_instance

            mock_qdrant_instance = MagicMock()
            mock_qdrant_instance.check_health.return_value = True
            mock_qdrant.return_value = mock_qdrant_instance

            response = client.get("/api/health")
            data = json.loads(response.data)

            assert response.status_code == 200
            assert data["status"] == "healthy"
            assert all(
                status == "healthy" 
                for status in data["services"].values()
            )

    def test_endpoints_discovery_integration(self, client):
        """Test endpoint discovery functionality."""
        response = client.get("/api/endpoints")
        data = json.loads(response.data)

        assert response.status_code == 200
        assert "endpoints" in data
        
        # Verify we have expected endpoint types
        methods = {ep["method"] for ep in data["endpoints"]}
        assert "GET" in methods
        assert "POST" in methods
        assert "PUT" in methods
        assert "DELETE" in methods

    def test_error_handling_llm_failure(self, client):
        """Test error handling when LLM service fails."""
        with patch("app.routes.get_llm_service") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.generate_structured_response.side_effect = Exception("LLM Error")
            mock_llm.return_value = mock_llm_instance

            response = client.post(
                "/api/query",
                json={"question": "Some question"},
                content_type="application/json"
            )
            data = json.loads(response.data)

            assert response.status_code == 503
            assert data["success"] is False
            assert "unavailable" in data["error"].lower()

    def test_error_handling_invalid_json_from_llm(self, client):
        """Test error handling when LLM returns invalid JSON."""
        with patch("app.routes.get_llm_service") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.generate_structured_response.side_effect = ValueError("Invalid JSON")
            mock_llm.return_value = mock_llm_instance

            response = client.post(
                "/api/query",
                json={"question": "Some question"},
                content_type="application/json"
            )
            data = json.loads(response.data)

            assert response.status_code == 422
            assert data["success"] is False


class TestModelValidation:
    """Tests for Pydantic model validation in the API."""

    @pytest.fixture
    def app(self):
        """Create application for testing."""
        return create_app({"TESTING": True})

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()

    def test_query_request_validation(self, client):
        """Test that query request validates properly."""
        # Missing question field
        response = client.post(
            "/api/query",
            json={},
            content_type="application/json"
        )
        assert response.status_code == 400

    def test_response_structure_validation(self, client):
        """Test that response structure is always consistent."""
        with patch("app.routes.get_llm_service") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.generate_structured_response.return_value = {
                "api_endpoint": "/api/test",
                "api_payload": None,
                "payload_instruction": "Test instruction"
            }
            mock_llm.return_value = mock_llm_instance

            response = client.post(
                "/api/query",
                json={"question": "test"},
                content_type="application/json"
            )
            data = json.loads(response.data)

            # Verify all expected fields are present
            assert "success" in data
            assert "data" in data
            assert "error" in data

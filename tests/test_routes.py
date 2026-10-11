"""
Unit tests for the Flask API routes
"""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_all_healthy(self, client):
        """Test health check when all services are healthy."""
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
            assert data["services"]["llm"] == "healthy"
            assert data["services"]["qdrant"] == "healthy"

    def test_health_check_degraded(self, client):
        """Test health check when some services are unhealthy."""
        with patch("app.routes.get_llm_service") as mock_llm, \
             patch("app.routes.get_qdrant_service") as mock_qdrant:
            
            mock_llm_instance = MagicMock()
            mock_llm_instance.check_health.return_value = False
            mock_llm.return_value = mock_llm_instance

            mock_qdrant_instance = MagicMock()
            mock_qdrant_instance.check_health.return_value = True
            mock_qdrant.return_value = mock_qdrant_instance

            response = client.get("/api/health")
            data = json.loads(response.data)

            assert response.status_code == 503
            assert data["status"] == "degraded"
            assert data["services"]["llm"] == "unhealthy"


class TestQueryEndpoint:
    """Tests for the query endpoint."""

    def test_query_success(self, client, mock_llm_response):
        """Test successful query processing."""
        with patch("app.routes.get_llm_service") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.generate_structured_response.return_value = mock_llm_response
            mock_llm.return_value = mock_llm_instance

            response = client.post(
                "/api/query",
                json={"question": "How do I list all inventory items?"},
                content_type="application/json"
            )
            data = json.loads(response.data)

            assert response.status_code == 200
            assert data["success"] is True
            assert data["data"]["api_endpoint"] == "/api/inventory"
            assert data["error"] is None

    def test_query_with_payload(self, client, mock_llm_response_with_payload):
        """Test query that returns a payload."""
        with patch("app.routes.get_llm_service") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.generate_structured_response.return_value = mock_llm_response_with_payload
            mock_llm.return_value = mock_llm_instance

            response = client.post(
                "/api/query",
                json={"question": "How do I create a new inventory item?"},
                content_type="application/json"
            )
            data = json.loads(response.data)

            assert response.status_code == 200
            assert data["success"] is True
            assert data["data"]["api_payload"] is not None
            assert data["data"]["api_payload"]["name"] == "Test Item"

    def test_query_missing_body(self, client):
        """Test query with missing request body."""
        response = client.post(
            "/api/query",
            content_type="application/json"
        )
        data = json.loads(response.data)

        assert response.status_code == 400
        assert data["success"] is False
        assert "required" in data["error"].lower()

    def test_query_empty_question(self, client):
        """Test query with empty question."""
        response = client.post(
            "/api/query",
            json={"question": ""},
            content_type="application/json"
        )
        data = json.loads(response.data)

        assert response.status_code == 400
        assert data["success"] is False

    def test_query_llm_validation_error(self, client):
        """Test query when LLM returns invalid response."""
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
            assert "failed" in data["error"].lower()

    def test_query_llm_service_unavailable(self, client):
        """Test query when LLM service is unavailable."""
        with patch("app.routes.get_llm_service") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm_instance.generate_structured_response.side_effect = Exception("Connection refused")
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


class TestEndpointsEndpoint:
    """Tests for the endpoints listing endpoint."""

    def test_list_endpoints(self, client):
        """Test listing available endpoints."""
        response = client.get("/api/endpoints")
        data = json.loads(response.data)

        assert response.status_code == 200
        assert "endpoints" in data
        assert len(data["endpoints"]) > 0
        
        # Verify structure of endpoints
        for endpoint in data["endpoints"]:
            assert "method" in endpoint
            assert "path" in endpoint
            assert "description" in endpoint

    def test_endpoints_contains_inventory_routes(self, client):
        """Test that inventory routes are listed."""
        response = client.get("/api/endpoints")
        data = json.loads(response.data)

        paths = [ep["path"] for ep in data["endpoints"]]
        assert "/api/inventory" in paths
        assert "/api/inventory/<id>" in paths

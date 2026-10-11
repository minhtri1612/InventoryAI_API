"""
Unit tests for the Qdrant Service
"""

import pytest
from unittest.mock import patch, MagicMock

from app.qdrant_service import QdrantService


class TestQdrantService:
    """Tests for QdrantService class."""

    @pytest.fixture
    def qdrant_service(self):
        """Create Qdrant service instance for testing."""
        return QdrantService(host="localhost", port=6333)

    def test_init(self, qdrant_service):
        """Test Qdrant service initialization."""
        assert qdrant_service.host == "localhost"
        assert qdrant_service.port == 6333
        assert qdrant_service._client is None

    @patch("app.qdrant_service.QdrantClient")
    def test_client_lazy_initialization(self, mock_client_class, qdrant_service):
        """Test that client is lazily initialized."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # First access creates client
        _ = qdrant_service.client
        mock_client_class.assert_called_once_with(host="localhost", port=6333)

        # Second access reuses client
        _ = qdrant_service.client
        assert mock_client_class.call_count == 1

    @patch("app.qdrant_service.QdrantClient")
    def test_ensure_collection_creates_new(self, mock_client_class, qdrant_service):
        """Test collection creation when it doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client

        result = qdrant_service.ensure_collection()

        assert result is True
        mock_client.create_collection.assert_called_once()

    @patch("app.qdrant_service.QdrantClient")
    def test_ensure_collection_exists(self, mock_client_class, qdrant_service):
        """Test that existing collection is not recreated."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "api_endpoints"
        mock_client.get_collections.return_value.collections = [mock_collection]
        mock_client_class.return_value = mock_client

        result = qdrant_service.ensure_collection()

        assert result is True
        mock_client.create_collection.assert_not_called()

    @patch("app.qdrant_service.QdrantClient")
    def test_search_similar_success(self, mock_client_class, qdrant_service):
        """Test successful vector search."""
        mock_client = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = 1
        mock_hit.score = 0.95
        mock_hit.payload = {"endpoint": "/api/inventory"}
        mock_client.search.return_value = [mock_hit]
        mock_client_class.return_value = mock_client

        query_vector = [0.1] * 384
        results = qdrant_service.search_similar(query_vector, limit=5)

        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["endpoint"] == "/api/inventory"

    @patch("app.qdrant_service.QdrantClient")
    def test_search_similar_error(self, mock_client_class, qdrant_service):
        """Test vector search with error."""
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_client

        query_vector = [0.1] * 384
        results = qdrant_service.search_similar(query_vector)

        assert results == []

    @patch("app.qdrant_service.QdrantClient")
    def test_upsert_endpoints_success(self, mock_client_class, qdrant_service):
        """Test successful endpoint upserting."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        endpoints = [{"endpoint": "/api/inventory", "method": "GET"}]
        vectors = [[0.1] * 384]
        
        result = qdrant_service.upsert_endpoints(endpoints, vectors)

        assert result is True
        mock_client.upsert.assert_called_once()

    def test_upsert_endpoints_mismatched_lengths(self, qdrant_service):
        """Test upserting with mismatched endpoint and vector lengths."""
        endpoints = [{"endpoint": "/api/inventory"}]
        vectors = [[0.1] * 384, [0.2] * 384]  # Two vectors for one endpoint

        with pytest.raises(ValueError) as excinfo:
            qdrant_service.upsert_endpoints(endpoints, vectors)

        assert "same length" in str(excinfo.value)

    @patch("app.qdrant_service.QdrantClient")
    def test_check_health_success(self, mock_client_class, qdrant_service):
        """Test health check when Qdrant is healthy."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        assert qdrant_service.check_health() is True

    @patch("app.qdrant_service.QdrantClient")
    def test_check_health_failure(self, mock_client_class, qdrant_service):
        """Test health check when Qdrant is unhealthy."""
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = Exception("Connection refused")
        mock_client_class.return_value = mock_client

        assert qdrant_service.check_health() is False

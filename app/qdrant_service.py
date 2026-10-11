"""
Qdrant Service for vector search operations
"""

import logging
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class QdrantService:
    """Service class for interacting with Qdrant vector database."""

    COLLECTION_NAME = "api_endpoints"
    VECTOR_SIZE = 384  # Typical size for small embedding models

    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize Qdrant service.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
        """
        self.host = host
        self.port = port
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    def ensure_collection(self) -> bool:
        """Ensure the API endpoints collection exists.
        
        Returns:
            True if collection exists or was created, False on error
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.COLLECTION_NAME}")
            return True
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            return False

    def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the collection.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of matching documents with scores
        """
        try:
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit
            )
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def upsert_endpoints(
        self, 
        endpoints: List[Dict[str, Any]], 
        vectors: List[List[float]]
    ) -> bool:
        """Upsert endpoint documents with their vectors.
        
        Args:
            endpoints: List of endpoint metadata documents
            vectors: Corresponding embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        if len(endpoints) != len(vectors):
            raise ValueError("Endpoints and vectors must have same length")

        try:
            points = [
                models.PointStruct(
                    id=i,
                    vector=vectors[i],
                    payload=endpoints[i]
                )
                for i in range(len(endpoints))
            ]
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points
            )
            return True
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            return False

    def check_health(self) -> bool:
        """Check if Qdrant server is healthy.
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

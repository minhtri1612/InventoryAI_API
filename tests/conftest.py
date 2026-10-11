"""
Pytest configuration and fixtures
"""

import pytest
from app import create_app


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app({
        "TESTING": True,
        "VLLM_URL": "http://localhost:8000/v1/completions",
        "VLLM_MODEL": "microsoft/phi-2",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": 6333,
        "MAX_TOKENS": 512,
        "TEMPERATURE": 0.1
    })
    yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def mock_llm_response():
    """Sample LLM response for testing."""
    return {
        "api_endpoint": "/api/inventory",
        "api_payload": None,
        "payload_instruction": "Send a GET request to retrieve all inventory items"
    }


@pytest.fixture
def mock_llm_response_with_payload():
    """Sample LLM response with payload."""
    return {
        "api_endpoint": "/api/inventory",
        "api_payload": {"name": "Test Item", "quantity": 10, "price": 9.99},
        "payload_instruction": "Send a POST request with the item details in the body"
    }

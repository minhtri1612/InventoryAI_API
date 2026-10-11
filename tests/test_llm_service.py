"""
Unit tests for the LLM Service
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import requests

from app.llm_service import LLMService


class TestLLMService:
    """Tests for LLMService class."""

    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance for testing."""
        return LLMService(
            vllm_url="http://localhost:8000/v1/completions",
            model="microsoft/phi-2",
            max_tokens=512,
            temperature=0.1
        )

    def test_init(self, llm_service):
        """Test LLM service initialization."""
        assert llm_service.vllm_url == "http://localhost:8000/v1/completions"
        assert llm_service.model == "microsoft/phi-2"
        assert llm_service.max_tokens == 512
        assert llm_service.temperature == 0.1

    def test_parse_valid_json(self, llm_service):
        """Test parsing valid JSON response."""
        text = '{"api_endpoint": "/api/inventory", "api_payload": null, "payload_instruction": "GET request"}'
        result = llm_service._parse_llm_response(text)
        
        assert result["api_endpoint"] == "/api/inventory"
        assert result["api_payload"] is None
        assert result["payload_instruction"] == "GET request"

    def test_parse_json_with_code_blocks(self, llm_service):
        """Test parsing JSON wrapped in markdown code blocks."""
        text = '```json\n{"api_endpoint": "/api/inventory", "api_payload": null, "payload_instruction": "GET request"}\n```'
        result = llm_service._parse_llm_response(text)
        
        assert result["api_endpoint"] == "/api/inventory"

    def test_parse_json_with_surrounding_text(self, llm_service):
        """Test parsing JSON embedded in other text."""
        text = 'Here is the response: {"api_endpoint": "/api/inventory", "api_payload": null, "payload_instruction": "GET request"} end'
        result = llm_service._parse_llm_response(text)
        
        assert result["api_endpoint"] == "/api/inventory"

    def test_parse_missing_required_field(self, llm_service):
        """Test parsing JSON missing required fields."""
        text = '{"api_endpoint": "/api/inventory"}'
        
        with pytest.raises(ValueError) as excinfo:
            llm_service._parse_llm_response(text)
        
        assert "Missing required field" in str(excinfo.value)

    def test_parse_invalid_json(self, llm_service):
        """Test parsing invalid JSON."""
        text = 'This is not JSON at all'
        
        with pytest.raises(ValueError) as excinfo:
            llm_service._parse_llm_response(text)
        
        assert "Could not find valid JSON" in str(excinfo.value)

    @patch("requests.post")
    def test_generate_structured_response_success(self, mock_post, llm_service):
        """Test successful structured response generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "text": '{"api_endpoint": "/api/inventory", "api_payload": null, "payload_instruction": "GET request"}'
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = llm_service.generate_structured_response("List all inventory")
        
        assert result["api_endpoint"] == "/api/inventory"
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_generate_structured_response_request_error(self, mock_post, llm_service):
        """Test response generation when request fails."""
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")

        with pytest.raises(requests.exceptions.RequestException):
            llm_service.generate_structured_response("List all inventory")

    @patch("requests.post")
    def test_generate_structured_response_invalid_json(self, mock_post, llm_service):
        """Test response generation when LLM returns invalid JSON."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"text": "Not a valid JSON response"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(ValueError):
            llm_service.generate_structured_response("List all inventory")

    @patch("requests.get")
    def test_check_health_success(self, mock_get, llm_service):
        """Test health check when service is healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert llm_service.check_health() is True

    @patch("requests.get")
    def test_check_health_failure(self, mock_get, llm_service):
        """Test health check when service is unhealthy."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection refused")

        assert llm_service.check_health() is False


class TestLLMServiceOutputValidation:
    """Tests to verify expected output structure from LLM."""

    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance for testing."""
        return LLMService(
            vllm_url="http://localhost:8000/v1/completions",
            model="microsoft/phi-2"
        )

    def test_expected_output_structure_list_items(self, llm_service):
        """Test that list items query produces expected structure."""
        response_text = '''
        {
            "api_endpoint": "/api/inventory",
            "api_payload": null,
            "payload_instruction": "Send a GET request to retrieve all inventory items. No payload required."
        }
        '''
        result = llm_service._parse_llm_response(response_text)
        
        assert "api_endpoint" in result
        assert "api_payload" in result
        assert "payload_instruction" in result
        assert result["api_endpoint"] == "/api/inventory"

    def test_expected_output_structure_create_item(self, llm_service):
        """Test that create item query produces expected structure with payload."""
        response_text = '''
        {
            "api_endpoint": "/api/inventory",
            "api_payload": {"name": "string", "quantity": "integer", "price": "float"},
            "payload_instruction": "Send a POST request with item name, quantity, and price in the request body."
        }
        '''
        result = llm_service._parse_llm_response(response_text)
        
        assert result["api_endpoint"] == "/api/inventory"
        assert result["api_payload"] is not None
        assert "name" in result["api_payload"]
        assert "quantity" in result["api_payload"]
        assert "price" in result["api_payload"]

    def test_expected_output_structure_delete_item(self, llm_service):
        """Test that delete item query produces expected structure."""
        response_text = '''
        {
            "api_endpoint": "/api/inventory/<id>",
            "api_payload": null,
            "payload_instruction": "Send a DELETE request with the item ID in the URL path."
        }
        '''
        result = llm_service._parse_llm_response(response_text)
        
        assert "/api/inventory" in result["api_endpoint"]
        assert result["api_payload"] is None

    def test_expected_output_structure_search(self, llm_service):
        """Test that search query produces expected structure."""
        response_text = '''
        {
            "api_endpoint": "/api/search",
            "api_payload": {"q": "search_term"},
            "payload_instruction": "Send a GET request with the search term as query parameter 'q'."
        }
        '''
        result = llm_service._parse_llm_response(response_text)
        
        assert result["api_endpoint"] == "/api/search"
        assert "q" in result["api_payload"]

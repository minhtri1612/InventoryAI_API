"""
LLM Service for interacting with vLLM (Phi-2)
"""

import json
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMService:
    """Service class for interacting with vLLM server."""

    # Default system prompt for structured JSON output
    DEFAULT_SYSTEM_PROMPT = """You are an API assistant. Given a natural language question about an inventory system, 
respond with ONLY valid JSON in this exact format:
{
    "api_endpoint": "<endpoint path>",
    "api_payload": <payload object or null>,
    "payload_instruction": "<how to build the request>"
}

Available endpoints:
- GET /api/inventory - List all inventory items
- GET /api/inventory/<id> - Get specific item
- POST /api/inventory - Create new item (requires: name, quantity, price)
- PUT /api/inventory/<id> - Update item
- DELETE /api/inventory/<id> - Delete item
- GET /api/search - Search items (query param: q)

Respond with ONLY the JSON, no explanation."""

    def __init__(
        self, 
        vllm_url: str, 
        model: str, 
        max_tokens: int = 512, 
        temperature: float = 0.1,
        system_prompt: str = None
    ):
        """Initialize LLM service.
        
        Args:
            vllm_url: URL of the vLLM server
            model: Model identifier for vLLM
            max_tokens: Maximum tokens for completion
            temperature: Temperature for generation
            system_prompt: Custom system prompt (optional, uses DEFAULT_SYSTEM_PROMPT if not provided)
        """
        self.vllm_url = vllm_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def generate_structured_response(self, question: str) -> Dict[str, Any]:
        """Generate structured JSON response from natural language question.
        
        Args:
            question: Natural language question from user
            
        Returns:
            Dictionary containing structured API metadata
            
        Raises:
            ValueError: If LLM response cannot be parsed as valid JSON
            requests.RequestException: If vLLM server is unavailable
        """
        prompt = f"{self.system_prompt}\n\nQuestion: {question}\n\nJSON Response:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": ["\n\n", "Question:"]
        }

        try:
            response = requests.post(
                self.vllm_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("choices", [{}])[0].get("text", "").strip()

            # Parse and validate JSON
            return self._parse_llm_response(generated_text)

        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parse and validate LLM response.
        
        Args:
            text: Raw text response from LLM
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If response is not valid JSON or missing required fields
        """
        # Try to extract JSON from response
        text = text.strip()
        
        # Handle case where JSON might be wrapped in markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(text[start:end])
            else:
                raise ValueError(f"Could not find valid JSON in response: {text}")

        # Validate required fields
        required_fields = ["api_endpoint", "payload_instruction"]
        for field in required_fields:
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")

        return parsed

    def check_health(self) -> bool:
        """Check if vLLM server is healthy.
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(
                self.vllm_url.replace("/v1/completions", "/health"),
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

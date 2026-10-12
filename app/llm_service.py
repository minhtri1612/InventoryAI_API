"""LLM service for interacting with Ollama."""

import json
import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class LLMService:
    """Service class for interacting with an Ollama server."""

    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "NONE"]},
            "api_endpoint": {"type": "string"},
            "api_payload": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            "payload_instruction": {"type": "string"},
            "clarification": {"type": "string"},
        },
        "required": [
            "supported",
            "method",
            "api_endpoint",
            "api_payload",
            "payload_instruction",
            "clarification",
        ],
    }

    DEFAULT_SYSTEM_PROMPT = """You translate an inventory question into one JSON action. You do not answer in prose.
Supported actions only:
- GET /api/search when the user describes a use, a function, or what an item is for, even with no product name and no id. api_payload is {"q": "the descriptive phrase"}. This is supported. Never ask for a name or id for this case.
- GET /api/inventory lists items or applies exact filters. Inclusive bounds are min_quantity, max_quantity, min_price, max_price. Strict bounds are quantity_lt, quantity_gt, price_lt, price_gt. "fewer than 5" is {"quantity_lt": 5}, not max_quantity 5 and not min_quantity 1. "at least 5" is {"quantity_gte": 5}. "more than 5" is {"quantity_gt": 5}. Zero stock is included by a strict less-than filter.
- GET /api/inventory/<id> reads one item only when the question includes a numeric id.
- POST /api/inventory proposes a create with name, quantity, and price from the question.
- PUT /api/inventory/<id> proposes an update only when a numeric id is present.
- DELETE /api/inventory/<id> proposes a delete only when a numeric id is present.
Set supported to false only when the question is not about inventory, or a create/update/delete/get-by-id is missing a required value. Do not invent an id.
A description such as "something to drive nails" is GET /api/search, supported true, clarification "".
payload_instruction is always a short non-empty sentence. clarification is "" when the action is complete.
Example: "Do we have a tool for driving nails?" -> {"supported": true, "method": "GET", "api_endpoint": "/api/search", "api_payload": {"q": "tool for driving nails"}, "payload_instruction": "Search by what the item is used for.", "clarification": ""}"""

    def __init__(
        self,
        ollama_base_url: str,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: str = None,
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def generate_structured_response(self, question: str) -> Dict[str, Any]:
        """Generate and parse structured API metadata for a natural language query."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "format": self.RESPONSE_SCHEMA,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("message", {}).get("content", "").strip()
            if not generated_text:
                raise ValueError("The model returned an empty inventory action")
            return self._parse_llm_response(generated_text)
        except requests.exceptions.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM response as JSON: %s", exc)
            raise ValueError(f"Invalid JSON response from LLM: {exc}") from exc

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parse an LLM response and ensure the required fields are present."""
        text = text.strip()
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
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError(f"Could not find valid JSON in response: {text}")
            parsed = json.loads(text[start:end])

        for field in ("api_endpoint", "payload_instruction"):
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")
        if not str(parsed.get("payload_instruction") or "").strip():
            raise ValueError(
                "Missing required field: payload_instruction. The model did not return a usable inventory action."
            )
        return parsed

    def check_health(self) -> bool:
        """Check whether the Ollama server responds to its tags endpoint."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

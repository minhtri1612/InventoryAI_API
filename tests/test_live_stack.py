"""Optional checks against a running Compose stack. Skip unless RUN_LIVE_STACK=1."""

import os

import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_STACK") != "1",
    reason="Set RUN_LIVE_STACK=1 to call the running API, Ollama, and Qdrant",
)

BASE_URL = os.getenv("LIVE_API_URL", "http://localhost:5000")


def test_live_health_and_semantic_search():
    health = requests.get(f"{BASE_URL}/api/health", timeout=10)
    assert health.status_code == 200
    body = health.json()
    assert body["services"]["llm"] == "healthy"
    assert body["services"]["qdrant"] == "healthy"

    found = requests.get(
        f"{BASE_URL}/api/search",
        params={"q": "something to drive nails"},
        timeout=60,
    )
    assert found.status_code == 200
    payload = found.json()
    assert payload["search_mode"] == "semantic"
    assert payload["data"]
    assert all(item["score"] >= 0.5 for item in payload["data"])

"""Tests for inventory CRUD and search."""

import json
from unittest.mock import patch


def test_inventory_crud_and_search(client):
    created = client.post(
        "/api/inventory",
        json={"name": "Widget", "quantity": 10, "price": 9.99},
    )
    assert created.status_code == 201
    item = json.loads(created.data)["data"]
    assert item["name"] == "Widget"
    item_id = item["id"]

    listed = json.loads(client.get("/api/inventory").data)["data"]
    assert len(listed) == 1

    fetched = client.get(f"/api/inventory/{item_id}")
    assert fetched.status_code == 200
    assert json.loads(fetched.data)["data"]["quantity"] == 10

    updated = client.put(
        f"/api/inventory/{item_id}",
        json={"quantity": 4},
    )
    assert updated.status_code == 200
    assert json.loads(updated.data)["data"]["quantity"] == 4
    assert json.loads(updated.data)["data"]["name"] == "Widget"

    found = client.get("/api/search?q=Wid")
    assert found.status_code == 200
    assert json.loads(found.data)["data"][0]["id"] == item_id

    missing = client.get("/api/search?q=nope")
    assert json.loads(missing.data)["data"] == []

    deleted = client.delete(f"/api/inventory/{item_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/inventory/{item_id}").status_code == 404


def test_query_confirms_writes_and_filters_reads(client):
    with patch("app.routes.get_llm_service") as mock_llm:
        mock_llm.return_value.generate_structured_response.return_value = {
            "method": "POST",
            "api_endpoint": "/api/inventory",
            "api_payload": {"name": "Hammer", "quantity": 2, "price": 15},
            "payload_instruction": "Create the item from the question",
        }
        preview = client.post("/api/query", json={"question": "Add a Hammer"})
        body = json.loads(preview.data)
        assert preview.status_code == 200
        assert body["confirmation_required"] is True
        assert body["confirmation_token"]
        assert json.loads(client.get("/api/inventory").data)["data"] == []

        confirmed = client.post(
            "/api/query/confirm",
            json={"confirmation_token": body["confirmation_token"]},
        )
        assert confirmed.status_code == 200
        assert json.loads(confirmed.data)["result"]["name"] == "Hammer"
        replay = client.post(
            "/api/query/confirm",
            json={"confirmation_token": body["confirmation_token"]},
        )
        assert replay.status_code == 409
        assert len(json.loads(client.get("/api/inventory").data)["data"]) == 1

        mock_llm.return_value.generate_structured_response.return_value = {
            "method": "GET",
            "api_endpoint": "/api/inventory",
            "api_payload": {"min_quantity": 5},
            "payload_instruction": "List items",
        }
        listed = json.loads(
            client.post("/api/query", json={"question": "Items with quantity at least 5"}).data
        )
        assert listed["result"] == []

        mock_llm.return_value.generate_structured_response.return_value = {
            "method": "DELETE",
            "api_endpoint": "/api/inventory/<id>",
            "api_payload": None,
            "payload_instruction": "Delete an item",
        }
        unclear = json.loads(
            client.post("/api/query", json={"question": "Delete it"}).data
        )
        assert unclear["confirmation_required"] is False
        assert unclear["result"] is None
        assert json.loads(client.get("/api/inventory").data)["data"][0]["name"] == "Hammer"


def test_semantic_search_uses_qdrant_ids(client):
    created = client.post(
        "/api/inventory",
        json={
            "name": "Hammer",
            "quantity": 2,
            "price": 15,
            "description": "steel head tool for driving nails",
            "category": "tools",
        },
    )
    item_id = json.loads(created.data)["data"]["id"]
    with patch("app.semantic_search.embed_text", return_value=[0.1] * 384), \
         patch("app.semantic_search._qdrant") as mock_qdrant:
        mock_qdrant.return_value.ensure_collection.return_value = True
        mock_qdrant.return_value.search_similar.return_value = [
            {"id": item_id, "score": 0.91, "payload": {"name": "Hammer"}}
        ]
        response = client.get("/api/search?q=something%20to%20drive%20nails")
        body = json.loads(response.data)
        assert response.status_code == 200
        assert body["search_mode"] == "semantic"
        assert body["data"][0]["id"] == item_id
        assert body["data"][0]["quantity"] == 2


def test_fewer_than_uses_strict_quantity_bound(client):
    for name, quantity in (("Empty", 0), ("Hammer", 2), ("Full", 5)):
        client.post("/api/inventory", json={"name": name, "quantity": quantity, "price": 1})
    with patch("app.routes.get_llm_service") as mock_llm:
        mock_llm.return_value.generate_structured_response.return_value = {
            "supported": True,
            "method": "GET",
            "api_endpoint": "/api/inventory",
            "api_payload": {"min_quantity": 1, "max_quantity": 5},
            "payload_instruction": "Filter quantity.",
            "clarification": "",
        }
        response = client.post(
            "/api/query",
            json={"question": "Which inventory items have fewer than 5 units in stock?"},
        )
        body = json.loads(response.data)
        assert body["data"]["api_payload"]["quantity_lt"] == 5
        assert "max_quantity" not in body["data"]["api_payload"]
        assert "min_quantity" not in body["data"]["api_payload"]
        names = {item["name"] for item in body["result"]}
        assert names == {"Empty", "Hammer"}


def test_low_semantic_score_is_omitted(client, app):
    created = client.post(
        "/api/inventory",
        json={"name": "Hammer", "quantity": 1, "price": 5, "description": "drives nails"},
    )
    item_id = json.loads(created.data)["data"]["id"]
    app.config["SEMANTIC_SCORE_THRESHOLD"] = 0.5
    with patch("app.semantic_search.embed_text", return_value=[0.1] * 384), \
         patch("app.semantic_search._qdrant") as mock_qdrant:
        mock_qdrant.return_value.ensure_collection.return_value = True
        mock_qdrant.return_value.search_similar.return_value = [
            {"id": item_id, "score": 0.2, "payload": {"name": "Hammer"}}
        ]
        body = json.loads(client.get("/api/search?q=unrelated").data)
        assert body["search_mode"] == "semantic"
        assert body["data"] == []


def test_failed_index_can_be_retried(client):
    with patch("app.semantic_search.embed_text", side_effect=ConnectionError("down")):
        created = client.post(
            "/api/inventory",
            json={"name": "Saw", "quantity": 1, "price": 4, "description": "cuts wood"},
        )
        item = json.loads(created.data)["data"]
        assert item["embedding_synced"] is False
        health = json.loads(client.get("/api/health").data)
        assert health["embedding_stale"] >= 1

    with patch("app.semantic_search.embed_text", return_value=[0.2] * 384), \
         patch("app.semantic_search._qdrant") as mock_qdrant:
        mock_qdrant.return_value.ensure_collection.return_value = True
        mock_qdrant.return_value.upsert_item.return_value = True
        retried = json.loads(client.post("/api/admin/reindex").data)
        assert retried["data"]["indexed"] >= 1
        assert retried["data"]["failed"] == 0


def test_api_key_and_backup(client, app):
    app.config["API_KEY"] = "secret"
    denied = client.get("/api/inventory")
    assert denied.status_code == 401
    allowed = client.get("/api/inventory", headers={"X-API-Key": "secret"})
    assert allowed.status_code == 200
    copied = client.post("/api/admin/backup", headers={"X-API-Key": "secret"})
    assert copied.status_code == 201
    assert json.loads(copied.data)["data"]["path"]


def test_descriptive_question_is_sent_to_search(client):
    created = client.post(
        "/api/inventory",
        json={"name": "Hammer", "quantity": 2, "price": 15, "description": "drives nails"},
    )
    item_id = json.loads(created.data)["data"]["id"]
    with patch("app.routes.get_llm_service") as mock_llm, \
         patch("app.semantic_search.embed_text", return_value=[0.1] * 384), \
         patch("app.semantic_search._qdrant") as mock_qdrant:
        mock_llm.return_value.generate_structured_response.return_value = {
            "supported": False,
            "method": "NONE",
            "api_endpoint": "",
            "api_payload": None,
            "payload_instruction": "Need a name or id.",
            "clarification": "Missing product name or id.",
        }
        mock_qdrant.return_value.ensure_collection.return_value = True
        mock_qdrant.return_value.search_similar.return_value = [
            {"id": item_id, "score": 0.8, "payload": {"name": "Hammer"}}
        ]
        response = client.post(
            "/api/query",
            json={"question": "Do we have a tool for driving nails?"},
        )
        body = json.loads(response.data)
        assert response.status_code == 200
        assert body["data"]["api_endpoint"] == "/api/search"
        assert body["data"]["supported"] is True
        assert body["result"][0]["id"] == item_id


def test_out_of_scope_question_does_not_write(client):
    with patch("app.routes.get_llm_service") as mock_llm:
        mock_llm.return_value.generate_structured_response.return_value = {
            "supported": False,
            "method": "NONE",
            "api_endpoint": "",
            "api_payload": None,
            "payload_instruction": "Outside inventory.",
            "clarification": "Ask about inventory items only.",
        }
        response = client.post("/api/query", json={"question": "What is the capital of France?"})
        body = json.loads(response.data)
        assert response.status_code == 200
        assert body["confirmation_required"] is False
        assert body["result"] is None
        assert "inventory" in body["message"]
        assert json.loads(client.get("/api/inventory").data)["data"] == []


def test_inventory_validation(client):
    empty = client.post("/api/inventory", json={})
    assert empty.status_code == 400

    no_body = client.post("/api/inventory")
    assert no_body.status_code == 400

    negative = client.post(
        "/api/inventory",
        json={"name": "Bad", "quantity": -1, "price": 1},
    )
    assert negative.status_code == 400

    assert client.get("/api/inventory/999").status_code == 404
    assert client.delete("/api/inventory/999").status_code == 404
    assert client.put("/api/inventory/999", json={"name": "X"}).status_code == 404

    blank_update = client.put("/api/inventory/1", json={})
    assert blank_update.status_code == 400

    assert client.get("/api/search").status_code == 400

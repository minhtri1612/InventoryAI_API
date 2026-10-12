"""Semantic inventory search. SQLite stays the source of item facts."""

import logging
from typing import Any, Dict, List, Tuple

import requests

from app.inventory_db import get_item, list_unsynced_items, search_items, set_embedding_synced
from app.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class SemanticUnavailable(Exception):
    """Embedding or Qdrant cannot serve a semantic search."""


def item_text(item: Dict[str, Any]) -> str:
    parts = [item.get("name") or "", item.get("category") or "", item.get("description") or ""]
    return ". ".join(part.strip() for part in parts if part and part.strip())


def embed_text(base_url: str, model: str, text: str) -> List[float]:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    vector = response.json().get("embedding")
    if not isinstance(vector, list) or not vector:
        raise SemanticUnavailable("Embedding model returned no vector")
    return vector


def index_item(app_config: Dict[str, Any], item: Dict[str, Any]) -> bool:
    """Embed one item and upsert it. Failure leaves SQLite unchanged."""
    try:
        vector = embed_text(
            app_config["OLLAMA_BASE_URL"],
            app_config["EMBEDDING_MODEL"],
            item_text(item),
        )
        service = _qdrant(app_config)
        synced = service.upsert_item(
            item["id"],
            vector,
            {"name": item["name"]},
        )
    except Exception as exc:
        logger.warning("Could not index item %s: %s", item.get("id"), exc)
        synced = False
    set_embedding_synced(app_config["DATABASE_PATH"], item["id"], synced)
    return synced


def remove_item(app_config: Dict[str, Any], item_id: int) -> None:
    try:
        _qdrant(app_config).delete_item(item_id)
    except Exception as exc:
        logger.warning("Could not delete vector %s: %s", item_id, exc)


def reindex_pending(app_config: Dict[str, Any]) -> Dict[str, int]:
    """Index items whose SQLite row is newer than their vector."""
    indexed = 0
    failed = 0
    for item in list_unsynced_items(app_config["DATABASE_PATH"]):
        if index_item(app_config, item):
            indexed += 1
        else:
            failed += 1
    return {"indexed": indexed, "failed": failed}


def find_items(app_config: Dict[str, Any], query: str) -> Tuple[List[Dict[str, Any]], str]:
    """Return items and the mode used: semantic, or name when vectors are unavailable."""
    database_path = app_config["DATABASE_PATH"]
    try:
        vector = embed_text(app_config["OLLAMA_BASE_URL"], app_config["EMBEDDING_MODEL"], query)
        service = _qdrant(app_config)
        if not service.ensure_collection():
            raise SemanticUnavailable("Qdrant collection is unavailable")
        hits = service.search_similar(vector, limit=5)
        threshold = float(app_config.get("SEMANTIC_SCORE_THRESHOLD", 0.5))
        items = []
        for hit in hits:
            if hit["score"] < threshold:
                continue
            item = get_item(database_path, int(hit["id"]))
            if item is not None:
                items.append({**item, "score": hit["score"]})
        return items, "semantic"
    except Exception as exc:
        logger.warning("Semantic search unavailable, using name search: %s", exc)
        return search_items(database_path, query), "name"


def _qdrant(app_config: Dict[str, Any]) -> QdrantService:
    return QdrantService(
        host=app_config["QDRANT_HOST"],
        port=int(app_config["QDRANT_PORT"]),
        vector_size=int(app_config["VECTOR_SIZE"]),
    )

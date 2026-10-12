"""SQLite storage for inventory items and one-time write confirmations."""

import json
import secrets
import sqlite3
import time
from typing import Any, Dict, List, Optional


def connect(database_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(database_path: str) -> None:
    with connect(database_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL
            )
            """
        )
        for statement in (
            "ALTER TABLE inventory_items ADD COLUMN description TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE inventory_items ADD COLUMN category TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE inventory_items ADD COLUMN embedding_synced INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_actions (
                token TEXT PRIMARY KEY,
                action_json TEXT NOT NULL,
                expires_at REAL NOT NULL,
                used INTEGER NOT NULL DEFAULT 0
            )
            """
        )


def row_to_item(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "quantity": row["quantity"],
        "price": row["price"],
        "description": row["description"] if "description" in row.keys() else "",
        "category": row["category"] if "category" in row.keys() else "",
        "embedding_synced": bool(row["embedding_synced"]) if "embedding_synced" in row.keys() else False,
    }


def list_items(
    database_path: str,
    name: Optional[str] = None,
    min_quantity: Optional[int] = None,
    max_quantity: Optional[int] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    quantity_lt: Optional[int] = None,
    quantity_lte: Optional[int] = None,
    quantity_gt: Optional[int] = None,
    quantity_gte: Optional[int] = None,
    price_lt: Optional[float] = None,
    price_lte: Optional[float] = None,
    price_gt: Optional[float] = None,
    price_gte: Optional[float] = None,
) -> List[Dict[str, Any]]:
    clauses = []
    params: List[Any] = []
    if name:
        clauses.append("name LIKE ?")
        params.append(f"%{name}%")
    comparisons = (
        ("quantity >= ?", min_quantity),
        ("quantity <= ?", max_quantity),
        ("quantity < ?", quantity_lt),
        ("quantity <= ?", quantity_lte),
        ("quantity > ?", quantity_gt),
        ("quantity >= ?", quantity_gte),
        ("price >= ?", min_price),
        ("price <= ?", max_price),
        ("price < ?", price_lt),
        ("price <= ?", price_lte),
        ("price > ?", price_gt),
        ("price >= ?", price_gte),
    )
    for clause, value in comparisons:
        if value is not None:
            clauses.append(clause)
            params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect(database_path) as conn:
        rows = conn.execute(
            f"SELECT id, name, quantity, price, description, category, embedding_synced FROM inventory_items{where} ORDER BY id",
            params,
        ).fetchall()
    return [row_to_item(row) for row in rows]


def get_item(database_path: str, item_id: int) -> Optional[Dict[str, Any]]:
    with connect(database_path) as conn:
        row = conn.execute(
            "SELECT id, name, quantity, price, description, category, embedding_synced FROM inventory_items WHERE id = ?",
            (item_id,),
        ).fetchone()
    return row_to_item(row) if row else None


def create_item(
    database_path: str,
    name: str,
    quantity: int,
    price: float,
    description: str = "",
    category: str = "",
) -> Dict[str, Any]:
    with connect(database_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO inventory_items (name, quantity, price, description, category)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, quantity, price, description, category),
        )
        item_id = cursor.lastrowid
    item = get_item(database_path, item_id)
    if item is None:
        raise RuntimeError("Failed to read created inventory item")
    return item


def update_item(
    database_path: str,
    item_id: int,
    name: Optional[str] = None,
    quantity: Optional[int] = None,
    price: Optional[float] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    existing = get_item(database_path, item_id)
    if existing is None:
        return None
    updated = {
        "name": existing["name"] if name is None else name,
        "quantity": existing["quantity"] if quantity is None else quantity,
        "price": existing["price"] if price is None else price,
        "description": existing["description"] if description is None else description,
        "category": existing["category"] if category is None else category,
    }
    with connect(database_path) as conn:
        conn.execute(
            """
            UPDATE inventory_items
            SET name = ?, quantity = ?, price = ?, description = ?, category = ?
            WHERE id = ?
            """,
            (
                updated["name"],
                updated["quantity"],
                updated["price"],
                updated["description"],
                updated["category"],
                item_id,
            ),
        )
    return get_item(database_path, item_id)


def delete_item(database_path: str, item_id: int) -> bool:
    with connect(database_path) as conn:
        cursor = conn.execute(
            "DELETE FROM inventory_items WHERE id = ?",
            (item_id,),
        )
        return cursor.rowcount > 0


def backup_database(database_path: str) -> str:
    """Copy the SQLite database to a sibling backups directory."""
    from datetime import datetime, timezone
    from pathlib import Path

    source = Path(database_path)
    directory = source.parent / "backups"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = directory / f"{source.stem}-{stamp}{source.suffix or '.db'}"
    with sqlite3.connect(database_path) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
    return str(destination)


def set_embedding_synced(database_path: str, item_id: int, synced: bool) -> None:
    with connect(database_path) as conn:
        conn.execute(
            "UPDATE inventory_items SET embedding_synced = ? WHERE id = ?",
            (1 if synced else 0, item_id),
        )


def list_unsynced_items(database_path: str) -> List[Dict[str, Any]]:
    with connect(database_path) as conn:
        rows = conn.execute(
            """
            SELECT id, name, quantity, price, description, category, embedding_synced
            FROM inventory_items
            WHERE embedding_synced = 0
            ORDER BY id
            """
        ).fetchall()
    return [row_to_item(row) for row in rows]


def search_items(database_path: str, query: str) -> List[Dict[str, Any]]:
    return list_items(database_path, name=query)


def save_pending_action(
    database_path: str, action: Dict[str, Any], ttl_seconds: int = 600
) -> str:
    token = secrets.token_urlsafe(16)
    with connect(database_path) as conn:
        conn.execute(
            "INSERT INTO pending_actions (token, action_json, expires_at, used) VALUES (?, ?, ?, 0)",
            (token, json.dumps(action), time.time() + ttl_seconds),
        )
    return token


def consume_pending_action(database_path: str, token: str) -> Dict[str, Any]:
    """Return the stored action once. Reject missing, used, or expired tokens."""
    with connect(database_path) as conn:
        row = conn.execute(
            "SELECT action_json, expires_at, used FROM pending_actions WHERE token = ?",
            (token,),
        ).fetchone()
        if row is None:
            raise LookupError("No pending confirmation matches this token")
        if row["used"]:
            raise PermissionError("This confirmation was already used")
        if row["expires_at"] < time.time():
            raise TimeoutError("This confirmation has expired")
        updated = conn.execute(
            "UPDATE pending_actions SET used = 1 WHERE token = ? AND used = 0",
            (token,),
        )
        if updated.rowcount != 1:
            raise PermissionError("This confirmation was already used")
        return json.loads(row["action_json"])

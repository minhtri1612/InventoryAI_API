"""Plan and execute inventory actions chosen by the LLM."""

import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from app.inventory_db import (
    create_item,
    delete_item,
    get_item,
    list_items,
    search_items,
    update_item,
)
from app.models import InventoryItemCreate, InventoryItemUpdate

_METHODS = {"GET", "POST", "PUT", "DELETE"}
_TYPE_PLACEHOLDERS = {"string", "integer", "int", "float", "number"}
_FILTER_INTS = (
    "min_quantity",
    "max_quantity",
    "quantity_lt",
    "quantity_lte",
    "quantity_gt",
    "quantity_gte",
)
_FILTER_FLOATS = (
    "min_price",
    "max_price",
    "price_lt",
    "price_lte",
    "price_gt",
    "price_gte",
)


class InventoryActionError(ValueError):
    """The chosen action could not be executed."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


class ActionOutcome:
    """Result of planning or running one inventory action."""

    def __init__(
        self,
        result: Any = None,
        confirmation_required: bool = False,
        message: Optional[str] = None,
    ):
        self.result = result
        self.confirmation_required = confirmation_required
        self.message = message


def plan_inventory_action(database_path: str, action: Dict[str, Any]) -> ActionOutcome:
    """Run reads immediately. Mutations return a preview and wait for confirmation."""
    return _dispatch(database_path, action, confirmed=False)


def execute_confirmed_action(database_path: str, action: Dict[str, Any]) -> Any:
    """Run a create, update, or delete that the caller has confirmed."""
    method, path, payload = _normalize(action)
    if method not in {"POST", "PUT", "DELETE"}:
        raise InventoryActionError("Only create, update, and delete can be confirmed", 400)
    outcome = _dispatch(database_path, action, confirmed=True)
    if outcome.confirmation_required or outcome.result is None:
        raise InventoryActionError(
            outcome.message or "This action is not specific enough to run",
            422,
        )
    if method == "GET":
        raise InventoryActionError("Only create, update, and delete can be confirmed", 400)
    return outcome.result


def _dispatch(database_path: str, action: Dict[str, Any], confirmed: bool) -> ActionOutcome:
    method, path, payload = _normalize(action)
    if not method or "<" in path or ">" in path or _is_type_template(payload):
        return ActionOutcome(
            message="Need a supported method, a numeric item id, and real field values before this can run."
        )
    if not _is_allowed(method, path):
        return ActionOutcome(message="That endpoint is not an allowed inventory action.")

    item_match = re.fullmatch(r"/api/inventory/(\d+)", path)
    if path == "/api/inventory" and method == "GET":
        return ActionOutcome(result=list_items(database_path, **_filters(payload)))
    if path == "/api/search" and method == "GET":
        query = str(payload.get("q") or payload.get("name") or "").strip()
        if not query:
            return ActionOutcome(message="Search needs the name to look up.")
        from app.semantic_search import find_items
        from flask import has_app_context, current_app
        if has_app_context():
            items, _mode = find_items(current_app.config, query)
            return ActionOutcome(result=items)
        return ActionOutcome(result=search_items(database_path, query))
    if item_match and method == "GET":
        return ActionOutcome(result=_require_item(database_path, int(item_match.group(1))))

    if path == "/api/inventory" and method == "POST":
        item = _validated_create(payload)
        preview = {"name": item.name, "quantity": item.quantity, "price": item.price}
        if not confirmed:
            return ActionOutcome(
                result=preview,
                confirmation_required=True,
                message="Confirm this create before it is saved.",
            )
        return ActionOutcome(result=create_item(
            database_path,
            item.name,
            item.quantity,
            item.price,
            description=item.description,
            category=item.category,
        ))

    item_id = int(item_match.group(1))
    current = _require_item(database_path, item_id)
    if method == "DELETE":
        if not confirmed:
            return ActionOutcome(
                result=current,
                confirmation_required=True,
                message="Confirm this delete before the item is removed.",
            )
        if not delete_item(database_path, item_id):
            raise InventoryActionError("Inventory item not found", 404)
        return ActionOutcome(result={"id": item_id})

    updated = _validated_update(payload)
    preview = {
        "id": item_id,
        "name": current["name"] if updated.name is None else updated.name,
        "quantity": current["quantity"] if updated.quantity is None else updated.quantity,
        "price": current["price"] if updated.price is None else updated.price,
    }
    if not confirmed:
        return ActionOutcome(
            result=preview,
            confirmation_required=True,
            message="Confirm this update before it is saved.",
        )
    saved = update_item(
        database_path,
        item_id,
        name=updated.name,
        quantity=updated.quantity,
        price=updated.price,
        description=updated.description,
        category=updated.category,
    )
    if saved is None:
        raise InventoryActionError("Inventory item not found", 404)
    return ActionOutcome(result=saved)


def _is_allowed(method: str, path: str) -> bool:
    if method == "GET" and path in {"/api/inventory", "/api/search"}:
        return True
    if method == "POST" and path == "/api/inventory":
        return True
    if method in {"GET", "PUT", "DELETE"} and re.fullmatch(r"/api/inventory/\d+", path):
        return True
    return False


def _normalize(action: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    endpoint = str(action.get("api_endpoint") or "").strip()
    method = str(action.get("method") or "").upper()
    payload = action.get("api_payload") or {}
    if not isinstance(payload, dict):
        payload = {}

    if " " in endpoint:
        maybe_method, rest = endpoint.split(None, 1)
        if maybe_method.upper() in _METHODS:
            method = maybe_method.upper()
            endpoint = rest.strip()

    parsed = urlparse(endpoint)
    path = parsed.path or endpoint
    if parsed.query:
        for key, values in parse_qs(parsed.query).items():
            payload.setdefault(key, values[0])

    if not method:
        instruction = str(action.get("payload_instruction") or "").upper()
        for candidate in ("DELETE", "POST", "PUT", "GET"):
            if candidate in instruction.split():
                method = candidate
                break
    return method, path, payload


def _is_type_template(payload: Dict[str, Any]) -> bool:
    return any(
        isinstance(value, str) and value.lower() in _TYPE_PLACEHOLDERS
        for value in payload.values()
    )


def _filters(payload: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    name = payload.get("name") or payload.get("q")
    if name:
        filters["name"] = str(name)
    try:
        for key in _FILTER_INTS:
            if payload.get(key) is not None:
                filters[key] = int(payload[key])
        for key in _FILTER_FLOATS:
            if payload.get(key) is not None:
                filters[key] = float(payload[key])
    except (TypeError, ValueError) as exc:
        raise InventoryActionError("Inventory filters must be numbers", 422) from exc
    return filters


def _validated_create(payload: Dict[str, Any]) -> InventoryItemCreate:
    try:
        return InventoryItemCreate(**payload)
    except ValidationError as exc:
        raise InventoryActionError(
            "Create needs name, quantity, and price.",
            422,
        ) from exc


def _validated_update(payload: Dict[str, Any]) -> InventoryItemUpdate:
    try:
        item = InventoryItemUpdate(**payload)
    except ValidationError as exc:
        raise InventoryActionError("Update fields are invalid.", 422) from exc
    if not item.has_changes():
        raise InventoryActionError("Update needs at least one of name, quantity, or price.", 422)
    return item


def _require_item(database_path: str, item_id: int) -> Dict[str, Any]:
    item = get_item(database_path, item_id)
    if item is None:
        raise InventoryActionError("Inventory item not found", 404)
    return item

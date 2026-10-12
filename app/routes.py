"""
Flask API Routes
"""

import logging
import re
from flask import Blueprint, request, jsonify, current_app
from pydantic import ValidationError

from app.models import (
    QueryRequest,
    StructuredAPIResponse,
    QueryResponse,
    HealthResponse,
    InventoryItemCreate,
    InventoryItemUpdate,
    ConfirmRequest,
)
from app.llm_service import LLMService
from app.qdrant_service import QdrantService
from app import inventory_db
from app.semantic_search import find_items, index_item, reindex_pending, remove_item
from app.inventory_actions import (
    InventoryActionError,
    execute_confirmed_action,
    plan_inventory_action,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.before_request
def require_api_key():
    """Require X-API-Key when API_KEY is configured. Health stays open for Compose."""
    expected = current_app.config.get("API_KEY") or ""
    if not expected or request.path == "/api/health":
        return None
    if request.headers.get("X-API-Key") != expected:
        return jsonify({"success": False, "data": None, "error": "Unauthorized"}), 401


def get_llm_service() -> LLMService:
    """Get LLM service instance from app config."""
    return LLMService(
        ollama_base_url=current_app.config["OLLAMA_BASE_URL"],
        model=current_app.config["OLLAMA_MODEL"],
        max_tokens=current_app.config["MAX_TOKENS"],
        temperature=current_app.config["TEMPERATURE"]
    )


def get_qdrant_service() -> QdrantService:
    """Get Qdrant service instance from app config."""
    return QdrantService(
        host=current_app.config["QDRANT_HOST"],
        port=current_app.config["QDRANT_PORT"]
    )


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for all services."""
    llm_service = get_llm_service()
    qdrant_service = get_qdrant_service()

    llm_healthy = llm_service.check_health()
    qdrant_healthy = qdrant_service.check_health()

    services = {
        "llm": "healthy" if llm_healthy else "unhealthy",
        "qdrant": "healthy" if qdrant_healthy else "unhealthy"
    }

    overall_status = "healthy" if all([llm_healthy, qdrant_healthy]) else "degraded"

    stale = len(inventory_db.list_unsynced_items(current_app.config["DATABASE_PATH"]))
    response = HealthResponse(
        status=overall_status,
        services=services,
        embedding_stale=stale,
    )
    status_code = 200 if overall_status == "healthy" else 503

    return jsonify(response.model_dump()), status_code


@api_bp.route("/query", methods=["POST"])
def query_endpoint():
    """
    Process natural language query and return structured API metadata.
    
    Request Body:
        {
            "question": "How do I list all inventory items?"
        }
    
    Response:
        {
            "success": true,
            "data": {
                "api_endpoint": "/api/inventory",
                "api_payload": null,
                "payload_instruction": "Send GET request to retrieve all items"
            },
            "error": null
        }
    """
    try:
        # Validate request
        data = request.get_json(silent=True)
        if not data:
            return jsonify(QueryResponse(
                success=False,
                error="Request body is required"
            ).model_dump()), 400

        try:
            query_request = QueryRequest(**data)
        except ValidationError as e:
            return jsonify(QueryResponse(
                success=False,
                error=str(e)
            ).model_dump()), 400

        # Get LLM service and generate response
        llm_service = get_llm_service()
        
        try:
            structured_data = _apply_comparison_bounds(
                query_request.question,
                _prefer_semantic_search(
                    query_request.question,
                    llm_service.generate_structured_response(query_request.question),
                ),
            )
            
            # Validate response structure
            api_response = StructuredAPIResponse(**structured_data)
            if api_response.supported is False:
                return jsonify(QueryResponse(
                    success=True,
                    data=api_response,
                    message=api_response.clarification or "This question is outside inventory lookup and changes.",
                ).model_dump()), 200
            try:
                outcome = plan_inventory_action(
                    current_app.config["DATABASE_PATH"],
                    structured_data,
                )
            except InventoryActionError as exc:
                logger.error("Inventory action failed: %s", exc)
                return jsonify(QueryResponse(
                    success=False,
                    data=api_response,
                    error=str(exc),
                ).model_dump()), exc.status_code

            token = None
            if outcome.confirmation_required:
                token = inventory_db.save_pending_action(
                    current_app.config["DATABASE_PATH"],
                    structured_data,
                )
            
            return jsonify(QueryResponse(
                success=True,
                data=api_response,
                result=outcome.result,
                confirmation_required=outcome.confirmation_required,
                confirmation_token=token,
                message=outcome.message,
            ).model_dump()), 200

        except ValueError as e:
            logger.error(f"LLM response validation failed: {e}")
            return jsonify(QueryResponse(
                success=False,
                error=f"Failed to generate valid response: {e}"
            ).model_dump()), 422

        except Exception as e:
            logger.error(f"LLM service error: {e}")
            return jsonify(QueryResponse(
                success=False,
                error="LLM service unavailable"
            ).model_dump()), 503

    except Exception as e:
        logger.exception(f"Unexpected error in query endpoint: {e}")
        return jsonify(QueryResponse(
            success=False,
            error="Internal server error"
        ).model_dump()), 500


@api_bp.route("/query/confirm", methods=["POST"])
def confirm_query_action():
    """Run a create, update, or delete that was returned for confirmation."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify(QueryResponse(
            success=False,
            error="Request body is required"
        ).model_dump()), 400
    try:
        confirm_request = ConfirmRequest(**data)
    except ValidationError as exc:
        return jsonify(QueryResponse(
            success=False,
            error=str(exc)
        ).model_dump()), 400
    database_path = current_app.config["DATABASE_PATH"]
    try:
        action = inventory_db.consume_pending_action(
            database_path,
            confirm_request.confirmation_token,
        )
    except LookupError as exc:
        return jsonify(QueryResponse(success=False, error=str(exc)).model_dump()), 404
    except PermissionError as exc:
        return jsonify(QueryResponse(success=False, error=str(exc)).model_dump()), 409
    except TimeoutError as exc:
        return jsonify(QueryResponse(success=False, error=str(exc)).model_dump()), 410
    try:
        result = execute_confirmed_action(database_path, action)
        _sync_vector(action.get("method"), result)
    except InventoryActionError as exc:
        return jsonify(QueryResponse(
            success=False,
            error=str(exc),
        ).model_dump()), exc.status_code
    return jsonify(QueryResponse(
        success=True,
        data=StructuredAPIResponse(**action),
        result=result,
    ).model_dump()), 200


@api_bp.route("/endpoints", methods=["GET"])
def list_endpoints():
    """List available API endpoints and their descriptions."""
    endpoints = [
        {
            "method": "GET",
            "path": "/api/inventory",
            "description": "List all inventory items",
            "payload": None
        },
        {
            "method": "GET",
            "path": "/api/inventory/<id>",
            "description": "Get a specific inventory item by ID",
            "payload": None
        },
        {
            "method": "POST",
            "path": "/api/inventory",
            "description": "Create a new inventory item",
            "payload": {"name": "string", "quantity": "integer", "price": "float"}
        },
        {
            "method": "PUT",
            "path": "/api/inventory/<id>",
            "description": "Update an existing inventory item",
            "payload": {"name": "string", "quantity": "integer", "price": "float"}
        },
        {
            "method": "DELETE",
            "path": "/api/inventory/<id>",
            "description": "Delete an inventory item",
            "payload": None
        },
        {
            "method": "GET",
            "path": "/api/search",
            "description": "Search inventory items",
            "payload": {"query_param": "q"}
        }
    ]
    return jsonify({"endpoints": endpoints}), 200


def _inventory_filters(args):
    filters = {}
    if args.get("name"):
        filters["name"] = args.get("name")
    for key in ("min_quantity", "max_quantity", "quantity_lt", "quantity_lte", "quantity_gt", "quantity_gte"):
        if args.get(key) not in (None, ""):
            filters[key] = int(args.get(key))
    for key in ("min_price", "max_price"):
        if args.get(key) not in (None, ""):
            filters[key] = float(args.get(key))
    return filters


def _apply_comparison_bounds(question, action):
    """Replace model min/max bounds with the comparison stated in the question."""
    bounds = _bounds_from_question(question)
    if not bounds:
        return action
    rewritten = dict(action)
    rewritten["supported"] = True
    rewritten["method"] = "GET"
    rewritten["api_endpoint"] = "/api/inventory"
    payload = dict(rewritten.get("api_payload") or {})
    if any(key.startswith("quantity_") for key in bounds):
        payload.pop("min_quantity", None)
        payload.pop("max_quantity", None)
    if any(key.startswith("price_") for key in bounds):
        payload.pop("min_price", None)
        payload.pop("max_price", None)
    payload.update(bounds)
    rewritten["api_payload"] = payload
    rewritten["clarification"] = ""
    rewritten["payload_instruction"] = "Use the comparison from the question, including items at the boundary only when the wording includes them."
    return rewritten


def _bounds_from_question(question):
    text = question.lower()
    bounds = {}
    quantity_patterns = (
        (r"(?:fewer|less) than (\d+(?:\.\d+)?)", "quantity_lt"),
        (r"(?:under|below) (\d+(?:\.\d+)?)(?:\s+units?)?", "quantity_lt"),
        (r"(?:at most|no more than) (\d+(?:\.\d+)?)", "quantity_lte"),
        (r"(?:more|greater) than (\d+(?:\.\d+)?)", "quantity_gt"),
        (r"(?:over|above) (\d+(?:\.\d+)?)(?:\s+units?)?", "quantity_gt"),
        (r"at least (\d+(?:\.\d+)?)", "quantity_gte"),
    )
    price_patterns = (
        (r"(?:cheaper than|costs? less than|price (?:is )?under|under) \$?(\d+(?:\.\d+)?)", "price_lt"),
        (r"(?:more expensive than|costs? more than|price (?:is )?over) \$?(\d+(?:\.\d+)?)", "price_gt"),
        (r"(?:at least|costs? at least) \$?(\d+(?:\.\d+)?)(?:\s+dollars?)?", "price_gte"),
    )
    if any(word in text for word in ("price", "cost", "dollar", "cheap", "expensive")):
        for pattern, key in price_patterns:
            match = re.search(pattern, text)
            if match:
                bounds[key] = float(match.group(1))
                return bounds
    for pattern, key in quantity_patterns:
        match = re.search(pattern, text)
        if match:
            number = float(match.group(1))
            bounds[key] = int(number) if number.is_integer() else number
            return bounds
    return bounds


def _prefer_semantic_search(question, action):
    """Route a descriptive lookup to search when the model asks for a name or id."""
    if action.get("supported") is not False:
        return action
    note = f"{action.get('clarification') or ''} {action.get('payload_instruction') or ''}".lower()
    if not re.search(r"\b(name|id|identifier)\b", note):
        return action
    lowered = question.lower()
    if any(word in lowered for word in ("delete", "remove", "update", "change", "create", "add ")):
        return action
    rewritten = dict(action)
    rewritten["supported"] = True
    rewritten["method"] = "GET"
    rewritten["api_endpoint"] = "/api/search"
    rewritten["api_payload"] = {"q": question}
    rewritten["clarification"] = ""
    rewritten["payload_instruction"] = "Search by description. A product name or id is not required."
    return rewritten


def _sync_vector(method, result):
    if not isinstance(result, dict):
        return
    method = (method or "").upper()
    if method == "DELETE" and "id" in result:
        remove_item(current_app.config, result["id"])
    elif method in {"POST", "PUT"} and result.get("name"):
        index_item(current_app.config, result)


def _item_response(data, status_code=200):
    return jsonify({"success": True, "data": data, "error": None}), status_code


def _item_error(message, status_code):
    return jsonify({"success": False, "data": None, "error": message}), status_code


@api_bp.route("/inventory", methods=["GET"])
def list_inventory():
    """List inventory items, optionally filtered by name, quantity, or price."""
    try:
        filters = _inventory_filters(request.args)
    except ValueError as exc:
        return _item_error(str(exc), 400)
    items = inventory_db.list_items(current_app.config["DATABASE_PATH"], **filters)
    return _item_response(items)


@api_bp.route("/inventory/<int:item_id>", methods=["GET"])
def get_inventory_item(item_id):
    """Get one inventory item by id."""
    item = inventory_db.get_item(current_app.config["DATABASE_PATH"], item_id)
    if item is None:
        return _item_error("Inventory item not found", 404)
    return _item_response(item)


@api_bp.route("/inventory", methods=["POST"])
def create_inventory_item():
    """Create an inventory item. Requires name, quantity, and price."""
    data = request.get_json(silent=True)
    if not data:
        return _item_error("Request body is required", 400)
    try:
        payload = InventoryItemCreate(**data)
    except ValidationError as exc:
        return _item_error(str(exc), 400)

    item = inventory_db.create_item(
        current_app.config["DATABASE_PATH"],
        payload.name,
        payload.quantity,
        payload.price,
        description=payload.description,
        category=payload.category,
    )
    item["embedding_synced"] = index_item(current_app.config, item)
    return _item_response(item, 201)


@api_bp.route("/inventory/<int:item_id>", methods=["PUT"])
def update_inventory_item(item_id):
    """Update fields on an existing inventory item."""
    data = request.get_json(silent=True)
    if not data:
        return _item_error("Request body is required", 400)
    try:
        payload = InventoryItemUpdate(**data)
    except ValidationError as exc:
        return _item_error(str(exc), 400)
    if not payload.has_changes():
        return _item_error("At least one of name, quantity, or price is required", 400)

    item = inventory_db.update_item(
        current_app.config["DATABASE_PATH"],
        item_id,
        name=payload.name,
        quantity=payload.quantity,
        price=payload.price,
        description=payload.description,
        category=payload.category,
    )
    if item is None:
        return _item_error("Inventory item not found", 404)
    item["embedding_synced"] = index_item(current_app.config, item)
    return _item_response(item)


@api_bp.route("/inventory/<int:item_id>", methods=["DELETE"])
def delete_inventory_item(item_id):
    """Delete an inventory item."""
    deleted = inventory_db.delete_item(current_app.config["DATABASE_PATH"], item_id)
    if not deleted:
        return _item_error("Inventory item not found", 404)
    remove_item(current_app.config, item_id)
    return _item_response({"id": item_id})


@api_bp.route("/admin/reindex", methods=["POST"])
def reindex_embeddings():
    """Index inventory rows that do not have a current vector."""
    result = reindex_pending(current_app.config)
    return _item_response(result)


@api_bp.route("/admin/backup", methods=["POST"])
def backup_inventory():
    """Write a consistent copy of the SQLite database."""
    path = inventory_db.backup_database(current_app.config["DATABASE_PATH"])
    return _item_response({"path": path}, 201)


@api_bp.route("/search", methods=["GET"])
def search_inventory():
    """Search inventory items by name. Query param: q."""
    query = request.args.get("q", "").strip()
    if not query:
        return _item_error("Query parameter q is required", 400)
    items, mode = find_items(current_app.config, query)
    response, status = _item_response(items)
    body = response.get_json()
    body["search_mode"] = mode
    return jsonify(body), status

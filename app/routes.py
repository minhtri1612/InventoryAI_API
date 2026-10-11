"""
Flask API Routes
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from pydantic import ValidationError

from app.models import QueryRequest, StructuredAPIResponse, QueryResponse, HealthResponse
from app.llm_service import LLMService
from app.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def get_llm_service() -> LLMService:
    """Get LLM service instance from app config."""
    return LLMService(
        vllm_url=current_app.config["VLLM_URL"],
        model=current_app.config["VLLM_MODEL"],
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

    response = HealthResponse(status=overall_status, services=services)
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
            structured_data = llm_service.generate_structured_response(query_request.question)
            
            # Validate response structure
            api_response = StructuredAPIResponse(**structured_data)
            
            return jsonify(QueryResponse(
                success=True,
                data=api_response
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

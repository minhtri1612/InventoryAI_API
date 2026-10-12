"""
Flask API Service for Natural Language to Structured JSON Translation
Integrates with Qdrant for vector search and Ollama for LLM inference
"""

import os

from flask import Flask
from flask_cors import CORS


def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # Load default configuration
    app.config.from_mapping(
        OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        OLLAMA_MODEL=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        QDRANT_HOST=os.getenv("QDRANT_HOST", "localhost"),
        QDRANT_PORT=int(os.getenv("QDRANT_PORT", "6333")),
        MAX_TOKENS=int(os.getenv("MAX_TOKENS", "512")),
        TEMPERATURE=float(os.getenv("TEMPERATURE", "0.1")),
        DATABASE_PATH=os.getenv("DATABASE_PATH", "inventory.db"),
        EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "all-minilm"),
        VECTOR_SIZE=int(os.getenv("VECTOR_SIZE", "384")),
        SEMANTIC_SCORE_THRESHOLD=float(os.getenv("SEMANTIC_SCORE_THRESHOLD", "0.5")),
        API_KEY=os.getenv("API_KEY", ""),
    )

    # Override with custom config if provided
    if config:
        app.config.update(config)

    from app.inventory_db import init_db
    init_db(app.config["DATABASE_PATH"])

    from app.semantic_search import reindex_pending
    try:
        reindex_pending(app.config)
    except Exception:
        pass

    # Register blueprints
    from app.routes import api_bp
    app.register_blueprint(api_bp)

    return app

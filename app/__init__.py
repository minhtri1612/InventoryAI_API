"""
Flask API Service for Natural Language to Structured JSON Translation
Integrates with Qdrant for vector search and vLLM (Phi-2) for LLM inference
"""

from flask import Flask
from flask_cors import CORS


def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # Load default configuration
    app.config.from_mapping(
        VLLM_URL="http://localhost:8000/v1/completions",
        VLLM_MODEL="microsoft/phi-2",
        QDRANT_HOST="localhost",
        QDRANT_PORT=6333,
        MAX_TOKENS=512,
        TEMPERATURE=0.1,
    )

    # Override with custom config if provided
    if config:
        app.config.update(config)

    # Register blueprints
    from app.routes import api_bp
    app.register_blueprint(api_bp)

    return app

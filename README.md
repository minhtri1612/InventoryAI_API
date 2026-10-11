# Inventory API Service

An intelligent API service that combines **Qdrant** for vector search, **Flask** for API endpoints, and **Local LLM (Phi-2 via vLLM)** to translate natural language queries into structured API metadata.

## Features

- **Natural Language Query Processing**: Ask questions in plain English and get structured API metadata
- **Strict JSON Output**: LLM responses are validated against a defined schema
- **Vector Search**: Powered by Qdrant for semantic similarity searches
- **Docker Compose Deployment**: Full stack deployment with a single command

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  Flask API  │────▶│ vLLM (Phi-2)│
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Qdrant    │
                    └─────────────┘
```

## API Response Structure

The LLM returns structured JSON containing:

```json
{
    "api_endpoint": "/api/inventory",
    "api_payload": {"name": "string", "quantity": "integer", "price": "float"},
    "payload_instruction": "Send a POST request with item details in the body"
}
```

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:5000/api/health
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python run.py
```

## API Endpoints

### Query Endpoint
Process natural language queries and return structured API metadata.

```bash
POST /api/query
Content-Type: application/json

{
    "question": "How do I list all inventory items?"
}
```

Response:
```json
{
    "success": true,
    "data": {
        "api_endpoint": "/api/inventory",
        "api_payload": null,
        "payload_instruction": "Send GET request to retrieve all items"
    },
    "error": null
}
```

### Health Check
```bash
GET /api/health
```

### List Available Endpoints
```bash
GET /api/endpoints
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_URL` | vLLM server URL | `http://localhost:8000/v1/completions` |
| `VLLM_MODEL` | Model identifier | `microsoft/phi-2` |
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `MAX_TOKENS` | Maximum tokens for LLM | `512` |
| `TEMPERATURE` | LLM temperature | `0.1` |
| `HUGGING_FACE_HUB_TOKEN` | (Optional) Hugging Face token for gated models | - |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test files
pytest tests/test_routes.py
pytest tests/test_llm_service.py
pytest tests/test_integration.py
```

## Project Structure

```
inventory/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── models.py            # Pydantic models
│   ├── routes.py            # API routes
│   ├── llm_service.py       # vLLM integration
│   └── qdrant_service.py    # Qdrant integration
├── tests/
│   ├── conftest.py          # Test fixtures
│   ├── test_routes.py       # Route tests
│   ├── test_llm_service.py  # LLM service tests
│   ├── test_qdrant_service.py # Qdrant service tests
│   └── test_integration.py  # Integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py
└── README.md
```

## License

MIT

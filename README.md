# Inventory API Service

An API service using **Flask**, **Ollama**, and **Qdrant** to translate natural language queries into structured API metadata. Ollama runs a local model and does not require an NVIDIA GPU.

## Features

- **Natural Language Inventory Actions**: Ollama turns a question into one inventory action. Reads run immediately. Creates, updates, and deletes wait for a one-time confirmation token.
- **Exact Filters in SQLite**: Name, quantity, price, and id lookups use SQL.
- **Semantic Search in Qdrant**: Descriptive questions use embeddings from `all-minilm`, then the matching rows are read from SQLite.
- **Docker Compose Deployment**: API, Ollama, and Qdrant start together.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  Flask API  │────▶│ Ollama      │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Qdrant    │
                    └─────────────┘
```

## API Response Structure

A descriptive question returns the matching items. The model chooses `GET /api/search`, then Qdrant finds ids and SQLite returns the current rows:

```json
{
    "success": true,
    "data": {
        "supported": true,
        "method": "GET",
        "api_endpoint": "/api/search",
        "api_payload": {"q": "tool for driving nails"},
        "payload_instruction": "Search by what the item is used for.",
        "clarification": ""
    },
    "result": [
        {"id": 2, "name": "Hammer", "quantity": 2, "price": 15.0, "description": "steel head tool for driving nails", "category": "tools", "score": 0.84}
    ],
    "confirmation_required": false,
    "confirmation_token": null,
    "message": null,
    "error": null
}
```

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start all services; the first run downloads the model
docker compose down
docker compose up -d --build

# Check service health
curl http://localhost:5000/api/health
```

### Local Development

```bash
# Install dependencies and ensure Ollama is running with the configured model
pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull all-minilm

# Configure local Ollama if using a non-default URL/model
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen2.5:3b

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
        "supported": true,
        "method": "GET",
        "api_endpoint": "/api/inventory",
        "api_payload": null,
        "payload_instruction": "List every inventory item.",
        "clarification": ""
    },
    "result": [],
    "confirmation_required": false,
    "confirmation_token": null,
    "message": null,
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
| `OLLAMA_BASE_URL` | Ollama server base URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model name already available in Ollama | `qwen2.5:3b` |
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `MAX_TOKENS` | Maximum tokens for LLM | `512` |
| `TEMPERATURE` | LLM temperature | `0.1` |
| `DATABASE_PATH` | SQLite file for inventory items | `inventory.db` |
| `EMBEDDING_MODEL` | Ollama embedding model | `all-minilm` |
| `VECTOR_SIZE` | Embedding dimensions stored in Qdrant | `384` |
| `SEMANTIC_SCORE_THRESHOLD` | Minimum Qdrant score returned for a semantic hit | `0.5` |
| `API_KEY` | If set, require header `X-API-Key` on every route except `/api/health` | empty |

## Inventory CRUD

SQLite stores each item, including description and category, and answers exact filters such as quantity and price. Qdrant stores embeddings from Ollama model `all-minilm` (384 dimensions) for semantic search. `GET /api/search?q=` embeds the phrase, finds nearby item ids, then reads those rows from SQLite. If embedding or Qdrant is down, search falls back to a name match and `search_mode` is `name`. Creating, updating, or deleting an item updates its vector. Existing rows are indexed when the API starts.

`POST /api/query` asks Ollama for one inventory action, using a JSON schema so `payload_instruction` is required. Reads and SQL filters run immediately. Questions outside inventory, or actions missing an id or real field values, return `message` and do not change data. Create, update, and delete return a preview plus a `confirmation_token`. `POST /api/query/confirm` runs only that stored action, once, within 10 minutes. Sending the same token again does not create another row.

```bash
# Create
curl -X POST http://localhost:5000/api/inventory \
  -H 'Content-Type: application/json' \
  -d '{"name": "Widget", "quantity": 10, "price": 9.99}'

# List, get, update, delete
curl http://localhost:5000/api/inventory
curl http://localhost:5000/api/inventory/1
curl -X PUT http://localhost:5000/api/inventory/1 \
  -H 'Content-Type: application/json' \
  -d '{"quantity": 8}'
curl -X DELETE http://localhost:5000/api/inventory/1

# Semantic search by what an item is for, or filter the list with SQL
curl 'http://localhost:5000/api/search?q=something%20to%20drive%20nails'
curl 'http://localhost:5000/api/inventory?min_quantity=5&max_price=20'

curl -X POST http://localhost:5000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Do we have a tool for driving nails?"}'

# Natural language read
curl -X POST http://localhost:5000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "List items with quantity at least 5"}'

# A create, update, or delete returns confirmation_token. Send that token once.
curl -X POST http://localhost:5000/api/query/confirm \
  -H 'Content-Type: application/json' \
  -d '{"confirmation_token": "<token from the preview>"}'
```

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
│   ├── llm_service.py       # Ollama chat and JSON schema
│   ├── qdrant_service.py    # Inventory vector collection
│   ├── semantic_search.py   # Embed, search Qdrant, read SQLite
│   ├── inventory_db.py      # SQLite items and confirmation tokens
│   └── inventory_actions.py # Allowlisted actions and write preview
├── tests/
│   ├── conftest.py          # Test fixtures
│   ├── test_routes.py       # Route tests
│   ├── test_llm_service.py  # LLM service tests
│   ├── test_qdrant_service.py # Qdrant service tests
│   ├── test_inventory.py    # CRUD, confirmation, and semantic search
│   └── test_integration.py  # Integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py
└── README.md
```

## License

MIT

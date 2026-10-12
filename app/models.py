"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request model for natural language queries."""
    question: str = Field(..., min_length=1, description="Natural language question")


class StructuredAPIResponse(BaseModel):
    """Structured response model from LLM."""
    supported: bool = Field(default=True, description="False when the question is outside inventory")
    clarification: Optional[str] = Field(default=None, description="What is missing or out of scope")
    method: Optional[str] = Field(default=None, description="HTTP method to call")
    api_endpoint: str = Field(..., description="The relevant API route")
    api_payload: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Payload reference for the API call"
    )
    payload_instruction: str = Field(
        ..., 
        description="Instructions on how to build the request body"
    )


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[StructuredAPIResponse] = Field(
        default=None, 
        description="Structured API metadata"
    )
    result: Optional[Any] = Field(
        default=None,
        description="Inventory rows for a read, or a preview of a change waiting for confirmation"
    )
    confirmation_required: bool = Field(
        default=False,
        description="True when a create, update, or delete is waiting for confirmation"
    )
    confirmation_token: Optional[str] = Field(
        default=None,
        description="One-time token for a previewed create, update, or delete"
    )
    message: Optional[str] = Field(
        default=None,
        description="Clarification when the question is not specific enough to run"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


class ConfirmRequest(BaseModel):
    """Body that spends a one-time confirmation token."""
    confirmation_token: str = Field(..., min_length=1)


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    services: Dict[str, str] = Field(..., description="Status of dependent services")
    embedding_stale: int = Field(default=0, description="Items saved in SQLite without a current vector")


class InventoryItemCreate(BaseModel):
    """Body for creating an inventory item."""
    name: str = Field(..., min_length=1)
    quantity: int = Field(..., ge=0)
    price: float = Field(..., ge=0)
    description: str = ""
    category: str = ""


class InventoryItemUpdate(BaseModel):
    """Body for updating an inventory item. At least one field is required."""
    name: Optional[str] = Field(default=None, min_length=1)
    quantity: Optional[int] = Field(default=None, ge=0)
    price: Optional[float] = Field(default=None, ge=0)
    description: Optional[str] = None
    category: Optional[str] = None

    def has_changes(self) -> bool:
        return any(
            value is not None
            for value in (self.name, self.quantity, self.price, self.description, self.category)
        )

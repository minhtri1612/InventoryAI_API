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
    error: Optional[str] = Field(default=None, description="Error message if any")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    services: Dict[str, str] = Field(..., description="Status of dependent services")

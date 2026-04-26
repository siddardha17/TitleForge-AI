"""
TitleForge AI — Pydantic Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional


class NormalizeRequest(BaseModel):
    raw_title: str = Field(
        ...,
        min_length=3,
        max_length=500,
        example="APPLE IPHONE 15 PRO MAX 256GB BLK",
        description="The raw, inconsistent vendor product title to normalize.",
    )


class NormalizeResponse(BaseModel):
    raw_title: str
    normalized_title: str
    model_backend: str
    processing_time_ms: float


class BulkNormalizeResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[dict]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_backend: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_backend: str
    model_id: Optional[str] = None
    ollama_model: Optional[str] = None
    device: Optional[str] = None

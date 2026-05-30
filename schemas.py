"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    document_ids: list[int] | None = Field(
        default=None,
        description="If set, search only chunks from these document IDs.",
    )


class QueryHit(BaseModel):
    document_id: int
    filename: str
    chunk_id: int
    chunk_index: int
    content: str
    cosine_similarity: float


class QueryResponse(BaseModel):
    question: str
    top_k: int
    document_ids: list[int] | None
    embedding_model: str
    hits: list[QueryHit]
    answer: str
    answer_sources: list[QueryHit] = []
    generation_model: str | None = None


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    upload_date: datetime
    expires_at: datetime
    user_token: str
    extracted_text_length: int
    extracted_text: str
    extracted_text_truncated: bool
    chunk_count: int
    embedded_chunk_count: int
    embedding_model: str | None


class DocumentSummary(BaseModel):
    id: int
    filename: str
    upload_date: datetime
    expires_at: datetime
    user_token: str
    chunk_count: int
    embedded_chunk_count: int


class DocumentListResponse(BaseModel):
    items: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class DocumentDetail(BaseModel):
    id: int
    filename: str
    upload_date: datetime
    expires_at: datetime
    user_token: str
    chunk_count: int
    embedded_chunk_count: int
    embedding_model: str | None


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    gemini: str
    groq: str
    uploads_dir: str
    database_error: str | None = None

"""
Pydantic models for the RAG (Retrieval-Augmented Generation) pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── API Contracts ──


class DocumentIngestRequest(BaseModel):
    """Request body for ingesting documents into ChromaDB."""

    documents: List[str] = Field(..., min_length=1, description="Text chunks to ingest")
    metadatas: List[Dict[str, Any]] = Field(
        ..., min_length=1, description="Per-chunk metadata dicts"
    )
    ids: Optional[List[str]] = Field(
        default=None, description="Optional chunk IDs (auto-generated if omitted)"
    )


class DocumentSearchRequest(BaseModel):
    """Request body for searching documents."""

    query: str = Field(..., min_length=1, description="Natural-language search query")
    n_results: int = Field(default=5, ge=1, le=50, description="Max results to return")
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        description="Cosine distance threshold (lower = more similar)",
    )


class DocumentSearchResult(BaseModel):
    """A single search result from ChromaDB."""

    id: str
    chunk_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    distance: float = 0.0


class DocumentSearchResponse(BaseModel):
    """Response wrapper for search results."""

    results: List[DocumentSearchResult]
    total: int


class RAGContextBlock(BaseModel):
    """Formatted RAG context ready for injection into the LLM prompt."""

    query: str
    results: List[DocumentSearchResult]
    formatted_text: str = ""


class CollectionStatsResponse(BaseModel):
    """ChromaDB collection statistics."""

    collection_name: str
    document_count: int = 0
    embedding_dimension: int = 0
    error: Optional[str] = None


# ── PDF Upload ──


class PDFUploadResponse(BaseModel):
    """Response for a successful PDF upload and ingestion."""

    filename: str
    num_chunks: int
    document_ids: List[str]
    collection_name: str
    status: str = "success"


# ── PDF List ──


class PDFListItem(BaseModel):
    """A single uploaded PDF entry (aggregated from its chunks)."""

    filename: str
    file_hash: str
    num_chunks: int
    uploaded_at: Optional[str] = None


class PDFListResponse(BaseModel):
    """Response for listing all uploaded PDFs."""

    pdfs: List[PDFListItem]
    total: int

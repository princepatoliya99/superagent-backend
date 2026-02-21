"""
RAG routes — document ingestion, search, stats, and health.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.models.rag import (
    CollectionStatsResponse,
    DocumentIngestRequest,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentSearchResult,
)
from app.services.chromadb_service import ChromaDBService
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])

# ── Module-level service references (set via configure()) ──

_chromadb: Optional[ChromaDBService] = None
_embedding: Optional[EmbeddingService] = None


def configure(
    chromadb_service: ChromaDBService,
    embedding_service: EmbeddingService,
) -> None:
    """Wire services into this route module (called at startup)."""
    global _chromadb, _embedding
    _chromadb = chromadb_service
    _embedding = embedding_service


def _chroma() -> ChromaDBService:
    if _chromadb is None or not _chromadb.is_initialized:
        raise HTTPException(status_code=503, detail="ChromaDB service unavailable")
    return _chromadb


# ── Endpoints ──


@router.post("/documents", summary="Ingest documents into ChromaDB")
async def ingest_documents(req: DocumentIngestRequest):
    svc = _chroma()

    if len(req.documents) != len(req.metadatas):
        raise HTTPException(
            status_code=400,
            detail="documents and metadatas must have the same length",
        )

    result = await svc.add_documents(
        documents=req.documents,
        metadatas=req.metadatas,
        ids=req.ids,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail="Failed to ingest documents")
    return result


@router.post("/search", response_model=DocumentSearchResponse, summary="Search documents")
async def search_documents(req: DocumentSearchRequest):
    svc = _chroma()

    raw_results = await svc.search(
        query_text=req.query,
        n_results=req.n_results,
        similarity_threshold=req.similarity_threshold,
    )

    results = [
        DocumentSearchResult(
            id=r["id"],
            chunk_text=r.get("chunk_text", ""),
            metadata=r.get("metadata", {}),
            distance=r.get("distance", 0.0),
        )
        for r in raw_results
    ]

    return DocumentSearchResponse(results=results, total=len(results))


@router.get("/stats", response_model=CollectionStatsResponse, summary="Collection statistics")
async def collection_stats():
    svc = _chroma()
    stats = await svc.get_collection_stats()
    return CollectionStatsResponse(**stats)


@router.delete("/documents/{document_id}", summary="Delete a document by ID")
async def delete_document(document_id: str):
    svc = _chroma()
    ok = await svc.delete_documents([document_id])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete document")
    return {"deleted": True, "id": document_id}


@router.get("/health", summary="RAG pipeline health check")
async def rag_health():
    embedding_ok = False
    chromadb_ok = False

    if _embedding and _embedding.is_initialized:
        embedding_ok = await _embedding.health_check()

    if _chromadb and _chromadb.is_initialized:
        chromadb_ok = await _chromadb.health_check()

    healthy = embedding_ok and chromadb_ok
    return {
        "healthy": healthy,
        "embedding_service": "ok" if embedding_ok else "unavailable",
        "chromadb_service": "ok" if chromadb_ok else "unavailable",
    }

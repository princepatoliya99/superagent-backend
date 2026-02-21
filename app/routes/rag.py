"""
RAG routes — document ingestion, search, stats, and health.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config.settings import settings
from app.models.rag import (
    CollectionStatsResponse,
    DocumentIngestRequest,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentSearchResult,
    PDFListItem,
    PDFListResponse,
    PDFUploadResponse,
)
from app.services.chromadb_service import ChromaDBService
from app.services.embedding_service import EmbeddingService
from app.services.pdf_service import PDFService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])

# ── Module-level service references (set via configure()) ──

_chromadb: Optional[ChromaDBService] = None
_embedding: Optional[EmbeddingService] = None
_pdf: Optional[PDFService] = None


def configure(
    chromadb_service: ChromaDBService,
    embedding_service: EmbeddingService,
    pdf_service: Optional[PDFService] = None,
) -> None:
    """Wire services into this route module (called at startup)."""
    global _chromadb, _embedding, _pdf
    _chromadb = chromadb_service
    _embedding = embedding_service
    _pdf = pdf_service


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


@router.get(
    "/pdfs",
    response_model=PDFListResponse,
    summary="List all uploaded PDFs",
)
async def list_pdfs():
    """Return a deduplicated list of all PDFs that have been uploaded."""
    svc = _chroma()

    try:
        raw = await svc.get_documents_by_source("pdf_upload")
    except Exception as exc:
        logger.error("Failed to list PDFs: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list PDFs")

    # Group chunks by file_hash to produce one entry per PDF
    pdf_map: dict[str, dict] = {}
    for meta in raw:
        fh = meta.get("file_hash", "unknown")
        if fh not in pdf_map:
            pdf_map[fh] = {
                "filename": meta.get("original_filename", "unknown.pdf"),
                "file_hash": fh,
                "num_chunks": 0,
                "uploaded_at": meta.get("uploaded_at"),
            }
        pdf_map[fh]["num_chunks"] += 1

    pdfs = [PDFListItem(**v) for v in pdf_map.values()]
    # Sort by uploaded_at descending (newest first), None last
    pdfs.sort(key=lambda p: p.uploaded_at or "", reverse=True)
    return PDFListResponse(pdfs=pdfs, total=len(pdfs))


@router.post(
    "/upload-pdf",
    response_model=PDFUploadResponse,
    summary="Upload a PDF, parse, chunk, embed, and store in ChromaDB",
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    chunk_size: Optional[int] = Query(
        default=None,
        ge=100,
        le=10000,
        description=f"Chunk size in characters (default: {settings.PDF_CHUNK_SIZE})",
    ),
    chunk_overlap: Optional[int] = Query(
        default=None,
        ge=0,
        le=2000,
        description=f"Chunk overlap in characters (default: {settings.PDF_CHUNK_OVERLAP})",
    ),
):
    """Upload a PDF file, parse it with LangChain, generate embeddings, and store in ChromaDB."""
    svc = _chroma()

    if _pdf is None or not _pdf.is_initialized:
        raise HTTPException(status_code=503, detail="PDF service unavailable")

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported, got: '{file.filename}'",
        )

    if file.content_type and file.content_type != "application/pdf":
        logger.warning(
            "Content-Type is '%s' but filename ends with .pdf — proceeding",
            file.content_type,
        )

    # Read file bytes
    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")
    finally:
        await file.close()

    # Parse and chunk
    try:
        documents, metadatas, ids = await _pdf.parse_and_chunk(
            file_bytes=file_bytes,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("PDF parsing failed for '%s'", file.filename)
        raise HTTPException(
            status_code=500, detail=f"PDF parsing failed: {exc}"
        )

    # Ingest into ChromaDB (embeddings are now generated inside add_documents)
    result = await svc.add_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=500, detail="Failed to store documents in ChromaDB"
        )

    logger.info(
        "PDF '%s' uploaded: %d chunks stored in collection '%s'",
        file.filename,
        len(documents),
        settings.CHROMADB_COLLECTION_NAME,
    )

    return PDFUploadResponse(
        filename=file.filename,
        num_chunks=len(documents),
        document_ids=ids,
        collection_name=settings.CHROMADB_COLLECTION_NAME,
    )

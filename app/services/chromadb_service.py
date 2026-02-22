"""
ChromaDB Cloud service for document storage and retrieval.

Manages a single collection with dense (Gemini) embeddings and optional
BM25 sparse embeddings.  Provides add, search, hybrid-search, and delete.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import chromadb

from app.config.settings import settings
from app.services.base.base_service import BaseService
from app.services.embedding_service import EmbeddingServiceChromaDBAdapter
from app.services.gemini_embedding_service import GeminiEmbeddingService

logger = logging.getLogger(__name__)


class ChromaDBService(BaseService):
    """Singleton-style service wrapping a ChromaDB Cloud collection."""

    def __init__(self, gemini_service: GeminiEmbeddingService) -> None:
        self._gemini = gemini_service
        self._ef: Optional[EmbeddingServiceChromaDBAdapter] = None
        self._client: Any = None  # chromadb.ClientAPI
        self._collection: Any = None  # chromadb.Collection

    # ── Lifecycle ──

    async def initialize(self) -> None:
        if not all([
            settings.CHROMADB_API_KEY,
            settings.CHROMADB_TENANT,
            settings.CHROMADB_DATABASE,
        ]):
            logger.warning(
                "ChromaDBService: missing CHROMADB_API_KEY / TENANT / DATABASE "
                "— service will remain uninitialised."
            )
            return

        if not self._gemini.is_initialized:
            logger.warning(
                "ChromaDBService: GeminiEmbeddingService not ready — "
                "cannot create embedding function."
            )
            return

        try:
            # Build the sync ChromaDB adapter for the embedding function
            self._ef = EmbeddingServiceChromaDBAdapter(self._gemini)

            # Connect to ChromaDB Cloud
            self._client = chromadb.CloudClient(
                api_key=settings.CHROMADB_API_KEY,
                tenant=settings.CHROMADB_TENANT,
                database=settings.CHROMADB_DATABASE,
            )

            # Get or create collection
            collection_name = settings.CHROMADB_COLLECTION_NAME
            try:
                self._collection = self._client.get_collection(name=collection_name)
                logger.info(
                    "Using existing ChromaDB collection '%s' (count=%d)",
                    collection_name,
                    self._collection.count(),
                )
            except Exception:
                self._collection = self._client.create_collection(
                    name=collection_name,
                )
                logger.info("Created ChromaDB collection '%s'", collection_name)

            await super().initialize()
            logger.info(
                "ChromaDBService ready: collection=%s, embedding_dim=%d",
                collection_name,
                self._ef.vector_dimension,
            )

        except Exception as exc:
            logger.error("ChromaDBService initialisation failed: %s", exc, exc_info=True)
            raise

    async def health_check(self) -> bool:
        if not self.is_initialized or self._collection is None:
            return False
        try:
            count = await asyncio.to_thread(self._collection.count)
            return count >= 0
        except Exception as exc:
            logger.error("ChromaDB health-check failed: %s", exc)
            return False

    async def shutdown(self) -> None:
        self._client = None
        self._collection = None
        self._ef = None
        await super().shutdown()

    # ── Document ingestion ──

    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add documents to the collection.  Embeddings are generated via the
        configured Gemini embedding function.

        Args:
            documents: Text chunks.
            metadatas: Per-chunk metadata dicts.
            ids: Optional chunk IDs (auto-generated if omitted).

        Returns:
            Dict with ``success``, ``total_chunks``, and latency metrics.
        """
        self._ensure_initialized()
        t0 = time.time()
        metrics: Dict[str, Any] = {
            "success": False,
            "total_chunks": len(documents),
            "total_latency_ms": 0.0,
        }

        if not documents:
            logger.error("add_documents: empty documents list")
            return metrics

        if len(documents) != len(metadatas):
            logger.error("add_documents: documents and metadatas length mismatch")
            return metrics

        if ids is None:
            ids = [f"chunk_{uuid.uuid4().hex}" for _ in range(len(documents))]

        # Sanitise metadata values (ChromaDB only accepts str/int/float/bool)
        processed = []
        for md in metadatas:
            clean = {}
            for k, v in md.items():
                if isinstance(v, list):
                    clean[k] = ",".join(str(x) for x in v)
                elif v is None:
                    clean[k] = ""
                else:
                    clean[k] = v
            processed.append(clean)

        # Generate embeddings for storage using RETRIEVAL_DOCUMENT task type
        assert self._ef is not None, "Embedding function not initialised"
        logger.info("Generating embeddings for %d document chunks…", len(documents))
        embeddings = await asyncio.to_thread(
            self._ef.get_embeddings, documents, "RETRIEVAL_DOCUMENT"
        )
        if not embeddings or len(embeddings) != len(documents):
            logger.error("Embedding generation failed or returned wrong count")
            return metrics

        # Filter out items whose embeddings are empty (failed generation).
        # embed_batch returns [] for texts that couldn't be embedded, and
        # ChromaDB will raise "list index out of range" on empty vectors.
        valid = [
            (i, doc, emb, meta)
            for i, (doc, emb, meta) in enumerate(zip(documents, embeddings, processed))
            if emb  # skip empty embeddings
        ]

        if not valid:
            logger.error("All embeddings failed — nothing to add")
            return metrics

        if len(valid) < len(documents):
            skipped = len(documents) - len(valid)
            logger.warning(
                "Skipping %d/%d chunks with failed embeddings",
                skipped,
                len(documents),
            )

        valid_ids = [ids[i] for i, *_ in valid]
        valid_docs = [doc for _, doc, _, _ in valid]
        valid_embs = [emb for _, _, emb, _ in valid]
        valid_meta = [meta for _, _, _, meta in valid]

        def _add():
            self._collection.add(
                ids=valid_ids,
                documents=valid_docs,
                embeddings=valid_embs,
                metadatas=valid_meta,
            )

        await asyncio.to_thread(_add)

        latency = (time.time() - t0) * 1000
        metrics.update(
            success=True,
            total_chunks=len(valid_ids),
            total_latency_ms=round(latency, 2),
        )
        logger.info(
            "Added %d documents to ChromaDB (%.2f ms)", len(valid_ids), latency
        )
        return metrics

    # ── Search ──

    async def search(
        self,
        query_text: str,
        n_results: int = 10,
        similarity_threshold: float = 0.85,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense vector search using an explicit query embedding.

        Args:
            query_text: Natural-language query.
            n_results: Max results to return.
            similarity_threshold: Cosine distance threshold (lower = more similar).
            where: Optional ChromaDB ``where`` filter dict.

        Returns:
            List of result dicts with ``id``, ``chunk_text``, ``metadata``, ``distance``.
        """
        self._ensure_initialized()
        assert self._ef is not None

        t0 = time.time()

        # Generate query embedding with RETRIEVAL_QUERY task type
        query_emb = self._ef.get_embedding(query_text, task_type="RETRIEVAL_QUERY")
        if not query_emb:
            logger.error("Failed to generate query embedding")
            return []

        query_params: Dict[str, Any] = {
            "query_embeddings": [query_emb],
            "n_results": n_results * 3,  # over-fetch, then filter
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = await asyncio.to_thread(lambda: self._collection.query(**query_params))

        formatted: List[Dict[str, Any]] = []
        if results and results.get("ids") and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                if distance >= similarity_threshold:
                    continue
                formatted.append({
                    "id": cid,
                    "chunk_text": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": distance,
                })
                if len(formatted) >= n_results:
                    break

        latency = (time.time() - t0) * 1000
        logger.info(
            "ChromaDB search returned %d results (%.2f ms)", len(formatted), latency
        )
        return formatted

    # ── Delete ──

    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by their IDs."""
        self._ensure_initialized()
        try:
            await asyncio.to_thread(lambda: self._collection.delete(ids=ids))
            logger.info("Deleted %d documents from ChromaDB", len(ids))
            return True
        except Exception as exc:
            logger.error("delete_documents failed: %s", exc)
            return False

    # ── List by source ──

    async def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Return all metadata dicts whose ``source`` field matches *source*.

        Used, for example, to enumerate uploaded PDFs without pulling full
        document text.
        """
        self._ensure_initialized()

        try:
            results = await asyncio.to_thread(
                lambda: self._collection.get(
                    where={"source": source},
                    include=["metadatas"],
                )
            )
            return results.get("metadatas", []) or []
        except Exception as exc:
            logger.error("get_documents_by_source failed: %s", exc)
            return []

    # ── Stats ──

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic collection statistics."""
        self._ensure_initialized()
        try:
            count = await asyncio.to_thread(self._collection.count)
            return {
                "collection_name": settings.CHROMADB_COLLECTION_NAME,
                "document_count": count,
                "embedding_dimension": self._ef.vector_dimension if self._ef else 0,
            }
        except Exception as exc:
            logger.error("get_collection_stats failed: %s", exc)
            return {
                "collection_name": settings.CHROMADB_COLLECTION_NAME,
                "document_count": 0,
                "error": str(exc),
            }

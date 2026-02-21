"""
Async embedding service and ChromaDB-compatible sync adapter.

``EmbeddingService`` wraps ``GeminiEmbeddingService`` with ``asyncio.to_thread``
so it can be called from async code.

``EmbeddingServiceChromaDBAdapter`` exposes the synchronous interface that
ChromaDB expects from an embedding function (callable with ``List[str]``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from app.services.base.base_service import BaseService
from app.services.gemini_embedding_service import GeminiEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingService(BaseService):
    """Async wrapper around :class:`GeminiEmbeddingService`."""

    def __init__(self, gemini_service: GeminiEmbeddingService) -> None:
        self._gemini = gemini_service

    # ── Lifecycle ──

    async def initialize(self) -> None:
        # GeminiEmbeddingService must already be initialised
        if not self._gemini.is_initialized:
            logger.warning(
                "EmbeddingService: underlying GeminiEmbeddingService is not "
                "initialised — embedding calls will fail."
            )
            return
        await super().initialize()

    async def health_check(self) -> bool:
        return self.is_initialized and await self._gemini.health_check()

    # ── Async API ──

    async def embed_content(
        self,
        content: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> Optional[List[float]]:
        """Generate an embedding asynchronously for a single text."""
        self._ensure_initialized()
        return await asyncio.to_thread(self._gemini.embed_text, content, task_type)

    async def embed_batch(
        self,
        contents: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[Optional[List[float]]]:
        """Generate embeddings asynchronously for multiple texts."""
        self._ensure_initialized()
        raw = await asyncio.to_thread(self._gemini.embed_batch, contents, task_type)
        return [vec if vec else None for vec in raw]

    def get_embedding_dimensions(self) -> int:
        return self._gemini.vector_dimension


class EmbeddingServiceChromaDBAdapter:
    """
    Synchronous adapter that satisfies ChromaDB's embedding-function protocol.

    ChromaDB expects:
    - ``__call__(input: List[str]) -> List[List[float]]``
    - optionally ``vector_dimension`` property
    """

    def __init__(self, gemini_service: GeminiEmbeddingService) -> None:
        self._gemini = gemini_service

    def get_embedding(
        self,
        text: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[float]:
        """Synchronous single-text embedding. Returns empty list on failure."""
        result = self._gemini.embed_text(text, task_type)
        return result if result is not None else []

    def get_embeddings(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        """Synchronous batch embedding."""
        return self._gemini.embed_batch(texts, task_type)

    @property
    def vector_dimension(self) -> int:
        return self._gemini.vector_dimension

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB callable: ``List[str] -> List[List[float]]``."""
        return self.get_embeddings(input, task_type="RETRIEVAL_DOCUMENT")

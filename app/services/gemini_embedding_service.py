"""
Gemini embedding service using the official google-genai SDK.

Provides synchronous embedding generation for single texts and batches.
Intended to be wrapped by ``EmbeddingService`` for async usage or
``EmbeddingServiceChromaDBAdapter`` for ChromaDB compatibility.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np
from google import genai
from google.genai import types

from app.config.settings import settings
from app.services.base.base_service import BaseService

logger = logging.getLogger(__name__)

# Gemini allows at most 100 items per batch; stay safely under the limit.
_BATCH_EMBED_MAX = 50


class GeminiEmbeddingService(BaseService):
    """Low-level Gemini embedding client (synchronous API calls)."""

    def __init__(self) -> None:
        self._api_key: str = settings.GEMINI_EMBED_API_KEY
        self._model_name: str = settings.GEMINI_EMBED_MODEL
        self._dimension: int = settings.GEMINI_EMBED_DIMENSION
        self._client: Optional[genai.Client] = None

    # ── Lifecycle ──

    async def initialize(self) -> None:
        if not self._api_key:
            logger.warning(
                "GeminiEmbeddingService: GEMINI_EMBED_API_KEY is empty — "
                "service will remain uninitialised."
            )
            return

        # Validate dimension range (128 – 3072 for gemini-embedding-001)
        if not (128 <= self._dimension <= 3072):
            logger.warning(
                "Invalid embedding dimension %d. Falling back to 768.",
                self._dimension,
            )
            self._dimension = 768

        self._client = genai.Client(api_key=self._api_key)
        await super().initialize()
        logger.info(
            "GeminiEmbeddingService ready: model=%s, dimension=%d",
            self._model_name,
            self._dimension,
        )

    async def health_check(self) -> bool:
        """Test-embed a short string to verify the service is operational."""
        if not self.is_initialized or self._client is None:
            return False
        try:
            vec = self.embed_text("health-check")
            return vec is not None and len(vec) == self._dimension
        except Exception as exc:
            logger.error("GeminiEmbeddingService health-check failed: %s", exc)
            return False

    # ── Public API (sync) ──

    def embed_text(
        self,
        text: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> Optional[List[float]]:
        """
        Generate an embedding vector for a single text string.

        Args:
            text: Input text.
            task_type: Gemini task type — ``RETRIEVAL_DOCUMENT``,
                ``RETRIEVAL_QUERY``, ``SEMANTIC_SIMILARITY``, etc.

        Returns:
            Normalised embedding vector, or ``None`` on failure.
        """
        self._ensure_initialized()
        assert self._client is not None

        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        try:
            t0 = time.perf_counter()
            result = self._client.models.embed_content(
                model=self._model_name,
                contents=text.strip(),
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._dimension,
                ),
            )

            if result.embeddings and len(result.embeddings) > 0:
                values = result.embeddings[0].values
                if values is None:
                    logger.error("Embedding values are None in API response")
                    return None
                normalised = self._normalise(values)
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.debug(
                    "[Embedding] single text — %.2f ms, %d dims",
                    latency_ms,
                    len(normalised),
                )
                return normalised

            logger.error("No embedding data in API response")
            return None

        except Exception as exc:
            logger.error("Gemini embed_text failed: %s", exc)
            return None

    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts in one call.

        Internally chunks into groups of ``_BATCH_EMBED_MAX`` to stay within
        the Gemini per-request limit.

        Args:
            texts: List of input texts.
            task_type: Gemini task type.

        Returns:
            List of normalised embedding vectors (empty list for failed items).
        """
        self._ensure_initialized()
        assert self._client is not None

        if not texts:
            return []

        # Track non-empty positions
        indexed = [(i, t.strip()) for i, t in enumerate(texts) if t and t.strip()]
        if not indexed:
            return [[] for _ in texts]

        embeddings: List[List[float]] = [[] for _ in texts]

        try:
            total_ms = 0.0

            for start in range(0, len(indexed), _BATCH_EMBED_MAX):
                chunk = indexed[start : start + _BATCH_EMBED_MAX]
                content_list = [t for _, t in chunk]

                t0 = time.perf_counter()
                result = self._client.models.embed_content(
                    model=self._model_name,
                    contents=content_list,  # type: ignore[arg-type]
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._dimension,
                    ),
                )
                total_ms += (time.perf_counter() - t0) * 1000

                if result.embeddings:
                    for idx, emb_obj in enumerate(result.embeddings):
                        orig_idx = chunk[idx][0]
                        if emb_obj.values is not None:
                            embeddings[orig_idx] = self._normalise(emb_obj.values)

            failed = [i for i, e in enumerate(embeddings) if len(e) == 0]
            if len(failed) == len(texts):
                raise RuntimeError(
                    f"All {len(texts)} embeddings failed. Check Gemini API key and config."
                )
            if failed:
                logger.warning(
                    "Failed %d / %d embeddings. Indices: %s",
                    len(failed),
                    len(texts),
                    failed,
                )

            logger.info(
                "[Embedding] batch — %.2f ms total, %d items",
                total_ms,
                len(texts),
            )
            return embeddings

        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("embed_batch error: %s", exc, exc_info=True)
            return [[] for _ in texts]

    # ── Properties ──

    @property
    def vector_dimension(self) -> int:
        return self._dimension

    # ── Private helpers ──

    def _normalise(self, values: List[float]) -> List[float]:
        """Normalise to unit length (skip for 3072-dim, already normalised)."""
        if self._dimension == 3072:
            return list(values)
        arr = np.array(values, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return list(values)
        return (arr / norm).tolist()

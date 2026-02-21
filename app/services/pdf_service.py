"""
PDF parsing and chunking service using LangChain.

Accepts PDF file uploads, extracts text via UnstructuredPDFLoader,
and splits into overlapping chunks via RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config.settings import settings
from app.services.base.base_service import BaseService

logger = logging.getLogger(__name__)


class PDFService(BaseService):
    """
    Parses PDF files and splits them into text chunks with metadata.

    Uses LangChain's ``UnstructuredPDFLoader`` for extraction and
    ``RecursiveCharacterTextSplitter`` for chunking.
    """

    async def initialize(self) -> None:
        await super().initialize()
        logger.info(
            "PDFService ready: chunk_size=%d, chunk_overlap=%d, max_file_mb=%d",
            settings.PDF_CHUNK_SIZE,
            settings.PDF_CHUNK_OVERLAP,
            settings.PDF_MAX_FILE_SIZE_MB,
        )

    async def health_check(self) -> bool:
        return self.is_initialized

    async def parse_and_chunk(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Parse a PDF and split into text chunks.

        Args:
            file_bytes: Raw PDF file content.
            filename: Original filename (used for metadata).
            chunk_size: Override for ``PDF_CHUNK_SIZE`` setting.
            chunk_overlap: Override for ``PDF_CHUNK_OVERLAP`` setting.

        Returns:
            Tuple of (documents, metadatas, ids) ready for ChromaDB ingestion.

        Raises:
            ValueError: If file is empty or exceeds size limit.
            RuntimeError: If PDF parsing fails.
        """
        self._ensure_initialized()

        # Validate file size
        max_bytes = settings.PDF_MAX_FILE_SIZE_MB * 1024 * 1024
        if not file_bytes:
            raise ValueError("Empty file provided")
        if len(file_bytes) > max_bytes:
            raise ValueError(
                f"File size ({len(file_bytes) / (1024 * 1024):.1f} MB) exceeds "
                f"limit of {settings.PDF_MAX_FILE_SIZE_MB} MB"
            )

        cs = chunk_size or settings.PDF_CHUNK_SIZE
        co = chunk_overlap or settings.PDF_CHUNK_OVERLAP

        # Write to a temp file for LangChain loader
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            logger.info("Parsing PDF '%s' (%d bytes)…", filename, len(file_bytes))

            # Lazy imports to avoid heavy startup cost
            from langchain_community.document_loaders import UnstructuredPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            # Load PDF
            loader = UnstructuredPDFLoader(tmp_path, mode="elements")
            raw_docs = loader.load()

            if not raw_docs:
                raise RuntimeError(f"No text extracted from PDF '{filename}'")

            logger.info(
                "Extracted %d elements from '%s'", len(raw_docs), filename
            )

            # Split into overlapping chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cs,
                chunk_overlap=co,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_documents(raw_docs)

            if not chunks:
                raise RuntimeError(
                    f"Text splitting produced zero chunks for '{filename}'"
                )

            logger.info(
                "Split '%s' into %d chunks (size=%d, overlap=%d)",
                filename,
                len(chunks),
                cs,
                co,
            )

            # Build output lists
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            ids: List[str] = []

            # Deterministic ID prefix based on file content
            file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]

            for idx, chunk in enumerate(chunks):
                text = chunk.page_content.strip()
                if not text:
                    continue

                chunk_id = f"pdf_{file_hash}_{idx:04d}"
                metadata = {
                    "source": "pdf_upload",
                    "original_filename": filename,
                    "file_hash": file_hash,
                    "chunk_index": idx,
                    "chunk_size": len(text),
                    "total_chunks": len(chunks),
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                }

                # Carry forward any page info from the loader
                if hasattr(chunk, "metadata") and chunk.metadata:
                    page = chunk.metadata.get("page_number")
                    if page is not None:
                        metadata["page_number"] = page
                    category = chunk.metadata.get("category")
                    if category:
                        metadata["element_category"] = category

                documents.append(text)
                metadatas.append(metadata)
                ids.append(chunk_id)

            logger.info(
                "Prepared %d non-empty chunks from '%s' for ingestion",
                len(documents),
                filename,
            )
            return documents, metadatas, ids

        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

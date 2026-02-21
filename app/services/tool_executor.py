"""
ToolExecutor — single dispatch point for all tool execution.

Routes tool calls to either:
- **Composio Tool Router** (MCP meta-tools and discovered toolkit tools)
- **Internal RAG** tools (document search for context injection)

Also provides helper methods for building RAG context blocks that
get injected into the LLM prompt before each call.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from app.config.settings import settings
from app.services.base.base_service import BaseService
from app.services.chromadb_service import ChromaDBService
from app.services.composio_service import ComposioService

logger = logging.getLogger(__name__)


class ToolExecutor(BaseService):
    """
    Unified tool execution router.

    Every tool call from the agentic loop passes through here. The executor
    inspects the tool name and delegates to the appropriate backend.
    """

    def __init__(
        self,
        composio_service: ComposioService,
        chromadb_service: ChromaDBService,
    ) -> None:
        self._composio = composio_service
        self._chromadb = chromadb_service

    # ── Lifecycle ──

    async def initialize(self) -> None:
        await super().initialize()

    async def health_check(self) -> bool:
        return self.is_initialized

    # ── Tool dispatch ──

    async def execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        user_id: str = "default",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool by name and return the result.

        Routing logic:
        - **Meta tools** (``COMPOSIO_SEARCH_TOOLS``, ``COMPOSIO_MANAGE_CONNECTIONS``,
          ``COMPOSIO_MULTI_EXECUTE_TOOL``, …) are forwarded to
          ``ComposioService.execute_session_meta_tool`` using the session API.
        - Other Composio tools (``{TOOLKIT}_{ACTION}``) are forwarded to
          ``ComposioService.execute_tool_for_user``.
        - ``RAG_SEARCH`` is handled internally via ChromaDBService.
        - Unknown tools return an error dict.

        Args:
            name: Tool name (e.g. ``COMPOSIO_SEARCH_TOOLS``, ``GMAIL_SEND_EMAIL``).
            arguments: Tool arguments dict.
            user_id: User identifier for Composio auth scoping.
            session_id: Tool Router session ID (required for meta tools).

        Returns:
            Result dict with ``data``, ``error``, ``successful`` keys.
        """
        self._ensure_initialized()

        if name == "RAG_SEARCH":
            return await self._execute_rag_search(arguments)

        # Default — treat as a Composio tool
        if self._composio.is_initialized:
            try:
                loop = asyncio.get_running_loop()

                # Meta tools must go through the session execute_meta API
                if ComposioService.is_meta_tool(name):
                    if not session_id:
                        return {
                            "data": {},
                            "error": (
                                f"Meta tool '{name}' requires a session_id but "
                                "none was provided."
                            ),
                            "successful": False,
                        }
                    result = await loop.run_in_executor(
                        None,
                        lambda: self._composio.execute_session_meta_tool(
                            session_id=session_id,
                            slug=name,
                            arguments=arguments,
                        ),
                    )
                else:
                    # Regular Composio tools → tools.execute()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self._composio.execute_tool_for_user(
                            slug=name, arguments=arguments, user_id=user_id
                        ),
                    )
                return result
            except Exception as exc:
                logger.error("ToolExecutor: Composio tool '%s' failed: %s", name, exc)
                return {"data": {}, "error": str(exc), "successful": False}

        return {
            "data": {},
            "error": f"No backend available for tool '{name}'",
            "successful": False,
        }

    # ── RAG context helpers ──

    async def get_rag_context(
        self,
        query: str,
        n_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search ChromaDB and return matching document chunks.

        Args:
            query: Natural-language query (typically the user message).
            n_results: Max results (defaults to ``RAG_TOP_K`` setting).
            similarity_threshold: Distance threshold (defaults to setting).

        Returns:
            List of result dicts from ChromaDB (``id``, ``chunk_text``, …).
        """
        if not settings.RAG_ENABLED:
            return []

        if not self._chromadb.is_initialized:
            logger.debug("ToolExecutor: ChromaDB not initialised — skipping RAG.")
            return []

        k = n_results or settings.RAG_TOP_K
        threshold = similarity_threshold or settings.RAG_SIMILARITY_THRESHOLD

        try:
            return await self._chromadb.search(
                query_text=query,
                n_results=k,
                similarity_threshold=threshold,
            )
        except Exception as exc:
            logger.error("RAG context retrieval failed: %s", exc)
            return []

    @staticmethod
    def format_rag_context_for_prompt(results: List[Dict[str, Any]]) -> str:
        """
        Format RAG search results into a context block suitable for
        prepending to the conversation history.

        Returns an empty string if there are no results.
        """
        if not results:
            return ""

        lines = ["## Relevant Context (from knowledge base)\n"]
        for i, r in enumerate(results, 1):
            text = r.get("chunk_text", "").strip()
            source = r.get("metadata", {}).get("original_filename", "unknown")
            distance = r.get("distance", 0.0)
            lines.append(
                f"### Source {i} (file: {source}, relevance: {1 - distance:.2f})\n{text}\n"
            )
        lines.append(
            "---\nUse the context above to answer the user's question when relevant.\n"
        )
        return "\n".join(lines)

    # ── Classification helper ──

    @staticmethod
    def is_composio_tool(name: str) -> bool:
        """Return ``True`` if the tool name looks like a Composio tool."""
        if name.startswith("COMPOSIO_"):
            return True
        # Composio tools follow TOOLKIT_ACTION pattern (e.g. GMAIL_SEND_EMAIL)
        parts = name.split("_")
        return len(parts) >= 2 and parts[0].isupper()

    # ── Internal handlers ──

    async def _execute_rag_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the internal ``RAG_SEARCH`` tool."""
        query = arguments.get("query", "")
        n_results = arguments.get("n_results", settings.RAG_TOP_K)

        if not query:
            return {"data": [], "error": "Missing 'query' argument", "successful": False}

        results = await self.get_rag_context(query, n_results=n_results)
        return {
            "data": {
                "results": results,
                "count": len(results),
            },
            "successful": True,
        }

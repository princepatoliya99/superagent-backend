"""
Abstract base class for LLM services.

Any LLM provider (OpenAI, Gemini, Claude …) must inherit from this
class and implement the raw dict-based ``chat_raw`` and
``chat_with_tools_raw`` methods.  These are the only two methods the
agentic loop (SuperAgentService) depends on.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.services.base.base_service import BaseService


class BaseLLMService(BaseService):
    """Abstract base for every LLM integration."""

    @abstractmethod
    async def chat_raw(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Send raw dict messages to the LLM and return a raw dict response.

        Args:
            messages: List of ``{"role": …, "content": …}`` dicts.

        Returns:
            ``{"role": "assistant", "content": "…"}``
        """
        ...

    @abstractmethod
    async def chat_with_tools_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any],
    ) -> dict[str, Any]:
        """
        Send raw dict messages with tool definitions to the LLM.

        Returns a dict with ``role``, ``content``, and optionally
        ``tool_calls`` (list of ``{"id", "name", "arguments"}`` dicts).

        Args:
            messages: List of ``{"role": …, "content": …}`` dicts.
            tools: OpenAI-compatible tool/function definitions.

        Returns:
            ``{"role": "assistant", "content": "…", "tool_calls": [...] | None}``
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the identifier of the underlying model."""
        ...

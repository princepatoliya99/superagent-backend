"""
Abstract base class for LLM services.

Any LLM provider (OpenAI, Gemini, Claude …) must inherit from this
class and implement the ``chat`` and ``chat_with_tools`` methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.models.chat import Message
from app.services.base.base_service import BaseService


class BaseLLMService(BaseService):
    """Abstract base for every LLM integration."""

    @abstractmethod
    async def chat(self, messages: list[Message]) -> Message:
        """
        Send a list of messages to the LLM and return the assistant reply.

        Args:
            messages: Ordered conversation history.

        Returns:
            The assistant's response as a ``Message``.
        """
        ...

    @abstractmethod
    async def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Message:
        """
        Send messages together with a list of tool definitions.

        The LLM may respond with ``tool_calls`` in the returned message.

        Args:
            messages: Ordered conversation history.
            tools: OpenAI-compatible tool/function definitions.

        Returns:
            The assistant's response (possibly containing tool calls).
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the identifier of the underlying model."""
        ...

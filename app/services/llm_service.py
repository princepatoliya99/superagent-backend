"""
Concrete LLM service backed by LiteLLM.

Supports any provider that LiteLLM supports (OpenAI, Gemini, Claude, …)
by simply changing the ``LLM_MODEL`` environment variable.
"""

from __future__ import annotations

import logging
from typing import Any

import litellm

from app.config.settings import settings
from app.models.chat import Message, Role, ToolCall
from app.services.base.base_llm_service import BaseLLMService

logger = logging.getLogger(__name__)


class LLMService(BaseLLMService):
    """LiteLLM-backed implementation of :class:`BaseLLMService`."""

    def __init__(self) -> None:
        self._model: str = settings.LLM_MODEL
        self._api_key: str = settings.LLM_API_KEY

    # ── Lifecycle ──

    async def initialize(self) -> None:
        # Set the API key so LiteLLM can pick it up
        if self._api_key:
            litellm.api_key = self._api_key
        await super().initialize()

    async def health_check(self) -> bool:
        """Quick health probe — try a tiny completion."""
        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(resp.choices)
        except Exception:
            logger.exception("LLM health-check failed")
            return False

    # ── Core API ──

    async def chat(self, messages: list[Message]) -> Message:
        self._ensure_initialized()

        raw_messages = self._to_raw(messages)
        logger.debug("LLM request (%d messages, model=%s)", len(raw_messages), self._model)

        response = await litellm.acompletion(
            model=self._model,
            messages=raw_messages,
        )

        choice = response.choices[0].message
        return Message(
            role=Role.ASSISTANT,
            content=choice.content or "",
        )

    async def chat_raw(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Send raw dict messages to the LLM and return a raw dict response.

        This is the preferred method when working with ``ConversationHistory``
        which stores messages as plain dicts.

        Args:
            messages: List of ``{"role": …, "content": …}`` dicts.

        Returns:
            ``{"role": "assistant", "content": "…"}``
        """
        self._ensure_initialized()

        logger.debug("LLM raw request (%d messages, model=%s)", len(messages), self._model)

        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
        )

        choice = response.choices[0].message
        return {"role": "assistant", "content": choice.content or ""}

    async def chat_with_tools_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any],
    ) -> dict[str, Any]:
        """
        Send raw dict messages with tool definitions to the LLM.

        Returns a dict with ``role``, ``content``, and optionally ``tool_calls``
        (list of ``{"id", "name", "arguments"}`` dicts).

        Args:
            messages: List of ``{"role": …, "content": …}`` dicts.
            tools: OpenAI-compatible tool definitions from Composio.

        Returns:
            ``{"role": "assistant", "content": "…", "tool_calls": [...] | None}``
        """
        self._ensure_initialized()
        import json as _json

        logger.debug(
            "LLM raw request with %d tools (%d messages, model=%s)",
            len(tools),
            len(messages),
            self._model,
        )

        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
            tools=tools,
        )

        choice = response.choices[0].message
        result: dict[str, Any] = {
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": None,
        }

        if choice.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": (
                        _json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    ),
                }
                for tc in choice.tool_calls
            ]

        return result

    async def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Message:
        self._ensure_initialized()

        raw_messages = self._to_raw(messages)
        logger.debug(
            "LLM request with %d tools (%d messages, model=%s)",
            len(tools),
            len(raw_messages),
            self._model,
        )

        response = await litellm.acompletion(
            model=self._model,
            messages=raw_messages,
            tools=tools,
        )

        choice = response.choices[0].message

        tool_calls: list[ToolCall] | None = None
        if choice.tool_calls:
            import json

            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=(
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    ),
                )
                for tc in choice.tool_calls
            ]

        return Message(
            role=Role.ASSISTANT,
            content=choice.content or "",
            tool_calls=tool_calls,
        )

    def get_model_name(self) -> str:
        return self._model

    # ── Helpers ──

    @staticmethod
    def _to_raw(messages: list[Message]) -> list[dict[str, str]]:
        """Convert ``Message`` objects to dicts that LiteLLM expects."""
        return [{"role": m.role.value, "content": m.content} for m in messages]

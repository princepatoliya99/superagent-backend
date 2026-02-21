"""
Concrete LLM service backed by LiteLLM.

Supports any provider that LiteLLM supports (OpenAI, Gemini, Claude, …)
by simply changing the ``LLM_MODEL`` environment variable.

Provides batch (non-streaming) completions with tool-call support,
Gemini schema sanitization, and structured usage / latency logging.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import Any

import litellm

from app.config.settings import settings
from app.services.base.base_llm_service import BaseLLMService

logger = logging.getLogger(__name__)


class LLMService(BaseLLMService):
    """LiteLLM-backed implementation of :class:`BaseLLMService`."""

    def __init__(self) -> None:
        self._model: str = settings.LLM_MODEL
        self._api_key: str = settings.LLM_API_KEY
        self._temperature: float = settings.LLM_TEMPERATURE

    # ── Lifecycle ──

    async def initialize(self) -> None:
        if self._api_key:
            litellm.api_key = self._api_key
        logger.info(
            "LLMService initialised: model=%s, temperature=%.2f",
            self._model,
            self._temperature,
        )
        await super().initialize()

    async def health_check(self) -> bool:
        """Quick health probe — try a tiny completion."""
        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(resp.choices)  # type: ignore[union-attr]
        except Exception:
            logger.exception("LLM health-check failed")
            return False

    # ── Schema Sanitization (Gemini compatibility) ──

    def _sanitize_schema_for_gemini(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively simplify JSON Schema for Gemini's stricter requirements."""
        if not isinstance(schema, dict):
            return schema

        schema = copy.deepcopy(schema)

        # Flatten anyOf / oneOf to the first variant
        for key in ("anyOf", "oneOf"):
            if key in schema:
                variants = schema.pop(key)
                if variants and isinstance(variants, list):
                    schema.update(variants[0])

        # Recurse into nested schema locations
        if "properties" in schema and isinstance(schema["properties"], dict):
            schema["properties"] = {
                k: self._sanitize_schema_for_gemini(v)
                for k, v in schema["properties"].items()
            }
        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = self._sanitize_schema_for_gemini(schema["items"])
        if "additionalProperties" in schema and isinstance(
            schema["additionalProperties"], dict
        ):
            schema["additionalProperties"] = self._sanitize_schema_for_gemini(
                schema["additionalProperties"]
            )

        return schema

    def _sanitize_tools_for_gemini(self, tools: list[Any]) -> list[Any]:
        """Sanitize tool definitions for Gemini (no-op for other models)."""
        if not self._model or "gemini" not in self._model.lower():
            return tools

        sanitized: list[Any] = []
        for tool in tools:
            tool = copy.deepcopy(tool)
            if (
                "function" in tool
                and "parameters" in tool["function"]
                and isinstance(tool["function"]["parameters"], dict)
            ):
                tool["function"]["parameters"] = self._sanitize_schema_for_gemini(
                    tool["function"]["parameters"]
                )
            sanitized.append(tool)
        return sanitized

    # ── Internal: single batch call ──

    async def _batch_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> Any:
        """
        Execute a single LiteLLM batch completion, log latency & usage.

        Args:
            messages: Raw message dicts.
            tools: Optional tool definitions (already sanitized).

        Returns:
            The raw LiteLLM ``ModelResponse``.
        """
        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "stream": False,
        }
        if tools:
            call_kwargs["tools"] = tools

        start = time.perf_counter()
        response = await litellm.acompletion(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        # Log latency
        logger.info("LLM batch response time: %.2f ms", latency_ms)

        # Log usage if present
        usage = getattr(response, "usage", None)
        if usage:
            logger.debug(
                "LLM usage — prompt_tokens=%s, completion_tokens=%s, total=%s",
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
            )

        return response

    # ── Core API ──

    async def chat_raw(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Send raw dict messages and return the assistant reply.

        Args:
            messages: List of ``{"role": …, "content": …}`` dicts.

        Returns:
            ``{"role": "assistant", "content": "…"}``
        """
        self._ensure_initialized()
        logger.debug(
            "LLM chat_raw: %d messages, model=%s", len(messages), self._model
        )

        response = await self._batch_completion(messages)

        if not response.choices:
            logger.error("LLM returned empty choices. Full response: %s", response)
            raise ValueError(
                f"LLM returned empty response. "
                f"Status: {getattr(response, 'status', 'unknown')}"
            )

        choice = response.choices[0].message
        return {"role": "assistant", "content": choice.content or ""}

    async def chat_with_tools_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any],
    ) -> dict[str, Any]:
        """
        Send raw dict messages with tool definitions.

        Returns a dict with ``role``, ``content``, and optionally
        ``tool_calls`` (list of ``{"id", "name", "arguments"}`` dicts).
        """
        self._ensure_initialized()
        logger.debug(
            "LLM chat_with_tools_raw: %d tools, %d messages, model=%s",
            len(tools),
            len(messages),
            self._model,
        )

        sanitized_tools = self._sanitize_tools_for_gemini(tools)
        response = await self._batch_completion(messages, tools=sanitized_tools)

        if not response.choices:
            logger.error("LLM returned empty choices. Full response: %s", response)
            raise ValueError(
                f"LLM returned empty response. "
                f"Status: {getattr(response, 'status', 'unknown')}"
            )

        choice = response.choices[0].message
        result: dict[str, Any] = {
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": None,
        }

        if choice.tool_calls:
            result["tool_calls"] = self._parse_tool_calls(choice.tool_calls)

        return result

    def get_model_name(self) -> str:
        return self._model

    # ── Helpers ──

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: list[Any]) -> list[dict[str, Any]]:
        """
        Parse LiteLLM tool-call objects into plain dicts.

        Args:
            raw_tool_calls: ``choice.message.tool_calls`` from LiteLLM.

        Returns:
            List of ``{"id": …, "name": …, "arguments": dict}``.
        """
        parsed: list[dict[str, Any]] = []
        for tc in raw_tool_calls:
            try:
                args = (
                    json.loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str)
                    else tc.function.arguments
                )
            except json.JSONDecodeError:
                args = {}
            parsed.append(
                {"id": tc.id, "name": tc.function.name, "arguments": args}
            )
        return parsed

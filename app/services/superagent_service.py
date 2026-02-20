"""
SuperAgent service — agentic orchestrator.

Coordinates LLM, conversation history, and Composio tool search/execution
in an agentic loop:

1. User sends a message
2. Search for relevant tools via Composio
3. LLM decides whether to call a tool or respond directly
4. If tool_calls → execute each tool → feed results back → LLM responds
5. Store everything in conversation history
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional

from app.models.conversation import ConversationHistory, ConversationRole
from app.services.composio_service import ComposioService
from app.services.conversation_service import ConversationService
from app.services.llm_service import LLMService
from app.services.base.base_service import BaseService

logger = logging.getLogger(__name__)

# Max iterations to prevent infinite tool-call loops
MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You are SuperAgent, a helpful AI assistant. "
    "You have access to real-world tools via Composio. "
    "When the user asks you to do something actionable (check email, "
    "search the web, manage calendar, etc.), use the provided tools. "
    "Always explain what you're doing before and after using a tool."
)


class SuperAgentService(BaseService):
    """
    Agentic orchestrator that ties together LLM, conversation history,
    and Composio tool search/execution.
    """

    def __init__(
        self,
        llm_service: LLMService,
        conversation_service: ConversationService,
        composio_service: ComposioService,
    ) -> None:
        self._llm = llm_service
        self._conversations = conversation_service
        self._composio = composio_service

    # ── Lifecycle ──

    async def initialize(self) -> None:
        await super().initialize()

    async def health_check(self) -> bool:
        return self.is_initialized

    # ── Core agentic loop ──

    async def handle_message(
        self,
        user_id: str,
        message: str,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Process a user message through the agentic loop.

        Yields intermediate events and the final reply as dicts:
        - ``{"type": "tool_search", "data": {"query": ..., "tools_found": N}}``
        - ``{"type": "tool_call", "data": {"name": ..., "arguments": ...}}``
        - ``{"type": "tool_result", "data": {"name": ..., "result": ...}}``
        - ``{"type": "reply", "data": {"conversation_id": ..., "content": ...}}``
        - ``{"type": "error", "data": {"message": ...}}``

        Args:
            user_id: User identifier (for Composio auth scoping).
            message: The user's text message.
            conversation_id: Optional existing conversation to continue.

        Yields:
            Event dicts describing each step.
        """
        self._ensure_initialized()

        # 1. Create / resume conversation
        cid = self._conversations.create_conversation(
            conversation_id=conversation_id,
            system_message=SYSTEM_PROMPT,
            should_init_system_message=True,
        )

        # 2. Add user message
        self._conversations.add_message(cid, ConversationRole.USER.value, message)

        # 3. Search for relevant tools (only if Composio is initialized)
        tools: list[Any] = []
        if self._composio.is_initialized:
            try:
                loop = asyncio.get_running_loop()
                tools = await loop.run_in_executor(
                    None,
                    lambda: self._composio.search_tools(user_id, message),
                )
                yield {
                    "type": "tool_search",
                    "data": {
                        "query": message,
                        "tools_found": len(tools),
                    },
                }
            except Exception as exc:
                logger.warning("Tool search failed: %s", exc)
                tools = []

        # 4. Send to LLM (with or without tools)
        history = self._conversations.get_formatted_history_for_model(cid)

        try:
            if tools:
                llm_response = await self._llm.chat_with_tools_raw(history, tools)
            else:
                llm_response = await self._llm.chat_raw(history)
        except Exception as exc:
            logger.exception("LLM call failed")
            yield {"type": "error", "data": {"message": f"LLM error: {exc}"}}
            return

        # 5. Agentic tool-call loop
        iterations = 0
        while llm_response.get("tool_calls") and iterations < MAX_TOOL_ITERATIONS:
            iterations += 1

            # Store the assistant message (with tool calls) in history
            self._conversations.add_message(
                cid,
                ConversationRole.ASSISTANT.value,
                llm_response.get("content", ""),
            )

            for tc in llm_response["tool_calls"]:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                tool_call_id = tc.get("id", "")

                yield {
                    "type": "tool_call",
                    "data": {"name": tool_name, "arguments": tool_args},
                }

                # Execute the tool
                try:
                    loop = asyncio.get_running_loop()
                    tool_result = await loop.run_in_executor(
                        None,
                        lambda tn=tool_name, ta=tool_args: self._composio.execute_tool_for_user(
                            slug=tn, arguments=ta, user_id=user_id
                        ),
                    )
                except Exception as exc:
                    tool_result = {"data": {}, "error": str(exc), "successful": False}

                yield {
                    "type": "tool_result",
                    "data": {"name": tool_name, "result": tool_result},
                }

                # Add tool result to history as a tool message
                self._conversations.add_message(
                    cid,
                    "tool",
                    json.dumps(
                        {
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "result": tool_result,
                        }
                    ),
                )

            # Re-call LLM with updated history (including tool results)
            history = self._conversations.get_formatted_history_for_model(cid)
            try:
                if tools:
                    llm_response = await self._llm.chat_with_tools_raw(history, tools)
                else:
                    llm_response = await self._llm.chat_raw(history)
            except Exception as exc:
                logger.exception("LLM follow-up call failed")
                yield {"type": "error", "data": {"message": f"LLM error: {exc}"}}
                return

        # 6. Store the final assistant reply
        final_content = llm_response.get("content", "")
        self._conversations.add_message(
            cid, ConversationRole.ASSISTANT.value, final_content
        )

        yield {
            "type": "reply",
            "data": {
                "conversation_id": cid,
                "content": final_content,
            },
        }

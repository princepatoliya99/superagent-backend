"""
SuperAgent service — agentic orchestrator.

Coordinates LLM, conversation history, Composio Tool Router, and RAG
context injection in an agentic loop:

1. User sends a message
2. Retrieve relevant RAG context (if enabled) and inject into prompt
3. Provide Composio Tool Router meta tools to the LLM
4. LLM decides whether to call a tool or respond directly
5. If tool_calls → dispatch via ToolExecutor → feed results back → repeat
6. Store everything in conversation history
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional

from app.config.settings import settings
from app.models.conversation import ConversationRole
from app.services.composio_service import ComposioService
from app.services.conversation_service import ConversationService
from app.services.llm_service import LLMService
from app.services.tool_executor import ToolExecutor
from app.services.base.base_service import BaseService

logger = logging.getLogger(__name__)

# Max iterations to prevent infinite tool-call loops
MAX_TOOL_ITERATIONS = 10

SYSTEM_PROMPT = (
    "You are SuperAgent, an intelligent AI assistant with access to real-world tools via Composio "
    "and a knowledge base via RAG.\n\n"
    "## Your Capabilities:\n"
    "- **Composio Tools**: You have meta tools (COMPOSIO_SEARCH_TOOLS, COMPOSIO_MANAGE_CONNECTIONS, "
    "COMPOSIO_MULTI_EXECUTE_TOOL) that let you discover, authenticate, and execute 1000+ external "
    "tools (Gmail, GitHub, Slack, etc.)\n"
    "- **Knowledge Base (RAG_SEARCH)**: You have a RAG_SEARCH tool to search uploaded documents "
    "(PDFs, text files) in the knowledge base.\n\n"
    "## IMPORTANT — When to Use Tools vs. Answer Directly:\n"
    "Not every question requires a tool call. Follow this decision process:\n"
    "1. **Answer directly (NO tool call)** if the question is about general knowledge, greetings, "
    "chitchat, opinions, or anything you can confidently answer from your training data.\n"
    "2. **Call RAG_SEARCH** if the user's question is specifically about the content of an uploaded "
    "document (PDF, text file, etc.) or references information that would only exist in the "
    "knowledge base. Examples: \"What does the uploaded PDF say about X?\", \"Summarize the document\", "
    "\"According to the paper, what is Y?\"\n"
    "3. **Call Composio tools** only when the user explicitly needs to interact with an external "
    "service (send email, create calendar event, open GitHub issue, etc.).\n\n"
    "**If the user asks a question that you can answer from your own knowledge — even if documents "
    "have been uploaded — do NOT call any tool. Only invoke RAG_SEARCH when the answer genuinely "
    "depends on the uploaded document content.**\n\n"
    "## Knowledge Base (RAG) Workflow:\n"
    "- When the user asks a question that clearly relates to uploaded documents, call RAG_SEARCH "
    "with a relevant natural-language query derived from the user's question.\n"
    "- Use the returned document chunks to inform your answer. Cite the source filename when possible.\n"
    "- If RAG_SEARCH returns no relevant results, say so honestly and answer with your best general knowledge.\n\n"
    "## Composio Tool Workflow (follow this EXACT order every time you need an external tool):\n"
    "1. **Search for Tools**: Call COMPOSIO_SEARCH_TOOLS to find the right tool for the task.\n"
    "2. **Manage Connections (ALWAYS REQUIRED)**: Call COMPOSIO_MANAGE_CONNECTIONS with the toolkit "
    "name(s) to ensure the user has an active authenticated connection. Pass the relevant "
    "toolkit(s) in the `toolkits` array (e.g. [\"googlecalendar\"]). If the response contains a "
    "`redirect_url`, share it with the user as a clickable link and **wait** — do NOT proceed to "
    "execute until the connection is confirmed active.\n"
    "3. **Execute**: Only AFTER COMPOSIO_MANAGE_CONNECTIONS succeeds, call "
    "COMPOSIO_MULTI_EXECUTE_TOOL to run the actual tool action.\n"
    "4. **Synthesize**: Combine tool results into a comprehensive answer.\n\n"
    "## CRITICAL Rules:\n"
    "- **Do NOT call any tool if the question can be answered from general knowledge.** "
    "Tool calls have latency — only use them when truly needed.\n"
    "- **NEVER skip step 2 of the Composio workflow.** You MUST call COMPOSIO_MANAGE_CONNECTIONS "
    "before COMPOSIO_MULTI_EXECUTE_TOOL, even if you think the user is already connected.\n"
    "- If COMPOSIO_MANAGE_CONNECTIONS returns a redirect_url, present it to the user as a clickable "
    "authentication link and tell them to complete the setup before you proceed.\n"
    "- Always pass the `session_id` returned from previous meta tool calls into subsequent calls.\n"
    "- Explain what you're doing at each step.\n\n"
    "Think step-by-step. Use tools only when necessary."
)


class SuperAgentService(BaseService):
    """
    Agentic orchestrator that ties together LLM, conversation history,
    Composio Tool Router meta tools, and RAG context injection.
    """

    def __init__(
        self,
        llm_service: LLMService,
        conversation_service: ConversationService,
        composio_service: ComposioService,
        tool_executor: ToolExecutor,
    ) -> None:
        self._llm = llm_service
        self._conversations = conversation_service
        self._composio = composio_service
        self._tool_executor = tool_executor

    # ── Lifecycle ──

    async def initialize(self) -> None:
        await super().initialize()

    async def health_check(self) -> bool:
        return self.is_initialized

    # ── Helper methods ──

    def _inject_rag_context(
        self,
        history: list[dict[str, Any]],
        rag_text: str,
    ) -> list[dict[str, Any]]:
        """
        Inject RAG context into the conversation history.

        Adds a system-level context message right after the system prompt
        (or at the beginning if there is no system message).
        """
        if not rag_text:
            return history

        context_msg = {"role": "system", "content": rag_text}

        # Find insertion point: after the first system message (if any)
        for i, msg in enumerate(history):
            if msg.get("role") == "system":
                updated = history[: i + 1] + [context_msg] + history[i + 1 :]
                return updated

        # No system message found — prepend
        return [context_msg] + history

    # ── Auth-required detection ──

    @staticmethod
    def _extract_auth_info(
        tool_result: dict[str, Any],
        tool_args: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Inspect a ``COMPOSIO_MANAGE_CONNECTIONS`` result for a redirect URL.

        Returns a ``connection_required`` payload dict if the user still needs to
        authenticate, or ``None`` if the connection is already active.
        """
        logger.debug(f"Checking auth info from tool_result keys: {tool_result.keys()}")
        
        data = tool_result.get("data") or {}

        # data may itself be a JSON-encoded string
        if isinstance(data, str):
            try:
                data = json.loads(data)
                logger.debug(f"Parsed data from string: {data.keys() if isinstance(data, dict) else type(data)}")
            except (json.JSONDecodeError, TypeError):
                data = {}

        # The redirect_url can be at the top level or nested inside
        # a per-toolkit entry
        redirect_url: str | None = None
        toolkit: str = ""

        # Check data.redirect_url
        if isinstance(data, dict) and data.get("redirect_url"):
            redirect_url = str(data["redirect_url"])
            logger.debug(f"Found redirect_url in data: {redirect_url}")

        # Also check if the entire tool_result has redirect_url at top level
        if not redirect_url and tool_result.get("redirect_url"):
            redirect_url = str(tool_result["redirect_url"])
            logger.debug(f"Found redirect_url at tool_result top level: {redirect_url}")

        # Check in content field (sometimes Composio returns it here)
        if not redirect_url and isinstance(data, dict):
            content = data.get("content")
            if isinstance(content, dict) and content.get("redirect_url"):
                redirect_url = str(content["redirect_url"])
                logger.debug(f"Found redirect_url in data.content: {redirect_url}")

        # Some responses nest per-toolkit results in a list or dict
        if not redirect_url and isinstance(data, dict):
            for key in ("results", "toolkits", "connections", "connection_details"):
                raw = data.get(key)
                
                # Handle as list
                if isinstance(raw, list):
                    for entry in raw:
                        if isinstance(entry, dict) and entry.get("redirect_url"):
                            redirect_url = str(entry["redirect_url"])
                            toolkit = toolkit or str(entry.get("toolkit", ""))
                            logger.debug(f"Found redirect_url in {key}[list]: {redirect_url}")
                            break
                    if redirect_url:
                        break
                
                # Handle as dict (toolkit_name -> toolkit_info)
                elif isinstance(raw, dict):
                    for tk_name, tk_info in raw.items():
                        if isinstance(tk_info, dict) and tk_info.get("redirect_url"):
                            redirect_url = str(tk_info["redirect_url"])
                            toolkit = toolkit or str(tk_info.get("toolkit", tk_name))
                            logger.debug(f"Found redirect_url in {key}[dict][{tk_name}]: {redirect_url}")
                            break
                    if redirect_url:
                        break

        if not redirect_url:
            logger.debug(f"No redirect_url found. Tool result structure: {json.dumps(tool_result, indent=2)[:500]}")
            return None

        # Try to derive the toolkit name from the tool arguments
        if not toolkit:
            toolkits_arg = tool_args.get("toolkits") or []
            if isinstance(toolkits_arg, list) and toolkits_arg:
                toolkit = str(toolkits_arg[0])
            elif isinstance(toolkits_arg, str):
                toolkit = toolkits_arg

        logger.info(f"Found redirect_url for toolkit '{toolkit}': {redirect_url}")
        
        return {
            "toolkit": toolkit,
            "redirect_url": redirect_url,
            "message": (
                f"Please connect your {toolkit or 'account'} to continue. "
                "Click the button below to authenticate."
            ),
        }

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
        - ``{"type": "rag_context", "data": {"results_count": N}}``
        - ``{"type": "tool_call", "data": {"name": ..., "arguments": ...}}``
        - ``{"type": "tool_result", "data": {"name": ..., "result": ...}}``
        - ``{"type": "connection_required", "data": {"toolkit": ..., "redirect_url": ..., ...}}``
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

        # 3. Get Composio Tool Router meta tools
        available_tools: list[Any] = []
        session_id: str = ""
        if self._composio.is_initialized:
            try:
                loop = asyncio.get_running_loop()
                tools_and_sid = await loop.run_in_executor(
                    None,
                    lambda: self._composio.get_session_tools(user_id),
                )
                available_tools, session_id = tools_and_sid
                logger.info(
                    "Loaded %d Tool Router meta tools for user %s (session %s)",
                    len(available_tools),
                    user_id,
                    session_id,
                )
            except Exception as exc:
                logger.warning("Failed to get session tools: %s", exc)

        # 4. Append RAG_SEARCH tool if RAG is enabled
        if settings.RAG_ENABLED and self._tool_executor.is_initialized:
            rag_tool_def = ToolExecutor.get_rag_tool_definition()
            available_tools.append(rag_tool_def)
            logger.info("Added RAG_SEARCH tool to available tools")

        # 5. Build history
        history = self._conversations.get_formatted_history_for_model(cid)

        # 6. Initial LLM call
        try:
            if available_tools:
                logger.debug(
                    "Calling LLM with %d tools and %d messages",
                    len(available_tools),
                    len(history),
                )
                llm_response = await self._llm.chat_with_tools_raw(history, available_tools)
            else:
                llm_response = await self._llm.chat_raw(history)
        except Exception as exc:
            logger.exception("LLM call failed")
            yield {"type": "error", "data": {"message": f"LLM error: {exc}"}}
            return

        # 7. Agentic tool-call loop
        iterations = 0
        while llm_response.get("tool_calls") and iterations < MAX_TOOL_ITERATIONS:
            iterations += 1

            # Store the assistant message WITH tool_calls in the proper format
            # (required for Gemini to match tool responses with tool calls)
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": llm_response.get("content") or "",
            }
            if llm_response.get("tool_calls"):
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["arguments"])
                                if isinstance(tc["arguments"], dict)
                                else tc["arguments"]
                            ),
                        },
                    }
                    for tc in llm_response["tool_calls"]
                ]

            # Add to history manually to preserve tool_calls structure
            self._conversations._conversations[cid].history.append(assistant_message)

            auth_required = False
            for tc in llm_response["tool_calls"]:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                tool_call_id = tc.get("id", "")

                yield {
                    "type": "tool_call",
                    "data": {"name": tool_name, "arguments": tool_args},
                }

                # Dispatch through ToolExecutor
                tool_result = await self._tool_executor.execute_tool(
                    name=tool_name,
                    arguments=tool_args,
                    user_id=user_id,
                    session_id=session_id,
                )

                yield {
                    "type": "tool_result",
                    "data": {"name": tool_name, "result": tool_result},
                }

                # ── Auth-required interception ──
                # If COMPOSIO_MANAGE_CONNECTIONS returned a redirect_url the
                # user has NOT yet connected.  Emit a connection_required event so
                # the UI can render a "Connect" button and stop the loop.
                if tool_name == "COMPOSIO_MANAGE_CONNECTIONS":
                    logger.info(f"COMPOSIO_MANAGE_CONNECTIONS result: {json.dumps(tool_result, indent=2)}")
                    auth_info = self._extract_auth_info(tool_result, tool_args)
                    if auth_info is not None:
                        logger.info(f"Emitting connection_required event: {auth_info}")
                        yield {
                            "type": "connection_required",
                            "data": {
                                "conversation_id": cid,
                                **auth_info,
                            },
                        }
                        auth_required = True
                    else:
                        logger.info("No redirect_url found, connection appears to be established")

                # Add tool result to history (format Gemini/OpenAI expects)
                tool_response_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(tool_result),
                }
                self._conversations._conversations[cid].history.append(
                    tool_response_message
                )

            # If the user needs to authenticate, stop the loop.
            # Store an assistant message and yield a reply so the frontend
            # receives the message via WebSocket instead of a bare return.
            if auth_required:
                auth_pause_msg = (
                    "I've detected that authentication is required. "
                    "I've shared the authentication link with you. "
                    "Once you complete the authentication, I will continue "
                    "with your original request."
                )
                self._conversations.add_message(
                    cid, ConversationRole.ASSISTANT.value, auth_pause_msg
                )
                yield {
                    "type": "reply",
                    "data": {
                        "conversation_id": cid,
                        "content": auth_pause_msg,
                    },
                }
                return

            # Re-call LLM with updated history
            history = self._conversations.get_formatted_history_for_model(cid)

            try:
                logger.debug(
                    "Re-calling LLM with %d messages and %d tools",
                    len(history),
                    len(available_tools),
                )
                if available_tools:
                    llm_response = await self._llm.chat_with_tools_raw(
                        history, available_tools
                    )
                else:
                    llm_response = await self._llm.chat_raw(history)
            except Exception as exc:
                logger.exception("LLM follow-up call failed")
                logger.error(
                    "History at time of failure: %s",
                    json.dumps(history[-5:], indent=2),
                )
                yield {"type": "error", "data": {"message": f"LLM error: {exc}"}}
                return

        # 8. Store the final assistant reply
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

    async def continue_after_auth(
        self,
        user_id: str,
        conversation_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Resume the agentic loop after the user completes OAuth.

        Injects a user message confirming auth completion, then re-runs
        the full agentic loop on the existing conversation so the LLM
        picks up where it left off.

        Yields the same event types as ``handle_message``.
        """
        self._ensure_initialized()

        continuation_msg = (
            "I have completed the authentication successfully. "
            "Please continue with my original request."
        )

        async for event in self.handle_message(
            user_id=user_id,
            message=continuation_msg,
            conversation_id=conversation_id,
        ):
            yield event

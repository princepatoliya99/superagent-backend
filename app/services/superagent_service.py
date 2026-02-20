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
MAX_TOOL_ITERATIONS = 10

# Maximum number of tools to provide to LLM at once (to avoid context overflow)
MAX_TOOLS_PER_REQUEST = 12

SYSTEM_PROMPT = (
    "You are SuperAgent, an intelligent AI assistant with access to real-world tools via Composio.\n\n"
    "## Your Workflow:\n"
    "1. **Analyze the Problem**: Break down the user's request into clear sub-problems or tasks\n"
    "2. **Search for Tools**: For each sub-problem, use the COMPOSIO_SEARCH tool to find relevant tools\n"
    "3. **Execute Tools**: Use the tools found by COMPOSIO_SEARCH to solve each sub-problem\n"
    "4. **Synthesize Results**: Combine the results and provide a comprehensive answer\n\n"
    "## Important Guidelines:\n"
    "- Always use COMPOSIO_SEARCH first to find the right tools for the task\n"
    "- Provide clear descriptions in your COMPOSIO_SEARCH query (e.g., 'fetch emails from Gmail', 'create GitHub issue')\n"
    "- Only use tools that were returned by COMPOSIO_SEARCH\n"
    "- Explain what you're doing before and after each step\n"
    "- If COMPOSIO_SEARCH returns no tools, inform the user that the capability isn't available\n\n"
    "Think step-by-step and use the tools systematically to help the user."
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

    # ── Helper methods ──

    def _create_composio_search_tool(self) -> dict[str, Any]:
        """
        Create the COMPOSIO_SEARCH tool definition for the LLM.
        
        This tool allows the LLM to dynamically search for and discover
        tools based on the task requirements.
        """
        return {
            "type": "function",
            "function": {
                "name": "COMPOSIO_SEARCH",
                "description": (
                    "Search for relevant tools in Composio's toolkit catalog. "
                    "Use this tool to find actions you can perform for specific services "
                    "(like Gmail, GitHub, Slack, etc.). Returns a list of available tools "
                    "that match your query. Always use this FIRST before attempting to use any other tools."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural language description of what you want to do. "
                                "Examples: 'fetch recent emails', 'create a GitHub issue', "
                                "'send a Slack message', 'list calendar events'"
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def _extract_toolkit_from_tool(self, tool: dict[str, Any]) -> Optional[str]:
        """
        Extract toolkit name from a tool definition.
        
        Composio tool names follow the pattern: TOOLKIT_ACTION_NAME
        e.g., GITHUB_CREATE_ISSUE, GMAIL_FETCH_EMAILS
        """
        if "function" in tool and "name" in tool["function"]:
            tool_name = tool["function"]["name"]
            # Extract toolkit prefix (everything before first underscore after toolkit name)
            parts = tool_name.split("_")
            if len(parts) >= 2:
                return parts[0].upper()
        return None

    def _extract_toolkit_from_tool_name(self, tool_name: str) -> Optional[str]:
        """
        Extract toolkit name from a tool name string.
        
        Composio tool names follow the pattern: TOOLKIT_ACTION_NAME
        e.g., GITHUB_CREATE_ISSUE -> GITHUB
        """
        parts = tool_name.split("_")
        if len(parts) >= 2:
            return parts[0].upper()
        return None

    def _prioritize_and_limit_tools(
        self,
        tools: list[dict[str, Any]],
        user_message: str,
        max_tools: int = MAX_TOOLS_PER_REQUEST
    ) -> list[dict[str, Any]]:
        """
        Prioritize and limit tools to avoid context overflow.
        
        Prioritization criteria:
        1. Tools from toolkits explicitly mentioned in user message (e.g., "gmail", "github")
        2. General tools
        
        Args:
            tools: List of tool definitions
            user_message: User's message (to detect explicit toolkit mentions)
            max_tools: Maximum number of tools to return
            
        Returns:
            Prioritized and limited list of tools
        """
        if len(tools) <= max_tools:
            return tools
        
        # Extract toolkit mentions from user message
        message_lower = user_message.lower()
        mentioned_toolkits = set()
        
        # Common toolkit keywords users might mention
        toolkit_keywords = {
            'gmail': 'GMAIL',
            'google': 'GMAIL',
            'github': 'GITHUB',
            'git': 'GITHUB',
            'slack': 'SLACK',
            'email': None,  # Generic, don't prioritize specific toolkit
            'calendar': 'GOOGLECALENDAR',
            'drive': 'GOOGLEDRIVE',
            'sheets': 'GOOGLESHEETS',
            'jira': 'JIRA',
            'trello': 'TRELLO',
        }
        
        for keyword, toolkit in toolkit_keywords.items():
            if keyword in message_lower and toolkit:
                mentioned_toolkits.add(toolkit)
        
        # Separate tools into prioritized and others
        prioritized = []
        others = []
        
        for tool in tools:
            toolkit = self._extract_toolkit_from_tool(tool)
            if toolkit and toolkit in mentioned_toolkits:
                prioritized.append(tool)
            else:
                others.append(tool)
        
        # Combine: prioritized tools first, then others up to max_tools
        result = prioritized[:max_tools]
        remaining_slots = max_tools - len(result)
        if remaining_slots > 0:
            result.extend(others[:remaining_slots])
        
        logger.info(
            f"Prioritized {len(prioritized)} tools (mentioned toolkits: {mentioned_toolkits}), "
            f"included {len(result)} out of {len(tools)} total tools"
        )
        
        return result

    async def _check_connection_status(
        self, user_id: str, toolkit: str
    ) -> dict[str, Any]:
        """
        Check if user has an active connection for a toolkit.
        
        Returns:
            {"connected": bool, "requires_auth": bool, "status": str}
        """
        try:
            # Call the async method directly
            connected_toolkits = await self._composio.get_connected_toolkits(user_id)
            
            # Check if toolkit is in connected list
            toolkit_lower = toolkit.lower()
            is_connected = any(
                self._composio._to_toolkit_slug(tk).lower() == toolkit_lower
                for tk in connected_toolkits
            )
            
            return {
                "connected": is_connected,
                "requires_auth": not is_connected,
                "status": "connected" if is_connected else "not_connected",
            }
        except Exception as exc:
            logger.warning(f"Failed to check connection status for {toolkit}: {exc}")
            return {"connected": False, "requires_auth": True, "status": "unknown"}

    async def _handle_toolkit_authorization(
        self, user_id: str, toolkit: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Handle toolkit authorization flow.
        
        Yields connection events including auth URL and waits for user to connect.
        """
        try:
            # Request authorization
            loop = asyncio.get_running_loop()
            auth_response = await loop.run_in_executor(
                None, lambda: self._composio.authorize_toolkit(user_id, toolkit)
            )
            
            # Check if auth is required
            if auth_response.status == self._composio.STATUS_NO_AUTH_REQUIRED:
                yield {
                    "type": "connection_established",
                    "data": {
                        "toolkit": toolkit,
                        "message": f"{toolkit} does not require authentication",
                    },
                }
                return
            
            # Yield auth URL for user to authenticate
            yield {
                "type": "connection_required",
                "data": {
                    "toolkit": toolkit,
                    "redirect_url": auth_response.redirect_url,
                    "connection_request_id": auth_response.connection_request_id,
                    "message": f"Please authenticate with {toolkit} using the provided link",
                },
            }
            
            # Wait for connection with timeout
            yield {
                "type": "connection_waiting",
                "data": {
                    "toolkit": toolkit,
                    "message": f"Waiting for {toolkit} authentication...",
                },
            }
            
            wait_response = await self._composio.wait_for_connection(
                connection_request_id=auth_response.connection_request_id or "",
                user_id=user_id,
                toolkit=toolkit,
            )
            
            if wait_response.status == self._composio.STATUS_CONNECTED:
                yield {
                    "type": "connection_established",
                    "data": {
                        "toolkit": toolkit,
                        "message": wait_response.message or f"Successfully connected to {toolkit}",
                    },
                }
            else:
                yield {
                    "type": "error",
                    "data": {
                        "message": f"Failed to connect to {toolkit}: {wait_response.status}"
                    },
                }
                
        except Exception as exc:
            logger.exception(f"Authorization failed for {toolkit}")
            yield {
                "type": "error",
                "data": {"message": f"Authorization error for {toolkit}: {exc}"},
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
        - ``{"type": "tool_search", "data": {"query": ..., "tools_found": N, "toolkits": [...]}}``
        - ``{"type": "connection_status", "data": {"toolkit": ..., "connected": bool, "requires_auth": bool}}``
        - ``{"type": "connection_required", "data": {"toolkit": ..., "redirect_url": ..., "message": ...}}``
        - ``{"type": "connection_waiting", "data": {"toolkit": ..., "message": ...}}``
        - ``{"type": "connection_established", "data": {"toolkit": ..., "message": ...}}``
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

        # 3. Start with COMPOSIO_SEARCH as the only available tool
        # The LLM will use this to discover other tools dynamically
        available_tools: list[Any] = []
        discovered_tools: dict[str, list[Any]] = {}  # Maps search query to tools
        
        if self._composio.is_initialized:
            # Add COMPOSIO_SEARCH as the initial tool
            available_tools = [self._create_composio_search_tool()]

        # 4. Send to LLM (with or without tools)
        history = self._conversations.get_formatted_history_for_model(cid)

        try:
            if available_tools:
                logger.debug(f"Calling LLM with {len(available_tools)} tools and {len(history)} messages")
                llm_response = await self._llm.chat_with_tools_raw(history, available_tools)
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

            # Store the assistant message WITH tool_calls in the proper format
            # This is required for Gemini to match tool responses with tool calls
            assistant_message = {
                "role": "assistant",
                "content": llm_response.get("content") or "",
            }
            
            # Add tool_calls if present (required by Gemini)
            if llm_response.get("tool_calls"):
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"],
                        }
                    }
                    for tc in llm_response["tool_calls"]
                ]
            
            # Add to history manually to preserve tool_calls structure
            self._conversations._conversations[cid].history.append(assistant_message)

            for tc in llm_response["tool_calls"]:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                tool_call_id = tc.get("id", "")

                yield {
                    "type": "tool_call",
                    "data": {"name": tool_name, "arguments": tool_args},
                }

                # Handle COMPOSIO_SEARCH specially
                if tool_name == "COMPOSIO_SEARCH":
                    search_query = tool_args.get("query", "")
                    
                    if not search_query:
                        tool_result = {
                            "data": {"tools": [], "message": "No search query provided"},
                            "error": "Missing query parameter",
                            "successful": False,
                        }
                    else:
                        try:
                            # Search for tools
                            loop = asyncio.get_running_loop()
                            found_tools = await loop.run_in_executor(
                                None,
                                lambda: self._composio.search_tools(user_id, search_query),
                            )
                            
                            # Store discovered tools
                            discovered_tools[search_query] = found_tools
                            
                            # Extract toolkits from found tools (for informational purposes)
                            toolkits_in_results = set()
                            for tool in found_tools:
                                toolkit = self._extract_toolkit_from_tool(tool)
                                if toolkit:
                                    toolkits_in_results.add(toolkit)
                            
                            # DON'T check connections for all toolkits yet
                            # Only check when LLM actually tries to use a specific tool
                            # This prevents unnecessary authorization prompts for tools the LLM won't use
                            
                            # Prioritize and limit tools to avoid context overflow
                            # This filters tools based on user's explicit mentions (e.g., "gmail")
                            # and limits total number to prevent hitting token limits
                            prioritized_tools = self._prioritize_and_limit_tools(
                                found_tools,
                                message,  # Use original user message for keyword detection
                                MAX_TOOLS_PER_REQUEST
                            )
                            
                            # Add prioritized tools to available tools
                            for tool in prioritized_tools:
                                if tool not in available_tools:
                                    available_tools.append(tool)
                            
                            # Format tool names for response
                            tool_names = [
                                t.get("function", {}).get("name", "unknown")
                                for t in prioritized_tools
                            ]
                            
                            tool_result = {
                                "data": {
                                    "tools_found": len(found_tools),
                                    "tools_available": len(prioritized_tools),
                                    "tool_names": tool_names,
                                    "toolkits_available": list(toolkits_in_results),
                                    "message": (
                                        f"Found {len(found_tools)} tools for '{search_query}'. "
                                        f"Providing {len(prioritized_tools)} most relevant tools. "
                                        f"Available toolkits: {', '.join(toolkits_in_results)}"
                                    ),
                                },
                                "successful": True,
                            }
                            
                            yield {
                                "type": "tool_search",
                                "data": {
                                    "query": search_query,
                                    "tools_found": len(found_tools),
                                    "toolkits": list(toolkits_in_results),
                                },
                            }
                            
                        except Exception as exc:
                            logger.exception(f"COMPOSIO_SEARCH failed for '{search_query}'")
                            tool_result = {
                                "data": {"tools": [], "message": str(exc)},
                                "error": str(exc),
                                "successful": False,
                            }
                else:
                    # Execute regular Composio tool
                    # First check if toolkit is connected
                    toolkit = self._extract_toolkit_from_tool_name(tool_name)
                    
                    if toolkit:
                        conn_status = await self._check_connection_status(user_id, toolkit)
                        
                        yield {
                            "type": "connection_status",
                            "data": {
                                "toolkit": toolkit,
                                "connected": conn_status["connected"],
                                "requires_auth": conn_status["requires_auth"],
                            },
                        }
                        
                        # If not connected, initiate authorization
                        if not conn_status["connected"]:
                            auth_successful = False
                            async for auth_event in self._handle_toolkit_authorization(
                                user_id, toolkit
                            ):
                                yield auth_event
                                
                                if auth_event["type"] == "connection_established":
                                    auth_successful = True
                                elif auth_event["type"] == "error":
                                    break
                            
                            if not auth_successful:
                                tool_result = {
                                    "data": {},
                                    "error": f"Authorization failed for {toolkit}",
                                    "successful": False
                                }
                                yield {
                                    "type": "tool_result",
                                    "data": {"name": tool_name, "result": tool_result},
                                }
                                
                                # Add error to history
                                tool_response_message = {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "content": json.dumps(tool_result),
                                }
                                self._conversations._conversations[cid].history.append(tool_response_message)
                                continue
                    
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

                # Add tool result to history in the format Gemini/OpenAI expects
                # Tool responses must have role="tool", tool_call_id, and content
                # Some models also expect "name" field
                tool_response_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(tool_result),
                }
                
                # Add to history manually to preserve proper structure
                self._conversations._conversations[cid].history.append(tool_response_message)

            # Re-call LLM with updated history and available tools
            history = self._conversations.get_formatted_history_for_model(cid)
            try:
                logger.debug(f"Re-calling LLM with {len(history)} messages and {len(available_tools)} tools")
                if available_tools:
                    llm_response = await self._llm.chat_with_tools_raw(history, available_tools)
                else:
                    llm_response = await self._llm.chat_raw(history)
            except Exception as exc:
                logger.exception("LLM follow-up call failed")
                logger.error(f"History at time of failure: {json.dumps(history[-5:], indent=2)}")  # Log last 5 messages
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

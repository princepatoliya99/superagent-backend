"""
Chat API routes (REST + WebSocket).

REST endpoints for sending messages, retrieving history,
listing conversations, and deleting conversations.

WebSocket endpoint ``/chat/ws`` for the SuperAgent agentic loop
with real-time streaming of intermediate steps.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.models.chat import ChatRequest, ChatResponse, ConversationSummary
from app.models.conversation import ConversationRole
from app.services.conversation_service import ConversationService
from app.services.llm_service import LLMService
from app.services.superagent_service import SuperAgentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

# Service instances are injected by main.py at startup
llm_service: LLMService | None = None
conversation_service: ConversationService | None = None
superagent_service: SuperAgentService | None = None


def configure(
    llm: LLMService,
    conversations: ConversationService,
    superagent: SuperAgentService,
) -> None:
    """Inject service instances (called from main.py at startup)."""
    global llm_service, conversation_service, superagent_service
    llm_service = llm
    conversation_service = conversations
    superagent_service = superagent


# ═══════════════════════════════════════════════════════════════════════
# WebSocket — SuperAgent agentic chat
# ═══════════════════════════════════════════════════════════════════════


@router.websocket("/ws")
async def ws_chat(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for the SuperAgent.

    **Client sends** JSON messages::

        {
            "message": "What is my last email in Gmail?",
            "user_id": "user_123",
            "conversation_id": "optional_existing_id"
        }

    **Server streams** JSON events::

        {"type": "tool_search",  "data": {"query": "...", "tools_found": 3}}
        {"type": "tool_call",    "data": {"name": "GMAIL_...", "arguments": {...}}}
        {"type": "tool_result",  "data": {"name": "GMAIL_...", "result": {...}}}
        {"type": "reply",        "data": {"conversation_id": "...", "content": "..."}}
        {"type": "error",        "data": {"message": "..."}}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Receive client message
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "data": {"message": "Invalid JSON"}}
                )
                continue

            message = payload.get("message", "").strip()
            user_id = payload.get("user_id", "default")
            conversation_id = payload.get("conversation_id")

            if not message:
                await websocket.send_json(
                    {"type": "error", "data": {"message": "Empty message"}}
                )
                continue

            if not superagent_service:
                await websocket.send_json(
                    {"type": "error", "data": {"message": "SuperAgent not initialized"}}
                )
                continue

            # Stream agentic loop events to the client
            async for event in superagent_service.handle_message(
                user_id=user_id,
                message=message,
                conversation_id=conversation_id,
            ):
                await websocket.send_json(event)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json(
                {"type": "error", "data": {"message": str(exc)}}
            )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# REST Endpoints
# ═══════════════════════════════════════════════════════════════════════


@router.post("", response_model=ChatResponse)
async def send_message(request: ChatRequest) -> ChatResponse:
    """Send a user message and receive an LLM reply."""
    assert llm_service and conversation_service

    cid = conversation_service.create_conversation(request.conversation_id)
    conversation_service.add_message(cid, ConversationRole.USER.value, request.message)
    history = conversation_service.get_formatted_history_for_model(cid)

    try:
        assistant_msg = await llm_service.chat_raw(history)
    except Exception as exc:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    conversation_service.add_message(cid, ConversationRole.ASSISTANT.value, assistant_msg["content"])
    updated_history = conversation_service.get_history(cid)

    return ChatResponse(
        conversation_id=cid,
        reply=assistant_msg["content"],
        history=updated_history,
    )


@router.get("/{conversation_id}/history")
async def get_history(conversation_id: str) -> List[Dict[str, Any]]:
    """Retrieve the full message history for a conversation."""
    assert conversation_service
    try:
        return conversation_service.get_history(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found.")


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, str]:
    """Delete a conversation by ID."""
    assert conversation_service
    if not conversation_service.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted", "conversation_id": conversation_id}


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations() -> list[ConversationSummary]:
    """List all conversations with lightweight summaries."""
    assert conversation_service
    return conversation_service.list_conversations()

"""
Pydantic models for chat requests, responses, and messages.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ──


class Role(str, Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ── Core Models ──


class ToolCall(BaseModel):
    """Represents a single tool invocation requested by the LLM."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str
    tool_calls: Optional[list[ToolCall]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── API Contracts ──


class ChatRequest(BaseModel):
    """Incoming chat request from the client."""

    conversation_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    """Response returned to the client after a chat turn."""

    conversation_id: str
    reply: str
    history: list[Message]


class ConversationSummary(BaseModel):
    """Lightweight summary of a conversation (for listing)."""

    conversation_id: str
    message_count: int
    created_at: datetime
    last_message_at: datetime

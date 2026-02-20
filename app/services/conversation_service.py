"""
Conversation service — manages multiple conversations,
each backed by a ``ConversationHistory`` instance.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models.chat import ConversationSummary
from app.models.conversation import ConversationHistory, ConversationRole
from app.services.base.base_service import BaseService
from app.telemetries.logger import logger

EVENT = "conversation_service"


class ConversationService(BaseService):
    """
    Manages a collection of ``ConversationHistory`` objects keyed by
    conversation ID.

    Each conversation is independently configurable with its own
    system message and max history length.
    """

    def __init__(self, default_max_history: int = 50) -> None:
        self._conversations: dict[str, ConversationHistory] = {}
        self._created_at: dict[str, datetime] = {}
        self._default_max_history = default_max_history

    # ── Lifecycle ──

    async def health_check(self) -> bool:
        return self.is_initialized

    # ── Conversation CRUD ──

    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        system_message: str = "",
        max_history_length: Optional[int] = None,
        should_init_system_message: bool = True,
    ) -> str:
        """Create (or reuse) a conversation and return its ID."""
        self._ensure_initialized()

        cid = conversation_id or uuid.uuid4().hex
        if cid in self._conversations:
            logger.debug(EVENT, message=f"Conversation {cid} already exists, reusing.")
            return cid

        self._conversations[cid] = ConversationHistory(
            max_history_length=max_history_length or self._default_max_history,
            system_message=system_message,
            should_init_system_message=should_init_system_message,
        )
        self._created_at[cid] = datetime.utcnow()
        logger.info(EVENT, message=f"Created conversation {cid}")
        return cid

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns ``True`` if it existed."""
        self._ensure_initialized()

        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            del self._created_at[conversation_id]
            logger.info(EVENT, message=f"Deleted conversation {conversation_id}")
            return True
        return False

    def list_conversations(self) -> list[ConversationSummary]:
        """Return lightweight summaries of all conversations."""
        self._ensure_initialized()

        summaries: list[ConversationSummary] = []
        for cid, conv in self._conversations.items():
            history = conv.get_history()
            msg_count = len(history)
            created = self._created_at[cid]
            summaries.append(
                ConversationSummary(
                    conversation_id=cid,
                    message_count=msg_count,
                    created_at=created,
                    last_message_at=created,  # in-memory, no per-message ts
                )
            )
        return summaries

    # ── Message operations (delegates to ConversationHistory) ──

    def add_message(self, conversation_id: str, role: str, content: Any) -> None:
        """Append a message to a conversation."""
        self._ensure_initialized()
        conv = self._get_conv(conversation_id)
        conv.add_conversation_message_to_history(role, content)

    def extend_history(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Bulk-append messages to a conversation."""
        self._ensure_initialized()
        conv = self._get_conv(conversation_id)
        conv.extend_history(messages)

    def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Return the full message history for a conversation."""
        self._ensure_initialized()
        conv = self._get_conv(conversation_id)
        return conv.get_history()

    def get_last_n_messages(self, conversation_id: str, n: int) -> List[Dict[str, Any]]:
        """Return the last *n* messages."""
        self._ensure_initialized()
        conv = self._get_conv(conversation_id)
        return conv.get_last_n_messages(n)

    def get_formatted_history_for_model(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Return a model-ready copy of the history."""
        self._ensure_initialized()
        conv = self._get_conv(conversation_id)
        return conv.get_formatted_history_for_model()

    def replace_last_message(self, conversation_id: str, role: str, content: Any) -> None:
        """Replace the most recent message in a conversation."""
        self._ensure_initialized()
        conv = self._get_conv(conversation_id)
        conv.replace_last_message(role, content)

    # ── System message ──

    def set_system_message(self, conversation_id: str, system_message: str) -> None:
        """Update the system message for a conversation."""
        self._ensure_initialized()
        conv = self._get_conv(conversation_id)
        conv.set_system_message(system_message)

    # ── Clear ──

    def clear_history(self, conversation_id: str) -> None:
        """Clear all messages in a conversation (including system)."""
        self._ensure_initialized()
        conv = self._get_conv(conversation_id)
        conv.clear_history()

    def clear_history_without_system_message(self, conversation_id: str) -> None:
        """Clear messages but preserve the system message."""
        self._ensure_initialized()
        conv = self._get_conv(conversation_id)
        conv.clear_history_without_system_message()

    # ── Direct access ──

    def get_conversation_history_object(self, conversation_id: str) -> ConversationHistory:
        """Return the raw ``ConversationHistory`` instance."""
        self._ensure_initialized()
        return self._get_conv(conversation_id)

    # ── Internal ──

    def _get_conv(self, conversation_id: str) -> ConversationHistory:
        if conversation_id not in self._conversations:
            raise KeyError(f"Conversation {conversation_id} not found.")
        return self._conversations[conversation_id]

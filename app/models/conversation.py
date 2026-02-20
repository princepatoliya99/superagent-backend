"""
ConversationHistory — per-conversation message store with system-message
preservation, truncation, and model-ready formatting.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List


from app.telemetries.logger import logger


class ConversationRole(Enum):
    """Roles that can participate in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationHistory:
    """
    Manages the message history for a single conversation.

    Features:
    - Automatic truncation to ``max_history_length`` while preserving the
      system message at index 0.
    - Helper methods for clearing, replacing, extending, and formatting.
    """

    def __init__(
        self,
        max_history_length: int = 10,
        system_message: str = "",
        should_init_system_message: bool = True,
    ) -> None:
        self.history: List[Dict[str, Any]] = []
        self.system_message = system_message
        if should_init_system_message:
            self.history.append({"role": "system", "content": self.system_message})
        self.max_history_length = max_history_length
        self.event_name = "conversation_history"
        logger.debug(self.event_name, message="Initialized conversation history manager")

    # ── Add / Extend ──

    def add_conversation_message_to_history(self, role: str, content: Any) -> None:
        """Append a message and truncate if necessary (system message preserved)."""
        message = {"role": role, "content": content}
        self.history.append(message)

        if len(self.history) > self.max_history_length:
            # Preserve system message at index 0
            self.history = [self.history[0]] + self.history[-(self.max_history_length - 1):]

        logger.debug(
            self.event_name,
            message=f"Added {role} message to history. Current length: {len(self.history)}",
        )

    def extend_history(self, messages: List[Dict[str, Any]]) -> None:
        """Bulk-append messages to history."""
        self.history.extend(messages)
        logger.debug(
            self.event_name,
            message=f"Extended history with {len(messages)} messages. Current length: {len(self.history)}",
        )

    # ── System message ──

    def set_system_message(self, system_message: str) -> None:
        """Update the system message (in-place at index 0)."""
        self.system_message = system_message
        if (
            self.history
            and len(self.history) > 0
            and self.history[0].get("role") == ConversationRole.SYSTEM.value
        ):
            self.history[0]["content"] = system_message

    # ── Retrieval ──

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the full history list."""
        return self.history

    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """Return the last *n* messages (or all if fewer exist)."""
        return self.history[-n:] if n <= len(self.history) else self.history.copy()

    def get_formatted_history_for_model(self) -> List[Dict[str, Any]]:
        """Return a copy of the history suitable for LLM input."""
        return self.history.copy()

    # ── Mutation ──

    def replace_last_message(self, role: str, content: Any) -> None:
        """Replace the most recent message (or add if history is empty)."""
        if not self.history:
            self.add_conversation_message_to_history(role, content)
            return

        self.history[-1] = {"role": role, "content": content}
        logger.debug(self.event_name, message="Replaced last message in history")

    # ── Clear ──

    def clear_history_without_system_message(self) -> None:
        """Remove all messages except the system message at index 0."""
        if (
            self.history
            and len(self.history) > 1
            and self.history[0].get("role") == ConversationRole.SYSTEM.value
        ):
            self.history[:] = self.history[:1]
        logger.debug(self.event_name, message="Cleared conversation history without system message")

    def clear_history(self) -> None:
        """Remove *all* messages including the system message."""
        self.history = []
        logger.debug(self.event_name, message="Cleared conversation history")

"""
Abstract base class for tool-execution services (e.g. Composio).

Any tool provider must inherit from this class and implement
``execute_tool`` and ``list_tools``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.services.base.base_service import BaseService


class BaseToolService(BaseService):
    """Abstract base for tool-execution backends."""

    @abstractmethod
    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool by name with the given arguments.

        Args:
            name: Tool identifier.
            arguments: Key-value arguments for the tool.

        Returns:
            The tool's output (type depends on the tool).
        """
        ...

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """
        Return a list of available tool definitions.

        Each dict should follow the OpenAI function-calling schema:
        ``{"type": "function", "function": {"name": …, "description": …, "parameters": …}}``.
        """
        ...

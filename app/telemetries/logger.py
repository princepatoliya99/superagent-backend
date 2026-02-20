"""
Structured logger for the application.

Wraps Python's stdlib logging with a convenience interface that
accepts an ``event_name`` as the first argument followed by keyword
pairs for structured context.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class StructuredLogger:
    """Logger that emits structured key-value messages."""

    def __init__(self, name: str = "superagent") -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s")
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG)

    # ── Public API ──

    def _format(self, event_name: str, **kwargs: Any) -> str:
        if kwargs:
            extras = " ".join(f"{k}={json.dumps(v) if isinstance(v, (dict, list)) else v}" for k, v in kwargs.items())
            return f"[{event_name}] {extras}"
        return f"[{event_name}]"

    def debug(self, event_name: str, **kwargs: Any) -> None:
        self._logger.debug(self._format(event_name, **kwargs))

    def info(self, event_name: str, **kwargs: Any) -> None:
        self._logger.info(self._format(event_name, **kwargs))

    def warning(self, event_name: str, **kwargs: Any) -> None:
        self._logger.warning(self._format(event_name, **kwargs))

    def error(self, event_name: str, **kwargs: Any) -> None:
        self._logger.error(self._format(event_name, **kwargs))

    def exception(self, event_name: str, **kwargs: Any) -> None:
        self._logger.exception(self._format(event_name, **kwargs))


# Singleton instance
logger = StructuredLogger()

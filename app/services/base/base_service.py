"""
Base class for all services in the application.

Every service must inherit from this class and implement
the lifecycle hooks: initialize, shutdown, and health_check.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Abstract base for every application service."""

    _initialized: bool = False

    # ── Lifecycle ──

    async def initialize(self) -> None:
        """
        Called once at application startup.

        Override this to set up connections, load resources, etc.
        The default implementation simply marks the service as initialised.
        """
        logger.info("%s initialising…", self.__class__.__name__)
        self._initialized = True
        logger.info("%s ready.", self.__class__.__name__)

    async def shutdown(self) -> None:
        """
        Called once at application shutdown.

        Override this to release connections, flush buffers, etc.
        """
        logger.info("%s shutting down…", self.__class__.__name__)
        self._initialized = False

    @abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` if the service is healthy and operational."""
        ...

    # ── Helpers ──

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _ensure_initialized(self) -> None:
        """Guard that raises if the service hasn't been initialised yet."""
        if not self._initialized:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been initialised. "
                "Call `await service.initialize()` first."
            )

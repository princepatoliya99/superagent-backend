"""
Composio Tool Router service.

Provides session management, toolkit authorization, MCP URL retrieval,
and user auth-config CRUD — all via Composio's Tool Router and SDK.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import aiohttp
from composio import Composio

from app.config.settings import settings
from app.models.composio import (
    AuthorizeToolkitResponse,
    DeleteAuthConfigResponse,
    DeletedConfigItem,
    FailedDeletionItem,
    UserAuthConfigItem,
    UserAuthConfigsResponse,
    WaitForConnectionResponse,
)
from app.services.base.base_tool_service import BaseToolService

logger = logging.getLogger(__name__)


class ComposioService(BaseToolService):
    """
    Full Composio Tool Router implementation.

    Tool Router provides a unified MCP endpoint for all toolkits with
    built-in search, authentication, and execution capabilities.
    """

    # Status constants
    STATUS_NO_AUTH_REQUIRED = "no_auth_required"
    STATUS_PENDING = "pending"
    STATUS_CONNECTED = "connected"
    STATUS_ACTIVE = "active"
    STATUS_COMPLETED = "completed"
    STATUS_NOT_FOUND = "not_found"
    STATUS_PARTIAL_SUCCESS = "partial_success"

    def __init__(self) -> None:
        self._api_key: str = settings.COMPOSIO_API_KEY
        self._org_key: str = settings.COMPOSIO_ORG_KEY
        self._base_url: str = settings.COMPOSIO_BASE_URL
        self._composio: Composio | None = None

    # =====================================================================
    # Lifecycle (BaseService)
    # =====================================================================

    async def initialize(self) -> None:
        if not self._api_key:
            logger.warning(
                "ComposioService: COMPOSIO_API_KEY is empty — "
                "service will remain uninitialised."
            )
            return

        self._composio = Composio(api_key=self._api_key)
        await super().initialize()
        logger.info("ComposioService initialised with Tool Router.")

    async def health_check(self) -> bool:
        return self.is_initialized and self._composio is not None

    # =====================================================================
    # BaseToolService interface
    # =====================================================================

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool (delegates to execute_tool_for_user with no user_id)."""
        self._ensure_initialized()
        return self.execute_tool_for_user(
            slug=name, arguments=arguments, user_id="default"
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        """List tools is not supported directly — use search_tools instead."""
        self._ensure_initialized()
        raise NotImplementedError(
            "Use search_tools(user_id, query) for dynamic tool discovery."
        )

    # =====================================================================
    # Tool Search & Execution (used by SuperAgent)
    # =====================================================================

    def search_tools(
        self,
        user_id: str,
        query: str,
    ) -> list[Any]:
        """
        Search for tools relevant to the user's query.

        Uses Composio's ``tools.get(user_id, search=query)`` which returns
        OpenAI-compatible tool definitions. These can be passed directly to
        an LLM that supports function calling.

        Args:
            user_id: User identifier (for auth scoping).
            query: Natural-language description of what the user wants to do.

        Returns:
            List of tool definitions in OpenAI function-calling format.
        """
        self._ensure_initialized()
        assert self._composio is not None

        try:
            tools = self._composio.tools.get(user_id=user_id, search=query)
            logger.info(
                "search_tools found %d tools for user %s query=%r",
                len(tools) if tools else 0,
                user_id,
                query[:80],
            )
            return tools if tools else []
        except Exception as exc:
            logger.error("search_tools failed for user %s: %s", user_id, exc)
            return []

    def execute_tool_for_user(
        self,
        slug: str,
        arguments: dict[str, Any],
        user_id: str,
    ) -> dict[str, Any]:
        """
        Execute a specific tool for a user.

        Uses Composio's ``tools.execute(slug, arguments, user_id=user_id)``.

        Args:
            slug: Tool slug (e.g. ``GMAIL_FETCH_EMAILS``).
            arguments: Tool arguments as a dict.
            user_id: User identifier.

        Returns:
            Tool execution response dict with ``data``, ``error``, ``successful``.
        """
        self._ensure_initialized()
        assert self._composio is not None

        try:
            result = self._composio.tools.execute(
                slug=slug,
                arguments=arguments,
                user_id=user_id,
            )
            logger.info("Executed tool %s for user %s", slug, user_id)
            return dict(result)
        except Exception as exc:
            logger.error("execute_tool_for_user failed (%s): %s", slug, exc)
            return {"data": {}, "error": str(exc), "successful": False}

    # =====================================================================
    # Session management
    # =====================================================================

    def create_session(
        self,
        user_id: str,
        toolkits: Optional[list[str]] = None,
        manage_connections: bool = True,
    ):
        """Create a Tool Router session for a user."""
        self._ensure_initialized()
        assert self._composio is not None

        try:
            session_kwargs: dict[str, Any] = {
                "user_id": user_id,
                "manage_connections": manage_connections,
            }
            if toolkits:
                session_kwargs["toolkits"] = toolkits

            session = self._composio.create(**session_kwargs)
            logger.info("Created Tool Router session for user %s", user_id)
            return session

        except Exception as exc:
            logger.error("Failed to create session for user %s: %s", user_id, exc)
            raise

    def get_mcp_url(
        self, user_id: str, toolkits: Optional[list[str]] = None
    ) -> str:
        """Get the unified MCP URL for a user."""
        session = self.create_session(user_id, toolkits)
        mcp_url = session.mcp.url
        logger.info("Got Tool Router MCP URL for user %s", user_id)
        return mcp_url

    def get_all_toolkits(self, user_id: str) -> list[dict[str, Any]]:
        """Paginate through all available toolkits."""
        self._ensure_initialized()

        try:
            session = self.create_session(user_id)
            all_toolkits: list[Any] = []
            cursor = None

            while True:
                result = session.toolkits(limit=20, next_cursor=cursor)
                all_toolkits.extend(result.items)
                cursor = result.next_cursor
                if not cursor:
                    break

            logger.info("Retrieved %d toolkits for user %s", len(all_toolkits), user_id)
            return all_toolkits

        except Exception as exc:
            logger.error("Failed to get toolkits for user %s: %s", user_id, exc)
            raise

    # =====================================================================
    # Toolkit authorization
    # =====================================================================

    def authorize_toolkit(
        self, user_id: str, toolkit: str
    ) -> AuthorizeToolkitResponse:
        """Authorize a toolkit for a user via Tool Router."""
        self._ensure_initialized()

        try:
            logger.info("Authorizing toolkit %s for user %s", toolkit, user_id)
            session = self.create_session(user_id)

            try:
                connection_request = session.authorize(toolkit)
                logger.info("Toolkit authorization initiated for user %s", user_id)

                return AuthorizeToolkitResponse(
                    redirect_url=connection_request.redirect_url,
                    connection_request_id=getattr(connection_request, "id", None),
                    user_id=user_id,
                    toolkit=toolkit,
                    status=self.STATUS_PENDING,
                )

            except Exception as auth_error:
                error_lower = str(auth_error).lower()
                is_no_auth = any(
                    phrase in error_lower
                    for phrase in (
                        "no auth toolkit",
                        "cannot create an auth config for a no auth toolkit",
                        "error code: 303",
                        "'code': 303",
                        '"code": 303',
                        "does not require",
                    )
                )
                if is_no_auth:
                    logger.info("Toolkit %s requires no auth", toolkit)
                    return AuthorizeToolkitResponse(
                        redirect_url=None,
                        connection_request_id=None,
                        user_id=user_id,
                        toolkit=toolkit,
                        status=self.STATUS_NO_AUTH_REQUIRED,
                        requires_auth=False,
                    )
                raise

        except Exception as exc:
            logger.error("authorize_toolkit failed for %s / %s: %s", user_id, toolkit, exc)
            raise Exception(f"Failed to authorize toolkit: {exc}") from exc

    async def wait_for_connection(
        self,
        connection_request_id: str,
        user_id: str,
        toolkit: Optional[str] = None,
    ) -> WaitForConnectionResponse:
        """Wait for a connection to be established."""
        self._ensure_initialized()
        assert self._composio is not None

        try:
            if not connection_request_id or not connection_request_id.strip():
                return WaitForConnectionResponse(
                    status=self.STATUS_CONNECTED,
                    user_id=user_id,
                    toolkit=toolkit,
                    message="Direct connection via Tool Router",
                )

            logger.info("Waiting for connection %s for user %s", connection_request_id, user_id)

            loop = asyncio.get_running_loop()
            try:
                connected_account = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._composio.connected_accounts.wait_for_connection(
                            connection_request_id
                        ),
                    ),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for connection %s", connection_request_id)
                await self._cleanup_initiated_connected_accounts(user_id)
                raise Exception("Timed out waiting for connection after 60 seconds")

            logger.info("Connection established: %s", connected_account.id)
            toolkit_raw = getattr(connected_account, "toolkit", None)
            connected_toolkit = (
                self._to_toolkit_slug(toolkit_raw) if toolkit_raw else toolkit
            )

            return WaitForConnectionResponse(
                status=self.STATUS_CONNECTED,
                user_id=user_id,
                toolkit=connected_toolkit,
                connection_id=connected_account.id,
                message=f"User connected to {connected_toolkit} via Tool Router",
            )

        except Exception as exc:
            logger.error("wait_for_connection failed: %s", exc)
            raise Exception(f"Failed to establish connection: {exc}") from exc

    # =====================================================================
    # User auth-config management
    # =====================================================================

    async def get_user_auth_configs(self, user_id: str) -> UserAuthConfigsResponse:
        """Get auth configs for a user via the Composio REST API."""
        self._ensure_initialized()

        try:
            url = f"{self._base_url}/connected_accounts"
            headers = {
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
                "x-org-key": self._org_key or "",
            }
            params = {"user_ids": user_id}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            items = data.get("items", [])
            filtered: list[UserAuthConfigItem] = []
            for item in items:
                filtered.append(
                    UserAuthConfigItem(
                        toolkit=item.get("toolkit", {}).get("slug", ""),
                        auth_config_id=item.get("auth_config", {}).get("id", ""),
                        connected_account_id=item.get("id", ""),
                        status=item.get("status", ""),
                        is_disabled=item.get("is_disabled", False),
                        is_complete_setup=True,
                    )
                )

            logger.info("Retrieved %d auth configs for user %s", len(filtered), user_id)
            return UserAuthConfigsResponse(items=filtered, total_items=len(filtered))

        except Exception as exc:
            logger.error("get_user_auth_configs failed for %s: %s", user_id, exc)
            raise Exception(f"Failed to fetch auth configs: {exc}") from exc

    async def get_connected_toolkits(self, user_id: str) -> list[Any]:
        """Get only connected toolkits for a user."""
        self._ensure_initialized()

        try:
            session = self.create_session(user_id)
            toolkits = session.toolkits(is_connected=True)
            logger.info("Retrieved %d connected toolkits for user %s", len(toolkits.items), user_id)
            return toolkits.items

        except Exception as exc:
            logger.error("get_connected_toolkits failed for %s: %s", user_id, exc)
            raise

    async def delete_user_auth_config(
        self,
        user_id: str,
        toolkit: Optional[str] = None,
    ) -> DeleteAuthConfigResponse:
        """Delete auth config(s) for a user, optionally filtered by toolkit."""
        self._ensure_initialized()

        try:
            auth_configs = await self.get_user_auth_configs(user_id)
            configs_to_delete = auth_configs.items

            if toolkit:
                slug = self._to_toolkit_slug(toolkit)
                configs_to_delete = [
                    c for c in configs_to_delete if self._to_toolkit_slug(c.toolkit) == slug
                ]

            if not configs_to_delete:
                return DeleteAuthConfigResponse(
                    status=self.STATUS_NOT_FOUND,
                    user_id=user_id,
                    deleted_configs=[],
                    toolkit=toolkit,
                    message=f"No auth configs found{' for toolkit ' + toolkit if toolkit else ''}",
                )

            deleted: list[DeletedConfigItem] = []
            failed: list[FailedDeletionItem] = []

            for config in configs_to_delete:
                try:
                    if config.connected_account_id:
                        await self._delete_connected_account(config.connected_account_id)
                        deleted.append(
                            DeletedConfigItem(
                                toolkit=config.toolkit,
                                connected_account_id=config.connected_account_id,
                            )
                        )
                except Exception as exc:
                    failed.append(
                        FailedDeletionItem(
                            toolkit=config.toolkit,
                            connected_account_id=config.connected_account_id,
                            error=str(exc),
                        )
                    )

            return DeleteAuthConfigResponse(
                status=self.STATUS_COMPLETED if not failed else self.STATUS_PARTIAL_SUCCESS,
                user_id=user_id,
                deleted_configs=deleted,
                failed_deletions=failed or None,
            )

        except Exception as exc:
            logger.error("delete_user_auth_config failed: %s", exc)
            raise

    # =====================================================================
    # Helpers
    # =====================================================================

    @staticmethod
    def _to_toolkit_slug(toolkit: Any) -> str:
        """Normalise a toolkit object / dict / str to its slug string."""
        if hasattr(toolkit, "slug"):
            return str(getattr(toolkit, "slug"))
        if isinstance(toolkit, dict):
            val = toolkit.get("slug") or toolkit.get("name") or toolkit.get("toolkit")
            return str(val) if val is not None else ""
        if isinstance(toolkit, str):
            return toolkit
        return ""

    async def _delete_connected_account(self, connected_account_id: str) -> None:
        """Delete a connected account via the Composio REST API."""
        url = f"{self._base_url}/connected_accounts/{connected_account_id}"
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
            "x-org-key": self._org_key or "",
        }

        async with aiohttp.ClientSession() as session:
            async with session.delete(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()

        logger.info("Deleted connected account %s", connected_account_id)

    async def _cleanup_initiated_connected_accounts(self, user_id: str) -> None:
        """Remove any connected accounts stuck in INITIATED state."""
        try:
            auth_configs = await self.get_user_auth_configs(user_id)
            deleted = 0
            for config in auth_configs.items:
                if config.status and config.status.upper() == "INITIATED" and config.connected_account_id:
                    try:
                        await self._delete_connected_account(config.connected_account_id)
                        deleted += 1
                    except Exception:
                        pass
            if deleted:
                logger.info("Cleaned up %d INITIATED accounts for user %s", deleted, user_id)
        except Exception as exc:
            logger.error("Cleanup error for user %s: %s", user_id, exc)

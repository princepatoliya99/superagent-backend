"""
Composio Tool Router API routes.

Endpoints for session/MCP URL management, toolkit authorization,
and user auth-config CRUD.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.models.composio import (
    AuthorizeToolkitResponse,
    DeleteAuthConfigResponse,
    UserAuthConfigsResponse,
    WaitForConnectionResponse,
)
from app.services.composio_service import ComposioService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/composio", tags=["Composio"])

# Service instance is injected at startup from main.py
composio_service: ComposioService | None = None


def configure(service: ComposioService) -> None:
    """Inject the ComposioService instance (called from main.py)."""
    global composio_service
    composio_service = service


def _svc() -> ComposioService:
    """Return the service or raise 503 if unavailable."""
    if composio_service is None or not composio_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Composio service is not initialised. Set COMPOSIO_API_KEY in .env.",
        )
    return composio_service


# ── MCP / Session ──


@router.get("/mcp-url")
async def get_mcp_url(
    user_id: str = Query(..., description="User identifier"),
    toolkits: Optional[str] = Query(None, description="Comma-separated toolkit slugs"),
) -> dict[str, str]:
    """Get the unified MCP URL for a user."""
    svc = _svc()
    tk_list = [t.strip() for t in toolkits.split(",")] if toolkits else None
    url = svc.get_mcp_url(user_id, tk_list)
    return {"mcp_url": url, "user_id": user_id}


@router.get("/toolkits")
async def list_toolkits(
    user_id: str = Query(..., description="User identifier"),
):
    """List all available toolkits."""
    svc = _svc()
    return svc.get_all_toolkits(user_id)


@router.get("/toolkits/connected")
async def list_connected_toolkits(
    user_id: str = Query(..., description="User identifier"),
):
    """List only the connected toolkits for a user."""
    svc = _svc()
    return await svc.get_connected_toolkits(user_id)


# ── Authorization ──


@router.post("/authorize", response_model=AuthorizeToolkitResponse)
async def authorize_toolkit(
    user_id: str = Query(..., description="User identifier"),
    toolkit: str = Query(..., description="Toolkit slug to authorize"),
) -> AuthorizeToolkitResponse:
    """Authorize a toolkit for a user. Returns a redirect URL if auth is needed."""
    svc = _svc()
    try:
        return svc.authorize_toolkit(user_id, toolkit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/wait-for-connection", response_model=WaitForConnectionResponse)
async def wait_for_connection(
    connection_request_id: str = Query(..., description="Connection request ID"),
    user_id: str = Query(..., description="User identifier"),
    toolkit: Optional[str] = Query(None, description="Toolkit slug"),
) -> WaitForConnectionResponse:
    """Block until a connection is established (max 60 s)."""
    svc = _svc()
    try:
        return await svc.wait_for_connection(connection_request_id, user_id, toolkit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── User Auth Configs ──


@router.get("/auth-configs", response_model=UserAuthConfigsResponse)
async def get_auth_configs(
    user_id: str = Query(..., description="User identifier"),
) -> UserAuthConfigsResponse:
    """Get all auth configs for a user."""
    svc = _svc()
    return await svc.get_user_auth_configs(user_id)


@router.delete("/auth-configs", response_model=DeleteAuthConfigResponse)
async def delete_auth_configs(
    user_id: str = Query(..., description="User identifier"),
    toolkit: Optional[str] = Query(None, description="Optional toolkit slug to filter"),
) -> DeleteAuthConfigResponse:
    """Delete auth config(s) for a user, optionally for a specific toolkit."""
    svc = _svc()
    return await svc.delete_user_auth_config(user_id, toolkit)

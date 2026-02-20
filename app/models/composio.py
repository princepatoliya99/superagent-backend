"""
Pydantic schemas for Composio Tool Router responses.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


# ── Authorization ──


class AuthorizeToolkitResponse(BaseModel):
    """Response from authorizing a toolkit."""

    redirect_url: Optional[str] = None
    connection_request_id: Optional[str] = None
    user_id: str
    toolkit: str
    status: str
    requires_auth: bool = True


class WaitForConnectionResponse(BaseModel):
    """Response from waiting for a connection to be established."""

    status: str
    user_id: str
    toolkit: Optional[str] = None
    connection_id: Optional[str] = None
    mcp_server_id: Optional[str] = None
    instance_id: Optional[str] = None
    message: Optional[str] = None
    mcp_server: Optional[Any] = None


# ── User Auth Configs ──


class UserAuthConfigItem(BaseModel):
    """A single user auth config entry."""

    toolkit: str
    auth_config_id: str
    connected_account_id: str
    status: str
    is_disabled: bool = False
    is_complete_setup: bool = True


class UserAuthConfigsResponse(BaseModel):
    """Response containing all auth configs for a user."""

    items: list[UserAuthConfigItem]
    total_items: int


# ── Deletion ──


class DeletedConfigItem(BaseModel):
    """A successfully deleted config entry."""

    toolkit: str
    connected_account_id: str


class FailedDeletionItem(BaseModel):
    """A config entry that failed to delete."""

    toolkit: str
    connected_account_id: Optional[str] = None
    error: str


class DeleteAuthConfigResponse(BaseModel):
    """Response from deleting auth configs."""

    status: str
    user_id: str
    deleted_configs: list[DeletedConfigItem]
    failed_deletions: Optional[list[FailedDeletionItem]] = None
    toolkit: Optional[str] = None
    message: Optional[str] = None

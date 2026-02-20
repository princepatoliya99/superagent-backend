"""
Application settings loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the chat backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──
    LLM_MODEL: str = "gemini/gemini-2.0-flash"
    LLM_API_KEY: str = ""

    # ── Composio ──
    COMPOSIO_API_KEY: str = ""
    COMPOSIO_ORG_KEY: str = ""
    COMPOSIO_BASE_URL: str = "https://backend.composio.dev/api/v2"

    # ── Server ──
    HOST: str = "0.0.0.0"
    PORT: int = 5050
    DEBUG: bool = False


# Singleton instance
settings = Settings()

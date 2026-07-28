"""Validated configuration for the Streamlit UI."""

from __future__ import annotations

from functools import lru_cache
from typing import Self

from pydantic import Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class UISettings(BaseSettings):
    """Configuration loaded from RAG_-prefixed environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    api_url: HttpUrl = HttpUrl("http://localhost:8000")
    api_key: SecretStr = Field(min_length=16)

    default_top_k: int = Field(default=5, ge=1, le=20)
    max_top_k: int = Field(default=20, ge=1, le=20)
    connect_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    request_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    max_upload_size_mb: int = Field(default=25, ge=1, le=200)

    page_title: str = Field(
        default="📚 Enterprise RAG Asistanı",
        min_length=1,
    )
    page_icon: str = Field(default="📚", min_length=1)

    @model_validator(mode="after")
    def default_top_k_must_not_exceed_maximum(self) -> Self:
        if self.default_top_k > self.max_top_k:
            raise ValueError("RAG_DEFAULT_TOP_K must not exceed RAG_MAX_TOP_K")
        return self

    @property
    def api_base_url(self) -> str:
        """Return the normalized backend URL without a trailing slash."""

        return str(self.api_url).rstrip("/")


@lru_cache(maxsize=1)
def get_ui_settings() -> UISettings:
    """Return the cached UI configuration."""

    return UISettings()

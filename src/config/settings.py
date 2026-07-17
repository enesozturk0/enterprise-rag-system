from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",          # proje kökündeki .env
        env_file_encoding="utf-8",
        extra="ignore",           # tanımsız env değişkenlerinde hata verme
    )

    # ── LLM ──────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model_name: str = "llama3.1:8b"
    embedding_model_name: str = "bge-m3"

    # ── Vektör DB ────────────────────────────────────────
    chroma_persist_dir: Path = Path("./data/chroma_db")
    chroma_collection_name: str = "rag_documents"

    # ── Chunking ─────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ── Embedding ────────────────────────────────────────
    embedding_batch_size: int = 32

    # ── API ──────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: SecretStr = SecretStr("change-me-in-production")

    # ── Logging ──────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"

    # ── Validators ───────────────────────────────────────
    @field_validator("chunk_size")
    @classmethod
    def chunk_size_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: ...) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"CHUNK_OVERLAP ({v}) must be < CHUNK_SIZE ({chunk_size})")
        if v < 0:
            raise ValueError("CHUNK_OVERLAP must be >= 0")
        return v

    @field_validator("embedding_batch_size")
    @classmethod
    def batch_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        return v

    @field_validator("api_port")
    @classmethod
    def port_in_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("API_PORT must be between 1-65535")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_storage_path(raw_value: str | None, fallback_name: str) -> str:
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = (_PROJECT_ROOT / path).resolve()
        return str(path)
    return str((_PROJECT_ROOT / fallback_name).resolve())


class Settings(BaseSettings):
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias="GROQ_MODEL",
    )
    groq_fallback_models: str = Field(
        default="openai/gpt-oss-20b,qwen/qwen3.6-27b,openai/gpt-oss-120b",
        validation_alias="GROQ_FALLBACK_MODELS",
    )
    groq_vision_model: str = Field(
        default="llama-3.2-11b-vision-preview",
        validation_alias="GROQ_VISION_MODEL",
    )

    def get_fallback_models(self) -> list[str]:
        """Parse and sanitize GROQ_FALLBACK_MODELS.

        Removes empty values, duplicates, and the primary model if present.
        """
        if not self.groq_fallback_models:
            return []
        raw = [m.strip() for m in self.groq_fallback_models.split(",") if m.strip()]
        deduped = []
        for m in raw:
            if m != self.groq_model and m not in deduped:
                deduped.append(m)
        return deduped

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        validation_alias="EMBEDDING_MODEL",
    )
    tesseract_lang: str = Field(
        default="eng",
        validation_alias="TESSERACT_LANG",
    )

    sqlite_db_path: str = Field(default="./data/app.db", validation_alias="SQLITE_DB_PATH")
    chroma_db_path: str = Field(default="./data/chroma_db", validation_alias="CHROMA_DB_PATH")
    # Chroma Cloud configuration (if present the app will prefer cloud)
    chroma_host: str | None = None
    chroma_api_key: str | None = None
    chroma_tenant: str | None = None
    chroma_database: str | None = None

    retrieval_top_k: int = 20
    rerank_top_k: int = 10
    groundedness_min_support: float = 0.6
    debug_agent_pipeline: bool = True
    jwt_secret_key: str = "drugbot-dev-secret-change-me-in-production"

    # Token budgeting
    max_context_tokens: int = Field(default=3500, validation_alias="MAX_CONTEXT_TOKENS")
    max_output_tokens: int = Field(default=1000, validation_alias="MAX_OUTPUT_TOKENS")
    max_history_turns: int = Field(default=4, validation_alias="MAX_HISTORY_TURNS")

    # Logging configuration
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_dir: str = Field(default="logs", validation_alias="LOG_DIR")
    log_max_bytes: int = Field(default=10485760, validation_alias="LOG_MAX_BYTES")  # 10 MB
    log_backup_count: int = Field(default=5, validation_alias="LOG_BACKUP_COUNT")

    @model_validator(mode="after")
    def _normalize_storage_paths(self):
        self.sqlite_db_path = _resolve_storage_path(self.sqlite_db_path, "data/app.db")
        self.chroma_db_path = _resolve_storage_path(self.chroma_db_path, "data/chroma_db")
        return self

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


def groq_llm_kwargs(
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> dict:
    """Build kwargs for instantiating ChatGroq."""
    kwargs = {
        "model": model or settings.groq_model,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if settings.groq_api_key:
        kwargs["groq_api_key"] = settings.groq_api_key
    return kwargs

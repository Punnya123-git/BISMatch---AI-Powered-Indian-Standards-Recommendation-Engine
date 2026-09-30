"""Application configuration.

Every runtime setting is read from environment variables (optionally through a
``.env`` file). No credential, model name or provider choice is ever hard-coded,
and no API key is ever committed to the repository.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/app -> backend -> <project root>
BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent

#: Verified catalogues that ship with the repository, richest first. Used when
#: STANDARDS_DATASET_PATH is not set, so a deployment works with no catalogue
#: configuration at all. These are real, committed BIS records - the fallback
#: is never invented data, and it is only used when the file actually exists.
SHIPPED_STANDARDS_DATASETS = (
    PROJECT_ROOT / "data" / "standards" / "standards_enriched_bis_verified.json",
    PROJECT_ROOT / "data" / "standards" / "standards.json",
)


def default_standards_dataset_path() -> Path | None:
    """The first shipped catalogue present on disk, or ``None`` if there is none."""
    for candidate in SHIPPED_STANDARDS_DATASETS:
        if candidate.is_file():
            return candidate
    return None


class Settings(BaseSettings):
    """Strongly typed view over the environment."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env", PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -----------------------------------------------------
    app_name: str = "SIH PS-108 Standards Recommendation API"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True
    api_prefix: str = "/api"
    log_level: str = "INFO"
    docs_url: str = "/docs"

    # --- HTTP ------------------------------------------------------------
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )

    # --- Filesystem ------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    max_upload_size_mb: int = 25
    allowed_upload_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".txt", ".md"]
    )

    # --- LLM provider (empty until a provider is selected) ----------------
    llm_provider: str = "unconfigured"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # --- Embedding provider (empty until a provider is selected) ----------
    embedding_provider: str = "unconfigured"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None

    # --- Vector store ----------------------------------------------------
    vector_store_provider: str = "chroma"
    vector_collection_name: str = "indian_standards"
    vector_db_path: Path | None = None

    # --- Retrieval / chunking --------------------------------------------
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_top_k: int = 10

    # --- SQL database (PostgreSQL-ready, not enabled yet) -----------------
    database_url: str | None = None

    # --- Standards dataset (must be supplied, never invented) -------------
    # Defaults to the verified catalogue that ships with the repository, so a
    # deployed instance is never left without a catalogue just because the
    # environment variable was not set. Set STANDARDS_DATASET_PATH to override.
    standards_dataset_path: Path | None = Field(
        default_factory=default_standards_dataset_path
    )

    # --- Start-up initialisation ------------------------------------------
    # Build the standards index at application start when the store does not
    # already match the dataset. This is what makes deployment independent of a
    # build-time indexing step (see app/standards/bootstrap.py).
    auto_index_on_startup: bool = True

    # --- Derived paths ---------------------------------------------------
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def resolved_vector_db_path(self) -> Path:
        return self.vector_db_path or (self.data_dir / "vector_store")

    @property
    def is_development(self) -> bool:
        return self.environment.lower() in {"development", "dev", "local"}

    @property
    def database_enabled(self) -> bool:
        """True once a SQL database URL is configured (not required yet)."""
        return bool(self.database_url)

    def ensure_directories(self) -> None:
        """Create the directories the application writes to."""
        for directory in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.resolved_vector_db_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance (safe to import anywhere)."""
    return Settings()

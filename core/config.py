"""
core/config.py — Single source of truth for all application settings.

Loads config.yaml, then lets environment variables (from .env) override
any value. Exposes a typed `Settings` object accessible via `get_settings()`.

Design notes:
- Pydantic v2 BaseModel used for nested sections (not BaseSettings nesting,
  which has quirks with YAML injection).
- `get_settings()` is cached so the YAML file is parsed only once.
- Dependency injection: FastAPI routes receive Settings via `Depends(get_settings)`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ──────────────────────────────────────────────────────────────────────────────
# Nested config sections (plain Pydantic models)
# ──────────────────────────────────────────────────────────────────────────────


class LLMSettings(BaseModel):
    provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    reasoning_model: str = "qwen3:8b"
    vision_model: str = "mlx-community/gemma-4-12b-it-4bit"
    # "ollama" = use Ollama vision API | "mlx" = use mlx_lm locally (Apple Silicon)
    vision_backend: str = "mlx"
    timeout: int = 120
    max_retries: int = 3
    scoring_temperature: float = 0.1
    qa_temperature: float = 0.4


class StorageSettings(BaseModel):
    database_url: str = "sqlite+aiosqlite:///./data/career_agent.db"
    chroma_path: str = "./data/chroma"
    documents_dir: str = "/Users/himanshu/Desktop/Working/Linkdin"
    screenshots_dir: str = "./data/screenshots"
    logs_dir: str = "./data/logs"


class ProfileSettings(BaseModel):
    path: str = "./profile/profile.yaml"
    default_resume: str = "Himanshu_ATS_LATEX_7_JUNE.pdf"


class DiscoverySettings(BaseModel):
    min_relevance_score: float = 75.0
    max_jobs_per_run: int = 50
    llm_score_batch_size: int = 5
    memory_lookback_days: int = 30
    greenhouse_companies: list[str] = []
    lever_companies: list[str] = []
    ashby_companies: list[str] = []


class ApplicationSettings(BaseModel):
    daily_limit: int = 10
    dry_run: bool = True
    inter_application_delay: int = 30


class BrowserSettings(BaseModel):
    headless: bool = False
    browser_type: str = "chromium"
    viewport_width: int = 1440
    viewport_height: int = 900
    timeout: int = 30
    action_delay_ms: int = 500


class SchedulerSettings(BaseModel):
    discovery_interval_hours: int = 6
    application_cron: str = "30 3 * * 1-5"
    screenshot_retention_days: int = 30


class LoggingSettings(BaseModel):
    level: str = "INFO"
    rotation: str = "50 MB"
    retention: str = "30 days"
    json_logs: bool = True

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log level must be one of {valid}, got '{v}'")
        return v.upper()


# ──────────────────────────────────────────────────────────────────────────────
# Root Settings
# ──────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """
    Root application settings.

    Loaded in priority order (highest wins):
      1. Environment variables (or .env file)
      2. config.yaml values injected during construction
      3. Field defaults defined above
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = LLMSettings()
    storage: StorageSettings = StorageSettings()
    profile: ProfileSettings = ProfileSettings()
    discovery: DiscoverySettings = DiscoverySettings()
    application: ApplicationSettings = ApplicationSettings()
    browser: BrowserSettings = BrowserSettings()
    scheduler: SchedulerSettings = SchedulerSettings()
    logging: LoggingSettings = LoggingSettings()

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        dirs = [
            self.storage.logs_dir,
            self.storage.screenshots_dir,
            self.storage.chroma_path,
            # Extract directory from the database URL
            _db_dir(self.storage.database_url),
        ]
        for d in dirs:
            if d:
                Path(d).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _db_dir(database_url: str) -> str:
    """Extract the directory component from a SQLite URL."""
    # e.g. "sqlite+aiosqlite:///./data/career_agent.db" → "./data"
    path_part = database_url.split("///")[-1]
    return str(Path(path_part).parent)


def _load_yaml(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML config file and return a flat-nested dict."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ──────────────────────────────────────────────────────────────────────────────
# Factory — cached singleton
# ──────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_settings(config_path: str = "config.yaml") -> Settings:
    """
    Build and return the Settings singleton.

    The YAML file is the base layer; environment variables override.
    """
    yaml_data = _load_yaml(config_path)

    # Override with env vars that match the pattern SECTION__KEY
    # Pydantic Settings handles this automatically via env_nested_delimiter.
    # We just pass the YAML data as the initial values.
    settings = Settings(**yaml_data)
    settings.ensure_directories()
    return settings


# FastAPI dependency alias
def settings_dep() -> Settings:
    """FastAPI dependency — use with Depends(settings_dep)."""
    return get_settings()

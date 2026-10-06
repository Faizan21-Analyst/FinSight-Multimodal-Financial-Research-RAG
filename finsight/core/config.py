"""Settings: secrets/env via pydantic-settings, tunables via configs/*.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from finsight.core.exceptions import ConfigError

def _find_project_root() -> Path:
    """Walk up from this file until a folder containing 'configs/' is found.
    Works whether the package lives in src/finsight or directly in finsight/."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sec_user_agent: str = ""
    qdrant_url: str = "http://localhost:6333"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "finsight"
    postgres_user: str = "finsight"
    postgres_password: str = "finsight"
    anthropic_api_key: str = ""
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/processed/llm_cache")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def require_sec_user_agent(self) -> str:
        if not self.sec_user_agent.strip():
            raise ConfigError("SEC_USER_AGENT is not set. EDGAR requires 'Name email@example.com'.")
        return self.sec_user_agent


class Company(BaseModel):
    ticker: str
    name: str
    cik: str
    fiscal_year_end: str


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_yaml(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    path = config_dir / name
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def load_app_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    return load_yaml("settings.yaml", config_dir)


def load_companies(config_dir: Path = CONFIG_DIR) -> list[Company]:
    raw = load_yaml("companies.yaml", config_dir)
    return [Company(**c) for c in raw["companies"]]
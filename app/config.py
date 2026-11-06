"""Application settings loaded from environment variables and the root .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or unusable."""


class Settings(BaseSettings):
    """Typed application settings.

    The API key is stored as ``SecretStr`` so it is masked in ``repr``, logs and
    serialized output. Use ``require_openai_api_key`` to read it.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    openai_model: str = "gpt-4o-mini"
    app_env: Literal["development", "testing", "production"] = "development"
    debug: bool = False
    database_url: str = "sqlite:///./lesson_memory.db"
    max_retries: int = Field(default=2, ge=0, le=2)

    def require_openai_api_key(self) -> str:
        """Return the API key, or raise a clear error if it is not configured.

        Call this when initializing an LLM client, never at import time.
        """
        key = (
            self.openai_api_key.get_secret_value().strip()
            if self.openai_api_key
            else ""
        )
        if not key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Add it to the .env file in the project "
                "root or export it as an environment variable."
            )
        return key


@lru_cache
def get_settings() -> Settings:
    """Return the shared, cached settings instance."""
    return Settings()

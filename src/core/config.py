"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Market data
    alpha_vantage_api_key: str = ""
    finnhub_api_key: str = ""

    # Social/web intelligence
    twitter_bearer_token: str = ""
    youtube_api_key: str = ""
    news_api_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""

    # Notifications
    notification_urls: str = ""

    # App settings
    prediction_interval_minutes: int = 60
    default_confidence_threshold: int = 60
    database_url: str = "sqlite+aiosqlite:///./stock_predictor.db"
    log_level: str = "INFO"

    # LLM settings
    llm_provider: str = Field(default="anthropic", description="'openai' or 'anthropic'")
    llm_model: str = Field(
        default="claude-opus-4-6",
        description="Model ID for the chosen provider",
    )
    llm_temperature: float = 0.2

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def active_api_key(self) -> str:
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return self.openai_api_key

    @property
    def notification_url_list(self) -> list[str]:
        if not self.notification_urls:
            return []
        return [u.strip() for u in self.notification_urls.split(",") if u.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

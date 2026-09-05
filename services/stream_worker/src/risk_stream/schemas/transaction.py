from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"

    redpanda_bootstrap_servers: str = "localhost:9092"

    transaction_topic: str = "transactions.raw"
    risk_decision_topic: str = "risk.decisions"

    consumer_group: str = "risk-sentinel-stream-worker"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "models"
    / "fraud_online_v5_catboost.cbm"
)

DEFAULT_CALIBRATOR_PATH = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "models"
    / "fraud_online_v5_isotonic_calibrator.joblib"
)


class Settings(BaseSettings):
    app_env: str = "development"

    redpanda_bootstrap_servers: str = "localhost:9092"

    transaction_topic: str = "transactions.raw"
    risk_decision_topic: str = "risk.decisions"
    transactions_dlq_topic: str = "transactions.raw.dlq"

    consumer_group: str = "risk-sentinel-persistence-v1"

    risk_model_path: str = str(DEFAULT_MODEL_PATH)
    risk_calibrator_path: str = str(DEFAULT_CALIBRATOR_PATH)

    model_version: str = "fraud-online-v5"
    calibration_version: str = "isotonic-online-v5"

    model_fail_closed: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

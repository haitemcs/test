"""Central configuration for local, test, and deployed environments."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Local-machine default. Docker Compose overrides this with hostname "db",
    # because containers reach each other by service name, not by localhost.
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/btcforecast"

    # Market-data settings used by the training script.
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    data_start: str = "2024-01-01"
    data_end: str = "2026-09-30"
    n_splits: int = 5

    # API metadata and locations of saved model artifacts.
    api_title: str = "BTC Volatility Forecast API"
    model_path: str = "artifacts/model.joblib"
    features_path: str = "artifacts/feature_columns.json"

    # Read .env during local development and ignore unrelated environment variables.
    # Values set in the process environment override values from the .env file.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Other modules import this shared settings object instead of reading variables alone.
settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/btcforecast"
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    data_start: str = "2024-01-01"
    data_end: str = "2026-09-30"
    n_splits: int = 5
    api_title: str = "BTC Volatility Forecast API"
    model_path: str = "artifacts/model.joblib"
    features_path: str = "artifacts/feature_columns.json"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

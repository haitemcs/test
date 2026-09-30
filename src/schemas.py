from datetime import datetime
from pydantic import BaseModel, Field

class Candle(BaseModel):
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    quote_asset_volume: float = Field(ge=0)
    number_of_trades: float = Field(ge=0)
    taker_buy_base_asset_volume: float = Field(ge=0)

class PredictionResponse(BaseModel):
    predicted_next_hour_range_pct: float
    model: str
    timestamp: datetime

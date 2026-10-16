"""Request and response shapes used by FastAPI."""

from datetime import datetime

from pydantic import BaseModel, Field


class Candle(BaseModel):
    """One market candle supplied by the client.

    Pydantic checks field types and the simple numeric constraints below before
    the route uses the data. Cross-field rules can be added as validators later.
    """

    timestamp: datetime
    open: float = Field(gt=0)  # Price must be greater than zero.
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)  # Volume may be zero, but not negative.
    quote_asset_volume: float = Field(ge=0)
    number_of_trades: float = Field(ge=0)
    taker_buy_base_asset_volume: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """The stable JSON shape returned after a successful prediction."""

    predicted_next_hour_range_pct: float
    model: str
    timestamp: datetime

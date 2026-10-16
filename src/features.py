"""Convert raw hourly market candles into model-ready numeric columns."""

import numpy as np
import pandas as pd

# Keep the feature list in one place so training and the API use the same columns.
FEATURES = [
    "return_pct", "hourly_range", "rt_cc", "range_ma_24", "range_ma_168",
    "volatility_24", "volatility_168", "volume_change", "volume_ratio",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "distance_ma_24", "distance_ma_168", "range_lag_1", "range_lag_2",
    "range_lag_3", "range_lag_6", "range_lag_12", "taker_buy_ratio",
    "order_flow_imbalance", "ofi_ma_24", "ofi_std_24", "avg_trade_size",
    "trade_size_ratio", "trade_ratio_lag_1", "trade_ratio_lag_2",
    "trade_ratio_lag_3", "ofi_lag_1", "ofi_lag_2", "ofi_lag_3",
]

# The value the model learns to predict. This is not one of the input features.
TARGET = "target_next_hour_range"


def make_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Clean candle data and create features, optionally including the training target.

    The input is copied so the caller's DataFrame is not modified. Rolling windows
    and lag calculations need past rows, so early incomplete rows are removed at the end.
    """
    x = df.copy()

    # Convert timestamps to a consistent timezone, sort old-to-new, and remove duplicates.
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Market APIs may return numbers as strings. Convert them to numeric values;
    # invalid strings become NaN and are removed if a required feature uses them.
    numeric = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_asset_volume",
    ]
    for column in numeric:
        x[column] = pd.to_numeric(x[column], errors="coerce")

    # Describe the current candle. Percentages make ranges easier to compare across prices.
    x["return_pct"] = (x["close"] / x["open"] - 1.0) * 100
    x["hourly_range"] = (x["high"] - x["low"]) / x["low"] * 100
    x["rt_cc"] = x["close"].pct_change() * 100

    # Rolling means summarize the last day/week; standard deviation summarizes variation.
    x["range_ma_24"] = x["hourly_range"].rolling(24).mean()
    x["range_ma_168"] = x["hourly_range"].rolling(168).mean()
    x["volatility_24"] = x["rt_cc"].rolling(24).std()
    x["volatility_168"] = x["rt_cc"].rolling(168).std()

    # Compare current volume with the previous candle and its 24-candle average.
    x["volume_change"] = x["volume"].pct_change() * 100
    x["volume_ma_24"] = x["volume"].rolling(24).mean()
    x["volume_ratio"] = x["volume"] / (x["volume_ma_24"] + 1e-8)

    # Sine/cosine encoding keeps cyclic times close together: midnight is near 23:00.
    tau = 2 * np.pi
    hour = x["timestamp"].dt.hour
    dow = x["timestamp"].dt.dayofweek
    x["hour_sin"] = np.sin(tau * hour / 24)
    x["hour_cos"] = np.cos(tau * hour / 24)
    x["dow_sin"] = np.sin(tau * dow / 7)
    x["dow_cos"] = np.cos(tau * dow / 7)
    x["is_weekend"] = (dow >= 5).astype(float)

    # Measure how far the current close is from its recent average price.
    x["ma_24_price"] = x["close"].rolling(24).mean()
    x["ma_168_price"] = x["close"].rolling(168).mean()
    x["distance_ma_24"] = (x["close"] - x["ma_24_price"]) / (x["ma_24_price"] + 1e-8)
    x["distance_ma_168"] = (x["close"] - x["ma_168_price"]) / (x["ma_168_price"] + 1e-8)

    # Lag means "the value from this many candles ago".
    for lag in (1, 2, 3, 6, 12):
        x[f"range_lag_{lag}"] = x["hourly_range"].shift(lag)

    # Approximate buying pressure from Binance's taker-buy volume.
    # The tiny constant avoids division by zero if volume is zero.
    x["taker_buy_ratio"] = x["taker_buy_base_asset_volume"] / (x["volume"] + 1e-8)
    x["order_flow_imbalance"] = (
        2 * x["taker_buy_base_asset_volume"] - x["volume"]
    ) / (x["volume"] + 1e-8)
    x["ofi_ma_24"] = x["order_flow_imbalance"].rolling(24).mean()
    x["ofi_std_24"] = x["order_flow_imbalance"].rolling(24).std()

    # Estimate average volume per trade and compare it with the recent average.
    x["avg_trade_size"] = x["volume"] / (x["number_of_trades"] + 1e-8)
    x["trade_size_ma_24"] = x["avg_trade_size"].rolling(24).mean()
    x["trade_size_ratio"] = x["avg_trade_size"] / (x["trade_size_ma_24"] + 1e-8)

    # Add lagged buying-ratio and order-flow measurements as additional signals.
    for lag in (1, 2, 3):
        x[f"trade_ratio_lag_{lag}"] = x["taker_buy_ratio"].shift(lag)
        x[f"ofi_lag_{lag}"] = x["order_flow_imbalance"].shift(lag)

    if include_target:
        # shift(-1) moves the next candle's range onto this row as the answer to learn.
        # TARGET must not be included in FEATURES, or training would leak the answer.
        x[TARGET] = x["hourly_range"].shift(-1)

    # Rolling calculations and lags create missing values near the start. The final
    # training row also has no target because its next candle is outside this dataset.
    required_columns = FEATURES + ([TARGET] if include_target else [])
    return x.dropna(subset=required_columns).reset_index(drop=True)

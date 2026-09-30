import numpy as np
import pandas as pd
from src.features import FEATURES, TARGET, make_features

def sample(n=220):
    t = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    base = 100 + np.arange(n) * 0.1
    return pd.DataFrame({
        "timestamp": t,
        "open": base,
        "high": base + 2,
        "low": base - 1,
        "close": base + 0.5,
        "volume": np.full(n, 1000.0),
        "quote_asset_volume": base * 1000,
        "number_of_trades": np.full(n, 100.0),
        "taker_buy_base_asset_volume": np.full(n, 500.0),
    })

def test_features_have_no_nan():
    data = make_features(sample())
    assert not data[FEATURES + [TARGET]].isna().any().any()
    assert len(data) > 0

def test_target_is_future_range():
    raw = sample()
    data = make_features(raw)
    assert data[TARGET].iloc[0] >= 0

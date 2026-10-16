"""Feature-engineering tests using generated data instead of live market downloads."""

import numpy as np
import pandas as pd

from src.features import FEATURES, TARGET, make_features


def sample(n=220):
    """Build predictable fake hourly candles so tests stay fast and repeatable."""
    timestamps = pd.date_range(
        "2026-01-01",
        periods=n,
        freq="h",
        tz="UTC",
    )

    base = 100 + np.arange(n) * 0.1

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base,
            "high": base + 2,
            "low": base - 1,
            "close": base + 0.5,
            "volume": np.full(n, 1000.0),
            "quote_asset_volume": base * 1000,
            "number_of_trades": np.full(n, 100.0),
            "taker_buy_base_asset_volume": np.full(n, 500.0),
        }
    )


def test_features_have_no_nan():
    """After incomplete rolling rows are removed, model columns should be complete."""
    data = make_features(sample())
    columns_to_check = FEATURES + [TARGET]

    assert not data[columns_to_check].isna().any().any()
    assert len(data) > 0


def test_target_is_future_range():
    """The target is a non-negative range percentage shifted from the next candle."""
    data = make_features(sample())

    assert data[TARGET].iloc[0] >= 0

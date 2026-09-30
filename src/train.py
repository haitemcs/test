import json
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit

from src.config import settings
from src.features import FEATURES, TARGET, make_features
from src.model import make_models, metrics, naive_prediction

BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_klines(start: str, end: str) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows = []
    current = start_ms
    while current < end_ms:
        response = requests.get(
            BINANCE_URL,
            params={"symbol": settings.symbol, "interval": settings.interval,
                    "startTime": current, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        current = batch[-1][0] + 1
        if len(batch) < 1000:
            break

    columns = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume",
               "quote_asset_volume", "number_of_trades",
               "taker_buy_base_asset_volume"]]

def run():
    raw = fetch_klines(settings.data_start, settings.data_end)
    data = make_features(raw)
    X, y = data[FEATURES], data[TARGET]

    tscv = TimeSeriesSplit(n_splits=settings.n_splits)
    model_scores = {name: [] for name in ["naive", *make_models().keys()]}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_scores["naive"].append(metrics(y_test, naive_prediction(y_train, len(test_idx))))
        for name, model in make_models().items():
            model.fit(X_train, y_train)
            model_scores[name].append(metrics(y_test, model.predict(X_test)))

    summary = {}
    for name, scores in model_scores.items():
        summary[name] = {
            metric: float(np.mean([s[metric] for s in scores]))
            for metric in ("mae", "rmse", "r2")
        }

    winner = min((n for n in summary if n != "naive"), key=lambda n: summary[n]["mae"])
    final_model = make_models()[winner]
    final_model.fit(X, y)

    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(final_model, settings.model_path)
    Path(settings.features_path).write_text(json.dumps(FEATURES, indent=2))
    Path("artifacts/metrics.json").write_text(json.dumps({
        "symbol": settings.symbol, "interval": settings.interval,
        "data_start": settings.data_start, "data_end": settings.data_end,
        "n_rows": len(data), "n_splits": settings.n_splits,
        "selected_model": winner, "results": summary,
    }, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"selected_model={winner}")

if __name__ == "__main__":
    run()

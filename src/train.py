"""Download market candles, compare models over time, and save the selected model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import TimeSeriesSplit

from src.config import settings
from src.features import FEATURES, TARGET, make_features
from src.model import make_models, metrics, naive_prediction

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(start: str, end: str) -> pd.DataFrame:
    """Download candle history from Binance in batches of up to 1000 rows."""
    # Binance expects timestamps in milliseconds. Convert the requested dates to UTC
    # so the boundaries do not depend on the computer's local timezone.
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows = []
    current = start_ms

    # The API has a 1000-candle response limit, so request the date range in batches.
    while current < end_ms:
        response = requests.get(
            BINANCE_URL,
            params={
                "symbol": settings.symbol,
                "interval": settings.interval,
                "startTime": current,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        response.raise_for_status()  # Stop here if Binance reports an HTTP error.
        batch = response.json()

        if not batch:
            break

        rows.extend(batch)
        current = batch[-1][0] + 1

        if len(batch) < 1000:
            break

    # Binance returns more fields than this project needs. Give them readable names,
    # create a UTC timestamp, and keep only columns used by feature engineering.
    columns = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    return df[
        [
            "timestamp", "open", "high", "low", "close", "volume",
            "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        ]
    ]


def run():
    """Run the complete download, feature, evaluation, and save workflow."""
    raw = fetch_klines(settings.data_start, settings.data_end)
    data = make_features(raw)

    # X contains the input measurements; y contains the future value we want to predict.
    X = data[FEATURES]
    y = data[TARGET]

    # Keep rows in time order. Each validation fold tests on rows that come after
    # its training rows, which better reflects forecasting into the future.
    splitter = TimeSeriesSplit(n_splits=settings.n_splits)
    model_scores = {"naive": []}
    models = make_models()

    for name in models:
        model_scores[name] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Compare every model with a simple baseline that predicts a recent average.
        baseline = naive_prediction(y_train, len(test_idx))
        model_scores["naive"].append(metrics(y_test, baseline))

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            model_scores[name].append(metrics(y_test, predictions))

    # Average each metric across folds so candidates can be compared using one summary.
    summary = {}
    for name, scores in model_scores.items():
        summary[name] = {}
        for metric_name in ("mae", "rmse", "r2"):
            values = []
            for score in scores:
                values.append(score[metric_name])
            summary[name][metric_name] = float(np.mean(values))

    # Select the ML candidate with the lowest average MAE. The baseline is reported
    # as a reference, but it is not saved as the final model.
    model_names = []
    for name in summary:
        if name != "naive":
            model_names.append(name)
    winner = model_names[0]
    for name in model_names:
        if summary[name]["mae"] < summary[winner]["mae"]:
            winner = name

    # Validation estimates performance. Refit the winner on all prepared rows before
    # saving it so the serving API can use the full available training dataset.
    final_model = models[winner]
    final_model.fit(X, y)

    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(final_model, settings.model_path)
    Path(settings.features_path).write_text(json.dumps(FEATURES, indent=2))

    # Save experiment settings and scores alongside the model for later inspection.
    Path("artifacts/metrics.json").write_text(
        json.dumps(
            {
                "symbol": settings.symbol,
                "interval": settings.interval,
                "data_start": settings.data_start,
                "data_end": settings.data_end,
                "n_rows": len(data),
                "n_splits": settings.n_splits,
                "selected_model": winner,
                "results": summary,
            },
            indent=2,
        )
    )

    print(json.dumps(summary, indent=2))
    print(f"selected_model={winner}")


if __name__ == "__main__":
    # This block runs when this file is executed as a script, not when imported.
    run()

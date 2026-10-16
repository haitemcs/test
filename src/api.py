"""HTTP routes for health checks and model predictions.

Training is handled separately by src.train. The API loads a saved model at startup
so requests can focus on preparing inputs, predicting, and recording the result.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import settings
from src.db import Prediction, SessionLocal, init_db
from src.features import FEATURES, make_features
from src.schemas import Candle, PredictionResponse

app = FastAPI(title=settings.api_title, version="0.1.0")

# Filled at startup. The server can report its health even if the model file is absent.
MODEL = None
MODEL_NAME = "unavailable"


@app.on_event("startup")
def startup():
    """Create missing database tables and load the saved model once."""
    global MODEL, MODEL_NAME

    init_db()

    model_file = Path(settings.model_path)
    if model_file.exists():
        MODEL = joblib.load(model_file)

        # The metrics file remembers which model won the last training experiment.
        metrics_file = Path("artifacts/metrics.json")
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            MODEL_NAME = metrics.get("selected_model", "trained_model")


@app.get("/health")
def health():
    """Report whether the API process is responding and whether a model loaded."""
    # A successful health response does not prove that every database operation works.
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model": MODEL_NAME,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(candle: Candle):
    """Single-candle example route; rolling features mean history is actually needed."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifact not found. Train first.")

    # A single row cannot supply the 24-hour and 168-hour rolling history. The feature
    # builder therefore returns no complete rows, so we fail clearly instead of guessing.
    raw = pd.DataFrame([candle.model_dump()])
    data = make_features(raw, include_target=False)

    if data.empty:
        raise HTTPException(
            status_code=422,
            detail="Prediction requires historical candles for rolling features; use /predict-history.",
        )

    last_row = data.iloc[[-1]]
    features = last_row[FEATURES]
    prediction = float(MODEL.predict(features)[0])

    # Open a short-lived session, add the prediction, and commit it to the database.
    with SessionLocal() as session:
        session.add(
            Prediction(
                timestamp=candle.timestamp,
                prediction=prediction,
                model_name=MODEL_NAME,
                request_json=candle.model_dump_json(),
            )
        )
        session.commit()

    return PredictionResponse(
        predicted_next_hour_range_pct=prediction,
        model=MODEL_NAME,
        timestamp=candle.timestamp,
    )


@app.post("/predict-history", response_model=PredictionResponse)
def predict_history(candles: list[Candle]):
    """Create features from a candle history, predict, and save the result."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifact not found. Train first.")

    # Rolling windows need earlier candles. This minimum is a basic guard; a production
    # API should also set a maximum request size to prevent huge requests.
    if len(candles) < 200:
        raise HTTPException(status_code=422, detail="Provide at least 200 hourly candles.")

    # Convert validated Pydantic objects into dictionaries, then a pandas DataFrame.
    raw_candles = []
    for candle in candles:
        raw_candles.append(candle.model_dump())

    raw = pd.DataFrame(raw_candles)
    data = make_features(raw, include_target=False)

    # Select the latest complete row and keep the exact feature order used in training.
    last_row = data.iloc[[-1]]
    features = last_row[FEATURES]
    prediction = float(MODEL.predict(features)[0])
    last = candles[-1]

    # Store the last five submitted candles as a compact record of the request.
    with SessionLocal() as session:
        recent_candles = []
        for candle in candles[-5:]:
            recent_candles.append(candle.model_dump(mode="json"))

        session.add(
            Prediction(
                timestamp=last.timestamp,
                prediction=prediction,
                model_name=MODEL_NAME,
                request_json=json.dumps(recent_candles),
            )
        )
        session.commit()

    return PredictionResponse(
        predicted_next_hour_range_pct=prediction,
        model=MODEL_NAME,
        timestamp=last.timestamp,
    )

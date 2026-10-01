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

MODEL = None
MODEL_NAME = "unavailable"

@app.on_event("startup")
def startup():
    global MODEL, MODEL_NAME
    init_db()
    if Path(settings.model_path).exists():
        MODEL = joblib.load(settings.model_path)
        if Path("artifacts/metrics.json").exists():
            MODEL_NAME = json.loads(Path("artifacts/metrics.json").read_text()).get(
                "selected_model", "trained_model"
            )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model": MODEL_NAME}

@app.post("/predict", response_model=PredictionResponse)
def predict(candle: Candle):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifact not found. Train first.")

    raw = pd.DataFrame([candle.model_dump()])
    data = make_features(raw, include_target=False)

    # A single candle cannot contain rolling history. The production endpoint therefore
    # expects a feature-ready history in a future streaming implementation. This explicit
    # error prevents silently producing a misleading forecast.
    if data.empty:
        raise HTTPException(
            status_code=422,
            detail="Prediction requires historical candles for rolling features; use /predict-history.",
        )

    last_row = data.iloc[[-1]]
    features = last_row[FEATURES]
    prediction = float(MODEL.predict(features)[0])
    with SessionLocal() as session:
        session.add(Prediction(
            timestamp=candle.timestamp,
            prediction=prediction,
            model_name=MODEL_NAME,
            request_json=candle.model_dump_json(),
        ))
        session.commit()

    return PredictionResponse(
        predicted_next_hour_range_pct=prediction,
        model=MODEL_NAME,
        timestamp=candle.timestamp,
    )

@app.post("/predict-history", response_model=PredictionResponse)
def predict_history(candles: list[Candle]):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifact not found. Train first.")
    if len(candles) < 200:
        raise HTTPException(status_code=422, detail="Provide at least 200 hourly candles.")

    raw_candles = []

    for candle in candles:
        raw_candles.append(candle.model_dump())

    raw = pd.DataFrame(raw_candles)
    data = make_features(raw, include_target=False)
    from src.features import FEATURES
    prediction = float(MODEL.predict(data.iloc[[-1]][FEATURES])[0])
    last = candles[-1]

    with SessionLocal() as session:
        session.add(Prediction(
            timestamp=last.timestamp,
            prediction=prediction,
            model_name=MODEL_NAME,
            request_json=json.dumps(
                [c.model_dump(mode="json") for c in candles[-5:]]
            ),
        ))
        session.commit()

    return PredictionResponse(
        predicted_next_hour_range_pct=prediction,
        model=MODEL_NAME,
        timestamp=last.timestamp,
    )

# BTC Volatility Forecasting

End-to-end time-series machine learning project for forecasting next-hour BTC/USDT volatility.

## Pipeline

**Binance OHLCV → feature engineering → chronological / walk-forward validation → model comparison → persisted model + experiment results → FastAPI → PostgreSQL → Docker → GitHub Actions CI → deployment-ready container**

The original research notebook is in [haitemcs/btc-volatility-forecast](https://github.com/haitemcs/btc-volatility-forecast). This repository turns that work into a reproducible engineering project.

## Target

Predict the next-hour BTC volatility proxy:

target_t = 100 * (High_(t+1) - Low_(t+1)) / Low_(t+1)

Features are constructed only from information available at time t. Validation is chronological; no random train/test split is used.

## Models

- Naive rolling-mean baseline
- Ridge regression
- HistGradientBoostingRegressor

Training reports MAE, RMSE and R² and selects the lowest-MAE model using walk-forward validation.

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

API docs: http://localhost:8000/docs

Health check:

```bash
curl http://localhost:8000/health
```

## Train

```bash
python -m src.train
```

Environment variables include DATA_START, DATA_END, SYMBOL, INTERVAL and N_SPLITS.

Training writes:
- artifacts/model.joblib
- artifacts/metrics.json
- artifacts/feature_columns.json

## PostgreSQL

The API records forecast requests and predictions in PostgreSQL. Tables are experiments and predictions.

## CI

GitHub Actions runs dependency installation, Ruff linting, unit tests and a Docker build.

## Deployment

The project is containerized and includes render.yaml as a deployment blueprint. Set DATABASE_URL in the hosting environment.

## Project structure

```
.github/workflows/ci.yml
artifacts/
src/
  api.py
  config.py
  db.py
  features.py
  model.py
  schemas.py
  train.py
tests/
Dockerfile
docker-compose.yml
pyproject.toml
render.yaml
README.md
```

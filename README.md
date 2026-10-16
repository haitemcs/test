# BTC Volatility Forecasting — Beginner's Backend + ML Study Guide

This repository connects Python, time-series machine learning, an HTTP API, PostgreSQL, Docker, tests, and continuous integration. The goal is to understand how raw data becomes a prediction, how a client requests it, how the service stores what happened, and how the application is packaged and checked.

The code is split into small files so you can study one responsibility at a time. Code comments explain important lines; this guide explains how the pieces fit together.

> **Important:** this predicts a historical-data-based volatility proxy. It is not a trading strategy, a price-direction forecast, or financial advice. Past validation results do not guarantee future performance.

## 1. What does the application do?

There are two workflows.

### A — Train a model

1. Download hourly BTC/USDT candle data from Binance.
2. Clean and sort the rows.
3. Convert raw candles into numeric features.
4. Define the target: the next candle's high-low range as a percentage of its low.
5. Compare a simple baseline with two machine-learning models using chronological validation.
6. Select the model with the lowest average validation MAE.
7. Train that model on all prepared rows and save it to `artifacts/model.joblib`.
8. Save evaluation metrics and the expected feature names beside the model.

### B — Serve predictions

1. A client sends historical candle data to FastAPI.
2. Pydantic checks the request's fields and basic value constraints.
3. Pandas builds the same features used during training.
4. The saved model predicts the next-hour range percentage.
5. SQLAlchemy writes the request and prediction to PostgreSQL.
6. FastAPI returns JSON.

The API does not train a model for every request. Training is separate; the API loads a saved model when it starts.

## 2. Architecture: follow the request

~~~text
Client (curl, browser, or another application)
        |
        | HTTP POST /predict-history + JSON candles
        v
FastAPI route in src/api.py
        |
        +--> Pydantic schema validates the request
        |
        +--> src/features.py builds rolling/time features
        |
        +--> artifacts/model.joblib predicts the target
        |
        +--> SQLAlchemy session writes a Prediction row
        |          |
        |          v
        |      PostgreSQL
        |
        v
JSON response to the client
~~~

Training is a separate path:

~~~text
Binance HTTP API
    -> src/train.py downloads candles
    -> src/features.py creates features and target
    -> chronological folds compare models in src/model.py
    -> winning model is retrained on all prepared rows
    -> artifacts/model.joblib + metrics.json + feature_columns.json
    -> API loads model.joblib during startup
~~~

### What each layer means

- **Client:** the program asking for a prediction, such as a frontend, script, or curl.
- **HTTP/API layer:** receives requests, validates inputs, chooses what code runs, and returns responses.
- **Schema:** describes valid request and response data. Pydantic rejects malformed inputs before business logic uses them.
- **Feature engineering:** transforms raw market observations into measurements the model can use.
- **Model:** a trained function that maps feature values to a prediction.
- **Database layer:** stores durable records so predictions do not disappear when the process restarts.
- **Container:** packages the app and dependencies for a predictable runtime.
- **Tests and CI:** automatically check selected behavior, including on pushes and pull requests.

This is a small learning project, so some responsibilities are combined. A larger service may add a service layer, migrations, structured logging, authentication, rate limiting, background jobs, and monitoring.

## 3. Reading waterfall: study the files in this order

Follow this order rather than opening files randomly.

### Stage 1 — Data contracts and settings

1. [src/schemas.py](src/schemas.py) — What does a candle look like? Which values are required? What does the API return?
2. [src/config.py](src/config.py) — How are settings such as the database URL and model path read?

Before moving on, explain the difference between a Python class, a field, a type such as float, and a validation rule such as greater than zero.

### Stage 2 — Feature engineering

3. [src/features.py](src/features.py) — How do raw candles become model inputs? Why do rolling calculations create missing rows? Why is the target shifted into the future?
4. [tests/test_features.py](tests/test_features.py) — How do tests check feature output and target construction?

Do not skip this stage. Training and API code must build the same features in the same order. A model can run without an exception and still be wrong if its inputs differ from training.

### Stage 3 — Machine learning

5. [src/model.py](src/model.py) — Which models are compared? What do MAE, RMSE, and R² measure?
6. [src/train.py](src/train.py) — How are data downloaded, split by time, evaluated, selected, and saved?
7. [artifacts/metrics.json](artifacts/metrics.json) — What did the recorded experiment report?
8. [artifacts/feature_columns.json](artifacts/feature_columns.json) — Which columns does the model expect?

Explain why a random train/test split can give an unrealistic estimate for time-series forecasting.

### Stage 4 — Database and API

9. [src/db.py](src/db.py) — How do Python classes map to database tables? What is an engine versus a session?
10. [src/api.py](src/api.py) — What happens at startup and during a prediction request? What if the model file is missing?
11. [tests/test_api.py](tests/test_api.py) — What does the health test verify, and what does it not verify?

### Stage 5 — Running the system

12. [pyproject.toml](pyproject.toml) — Which libraries are runtime dependencies and which tools are for development?
13. [Dockerfile](Dockerfile) — How is the API image built?
14. [docker-compose.yml](docker-compose.yml) — How are the API and PostgreSQL started together?
15. [.github/workflows/ci.yml](.github/workflows/ci.yml) — What checks run automatically?
16. [render.yaml](render.yaml) — Which deployment settings are described, and which environment values must be supplied separately?

## 4. Python concepts used here

You do not need to understand every advanced-looking line at once.

- **Import:** `from src.features import make_features` means use the function named make_features from that module.
- **Function:** `def make_features(...):` defines reusable steps. Parameters are inputs; `return` sends a result back.
- **Class:** groups data and behavior. Here, classes describe settings, request schemas, model results, and database tables.
- **Type annotation:** `timestamp: datetime` documents the expected value type. Pydantic also uses annotations to validate request data.
- **List:** an ordered collection, such as `[1, 2, 3]`.
- **Dictionary:** key/value pairs, such as `{"mae": 0.25}`.
- **Loop:** `for candle in candles:` repeats a block for each item.
- **Conditional:** `if MODEL is None:` runs a block only when the condition is true.
- **Exception:** `raise HTTPException(...)` stops the request path and tells FastAPI to return an HTTP error.
- **Context manager:** `with SessionLocal() as session:` manages a resource for a block and closes the session afterward.
- **DataFrame:** a table of rows and columns provided by pandas.
- **Model pipeline:** a sequence of preprocessing and model steps. The Ridge pipeline scales features before regression.
- **Environment variable:** configuration supplied outside source code, such as DATABASE_URL.

When syntax feels dense, translate it into plain language first. For example, `data[FEATURES]` means “select only the columns whose names are listed in FEATURES.”

## 5. The target and why history matters

The target is:

~~~text
100 * (next candle's high - next candle's low) / next candle's low
~~~

It measures the next hourly candle's high-low range as a percentage of its low. It does not predict whether BTC will go up or down, and it does not directly estimate profit or loss.

A feature must only use information available at prediction time. The target uses the next candle, but must never be an input feature. In src/features.py, `shift(-1)` moves the next row's range into the current row's target. That gives training an answer without leaking that answer into the model inputs.

Rolling features need earlier candles. A 168-row rolling mean needs 168 hourly observations before it has a complete value. Early rows are therefore dropped.

## 6. Model evaluation in plain English

The project compares:

- **Naive baseline:** predicts a recent average range. A complex model should be compared with something simple.
- **Ridge regression:** a linear model with regularization. Its pipeline scales features first.
- **Histogram-based gradient boosting:** a tree-based model that can learn non-linear relationships.

Chronological folds train on earlier rows and evaluate on later rows. This resembles real forecasting better than randomly mixing past and future observations.

- **MAE (Mean Absolute Error):** average absolute distance between predictions and targets. Lower is better.
- **RMSE (Root Mean Squared Error):** similar to MAE but penalizes large errors more. Lower is better.
- **R²:** compares the model with predicting the average target. Near 1 is strong, around 0 is similar to that baseline, and below 0 is worse on the evaluated data.

The saved metrics describe one experiment, not guaranteed future performance. Markets change, and data windows and validation methods matter.

## 7. Run locally

### Option A — API and database with Docker

Prerequisite: Docker with the Compose plugin.

~~~bash
cp .env.example .env
docker compose up --build
~~~

Open the API docs at http://localhost:8000/docs and health endpoint at http://localhost:8000/health.

Compose starts PostgreSQL and waits for its health check before starting the API. The API container connects using hostname `db`. Inside a container, localhost means that same container, not the database container.

Docker starts the API with the model artifact already in artifacts/. It does not automatically download data or retrain the model.

### Option B — Train a new model

Use a Python environment with the project dependencies installed:

~~~bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e ".[dev]"
python -m src.train
~~~

Training downloads candles from Binance and requires internet access. It may take time. It writes the model, feature list, and metrics into artifacts/.

If you train on your host while Docker is running, Compose maps the local artifacts directory into the container. Restart the API after replacing the model so it loads the new artifact.

### Health request

~~~bash
curl http://localhost:8000/health
~~~

A successful response means the HTTP app responds. Check model_loaded too: a healthy process does not necessarily mean a model is available.

### Prediction request

The useful route is POST /predict-history, which requires at least 200 hourly candles. Each candle must contain the fields defined by Candle in src/schemas.py. The response contains the predicted range percentage, model name, and timestamp. Try the route interactively using Swagger UI at http://localhost:8000/docs.

The POST /predict single-candle route cannot build the rolling history needed by the feature pipeline, so it returns a validation error rather than inventing a forecast. Use /predict-history until a real history store or streaming feature state is implemented. This is a current limitation, not a claim that the one-candle endpoint is fully useful.

## 8. Database basics

SQLAlchemy is the Python layer used to describe and interact with PostgreSQL.

- **engine:** database connection configuration and connection management.
- **SessionLocal:** a factory that creates a session for a unit of database work.
- **Base:** parent class for table definitions.
- **Experiment:** describes model evaluation results. The current training script writes metrics to JSON; it does not insert experiment rows.
- **Prediction:** stores a prediction, timestamp, model name, and input request as JSON text.
- **init_db():** creates missing tables when the application starts.

A session is not the database itself. Think of it as a short-lived workspace for database operations. The API adds a row and commits it so the record is persisted.

For a production-grade application, replace automatic table creation with versioned migrations, use secure database credentials, add indexes and retention rules, and consider whether request payloads should be stored.

## 9. Tests and code quality

Run checks locally:

~~~bash
ruff check .
pytest -q
docker build -t btc-volatility-forecast:local .
~~~

Feature tests use generated sample data, so they do not download market history. They check that feature generation produces usable rows and that a target exists. The API test checks the health endpoint.

**What the current tests do not prove:** they do not verify prediction accuracy, every invalid input, PostgreSQL persistence, model loading, or the complete Docker Compose flow. Tests are executable checks of specific behavior, not proof that every part of the application is correct.

GitHub Actions runs linting, tests, and a Docker image build on pushes to main and pull requests.

## 10. Environment and deployment

Settings live in src/config.py and can be overridden with environment variables. .env.example documents local settings. Never commit real passwords, API keys, or production credentials; use your host's environment/secret settings.

- **Dockerfile:** builds the API image.
- **docker-compose.yml:** runs the API and PostgreSQL for local development.
- **render.yaml:** describes a deployment service and expects DATABASE_URL to be configured separately.

A deployment blueprint is not the same as a verified production deployment. Before calling this production-ready, add secure secrets, migrations, stronger tests, logging, monitoring, request-size limits, and model versioning/retraining procedures.

## 11. Current limitations worth understanding

These are learning tasks, not reasons to add libraries blindly:

1. **Single-candle prediction:** rolling features require history, so use /predict-history. A future version could fetch recent candles or maintain feature state.
2. **Input validation:** basic numeric bounds exist, but candle consistency, sensible timestamps, and maximum history size need stronger checks.
3. **Training reproducibility:** record dataset details, library versions, and training timestamp with each model.
4. **Experiment persistence:** Experiment exists as a table model, but training writes metrics to JSON instead of PostgreSQL.
5. **Database schema changes:** create_all creates missing tables but does not manage changes to existing tables. Add Alembic migrations as the schema evolves.
6. **Model/API compatibility:** validate that the saved model's feature names match the API feature list on startup.
7. **Production operations:** add structured logs, error handling, authentication/rate limiting if needed, database indexes, and monitoring.
8. **Time-series rigor:** consider incomplete candles, data gaps, repeated downloads, and hyperparameter selection.
9. **Security:** limit history request size and avoid storing sensitive input data without a reason.

Fix these one at a time and write a test for each behavior.

## 12. Exercises: learn by changing code

Try these in order. Write down what you expect before searching for an answer.

### Beginner
- Change the API title in configuration and observe the docs.
- Add a field to the response schema and return it from the route.
- Explain each step in make_features in your own words.
- Write a test that passes unsorted timestamps and verifies output is sorted.
- Write a test for a rolling feature after enough rows exist.

### Intermediate
- Validate that high is not below low and that high/low contain open and close.
- Add a maximum candle count to /predict-history.
- Test the 199-candle error and a valid 200+ candle request.
- Test that the API returns 503 when no model is loaded.
- Store experiment metrics in PostgreSQL instead of only JSON.
- Add migrations and test them against a fresh database.

### Advanced
- Add a model version to artifacts and save it with each prediction.
- Prevent incomplete hourly candles from entering training.
- Compare with a stronger time-series baseline and report scores for every fold.
- Add structured logging without exposing secrets.
- Test API and database behavior together with an isolated test database.
- Document retraining, rollback, and model promotion.

## 13. Recommended reading

Read the official documentation in this order. Focus on concepts used here instead of trying to read every page.

1. [Python tutorial](https://docs.python.org/3/tutorial/) — functions, lists, dictionaries, modules, exceptions, and classes.
2. [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/) — routes, request bodies, response models, errors, and API docs.
3. [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/) — request validation.
4. [Pydantic settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — environment-based configuration.
5. [Pandas tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/) — DataFrames, sorting, missing data, and rolling calculations.
6. [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) — metrics and validation.
7. [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) — chronological cross-validation.
8. [SQLAlchemy ORM quick start](https://docs.sqlalchemy.org/en/20/orm/quickstart.html) — engines, sessions, models, and persistence.
9. [Docker getting started](https://docs.docker.com/get-started/) and [Compose](https://docs.docker.com/compose/) — images, containers, services, volumes, and health checks.
10. [pytest getting started](https://docs.pytest.org/en/stable/getting-started.html) — assertions and test organization.
11. [GitHub Actions quickstart](https://docs.github.com/en/actions/writing-workflows/quickstart) — automated checks on repository changes.

## 14. A practical study method

For each file:

1. Read the matching README section.
2. Read the code and comments once without changing anything.
3. Write a short explanation in your own words.
4. Trace one example through the code.
5. Change one small thing.
6. Add or update a test.
7. Run the checks and inspect the diff.
8. If you cannot explain why a line exists, pause and investigate before copying more code.

The goal is not to memorize this project. It is to rebuild a smaller version yourself and explain why each layer exists.

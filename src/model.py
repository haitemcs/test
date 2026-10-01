from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    r2: float

def make_models():
    models = {}

    models["ridge"] = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    models["hist_gradient_boosting"] = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    return models

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

def naive_prediction(train_y, n):
    last_24_values = train_y[-24:]
    average = float(np.mean(last_24_values))

    return np.repeat(average, n)

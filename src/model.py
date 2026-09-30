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
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42
        ),
    }

def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

def naive_prediction(train_y, n):
    return np.repeat(float(np.mean(train_y[-24:])), n)

"""Model definitions and evaluation helpers used by the training workflow."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    """Group a model name with the scores measured for it."""

    name: str
    mae: float
    rmse: float
    r2: float


def make_models():
    """Create fresh model instances so candidates can be compared fairly."""
    models = {}

    # Ridge is sensitive to feature scales, so standardize features before fitting.
    # A pipeline keeps the scaler and regression model together for later prediction.
    models["ridge"] = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    # Gradient boosting builds small decision trees in sequence. These settings limit
    # model complexity; random_state makes its randomness repeatable.
    models["hist_gradient_boosting"] = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    return models


def metrics(y_true, y_pred):
    """Calculate common regression metrics from actual and predicted values."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Convert NumPy numeric values to normal Python floats for JSON serialization.
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def naive_prediction(train_y, n):
    """Use the mean of the last 24 training targets as a simple baseline."""
    last_24_values = train_y[-24:]
    average = float(np.mean(last_24_values))

    # Return the same baseline value for each row in the validation fold.
    return np.repeat(average, n)

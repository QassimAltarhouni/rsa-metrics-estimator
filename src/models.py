import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import numpy as np


def train_and_select(X, y, seed=42):
    """
    Train candidate models and select best one
    """
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1
        )
    }

    best_model = None
    best_score = float("inf")

    for name, model in models.items():
        scores = -cross_val_score(
            model,
            X,
            y,
            scoring="neg_mean_absolute_error",
            cv=3,
            n_jobs=-1
        )
        score = scores.mean()

        if score < best_score:
            best_score = score
            best_model = model

    best_model.fit(X, y)
    return best_model


def predict_with_bundle(bundle, X):
    """
    Run inference using saved model bundle
    """
    model = bundle["model"]
    return model.predict(X)

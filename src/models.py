# path: src/models.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import LOG1P_TARGETS, TARGETS
from src.math_utils import mean_absolute_percentage_error_safe, weighted_absolute_percentage_error


def _split_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build X, y from dataframe:
      - y: TARGETS
      - X: all numeric columns except TARGETS
      - drop common meta columns if present
    """
    df = df.copy()

    y_df = df[TARGETS].astype(float)

    drop_cols = set(TARGETS)
    for meta in ("topology", "request_set"):
        if meta in df.columns:
            drop_cols.add(meta)

    x_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # keep numeric only
    x_df = x_df.select_dtypes(include=[np.number]).fillna(0.0)

    if x_df.shape[1] == 0:
        raise RuntimeError("No numeric feature columns found after dropping targets/meta.")

    return x_df.to_numpy(dtype=float), y_df.to_numpy(dtype=float), list(x_df.columns)


def _transform_targets(y: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    idx = [i for i, t in enumerate(TARGETS) if t in LOG1P_TARGETS]
    y_t = y.copy()
    if idx:
        y_t[:, idx] = np.log1p(np.maximum(y_t[:, idx], 0.0))
    return y_t, idx


def _inverse_transform_targets(y_pred: np.ndarray, idx: List[int]) -> np.ndarray:
    y2 = y_pred.copy()
    if idx:
        y2[:, idx] = np.expm1(y2[:, idx])
    return y2


def _metrics_row(model: str, target: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = mean_absolute_percentage_error_safe(y_true, y_pred)
    wape = weighted_absolute_percentage_error(y_true, y_pred)
    return {"model": model, "target": target, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "WAPE": wape}


def train_and_select(
    samples: pd.DataFrame,
    outdir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train model and write:
      - metrics_all.csv
      - metrics_best.csv
      - bundle.joblib
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, feature_columns = _split_xy(samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(seed), shuffle=True
    )

    y_train_t, t_idx = _transform_targets(y_train)

    model_name = "RandomForest"
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(X_train, y_train_t)

    y_pred_t = np.asarray(model.predict(X_test), dtype=float)
    y_pred = _inverse_transform_targets(y_pred_t, t_idx)

    rows: List[Dict[str, Any]] = []
    for i, t in enumerate(TARGETS):
        rows.append(_metrics_row(model_name, t, y_test[:, i], y_pred[:, i]))

    df_metrics = pd.DataFrame(rows).sort_values(["target"]).reset_index(drop=True)

    # only one model => best == itself
    df_best = df_metrics.copy()

    (outdir / "metrics_all.csv").write_text(df_metrics.to_csv(index=False), encoding="utf-8")
    (outdir / "metrics_best.csv").write_text(df_best.to_csv(index=False), encoding="utf-8")

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "targets": TARGETS,
        "log1p_targets": sorted(LOG1P_TARGETS),
        "transform_idx": t_idx,
        "seed": seed,
    }
    joblib.dump(bundle, outdir / "bundle.joblib")

    print("\n=== Metrics (all) ===")
    print(df_metrics.to_string(index=False))
    print(f"\nSaved: {outdir / 'metrics_all.csv'}")
    print(f"Saved: {outdir / 'metrics_best.csv'}")
    print(f"Saved: {outdir / 'bundle.joblib'}")

    return bundle


def predict_with_bundle(bundle: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """
    Run inference using saved model bundle.
    X can be a DataFrame with at least bundle['feature_columns'].
    """
    feature_columns: List[str] = list(bundle["feature_columns"])
    x_df = X.reindex(columns=feature_columns).fillna(0.0)
    x = x_df.to_numpy(dtype=float)

    y_pred_t = np.asarray(bundle["model"].predict(x), dtype=float)
    return _inverse_transform_targets(y_pred_t, list(bundle["transform_idx"]))

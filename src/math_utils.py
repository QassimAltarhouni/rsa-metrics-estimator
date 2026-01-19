from __future__ import annotations

import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across numpy & random."""
    random.seed(seed)
    np.random.seed(seed)


def safe_skew(x: np.ndarray) -> float:
    """Population skewness (3rd standardized moment)."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    mu = float(x.mean())
    s = float(x.std(ddof=0))
    if s == 0.0:
        return 0.0
    z = (x - mu) / s
    return float(np.mean(z**3))


def safe_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (4th standardized moment - 3)."""
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return 0.0
    mu = float(x.mean())
    s = float(x.std(ddof=0))
    if s == 0.0:
        return 0.0
    z = (x - mu) / s
    return float(np.mean(z**4) - 3.0)


def gini(x: np.ndarray) -> float:
    """Gini coefficient for non-negative values."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    x = np.clip(x, 0, None)
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def entropy_from_counts(c: np.ndarray) -> float:
    """Shannon entropy (base-2) from counts."""
    c = np.asarray(c, dtype=float)
    total = float(c.sum())
    if total <= 0.0:
        return 0.0
    p = c / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ---- ADDED: MAPE ----
def mean_absolute_percentage_error_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    eps: float = 1e-9,
) -> float:
    """
    MAPE (%) with safe denominator clipping for zeros/near-zeros.

    mean(|y - yhat| / max(|y|, eps)) * 100
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


# ---- ADDED: WAPE ----
def weighted_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """
    WAPE (%) = sum(|y - yhat|) / max(sum(|y|), eps) * 100
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    num = float(np.sum(np.abs(yt - yp)))
    den = float(np.sum(np.abs(yt)))
    if den <= eps:
        return 0.0 if num <= eps else float("inf")
    return float((num / den) * 100.0)




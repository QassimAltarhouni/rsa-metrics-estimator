import numpy as np

def safe_skew(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    s = x.std()
    return 0.0 if s == 0 else float(((x - x.mean()) / s).mean() ** 3)

def safe_kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return 0.0
    s = x.std()
    return 0.0 if s == 0 else float(((x - x.mean()) / s).mean() ** 4 - 3)

def gini(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    x = np.sort(np.clip(x, 0, None))
    n = x.size
    return float((n + 1 - 2 * (x.cumsum() / x.sum()).sum()) / n)

def entropy_from_counts(c: np.ndarray) -> float:
    if c.sum() == 0:
        return 0.0
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
import random
import numpy as np


def set_global_seed(seed: int = 42):
    """
    Set seed for reproducibility across numpy & random
    """
    random.seed(seed)
    np.random.seed(seed)

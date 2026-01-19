from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import TARGETS


def load_dataset(root: Path, topology: str, seed: int = 42) -> pd.DataFrame:
    """
    Load training dataset from features_full.csv.
    Expected: features_full.csv contains feature columns + TARGETS columns.
    """
    topo_dir = Path(root) / topology
    features_path = topo_dir / "features_full.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    df = pd.read_csv(features_path)
    if df.empty:
        raise RuntimeError("features_full.csv is empty")

    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise RuntimeError(f"Missing target columns in features_full.csv: {missing}")

    return df


from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import TARGETS


def load_dataset(root: Path, topology: str, seed: int = 42) -> pd.DataFrame:
    """
    Load dataset from features_full.csv (must include TARGETS columns).
    """
    topo_dir = Path(root) / topology
    features_path = topo_dir / "features_full.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing file: {features_path}")

    df = pd.read_csv(features_path)
    if df.empty:
        raise RuntimeError(f"{features_path} is empty")

    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise RuntimeError(f"Missing TARGET columns in features_full.csv: {missing}")

    return df


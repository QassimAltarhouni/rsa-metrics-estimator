from pathlib import Path
import pandas as pd


def load_dataset(root: Path, topology: str, seed: int = 42):
    topo_dir = root / topology

    features_path = topo_dir / "features_full.csv"
    metrics_path = topo_dir / "metrics_all.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    X = pd.read_csv(features_path)
    y = pd.read_csv(metrics_path)

    if X.empty or y.empty:
        raise RuntimeError("Features or metrics CSV is empty")

    # Combine for convenience
    dataset = pd.concat([X, y], axis=1)

    print(f"Loaded dataset: {dataset.shape}")

    return dataset

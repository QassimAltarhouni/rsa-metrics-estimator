from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import load_dataset
from src.math_utils import set_global_seed
from src.models import train_and_select


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RSA Metrics Estimator")

    parser.add_argument("--zip", type=str, required=True,
                        help="Dummy zip path (not used, kept for compatibility)")
    parser.add_argument("--workdir", type=str, required=True,
                        help="Root project directory (must contain Data/)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology", type=str, default="Euro28")

    args = parser.parse_args()

    set_global_seed(args.seed)

    root = Path(args.workdir).resolve()
    data_root = root / "Data"
    if not data_root.exists():
        raise FileNotFoundError(f"'Data' folder not found inside {root}")

    topology = args.topology
    print(f"\n===== TRAINING TOPOLOGY: {topology} =====")

    samples = load_dataset(root=data_root, topology=topology, seed=args.seed)
    if samples.empty:
        raise RuntimeError("Dataset is empty. No samples were loaded.")

    outdir = Path(args.outdir).resolve() / topology
    train_and_select(samples=samples, outdir=outdir, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    main()

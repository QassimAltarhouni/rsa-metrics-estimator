import argparse
from pathlib import Path

from src.dataset import load_dataset
from src.models import train_and_select
from src.math_utils import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Train RSA Metrics Estimator")

    parser.add_argument("--zip", type=str, required=True,
                        help="Dummy zip path (not used, kept for compatibility)")
    parser.add_argument("--workdir", type=str, required=True,
                        help="Root project directory (must contain Data/)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 🔐 Set seed
    set_global_seed(args.seed)

    # ✅ CRITICAL FIX
    root = Path(args.workdir).resolve()
    data_root = root / "Data"

    if not data_root.exists():
        raise FileNotFoundError(f"'Data' folder not found inside {root}")

    topology = "Euro28"   # <-- YOU HARD-CODED THIS IN DATA STRUCTURE

    print(f"\n===== TRAINING TOPOLOGY: {topology} =====")

    #samples = load_dataset(
     #   root=data_root,
      #  topology=topology,
       # seed=args.seed

    samples = load_dataset(
        root=Path("Data"),
        topology="Euro28",
        seed=args.seed
    )


    if samples.empty:
        raise RuntimeError("Dataset is empty. No samples were loaded.")

    train_and_select(
        samples=samples,
        outdir=Path(args.outdir),
        test_size=args.test_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

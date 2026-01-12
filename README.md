# RSA Metrics Estimator

Train regression models that estimate RSA metrics (e.g., highest slot, sum of slots) from
demand requests in the `RSA_estimation` dataset.

## Requirements

- Python 3.9+
- Packages: `numpy`, `pandas`, `scikit-learn`, `networkx`, `joblib`

Install dependencies in your environment (example):

```bash
pip install numpy pandas scikit-learn networkx joblib
```

## Data layout

The script expects a `RSA_estimation/` folder extracted from the provided zip. It can be
nested under the `--workdir` directory; the script searches for it automatically.
If you already extracted the dataset, you can create a `.extracted_ok` marker file in the
work directory to skip extraction.

## Usage

### Train models

```powershell
# Optional: mark an already-extracted dataset
New-Item -ItemType File -Path .\.extracted_ok -Force

# Train
python .\train_rsa_estimator.py --zip .\dummy.zip --workdir . --outdir .\_out --test_size 0.2 --seed 42
```

### Train and predict from a requests CSV

```powershell
python .\train_rsa_estimator.py \
  --zip .\dummy.zip \
  --workdir . \
  --outdir .\_out \
  --predict_requests .\some_requests.csv
```

## Outputs

For each topology (e.g., `Euro28`, `US26`) the script writes to `--outdir/<topology>`:

- `bundle_<topology>.joblib`: best-per-target models plus feature transforms
- `metrics_all.csv`: metrics for all models and targets
- `metrics_best.csv`: best model per target
- `preds_test.csv`: test split predictions
- `features_full.csv`: features + targets for debugging
- `best_models.json`: mapping of each target to its best model
- `predict_<topology>_<timestamp>.csv`: predictions (only when `--predict_requests` is set)

## Notes

- `--zip` is required even if the data is already extracted; any placeholder file path
  is acceptable if `RSA_estimation/` is present under `--workdir`.
- Default output directory is `./_out_v3` if `--outdir` is not provided.
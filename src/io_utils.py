from pathlib import Path
import zipfile


def ensure_extracted(zip_path, workdir):
    """
    Ensure ZIP dataset is extracted only once
    """
    zip_path = Path(zip_path)
    workdir = Path(workdir)
    marker = workdir / ".extracted_ok"

    if marker.exists():
        return

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(workdir)

    marker.touch()


def parse_results_txt(results_path):
    """
    Parse RSA results.txt file and return metrics as dictionary
    """
    results_path = Path(results_path)

    metrics = {}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            key, value = line.strip().split("\t")
            metrics[key] = float(value)

    return metrics

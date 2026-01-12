# path: train_rsa_estimator_v3.py
"""
V3: 3 algorithms + pick best per target + prediction bundle.

Run training:
  python .\train_rsa_estimator_v3.py --zip .\dummy.zip --workdir . --outdir .\_out_v3 --test_size 0.2 --seed 42

Train + predict on a requests.csv:
  python .\train_rsa_estimator_v3.py --zip .\dummy.zip --workdir . --outdir .\_out_v3 --predict_requests .\some_requests.csv

Outputs (per topology):
  - bundle_<topology>.joblib           (best model per target + columns + transforms)
  - metrics_all.csv                    (all models x targets)
  - metrics_best.csv                   (best-per-target summary)
  - preds_test.csv                     (test split predictions)
  - features_full.csv                  (features + targets for debugging)
  - predict_<timestamp>.csv            (if --predict_requests given)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any, Optional

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.decomposition import NMF
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

TARGETS = ["highestSlot", "avgHighestSlot", "sumOfSlots", "avgActiveTransceivers"]
LOG1P_TARGETS = {"highestSlot", "avgHighestSlot", "sumOfSlots"}  # slot-like


# -----------------------------
# Repro
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------
# IO / extraction
# -----------------------------
def _find_rsa_estimation_dir(workdir: Path) -> Optional[Path]:
    direct = workdir / "RSA_estimation"
    if direct.exists():
        return direct

    # common: extracted repo root contains RSA_estimation/
    for p in workdir.rglob("RSA_estimation"):
        if p.is_dir():
            return p
    return None


def ensure_extracted(zip_path: Path, workdir: Path) -> Path:
    """
    Extract zip if needed, then find RSA_estimation folder (even if nested).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    marker = workdir / ".extracted_ok"

    found = _find_rsa_estimation_dir(workdir)
    if found is not None and marker.exists():
        return found
    if found is not None and not marker.exists():
        # accept already-extracted state
        return found

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(workdir)

    marker.write_text("ok", encoding="utf-8")

    found = _find_rsa_estimation_dir(workdir)
    if found is None:
        raise FileNotFoundError(f"Could not find RSA_estimation under {workdir}")
    return found


def parse_results_txt(results_path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        k, v = line.split("\t")
        out[k.strip()] = float(v.strip())
    missing = [t for t in TARGETS if t not in out]
    if missing:
        raise ValueError(f"Missing targets {missing} in {results_path}")
    return out


# -----------------------------
# Math helpers
# -----------------------------
def safe_skew(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    mu = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 3))


def safe_kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return 0.0
    mu = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 4) - 3.0)


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    x = np.clip(x, 0, None)
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def quantiles(x: np.ndarray, qs: Iterable[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {f"q{int(q*100):02d}": 0.0 for q in qs}
    vals = np.quantile(x, list(qs))
    return {f"q{int(q*100):02d}": float(v) for q, v in zip(qs, vals)}


# -----------------------------
# Graph build
# -----------------------------
def build_request_graph(req: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for s, d, b in req[["source", "destination", "bitrate"]].itertuples(index=False):
        s_i = int(s)
        d_i = int(d)
        b_f = float(b)
        if g.has_edge(s_i, d_i):
            g[s_i][d_i]["w_sum"] += b_f
            g[s_i][d_i]["cnt"] += 1
        else:
            g.add_edge(s_i, d_i, w_sum=b_f, cnt=1)
    return g


def adjacency_eigenspectrum_features(req: pd.DataFrame, k: int = 12) -> Dict[str, float]:
    nodes = np.unique(np.concatenate([req["source"].values, req["destination"].values]))
    nodes = np.asarray(nodes, dtype=int)
    n = nodes.size
    if n == 0:
        return {f"eig_{i}": 0.0 for i in range(k)}

    idx = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((n, n), dtype=float)
    for s, d, b in req[["source", "destination", "bitrate"]].itertuples(index=False):
        A[idx[int(s)], idx[int(d)]] += float(b)

    S = (A + A.T) / 2.0
    eigvals = np.linalg.eigvalsh(S)
    eigvals = np.sort(np.abs(eigvals))[::-1][:k]
    if eigvals.size < k:
        eigvals = np.pad(eigvals, (0, k - eigvals.size), constant_values=0.0)
    return {f"eig_{i}": float(eigvals[i]) for i in range(k)}


# -----------------------------
# V2/V3 graph features
# -----------------------------
def hashed_wl_features(g: nx.DiGraph, n_bins: int = 128, iters: int = 2) -> Dict[str, float]:
    if g.number_of_nodes() == 0:
        return {f"wl_{i}": 0.0 for i in range(n_bins)}

    in_deg = dict(g.in_degree())
    out_deg = dict(g.out_degree())
    in_str = dict(g.in_degree(weight="w_sum"))
    out_str = dict(g.out_degree(weight="w_sum"))

    labels: Dict[int, str] = {}
    for v in g.nodes():
        labels[int(v)] = f"{in_deg[v]}|{out_deg[v]}|{int(round(in_str[v]))}|{int(round(out_str[v]))}"

    bins = np.zeros(n_bins, dtype=float)

    def _bin(h: str) -> int:
        digest = hashlib.md5(h.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % n_bins

    for lab in labels.values():
        bins[_bin(lab)] += 1.0

    for _ in range(iters):
        new_labels: Dict[int, str] = {}
        for v in g.nodes():
            neigh = sorted(labels[int(u)] for u in g.predecessors(v)) + sorted(
                labels[int(u)] for u in g.successors(v)
            )
            joined = labels[int(v)] + "||" + "|".join(neigh)
            digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
            new_labels[int(v)] = digest
        labels = new_labels
        for lab in labels.values():
            bins[_bin(lab)] += 1.0

    bins = bins / max(1.0, float(g.number_of_nodes()))
    return {f"wl_{i}": float(bins[i]) for i in range(n_bins)}


def build_demand_matrix(req: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    nodes = np.unique(np.concatenate([req["source"].values, req["destination"].values]))
    nodes = np.asarray(nodes, dtype=int)
    n = nodes.size
    if n == 0:
        return np.zeros((0, 0), dtype=float), nodes

    idx = {node: i for i, node in enumerate(nodes)}
    M = np.zeros((n, n), dtype=float)
    for s, d, b in req[["source", "destination", "bitrate"]].itertuples(index=False):
        M[idx[int(s)], idx[int(d)]] += float(b)
    return M, nodes


def demand_matrix_svd_features_from_matrix(M: np.ndarray, k: int = 10) -> Dict[str, float]:
    n = M.shape[0]
    if n == 0:
        return {f"svd_{i}": 0.0 for i in range(k)}

    svals = np.linalg.svd(M, compute_uv=False)[:k]
    if svals.size < k:
        svals = np.pad(svals, (0, k - svals.size), constant_values=0.0)

    total = float(M.sum())
    if total > 0:
        svals = svals / total
    return {f"svd_{i}": float(svals[i]) for i in range(k)}


def demand_matrix_stats_features(M: np.ndarray) -> Dict[str, float]:
    if M.size == 0:
        return {
            "dm_sparsity": 1.0,
            "dm_total": 0.0,
            "dm_row_mean": 0.0,
            "dm_row_std": 0.0,
            "dm_row_max": 0.0,
            "dm_row_q90": 0.0,
            "dm_col_mean": 0.0,
            "dm_col_std": 0.0,
            "dm_col_max": 0.0,
            "dm_col_q90": 0.0,
            "dm_top_row_share": 0.0,
            "dm_top_col_share": 0.0,
            "dm_row_entropy": 0.0,
            "dm_col_entropy": 0.0,
            "dm_row_gini": 0.0,
            "dm_col_gini": 0.0,
        }

    total = float(M.sum())
    row_sum = M.sum(axis=1)
    col_sum = M.sum(axis=0)

    sparsity = float(np.mean(M == 0.0))
    row_q90 = float(np.quantile(row_sum, 0.90)) if row_sum.size else 0.0
    col_q90 = float(np.quantile(col_sum, 0.90)) if col_sum.size else 0.0

    top_row_share = float(row_sum.max() / total) if total > 0 else 0.0
    top_col_share = float(col_sum.max() / total) if total > 0 else 0.0

    return {
        "dm_sparsity": sparsity,
        "dm_total": total,
        "dm_row_mean": float(row_sum.mean()) if row_sum.size else 0.0,
        "dm_row_std": float(row_sum.std(ddof=0)) if row_sum.size else 0.0,
        "dm_row_max": float(row_sum.max()) if row_sum.size else 0.0,
        "dm_row_q90": row_q90,
        "dm_col_mean": float(col_sum.mean()) if col_sum.size else 0.0,
        "dm_col_std": float(col_sum.std(ddof=0)) if col_sum.size else 0.0,
        "dm_col_max": float(col_sum.max()) if col_sum.size else 0.0,
        "dm_col_q90": col_q90,
        "dm_top_row_share": top_row_share,
        "dm_top_col_share": top_col_share,
        "dm_row_entropy": entropy_from_counts(row_sum),
        "dm_col_entropy": entropy_from_counts(col_sum),
        "dm_row_gini": gini(row_sum),
        "dm_col_gini": gini(col_sum),
    }


def demand_matrix_nmf_features(M: np.ndarray, k: int = 6, seed: int = 42) -> Dict[str, float]:
    if M.size == 0 or k <= 0:
        return {f"nmf_{i}": 0.0 for i in range(max(k, 0))}

    n = M.shape[0]
    k_eff = min(k, max(1, n - 1))
    if k_eff <= 0:
        return {f"nmf_{i}": 0.0 for i in range(k)}

    try:
        model = NMF(n_components=k_eff, init="nndsvda", random_state=seed, max_iter=300)
        W = model.fit_transform(M)
        H = model.components_
        comp_strength = np.sum(W, axis=0) + np.sum(H, axis=1)
        comp_strength = comp_strength / max(1.0, comp_strength.sum())
        if comp_strength.size < k:
            comp_strength = np.pad(comp_strength, (0, k - comp_strength.size), constant_values=0.0)
        return {f"nmf_{i}": float(comp_strength[i]) for i in range(k)}
    except Exception:
        return {f"nmf_{i}": 0.0 for i in range(k)}


def bitrate_hist_features(
    bitrate: np.ndarray,
    bins: Tuple[float, ...] = (0, 50, 100, 150, 200, 400, 800, 1600),
) -> Dict[str, float]:
    if bitrate.size == 0:
        return {f"bh_{i}": 0.0 for i in range(len(bins) - 1)}
    hist, _ = np.histogram(bitrate, bins=np.array(bins, dtype=float))
    hist = hist.astype(float) / float(bitrate.size)
    return {f"bh_{i}": float(hist[i]) for i in range(hist.size)}


def topk_edge_weight_features(g: nx.DiGraph, k: int = 20) -> Dict[str, float]:
    w = [float(data.get("w_sum", 0.0)) for _, _, data in g.edges(data=True)]
    if not w:
        return {f"ew_{i}": 0.0 for i in range(k)}
    w = np.sort(np.array(w, dtype=float))[::-1][:k]
    if w.size < k:
        w = np.pad(w, (0, k - w.size), constant_values=0.0)
    total = float(np.sum(w))
    if total > 0:
        w = w / total
    return {f"ew_{i}": float(w[i]) for i in range(k)}


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(
    req: pd.DataFrame,
    eig_k: int = 24,
    wl_bins: int = 128,
    nmf_k: int = 6,
    seed: int = 42,
) -> Dict[str, float]:
    n_req = int(len(req))
    nodes = np.unique(np.concatenate([req["source"].values, req["destination"].values]))
    n_nodes = int(nodes.size)

    bitrate = req["bitrate"].to_numpy(dtype=float)

    feats: Dict[str, float] = {
        "n_requests": float(n_req),
        "n_nodes_in_requests": float(n_nodes),
        "bitrate_mean": float(bitrate.mean()) if n_req else 0.0,
        "bitrate_std": float(bitrate.std(ddof=0)) if n_req else 0.0,
        "bitrate_min": float(bitrate.min()) if n_req else 0.0,
        "bitrate_max": float(bitrate.max()) if n_req else 0.0,
        "bitrate_skew": safe_skew(bitrate) if n_req else 0.0,
        "bitrate_kurt": safe_kurtosis(bitrate) if n_req else 0.0,
        "bitrate_gini": gini(bitrate) if n_req else 0.0,
    }
    feats.update(quantiles(bitrate, qs=[0.05, 0.25, 0.50, 0.75, 0.95]))
    feats.update(bitrate_hist_features(bitrate))

    pairs = list(zip(req["source"].astype(int), req["destination"].astype(int)))
    feats["unique_pair_ratio"] = float(len(set(pairs)) / n_req) if n_req else 0.0

    feats["src_entropy"] = entropy_from_counts(req["source"].value_counts().to_numpy())
    feats["dst_entropy"] = entropy_from_counts(req["destination"].value_counts().to_numpy())

    g = build_request_graph(req)
    m = g.number_of_edges()

    feats["g_edges"] = float(m)
    feats["g_density"] = float(nx.density(g)) if n_nodes > 1 else 0.0
    feats["g_reciprocity"] = float(nx.reciprocity(g)) if m > 0 else 0.0

    in_deg = np.array([d for _, d in g.in_degree()], dtype=float) if n_nodes else np.array([])
    out_deg = np.array([d for _, d in g.out_degree()], dtype=float) if n_nodes else np.array([])
    in_str = np.array([s for _, s in g.in_degree(weight="w_sum")], dtype=float) if n_nodes else np.array([])
    out_str = np.array([s for _, s in g.out_degree(weight="w_sum")], dtype=float) if n_nodes else np.array([])

    def stats_block(name: str, x: np.ndarray) -> Dict[str, float]:
        if x.size == 0:
            return {f"{name}_{k}": 0.0 for k in ["mean", "std", "max", "q50", "q90"]}
        return {
            f"{name}_mean": float(x.mean()),
            f"{name}_std": float(x.std(ddof=0)),
            f"{name}_max": float(x.max()),
            f"{name}_q50": float(np.quantile(x, 0.50)),
            f"{name}_q90": float(np.quantile(x, 0.90)),
        }

    feats.update(stats_block("in_deg", in_deg))
    feats.update(stats_block("out_deg", out_deg))
    feats.update(stats_block("in_str", in_str))
    feats.update(stats_block("out_str", out_str))

    if m > 0:
        gu = g.to_undirected()
        try:
            feats["g_transitivity"] = float(nx.transitivity(gu))
        except Exception:
            feats["g_transitivity"] = 0.0
        try:
            feats["g_avg_clustering"] = float(nx.average_clustering(gu, weight=None))
        except Exception:
            feats["g_avg_clustering"] = 0.0
        try:
            feats["g_degree_assort"] = float(nx.degree_assortativity_coefficient(gu))
        except Exception:
            feats["g_degree_assort"] = 0.0
        try:
            pr = nx.pagerank(g, weight="w_sum")
            pr_vals = np.array(list(pr.values()), dtype=float)
            feats["pr_mean"] = float(pr_vals.mean())
            feats["pr_std"] = float(pr_vals.std(ddof=0))
            feats["pr_max"] = float(pr_vals.max())
        except Exception:
            feats["pr_mean"] = 0.0
            feats["pr_std"] = 0.0
            feats["pr_max"] = 0.0
    else:
        feats["g_transitivity"] = 0.0
        feats["g_avg_clustering"] = 0.0
        feats["g_degree_assort"] = 0.0
        feats["pr_mean"] = 0.0
        feats["pr_std"] = 0.0
        feats["pr_max"] = 0.0

    feats.update(adjacency_eigenspectrum_features(req, k=eig_k))

    demand_matrix, _ = build_demand_matrix(req)
    feats.update(demand_matrix_svd_features_from_matrix(demand_matrix, k=10))
    feats.update(demand_matrix_stats_features(demand_matrix))
    feats.update(demand_matrix_nmf_features(demand_matrix, k=nmf_k, seed=seed))

    feats.update(topk_edge_weight_features(g, k=20))
    feats.update(hashed_wl_features(g, n_bins=wl_bins, iters=2))
    return feats


# -----------------------------
# Dataset
# -----------------------------
@dataclass(frozen=True)
class Sample:
    topology: str
    request_set: str
    features: Dict[str, float]
    targets: Dict[str, float]


def load_dataset(
    root: Path,
    topology: str,
    eig_k: int,
    wl_bins: int,
    nmf_k: int,
    seed: int,
) -> List[Sample]:
    topo_dir = root / topology
    if not topo_dir.exists():
        raise FileNotFoundError(f"Topology folder not found: {topo_dir}")

    samples: List[Sample] = []
    for req_dir in sorted(topo_dir.glob("request-set_*")):
        req_csv = req_dir / "requests.csv"
        res_txt = req_dir / "results.txt"
        if not req_csv.exists() or not res_txt.exists():
            continue

        req = pd.read_csv(req_csv)
        req["source"] = req["source"].astype(int)
        req["destination"] = req["destination"].astype(int)
        req["bitrate"] = req["bitrate"].astype(float)

        feats = extract_features(req, eig_k=eig_k, wl_bins=wl_bins, nmf_k=nmf_k, seed=seed)
        targs = parse_results_txt(res_txt)

        samples.append(Sample(topology=topology, request_set=req_dir.name, features=feats, targets=targs))

    if not samples:
        raise ValueError(f"No samples found for topology {topology} in {topo_dir}")
    return samples


# -----------------------------
# Target transforms
# -----------------------------
def transform_targets(y: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    y2 = y.copy()
    idx = [i for i, t in enumerate(TARGETS) if t in LOG1P_TARGETS]
    if idx:
        y2[:, idx] = np.log1p(np.maximum(y2[:, idx], 0.0))
    return y2, idx


def inverse_transform_targets(y_pred: np.ndarray, idx: List[int]) -> np.ndarray:
    y2 = y_pred.copy()
    if idx:
        y2[:, idx] = np.expm1(y2[:, idx])
    return y2


def transform_single_target(y: np.ndarray, target_name: str) -> Tuple[np.ndarray, bool]:
    if target_name in LOG1P_TARGETS:
        return np.log1p(np.maximum(y, 0.0)), True
    return y.astype(float), False


def inverse_transform_single_target(y_pred: np.ndarray, was_log: bool) -> np.ndarray:
    if was_log:
        return np.expm1(y_pred)
    return y_pred


# -----------------------------
# Models (3 algorithms)
# -----------------------------
def make_multioutput_models(
    seed: int,
    n_estimators: int,
    max_features: float,
    bootstrap: bool,
) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            max_features=max_features,
            bootstrap=bootstrap,
            min_samples_leaf=1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=max(400, n_estimators // 2),
            random_state=seed,
            n_jobs=-1,
            max_features=max_features,
            bootstrap=bootstrap,
            min_samples_leaf=1,
        ),
        "HistGB": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                random_state=seed,
                max_iter=600,
                learning_rate=0.05,
                max_depth=None,
            )
        ),
    }
    return models


def make_single_target_model(
    model_name: str,
    seed: int,
    n_estimators: int,
    max_features: float,
    bootstrap: bool,
) -> Any:
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            max_features=max_features,
            bootstrap=bootstrap,
            min_samples_leaf=1,
        )
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=max(400, n_estimators // 2),
            random_state=seed,
            n_jobs=-1,
            max_features=max_features,
            bootstrap=bootstrap,
            min_samples_leaf=1,
        )
    if model_name == "HistGB":
        return HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=600,
            learning_rate=0.05,
            max_depth=None,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


# -----------------------------
# Metrics / selection
# -----------------------------
def _metrics_row(model: str, target: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"model": model, "target": target, "MAE": mae, "RMSE": rmse, "R2": r2}


def pick_best_models_per_target(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Best = highest R2; tie-breaker lowest RMSE.
    """
    out_rows = []
    for t in TARGETS:
        d = df_metrics[df_metrics["target"] == t].copy()
        d = d.sort_values(["R2", "RMSE"], ascending=[False, True])
        best = d.iloc[0].to_dict()
        out_rows.append(best)
    return pd.DataFrame(out_rows)


# -----------------------------
# Predict helpers
# -----------------------------
def features_from_requests_csv(
    requests_csv: Path,
    eig_k: int,
    wl_bins: int,
    nmf_k: int,
    seed: int,
) -> Dict[str, float]:
    req = pd.read_csv(requests_csv)
    req["source"] = req["source"].astype(int)
    req["destination"] = req["destination"].astype(int)
    req["bitrate"] = req["bitrate"].astype(float)
    return extract_features(req, eig_k=eig_k, wl_bins=wl_bins, nmf_k=nmf_k, seed=seed)


def align_feature_vector(feat: Dict[str, float], feature_columns: List[str]) -> np.ndarray:
    row = {c: float(feat.get(c, 0.0)) for c in feature_columns}
    return pd.DataFrame([row], columns=feature_columns).to_numpy(dtype=float)


# -----------------------------
# Training pipeline
# -----------------------------
def train_and_select(
    samples: List[Sample],
    outdir: Path,
    seed: int,
    test_size: float,
    n_estimators: int,
    max_features: float,
    bootstrap: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df_feat = pd.DataFrame([s.features for s in samples]).fillna(0.0)
    df_meta = pd.DataFrame([{"topology": s.topology, "request_set": s.request_set} for s in samples])
    df_y = pd.DataFrame([s.targets for s in samples])[TARGETS]

    feature_columns = list(df_feat.columns)

    X = df_feat.to_numpy(dtype=float)
    y = df_y.to_numpy(dtype=float)

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, df_meta, test_size=test_size, random_state=seed, shuffle=True
    )

    y_train_t, t_idx = transform_targets(y_train)

    models = make_multioutput_models(seed, n_estimators, max_features, bootstrap)

    # Evaluate all 3 models
    metrics_rows: List[Dict[str, Any]] = []
    for name, model in models.items():
        model.fit(X_train, y_train_t)
        y_pred_t = np.asarray(model.predict(X_test), dtype=float)
        y_pred = inverse_transform_targets(y_pred_t, t_idx)

        for i, t in enumerate(TARGETS):
            metrics_rows.append(_metrics_row(name, t, y_test[:, i], y_pred[:, i]))

    df_metrics = pd.DataFrame(metrics_rows).sort_values(["target", "model"]).reset_index(drop=True)
    df_best = pick_best_models_per_target(df_metrics)

    print("\n=== Metrics (all models) ===")
    print(df_metrics.to_string(index=False))
    print("\n=== Best model per target ===")
    print(df_best.to_string(index=False))

    # Save test predictions from the single best-per-target ensemble
    best_map = {row["target"]: row["model"] for row in df_best.to_dict(orient="records")}

    # Train best models per target on TRAIN split, predict TEST split (for consistent report)
    test_pred = meta_test.reset_index(drop=True).copy()
    for i, t in enumerate(TARGETS):
        model_name = best_map[t]
        reg = make_single_target_model(model_name, seed, n_estimators, max_features, bootstrap)

        y_tr_1 = y_train[:, i]
        y_tr_t, was_log = transform_single_target(y_tr_1, t)

        reg.fit(X_train, y_tr_t)
        y_pred_t_1 = np.asarray(reg.predict(X_test), dtype=float)
        y_pred_1 = inverse_transform_single_target(y_pred_t_1, was_log)

        test_pred[f"y_true_{t}"] = y_test[:, i]
        test_pred[f"y_pred_{t}"] = y_pred_1
        test_pred[f"best_model_{t}"] = model_name

    # Fit FINAL best-per-target models on FULL dataset
    final_models: Dict[str, Any] = {}
    final_model_names: Dict[str, str] = {}
    final_transform_flags: Dict[str, bool] = {}

    for i, t in enumerate(TARGETS):
        model_name = best_map[t]
        reg = make_single_target_model(model_name, seed, n_estimators, max_features, bootstrap)

        y_full_1 = y[:, i]
        y_full_t, was_log = transform_single_target(y_full_1, t)

        reg.fit(X, y_full_t)

        final_models[t] = reg
        final_model_names[t] = model_name
        final_transform_flags[t] = was_log

    topology_name = df_meta["topology"].iloc[0] if len(df_meta) else "unknown"

    bundle = {
        "topology": topology_name,
        "feature_columns": feature_columns,
        "targets": TARGETS,
        "log1p_targets": sorted(LOG1P_TARGETS),
        "model_name_per_target": final_model_names,
        "was_log_per_target": final_transform_flags,
        "models": final_models,
        "train_seed": seed,
    }

    # Save artifacts
    (outdir / "metrics_all.csv").write_text(df_metrics.to_csv(index=False), encoding="utf-8")
    (outdir / "metrics_best.csv").write_text(df_best.to_csv(index=False), encoding="utf-8")
    test_pred.to_csv(outdir / "preds_test.csv", index=False)

    df_full = pd.concat([df_meta, df_feat, df_y], axis=1)
    df_full.to_csv(outdir / "features_full.csv", index=False)

    joblib.dump(bundle, outdir / f"bundle_{topology_name}.joblib")
    (outdir / "best_models.json").write_text(json.dumps(best_map, indent=2), encoding="utf-8")

    print(f"\nSaved: {outdir / 'metrics_all.csv'}")
    print(f"Saved: {outdir / 'metrics_best.csv'}")
    print(f"Saved: {outdir / 'preds_test.csv'}")
    print(f"Saved: {outdir / 'features_full.csv'}")
    print(f"Saved: {outdir / f'bundle_{topology_name}.joblib'}")
    print(f"Saved: {outdir / 'best_models.json'}")


def predict_with_bundle(
    bundle_path: Path,
    requests_csv: Path,
    outdir: Path,
    eig_k: int,
    wl_bins: int,
    nmf_k: int,
    seed: int,
) -> Path:
    bundle = joblib.load(bundle_path)

    feat = features_from_requests_csv(requests_csv, eig_k=eig_k, wl_bins=wl_bins, nmf_k=nmf_k, seed=seed)
    x = align_feature_vector(feat, bundle["feature_columns"])

    rows = []
    for t in bundle["targets"]:
        reg = bundle["models"][t]
        was_log = bool(bundle["was_log_per_target"][t])

        y_pred_t = float(np.asarray(reg.predict(x), dtype=float).reshape(-1)[0])
        y_pred = float(inverse_transform_single_target(np.array([y_pred_t], dtype=float), was_log)[0])

        rows.append(
            {
                "target": t,
                "y_pred": y_pred,
                "model": bundle["model_name_per_target"][t],
            }
        )

    df = pd.DataFrame(rows)
    print("\n=== Prediction ===")
    print(df.to_string(index=False))

    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"predict_{bundle['topology']}_{ts}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=str, required=True, help="Path to RSA_estimation zip (can be dummy if extracted).")
    p.add_argument("--workdir", type=str, default=".", help="Extraction directory containing RSA_estimation/")
    p.add_argument("--outdir", type=str, default="./_out_v3", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)

    # feature params
    p.add_argument("--eig_k", type=int, default=24)
    p.add_argument("--wl_bins", type=int, default=128)
    p.add_argument("--nmf_k", type=int, default=6)

    # model params
    p.add_argument("--n_estimators", type=int, default=1200)
    p.add_argument("--max_features", type=float, default=0.7)
    p.add_argument("--bootstrap", action="store_true")

    # predict
    p.add_argument("--predict_requests", type=str, default="", help="If set, run prediction for this requests.csv")

    args = p.parse_args()
    set_global_seed(args.seed)

    zip_path = Path(args.zip).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    root = ensure_extracted(zip_path, workdir)

    for topology in ["Euro28", "US26"]:
        print("\n\n#############################")
        print(f"# Topology: {topology}")
        print("#############################")

        samples = load_dataset(
            root,
            topology=topology,
            eig_k=args.eig_k,
            wl_bins=args.wl_bins,
            nmf_k=args.nmf_k,
            seed=args.seed,
        )

        topo_out = outdir / topology
        train_and_select(
            samples=samples,
            outdir=topo_out,
            seed=args.seed,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            max_features=args.max_features,
            bootstrap=args.bootstrap,
        )

        if args.predict_requests:
            bundle_path = topo_out / f"bundle_{topology}.joblib"
            predict_with_bundle(
                bundle_path=bundle_path,
                requests_csv=Path(args.predict_requests).expanduser().resolve(),
                outdir=topo_out,
                eig_k=args.eig_k,
                wl_bins=args.wl_bins,
                nmf_k=args.nmf_k,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()

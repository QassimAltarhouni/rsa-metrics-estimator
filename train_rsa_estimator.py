# path: train_rsa_estimator_v4.py
r"""
V4: 3 algorithms + RepeatedKFold CV (mean±std) + optional sample weighting + optional SelectKBest.

Train:
  python .\train_rsa_estimator_v4.py --zip .\dummy.zip --workdir . --outdir .\_out_v4 --seed 42 ^
    --cv_splits 5 --cv_repeats 10 --use_weights --weight_alpha 1.0 --select_k 150

Predict (optional):
  python .\train_rsa_estimator_v4.py --zip .\dummy.zip --workdir . --outdir .\_out_v4 --seed 42 ^
    --use_weights --weight_alpha 1.0 --select_k 150 --predict_requests .\some_requests.csv
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split

TARGETS = ["highestSlot", "avgHighestSlot", "sumOfSlots", "avgActiveTransceivers"]
LOG1P_TARGETS = {"highestSlot", "avgHighestSlot", "sumOfSlots"}


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
    for p in workdir.rglob("RSA_estimation"):
        if p.is_dir():
            return p
    return None


def ensure_extracted(zip_path: Path, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    marker = workdir / ".extracted_ok"

    found = _find_rsa_estimation_dir(workdir)
    if found is not None and marker.exists():
        return found
    if found is not None and not marker.exists():
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
# Graph features
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
# Targets
# -----------------------------
def transform_y(y: np.ndarray, target: str) -> Tuple[np.ndarray, bool]:
    if target in LOG1P_TARGETS:
        return np.log1p(np.maximum(y.astype(float), 0.0)), True
    return y.astype(float), False


def inverse_transform_y(y_pred: np.ndarray, was_log: bool) -> np.ndarray:
    if was_log:
        return np.expm1(y_pred.astype(float))
    return y_pred.astype(float)


# -----------------------------
# Models (3 algorithms)
# -----------------------------
def make_regressor(model_name: str, seed: int, n_estimators: int, max_features: float, bootstrap: bool) -> Any:
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
# Weighting / CV evaluation
# -----------------------------
def compute_sample_weight(y_raw: np.ndarray, alpha: float) -> np.ndarray:
    y = np.asarray(y_raw, dtype=float)
    med = float(np.median(y)) if y.size else 0.0
    denom = med + 1e-9
    w = 1.0 + alpha * (y / denom)
    return np.clip(w, 0.1, 50.0)


def eval_cv_single_target(
    X: np.ndarray,
    y_raw: np.ndarray,
    target: str,
    model_name: str,
    seed: int,
    n_estimators: int,
    max_features: float,
    bootstrap: bool,
    rkf: RepeatedKFold,
    select_k: int,
    use_weights: bool,
    weight_alpha: float,
) -> Dict[str, Any]:
    y_t, was_log = transform_y(y_raw, target)
    w_full = compute_sample_weight(y_raw, weight_alpha) if use_weights else None

    maes: List[float] = []
    rmses: List[float] = []
    r2s: List[float] = []

    for tr, te in rkf.split(X):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_t[tr], y_t[te]
        w_tr = w_full[tr] if w_full is not None else None

        selector = None
        if 0 < select_k < X.shape[1]:
            selector = SelectKBest(score_func=mutual_info_regression, k=select_k)
            selector.fit(X_tr, y_tr)
            X_tr = selector.transform(X_tr)
            X_te = selector.transform(X_te)

        reg = make_regressor(model_name, seed, n_estimators, max_features, bootstrap)

        if w_tr is not None:
            reg.fit(X_tr, y_tr, sample_weight=w_tr)
        else:
            reg.fit(X_tr, y_tr)

        pred_t = np.asarray(reg.predict(X_te), dtype=float)
        pred = inverse_transform_y(pred_t, was_log)
        true = inverse_transform_y(np.asarray(y_te, dtype=float), was_log)

        maes.append(float(mean_absolute_error(true, pred)))
        rmses.append(float(math.sqrt(mean_squared_error(true, pred))))
        r2s.append(float(r2_score(true, pred)))

    return {
        "model": model_name,
        "target": target,
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes, ddof=0)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses, ddof=0)),
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s, ddof=0)),
    }


def pick_best_per_target_cv(df_cv: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t in TARGETS:
        d = df_cv[df_cv["target"] == t].copy()
        d = d.sort_values(["R2_mean", "RMSE_mean"], ascending=[False, True])
        out.append(d.iloc[0].to_dict())
    return pd.DataFrame(out)


def fit_final_bundle(
    X: np.ndarray,
    feat_cols: List[str],
    y: np.ndarray,
    df_meta: pd.DataFrame,
    best_map: Dict[str, str],
    seed: int,
    n_estimators: int,
    max_features: float,
    bootstrap: bool,
    select_k: int,
    use_weights: bool,
    weight_alpha: float,
) -> Dict[str, Any]:
    topology_name = df_meta["topology"].iloc[0] if len(df_meta) else "unknown"

    models: Dict[str, Any] = {}
    selectors: Dict[str, Any] = {}
    feature_cols_per_target: Dict[str, List[str]] = {}
    was_log_per_target: Dict[str, bool] = {}

    for i, t in enumerate(TARGETS):
        model_name = best_map[t]
        y_raw = y[:, i]
        y_t, was_log = transform_y(y_raw, t)
        w = compute_sample_weight(y_raw, weight_alpha) if use_weights else None

        selector = None
        X_fit = X
        selected_cols = feat_cols

        if 0 < select_k < X.shape[1]:
            selector = SelectKBest(score_func=mutual_info_regression, k=select_k)
            selector.fit(X, y_t)
            idx = selector.get_support(indices=True)
            selected_cols = [feat_cols[j] for j in idx]
            X_fit = selector.transform(X)

        reg = make_regressor(model_name, seed, n_estimators, max_features, bootstrap)
        if w is not None:
            reg.fit(X_fit, y_t, sample_weight=w)
        else:
            reg.fit(X_fit, y_t)

        models[t] = reg
        selectors[t] = selector
        feature_cols_per_target[t] = selected_cols
        was_log_per_target[t] = was_log

    return {
        "topology": topology_name,
        "targets": TARGETS,
        "log1p_targets": sorted(LOG1P_TARGETS),
        "model_name_per_target": best_map,
        "was_log_per_target": was_log_per_target,
        "feature_columns_full": feat_cols,
        "feature_columns_per_target": feature_cols_per_target,
        "selectors_per_target": selectors,
        "models_per_target": models,
        "train_seed": seed,
    }


def predict_with_bundle(bundle_path: Path, requests_csv: Path, eig_k: int, wl_bins: int, nmf_k: int, seed: int, outdir: Path) -> Path:
    bundle = joblib.load(bundle_path)

    req = pd.read_csv(requests_csv)
    req["source"] = req["source"].astype(int)
    req["destination"] = req["destination"].astype(int)
    req["bitrate"] = req["bitrate"].astype(float)

    feat = extract_features(req, eig_k=eig_k, wl_bins=wl_bins, nmf_k=nmf_k, seed=seed)
    full_cols = bundle["feature_columns_full"]
    x_full = pd.DataFrame([{c: float(feat.get(c, 0.0)) for c in full_cols}], columns=full_cols).to_numpy(dtype=float)

    rows = []
    for t in bundle["targets"]:
        selector = bundle["selectors_per_target"][t]
        model = bundle["models_per_target"][t]
        was_log = bool(bundle["was_log_per_target"][t])

        x = selector.transform(x_full) if selector is not None else x_full
        pred_t = float(np.asarray(model.predict(x), dtype=float).reshape(-1)[0])
        pred = float(inverse_transform_y(np.array([pred_t], dtype=float), was_log)[0])

        rows.append({"target": t, "y_pred": pred, "model": bundle["model_name_per_target"][t]})

    df = pd.DataFrame(rows)
    print("\n=== Prediction ===")
    print(df.to_string(index=False))

    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"predict_{bundle['topology']}_{ts}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return out_path


def train_pipeline(
    samples: List[Sample],
    outdir: Path,
    seed: int,
    test_size: float,
    n_estimators: int,
    max_features: float,
    bootstrap: bool,
    cv_splits: int,
    cv_repeats: int,
    select_k: int,
    use_weights: bool,
    weight_alpha: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df_feat = pd.DataFrame([s.features for s in samples]).fillna(0.0)
    df_meta = pd.DataFrame([{"topology": s.topology, "request_set": s.request_set} for s in samples])
    df_y = pd.DataFrame([s.targets for s in samples])[TARGETS]

    X = df_feat.to_numpy(dtype=float)
    y = df_y.to_numpy(dtype=float)
    feat_cols = list(df_feat.columns)

    rkf = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=seed)
    model_names = ["ExtraTrees", "RandomForest", "HistGB"]

    cv_rows: List[Dict[str, Any]] = []
    for t_i, t in enumerate(TARGETS):
        y_raw = y[:, t_i]
        for m in model_names:
            cv_rows.append(
                eval_cv_single_target(
                    X=X,
                    y_raw=y_raw,
                    target=t,
                    model_name=m,
                    seed=seed,
                    n_estimators=n_estimators,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    rkf=rkf,
                    select_k=select_k,
                    use_weights=use_weights,
                    weight_alpha=weight_alpha,
                )
            )

    df_cv = pd.DataFrame(cv_rows).sort_values(["target", "model"]).reset_index(drop=True)
    df_best = pick_best_per_target_cv(df_cv)
    best_map = {row["target"]: row["model"] for row in df_best.to_dict(orient="records")}

    print("\n=== CV Metrics (mean±std across folds) ===")
    print(df_cv.to_string(index=False))
    print("\n=== Best model per target (from CV) ===")
    print(df_best.to_string(index=False))

    df_cv.to_csv(outdir / "cv_metrics_all.csv", index=False)
    df_best.to_csv(outdir / "cv_metrics_best.csv", index=False)
    (outdir / "best_models_cv.json").write_text(json.dumps(best_map, indent=2), encoding="utf-8")

    # Holdout (sanity only)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, df_meta, test_size=test_size, random_state=seed, shuffle=True
    )
    preds = meta_test.reset_index(drop=True).copy()

    for i, t in enumerate(TARGETS):
        model_name = best_map[t]
        y_tr_raw = y_train[:, i]
        y_tr_t, was_log = transform_y(y_tr_raw, t)
        w_tr = compute_sample_weight(y_tr_raw, weight_alpha) if use_weights else None

        selector = None
        X_tr = X_train
        X_te = X_test
        if 0 < select_k < X.shape[1]:
            selector = SelectKBest(score_func=mutual_info_regression, k=select_k)
            selector.fit(X_tr, y_tr_t)
            X_tr = selector.transform(X_tr)
            X_te = selector.transform(X_te)

        reg = make_regressor(model_name, seed, n_estimators, max_features, bootstrap)
        if w_tr is not None:
            reg.fit(X_tr, y_tr_t, sample_weight=w_tr)
        else:
            reg.fit(X_tr, y_tr_t)

        pred_t = np.asarray(reg.predict(X_te), dtype=float)
        pred = inverse_transform_y(pred_t, was_log)

        preds[f"y_true_{t}"] = y_test[:, i]
        preds[f"y_pred_{t}"] = pred
        preds[f"best_model_{t}"] = model_name

    preds.to_csv(outdir / "preds_holdout.csv", index=False)

    bundle = fit_final_bundle(
        X=X,
        feat_cols=feat_cols,
        y=y,
        df_meta=df_meta,
        best_map=best_map,
        seed=seed,
        n_estimators=n_estimators,
        max_features=max_features,
        bootstrap=bootstrap,
        select_k=select_k,
        use_weights=use_weights,
        weight_alpha=weight_alpha,
    )

    topo = bundle["topology"]
    joblib.dump(bundle, outdir / f"bundle_{topo}.joblib")

    print(f"\nSaved: {outdir / 'cv_metrics_all.csv'}")
    print(f"Saved: {outdir / 'cv_metrics_best.csv'}")
    print(f"Saved: {outdir / 'best_models_cv.json'}")
    print(f"Saved: {outdir / 'preds_holdout.csv'}")
    print(f"Saved: {outdir / f'bundle_{topo}.joblib'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=str, required=True)
    p.add_argument("--workdir", type=str, default=".")
    p.add_argument("--outdir", type=str, default="./_out_v4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)

    p.add_argument("--eig_k", type=int, default=24)
    p.add_argument("--wl_bins", type=int, default=128)
    p.add_argument("--nmf_k", type=int, default=6)

    p.add_argument("--n_estimators", type=int, default=1200)
    p.add_argument("--max_features", type=float, default=0.7)
    p.add_argument("--bootstrap", action="store_true")

    p.add_argument("--cv_splits", type=int, default=5)
    p.add_argument("--cv_repeats", type=int, default=10)

    p.add_argument("--select_k", type=int, default=0, help="Enable SelectKBest; 0 disables.")
    p.add_argument("--use_weights", action="store_true", help="Enable sample weights.")
    p.add_argument("--weight_alpha", type=float, default=1.0)

    p.add_argument("--predict_requests", type=str, default="")

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
        train_pipeline(
            samples=samples,
            outdir=topo_out,
            seed=args.seed,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            max_features=args.max_features,
            bootstrap=args.bootstrap,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            select_k=args.select_k,
            use_weights=args.use_weights,
            weight_alpha=args.weight_alpha,
        )

        if args.predict_requests:
            bundle_path = topo_out / f"bundle_{topology}.joblib"
            predict_with_bundle(
                bundle_path=bundle_path,
                requests_csv=Path(args.predict_requests).expanduser().resolve(),
                eig_k=args.eig_k,
                wl_bins=args.wl_bins,
                nmf_k=args.nmf_k,
                seed=args.seed,
                outdir=topo_out,
            )


if __name__ == "__main__":
    main()

# # path: train_rsa_estimator.py
# """
# RSA objective/metric estimator (topology-agnostic).
# - Input: a set of requests (source, destination, bitrate)
# - Output: predicts 4 RSA metrics without running RSA:
#   highestSlot, avgHighestSlot, sumOfSlots, avgActiveTransceivers
#
# How to run (example):
#   python train_rsa_estimator.py \
#     --zip /mnt/data/RSA_estimation-20251027T181509Z-1-001.zip \
#     --workdir ./_data \
#     --outdir ./_out \
#     --test_size 0.2 \
#     --seed 42
#
# Outputs:
#   - _out/models/<topology>.joblib
#   - _out/features_<topology>.csv
#   - _out/preds_<topology>.csv
#   - printed metrics (MAE/RMSE/R2) per target
# """
#
# from __future__ import annotations
#
# import argparse
# import json
# import math
# import os
# import random
# import zipfile
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Iterable, List, Tuple
#
# import joblib
# import networkx as nx
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
#
#
# TARGETS = ["highestSlot", "avgHighestSlot", "sumOfSlots", "avgActiveTransceivers"]
#
#
# def set_global_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#
#
# def ensure_extracted(zip_path: Path, workdir: Path) -> Path:
#     workdir.mkdir(parents=True, exist_ok=True)
#     marker = workdir / ".extracted_ok"
#     root_guess = workdir / "RSA_estimation"
#     if marker.exists() and root_guess.exists():
#         return root_guess
#
#     with zipfile.ZipFile(zip_path, "r") as zf:
#         zf.extractall(workdir)
#
#     marker.write_text("ok", encoding="utf-8")
#     if not root_guess.exists():
#         raise FileNotFoundError(f"Expected folder {root_guess} after extraction.")
#     return root_guess
#
#
# def parse_results_txt(results_path: Path) -> Dict[str, float]:
#     out: Dict[str, float] = {}
#     for line in results_path.read_text(encoding="utf-8").splitlines():
#         if not line.strip():
#             continue
#         k, v = line.split("\t")
#         out[k.strip()] = float(v.strip())
#     missing = [t for t in TARGETS if t not in out]
#     if missing:
#         raise ValueError(f"Missing targets {missing} in {results_path}")
#     return out
#
#
# def safe_skew(x: np.ndarray) -> float:
#     if x.size < 3:
#         return 0.0
#     mu = x.mean()
#     s = x.std(ddof=0)
#     if s == 0:
#         return 0.0
#     return float(np.mean(((x - mu) / s) ** 3))
#
#
# def safe_kurtosis(x: np.ndarray) -> float:
#     if x.size < 4:
#         return 0.0
#     mu = x.mean()
#     s = x.std(ddof=0)
#     if s == 0:
#         return 0.0
#     return float(np.mean(((x - mu) / s) ** 4) - 3.0)
#
#
# def gini(x: np.ndarray) -> float:
#     # Gini on non-negative values
#     x = np.asarray(x, dtype=float)
#     if x.size == 0:
#         return 0.0
#     x = np.clip(x, 0, None)
#     if np.all(x == 0):
#         return 0.0
#     x_sorted = np.sort(x)
#     n = x_sorted.size
#     cum = np.cumsum(x_sorted)
#     return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)
#
#
# def entropy_from_counts(counts: np.ndarray) -> float:
#     counts = counts.astype(float)
#     total = counts.sum()
#     if total <= 0:
#         return 0.0
#     p = counts / total
#     p = p[p > 0]
#     return float(-(p * np.log2(p)).sum())
#
#
# def quantiles(x: np.ndarray, qs: Iterable[float]) -> Dict[str, float]:
#     x = np.asarray(x, dtype=float)
#     if x.size == 0:
#         return {f"q{int(q*100):02d}": 0.0 for q in qs}
#     vals = np.quantile(x, list(qs))
#     return {f"q{int(q*100):02d}": float(v) for q, v in zip(qs, vals)}
#
#
# def build_request_graph(req: pd.DataFrame) -> nx.DiGraph:
#     # Directed weighted graph where edge weight = sum bitrate, edge count = number of demands
#     g = nx.DiGraph()
#     for s, d, b in req[["source", "destination", "bitrate"]].itertuples(index=False):
#         if g.has_edge(s, d):
#             g[s][d]["w_sum"] += float(b)
#             g[s][d]["cnt"] += 1
#         else:
#             g.add_edge(s, d, w_sum=float(b), cnt=1)
#     return g
#
#
# def adjacency_eigenspectrum_features(req: pd.DataFrame, k: int = 12) -> Dict[str, float]:
#     # Top-k eigenvalues of symmetric weighted adjacency (A + A^T)/2 over nodes appearing in req
#     nodes = np.unique(np.concatenate([req["source"].values, req["destination"].values]))
#     nodes = np.asarray(nodes, dtype=int)
#     n = nodes.size
#     if n == 0:
#         return {f"eig_{i}": 0.0 for i in range(k)}
#
#     idx = {node: i for i, node in enumerate(nodes)}
#     A = np.zeros((n, n), dtype=float)
#     for s, d, b in req[["source", "destination", "bitrate"]].itertuples(index=False):
#         A[idx[int(s)], idx[int(d)]] += float(b)
#
#     S = (A + A.T) / 2.0
#     # Symmetric => real eigenvalues
#     eigvals = np.linalg.eigvalsh(S)
#     eigvals = np.sort(np.abs(eigvals))[::-1]  # magnitude, descending
#     eigvals = eigvals[:k]
#     if eigvals.size < k:
#         eigvals = np.pad(eigvals, (0, k - eigvals.size), constant_values=0.0)
#
#     return {f"eig_{i}": float(eigvals[i]) for i in range(k)}
#
#
# def extract_features(req: pd.DataFrame, eig_k: int = 12) -> Dict[str, float]:
#     # Basic set stats
#     n_req = int(len(req))
#     nodes = np.unique(np.concatenate([req["source"].values, req["destination"].values]))
#     n_nodes = int(nodes.size)
#
#     bitrate = req["bitrate"].to_numpy(dtype=float)
#     bitrate_stats = {
#         "n_requests": float(n_req),
#         "n_nodes_in_requests": float(n_nodes),
#         "bitrate_mean": float(bitrate.mean()) if n_req else 0.0,
#         "bitrate_std": float(bitrate.std(ddof=0)) if n_req else 0.0,
#         "bitrate_min": float(bitrate.min()) if n_req else 0.0,
#         "bitrate_max": float(bitrate.max()) if n_req else 0.0,
#         "bitrate_skew": safe_skew(bitrate) if n_req else 0.0,
#         "bitrate_kurt": safe_kurtosis(bitrate) if n_req else 0.0,
#         "bitrate_gini": gini(bitrate) if n_req else 0.0,
#     }
#     bitrate_stats.update(quantiles(bitrate, qs=[0.05, 0.25, 0.50, 0.75, 0.95]))
#
#     # Pair diversity / entropy
#     pairs = list(zip(req["source"].astype(int), req["destination"].astype(int)))
#     n_unique_pairs = len(set(pairs))
#     bitrate_stats["unique_pair_ratio"] = float(n_unique_pairs / n_req) if n_req else 0.0
#
#     src_counts = req["source"].value_counts().to_numpy()
#     dst_counts = req["destination"].value_counts().to_numpy()
#     bitrate_stats["src_entropy"] = entropy_from_counts(src_counts)
#     bitrate_stats["dst_entropy"] = entropy_from_counts(dst_counts)
#
#     # Graph features from request-graph
#     g = build_request_graph(req)
#     m = g.number_of_edges()
#
#     graph_feats: Dict[str, float] = {
#         "g_edges": float(m),
#         "g_density": float(nx.density(g)) if n_nodes > 1 else 0.0,
#         "g_reciprocity": float(nx.reciprocity(g)) if m > 0 else 0.0,
#     }
#
#     # Degree / strength stats
#     in_deg = np.array([d for _, d in g.in_degree()], dtype=float) if n_nodes else np.array([])
#     out_deg = np.array([d for _, d in g.out_degree()], dtype=float) if n_nodes else np.array([])
#     in_str = np.array([s for _, s in g.in_degree(weight="w_sum")], dtype=float) if n_nodes else np.array([])
#     out_str = np.array([s for _, s in g.out_degree(weight="w_sum")], dtype=float) if n_nodes else np.array([])
#
#     def stats_block(name: str, x: np.ndarray) -> Dict[str, float]:
#         if x.size == 0:
#             return {f"{name}_{k}": 0.0 for k in ["mean", "std", "max", "q50", "q90"]}
#         return {
#             f"{name}_mean": float(x.mean()),
#             f"{name}_std": float(x.std(ddof=0)),
#             f"{name}_max": float(x.max()),
#             f"{name}_q50": float(np.quantile(x, 0.50)),
#             f"{name}_q90": float(np.quantile(x, 0.90)),
#         }
#
#     graph_feats.update(stats_block("in_deg", in_deg))
#     graph_feats.update(stats_block("out_deg", out_deg))
#     graph_feats.update(stats_block("in_str", in_str))
#     graph_feats.update(stats_block("out_str", out_str))
#
#     # Clustering / transitivity (use undirected projection)
#     if m > 0:
#         gu = g.to_undirected()
#         try:
#             graph_feats["g_transitivity"] = float(nx.transitivity(gu))
#         except Exception:
#             graph_feats["g_transitivity"] = 0.0
#         try:
#             graph_feats["g_avg_clustering"] = float(nx.average_clustering(gu, weight=None))
#         except Exception:
#             graph_feats["g_avg_clustering"] = 0.0
#         try:
#             graph_feats["g_degree_assort"] = float(nx.degree_assortativity_coefficient(gu))
#         except Exception:
#             graph_feats["g_degree_assort"] = 0.0
#     else:
#         graph_feats["g_transitivity"] = 0.0
#         graph_feats["g_avg_clustering"] = 0.0
#         graph_feats["g_degree_assort"] = 0.0
#
#     # Pagerank stats (weighted)
#     if m > 0:
#         try:
#             pr = nx.pagerank(g, weight="w_sum")
#             pr_vals = np.array(list(pr.values()), dtype=float)
#             graph_feats["pr_mean"] = float(pr_vals.mean())
#             graph_feats["pr_std"] = float(pr_vals.std(ddof=0))
#             graph_feats["pr_max"] = float(pr_vals.max())
#         except Exception:
#             graph_feats["pr_mean"] = 0.0
#             graph_feats["pr_std"] = 0.0
#             graph_feats["pr_max"] = 0.0
#     else:
#         graph_feats["pr_mean"] = 0.0
#         graph_feats["pr_std"] = 0.0
#         graph_feats["pr_max"] = 0.0
#
#     # Innovative: eigen-spectrum embedding
#     eig_feats = adjacency_eigenspectrum_features(req, k=eig_k)
#
#     return {**bitrate_stats, **graph_feats, **eig_feats}
#
#
# @dataclass(frozen=True)
# class Sample:
#     topology: str
#     request_set: str
#     features: Dict[str, float]
#     targets: Dict[str, float]
#
#
# def load_dataset(root: Path, topology: str, eig_k: int) -> List[Sample]:
#     topo_dir = root / topology
#     if not topo_dir.exists():
#         raise FileNotFoundError(f"Topology folder not found: {topo_dir}")
#
#     samples: List[Sample] = []
#     for req_dir in sorted(topo_dir.glob("request-set_*")):
#         req_csv = req_dir / "requests.csv"
#         res_txt = req_dir / "results.txt"
#         if not req_csv.exists() or not res_txt.exists():
#             continue
#
#         req = pd.read_csv(req_csv)
#         # Defensive typing
#         req["source"] = req["source"].astype(int)
#         req["destination"] = req["destination"].astype(int)
#         req["bitrate"] = req["bitrate"].astype(float)
#
#         feats = extract_features(req, eig_k=eig_k)
#         targs = parse_results_txt(res_txt)
#
#         samples.append(
#             Sample(
#                 topology=topology,
#                 request_set=req_dir.name,
#                 features=feats,
#                 targets=targs,
#             )
#         )
#     if not samples:
#         raise ValueError(f"No samples found for topology {topology} in {topo_dir}")
#     return samples
#
#
# def train_and_evaluate(
#     samples: List[Sample],
#     outdir: Path,
#     seed: int,
#     test_size: float,
#     n_estimators: int,
#     max_features: str,
# ) -> None:
#     outdir.mkdir(parents=True, exist_ok=True)
#
#     df_feat = pd.DataFrame([s.features for s in samples])
#     df_meta = pd.DataFrame(
#         [{"topology": s.topology, "request_set": s.request_set} for s in samples]
#     )
#     df_y = pd.DataFrame([s.targets for s in samples])[TARGETS]
#
#     X = df_feat.to_numpy(dtype=float)
#     y = df_y.to_numpy(dtype=float)
#
#     X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
#         X, y, df_meta, test_size=test_size, random_state=seed, shuffle=True
#     )
#
#     model = ExtraTreesRegressor(
#         n_estimators=n_estimators,
#         random_state=seed,
#         n_jobs=-1,
#         max_features=max_features,
#         min_samples_leaf=1,
#     )
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#
#     # Report metrics per target
#     print("\n=== Evaluation (per target) ===")
#     rows = []
#     for i, t in enumerate(TARGETS):
#         mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
#         rmse = math.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
#         r2 = r2_score(y_test[:, i], y_pred[:, i])
#         rows.append({"target": t, "MAE": mae, "RMSE": rmse, "R2": r2})
#     df_metrics = pd.DataFrame(rows)
#     print(df_metrics.to_string(index=False))
#
#     # Save artifacts
#     topo_name = meta_test["topology"].iloc[0] if len(meta_test) else "unknown"
#     joblib.dump(model, outdir / f"model_{topo_name}.joblib")
#
#     # Save full feature table (for your report)
#     df_full = pd.concat([df_meta, df_feat, df_y], axis=1)
#     df_full.to_csv(outdir / f"features_{topo_name}.csv", index=False)
#
#     # Save predictions on test split
#     df_pred = meta_test.reset_index(drop=True).copy()
#     for i, t in enumerate(TARGETS):
#         df_pred[f"y_true_{t}"] = y_test[:, i]
#         df_pred[f"y_pred_{t}"] = y_pred[:, i]
#     df_pred.to_csv(outdir / f"preds_{topo_name}.csv", index=False)
#
#     print(f"\nSaved: {outdir / f'model_{topo_name}.joblib'}")
#     print(f"Saved: {outdir / f'features_{topo_name}.csv'}")
#     print(f"Saved: {outdir / f'preds_{topo_name}.csv'}")
#
#
# def main() -> None:
#     p = argparse.ArgumentParser()
#     p.add_argument("--zip", type=str, required=True, help="Path to RSA_estimation zip")
#     p.add_argument("--workdir", type=str, default="./_data", help="Extraction directory")
#     p.add_argument("--outdir", type=str, default="./_out", help="Output directory")
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--test_size", type=float, default=0.2)
#     p.add_argument("--eig_k", type=int, default=12, help="Top-k eigen spectrum features")
#     p.add_argument("--n_estimators", type=int, default=600)
#     p.add_argument(
#         "--max_features",
#         type=str,
#         default="sqrt",
#         help="ExtraTrees max_features (e.g., 'sqrt', 'log2', 0.7)",
#     )
#     args = p.parse_args()
#
#     set_global_seed(args.seed)
#
#     zip_path = Path(args.zip).expanduser().resolve()
#     workdir = Path(args.workdir).expanduser().resolve()
#     outdir = Path(args.outdir).expanduser().resolve()
#
#     root = ensure_extracted(zip_path, workdir)
#
#     # Train separately per topology (as required for evaluation on both)
#     for topology in ["Euro28", "US26"]:
#         print(f"\n\n#############################")
#         print(f"# Topology: {topology}")
#         print(f"#############################")
#         samples = load_dataset(root, topology=topology, eig_k=args.eig_k)
#         train_and_evaluate(
#             samples=samples,
#             outdir=outdir / topology,
#             seed=args.seed,
#             test_size=args.test_size,
#             n_estimators=args.n_estimators,
#             max_features=args.max_features,
#         )
#
#
# if __name__ == "__main__":
#     set_global_seed(42)
#     main()


# path: train_rsa_estimator_v2.py
"""
V2: Stronger graph-based features + target log-transform (for slot-like metrics).

Run (same as before):
  python .\train_rsa_estimator_v2.py --zip .\dummy.zip --workdir . --outdir .\_out --test_size 0.2 --seed 42

Notes:
- If you already have RSA_estimation/ extracted, create .extracted_ok to skip zip extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGETS = ["highestSlot", "avgHighestSlot", "sumOfSlots", "avgActiveTransceivers"]


# -----------------------------
# Repro
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------
# IO
# -----------------------------
def ensure_extracted(zip_path: Path, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    marker = workdir / ".extracted_ok"
    root_guess = workdir / "RSA_estimation"

    if marker.exists() and root_guess.exists():
        return root_guess

    # If already extracted but marker missing, accept it
    if root_guess.exists():
        return root_guess

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(workdir)

    marker.write_text("ok", encoding="utf-8")
    if not root_guess.exists():
        raise FileNotFoundError(f"Expected folder {root_guess} after extraction.")
    return root_guess


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
        s = int(s)
        d = int(d)
        b = float(b)
        if g.has_edge(s, d):
            g[s][d]["w_sum"] += b
            g[s][d]["cnt"] += 1
        else:
            g.add_edge(s, d, w_sum=b, cnt=1)
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
    eigvals = np.sort(np.abs(eigvals))[::-1]
    eigvals = eigvals[:k]
    if eigvals.size < k:
        eigvals = np.pad(eigvals, (0, k - eigvals.size), constant_values=0.0)

    return {f"eig_{i}": float(eigvals[i]) for i in range(k)}


# -----------------------------
# New V2 features (innovative)
# -----------------------------
def hashed_wl_features(
    g: nx.DiGraph,
    n_bins: int = 128,
    iters: int = 2,
) -> Dict[str, float]:
    """
    Weisfeiler–Lehman (WL) subtree features with feature hashing into fixed bins.
    Works well as a compact graph structure signature.
    """
    if g.number_of_nodes() == 0:
        return {f"wl_{i}": 0.0 for i in range(n_bins)}

    # initial node labels: (in_deg, out_deg) + rounded strengths
    in_deg = dict(g.in_degree())
    out_deg = dict(g.out_degree())
    in_str = dict(g.in_degree(weight="w_sum"))
    out_str = dict(g.out_degree(weight="w_sum"))

    labels: Dict[int, str] = {}
    for v in g.nodes():
        labels[int(v)] = f"{in_deg[v]}|{out_deg[v]}|{int(round(in_str[v]))}|{int(round(out_str[v]))}"

    bins = np.zeros(n_bins, dtype=float)

    def _bin(h: str) -> int:
        # stable hash -> bin
        digest = hashlib.md5(h.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % n_bins

    # count initial labels
    for v, lab in labels.items():
        bins[_bin(lab)] += 1.0

    # WL iterations
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
        for v, lab in labels.items():
            bins[_bin(lab)] += 1.0

    # normalize by nodes (helps generalization)
    bins = bins / max(1.0, float(g.number_of_nodes()))
    return {f"wl_{i}": float(bins[i]) for i in range(n_bins)}


def demand_matrix_svd_features(req: pd.DataFrame, k: int = 10) -> Dict[str, float]:
    """
    Build OD demand matrix (sum bitrate) and extract top-k singular values.
    Strong fixed-length summary of traffic pattern.
    """
    nodes = np.unique(np.concatenate([req["source"].values, req["destination"].values]))
    nodes = np.asarray(nodes, dtype=int)
    n = nodes.size
    if n == 0:
        return {f"svd_{i}": 0.0 for i in range(k)}

    idx = {node: i for i, node in enumerate(nodes)}
    M = np.zeros((n, n), dtype=float)
    for s, d, b in req[["source", "destination", "bitrate"]].itertuples(index=False):
        M[idx[int(s)], idx[int(d)]] += float(b)

    # SVD singular values
    svals = np.linalg.svd(M, compute_uv=False)
    svals = svals[:k]
    if svals.size < k:
        svals = np.pad(svals, (0, k - svals.size), constant_values=0.0)

    # scale-invariant: normalize by sum bitrate
    total = float(M.sum())
    if total > 0:
        svals = svals / total

    return {f"svd_{i}": float(svals[i]) for i in range(k)}


def bitrate_hist_features(bitrate: np.ndarray, bins: Tuple[float, ...] = (0, 50, 100, 150, 200, 400, 800, 1600)) -> Dict[str, float]:
    """
    Histogram counts normalized by n_requests.
    """
    if bitrate.size == 0:
        return {f"bh_{i}": 0.0 for i in range(len(bins) - 1)}
    hist, _ = np.histogram(bitrate, bins=np.array(bins, dtype=float))
    hist = hist.astype(float) / float(bitrate.size)
    return {f"bh_{i}": float(hist[i]) for i in range(hist.size)}


def topk_edge_weight_features(g: nx.DiGraph, k: int = 20) -> Dict[str, float]:
    """
    Sort edge weights and keep top-k (padded).
    Captures concentration / bottleneck-ish patterns.
    """
    w = [float(data.get("w_sum", 0.0)) for _, _, data in g.edges(data=True)]
    if not w:
        return {f"ew_{i}": 0.0 for i in range(k)}
    w = np.sort(np.array(w, dtype=float))[::-1]
    w = w[:k]
    if w.size < k:
        w = np.pad(w, (0, k - w.size), constant_values=0.0)
    # normalize by total traffic
    total = float(np.sum(w))
    if total > 0:
        w = w / total
    return {f"ew_{i}": float(w[i]) for i in range(k)}


# -----------------------------
# Feature extraction (V2)
# -----------------------------
def extract_features(req: pd.DataFrame, eig_k: int = 24, wl_bins: int = 128) -> Dict[str, float]:
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

    # V2 additions
    feats.update(adjacency_eigenspectrum_features(req, k=eig_k))
    feats.update(demand_matrix_svd_features(req, k=10))
    feats.update(topk_edge_weight_features(g, k=20))
    feats.update(hashed_wl_features(g, n_bins=wl_bins, iters=2))
    return feats


# -----------------------------
# Dataset & training
# -----------------------------
@dataclass(frozen=True)
class Sample:
    topology: str
    request_set: str
    features: Dict[str, float]
    targets: Dict[str, float]


def load_dataset(root: Path, topology: str, eig_k: int, wl_bins: int) -> List[Sample]:
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

        feats = extract_features(req, eig_k=eig_k, wl_bins=wl_bins)
        targs = parse_results_txt(res_txt)

        samples.append(
            Sample(
                topology=topology,
                request_set=req_dir.name,
                features=feats,
                targets=targs,
            )
        )
    if not samples:
        raise ValueError(f"No samples found for topology {topology} in {topo_dir}")
    return samples


def transform_targets(y: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Apply log1p to slot-like + sum targets to stabilize learning.
    Indices: highestSlot, avgHighestSlot, sumOfSlots (0,1,2)
    """
    y2 = y.copy()
    idx = [0, 1, 2]
    y2[:, idx] = np.log1p(np.maximum(y2[:, idx], 0.0))
    return y2, idx


def inverse_transform_targets(y_pred: np.ndarray, idx: List[int]) -> np.ndarray:
    y2 = y_pred.copy()
    y2[:, idx] = np.expm1(y2[:, idx])
    return y2


def train_and_evaluate(
    samples: List[Sample],
    outdir: Path,
    seed: int,
    test_size: float,
    n_estimators: int,
    max_features: float,
    bootstrap: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df_feat = pd.DataFrame([s.features for s in samples])
    df_meta = pd.DataFrame([{"topology": s.topology, "request_set": s.request_set} for s in samples])
    df_y = pd.DataFrame([s.targets for s in samples])[TARGETS]

    X = df_feat.to_numpy(dtype=float)
    y = df_y.to_numpy(dtype=float)

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, df_meta, test_size=test_size, random_state=seed, shuffle=True
    )

    y_train_t, t_idx = transform_targets(y_train)

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        max_features=max_features,
        bootstrap=bootstrap,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train_t)

    y_pred_t = model.predict(X_test)
    y_pred = inverse_transform_targets(np.asarray(y_pred_t, dtype=float), t_idx)

    print("\n=== Evaluation (per target) ===")
    rows = []
    for i, t in enumerate(TARGETS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        rows.append({"target": t, "MAE": mae, "RMSE": rmse, "R2": r2})
    df_metrics = pd.DataFrame(rows)
    print(df_metrics.to_string(index=False))

    topo_name = meta_test["topology"].iloc[0] if len(meta_test) else "unknown"
    joblib.dump(model, outdir / f"model_{topo_name}.joblib")

    df_full = pd.concat([df_meta, df_feat, df_y], axis=1)
    df_full.to_csv(outdir / f"features_{topo_name}.csv", index=False)

    df_pred = meta_test.reset_index(drop=True).copy()
    for i, t in enumerate(TARGETS):
        df_pred[f"y_true_{t}"] = y_test[:, i]
        df_pred[f"y_pred_{t}"] = y_pred[:, i]
    df_pred.to_csv(outdir / f"preds_{topo_name}.csv", index=False)

    df_metrics.to_csv(outdir / f"metrics_{topo_name}.csv", index=False)

    print(f"\nSaved: {outdir / f'model_{topo_name}.joblib'}")
    print(f"Saved: {outdir / f'features_{topo_name}.csv'}")
    print(f"Saved: {outdir / f'preds_{topo_name}.csv'}")
    print(f"Saved: {outdir / f'metrics_{topo_name}.csv'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=str, required=True, help="Path to RSA_estimation zip (can be dummy if extracted).")
    p.add_argument("--workdir", type=str, default=".", help="Extraction directory containing RSA_estimation/")
    p.add_argument("--outdir", type=str, default="./_out_v2", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)

    # feature params
    p.add_argument("--eig_k", type=int, default=24)
    p.add_argument("--wl_bins", type=int, default=128)

    # model params
    p.add_argument("--n_estimators", type=int, default=1200)
    p.add_argument("--max_features", type=float, default=0.7)
    p.add_argument("--bootstrap", action="store_true", help="Enable bootstrap sampling")

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
        samples = load_dataset(root, topology=topology, eig_k=args.eig_k, wl_bins=args.wl_bins)
        train_and_evaluate(
            samples=samples,
            outdir=outdir / topology,
            seed=args.seed,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            max_features=args.max_features,
            bootstrap=args.bootstrap,
        )

if __name__ == "__main__":
    main()

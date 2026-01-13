import numpy as np
from math_utils import safe_skew, safe_kurtosis, gini, entropy_from_counts


def bitrate_features(bitrate: np.ndarray) -> dict:
    """
    Extract statistical features from bitrate array
    """
    if bitrate is None or bitrate.size == 0:
        return {}

    return {
        "bitrate_mean": float(bitrate.mean()),
        "bitrate_std": float(bitrate.std()),
        "bitrate_min": float(bitrate.min()),
        "bitrate_max": float(bitrate.max()),
        "bitrate_skew": float(safe_skew(bitrate)),
        "bitrate_kurt": float(safe_kurtosis(bitrate)),
        "bitrate_gini": float(gini(bitrate)),
    }


def extract_features(
    req: dict,
    eig_k: int = 24,
    wl_bins: int = 128,
    nmf_k: int = 6,
    seed: int = 42,
) -> dict:
    """
    Main feature extraction entry point used by dataset.py

    Parameters
    ----------
    req : dict
        One request/sample containing traffic statistics
    eig_k, wl_bins, nmf_k, seed :
        Placeholder parameters for future extensions

    Returns
    -------
    dict
        Flat feature dictionary for ML models
    """

    features = {}

    # ---- Bitrate features (Group A example) ----
    bitrate = np.asarray(req.get("bitrate", []), dtype=float)
    features.update(bitrate_features(bitrate))

    # ---- Packet count / distribution features (Group B placeholder) ----
    pkt_sizes = np.asarray(req.get("packet_sizes", []), dtype=float)
    if pkt_sizes.size > 0:
        features.update({
            "pkt_mean": float(pkt_sizes.mean()),
            "pkt_std": float(pkt_sizes.std()),
            "pkt_entropy": float(entropy_from_counts(pkt_sizes.astype(int))),
        })

    # ---- Traffic volume / connectivity (Group C placeholder) ----
    total_bytes = float(np.sum(bitrate)) if bitrate.size > 0 else 0.0
    features["total_bytes"] = total_bytes
    features["num_packets"] = int(pkt_sizes.size)

    return features

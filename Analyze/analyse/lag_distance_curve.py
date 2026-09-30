"""Lag-Distance Curve for video representation.

For different temporal lags Δt:

d(Δt) = 1 / (T - Δt) * Σ ||s_{t+Δt} - s_t||_2

where s_t is the pooled feature of frame t.
This curve shows how representations change at different temporal scales.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_lag_distance_curve(pooled_features, max_lag=None):
    """Compute average L2 distance for each temporal lag."""
    pooled_features = np.asarray(pooled_features)
    num_frames = len(pooled_features)
    if num_frames < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), []

    if max_lag is None:
        max_lag = num_frames - 1
    max_lag = min(int(max_lag), num_frames - 1)

    lags = np.arange(1, max_lag + 1, dtype=np.int64)
    distances = []
    per_lag_distances = []

    for lag in lags:
        diffs = pooled_features[lag:] - pooled_features[:-lag]
        lag_distances = np.linalg.norm(diffs, axis=1)
        distances.append(float(lag_distances.mean()))
        per_lag_distances.append(lag_distances.tolist())

    return lags, np.array(distances, dtype=np.float32), per_lag_distances


def make_lag_distance_figure(lags, distances, save_path):
    """Save lag-distance curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(lags, distances, marker="o", markersize=3,
            linewidth=1.2, color="darkorange")
    ax.set_xlabel("Temporal Lag Δt (frames)")
    ax.set_ylabel("Mean L2 Distance")
    ax.set_title("Lag-Distance Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_lag_distance_summary(lags, distances, per_lag_distances,
                              num_frames, save_path):
    """Save JSON summary."""
    summary = {
        "num_frames": int(num_frames),
        "max_lag": int(lags[-1]) if len(lags) else 0,
        "lags": lags.tolist(),
        "mean_l2_distances": distances.tolist(),
        "per_lag_l2_distances": [
            {"lag": int(lag), "distances": values}
            for lag, values in zip(lags, per_lag_distances)
        ],
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

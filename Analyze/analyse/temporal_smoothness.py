"""Temporal Smoothness for video representation.

TS = 1 / (T - 1) * Σ ||s_{t+1} - s_t||_2

where s_t is the pooled feature of frame t.
Small TS means adjacent frame representations are stable.
Large TS means representation jitter is strong.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_temporal_smoothness(pooled_features):
    """Compute average L2 distance between adjacent pooled frame features."""
    pooled_features = np.asarray(pooled_features)
    if len(pooled_features) < 2:
        return 0.0, np.array([], dtype=np.float32)
    diffs = pooled_features[1:] - pooled_features[:-1]
    distances = np.linalg.norm(diffs, axis=1)
    return float(distances.mean()), distances


def make_temporal_smoothness_figure(distances, temporal_smoothness, save_path):
    """Save per-transition distance curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(np.arange(1, len(distances) + 1), distances,
            marker="o", markersize=3, linewidth=1.2, color="teal")
    ax.axhline(temporal_smoothness, color="red", linestyle="--",
               linewidth=1, label=f"TS = {temporal_smoothness:.4f}")
    ax.set_xlabel("Frame transition t → t+1")
    ax.set_ylabel("L2 distance")
    ax.set_title("Temporal Smoothness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_temporal_smoothness_summary(temporal_smoothness, distances,
                                     num_frames, save_path):
    """Save JSON summary."""
    summary = {
        "temporal_smoothness": temporal_smoothness,
        "num_frames": int(num_frames),
        "num_transitions": int(len(distances)),
        "mean_adjacent_l2": temporal_smoothness,
        "std_adjacent_l2": float(distances.std()) if len(distances) else 0.0,
        "min_adjacent_l2": float(distances.min()) if len(distances) else 0.0,
        "max_adjacent_l2": float(distances.max()) if len(distances) else 0.0,
        "adjacent_l2_distances": distances.tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

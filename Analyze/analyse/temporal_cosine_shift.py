"""Temporal Cosine Shift for video representation.

TCS = 1 / (T - 1) * Σ (1 - cos(s_t, s_{t+1}))

where s_t is the pooled feature of frame t.
Compared with Temporal Smoothness, this removes the influence of feature norm scale.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_temporal_cosine_shift(pooled_features):
    """Compute average cosine shift between adjacent pooled frame features."""
    pooled_features = np.asarray(pooled_features)
    if len(pooled_features) < 2:
        return 0.0, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    prev = pooled_features[:-1]
    nxt = pooled_features[1:]
    prev_norm = prev / (np.linalg.norm(prev, axis=1, keepdims=True) + 1e-8)
    nxt_norm = nxt / (np.linalg.norm(nxt, axis=1, keepdims=True) + 1e-8)
    cosine = (prev_norm * nxt_norm).sum(axis=1)
    shifts = 1.0 - cosine
    return float(shifts.mean()), shifts, cosine


def make_temporal_cosine_shift_figure(shifts, temporal_cosine_shift, save_path):
    """Save per-transition cosine shift curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(np.arange(1, len(shifts) + 1), shifts,
            marker="o", markersize=3, linewidth=1.2, color="purple")
    ax.axhline(temporal_cosine_shift, color="red", linestyle="--",
               linewidth=1, label=f"TCS = {temporal_cosine_shift:.4f}")
    ax.set_xlabel("Frame transition t → t+1")
    ax.set_ylabel("1 - cosine similarity")
    ax.set_title("Temporal Cosine Shift")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_temporal_cosine_shift_summary(temporal_cosine_shift, shifts,
                                       cosine, num_frames, save_path):
    """Save JSON summary."""
    summary = {
        "temporal_cosine_shift": temporal_cosine_shift,
        "num_frames": int(num_frames),
        "num_transitions": int(len(shifts)),
        "mean_shift": temporal_cosine_shift,
        "std_shift": float(shifts.std()) if len(shifts) else 0.0,
        "min_shift": float(shifts.min()) if len(shifts) else 0.0,
        "max_shift": float(shifts.max()) if len(shifts) else 0.0,
        "mean_adjacent_cosine": float(cosine.mean()) if len(cosine) else 0.0,
        "adjacent_cosine_shifts": shifts.tolist(),
        "adjacent_cosines": cosine.tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

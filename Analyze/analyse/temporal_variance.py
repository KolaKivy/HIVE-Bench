"""Temporal Variance for video representation.

For a trajectory of pooled frame features S ∈ R^{T×D}:

v_d^time = Var(S[:, d])

This measures which feature channels change over time.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_temporal_variance(pooled_features):
    """Compute per-channel temporal variance over pooled frame features."""
    pooled_features = np.asarray(pooled_features)
    if len(pooled_features) == 0:
        return np.array([], dtype=np.float32), 0.0, 0.0
    channel_vars = pooled_features.var(axis=0)
    return channel_vars, float(channel_vars.mean()), float(channel_vars.max())


def make_temporal_variance_figure(channel_vars, mean_var, max_var, save_path):
    """Save channel temporal variance spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(np.arange(len(channel_vars)), channel_vars,
                 linewidth=1.0, color="steelblue")
    axes[0].axhline(mean_var, color="red", linestyle="--",
                    linewidth=1, label=f"mean={mean_var:.6f}")
    axes[0].set_xlabel("Channel index")
    axes[0].set_ylabel("Temporal Variance")
    axes[0].set_title("Per-channel Temporal Variance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    sorted_vars = np.sort(channel_vars)[::-1]
    axes[1].plot(np.arange(len(sorted_vars)), sorted_vars,
                 linewidth=1.0, color="darkorange")
    axes[1].set_xlabel("Sorted channel rank")
    axes[1].set_ylabel("Temporal Variance")
    axes[1].set_title(f"Sorted Spectrum (max={max_var:.6f})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_temporal_variance_summary(channel_vars, mean_var, max_var,
                                   num_frames, save_path):
    """Save JSON summary."""
    if len(channel_vars):
        sorted_idx = np.argsort(channel_vars)[::-1]
        top_k = min(20, len(channel_vars))
        top_channels = [
            {"channel": int(idx), "variance": float(channel_vars[idx])}
            for idx in sorted_idx[:top_k]
        ]
    else:
        top_channels = []

    summary = {
        "num_frames": int(num_frames),
        "num_channels": int(len(channel_vars)),
        "mean_temporal_variance": mean_var,
        "max_temporal_variance": max_var,
        "std_temporal_variance": float(channel_vars.std()) if len(channel_vars) else 0.0,
        "min_temporal_variance": float(channel_vars.min()) if len(channel_vars) else 0.0,
        "top_channels": top_channels,
        "channel_temporal_variances": channel_vars.tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

"""Temporal Effective Rank for video representation.

Treat pooled frame features S ∈ R^{T×D} as a sample matrix.
Steps:
    1. Center S along time.
    2. Compute temporal covariance in feature space.
    3. Compute effective rank / participation ratio.

Effective rank = (Σ λ)^2 / Σ λ^2

Large → dynamic process spans many independent variation directions.
Small → process is compressed into a few temporal modes.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_temporal_effective_rank(pooled_features):
    """Compute temporal effective rank from pooled frame features."""
    S = np.asarray(pooled_features, dtype=np.float32)
    if len(S) < 2:
        return 0.0, np.array([], dtype=np.float32)

    centered = S - S.mean(axis=0, keepdims=True)
    cov = centered @ centered.T / max(len(S) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)[::-1]

    eig_sum = eigenvalues.sum()
    eff_rank = float((eig_sum ** 2) / (np.square(eigenvalues).sum() + 1e-12))
    return eff_rank, eigenvalues


def make_temporal_rank_figure(eigenvalues, eff_rank, save_path):
    """Save eigenvalue spectrum figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues,
                 marker="o", markersize=2, linewidth=1.0, color="steelblue")
    axes[0].set_xlabel("Temporal eigenvalue rank")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title(f"Temporal Spectrum (EffRank={eff_rank:.2f})")
    axes[0].grid(True, alpha=0.3)

    if len(eigenvalues):
        total = eigenvalues.sum() + 1e-12
        cumulative = np.cumsum(eigenvalues) / total
    else:
        cumulative = np.array([], dtype=np.float32)
    axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative,
                 linewidth=1.2, color="darkorange")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Temporal eigenvalue rank")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("Cumulative Spectrum")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_temporal_rank_summary(eff_rank, eigenvalues, num_frames, save_path):
    """Save JSON summary."""
    total = float(eigenvalues.sum()) if len(eigenvalues) else 0.0
    if total > 0:
        cumulative = np.cumsum(eigenvalues) / total
        rank_90 = int(np.searchsorted(cumulative, 0.9) + 1)
        rank_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    else:
        cumulative = np.array([], dtype=np.float32)
        rank_90 = 0
        rank_95 = 0

    summary = {
        "temporal_effective_rank": eff_rank,
        "num_frames": int(num_frames),
        "num_eigenvalues": int(len(eigenvalues)),
        "rank_90": rank_90,
        "rank_95": rank_95,
        "top_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) else 0.0,
        "eigenvalue_sum": total,
        "eigenvalues": eigenvalues.tolist(),
        "cumulative_explained_variance": cumulative.tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

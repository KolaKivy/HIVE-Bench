"""Token Covariance Rank (Effective Rank / Participation Ratio).

Given Z ∈ R^{N×D} (N tokens, D dims):
    1. Center tokens along token dimension.
    2. Compute covariance Σ_token ∈ R^{D×D}.
    3. Compute effective rank via participation ratio (PR):
       PR = (Σ λ_i)^2 / Σ λ_i^2

Measures how many dimensions the local structure occupies.
- Large: high intra-image structural complexity.
- Small: tokens are homogeneous.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def compute_token_cov_rank(features):
    """Compute effective rank (participation ratio) of token covariance.

    Args:
        features: np.ndarray (N, D) patch features

    Returns:
        eff_rank:    float, participation ratio
        eigenvalues: np.ndarray (D,), sorted eigenvalues (descending)
    """
    # 1. center
    centered = features - features.mean(axis=0, keepdims=True)
    # 2. covariance  D×D
    N = centered.shape[0]
    cov = (centered.T @ centered) / (N - 1)
    # 3. eigenvalues
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)  # numerical safety
    eigvals = eigvals[::-1]  # descending
    # participation ratio
    sum_lambda = eigvals.sum()
    sum_lambda2 = (eigvals ** 2).sum()
    eff_rank = float(sum_lambda ** 2 / (sum_lambda2 + 1e-12))
    return eff_rank, eigvals


def make_rank_figure(image_path, eigenvalues, eff_rank, save_path):
    """Save a figure: original image (left), eigenvalue spectrum (right)."""
    img = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].semilogy(eigenvalues, color="teal", linewidth=1.2)
    axes[1].axvline(x=eff_rank, color="red", linestyle="--", linewidth=1,
                    label=f"Eff. Rank = {eff_rank:.1f}")
    axes[1].set_xlabel("Component index")
    axes[1].set_ylabel("Eigenvalue (log)")
    axes[1].set_title("Covariance Eigenvalue Spectrum")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_rank_summary(eff_rank, eigenvalues, save_path):
    """Save a JSON summary of covariance rank statistics."""
    total_var = float(eigenvalues.sum())
    # cumulative explained variance ratio
    cumvar = np.cumsum(eigenvalues) / (total_var + 1e-12)
    rank_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    rank_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    summary = {
        "effective_rank": eff_rank,
        "total_variance": total_var,
        "rank_90pct": rank_90,
        "rank_95pct": rank_95,
        "rank_99pct": rank_99,
        "num_dims": int(len(eigenvalues)),
        "top5_eigenvalues": eigenvalues[:5].tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

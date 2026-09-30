"""Token Norm Entropy: measure diversity of token intensity distribution.

Steps:
    1. Compute r_i = ||z_i||_2 for each patch token.
    2. Build histogram of {r_i} to get bin probabilities q_b.
    3. H_token-norm = -Σ q_b log(q_b)

Low entropy → dominated by few strong tokens or all tokens nearly equal.
High entropy → richer distribution of token strengths.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def compute_token_norm_entropy(features, num_bins=30):
    """Compute entropy of the token-norm histogram.

    Args:
        features: np.ndarray (N, D) patch features
        num_bins: int, number of histogram bins

    Returns:
        norms:   np.ndarray (N,), per-token L2 norms
        entropy: float, H_token-norm
        hist:    np.ndarray (num_bins,), bin counts
        bin_edges: np.ndarray (num_bins+1,)
    """
    norms = np.linalg.norm(features, axis=1)  # (N,)
    hist, bin_edges = np.histogram(norms, bins=num_bins)
    # normalize to probabilities
    q = hist.astype(np.float64) / hist.sum()
    # entropy: -Σ q_b log(q_b), skip zero bins
    nonzero = q > 0
    entropy = -float((q[nonzero] * np.log(q[nonzero])).sum())
    return norms, entropy, hist, bin_edges


def make_entropy_figure(image_path, norms, entropy, hist, bin_edges, save_path):
    """Save a figure: original image (left), norm histogram (right)."""
    img = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axes[1].bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]) * 0.9,
                color="steelblue", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Token Norm")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Norm Histogram  (H={entropy:.4f})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_entropy_summary(norms, entropy, save_path):
    """Save a JSON summary of entropy statistics."""
    summary = {
        "entropy": entropy,
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "num_tokens": int(len(norms)),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

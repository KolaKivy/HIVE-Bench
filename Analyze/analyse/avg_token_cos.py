"""Average Pairwise Token Cosine Similarity.

AvgTokenCos = 1 / (N(N-1)) * Σ_{i≠j} cos(z_i, z_j)

Measures whether tokens within an image are too similar (homogenized).
- Large: tokens homogenized, local information may be smoothed out.
- Small: tokens are more discriminative.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def compute_avg_token_cos(features):
    """Compute average pairwise cosine similarity among all tokens.

    Args:
        features: np.ndarray (N, D) patch features

    Returns:
        avg_cos:    float, average pairwise cosine similarity
        cos_matrix: np.ndarray (N, N), full cosine similarity matrix
    """
    # L2 normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normed = features / (norms + 1e-8)
    # cosine similarity matrix
    cos_matrix = normed @ normed.T  # (N, N)
    N = cos_matrix.shape[0]
    # exclude diagonal (i==j), average over N*(N-1) pairs
    avg_cos = float((cos_matrix.sum() - np.trace(cos_matrix)) / (N * (N - 1)))
    return avg_cos, cos_matrix


def make_cos_figure(image_path, cos_matrix, avg_cos, save_path):
    """Save a figure: original image (left), cosine similarity matrix (right)."""
    img = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(cos_matrix, cmap="coolwarm", vmin=-1, vmax=1,
                        interpolation="nearest")
    axes[1].set_title(f"Cosine Sim Matrix (avg={avg_cos:.4f})")
    axes[1].set_xlabel("Token j")
    axes[1].set_ylabel("Token i")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_cos_summary(avg_cos, cos_matrix, save_path):
    """Save a JSON summary of cosine similarity statistics."""
    N = cos_matrix.shape[0]
    # off-diagonal values
    mask = ~np.eye(N, dtype=bool)
    off_diag = cos_matrix[mask]
    summary = {
        "avg_cosine_sim": avg_cos,
        "std_cosine_sim": float(off_diag.std()),
        "min_cosine_sim": float(off_diag.min()),
        "max_cosine_sim": float(off_diag.max()),
        "num_tokens": int(N),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

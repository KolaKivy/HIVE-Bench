"""Mean Token Norm: measure overall patch response intensity.

r_i = ||z_i||_2        (L2 norm of each patch token)
r̄   = (1/N) Σ r_i     (mean over all patches)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def compute_mean_token_norm(features):
    """Compute per-token L2 norms and their mean.

    Args:
        features: np.ndarray (N, D) patch features

    Returns:
        norms:     np.ndarray (N,), per-token L2 norms
        mean_norm: float, mean of all token norms
        std_norm:  float, std of all token norms
    """
    norms = np.linalg.norm(features, axis=1)  # (N,)
    return norms, float(norms.mean()), float(norms.std())


def make_norm_heatmap(image_path, norms, grid_h, grid_w, save_path):
    """Save a figure with original image (left) and token-norm heatmap (right)."""
    img = Image.open(image_path).convert("RGB")
    norm_map = norms.reshape(grid_h, grid_w)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(norm_map, cmap="hot", interpolation="nearest")
    axes[1].set_title("Token Norm Heatmap")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_norm_summary(norms, mean_norm, std_norm, save_path):
    """Save a JSON summary of norm statistics."""
    summary = {
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "num_tokens": int(len(norms)),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

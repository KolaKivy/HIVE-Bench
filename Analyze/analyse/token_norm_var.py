"""Token Norm Variance: measure whether local patch responses are balanced.

Var(r_1, ..., r_N)  where r_i = ||z_i||_2

Large variance → responses concentrated in few regions.
Small variance → patches respond more uniformly.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def compute_token_norm_var(features):
    """Compute variance of per-token L2 norms.

    Args:
        features: np.ndarray (N, D) patch features

    Returns:
        norms:    np.ndarray (N,), per-token L2 norms
        var_norm: float, variance of token norms
    """
    norms = np.linalg.norm(features, axis=1)  # (N,)
    var_norm = float(norms.var())
    return norms, var_norm


def make_var_heatmap(image_path, norms, grid_h, grid_w, var_norm, save_path):
    """Save a figure with original image (left) and norm deviation heatmap (right)."""
    img = Image.open(image_path).convert("RGB")
    norm_map = norms.reshape(grid_h, grid_w)
    mean_val = norms.mean()
    # show deviation from mean to highlight variance structure
    dev_map = (norm_map - mean_val) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(dev_map, cmap="magma", interpolation="nearest")
    axes[1].set_title(f"Squared Deviation (Var={var_norm:.4f})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_var_summary(norms, var_norm, save_path):
    """Save a JSON summary of norm variance statistics."""
    summary = {
        "variance": var_norm,
        "std": float(norms.std()),
        "mean_norm": float(norms.mean()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "num_tokens": int(len(norms)),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

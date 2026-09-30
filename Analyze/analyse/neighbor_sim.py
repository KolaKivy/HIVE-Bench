"""Neighbor Similarity: average cosine similarity between spatially adjacent patches.

NS = (1/|N|) Σ_{(i,j)∈N} cos(z_i, z_j)

where N is the set of 4-connected neighbor pairs on the patch grid.

Measures local spatial smoothness.
- Large: locally smooth features.
- Too large: over-smooth, boundaries erased.
- Too small: noisy / unstable.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def compute_neighbor_sim(features, grid_h, grid_w):
    """Compute average cosine similarity between 4-connected neighbor patches.

    Args:
        features: np.ndarray (N, D) patch features, row-major order
        grid_h: int
        grid_w: int

    Returns:
        avg_ns:      float, average neighbor cosine similarity
        ns_map:      np.ndarray (grid_h, grid_w), per-patch mean neighbor similarity
    """
    feat_grid = features.reshape(grid_h, grid_w, -1)
    # L2 normalize
    norms = np.linalg.norm(feat_grid, axis=2, keepdims=True)
    normed = feat_grid / (norms + 1e-8)

    ns_map = np.zeros((grid_h, grid_w), dtype=np.float64)
    count_map = np.zeros((grid_h, grid_w), dtype=np.float64)
    total_sim = 0.0
    num_pairs = 0

    # horizontal pairs
    if grid_w > 1:
        h_sim = (normed[:, :-1, :] * normed[:, 1:, :]).sum(axis=2)
        total_sim += h_sim.sum()
        num_pairs += h_sim.size
        ns_map[:, :-1] += h_sim
        ns_map[:, 1:] += h_sim
        count_map[:, :-1] += 1
        count_map[:, 1:] += 1

    # vertical pairs
    if grid_h > 1:
        v_sim = (normed[:-1, :, :] * normed[1:, :, :]).sum(axis=2)
        total_sim += v_sim.sum()
        num_pairs += v_sim.size
        ns_map[:-1, :] += v_sim
        ns_map[1:, :] += v_sim
        count_map[:-1, :] += 1
        count_map[1:, :] += 1

    avg_ns = float(total_sim / max(num_pairs, 1))
    ns_map = ns_map / np.maximum(count_map, 1)
    return avg_ns, ns_map


def make_ns_figure(image_path, ns_map, avg_ns, save_path):
    """Save a figure: original image (left), neighbor similarity map (right)."""
    img = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(ns_map, cmap="viridis", interpolation="nearest",
                        vmin=0, vmax=1)
    axes[1].set_title(f"Neighbor Similarity (avg={avg_ns:.4f})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_ns_summary(avg_ns, ns_map, save_path):
    """Save a JSON summary of neighbor similarity statistics."""
    summary = {
        "avg_neighbor_sim": avg_ns,
        "std_neighbor_sim": float(ns_map.std()),
        "min_neighbor_sim": float(ns_map.min()),
        "max_neighbor_sim": float(ns_map.max()),
        "grid_h": int(ns_map.shape[0]),
        "grid_w": int(ns_map.shape[1]),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

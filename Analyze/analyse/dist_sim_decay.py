"""Distance-Similarity Decay: how cosine similarity decays with spatial distance.

Steps:
    1. Compute spatial distance d_ij^spatial for every patch pair.
    2. Compute feature similarity s_ij = cos(z_i, z_j).
    3. Bucket by distance, compute mean similarity per bucket.

s(d) = E[ cos(z_i, z_j) | d_ij^spatial = d ]

Fast decay  → clear local structure, strong spatial contrast.
Slow decay  → tokens uniformly similar, spatial structure smoothed out.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def compute_dist_sim_decay(features, grid_h, grid_w, num_buckets=50):
    """Compute distance-similarity decay curve.

    Args:
        features: np.ndarray (N, D) patch features, row-major
        grid_h, grid_w: int
        num_buckets: int, number of distance buckets

    Returns:
        bucket_centers: np.ndarray (num_buckets,), distance bucket centers
        bucket_means:   np.ndarray (num_buckets,), mean cosine sim per bucket
        bucket_counts:  np.ndarray (num_buckets,), number of pairs per bucket
    """
    N = grid_h * grid_w
    # patch grid coordinates
    rows, cols = np.divmod(np.arange(N), grid_w)

    # spatial distances (Euclidean on grid)
    dr = rows[:, None] - rows[None, :]  # (N, N)
    dc = cols[:, None] - cols[None, :]
    spatial_dist = np.sqrt(dr.astype(np.float32) ** 2 + dc.astype(np.float32) ** 2)

    # cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normed = features / (norms + 1e-8)
    cos_sim = normed @ normed.T  # (N, N)

    # upper triangle (exclude diagonal)
    triu_idx = np.triu_indices(N, k=1)
    dists = spatial_dist[triu_idx]
    sims = cos_sim[triu_idx]

    # bucket
    max_dist = dists.max()
    bin_edges = np.linspace(0, max_dist + 1e-6, num_buckets + 1)
    bucket_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bucket_idx = np.digitize(dists, bin_edges) - 1
    bucket_idx = np.clip(bucket_idx, 0, num_buckets - 1)

    bucket_means = np.zeros(num_buckets)
    bucket_counts = np.zeros(num_buckets, dtype=np.int64)
    for b in range(num_buckets):
        mask = bucket_idx == b
        bucket_counts[b] = mask.sum()
        if bucket_counts[b] > 0:
            bucket_means[b] = sims[mask].mean()

    return bucket_centers, bucket_means, bucket_counts


def make_decay_figure(image_path, bucket_centers, bucket_means, bucket_counts,
                      save_path):
    """Save a figure: original image (left), decay curve (right)."""
    img = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # only plot buckets with data
    valid = bucket_counts > 0
    axes[1].plot(bucket_centers[valid], bucket_means[valid],
                 marker="o", markersize=3, linewidth=1.2, color="teal")
    axes[1].set_xlabel("Spatial Distance (patches)")
    axes[1].set_ylabel("Mean Cosine Similarity")
    axes[1].set_title("Distance-Similarity Decay")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_decay_summary(bucket_centers, bucket_means, bucket_counts, save_path):
    """Save a JSON summary of the decay curve."""
    valid = bucket_counts > 0
    summary = {
        "distances": bucket_centers[valid].tolist(),
        "mean_similarities": bucket_means[valid].tolist(),
        "pair_counts": bucket_counts[valid].tolist(),
        "sim_at_dist_1": None,
        "sim_at_max_dist": None,
    }
    if valid.any():
        summary["sim_at_dist_1"] = float(bucket_means[valid][0])
        summary["sim_at_max_dist"] = float(bucket_means[valid][-1])
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

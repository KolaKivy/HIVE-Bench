"""Token-to-Global Similarity (and its variance).

g = z̄ = (1/N) Σ z_i          (global mean token)
s_i = cos(z_i, g)             (per-token similarity to global)

Metrics:
- mean(s_i): how strongly the global component dominates.
- Var(s_i):  whether all tokens are equally coupled to global.
  - Small var + high mean: global dominates, little local individuality.
  - Large var: different regions relate differently to the global.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def compute_token_to_global(features):
    """Compute per-token cosine similarity to the global mean token.

    Args:
        features: np.ndarray (N, D) patch features

    Returns:
        sims:     np.ndarray (N,), per-token cos similarity to global
        mean_sim: float
        var_sim:  float
    """
    g = features.mean(axis=0, keepdims=True)  # (1, D)
    # normalize
    g_norm = g / (np.linalg.norm(g) + 1e-8)
    feat_norms = np.linalg.norm(features, axis=1, keepdims=True)
    feat_normed = features / (feat_norms + 1e-8)
    sims = (feat_normed * g_norm).sum(axis=1)  # (N,)
    return sims, float(sims.mean()), float(sims.var())


def make_token_to_global_figure(image_path, sims, grid_h, grid_w,
                                mean_sim, var_sim, save_path):
    """Save a figure: original (left), token-to-global similarity map (right)."""
    img = Image.open(image_path).convert("RGB")
    sim_map = sims.reshape(grid_h, grid_w)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(sim_map, cmap="RdYlGn", interpolation="nearest",
                        vmin=0, vmax=1)
    axes[1].set_title(f"Token-to-Global Sim (mean={mean_sim:.4f}, var={var_sim:.4f})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_token_to_global_summary(sims, mean_sim, var_sim, save_path):
    """Save a JSON summary."""
    summary = {
        "mean_token_to_global_sim": mean_sim,
        "var_token_to_global_sim": var_sim,
        "std_token_to_global_sim": float(sims.std()),
        "min_sim": float(sims.min()),
        "max_sim": float(sims.max()),
        "num_tokens": int(len(sims)),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

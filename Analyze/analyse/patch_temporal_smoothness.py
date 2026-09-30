import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_patch_temporal_smoothness(patch_features):
    Z = np.asarray(patch_features, dtype=np.float32)
    if len(Z) < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    step_distances = np.linalg.norm(Z[1:] - Z[:-1], axis=2)
    patch_smoothness = step_distances.mean(axis=0)
    return patch_smoothness, step_distances


def make_patch_temporal_smoothness_figure(patch_smoothness, gh, gw, save_path):
    heatmap = patch_smoothness.reshape(gh, gw)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heatmap, cmap="magma")
    ax.set_title("Patch Temporal Smoothness")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean L2 step distance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_patch_temporal_smoothness_summary(patch_smoothness, step_distances, gh, gw, num_frames, save_path):
    top_idx = np.argsort(patch_smoothness)[::-1][:10] if len(patch_smoothness) else []
    summary = {
        "mean_patch_temporal_smoothness": float(patch_smoothness.mean()) if len(patch_smoothness) else 0.0,
        "max_patch_temporal_smoothness": float(patch_smoothness.max()) if len(patch_smoothness) else 0.0,
        "min_patch_temporal_smoothness": float(patch_smoothness.min()) if len(patch_smoothness) else 0.0,
        "std_patch_temporal_smoothness": float(patch_smoothness.std()) if len(patch_smoothness) else 0.0,
        "mean_step_patch_temporal_smoothness": float(step_distances.mean()) if len(step_distances) else 0.0,
        "num_frames": int(num_frames),
        "num_patches": int(len(patch_smoothness)),
        "grid_h": int(gh),
        "grid_w": int(gw),
        "top_dynamic_patch_indices": [int(idx) for idx in top_idx],
        "top_dynamic_patch_values": [float(patch_smoothness[idx]) for idx in top_idx],
        "patch_temporal_smoothness": patch_smoothness.tolist(),
        "patch_temporal_smoothness_map": patch_smoothness.reshape(gh, gw).tolist() if len(patch_smoothness) else [],
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

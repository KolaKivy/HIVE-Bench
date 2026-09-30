import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_temporal_token_norm_entropy(patch_features, num_bins=16):
    Z = np.asarray(patch_features, dtype=np.float32)
    if len(Z) < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    norms = np.linalg.norm(Z, axis=2)
    entropy = np.zeros(norms.shape[1], dtype=np.float32)
    for patch_idx in range(norms.shape[1]):
        values = norms[:, patch_idx]
        hist, _ = np.histogram(values, bins=int(num_bins), density=False)
        total = hist.sum()
        if total == 0:
            continue
        probs = hist.astype(np.float64) / total
        probs = probs[probs > 0]
        entropy[patch_idx] = float(-(probs * np.log(probs + 1e-12)).sum())

    normalized_entropy = entropy / np.log(int(num_bins)) if int(num_bins) > 1 else entropy
    return entropy, normalized_entropy


def make_temporal_token_norm_entropy_figure(entropy, normalized_entropy, gh, gw, save_path):
    entropy_map = entropy.reshape(gh, gw)
    normalized_map = normalized_entropy.reshape(gh, gw)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(entropy_map, cmap="viridis")
    axes[0].set_title("Temporal Token Norm Entropy")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(normalized_map, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Normalized Entropy")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_temporal_token_norm_entropy_summary(entropy, normalized_entropy, gh, gw, num_frames, num_bins, save_path):
    top_idx = np.argsort(normalized_entropy)[::-1][:10] if len(normalized_entropy) else []
    summary = {
        "mean_temporal_token_norm_entropy": float(entropy.mean()) if len(entropy) else 0.0,
        "max_temporal_token_norm_entropy": float(entropy.max()) if len(entropy) else 0.0,
        "min_temporal_token_norm_entropy": float(entropy.min()) if len(entropy) else 0.0,
        "std_temporal_token_norm_entropy": float(entropy.std()) if len(entropy) else 0.0,
        "mean_normalized_temporal_token_norm_entropy": float(normalized_entropy.mean()) if len(normalized_entropy) else 0.0,
        "max_normalized_temporal_token_norm_entropy": float(normalized_entropy.max()) if len(normalized_entropy) else 0.0,
        "num_frames": int(num_frames),
        "num_patches": int(len(entropy)),
        "grid_h": int(gh),
        "grid_w": int(gw),
        "num_bins": int(num_bins),
        "top_entropy_patch_indices": [int(idx) for idx in top_idx],
        "top_entropy_patch_values": [float(normalized_entropy[idx]) for idx in top_idx],
        "temporal_token_norm_entropy": entropy.tolist(),
        "normalized_temporal_token_norm_entropy": normalized_entropy.tolist(),
        "normalized_temporal_token_norm_entropy_map": normalized_entropy.reshape(gh, gw).tolist() if len(normalized_entropy) else [],
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

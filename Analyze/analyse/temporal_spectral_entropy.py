import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_temporal_spectral_entropy(pooled_features):
    S = np.asarray(pooled_features, dtype=np.float32)
    if len(S) < 2:
        return 0.0, 0.0, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    centered = S - S.mean(axis=0, keepdims=True)
    cov = centered @ centered.T / max(len(S) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)[::-1]

    total = eigenvalues.sum()
    if total <= 0:
        probs = np.zeros_like(eigenvalues)
        return 0.0, 0.0, eigenvalues, probs

    probs = eigenvalues / total
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    normalized_entropy = float(entropy / np.log(len(probs))) if len(probs) > 1 else 0.0
    return entropy, normalized_entropy, eigenvalues, probs


def make_temporal_spectral_entropy_figure(eigenvalues, probs, entropy, normalized_entropy, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues,
                 marker="o", markersize=2, linewidth=1.0, color="steelblue")
    axes[0].set_xlabel("Temporal eigenvalue rank")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Temporal Covariance Spectrum")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(np.arange(1, len(probs) + 1), probs, color="darkorange")
    axes[1].set_xlabel("Temporal eigenvalue rank")
    axes[1].set_ylabel("Normalized eigenvalue weight")
    axes[1].set_title(f"Spectral Entropy={entropy:.3f}, Norm={normalized_entropy:.3f}")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_temporal_spectral_entropy_summary(entropy, normalized_entropy, eigenvalues, probs, num_frames, save_path):
    summary = {
        "temporal_spectral_entropy": entropy,
        "normalized_temporal_spectral_entropy": normalized_entropy,
        "num_frames": int(num_frames),
        "num_eigenvalues": int(len(eigenvalues)),
        "top_probability": float(probs[0]) if len(probs) else 0.0,
        "eigenvalue_sum": float(eigenvalues.sum()) if len(eigenvalues) else 0.0,
        "eigenvalues": eigenvalues.tolist(),
        "eigenvalue_probabilities": probs.tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_total_trajectory_variation(pooled_features):
    S = np.asarray(pooled_features, dtype=np.float32)
    if len(S) < 2:
        return 0.0, 0.0, np.array([], dtype=np.float32)

    step_distances = np.linalg.norm(S[1:] - S[:-1], axis=1)
    ttv = float(step_distances.sum())
    normalized_ttv = float(ttv / len(S))
    return ttv, normalized_ttv, step_distances


def make_ttv_figure(step_distances, ttv, normalized_ttv, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(1, len(step_distances) + 1)
    axes[0].plot(x, step_distances, linewidth=1.2, color="steelblue")
    axes[0].axhline(step_distances.mean() if len(step_distances) else 0.0,
                    color="darkorange", linestyle="--", linewidth=1,
                    label="mean step")
    axes[0].set_xlabel("Step index")
    axes[0].set_ylabel("L2 distance")
    axes[0].set_title("Frame-to-frame trajectory variation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    cumulative = np.cumsum(step_distances) if len(step_distances) else np.array([])
    axes[1].plot(x, cumulative, linewidth=1.2, color="seagreen")
    axes[1].set_xlabel("Step index")
    axes[1].set_ylabel("Cumulative L2 distance")
    axes[1].set_title(f"TTV={ttv:.3f}, TTV/T={normalized_ttv:.4f}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_ttv_summary(ttv, normalized_ttv, step_distances, num_frames, save_path):
    summary = {
        "total_trajectory_variation": ttv,
        "normalized_total_trajectory_variation": normalized_ttv,
        "mean_step_variation": float(step_distances.mean()) if len(step_distances) else 0.0,
        "max_step_variation": float(step_distances.max()) if len(step_distances) else 0.0,
        "std_step_variation": float(step_distances.std()) if len(step_distances) else 0.0,
        "num_frames": int(num_frames),
        "num_steps": int(len(step_distances)),
        "step_distances": step_distances.tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

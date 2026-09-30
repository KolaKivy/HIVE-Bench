import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrajectoryVarRatioAccumulator:
    def __init__(self):
        self.records = []

    def add(self, name, pooled_features):
        S = np.asarray(pooled_features, dtype=np.float32)
        if len(S) < 2:
            within_var = 0.0
        else:
            within_var = float(S.var(axis=0).mean())
        mean_feature = S.mean(axis=0) if len(S) else None
        record = {
            "name": name,
            "within_trajectory_variance": within_var,
            "num_frames": int(len(S)),
            "mean_feature": mean_feature,
        }
        self.records.append(record)
        return record

    def finalize(self):
        valid_records = [record for record in self.records if record["mean_feature"] is not None]
        if len(valid_records) < 2:
            between_var = 0.0
        else:
            means = np.stack([record["mean_feature"] for record in valid_records], axis=0)
            between_var = float(means.var(axis=0).mean())

        within_values = np.array(
            [record["within_trajectory_variance"] for record in valid_records],
            dtype=np.float64,
        )
        mean_within = float(within_values.mean()) if len(within_values) else 0.0
        ratio = float(mean_within / (between_var + 1e-12))

        videos = [
            {
                "name": record["name"],
                "within_trajectory_variance": record["within_trajectory_variance"],
                "num_frames": record["num_frames"],
            }
            for record in valid_records
        ]
        return {
            "trajectory_var_ratio": ratio,
            "mean_within_trajectory_variance": mean_within,
            "between_trajectory_variance": between_var,
            "num_videos": len(valid_records),
            "videos": videos,
        }


def make_trajectory_var_ratio_figure(summary, save_path):
    videos = summary["videos"]
    names = [record["name"] for record in videos]
    within_values = [record["within_trajectory_variance"] for record in videos]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(np.arange(len(within_values)), within_values, color="steelblue")
    axes[0].axhline(
        summary["mean_within_trajectory_variance"],
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"mean={summary['mean_within_trajectory_variance']:.6f}",
    )
    axes[0].set_title("Within-trajectory variance")
    axes[0].set_xticks(np.arange(len(within_values)))
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    labels = ["mean within", "between", "ratio"]
    values = [
        summary["mean_within_trajectory_variance"],
        summary["between_trajectory_variance"],
        summary["trajectory_var_ratio"],
    ]
    axes[1].bar(labels, values, color=["steelblue", "darkorange", "seagreen"])
    axes[1].set_title("Trajectory variance ratio")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_trajectory_var_ratio_summary(summary, json_path, fig_path):
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    make_trajectory_var_ratio_figure(summary, fig_path)

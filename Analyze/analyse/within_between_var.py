"""Within-image / Between-image Variance Ratio.

R = within-image variance / between-image variance

- Within-image variance: average variance of tokens relative to each image's mean.
- Between-image variance: variance of pooled features (image means) across images.

Large R → encoder emphasizes local/spatial structure within images.
Small R → encoder emphasizes global identity differences between images.
"""

import json
import numpy as np


class WithinBetweenVarAccumulator:
    """Accumulates per-image statistics, computes final ratio."""

    def __init__(self):
        self.within_vars = []   # per-image within-variance (scalar)
        self.image_means = []   # per-image mean token (D,)
        self.image_names = []

    def add_image(self, features, name):
        """Add one image's features.

        Args:
            features: np.ndarray (N, D)
            name: str, image identifier
        """
        mean_token = features.mean(axis=0)  # (D,)
        # within-image variance: mean over tokens of ||z_i - mean||^2
        within_var = float(((features - mean_token) ** 2).sum(axis=1).mean())
        self.within_vars.append(within_var)
        self.image_means.append(mean_token)
        self.image_names.append(name)

    def compute(self):
        """Compute the final ratio.

        Returns:
            ratio:        float, R = within / between
            within_var:   float, average within-image variance
            between_var:  float, between-image variance
            per_image:    list of dicts with per-image within_var
        """
        within_var = float(np.mean(self.within_vars))

        # between-image variance: variance of image means
        means_array = np.array(self.image_means)  # (M, D)
        global_mean = means_array.mean(axis=0)
        between_var = float(((means_array - global_mean) ** 2).sum(axis=1).mean())

        ratio = within_var / (between_var + 1e-12)

        per_image = [
            {"name": n, "within_var": float(wv)}
            for n, wv in zip(self.image_names, self.within_vars)
        ]
        return ratio, within_var, between_var, per_image


def save_within_between_summary(ratio, within_var, between_var, per_image,
                                save_path):
    """Save the final summary JSON."""
    summary = {
        "variance_ratio": ratio,
        "within_image_variance": within_var,
        "between_image_variance": between_var,
        "num_images": len(per_image),
        "per_image": per_image,
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

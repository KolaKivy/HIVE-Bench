"""PCA feature visualization: project patch features to RGB."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA


def pca_to_rgb(features, grid_h, grid_w):
    """Project (N, D) patch features to (grid_h, grid_w, 3) RGB image via PCA.

    Uses sigmoid(2x) mapping for vibrant colors (following DINOv3 paper).
    """
    pca = PCA(n_components=3, whiten=True)
    proj = pca.fit_transform(features)                       # (N, 3)
    proj = 1.0 / (1.0 + np.exp(-2.0 * proj))                # sigmoid(2x)
    return proj.reshape(grid_h, grid_w, 3)


def make_side_by_side(image_path, pca_rgb, save_path):
    """Save a figure with original image (left) and PCA feature map (right), same height."""
    img = Image.open(image_path).convert("RGB")
    h, w = pca_rgb.shape[:2]

    # upsample PCA map to match original image height for visual consistency
    orig_w, orig_h = img.size
    pca_img = Image.fromarray((pca_rgb * 255).astype(np.uint8))
    pca_img = pca_img.resize((orig_w, orig_h), Image.NEAREST)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(pca_img)
    axes[1].set_title("PCA Feature Map")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

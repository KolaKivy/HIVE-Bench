"""Frequency-domain metrics for ViT/CNN patch token feature maps.

This module is designed to match the calling style used in the existing
`analyse/*` modules:

    metrics, power_spectrum, radial_freq, radial_power = compute_frequency_metrics(tokens, H, W)
    make_frequency_figure(ref_img_path, power_spectrum, radial_freq, radial_power, metrics, save_path)
    save_frequency_summary(metrics, radial_freq, radial_power, save_path)

The six reported metrics are:
    1. low_ratio
    2. mid_ratio
    3. high_ratio
    4. spectral_entropy
    5. spectral_centroid
    6. spectral_bandwidth

Supported feature shapes:
    - (N, C), with grid_h and grid_w required; current run.py uses this.
    - (C, H, W)
    - (H, W, C), with channel_last=True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _to_chw(feature: np.ndarray, grid_h: int | None = None, grid_w: int | None = None, channel_last: bool = False) -> np.ndarray:
    """Convert feature to CHW numpy array.

    Args:
        feature: Feature array. Supported shapes: (N, C), (C, H, W), (H, W, C).
        grid_h: Token grid height. Required when feature is (N, C).
        grid_w: Token grid width. Required when feature is (N, C).
        channel_last: Set True when a 3D feature is (H, W, C).

    Returns:
        np.ndarray: Feature with shape (C, H, W), dtype float64.
    """
    x = np.asarray(feature)

    if x.ndim == 2:
        if grid_h is None or grid_w is None:
            raise ValueError("grid_h and grid_w are required when feature has shape (N, C).")
        n, c = x.shape
        if n != grid_h * grid_w:
            raise ValueError(f"feature has N={n}, but grid_h*grid_w={grid_h * grid_w}.")
        # (N, C) -> (H, W, C) -> (C, H, W)
        x = x.reshape(grid_h, grid_w, c).transpose(2, 0, 1)

    elif x.ndim == 3:
        if channel_last:
            # (H, W, C) -> (C, H, W)
            x = x.transpose(2, 0, 1)
        # else: assume already (C, H, W)

    else:
        raise ValueError(f"Unsupported feature shape {x.shape}; expected (N,C), (C,H,W), or (H,W,C).")

    return x.astype(np.float64, copy=False)


def _frequency_radius(h: int, w: int) -> np.ndarray:
    """Return normalized 2D frequency radius map with shape (H, W)."""
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    r = np.sqrt(fy_grid ** 2 + fx_grid ** 2)
    r = r / (r.max() + 1e-12)
    return r


def _radial_average(power_spectrum: np.ndarray, radius: np.ndarray, num_bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial averaged power spectrum.

    Args:
        power_spectrum: Mean power spectrum, shape (H, W), unshifted FFT order.
        radius: Normalized radius map, shape (H, W), values in [0, 1].
        num_bins: Number of radial bins. Defaults to max(8, min(H, W)//2 + 1).

    Returns:
        radial_freq: Bin centers.
        radial_power: Mean power inside each radial bin.
    """
    h, w = power_spectrum.shape
    if num_bins is None:
        num_bins = max(8, min(h, w) // 2 + 1)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    radial_freq = 0.5 * (bins[:-1] + bins[1:])
    radial_power = np.zeros(num_bins, dtype=np.float64)

    r_flat = radius.reshape(-1)
    p_flat = power_spectrum.reshape(-1)

    for i in range(num_bins):
        if i == num_bins - 1:
            mask = (r_flat >= bins[i]) & (r_flat <= bins[i + 1])
        else:
            mask = (r_flat >= bins[i]) & (r_flat < bins[i + 1])
        radial_power[i] = float(p_flat[mask].mean()) if np.any(mask) else 0.0

    return radial_freq, radial_power


def compute_frequency_metrics(
    feature: np.ndarray,
    grid_h: int | None = None,
    grid_w: int | None = None,
    low_thr: float = 0.15,
    mid_thr: float = 0.35,
    remove_dc: bool = True,
    normalize_channel: bool = False,
    channel_last: bool = False,
    num_radial_bins: int | None = None,
    eps: float = 1e-12,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Compute six frequency-domain metrics for one image/frame feature.

    This function calculates metrics per channel first, then averages over channels.
    This avoids letting a few high-energy channels dominate the final ratios.

    Args:
        feature: Feature array. In your current framework, pass tokens with shape (H*W, C).
        grid_h: Token grid height when feature is (N, C).
        grid_w: Token grid width when feature is (N, C).
        low_thr: Low-frequency radius threshold in normalized [0, 1] radius.
        mid_thr: Mid-frequency radius threshold in normalized [0, 1] radius.
        remove_dc: If True, subtract each channel's spatial mean before FFT.
        normalize_channel: If True, additionally divide each channel by spatial std.
        channel_last: Set True if a 3D feature is given as (H, W, C).
        num_radial_bins: Number of bins for radial spectrum visualization.
        eps: Numerical stability constant.

    Returns:
        metrics: Dict containing low/mid/high ratios, entropy, centroid, bandwidth, plus stds.
        power_spectrum: Channel-averaged 2D power spectrum, shape (H, W), unshifted.
        radial_freq: Radial frequency bin centers.
        radial_power: Radial averaged power values.
    """
    if not (0.0 <= low_thr < mid_thr <= 1.0):
        raise ValueError("Require 0 <= low_thr < mid_thr <= 1.")

    x = _to_chw(feature, grid_h, grid_w, channel_last=channel_last)  # (C, H, W)
    c, h, w = x.shape

    if remove_dc:
        x = x - x.mean(axis=(1, 2), keepdims=True)

    if normalize_channel:
        x = x / (x.std(axis=(1, 2), keepdims=True) + eps)

    # FFT over the spatial token grid.
    fft = np.fft.fft2(x, axes=(-2, -1), norm="ortho")
    power = np.abs(fft) ** 2  # (C, H, W)
    power_spectrum = power.mean(axis=0)  # (H, W), for visualization

    radius = _frequency_radius(h, w)
    r_flat = radius.reshape(-1)
    power_flat = power.reshape(c, -1)

    low_mask = r_flat <= low_thr
    mid_mask = (r_flat > low_thr) & (r_flat <= mid_thr)
    high_mask = r_flat > mid_thr

    total_energy = power_flat.sum(axis=1) + eps
    low_energy = power_flat[:, low_mask].sum(axis=1)
    mid_energy = power_flat[:, mid_mask].sum(axis=1)
    high_energy = power_flat[:, high_mask].sum(axis=1)

    low_ratio_per_channel = low_energy / total_energy
    mid_ratio_per_channel = mid_energy / total_energy
    high_ratio_per_channel = high_energy / total_energy

    probs = power_flat / total_energy[:, None]
    entropy_per_channel = -(probs * np.log(probs + eps)).sum(axis=1) / np.log(power_flat.shape[1])

    centroid_per_channel = (probs * r_flat[None, :]).sum(axis=1)
    bandwidth_per_channel = np.sqrt((probs * (r_flat[None, :] - centroid_per_channel[:, None]) ** 2).sum(axis=1))

    radial_freq, radial_power = _radial_average(power_spectrum, radius, num_bins=num_radial_bins)

    metrics = {
        "low_ratio": float(np.mean(low_ratio_per_channel)),
        "mid_ratio": float(np.mean(mid_ratio_per_channel)),
        "high_ratio": float(np.mean(high_ratio_per_channel)),
        "spectral_entropy": float(np.mean(entropy_per_channel)),
        "spectral_centroid": float(np.mean(centroid_per_channel)),
        "spectral_bandwidth": float(np.mean(bandwidth_per_channel)),
        "low_ratio_std": float(np.std(low_ratio_per_channel)),
        "mid_ratio_std": float(np.std(mid_ratio_per_channel)),
        "high_ratio_std": float(np.std(high_ratio_per_channel)),
        "spectral_entropy_std": float(np.std(entropy_per_channel)),
        "spectral_centroid_std": float(np.std(centroid_per_channel)),
        "spectral_bandwidth_std": float(np.std(bandwidth_per_channel)),
        "low_thr": float(low_thr),
        "mid_thr": float(mid_thr),
        "remove_dc": bool(remove_dc),
        "normalize_channel": bool(normalize_channel),
        "grid_h": int(h),
        "grid_w": int(w),
        "num_channels": int(c),
        "low_bin_count": int(np.sum(low_mask)),
        "mid_bin_count": int(np.sum(mid_mask)),
        "high_bin_count": int(np.sum(high_mask)),
    }

    return metrics, power_spectrum, radial_freq, radial_power


# Optional lightweight wrappers, if you want separate metric calls elsewhere.
def compute_frequency_energy_ratios(feature: np.ndarray, grid_h: int | None = None, grid_w: int | None = None, **kwargs: Any) -> dict[str, float]:
    metrics, _, _, _ = compute_frequency_metrics(feature, grid_h, grid_w, **kwargs)
    return {k: metrics[k] for k in ["low_ratio", "mid_ratio", "high_ratio"]}


def compute_spectral_entropy(feature: np.ndarray, grid_h: int | None = None, grid_w: int | None = None, **kwargs: Any) -> float:
    metrics, _, _, _ = compute_frequency_metrics(feature, grid_h, grid_w, **kwargs)
    return float(metrics["spectral_entropy"])


def compute_spectral_centroid(feature: np.ndarray, grid_h: int | None = None, grid_w: int | None = None, **kwargs: Any) -> float:
    metrics, _, _, _ = compute_frequency_metrics(feature, grid_h, grid_w, **kwargs)
    return float(metrics["spectral_centroid"])


def compute_spectral_bandwidth(feature: np.ndarray, grid_h: int | None = None, grid_w: int | None = None, **kwargs: Any) -> float:
    metrics, _, _, _ = compute_frequency_metrics(feature, grid_h, grid_w, **kwargs)
    return float(metrics["spectral_bandwidth"])


def make_frequency_figure(
    ref_img_path: str | None,
    power_spectrum: np.ndarray,
    radial_freq: np.ndarray,
    radial_power: np.ndarray,
    metrics: dict[str, Any],
    save_path: str,
) -> None:
    """Save a visualization figure for one image/frame.

    The figure contains reference image, log power spectrum, radial power spectrum,
    and low/mid/high energy ratios.
    """
    ncols = 4 if ref_img_path is not None and Path(ref_img_path).exists() else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.0))
    if ncols == 1:
        axes = [axes]

    ax_idx = 0
    if ncols == 4:
        img = cv2.imread(str(ref_img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[ax_idx].imshow(img)
            axes[ax_idx].set_title("Reference")
            axes[ax_idx].axis("off")
            ax_idx += 1

    shifted_log_power = np.log1p(np.fft.fftshift(power_spectrum))
    axes[ax_idx].imshow(shifted_log_power)
    axes[ax_idx].set_title("Log power spectrum")
    axes[ax_idx].axis("off")
    ax_idx += 1

    axes[ax_idx].plot(radial_freq, np.log1p(radial_power), marker="o")
    axes[ax_idx].set_title("Radial spectrum")
    axes[ax_idx].set_xlabel("Normalized frequency radius")
    axes[ax_idx].set_ylabel("log(1 + power)")
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1

    band_names = ["Low", "Mid", "High"]
    band_values = [metrics["low_ratio"], metrics["mid_ratio"], metrics["high_ratio"]]
    axes[ax_idx].bar(band_names, band_values)
    axes[ax_idx].set_ylim(0.0, 1.0)
    axes[ax_idx].set_title("Band energy ratios")
    axes[ax_idx].set_ylabel("Ratio")

    text = (
        f"Entropy: {metrics['spectral_entropy']:.4f}\n"
        f"Centroid: {metrics['spectral_centroid']:.4f}\n"
        f"Bandwidth: {metrics['spectral_bandwidth']:.4f}"
    )
    axes[ax_idx].text(0.02, 0.98, text, transform=axes[ax_idx].transAxes, va="top", ha="left")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_frequency_summary(
    metrics: dict[str, Any],
    radial_freq: np.ndarray,
    radial_power: np.ndarray,
    save_path: str,
) -> None:
    """Save frequency metric summary as JSON."""
    summary = {
        "metrics": metrics,
        "radial_spectrum": {
            "frequency_radius": np.asarray(radial_freq, dtype=float).tolist(),
            "power": np.asarray(radial_power, dtype=float).tolist(),
            "log1p_power": np.log1p(np.asarray(radial_power, dtype=float)).tolist(),
        },
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

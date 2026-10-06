"""Temporal analysis execution and dispatch."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from omegaconf import DictConfig

from analyse import (
    autocorrelation,
    lag_distance_curve,
    patch_temporal_smoothness,
    temporal_cosine_shift,
    temporal_effective_rank,
    temporal_smoothness,
    temporal_spectral_entropy,
    temporal_token_norm_entropy,
    temporal_variance,
    total_trajectory_variation,
)
from tools.summary import capture_summary


@dataclass(frozen=True)
class TemporalFeatures:
    pooled_features: np.ndarray
    patch_features: np.ndarray
    grid_h: int
    grid_w: int
    T: int


TemporalHandler = Callable[
    [TemporalFeatures, DictConfig, Path],
    dict[str, Any],
]


def _run_temporal_smoothness(
    features: TemporalFeatures,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    ts, distances = temporal_smoothness.compute_temporal_smoothness(
        features.pooled_features,
    )
    temporal_smoothness.make_temporal_smoothness_figure(
        distances,
        ts,
        str(output_dir / "temporal_smoothness.png"),
    )
    return capture_summary(
        temporal_smoothness.save_temporal_smoothness_summary,
        output_dir,
        ts,
        distances,
        features.T,
    )


def _run_temporal_cosine_shift(
    features: TemporalFeatures,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    tcs, shifts, cosine = temporal_cosine_shift.compute_temporal_cosine_shift(
        features.pooled_features,
    )
    temporal_cosine_shift.make_temporal_cosine_shift_figure(
        shifts,
        tcs,
        str(output_dir / "temporal_cosine_shift.png"),
    )
    return capture_summary(
        temporal_cosine_shift.save_temporal_cosine_shift_summary,
        output_dir,
        tcs,
        shifts,
        cosine,
        features.T,
    )


def _run_lag_distance_curve(
    features: TemporalFeatures,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    lags, distances, per_lag_distances = (
        lag_distance_curve.compute_lag_distance_curve(features.pooled_features)
    )
    lag_distance_curve.make_lag_distance_figure(
        lags,
        distances,
        str(output_dir / "lag_distance_curve.png"),
    )
    return capture_summary(
        lag_distance_curve.save_lag_distance_summary,
        output_dir,
        lags,
        distances,
        per_lag_distances,
        features.T,
    )


def _run_temporal_variance(
    features: TemporalFeatures,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    channel_vars, mean_var, max_var = temporal_variance.compute_temporal_variance(
        features.pooled_features,
    )
    temporal_variance.make_temporal_variance_figure(
        channel_vars,
        mean_var,
        max_var,
        str(output_dir / "temporal_variance.png"),
    )
    return capture_summary(
        temporal_variance.save_temporal_variance_summary,
        output_dir,
        channel_vars,
        mean_var,
        max_var,
        features.T,
    )


def _run_temporal_effective_rank(
    features: TemporalFeatures,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    eff_rank, eigenvalues = temporal_effective_rank.compute_temporal_effective_rank(
        features.pooled_features,
    )
    temporal_effective_rank.make_temporal_rank_figure(
        eigenvalues,
        eff_rank,
        str(output_dir / "temporal_effective_rank.png"),
    )
    return capture_summary(
        temporal_effective_rank.save_temporal_rank_summary,
        output_dir,
        eff_rank,
        eigenvalues,
        features.T,
    )


def _run_temporal_spectral_entropy(
    features: TemporalFeatures,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    entropy, normalized_entropy, eigenvalues, probs = (
        temporal_spectral_entropy.compute_temporal_spectral_entropy(
            features.pooled_features,
        )
    )
    temporal_spectral_entropy.make_temporal_spectral_entropy_figure(
        eigenvalues,
        probs,
        entropy,
        normalized_entropy,
        str(output_dir / "temporal_spectral_entropy.png"),
    )
    return capture_summary(
        temporal_spectral_entropy.save_temporal_spectral_entropy_summary,
        output_dir,
        entropy,
        normalized_entropy,
        eigenvalues,
        probs,
        features.T,
    )


def _run_autocorrelation(
    features: TemporalFeatures,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    max_lag = min(features.T // 2, 50)
    lags, autocorr = autocorrelation.compute_autocorrelation(
        features.pooled_features,
        max_lag=max_lag,
    )
    autocorrelation.make_autocorrelation_figure(
        lags,
        autocorr,
        str(output_dir / "autocorrelation.png"),
    )
    return capture_summary(
        autocorrelation.save_autocorrelation_summary,
        output_dir,
        lags,
        autocorr,
        features.T,
    )


def _run_total_trajectory_variation(
    features: TemporalFeatures,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    ttv, normalized_ttv, step_distances = (
        total_trajectory_variation.compute_total_trajectory_variation(
            features.pooled_features,
        )
    )
    total_trajectory_variation.make_ttv_figure(
        step_distances,
        ttv,
        normalized_ttv,
        str(output_dir / "total_trajectory_variation.png"),
    )
    return capture_summary(
        total_trajectory_variation.save_ttv_summary,
        output_dir,
        ttv,
        normalized_ttv,
        step_distances,
        features.T,
    )


def _run_patch_temporal_smoothness(
    features: TemporalFeatures,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    patch_smoothness, step_distances = (
        patch_temporal_smoothness.compute_patch_temporal_smoothness(
            features.patch_features,
        )
    )
    patch_temporal_smoothness.make_patch_temporal_smoothness_figure(
        patch_smoothness,
        features.grid_h,
        features.grid_w,
        str(output_dir / "patch_temporal_smoothness.png"),
    )
    return capture_summary(
        patch_temporal_smoothness.save_patch_temporal_smoothness_summary,
        output_dir,
        patch_smoothness,
        step_distances,
        features.grid_h,
        features.grid_w,
        features.T,
    )


def _run_temporal_token_norm_entropy(
    features: TemporalFeatures,
    cfg: DictConfig,
    output_dir: Path,
) -> dict[str, Any]:
    num_bins = cfg.get("num_bins", 16)
    entropy, normalized_entropy = (
        temporal_token_norm_entropy.compute_temporal_token_norm_entropy(
            features.patch_features,
            num_bins,
        )
    )
    temporal_token_norm_entropy.make_temporal_token_norm_entropy_figure(
        entropy,
        normalized_entropy,
        features.grid_h,
        features.grid_w,
        str(output_dir / "temporal_token_norm_entropy.png"),
    )
    return capture_summary(
        temporal_token_norm_entropy.save_temporal_token_norm_entropy_summary,
        output_dir,
        entropy,
        normalized_entropy,
        features.grid_h,
        features.grid_w,
        features.T,
        num_bins,
    )


TEMPORAL_HANDLERS: dict[str, TemporalHandler] = {
    "temporal_smoothness": _run_temporal_smoothness,
    "temporal_cosine_shift": _run_temporal_cosine_shift,
    "lag_distance_curve": _run_lag_distance_curve,
    "temporal_variance": _run_temporal_variance,
    "temporal_effective_rank": _run_temporal_effective_rank,
    "temporal_spectral_entropy": _run_temporal_spectral_entropy,
    "autocorrelation": _run_autocorrelation,
    "total_trajectory_variation": _run_total_trajectory_variation,
    "patch_temporal_smoothness": _run_patch_temporal_smoothness,
    "temporal_token_norm_entropy": _run_temporal_token_norm_entropy,
}


def process_temporal_analysis(
    analysis_names: list[str],
    frames: list[np.ndarray],
    model,
    cfg: DictConfig,
    output_dir: Path,
):
    """Process temporal analysis methods that require multiple frames."""
    batch_size = cfg.get("batch_size", 8)
    grid_h, grid_w = None, None
    analysis_summaries = {}

    all_pooled_features = []
    all_patch_features = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        if not batch_frames:
            continue

        visual_tokens = model(batch_frames)
        pooled_np = visual_tokens.mean(dim=(2, 3)).detach().float().cpu().numpy()
        visual_tokens_np = visual_tokens.detach().float().cpu().numpy()
        B, C, H, W = visual_tokens_np.shape

        if grid_h is None:
            grid_h, grid_w = H, W

        all_pooled_features.append(pooled_np)
        patch_feats = visual_tokens_np.reshape(B, C, -1).transpose(0, 2, 1)
        all_patch_features.append(patch_feats)

    pooled_features = np.concatenate(all_pooled_features, axis=0)
    patch_features = np.concatenate(all_patch_features, axis=0)
    T = pooled_features.shape[0]

    print(f"Processing temporal analysis with {T} frames")
    features = TemporalFeatures(
        pooled_features=pooled_features,
        patch_features=patch_features,
        grid_h=grid_h,
        grid_w=grid_w,
        T=T,
    )

    for analysis_name in analysis_names:
        handler = TEMPORAL_HANDLERS.get(analysis_name)
        if handler is None:
            raise ValueError(f"Unknown temporal analysis method: {analysis_name}")
        analysis_summaries[analysis_name] = handler(features, cfg, output_dir)

    print(f"Temporal analysis results saved to: {output_dir}")
    return analysis_summaries

"""Encoder test framework for analyzing visual encoder representations.

Usage:
    # Single-frame analysis (averaged over all frames)
    python run.py model=dinov3 analysis=pca_vis video_path=playground/videos/example.mp4
    python run.py model=clip analysis=avg_token_cos video_path=playground/videos/example.mp4
    
    # Temporal analysis
    python run.py model=dinov3 analysis=temporal_smoothness video_path=playground/videos/example.mp4
    python run.py model=dinov3 analysis=autocorrelation video_path=playground/videos/example.mp4 frame_start=0 frame_end=100
    
    # Specify stride
    python run.py model=sam analysis=temporal_variance video_path=playground/videos/example.mp4 stride=5
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}

# Register custom resolvers for formatting the hydra output directory
OmegaConf.register_new_resolver(
    "format_model",
    lambda m: m.split("/")[-1]
)
OmegaConf.register_new_resolver(
    "format_dir",
    lambda a, add: (a + "_" + add) if add else a
)
OmegaConf.register_new_resolver(
    "format_path",
    lambda m, a, add: (m.split("/")[-1] + "_" + a + "_" + add) if add else (m.split("/")[-1] + "_" + a)
)

# Checkpoint loading for fine-tuned encoders
POLICY_VISION_PREFIX = "vision_encoder."


def _strip_vision_prefixes(key: str) -> str:
    """Drop wrapper prefixes (policy prefix, DDP/compile wrappers) from a checkpoint key."""
    for prefix in (POLICY_VISION_PREFIX, "module.", "_orig_mod."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _map_ckpt_key_to_target(key: str, target_keys) -> str | None:
    """Map a checkpoint key onto a target parameter via unique dotted-suffix match.

    The attribute holding the encoder differs between the policy wrapper used for
    training and the analysis wrapper, e.g. the same weight appears as
    ``vision_encoder.aggregator.*`` (VGGT), ``vision_encoder.model.model.*``
    (C-RADIO), ``vision_encoder.model.*`` -> ``vision_model.*`` (SigLIP2) or
    ``vision_encoder.model.*`` -> ``embeddings.*`` (DINOv3). Hard-coding one
    prefix therefore silently matches nothing for the other backbones, so match
    the longest unique dotted suffix instead.
    """
    if key in target_keys:
        return key
    parts = key.split(".")
    for size in range(len(parts) - 1, 0, -1):
        tail = ".".join(parts[-size:])
        candidates = [
            candidate
            for candidate in target_keys
            if candidate == tail or candidate.endswith("." + tail)
        ]
        if len(candidates) == 1:
            return candidates[0]
    return None


def load_vision_checkpoint(model, ckpt_dir: str, device) -> None:
    """Restore ``vision_encoder.*`` weights from a policy checkpoint into ``model.model``.

    Fails loudly instead of silently doing nothing: it reports how many tensors
    matched, complains about unmatched keys, and re-reads a sample of the
    restored parameters to prove the weights actually landed in the module.
    """
    payload = torch.load(ckpt_dir, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a state dict in {ckpt_dir}, got {type(payload).__name__}"
        )

    vision_items = {k: v for k, v in payload.items() if k.startswith(POLICY_VISION_PREFIX)}
    if not vision_items:
        # Tolerate encoder-only state dicts.
        vision_items = {k: v for k, v in payload.items() if torch.is_tensor(v)}
    if not vision_items:
        raise RuntimeError(f"No tensors found in checkpoint: {ckpt_dir}")

    target_module = model.model if hasattr(model, "model") else model
    target = target_module.state_dict()

    restored: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []
    shape_mismatch: list[tuple[str, tuple, tuple]] = []
    for key, value in vision_items.items():
        mapped = _map_ckpt_key_to_target(_strip_vision_prefixes(key), target)
        if mapped is None:
            unmatched.append(key)
        elif tuple(value.shape) != tuple(target[mapped].shape):
            shape_mismatch.append((key, tuple(value.shape), tuple(target[mapped].shape)))
        else:
            restored[mapped] = value

    missing, unexpected = target_module.load_state_dict(restored, strict=False)

    print(f"Loaded checkpoint from: {ckpt_dir}")
    print(
        f"  vision_encoder tensors: {len(vision_items)}, restored: {len(restored)}, "
        f"unmatched: {len(unmatched)}, shape mismatch: {len(shape_mismatch)}"
    )
    print(
        f"  target params not covered by the checkpoint: {len(missing)} "
        "(frozen/untrained modules such as prediction heads are expected here)"
    )
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]} (total {len(unexpected)})")
    if unmatched:
        print(f"  Unmatched checkpoint keys (first 5): {unmatched[:5]}")
    if shape_mismatch:
        print(f"  Shape mismatches (first 5): {shape_mismatch[:5]}")

    if not restored:
        raise RuntimeError(
            f"No checkpoint tensors could be mapped onto {type(target_module).__name__}; "
            f"check that the encoder architecture in {ckpt_dir} matches the configured model."
        )

    # Guard against a silent no-op: verify a sample of restored tensors on device.
    applied = target_module.state_dict()
    not_applied = [
        key
        for key in list(restored)[:5]
        if not torch.equal(applied[key].detach().cpu(), restored[key].detach().cpu())
    ]
    if not_applied:
        raise RuntimeError(
            f"Checkpoint restore verification failed for: {not_applied[:3]}"
        )
    print(f"  verified {len(list(restored)[:5])} restored tensors match the checkpoint")


# Import all analysis functions
from analyse import (
    autocorrelation,
    avg_token_cos,
    dist_sim_decay,
    lag_distance_curve,
    mean_token_norm,
    neighbor_sim,
    patch_temporal_smoothness,
    pca_vis,
    temporal_cosine_shift,
    temporal_effective_rank,
    temporal_smoothness,
    temporal_spectral_entropy,
    temporal_token_norm_entropy,
    temporal_variance,
    token_cov_rank,
    token_norm_entropy,
    token_norm_var,
    token_to_global,
    total_trajectory_variation,
    trajectory_var_ratio,
    within_between_var,
    frequency_metrics, 
)


def extract_frames(video_path: str, frame_start: int = None, frame_end: int = None, stride: int = 1):
    """Extract frames from video with optional range and stride.
    
    Returns:
        list[np.ndarray]: List of RGB frames
        list[int]: Frame indices
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_start is None:
        frame_start = 0
    if frame_end is None:
        frame_end = total_frames
    
    frame_indices = list(range(frame_start, frame_end, stride))
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames, frame_indices


def find_video_files(
    video_dir: Path, 
    sample_interval: int = 1,
    start_index: int = 0
) -> list[Path]:
    """
    Return all supported video files directly inside a directory, sampled every N files.
    
    Args:
        video_dir: Directory containing video files
        sample_interval: Take one video every N files
        start_index: Starting index for sampling (0-based)
    
    Returns:
        List of sampled video paths
    """
    all_videos = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    
    
    return all_videos[start_index::sample_interval]

def safe_output_name(path: Path, used_names: set[str]) -> str:
    """Create a stable, filesystem-safe output folder name for a video."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    if not stem:
        stem = "video"

    candidate = stem
    suffix = 2
    while candidate in used_names:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def select_representative_frame_position(frame_indices: list[int], representative_frame_index: int | None) -> int:
    """Select the extracted-frame position closest to the requested source frame."""
    if not frame_indices:
        raise ValueError("Cannot select representative frame from an empty frame list")
    if representative_frame_index is None:
        return 0
    return min(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - representative_frame_index))


def normalize_analysis_names(analysis: Any) -> list[str]:
    """Normalize Hydra/string analysis config into a de-duplicated list."""
    if isinstance(analysis, (list, tuple, ListConfig)):
        raw_names = list(analysis)
    elif isinstance(analysis, str):
        stripped = analysis.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
            raw_names = stripped.split(",")
        elif "," in stripped:
            raw_names = stripped.split(",")
        else:
            raw_names = [stripped]
    else:
        raw_names = [analysis]

    analysis_names = []
    seen = set()
    for name in raw_names:
        name = str(name).strip().strip("'\"")
        if not name or name in seen:
            continue
        analysis_names.append(name)
        seen.add(name)
    return analysis_names


def capture_summary(save_fn, temp_dir: Path, *args) -> dict[str, Any]:
    """Run an existing save_* function and return the JSON it would write."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=".summary_",
        dir=temp_dir,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_fn(*args, str(tmp_path))
        with open(tmp_path, "r") as f:
            return json.load(f)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_grouped_summary(
    summary_path: Path,
    analysis_summaries: dict[str, dict[str, Any]],
    selected_analyses: list[str],
    metadata: dict[str, Any] | None = None,
):
    """Write a summary where each metric owns its original summary fields."""
    payload = {
        "selected_analyses": selected_analyses,
        "analyses": analysis_summaries,
    }
    if metadata:
        payload["metadata"] = metadata

    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)


def append_metric(accumulator: dict[str, dict[str, list[Any]]], analysis_name: str, key: str, value: Any):
    accumulator.setdefault(analysis_name, {}).setdefault(key, []).append(value)


def summarize_single_frame_metrics(metrics_accumulator: dict[str, dict[str, list[Any]]]) -> dict[str, dict[str, Any]]:
    averaged = {}
    for analysis_name, analysis_metrics in metrics_accumulator.items():
        summary = {}
        for key, values in analysis_metrics.items():
            if key == "decay_curves":
                all_centers, all_means, all_counts = zip(*values)
                avg_means = np.mean([m for m in all_means], axis=0)
                avg_counts = np.mean([c for c in all_counts], axis=0)
                summary["mean_decay_curve"] = {
                    "distances": all_centers[0].tolist(),
                    "mean_similarities": avg_means.tolist(),
                    "pair_counts": avg_counts.astype(int).tolist()
                }
            elif key == "frequency_metrics":
                metric_names = [
                    "low_ratio", "mid_ratio", "high_ratio",
                    "spectral_entropy", "spectral_centroid", "spectral_bandwidth",
                ]

                for metric_name in metric_names:
                    arr = np.array([m[metric_name] for m in values], dtype=float)
                    summary[f"mean_{metric_name}"] = float(np.mean(arr))
                    summary[f"std_{metric_name}"] = float(np.std(arr))

                for meta_name in [
                    "low_thr", "mid_thr", "remove_dc", "normalize_channel",
                    "grid_h", "grid_w", "num_channels",
                    "low_bin_count", "mid_bin_count", "high_bin_count",
                ]:
                    if meta_name in values[0]:
                        summary[meta_name] = values[0][meta_name]
            elif values and np.isscalar(values[0]):
                summary[f"mean_{key}"] = float(np.mean(values))
                summary[f"std_{key}"] = float(np.std(values))

        if summary:
            averaged[analysis_name] = summary
    return averaged


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def average_json_values(values: list[Any]) -> Any:
    """Average compatible numeric JSON values while preserving nested structure."""
    values = [value for value in values if value is not None]
    if not values:
        return None

    if all(is_number(value) for value in values):
        return float(np.mean(values))

    if all(isinstance(value, dict) for value in values):
        averaged = {}
        common_keys = set(values[0].keys())
        for value in values[1:]:
            common_keys &= set(value.keys())

        for key in sorted(common_keys):
            averaged_value = average_json_values([value[key] for value in values])
            if averaged_value is not None:
                averaged[key] = averaged_value
        return averaged or None

    if all(isinstance(value, list) for value in values):
        lengths = [len(value) for value in values]
        if len(set(lengths)) != 1:
            return values[0] if all(value == values[0] for value in values) else None
        if lengths[0] == 0:
            return []

        averaged_list = []
        for items in zip(*values):
            averaged_value = average_json_values(list(items))
            if averaged_value is None:
                return values[0] if all(value == values[0] for value in values) else None
            averaged_list.append(averaged_value)
        return averaged_list

    return values[0] if all(value == values[0] for value in values) else None


def average_video_analysis_summaries(video_summaries: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Average per-video grouped analysis summaries."""
    averaged = {}
    analysis_names = []
    for summary in video_summaries:
        for analysis_name in summary:
            if analysis_name not in analysis_names:
                analysis_names.append(analysis_name)

    for analysis_name in analysis_names:
        values = [
            summary[analysis_name]
            for summary in video_summaries
            if analysis_name in summary
        ]
        averaged_value = average_json_values(values)
        if averaged_value is not None:
            averaged[analysis_name] = averaged_value
    return averaged


def process_single_frame_analysis(
    analysis_names: list[str],
    frames: list[np.ndarray],
    model,
    cfg: DictConfig,
    output_dir: Path,
    frame_indices: list[int] | None = None,
    save_frame_outputs: bool = True,
    save_frame_summaries: bool = True,
    representative_frame_position: int | None = None,
):
    """Process single-frame analysis methods.
    
    For these methods, we compute the metric for each frame and output the average.
    When save_frame_outputs is False, only the representative frame writes PNGs.
    """
    batch_size = cfg.get("batch_size", 8)
    grid_h, grid_w = None, None
    
    # Accumulators for averaging, grouped by analysis name.
    metrics_accumulator = {}
    
    for i in tqdm(range(0, len(frames), batch_size), desc="Processing frames", total=len(frames)//batch_size):
        batch_frames = frames[i:i + batch_size]
        if not batch_frames:
            continue
        
        # Extract visual patch tokens as BCHW.
        visual_tokens = model(batch_frames)
        
        # Convert to numpy
        visual_tokens_np = visual_tokens.cpu().numpy()  # (B, C, H, W)
        B, C, H, W = visual_tokens_np.shape
        
        # Set grid dimensions from first batch
        if grid_h is None:
            grid_h, grid_w = H, W
        
        # Process each frame in batch
        for b in range(B):
            frame_idx = i + b
            if frame_idx >= len(frames):
                break
            
            # Reshape visual tokens to (N, D)
            tokens = visual_tokens_np[b].reshape(C, -1).T  # (H*W, C)

            source_frame_idx = frame_indices[frame_idx] if frame_indices is not None else frame_idx
            should_save_frame_outputs = save_frame_outputs or frame_idx == representative_frame_position
            frame_output_dir = None
            ref_img_path = None
            if should_save_frame_outputs:
                frame_output_dir = output_dir / f"frame_{source_frame_idx:04d}"
                frame_output_dir.mkdir(parents=True, exist_ok=True)
                ref_img_path = frame_output_dir / "reference.png"
                cv2.imwrite(str(ref_img_path), cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR))
            
            frame_summaries = {}

            # Run all requested analyses for this frame and group their summaries.
            for analysis_name in analysis_names:
                if analysis_name == "pca_vis":
                    if should_save_frame_outputs:
                        pca_rgb = pca_vis.pca_to_rgb(tokens, grid_h, grid_w)
                        pca_vis.make_side_by_side(str(ref_img_path), pca_rgb, str(frame_output_dir / "pca_vis.png"))

                elif analysis_name == "avg_token_cos":
                    avg_cos, cos_matrix = avg_token_cos.compute_avg_token_cos(tokens)
                    append_metric(metrics_accumulator, analysis_name, "avg_cos_values", avg_cos)

                    if should_save_frame_outputs:
                        avg_token_cos.make_cos_figure(str(ref_img_path), cos_matrix, avg_cos, str(frame_output_dir / "avg_token_cos.png"))
                    if save_frame_summaries:
                        frame_summaries[analysis_name] = capture_summary(
                            avg_token_cos.save_cos_summary,
                            frame_output_dir,
                            avg_cos,
                            cos_matrix,
                        )

                elif analysis_name == "dist_sim_decay":
                    bucket_centers, bucket_means, bucket_counts = dist_sim_decay.compute_dist_sim_decay(tokens, grid_h, grid_w)
                    append_metric(metrics_accumulator, analysis_name, "decay_curves", (bucket_centers, bucket_means, bucket_counts))

                    if should_save_frame_outputs:
                        dist_sim_decay.make_decay_figure(
                            str(ref_img_path),
                            bucket_centers,
                            bucket_means,
                            bucket_counts,
                            str(frame_output_dir / "dist_sim_decay.png"),
                        )
                    if save_frame_summaries:
                        frame_summaries[analysis_name] = capture_summary(
                            dist_sim_decay.save_decay_summary,
                            frame_output_dir,
                            bucket_centers,
                            bucket_means,
                            bucket_counts,
                        )

                elif analysis_name == "mean_token_norm":
                    norms, mean_norm, std_norm = mean_token_norm.compute_mean_token_norm(tokens)
                    append_metric(metrics_accumulator, analysis_name, "mean_norms", mean_norm)
                    append_metric(metrics_accumulator, analysis_name, "std_norms", std_norm)

                    if should_save_frame_outputs:
                        mean_token_norm.make_norm_heatmap(str(ref_img_path), norms, grid_h, grid_w, str(frame_output_dir / "mean_token_norm.png"))
                    if save_frame_summaries:
                        frame_summaries[analysis_name] = capture_summary(
                            mean_token_norm.save_norm_summary,
                            frame_output_dir,
                            norms,
                            mean_norm,
                            std_norm,
                        )

                elif analysis_name == "neighbor_sim":
                    avg_ns, ns_map = neighbor_sim.compute_neighbor_sim(tokens, grid_h, grid_w)
                    append_metric(metrics_accumulator, analysis_name, "neighbor_sim_values", avg_ns)

                    if should_save_frame_outputs:
                        neighbor_sim.make_ns_figure(str(ref_img_path), ns_map, avg_ns, str(frame_output_dir / "neighbor_sim.png"))
                    if save_frame_summaries:
                        frame_summaries[analysis_name] = capture_summary(
                            neighbor_sim.save_ns_summary,
                            frame_output_dir,
                            avg_ns,
                            ns_map,
                        )

                elif analysis_name == "token_cov_rank":
                    eff_rank, eigenvalues = token_cov_rank.compute_token_cov_rank(tokens)
                    append_metric(metrics_accumulator, analysis_name, "eff_ranks", eff_rank)

                    if should_save_frame_outputs:
                        token_cov_rank.make_rank_figure(str(ref_img_path), eigenvalues, eff_rank, str(frame_output_dir / "token_cov_rank.png"))
                    if save_frame_summaries:
                        frame_summaries[analysis_name] = capture_summary(
                            token_cov_rank.save_rank_summary,
                            frame_output_dir,
                            eff_rank,
                            eigenvalues,
                        )

                elif analysis_name == "token_norm_entropy":
                    norms, entropy, hist, bin_edges = token_norm_entropy.compute_token_norm_entropy(tokens)
                    append_metric(metrics_accumulator, analysis_name, "entropies", entropy)

                    if should_save_frame_outputs:
                        token_norm_entropy.make_entropy_figure(
                            str(ref_img_path),
                            norms,
                            entropy,
                            hist,
                            bin_edges,
                            str(frame_output_dir / "token_norm_entropy.png"),
                        )
                    if save_frame_summaries:
                        frame_summaries[analysis_name] = capture_summary(
                            token_norm_entropy.save_entropy_summary,
                            frame_output_dir,
                            norms,
                            entropy,
                        )

                elif analysis_name == "token_norm_var":
                    norms, var_norm = token_norm_var.compute_token_norm_var(tokens)
                    append_metric(metrics_accumulator, analysis_name, "var_norms", var_norm)

                    if should_save_frame_outputs:
                        token_norm_var.make_var_heatmap(
                            str(ref_img_path),
                            norms,
                            grid_h,
                            grid_w,
                            var_norm,
                            str(frame_output_dir / "token_norm_var.png"),
                        )
                    if save_frame_summaries:
                        frame_summaries[analysis_name] = capture_summary(
                            token_norm_var.save_var_summary,
                            frame_output_dir,
                            norms,
                            var_norm,
                        )

                elif analysis_name == "token_to_global":
                    sims, mean_sim, var_sim = token_to_global.compute_token_to_global(tokens)
                    append_metric(metrics_accumulator, analysis_name, "mean_sims", mean_sim)
                    append_metric(metrics_accumulator, analysis_name, "var_sims", var_sim)

                    if should_save_frame_outputs:
                        token_to_global.make_token_to_global_figure(
                            str(ref_img_path),
                            sims,
                            grid_h,
                            grid_w,
                            mean_sim,
                            var_sim,
                            str(frame_output_dir / "token_to_global.png"),
                        )
                    if save_frame_summaries:
                        frame_summaries[analysis_name] = capture_summary(
                            token_to_global.save_token_to_global_summary,
                            frame_output_dir,
                            sims,
                            mean_sim,
                            var_sim,
                        )

                elif analysis_name == "frequency_metrics":
                    low_thr = cfg.get("low_thr", 0.15)
                    mid_thr = cfg.get("mid_thr", 0.35)
                    remove_dc = cfg.get("remove_dc", True)
                    normalize_channel = cfg.get("normalize_channel", False)
                    num_radial_bins = cfg.get("num_radial_bins", None)

                    metrics, power_spectrum, radial_freq, radial_power = frequency_metrics.compute_frequency_metrics(
                        tokens,
                        grid_h,
                        grid_w,
                        low_thr=low_thr,
                        mid_thr=mid_thr,
                        remove_dc=remove_dc,
                        normalize_channel=normalize_channel,
                        num_radial_bins=num_radial_bins,
                    )

                    append_metric(metrics_accumulator, analysis_name, "frequency_metrics", metrics)

                    if should_save_frame_outputs:
                        frequency_metrics.make_frequency_figure(
                            str(ref_img_path),
                            power_spectrum,
                            radial_freq,
                            radial_power,
                            metrics,
                            str(frame_output_dir / "frequency_metrics.png"),
                        )

                    if save_frame_summaries:
                        frame_summaries[analysis_name] = capture_summary(
                            frequency_metrics.save_frequency_summary,
                            frame_output_dir,
                            metrics,
                            radial_freq,
                            radial_power,
                        )

            if save_frame_summaries and frame_summaries:
                write_grouped_summary(
                    frame_output_dir / "summary.json",
                    frame_summaries,
                    analysis_names,
                    metadata={"frame_index": source_frame_idx},
                )
    
    # Compute and save averaged metrics
    if metrics_accumulator:
        summary = summarize_single_frame_metrics(metrics_accumulator)
        summary_path = output_dir / "averaged_summary.json"
        write_grouped_summary(
            summary_path,
            summary,
            analysis_names,
            metadata={"frame_count": len(frames)},
        )
        print(f"Averaged summary saved to: {summary_path}")
        return summary

    return {}


def process_temporal_analysis(analysis_names: list[str], frames: list[np.ndarray], model, cfg: DictConfig, output_dir: Path):
    """Process temporal analysis methods that require multiple frames."""
    batch_size = cfg.get("batch_size", 8)
    grid_h, grid_w = None, None
    analysis_summaries = {}
    
    # Collect features from all frames
    all_pooled_features = []
    all_patch_features = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        if not batch_frames:
            continue
        
        # Extract visual patch tokens as BCHW.
        visual_tokens = model(batch_frames)
        
        # Convert to numpy
        pooled_np = visual_tokens.mean(dim=(2, 3)).cpu().numpy()  # (B, D)
        visual_tokens_np = visual_tokens.cpu().numpy()  # (B, C, H, W)
        B, C, H, W = visual_tokens_np.shape
        
        # Set grid dimensions from first batch
        if grid_h is None:
            grid_h, grid_w = H, W
        
        # Collect spatially averaged patch features.
        all_pooled_features.append(pooled_np)  # (B, D)
        
        # Collect patch features reshaped to (B, N, D)
        patch_feats = visual_tokens_np.reshape(B, C, -1).transpose(0, 2, 1)  # (B, H*W, C)
        all_patch_features.append(patch_feats)
    
    # Concatenate all batches
    pooled_features = np.concatenate(all_pooled_features, axis=0)  # (T, D)
    patch_features = np.concatenate(all_patch_features, axis=0)  # (T, N, D)
    T = pooled_features.shape[0]
    
    print(f"Processing temporal analysis with {T} frames")
    
    # Run temporal analyses with shared extracted features.
    for analysis_name in analysis_names:
        if analysis_name == "temporal_smoothness":
            ts, distances = temporal_smoothness.compute_temporal_smoothness(pooled_features)
            temporal_smoothness.make_temporal_smoothness_figure(distances, ts, str(output_dir / "temporal_smoothness.png"))
            analysis_summaries[analysis_name] = capture_summary(
                temporal_smoothness.save_temporal_smoothness_summary,
                output_dir,
                ts,
                distances,
                T,
            )

        elif analysis_name == "temporal_cosine_shift":
            tcs, shifts, cosine = temporal_cosine_shift.compute_temporal_cosine_shift(pooled_features)
            temporal_cosine_shift.make_temporal_cosine_shift_figure(shifts, tcs, str(output_dir / "temporal_cosine_shift.png"))
            analysis_summaries[analysis_name] = capture_summary(
                temporal_cosine_shift.save_temporal_cosine_shift_summary,
                output_dir,
                tcs,
                shifts,
                cosine,
                T,
            )

        elif analysis_name == "lag_distance_curve":
            lags, distances, per_lag_distances = lag_distance_curve.compute_lag_distance_curve(pooled_features)
            lag_distance_curve.make_lag_distance_figure(lags, distances, str(output_dir / "lag_distance_curve.png"))
            analysis_summaries[analysis_name] = capture_summary(
                lag_distance_curve.save_lag_distance_summary,
                output_dir,
                lags,
                distances,
                per_lag_distances,
                T,
            )

        elif analysis_name == "temporal_variance":
            channel_vars, mean_var, max_var = temporal_variance.compute_temporal_variance(pooled_features)
            temporal_variance.make_temporal_variance_figure(channel_vars, mean_var, max_var, str(output_dir / "temporal_variance.png"))
            analysis_summaries[analysis_name] = capture_summary(
                temporal_variance.save_temporal_variance_summary,
                output_dir,
                channel_vars,
                mean_var,
                max_var,
                T,
            )

        elif analysis_name == "temporal_effective_rank":
            eff_rank, eigenvalues = temporal_effective_rank.compute_temporal_effective_rank(pooled_features)
            temporal_effective_rank.make_temporal_rank_figure(eigenvalues, eff_rank, str(output_dir / "temporal_effective_rank.png"))
            analysis_summaries[analysis_name] = capture_summary(
                temporal_effective_rank.save_temporal_rank_summary,
                output_dir,
                eff_rank,
                eigenvalues,
                T,
            )

        elif analysis_name == "temporal_spectral_entropy":
            entropy, normalized_entropy, eigenvalues, probs = temporal_spectral_entropy.compute_temporal_spectral_entropy(pooled_features)
            temporal_spectral_entropy.make_temporal_spectral_entropy_figure(
                eigenvalues,
                probs,
                entropy,
                normalized_entropy,
                str(output_dir / "temporal_spectral_entropy.png"),
            )
            analysis_summaries[analysis_name] = capture_summary(
                temporal_spectral_entropy.save_temporal_spectral_entropy_summary,
                output_dir,
                entropy,
                normalized_entropy,
                eigenvalues,
                probs,
                T,
            )

        elif analysis_name == "autocorrelation":
            max_lag = min(T // 2, 50)  # Reasonable default
            lags, autocorr = autocorrelation.compute_autocorrelation(pooled_features, max_lag=max_lag)
            autocorrelation.make_autocorrelation_figure(lags, autocorr, str(output_dir / "autocorrelation.png"))
            analysis_summaries[analysis_name] = capture_summary(
                autocorrelation.save_autocorrelation_summary,
                output_dir,
                lags,
                autocorr,
                T,
            )

        elif analysis_name == "total_trajectory_variation":
            ttv, normalized_ttv, step_distances = total_trajectory_variation.compute_total_trajectory_variation(pooled_features)
            total_trajectory_variation.make_ttv_figure(step_distances, ttv, normalized_ttv, str(output_dir / "total_trajectory_variation.png"))
            analysis_summaries[analysis_name] = capture_summary(
                total_trajectory_variation.save_ttv_summary,
                output_dir,
                ttv,
                normalized_ttv,
                step_distances,
                T,
            )

        elif analysis_name == "patch_temporal_smoothness":
            patch_smoothness, step_distances = patch_temporal_smoothness.compute_patch_temporal_smoothness(patch_features)
            patch_temporal_smoothness.make_patch_temporal_smoothness_figure(
                patch_smoothness,
                grid_h,
                grid_w,
                str(output_dir / "patch_temporal_smoothness.png"),
            )
            analysis_summaries[analysis_name] = capture_summary(
                patch_temporal_smoothness.save_patch_temporal_smoothness_summary,
                output_dir,
                patch_smoothness,
                step_distances,
                grid_h,
                grid_w,
                T,
            )

        elif analysis_name == "temporal_token_norm_entropy":
            num_bins = cfg.get("num_bins", 16)
            entropy, normalized_entropy = temporal_token_norm_entropy.compute_temporal_token_norm_entropy(patch_features, num_bins)
            temporal_token_norm_entropy.make_temporal_token_norm_entropy_figure(
                entropy,
                normalized_entropy,
                grid_h,
                grid_w,
                str(output_dir / "temporal_token_norm_entropy.png"),
            )
            analysis_summaries[analysis_name] = capture_summary(
                temporal_token_norm_entropy.save_temporal_token_norm_entropy_summary,
                output_dir,
                entropy,
                normalized_entropy,
                grid_h,
                grid_w,
                T,
                num_bins,
            )

        else:
            raise ValueError(f"Unknown temporal analysis method: {analysis_name}")

    print(f"Temporal analysis results saved to: {output_dir}")
    return analysis_summaries


def process_within_between_var_analysis(
    frames: list[np.ndarray],
    frame_indices: list[int],
    model,
    cfg: DictConfig,
    output_dir: Path,
):
    """Compute within-image and between-image variance over all selected video frames."""
    batch_size = cfg.get("batch_size", 8)
    accumulator = within_between_var.WithinBetweenVarAccumulator()

    for i in tqdm(range(0, len(frames), batch_size), desc="Processing frames"):
        batch_frames = frames[i:i + batch_size]
        if not batch_frames:
            continue

        with torch.no_grad():
            visual_tokens = model(batch_frames)
        visual_tokens_np = visual_tokens.cpu().numpy()  # (B, C, H, W)
        B, C, _, _ = visual_tokens_np.shape

        for b in range(B):
            source_frame_idx = frame_indices[i + b] if i + b < len(frame_indices) else i + b
            tokens = visual_tokens_np[b].reshape(C, -1).T  # (H*W, C)
            accumulator.add_image(tokens, name=f"frame_{source_frame_idx:06d}")

    ratio, within_var, between_var, per_image = accumulator.compute()
    summary = capture_summary(
        within_between_var.save_within_between_summary,
        output_dir,
        ratio,
        within_var,
        between_var,
        per_image,
    )

    print(f"Within-image variance: {within_var:.6f}")
    print(f"Between-image variance: {between_var:.6f}")
    print(f"Variance ratio: {ratio:.6f}")
    print(f"Within/between variance results saved to: {output_dir}")
    return {"within_between_var": summary}


def extract_pooled_features(
    frames: list[np.ndarray],
    model,
    batch_size: int,
) -> np.ndarray:
    """Extract spatially pooled features for each frame. Returns (T, D)."""
    all_pooled_features = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        if not batch_frames:
            continue
        with torch.no_grad():
            visual_tokens = model(batch_frames)
        all_pooled_features.append(visual_tokens.mean(dim=(2, 3)).cpu().numpy())

    if not all_pooled_features:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(all_pooled_features, axis=0)


def finalize_trajectory_var_ratio(
    accumulator: "trajectory_var_ratio.TrajectoryVarRatioAccumulator",
    output_dir: Path,
) -> dict[str, Any]:
    """Finalize trajectory_var_ratio across videos and save method artifacts."""
    summary = accumulator.finalize()
    if summary["num_videos"] < 2:
        raise ValueError(
            "trajectory_var_ratio requires features from at least 2 videos, "
            f"got {summary['num_videos']}"
        )

    json_path = output_dir / "trajectory_var_ratio.json"
    fig_path = output_dir / "trajectory_var_ratio.png"
    trajectory_var_ratio.save_trajectory_var_ratio_summary(
        summary,
        str(json_path),
        str(fig_path),
    )

    result = {
        "trajectory_var_ratio": summary["trajectory_var_ratio"],
        "mean_within_trajectory_variance": summary["mean_within_trajectory_variance"],
        "between_trajectory_variance": summary["between_trajectory_variance"],
        "num_videos": summary["num_videos"],
        "videos": summary["videos"],
    }
    print(f"trajectory_var_ratio: {result['trajectory_var_ratio']:.6f}")
    print(f"mean_within_trajectory_variance: {result['mean_within_trajectory_variance']:.6f}")
    print(f"between_trajectory_variance: {result['between_trajectory_variance']:.6f}")
    return result


def run_analyses_for_frames(
    analysis_names: list[str],
    frames: list[np.ndarray],
    frame_indices: list[int],
    model,
    cfg: DictConfig,
    output_dir: Path,
    selected_single_frame_methods: list[str],
    selected_temporal_methods: list[str],
    selected_multi_frame_methods: list[str],
    single_frame_save_outputs: bool = True,
    single_frame_save_summaries: bool = True,
    representative_frame_position: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Run selected analyses for one video's extracted frames."""
    combined_summary = {}
    root_summary = {}

    if selected_single_frame_methods:
        print(f"Running single-frame analyses: {selected_single_frame_methods}")
        single_frame_summary = process_single_frame_analysis(
            selected_single_frame_methods,
            frames,
            model,
            cfg,
            output_dir,
            frame_indices=frame_indices if not single_frame_save_outputs else None,
            save_frame_outputs=single_frame_save_outputs,
            save_frame_summaries=single_frame_save_summaries,
            representative_frame_position=representative_frame_position,
        )
        combined_summary.update(single_frame_summary)

    if selected_temporal_methods:
        if len(frames) < 2:
            raise ValueError(f"Temporal analyses {selected_temporal_methods} require at least 2 frames, got {len(frames)}")
        print(f"Running temporal analyses: {selected_temporal_methods}")
        root_summary.update(process_temporal_analysis(selected_temporal_methods, frames, model, cfg, output_dir))

    if selected_multi_frame_methods:
        if len(frames) < 2:
            raise ValueError(f"Multi-frame analyses {selected_multi_frame_methods} require at least 2 frames, got {len(frames)}")
        print(f"Running multi-frame analyses: {selected_multi_frame_methods}")
        for method_name in selected_multi_frame_methods:
            if method_name == "within_between_var":
                root_summary.update(process_within_between_var_analysis(frames, frame_indices, model, cfg, output_dir))

    if root_summary:
        summary_path = output_dir / "summary.json"
        write_grouped_summary(
            summary_path,
            root_summary,
            analysis_names,
            metadata={"frame_count": len(frames)},
        )
        print(f"Grouped summary saved to: {summary_path}")
        combined_summary.update(root_summary)

    return combined_summary


def process_video_file(
    video_path: Path,
    analysis_names: list[str],
    model,
    cfg: DictConfig,
    output_dir: Path,
    selected_single_frame_methods: list[str],
    selected_temporal_methods: list[str],
    selected_multi_frame_methods: list[str],
    frame_start: int | None,
    frame_end: int | None,
    stride: int,
    folder_mode: bool = False,
) -> dict[str, dict[str, Any]]:
    """Extract one video and run all selected analyses."""
    representative_frame_index = cfg.get("representative_frame_index", 0)
    if representative_frame_index is not None:
        representative_frame_index = int(representative_frame_index)

    print(f"Extracting frames from video: {video_path}")
    if folder_mode:
        frames, frame_indices = extract_frames(str(video_path), None, None, stride)
    else:
        frames, frame_indices = extract_frames(str(video_path), frame_start, frame_end, stride)
    print(f"Extracted {len(frames)} frames")

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    representative_frame_position = None
    if folder_mode:
        representative_frame_position = select_representative_frame_position(
            frame_indices,
            representative_frame_index,
        )
        print(
            "Representative frame for saved PNGs: "
            f"source frame {frame_indices[representative_frame_position]}"
        )

    return run_analyses_for_frames(
        analysis_names,
        frames,
        frame_indices,
        model,
        cfg,
        output_dir,
        selected_single_frame_methods,
        selected_temporal_methods,
        selected_multi_frame_methods,
        single_frame_save_outputs=not folder_mode,
        single_frame_save_summaries=not folder_mode,
        representative_frame_position=representative_frame_position,
    )


def process_video_directory(
    video_dir: Path,
    analysis_names: list[str],
    model,
    cfg: DictConfig,
    output_dir: Path,
    selected_single_frame_methods: list[str],
    selected_temporal_methods: list[str],
    selected_multi_frame_methods: list[str],
    stride: int,
    selected_multi_video_methods: list[str] | None = None,
) -> None:
    """Run analyses for every video in a directory and save a cross-video summary."""
    selected_multi_video_methods = selected_multi_video_methods or []
    video_paths = find_video_files(video_dir, sample_interval=5)
    if not video_paths:
        raise ValueError(f"No supported video files found in directory: {video_dir}")

    print(f"Found {len(video_paths)} videos in directory: {video_dir}")
    used_names = set()
    per_video_summaries = []
    video_records = []

    has_per_video_methods = bool(
        selected_single_frame_methods
        or selected_temporal_methods
        or selected_multi_frame_methods
    )
    batch_size = cfg.get("batch_size", 8)
    trajectory_accumulator = None
    if "trajectory_var_ratio" in selected_multi_video_methods:
        print("Running multi-video analyses: ['trajectory_var_ratio']")
        trajectory_accumulator = trajectory_var_ratio.TrajectoryVarRatioAccumulator()

    for video_idx, video_path in tqdm(enumerate(video_paths, start=1), total=len(video_paths), desc="Processing videos", ncols=80):
        video_output_name = safe_output_name(video_path, used_names)
        video_output_dir = output_dir / video_output_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{video_idx}/{len(video_paths)}] Processing video: {video_path.name}")

        # Extract once so per-video and multi-video methods share the same frames.
        frames, frame_indices = extract_frames(str(video_path), None, None, stride)
        print(f"Extracted {len(frames)} frames")
        if len(frames) == 0:
            if has_per_video_methods:
                raise ValueError(f"No frames extracted from video: {video_path}")
            print(f"Skipping {video_path.name}: no frames extracted")
            continue

        summary = {}
        if has_per_video_methods:
            representative_frame_index = cfg.get("representative_frame_index", 0)
            if representative_frame_index is not None:
                representative_frame_index = int(representative_frame_index)
            representative_frame_position = select_representative_frame_position(
                frame_indices,
                representative_frame_index,
            )
            print(
                "Representative frame for saved PNGs: "
                f"source frame {frame_indices[representative_frame_position]}"
            )
            per_video_analysis_names = [
                name for name in analysis_names if name not in selected_multi_video_methods
            ]
            summary = run_analyses_for_frames(
                per_video_analysis_names,
                frames,
                frame_indices,
                model,
                cfg,
                video_output_dir,
                selected_single_frame_methods,
                selected_temporal_methods,
                selected_multi_frame_methods,
                single_frame_save_outputs=False,
                single_frame_save_summaries=False,
                representative_frame_position=representative_frame_position,
            )

        if trajectory_accumulator is not None:
            pooled_features = extract_pooled_features(frames, model, batch_size)
            record = trajectory_accumulator.add(video_output_name, pooled_features)
            print(
                f"  trajectory_var_ratio frames={record['num_frames']}, "
                f"within_var={record['within_trajectory_variance']:.6f}"
            )

        if summary:
            per_video_summaries.append(summary)
        video_records.append(
            {
                "video_path": str(video_path),
                "output_dir": str(video_output_dir),
                "analyses": sorted(summary.keys()),
            }
        )

    all_video_summary = average_video_analysis_summaries(per_video_summaries)
    if trajectory_accumulator is not None:
        all_video_summary["trajectory_var_ratio"] = finalize_trajectory_var_ratio(
            trajectory_accumulator,
            output_dir,
        )

    summary_path = output_dir / "all_videos_summary.json"
    write_grouped_summary(
        summary_path,
        all_video_summary,
        analysis_names,
        metadata={
            "video_count": len(video_paths),
            "stride": int(stride),
            "representative_frame_index": cfg.get("representative_frame_index", 0),
            "videos": video_records,
        },
    )
    print(f"All-video summary saved to: {summary_path}")


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for encoder testing framework."""
    # Configuration
    device = cfg.get("device", "cuda")
    video_path = cfg.get("video_path")
    frame_start = cfg.get("frame_start", None)
    frame_end = cfg.get("frame_end", None)
    stride = cfg.get("stride", 1)
    analysis = cfg.get("analysis", None)
    analysis_name = cfg.get("analysis_name", None)
    ckpt_dir = cfg.model.get("ckpt_dir", None)
    if analysis is None:
        analysis = analysis_name
    
    if video_path is None:
        raise ValueError("video_path must be specified")
    if analysis is None:
        raise ValueError("analysis must be specified (e.g., analysis=pca_vis or analysis=[avg_token_cos,dist_sim_decay])")
    analysis_names = normalize_analysis_names(analysis)
    if not analysis_names:
        raise ValueError("analysis must contain at least one method")
    
    # Determine output directory
    model_name = cfg.model.model_name.split("/")[-1]
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model: {model_name}")
    print(f"Analysis: {analysis_names}")
    video_path_obj = Path(video_path)

    print(f"Video: {video_path}")
    print(f"Frame range: {frame_start} - {frame_end}, stride: {stride}")
    print(f"Output directory: {output_dir}")

    # Categorize analysis methods
    single_frame_methods = [
        "pca_vis", "avg_token_cos", "dist_sim_decay", "mean_token_norm",
        "neighbor_sim", "token_cov_rank", "token_norm_entropy", "token_norm_var",
        "token_to_global", "frequency_metrics"
    ]
    
    temporal_methods = [
        "temporal_smoothness", "temporal_cosine_shift", "lag_distance_curve",
        "temporal_variance", "temporal_effective_rank", "temporal_spectral_entropy",
        "autocorrelation", "total_trajectory_variation", "patch_temporal_smoothness",
        "temporal_token_norm_entropy"
    ]

    multi_frame_methods = [
        "within_between_var"
    ]

    multi_video_methods = [
        "trajectory_var_ratio"
    ]
    
    available_methods = (
        single_frame_methods
        + temporal_methods
        + multi_frame_methods
        + multi_video_methods
    )
    unknown_methods = [name for name in analysis_names if name not in available_methods]
    if unknown_methods:
        raise ValueError(f"Unknown analysis method(s): {unknown_methods}. Available methods: {available_methods}")

    selected_single_frame_methods = [name for name in analysis_names if name in single_frame_methods]
    selected_temporal_methods = [name for name in analysis_names if name in temporal_methods]
    selected_multi_frame_methods = [name for name in analysis_names if name in multi_frame_methods]
    selected_multi_video_methods = [name for name in analysis_names if name in multi_video_methods]

    # Load model
    print("Loading model...")
    model = hydra.utils.instantiate(cfg.model, device=device)
    if ckpt_dir is not None:
        load_vision_checkpoint(model, ckpt_dir, device)
    model.eval()

    if video_path_obj.is_dir():
        print("Video path is a directory; frame_start/frame_end will be ignored.")
        process_video_directory(
            video_path_obj,
            analysis_names,
            model,
            cfg,
            output_dir,
            selected_single_frame_methods,
            selected_temporal_methods,
            selected_multi_frame_methods,
            stride,
            selected_multi_video_methods=selected_multi_video_methods,
        )
    else:
        if selected_multi_video_methods:
            raise ValueError(
                f"Multi-video analyses {selected_multi_video_methods} require "
                "video_path to be a directory containing multiple videos"
            )
        process_video_file(
            video_path_obj,
            analysis_names,
            model,
            cfg,
            output_dir,
            selected_single_frame_methods,
            selected_temporal_methods,
            selected_multi_frame_methods,
            frame_start,
            frame_end,
            stride,
            folder_mode=False,
        )
    
    print("Done!")


if __name__ == "__main__":
    main()

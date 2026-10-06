"""Single-frame analysis execution and dispatch."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from analyse import (
    avg_token_cos,
    dist_sim_decay,
    frequency_metrics,
    mean_token_norm,
    neighbor_sim,
    pca_vis,
    token_cov_rank,
    token_norm_entropy,
    token_norm_var,
    token_to_global,
)
from tools.summary import (
    append_metric,
    capture_summary,
    summarize_single_frame_metrics,
    write_grouped_summary,
)


@dataclass
class SingleFrameContext:
    tokens: np.ndarray
    grid_h: int
    grid_w: int
    cfg: DictConfig
    output_dir: Path
    frame_output_dir: Path | None
    ref_img_path: Path | None
    should_save_frame_outputs: bool
    save_frame_summaries: bool


@dataclass
class SingleFrameResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    frame_summary: dict[str, Any] | None = None


def _run_pca_vis(context: SingleFrameContext) -> SingleFrameResult:
    if context.should_save_frame_outputs:
        pca_rgb = pca_vis.pca_to_rgb(context.tokens, context.grid_h, context.grid_w)
        pca_vis.make_side_by_side(
            str(context.ref_img_path),
            pca_rgb,
            str(context.frame_output_dir / "pca_vis.png"),
        )
    return SingleFrameResult()


def _run_avg_token_cos(context: SingleFrameContext) -> SingleFrameResult:
    avg_cos, cos_matrix = avg_token_cos.compute_avg_token_cos(context.tokens)
    result = SingleFrameResult(metrics={"avg_cos_values": avg_cos})

    if context.should_save_frame_outputs:
        avg_token_cos.make_cos_figure(
            str(context.ref_img_path),
            cos_matrix,
            avg_cos,
            str(context.frame_output_dir / "avg_token_cos.png"),
        )
    if context.save_frame_summaries:
        result.frame_summary = capture_summary(
            avg_token_cos.save_cos_summary,
            context.frame_output_dir,
            avg_cos,
            cos_matrix,
        )
    return result


def _run_dist_sim_decay(context: SingleFrameContext) -> SingleFrameResult:
    bucket_centers, bucket_means, bucket_counts = dist_sim_decay.compute_dist_sim_decay(
        context.tokens,
        context.grid_h,
        context.grid_w,
    )
    result = SingleFrameResult(
        metrics={
            "decay_curves": (bucket_centers, bucket_means, bucket_counts),
        }
    )

    if context.should_save_frame_outputs:
        dist_sim_decay.make_decay_figure(
            str(context.ref_img_path),
            bucket_centers,
            bucket_means,
            bucket_counts,
            str(context.frame_output_dir / "dist_sim_decay.png"),
        )
    if context.save_frame_summaries:
        result.frame_summary = capture_summary(
            dist_sim_decay.save_decay_summary,
            context.frame_output_dir,
            bucket_centers,
            bucket_means,
            bucket_counts,
        )
    return result


def _run_mean_token_norm(context: SingleFrameContext) -> SingleFrameResult:
    norms, mean_norm, std_norm = mean_token_norm.compute_mean_token_norm(context.tokens)
    result = SingleFrameResult(
        metrics={
            "mean_norms": mean_norm,
            "std_norms": std_norm,
        }
    )

    if context.should_save_frame_outputs:
        mean_token_norm.make_norm_heatmap(
            str(context.ref_img_path),
            norms,
            context.grid_h,
            context.grid_w,
            str(context.frame_output_dir / "mean_token_norm.png"),
        )
    if context.save_frame_summaries:
        result.frame_summary = capture_summary(
            mean_token_norm.save_norm_summary,
            context.frame_output_dir,
            norms,
            mean_norm,
            std_norm,
        )
    return result


def _run_neighbor_sim(context: SingleFrameContext) -> SingleFrameResult:
    avg_ns, ns_map = neighbor_sim.compute_neighbor_sim(
        context.tokens,
        context.grid_h,
        context.grid_w,
    )
    result = SingleFrameResult(metrics={"neighbor_sim_values": avg_ns})

    if context.should_save_frame_outputs:
        neighbor_sim.make_ns_figure(
            str(context.ref_img_path),
            ns_map,
            avg_ns,
            str(context.frame_output_dir / "neighbor_sim.png"),
        )
    if context.save_frame_summaries:
        result.frame_summary = capture_summary(
            neighbor_sim.save_ns_summary,
            context.frame_output_dir,
            avg_ns,
            ns_map,
        )
    return result


def _run_token_cov_rank(context: SingleFrameContext) -> SingleFrameResult:
    eff_rank, eigenvalues = token_cov_rank.compute_token_cov_rank(context.tokens)
    result = SingleFrameResult(metrics={"eff_ranks": eff_rank})

    if context.should_save_frame_outputs:
        token_cov_rank.make_rank_figure(
            str(context.ref_img_path),
            eigenvalues,
            eff_rank,
            str(context.frame_output_dir / "token_cov_rank.png"),
        )
    if context.save_frame_summaries:
        result.frame_summary = capture_summary(
            token_cov_rank.save_rank_summary,
            context.frame_output_dir,
            eff_rank,
            eigenvalues,
        )
    return result


def _run_token_norm_entropy(context: SingleFrameContext) -> SingleFrameResult:
    norms, entropy, hist, bin_edges = token_norm_entropy.compute_token_norm_entropy(
        context.tokens,
    )
    result = SingleFrameResult(metrics={"entropies": entropy})

    if context.should_save_frame_outputs:
        token_norm_entropy.make_entropy_figure(
            str(context.ref_img_path),
            norms,
            entropy,
            hist,
            bin_edges,
            str(context.frame_output_dir / "token_norm_entropy.png"),
        )
    if context.save_frame_summaries:
        result.frame_summary = capture_summary(
            token_norm_entropy.save_entropy_summary,
            context.frame_output_dir,
            norms,
            entropy,
        )
    return result


def _run_token_norm_var(context: SingleFrameContext) -> SingleFrameResult:
    norms, var_norm = token_norm_var.compute_token_norm_var(context.tokens)
    result = SingleFrameResult(metrics={"var_norms": var_norm})

    if context.should_save_frame_outputs:
        token_norm_var.make_var_heatmap(
            str(context.ref_img_path),
            norms,
            context.grid_h,
            context.grid_w,
            var_norm,
            str(context.frame_output_dir / "token_norm_var.png"),
        )
    if context.save_frame_summaries:
        result.frame_summary = capture_summary(
            token_norm_var.save_var_summary,
            context.frame_output_dir,
            norms,
            var_norm,
        )
    return result


def _run_token_to_global(context: SingleFrameContext) -> SingleFrameResult:
    sims, mean_sim, var_sim = token_to_global.compute_token_to_global(context.tokens)
    result = SingleFrameResult(
        metrics={
            "mean_sims": mean_sim,
            "var_sims": var_sim,
        }
    )

    if context.should_save_frame_outputs:
        token_to_global.make_token_to_global_figure(
            str(context.ref_img_path),
            sims,
            context.grid_h,
            context.grid_w,
            mean_sim,
            var_sim,
            str(context.frame_output_dir / "token_to_global.png"),
        )
    if context.save_frame_summaries:
        result.frame_summary = capture_summary(
            token_to_global.save_token_to_global_summary,
            context.frame_output_dir,
            sims,
            mean_sim,
            var_sim,
        )
    return result


def _run_frequency_metrics(context: SingleFrameContext) -> SingleFrameResult:
    low_thr = context.cfg.get("low_thr", 0.15)
    mid_thr = context.cfg.get("mid_thr", 0.35)
    remove_dc = context.cfg.get("remove_dc", True)
    normalize_channel = context.cfg.get("normalize_channel", False)
    num_radial_bins = context.cfg.get("num_radial_bins", None)

    metrics, power_spectrum, radial_freq, radial_power = (
        frequency_metrics.compute_frequency_metrics(
            context.tokens,
            context.grid_h,
            context.grid_w,
            low_thr=low_thr,
            mid_thr=mid_thr,
            remove_dc=remove_dc,
            normalize_channel=normalize_channel,
            num_radial_bins=num_radial_bins,
        )
    )
    result = SingleFrameResult(metrics={"frequency_metrics": metrics})

    if context.should_save_frame_outputs:
        frequency_metrics.make_frequency_figure(
            str(context.ref_img_path),
            power_spectrum,
            radial_freq,
            radial_power,
            metrics,
            str(context.frame_output_dir / "frequency_metrics.png"),
        )
    if context.save_frame_summaries:
        result.frame_summary = capture_summary(
            frequency_metrics.save_frequency_summary,
            context.frame_output_dir,
            metrics,
            radial_freq,
            radial_power,
        )
    return result


SINGLE_FRAME_HANDLERS = {
    "pca_vis": _run_pca_vis,
    "avg_token_cos": _run_avg_token_cos,
    "dist_sim_decay": _run_dist_sim_decay,
    "mean_token_norm": _run_mean_token_norm,
    "neighbor_sim": _run_neighbor_sim,
    "token_cov_rank": _run_token_cov_rank,
    "token_norm_entropy": _run_token_norm_entropy,
    "token_norm_var": _run_token_norm_var,
    "token_to_global": _run_token_to_global,
    "frequency_metrics": _run_frequency_metrics,
}


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
    """Process single-frame analysis methods."""
    batch_size = cfg.get("batch_size", 8)
    grid_h, grid_w = None, None
    metrics_accumulator = {}

    for i in tqdm(
        range(0, len(frames), batch_size),
        desc="Processing frames",
        total=len(frames) // batch_size,
    ):
        batch_frames = frames[i:i + batch_size]
        if not batch_frames:
            continue

        visual_tokens = model(batch_frames)
        visual_tokens_np = visual_tokens.detach().float().cpu().numpy()
        B, C, H, W = visual_tokens_np.shape

        if grid_h is None:
            grid_h, grid_w = H, W

        for b in range(B):
            frame_idx = i + b
            if frame_idx >= len(frames):
                break

            tokens = visual_tokens_np[b].reshape(C, -1).T
            source_frame_idx = (
                frame_indices[frame_idx]
                if frame_indices is not None
                else frame_idx
            )
            should_save_frame_outputs = (
                save_frame_outputs
                or frame_idx == representative_frame_position
            )
            frame_output_dir = None
            ref_img_path = None
            if should_save_frame_outputs:
                frame_output_dir = output_dir / f"frame_{source_frame_idx:04d}"
                frame_output_dir.mkdir(parents=True, exist_ok=True)
                ref_img_path = frame_output_dir / "reference.png"
                cv2.imwrite(
                    str(ref_img_path),
                    cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR),
                )

            frame_summaries = {}
            for analysis_name in analysis_names:
                handler = SINGLE_FRAME_HANDLERS.get(analysis_name)
                if handler is None:
                    raise ValueError(
                        f"Unknown single-frame analysis method: {analysis_name}"
                    )

                context = SingleFrameContext(
                    tokens=tokens,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    cfg=cfg,
                    output_dir=output_dir,
                    frame_output_dir=frame_output_dir,
                    ref_img_path=ref_img_path,
                    should_save_frame_outputs=should_save_frame_outputs,
                    save_frame_summaries=save_frame_summaries,
                )
                result = handler(context)
                for key, value in result.metrics.items():
                    append_metric(metrics_accumulator, analysis_name, key, value)
                if result.frame_summary is not None:
                    frame_summaries[analysis_name] = result.frame_summary

            if save_frame_summaries and frame_summaries:
                write_grouped_summary(
                    frame_output_dir / "summary.json",
                    frame_summaries,
                    analysis_names,
                    metadata={"frame_index": source_frame_idx},
                )

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

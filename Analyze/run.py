"""Encoder test framework for analyzing visual encoder representations.

Usage:
    # Single-frame analysis (averaged over all frames)
    python run.py model=dinov3 analysis=pca_vis video_path=/path/to/video.mp4
    python run.py model=clip analysis=avg_token_cos video_path=/path/to/video.mp4

    # Temporal analysis
    python run.py model=dinov3 analysis=temporal_smoothness video_path=/path/to/video.mp4
    python run.py model=dinov3 analysis=autocorrelation video_path=/path/to/video.mp4 frame_start=0 frame_end=100

    # Specify stride
    python run.py model=sam analysis=temporal_variance video_path=/path/to/video.mp4 stride=5
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from tools.checkpoint import load_vision_checkpoint
from tools.pipeline import process_video_directory, process_video_file
from tools.registry import (
    ANALYSIS_METHOD_GROUPS,
    available_methods,
    normalize_analysis_names,
)


# Register custom resolvers for formatting the Hydra output directory.
OmegaConf.register_new_resolver(
    "format_model",
    lambda m: m.split("/")[-1],
)
OmegaConf.register_new_resolver(
    "format_dir",
    lambda a, add: (a + "_" + add) if add else a,
)
OmegaConf.register_new_resolver(
    "format_path",
    lambda m, a, add: (m.split("/")[-1] + "_" + a + "_" + add)
    if add
    else (m.split("/")[-1] + "_" + a),
)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for encoder testing framework."""
    device = cfg.get("device", "cuda")
    video_path = cfg.get("video_path")
    frame_start = cfg.get("frame_start", None)
    frame_end = cfg.get("frame_end", None)
    stride = cfg.get("stride", 1)
    analysis = cfg.get("analysis", None)
    ckpt_dir = cfg.model.get("ckpt_dir", None)

    if video_path is None:
        raise ValueError("video_path must be specified")
    if analysis is None:
        raise ValueError(
            "analysis must be specified (e.g., analysis=pca_vis "
            "or analysis=[avg_token_cos,dist_sim_decay])"
        )
    analysis_names = normalize_analysis_names(analysis)
    if not analysis_names:
        raise ValueError("analysis must contain at least one method")

    model_name = cfg.model.model_name.split("/")[-1]
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Analysis: {analysis_names}")
    video_path_obj = Path(video_path)

    print(f"Video: {video_path}")
    print(f"Frame range: {frame_start} - {frame_end}, stride: {stride}")
    print(f"Output directory: {output_dir}")

    single_frame_methods = ANALYSIS_METHOD_GROUPS["single_frame"]
    temporal_methods = ANALYSIS_METHOD_GROUPS["temporal"]
    multi_frame_methods = ANALYSIS_METHOD_GROUPS["multi_frame"]
    multi_video_methods = ANALYSIS_METHOD_GROUPS["multi_video"]

    unknown_methods = [
        name for name in analysis_names if name not in available_methods()
    ]
    if unknown_methods:
        raise ValueError(
            f"Unknown analysis method(s): {unknown_methods}. "
            f"Available methods: {available_methods()}"
        )

    selected_single_frame_methods = [
        name for name in analysis_names if name in single_frame_methods
    ]
    selected_temporal_methods = [
        name for name in analysis_names if name in temporal_methods
    ]
    selected_multi_frame_methods = [
        name for name in analysis_names if name in multi_frame_methods
    ]
    selected_multi_video_methods = [
        name for name in analysis_names if name in multi_video_methods
    ]

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

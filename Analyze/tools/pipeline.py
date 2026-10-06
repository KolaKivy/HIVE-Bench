"""Video-level and directory-level analysis orchestration."""

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from tools.multi_frame import (
    MULTI_FRAME_HANDLERS,
    MULTI_VIDEO_HANDLERS,
    extract_pooled_features,
)
from tools.single_frame import process_single_frame_analysis
from tools.summary import (
    average_video_analysis_summaries,
    write_grouped_summary,
)
from tools.temporal import process_temporal_analysis
from tools.video import (
    extract_frames,
    find_video_files,
    safe_output_name,
    select_representative_frame_position,
)


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
            raise ValueError(
                f"Temporal analyses {selected_temporal_methods} require at least 2 frames, "
                f"got {len(frames)}"
            )
        print(f"Running temporal analyses: {selected_temporal_methods}")
        root_summary.update(
            process_temporal_analysis(
                selected_temporal_methods,
                frames,
                model,
                cfg,
                output_dir,
            )
        )

    if selected_multi_frame_methods:
        if len(frames) < 2:
            raise ValueError(
                f"Multi-frame analyses {selected_multi_frame_methods} require at least 2 frames, "
                f"got {len(frames)}"
            )
        print(f"Running multi-frame analyses: {selected_multi_frame_methods}")
        for method_name in selected_multi_frame_methods:
            handler = MULTI_FRAME_HANDLERS.get(method_name)
            if handler is not None:
                root_summary.update(
                    handler(frames, frame_indices, model, cfg, output_dir)
                )

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
        frames, frame_indices = extract_frames(
            str(video_path),
            frame_start,
            frame_end,
            stride,
        )
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
    trajectory_method = (
        MULTI_VIDEO_HANDLERS.get("trajectory_var_ratio")
        if "trajectory_var_ratio" in selected_multi_video_methods
        else None
    )
    if trajectory_method is not None:
        print("Running multi-video analyses: ['trajectory_var_ratio']")
    trajectory_accumulator = (
        trajectory_method.create_accumulator()
        if trajectory_method is not None
        else None
    )

    for video_idx, video_path in tqdm(
        enumerate(video_paths, start=1),
        total=len(video_paths),
        desc="Processing videos",
        ncols=80,
    ):
        video_output_name = safe_output_name(video_path, used_names)
        video_output_dir = output_dir / video_output_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{video_idx}/{len(video_paths)}] Processing video: {video_path.name}")

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
    if trajectory_method is not None:
        all_video_summary[trajectory_method.name] = trajectory_method.finalize(
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

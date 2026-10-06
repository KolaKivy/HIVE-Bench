"""Multi-frame and multi-video analysis execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from analyse import trajectory_var_ratio, within_between_var
from tools.summary import capture_summary


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
        visual_tokens_np = visual_tokens.detach().float().cpu().numpy()
        B, C, _, _ = visual_tokens_np.shape

        for b in range(B):
            source_frame_idx = (
                frame_indices[i + b]
                if i + b < len(frame_indices)
                else i + b
            )
            tokens = visual_tokens_np[b].reshape(C, -1).T
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
        all_pooled_features.append(visual_tokens.mean(dim=(2, 3)).detach().float().cpu().numpy())

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
    print(
        "mean_within_trajectory_variance: "
        f"{result['mean_within_trajectory_variance']:.6f}"
    )
    print(f"between_trajectory_variance: {result['between_trajectory_variance']:.6f}")
    return result


MULTI_FRAME_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "within_between_var": process_within_between_var_analysis,
}


@dataclass(frozen=True)
class MultiVideoMethod:
    name: str
    create_accumulator: Callable[[], Any]
    finalize: Callable[[Any, Path], dict[str, Any]]


MULTI_VIDEO_HANDLERS: dict[str, MultiVideoMethod] = {
    "trajectory_var_ratio": MultiVideoMethod(
        name="trajectory_var_ratio",
        create_accumulator=trajectory_var_ratio.TrajectoryVarRatioAccumulator,
        finalize=finalize_trajectory_var_ratio,
    ),
}

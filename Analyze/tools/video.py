"""Video discovery and frame extraction helpers."""

import re
from pathlib import Path

import cv2


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


def extract_frames(
    video_path: str,
    frame_start: int | None = None,
    frame_end: int | None = None,
    stride: int = 1,
):
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
    start_index: int = 0,
) -> list[Path]:
    """Return all supported video files directly inside a directory, sampled every N files."""
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


def select_representative_frame_position(
    frame_indices: list[int],
    representative_frame_index: int | None,
) -> int:
    """Select the extracted-frame position closest to the requested source frame."""
    if not frame_indices:
        raise ValueError("Cannot select representative frame from an empty frame list")
    if representative_frame_index is None:
        return 0
    return min(
        range(len(frame_indices)),
        key=lambda i: abs(frame_indices[i] - representative_frame_index),
    )

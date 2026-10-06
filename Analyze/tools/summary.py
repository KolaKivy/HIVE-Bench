"""Helpers for capturing and averaging grouped analysis summaries."""

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


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


def append_metric(
    accumulator: dict[str, dict[str, list[Any]]],
    analysis_name: str,
    key: str,
    value: Any,
):
    accumulator.setdefault(analysis_name, {}).setdefault(key, []).append(value)


def summarize_single_frame_metrics(
    metrics_accumulator: dict[str, dict[str, list[Any]]],
) -> dict[str, dict[str, Any]]:
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
                    "pair_counts": avg_counts.astype(int).tolist(),
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
    return isinstance(
        value,
        (int, float, np.integer, np.floating),
    ) and not isinstance(value, bool)


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


def average_video_analysis_summaries(
    video_summaries: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
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

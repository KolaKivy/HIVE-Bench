"""Analysis method metadata and normalized CLI input handling."""

from typing import Any

from omegaconf import ListConfig

from tools.multi_frame import (
    MULTI_FRAME_HANDLERS,
    MULTI_VIDEO_HANDLERS,
)
from tools.single_frame import SINGLE_FRAME_HANDLERS
from tools.temporal import TEMPORAL_HANDLERS


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


ANALYSIS_METHOD_GROUPS = {
    "single_frame": list(SINGLE_FRAME_HANDLERS),
    "temporal": list(TEMPORAL_HANDLERS),
    "multi_frame": list(MULTI_FRAME_HANDLERS),
    "multi_video": list(MULTI_VIDEO_HANDLERS),
}


def available_methods() -> list[str]:
    """Return supported methods in the same order used by the CLI validation error."""
    return [
        method_name
        for method_names in ANALYSIS_METHOD_GROUPS.values()
        for method_name in method_names
    ]

"""Canonical framework names and saved-checkpoint metadata migration."""

from collections.abc import Mapping


# Only saved identifiers need migration; old Python module paths are not aliases.
LEGACY_FRAMEWORK_NAMES = {
    "DinoGR00T": "VisionGR00T",
    "Dinov3CLIPGR00T": "VisionCLIPGR00T",
    "QwenVisionGR00T": "VLMVisionGR00T",
}


def normalize_framework_config(config):
    """Normalize framework.name in a dictionary, OmegaConf, or namespace in place."""
    if isinstance(config, Mapping):
        framework = config.get("framework")
    else:
        framework = getattr(config, "framework", None)
    if framework is None:
        return config
    name = framework.get("name") if isinstance(framework, Mapping) else getattr(framework, "name", None)
    canonical = LEGACY_FRAMEWORK_NAMES.get(name, name)
    if canonical != name:
        if isinstance(framework, Mapping):
            framework["name"] = canonical
        else:
            framework.name = canonical
    return config

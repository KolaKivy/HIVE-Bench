"""Benchmark-local LeRobot registry discovery."""
from __future__ import annotations
import importlib.util
import logging
import sys
from pathlib import Path
from hivebench.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP as _BASE_CONFIG_MAP
from hivebench.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from hivebench.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES as _BASE_MIXTURES
logger = logging.getLogger(__name__)
# Older HIVE-Bench enum lacks newer upstream-only tags. They are data-loader
# metadata here; map unknown tags to the generic projector without touching models.
for _tag_name in ("ARX_X5",):
    if not hasattr(EmbodimentTag, _tag_name):
        setattr(EmbodimentTag, _tag_name, EmbodimentTag.NEW_EMBODIMENT)
ROBOT_TYPE_CONFIG_MAP = dict(_BASE_CONFIG_MAP)
DATASET_NAMED_MIXTURES = dict(_BASE_MIXTURES)
_LEGACY_TAG_OVERRIDES = {}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}
_DISCOVERED = False
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
def _derive_tags():
    tags = {k: getattr(v, "embodiment_tag", EmbodimentTag.NEW_EMBODIMENT) for k, v in ROBOT_TYPE_CONFIG_MAP.items()}
    tags.update(_LEGACY_TAG_OVERRIDES)
    return tags
def discover_and_merge():
    global _DISCOVERED, ROBOT_TYPE_TO_EMBODIMENT_TAG
    if _DISCOVERED: return
    _DISCOVERED = True
    repo_root = Path(__file__).resolve().parents[4]
    bench_root = repo_root / "Bench"
    for registry_dir in sorted(bench_root.glob("*/train_files/data_registry")) if bench_root.is_dir() else []:
        config_file = registry_dir / "data_config.py"
        if not config_file.is_file(): continue
        bench_name = registry_dir.parents[1].name
        module = _load(config_file, f"_hivebench_data_registry_{bench_name}")
        if module is None: continue
        if hasattr(module, "ROBOT_TYPE_CONFIG_MAP"): ROBOT_TYPE_CONFIG_MAP.update(module.ROBOT_TYPE_CONFIG_MAP)
        if hasattr(module, "DATASET_NAMED_MIXTURES"): DATASET_NAMED_MIXTURES.update(module.DATASET_NAMED_MIXTURES)
        if hasattr(module, "ROBOT_TYPE_TO_EMBODIMENT_TAG"): _LEGACY_TAG_OVERRIDES.update(module.ROBOT_TYPE_TO_EMBODIMENT_TAG)
    ROBOT_TYPE_TO_EMBODIMENT_TAG = _derive_tags()
discover_and_merge()

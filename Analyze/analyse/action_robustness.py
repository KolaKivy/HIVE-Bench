"""Evaluate texture robustness of RoboTwin action policies.

The static-texture datasets contain 100 independent initial observations per
task: object pose, robot state, and camera are fixed while table/wall textures
change.  This script evaluates every selected policy using the *same* initial
diffusion noise for every texture, so output variance measures sensitivity to
appearance rather than diffusion sampling noise.

The script deliberately reads RoboTwin HDF5 directly.  It does not need, and
must not overwrite, the LeRobot training dataset.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterator

import cv2
import h5py
import numpy as np
import torch
import yaml


# Hugging Face reads these variables while its modules are imported.  Honour
# our CLI flag at that point instead of setting it later in ``main``.
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


ANALYZE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
HIVEBENCH_ROOT = Path(os.environ.get("HIVEBENCH_ROOT", PROJECT_ROOT)).expanduser()
if str(HIVEBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(HIVEBENCH_ROOT))
HIVE_POLICY_ROOT = PROJECT_ROOT / "Policy"
if str(HIVE_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(HIVE_POLICY_ROOT))
HIVE_CHECKPOINT_ROOT = Path(
    os.environ.get("HIVE_CHECKPOINT_ROOT", PROJECT_ROOT / "playground/Checkpoints")
).expanduser()
HIVE_VLM_RUNS = {
    "qwenvision_robotwin_fulllayer_new",
    "qwenvision_robotwin_layer16_new",
    "depthvlm_layer16_robotwin_new",
    "depthvlm_fulllayer_robotwin_new",
    "xiaomi_robotwin_fulllayer_new",
    "xiaomi_robotwin_new",
}
LOCAL_VLM_PATHS = {
    "qwen3-vl": HIVEBENCH_ROOT / "playground/Pretrained_models/Qwen3-VL-4B-Instruct",
    "depthvlm": HIVEBENCH_ROOT / "playground/Pretrained_models/DepthVLM-4B",
    "xiaomi": HIVEBENCH_ROOT / "playground/Pretrained_models/Xiaomi-Robotics",
}

# These research backbones are installed from local source checkouts rather
# than PyPI.  Add them here so the evaluator is self-contained; users no
# longer need to remember matching PYTHONPATH exports before a run.
OPTIONAL_BACKBONE_ROOTS = (
    Path(os.environ.get("SPA_ROOT", THIRD_PARTY_ROOT / "SPA")).expanduser(),
    Path(os.environ.get("VGGT_OMEGA_ROOT", THIRD_PARTY_ROOT / "VGGT_Omega")).expanduser(),
)
for backbone_root in OPTIONAL_BACKBONE_ROOTS:
    if backbone_root.is_dir() and str(backbone_root) not in sys.path:
        sys.path.insert(0, str(backbone_root))

VOLTRON_CACHE = ANALYZE_ROOT / "cache"

from hivebench.model.framework.base_framework import baseframework, build_framework
from hivebench.model.framework.share_tools import dict_to_namespace, read_mode_config


TASK_PROMPTS = {
    "adjust_bottle": "Move the bottle to the target position.",
    "beat_block_hammer": "Pick up the hammer and hit the block.",
    "click_alarmclock": "Press the top button on the alarm clock.",
    "handover_block": "Hand over the block and place it on the target.",
    "move_playingcard_away": "Move the playing cards away.",
    "open_laptop": "Open the laptop.",
    "place_can_basket": "Place the can in the basket.",
    "rotate_qrcode": "Rotate the QR code sign.",
    "stamp_seal": "Stamp the seal on the target.",
    "turn_switch": "Turn the switch.",
    "place_burger_fries": "Place the burger and fries on the tray.",
    "lift_pot": "Lift the pot.",
}

HIVEBENCH_CAMERA_KEYS = ("front_camera", "left_camera", "right_camera")
HIVE_VLM_CAMERA_KEYS = ("head_camera", "left_camera", "right_camera")
EXCLUDED_CHECKPOINTS = {"all_task_robotwin_dinov2_delta"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(
            os.environ.get("HIVEBENCH_CHECKPOINT_ROOT", HIVEBENCH_ROOT / "playground/Checkpoints")
        ).expanduser(),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.environ.get("ROBOTWIN_DATA_ROOT", THIRD_PARTY_ROOT / "RoboTwin/data")
        ).expanduser(),
    )
    parser.add_argument(
        "--dataset-name",
        default="demo_randomized_static_texture_unseen_100",
        help="Per-task static collection directory name under dataset-root/<task>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/action_robustness"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force Hugging Face/Transformers offline mode and use locally cached backbone files only.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional checkpoint directory names. Default evaluates HIVE-Bench all_task_robotwin_* plus supported HIVE VLM runs.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task names; default evaluates all 12 static-texture tasks.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per task for a smoke test. Omit for all 100 frames.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Regenerate aggregate CSV/JSON files from completed model summaries without loading a policy.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def checkpoint_paths(root: Path, selected: list[str] | None) -> list[Path]:
    paths = []
    wanted = set(selected) if selected else None
    for run_dir in sorted(root.glob("all_task_robotwin_*")):
        if run_dir.name in EXCLUDED_CHECKPOINTS:
            continue
        if wanted is not None and run_dir.name not in wanted:
            continue
        checkpoint = choose_checkpoint(run_dir)
        if not (run_dir / "config.yaml").is_file() or not (run_dir / "dataset_statistics.json").is_file():
            raise FileNotFoundError(f"Checkpoint metadata missing next to {checkpoint}")
        paths.append(checkpoint)
    # HIVE VLMVisionGR00T checkpoints use a separate root and do not follow
    # the all_task_robotwin_* name convention.  Limit automatic discovery to
    # the explicitly supported VLM experiment list above.
    hive_names = HIVE_VLM_RUNS if wanted is None else wanted & HIVE_VLM_RUNS
    for name in sorted(hive_names):
        run_dir = HIVE_CHECKPOINT_ROOT / name
        if not run_dir.is_dir():
            continue
        if not (run_dir / "config.yaml").is_file() or not (run_dir / "dataset_statistics.json").is_file():
            continue
        paths.append(choose_checkpoint(run_dir))
    if wanted is not None:
        found = {path.parent.parent.name for path in paths}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Requested models not found (or excluded): {missing}")
    if not paths:
        raise ValueError(f"No eligible checkpoints under {root}")
    return paths


def choose_checkpoint(run_dir: Path) -> Path:
    """Prefer repository-provided key-compatible copies of a final checkpoint."""
    final_dir = run_dir / "final_model"
    candidates = [
        final_dir / "pytorch_model_fixed.pt",
        final_dir / "pytorch_model_correct.pt",
        final_dir / "pytorch_model_compatible.pt",
        final_dir / "pytorch_model.pt",
        final_dir / "pytorch_model_converted.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    step_checkpoints = sorted(
        (run_dir / "checkpoints").glob("steps_*_pytorch_model.pt"),
        key=lambda path: int(path.name.split("_")[1]),
        reverse=True,
    )
    if step_checkpoints:
        return step_checkpoints[0]
    raise FileNotFoundError(f"No final model checkpoint in {final_dir}")


@contextmanager
def local_voltron_dependencies() -> Iterator[None]:
    """Load Voltron from its local cache without downloading DistilBERT.

    Voltron's visual-only inference never calls its tokenizer or language
    encoder, but the upstream constructor still tries to download both.  The
    trained RoboTwin checkpoint contains the language-encoder parameters, so
    instantiate the standard DistilBERT architecture locally and let the
    checkpoint restore its exact weights immediately afterwards.
    """
    import transformers
    import voltron

    original_load = voltron.load
    original_tokenizer = transformers.AutoTokenizer.from_pretrained
    original_model = transformers.AutoModel.from_pretrained

    class UnusedTokenizer:
        # DistilBERT's standard special-token ids.  Voltron visual inference
        # does not tokenize language, but its constructor and a few helper
        # methods inspect these attributes.
        cls_token_id = 101
        sep_token_id = 102
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, language, return_tensors="pt", max_length=20, padding="max_length", truncation=True):
            batch = len(language) if isinstance(language, (list, tuple)) else 1
            input_ids = torch.full((batch, max_length), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((batch, max_length), dtype=torch.long)
            input_ids[:, 0] = self.cls_token_id
            input_ids[:, 1] = self.sep_token_id
            attention_mask[:, :2] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def local_model(model_name, *args, **kwargs):
        if model_name == "distilbert-base-uncased":
            return transformers.AutoModel.from_config(transformers.DistilBertConfig())
        return original_model(model_name, *args, **kwargs)

    def cached_load(model_id, device="cpu", freeze=True, cache=VOLTRON_CACHE):
        return original_load(model_id, device=device, freeze=freeze, cache=str(cache))

    transformers.AutoTokenizer.from_pretrained = UnusedTokenizer.from_pretrained
    transformers.AutoModel.from_pretrained = local_model
    voltron.load = cached_load
    try:
        yield
    finally:
        voltron.load = original_load
        transformers.AutoTokenizer.from_pretrained = original_tokenizer
        transformers.AutoModel.from_pretrained = original_model


def migrate_checkpoint_keys(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Adapt old text/RADIO wrapper prefixes to the instantiated model."""
    expected = set(model.state_dict())
    migrated = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in (
            ("text_encoder.embeddings.", "text_encoder.text_model.embeddings."),
            ("text_encoder.encoder.", "text_encoder.text_model.encoder."),
            ("text_encoder.final_layer_norm.", "text_encoder.text_model.final_layer_norm."),
            ("clip_text_model.embeddings.", "clip_text_model.text_model.embeddings."),
            ("clip_text_model.encoder.", "clip_text_model.text_model.encoder."),
            ("clip_text_model.final_layer_norm.", "clip_text_model.text_model.final_layer_norm."),
        ):
            new_key = new_key.replace(old, new)

        # C-RADIO checkpoints exist in both layouts.  The outer RADIO wrapper
        # has a `.model`, and its Torch Hub backbone has another `.model`.
        # Select the layout that the current framework actually instantiated.
        candidates = [new_key]
        if new_key.startswith("vision_encoder.model.model."):
            candidates.append(new_key.replace("vision_encoder.model.model.", "vision_encoder.model.", 1))
        elif new_key.startswith("vision_encoder.model."):
            candidates.append(new_key.replace("vision_encoder.model.", "vision_encoder.model.model.", 1))
        new_key = next((candidate for candidate in candidates if candidate in expected), new_key)

        if new_key in migrated:
            raise RuntimeError(f"Checkpoint key collision after migration: {key} -> {new_key}")
        migrated[new_key] = value
    return migrated


def is_hive_checkpoint(checkpoint: Path) -> bool:
    return HIVE_CHECKPOINT_ROOT in checkpoint.parents


def local_vlm_path(config) -> Path:
    configured = str(config.framework.qwenvl.base_vlm).lower()
    for marker, path in LOCAL_VLM_PATHS.items():
        if marker in configured:
            if not path.is_dir():
                raise FileNotFoundError(f"Required local VLM is missing: {path}")
            return path
    raise ValueError(f"No local VLM mapping configured for {config.framework.qwenvl.base_vlm!r}")


def load_hive_policy(checkpoint: Path):
    """Load HIVE VLMVisionGR00T checkpoints with local VLM-path remapping."""
    from hivebench.model.framework import build_framework as build_hive_framework
    from hivebench.model.framework.share_tools import dict_to_namespace as hive_dict_to_namespace
    from hivebench.model.framework.share_tools import read_mode_config as read_hive_mode_config

    model_config, norm_stats = read_hive_mode_config(str(checkpoint))
    config = hive_dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None
    if config.framework.name != "VLMVisionGR00T":
        raise ValueError(f"Unsupported HIVE framework: {config.framework.name!r}")
    config.framework.qwenvl.base_vlm = str(local_vlm_path(config))
    model = build_hive_framework(config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(migrate_checkpoint_keys(model, state_dict), strict=True)
    model.norm_stats = norm_stats
    return model


def load_policy(checkpoint: Path):
    """Build the configured framework and load legacy/current key layouts."""
    if is_hive_checkpoint(checkpoint):
        return load_hive_policy(checkpoint)
    model_config, norm_stats = read_mode_config(str(checkpoint))
    config = dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None
    needs_voltron = config.framework.vision_model == "voltron"
    with local_voltron_dependencies() if needs_voltron else nullcontext():
        model = build_framework(config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(migrate_checkpoint_keys(model, state_dict), strict=True)
    model.norm_stats = norm_stats
    return model


def dataset_path(root: Path, task: str, dataset_name: str) -> Path:
    path = root / task / dataset_name / "data" / "episode0.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"Missing static-texture episode: {path}")
    return path


def verify_static_scene(root: Path, task: str, dataset_name: str, expected_frames: int | None) -> dict:
    info_path = root / task / dataset_name / "scene_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing scene metadata: {info_path}")
    metadata = json.loads(info_path.read_text(encoding="utf-8"))
    frames = metadata.get("frames", [])
    if expected_frames is not None and len(frames) < expected_frames:
        raise ValueError(f"{task}: requested {expected_frames} frames, found only {len(frames)}")
    if not frames:
        raise ValueError(f"{task}: no frames in {info_path}")
    first_pose = frames[0].get("object_poses", {})
    if any(frame.get("object_poses", {}) != first_pose for frame in frames):
        raise ValueError(f"{task}: task-object pose changes across static frames")
    pairs = {(frame.get("wall_texture"), frame.get("table_texture")) for frame in frames}
    if len(pairs) != len(frames):
        raise ValueError(f"{task}: texture pairs are not unique ({len(pairs)}/{len(frames)})")
    return metadata


def decode_rgb(encoded: np.ndarray | bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(bytes(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG frame")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def iter_task_batches(
    path: Path,
    batch_size: int,
    max_frames: int | None,
    camera_keys: tuple[str, str, str],
) -> Iterator[tuple[list[list[np.ndarray]], np.ndarray]]:
    with h5py.File(path, "r") as file:
        state = file["joint_action/vector"]
        total_frames = len(state) if max_frames is None else min(len(state), max_frames)
        camera_datasets = [file[f"observation/{camera}/rgb"] for camera in camera_keys]
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            raw_states = np.asarray(state[start:end], dtype=np.float32)
            images = []
            for index in range(start, end):
                images.append([decode_rgb(camera[index]) for camera in camera_datasets])
            yield images, raw_states


class RobotWinCheckpointNormalizer:
    """Exact state/action normalization for the RoboTwin training convention.

    The saved `dataset_statistics.json` has flat 14-D arrays ordered as
    [left joints(6), right joints(6), left gripper(1), right gripper(1)].
    Joint values use min-max normalization to [-1, 1], while grippers use the
    training-time binary threshold (`value > 0.5`).
    """

    state_order = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_order = [
        "action.left_joints",
        "action.right_joints",
        "action.left_gripper",
        "action.right_gripper",
    ]

    def __init__(self, checkpoint: Path):
        run_dir = checkpoint.parent.parent
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        action_mode = config["datasets"]["vla_data"].get("action_mode", "abs")
        if action_mode != "abs":
            raise ValueError(f"Only absolute-action checkpoints are supported, got {action_mode!r}")

        statistics = json.loads((run_dir / "dataset_statistics.json").read_text(encoding="utf-8"))
        if len(statistics) != 1:
            raise ValueError(f"Expected one embodiment statistics entry, got {list(statistics)}")
        self.statistics_key = next(iter(statistics))
        stats = statistics[self.statistics_key]
        self.state_min = self._vector(stats["state"], "min")
        self.state_max = self._vector(stats["state"], "max")
        self.action_min = self._vector(stats["action"], "min")
        self.action_max = self._vector(stats["action"], "max")

    @staticmethod
    def _vector(stats: dict, name: str) -> np.ndarray:
        value = np.asarray(stats[name], dtype=np.float32)
        if value.shape != (14,):
            raise ValueError(f"Expected a 14-D {name} vector, got {value.shape}")
        return value

    @staticmethod
    def _min_max_apply(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
        output = np.zeros_like(values, dtype=np.float32)
        valid = minimum != maximum
        output[..., valid] = 2 * (values[..., valid] - minimum[valid]) / (maximum[valid] - minimum[valid]) - 1
        return output

    def apply_state(self, raw_state: np.ndarray) -> np.ndarray:
        if raw_state.shape[-1] != 14:
            raise ValueError(f"Expected raw RoboTwin qpos shape (..., 14), got {raw_state.shape}")
        # HDF5 order: left joints, left gripper, right joints, right gripper.
        ordered = np.concatenate(
            [raw_state[..., :6], raw_state[..., 7:13], raw_state[..., 6:7], raw_state[..., 13:14]],
            axis=-1,
        ).astype(np.float32, copy=False)
        normalized = self._min_max_apply(ordered[..., :12], self.state_min[:12], self.state_max[:12])
        grippers = (ordered[..., 12:] > 0.5).astype(np.float32)
        return np.concatenate([normalized, grippers], axis=-1)

    def unapply_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        if normalized_actions.shape[-1] != 14:
            raise ValueError(f"Expected normalized actions shape (..., 14), got {normalized_actions.shape}")
        output = np.empty_like(normalized_actions, dtype=np.float32)
        output[..., :12] = (
            (normalized_actions[..., :12] + 1) / 2 * (self.action_max[:12] - self.action_min[:12])
            + self.action_min[:12]
        )
        output[..., 12:] = (normalized_actions[..., 12:] > 0.5).astype(np.float32)
        return output


@contextmanager
def fixed_diffusion_noise(noise: torch.Tensor):
    """Temporarily make the DiT action head use identical initial noise per item.

    GR00T_ActionHeader.predict_action calls ``torch.randn`` exactly once to
    initialise an action chunk. The model is in eval mode, so replacing that
    call is sufficient to remove diffusion sampling as a source of variance
    while retaining batched visual inference.
    """
    original_randn = torch.randn

    def deterministic_randn(*args, **kwargs):
        size = kwargs.get("size", args[0] if args else None)
        if size is not None:
            size = tuple(size)
            if len(size) == 3 and tuple(size[1:]) == tuple(noise.shape[1:]):
                dtype = kwargs.get("dtype", noise.dtype)
                device = kwargs.get("device", noise.device)
                return noise.to(device=device, dtype=dtype).expand(size[0], -1, -1).clone()
        return original_randn(*args, **kwargs)

    torch.randn = deterministic_randn
    try:
        yield
    finally:
        torch.randn = original_randn


def action_statistics(actions: np.ndarray) -> dict:
    variance = actions.var(axis=0)
    mean = actions.mean(axis=0)
    return {
        "mean": mean.tolist(),
        "variance": variance.tolist(),
        "scalar_variance": float(variance.mean()),
        "scalar_std": float(np.sqrt(variance.mean())),
        "variance_by_horizon": variance.mean(axis=1).tolist(),
        "variance_by_action_dim": variance.mean(axis=0).tolist(),
    }


def total_metrics(actions: np.ndarray, texture_variance: float) -> dict:
    """Return the two encoder-level statistics and a labelled diagnostic.

    ``texture_variance`` is centred within each task before averaging.  In
    contrast, the pooled variance below also includes intentional action
    differences between distinct tasks and is retained only for reference.
    """
    return {
        "total_action_mean": float(actions.mean()),
        "total_texture_variance": float(texture_variance),
        "pooled_action_variance_including_task_difference": float(actions.var()),
    }


def macro_task_metrics(task_summaries: dict, space: str = "normalized") -> tuple[float, float]:
    """Average one scalar mean and one scalar variance across the 12 tasks."""
    task_means = [float(np.asarray(item[space]["mean"], dtype=np.float32).mean()) for item in task_summaries.values()]
    task_variances = [float(item[space]["scalar_variance"]) for item in task_summaries.values()]
    return float(np.mean(task_means)), float(np.mean(task_variances))


def add_total_metrics(summary: dict) -> dict:
    """Backfill encoder totals for a completed model summary without inference."""
    normalized_mean, normalized_variance = macro_task_metrics(summary["tasks"], "normalized")
    unnormalized_mean, unnormalized_variance = macro_task_metrics(summary["tasks"], "unnormalized")
    summary["total_metrics"] = {
        "normalized": {
            "total_mean": normalized_mean,
            "total_variance": normalized_variance,
        },
        "unnormalized": {
            "total_mean": unnormalized_mean,
            "total_variance": unnormalized_variance,
        },
    }
    return summary


def unnormalize_batch(actions: np.ndarray, normalizer: RobotWinCheckpointNormalizer) -> np.ndarray:
    return normalizer.unapply_actions(actions)


def checkpoint_stat_hash(checkpoint: Path) -> str:
    return hashlib.sha256((checkpoint.parent.parent / "dataset_statistics.json").read_bytes()).hexdigest()


def evaluate_model(
    checkpoint: Path,
    tasks: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    model_name = checkpoint.parent.parent.name
    print(f"\n[{model_name}] loading {checkpoint}", flush=True)
    model = load_policy(checkpoint).to(args.device).eval()
    normalizer = RobotWinCheckpointNormalizer(checkpoint)
    camera_keys = HIVE_VLM_CAMERA_KEYS if is_hive_checkpoint(checkpoint) else HIVEBENCH_CAMERA_KEYS

    action_horizon = int(model.action_model.action_horizon)
    action_dim = int(model.config.framework.action_model.action_dim)
    generator = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    initial_noise = torch.randn((1, action_horizon, action_dim), generator=generator, dtype=torch.float32)

    normalized_by_task = []
    raw_by_task = []
    task_summaries = {}
    rows = []
    with torch.inference_mode():
        for task in tasks:
            hdf5_path = dataset_path(args.dataset_root, task, args.dataset_name)
            verify_static_scene(args.dataset_root, task, args.dataset_name, args.max_frames)
            normalized_chunks = []
            for images, raw_states in iter_task_batches(
                hdf5_path, args.batch_size, args.max_frames, camera_keys
            ):
                normalized_states = normalizer.apply_state(raw_states)
                examples = [
                    # Training samples carry one observation step: [1, 14].
                    # The framework batches this to [B, 1, 14] before its
                    # state encoder, so retain that singleton time dimension.
                    {"image": image, "lang": TASK_PROMPTS[task], "state": state[None, :]}
                    for image, state in zip(images, normalized_states)
                ]
                with fixed_diffusion_noise(initial_noise):
                    output = model.predict_action(examples=examples)
                normalized_chunks.append(np.asarray(output["normalized_actions"], dtype=np.float32))

            normalized_actions = np.concatenate(normalized_chunks, axis=0)
            raw_actions = unnormalize_batch(normalized_actions, normalizer).astype(np.float32)
            normalized_by_task.append(normalized_actions)
            raw_by_task.append(raw_actions)
            normalized_stats = action_statistics(normalized_actions)
            raw_stats = action_statistics(raw_actions)
            task_summaries[task] = {
                "prompt": TASK_PROMPTS[task],
                "num_frames": int(normalized_actions.shape[0]),
                "normalized": normalized_stats,
                "unnormalized": raw_stats,
            }
            rows.append({
                "model": model_name,
                "task": task,
                "num_frames": int(normalized_actions.shape[0]),
                "normalized_scalar_variance": normalized_stats["scalar_variance"],
                "normalized_scalar_std": normalized_stats["scalar_std"],
                "unnormalized_scalar_variance": raw_stats["scalar_variance"],
                "unnormalized_scalar_std": raw_stats["scalar_std"],
            })
            print(
                f"  {task}: normalized variance={normalized_stats['scalar_variance']:.7g}",
                flush=True,
            )

    normalized_array = np.stack(normalized_by_task, axis=0)
    raw_array = np.stack(raw_by_task, axis=0)
    macro_normalized = float(np.mean([summary["normalized"]["scalar_variance"] for summary in task_summaries.values()]))
    macro_raw = float(np.mean([summary["unnormalized"]["scalar_variance"] for summary in task_summaries.values()]))
    summary = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "checkpoint_statistics_sha256": checkpoint_stat_hash(checkpoint),
        "excluded_models": sorted(EXCLUDED_CHECKPOINTS),
        "device": args.device,
        "diffusion_noise_seed": args.noise_seed,
        "diffusion_noise_policy": "identical fixed initial action noise for every task/frame",
        "camera_order": list(camera_keys),
        "state_input_order": RobotWinCheckpointNormalizer.state_order,
        "statistics_key": normalizer.statistics_key,
        "action_shape": list(normalized_array.shape[2:]),
        "tasks": task_summaries,
        "macro_texture_variance": {
            "normalized": macro_normalized,
            "unnormalized": macro_raw,
        },
        "pooled_descriptive_statistics": {
            "normalized": action_statistics(normalized_array.reshape(-1, *normalized_array.shape[2:])),
            "unnormalized": action_statistics(raw_array.reshape(-1, *raw_array.shape[2:])),
            "warning": "Pooled variance includes expected action differences between tasks; use macro_texture_variance for robustness ranking.",
        },
    }
    normalized_mean, normalized_variance = macro_task_metrics(task_summaries, "normalized")
    raw_mean, raw_variance = macro_task_metrics(task_summaries, "unnormalized")
    summary["total_metrics"] = {
        "normalized": {"total_mean": normalized_mean, "total_variance": normalized_variance},
        "unnormalized": {"total_mean": raw_mean, "total_variance": raw_variance},
    }
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "actions.npz",
        tasks=np.asarray(tasks),
        normalized_actions=normalized_array,
        unnormalized_actions=raw_array,
    )
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del model, normalizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_summary_row(summary: dict) -> dict:
    totals = summary["total_metrics"]["normalized"]
    return {
        "model": summary["model"],
        "total_mean": totals["total_mean"],
        "total_variance": totals["total_variance"],
    }


def task_rows_from_summary(summary: dict) -> list[dict]:
    rows = []
    for task, task_summary in summary["tasks"].items():
        normalized = task_summary["normalized"]
        raw = task_summary["unnormalized"]
        rows.append({
            "model": summary["model"],
            "task": task,
            "num_frames": task_summary["num_frames"],
            "normalized_scalar_variance": normalized["scalar_variance"],
            "normalized_scalar_std": normalized["scalar_std"],
            "unnormalized_scalar_variance": raw["scalar_variance"],
            "unnormalized_scalar_std": raw["scalar_std"],
        })
    return rows


def load_existing_summary(output_dir: Path, model_name: str) -> dict:
    summary_path = output_dir / model_name / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    add_total_metrics(summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_aggregate_outputs(
    output_dir: Path,
    model_summaries: list[dict],
    all_rows: list[dict],
    failures: dict[str, str],
) -> None:
    model_summaries.sort(key=lambda item: item["total_variance"])
    write_csv(all_rows, output_dir / "per_task_metrics.csv")
    write_csv(model_summaries, output_dir / "encoder_total_metrics.csv")
    write_csv(
        [{"model": row["model"], "total_mean": row["total_mean"]} for row in model_summaries],
        output_dir / "encoder_total_mean.csv",
    )
    # Retain the original filename as a convenient ranking table, now with the
    # explicitly named two totals for every encoder.
    write_csv(model_summaries, output_dir / "model_ranking.csv")
    (output_dir / "summary.json").write_text(
        json.dumps({"models": model_summaries, "failures": failures}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.offline:
        # Must be set before any vision backbone calls `from_pretrained`.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tasks = args.tasks or list(TASK_PROMPTS)
    unknown_tasks = sorted(set(tasks) - set(TASK_PROMPTS))
    if unknown_tasks:
        raise ValueError(f"No fixed prompt defined for: {unknown_tasks}")
    checkpoints = checkpoint_paths(args.checkpoint_root, args.models)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tasks": tasks,
        "checkpoint_count": len(checkpoints),
        "checkpoints": [str(path) for path in checkpoints],
        "excluded_checkpoints": sorted(EXCLUDED_CHECKPOINTS),
        "dataset_root": str(args.dataset_root),
        "dataset_name": args.dataset_name,
        "noise_seed": args.noise_seed,
        "total_metric_definition": "For each task, average the scalar action mean/variance over its 100 texture frames, then average those 12 task values. Root metrics use normalized actions.",
        "optional_backbone_roots": [str(path) for path in OPTIONAL_BACKBONE_ROOTS if path.is_dir()],
        "prompts": {task: TASK_PROMPTS[task] for task in tasks},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.summarize_only:
        previous_summary_path = args.output_dir / "summary.json"
        previous_failures = (
            json.loads(previous_summary_path.read_text(encoding="utf-8")).get("failures", {})
            if previous_summary_path.is_file()
            else {}
        )
        completed_summaries = []
        completed_rows = []
        for checkpoint in checkpoints:
            model_name = checkpoint.parent.parent.name
            summary_path = args.output_dir / model_name / "summary.json"
            if not summary_path.is_file():
                continue
            summary = load_existing_summary(args.output_dir, model_name)
            completed_summaries.append(model_summary_row(summary))
            completed_rows.extend(task_rows_from_summary(summary))
        write_aggregate_outputs(args.output_dir, completed_summaries, completed_rows, previous_failures)
        print(f"Summarized {len(completed_summaries)} completed models. Results: {args.output_dir}")
        return

    all_rows = []
    model_summaries = []
    previous_root_summary = args.output_dir / "summary.json"
    previous = (
        json.loads(previous_root_summary.read_text(encoding="utf-8"))
        if previous_root_summary.is_file()
        else {}
    )
    requested_names = {checkpoint.parent.parent.name for checkpoint in checkpoints}
    # A partial rerun (for example, only the six failed encoders) keeps the
    # successful models from the previous run in the root aggregates.
    if args.models is not None:
        for model_name in sorted(path.name for path in args.output_dir.iterdir() if path.is_dir()):
            if model_name in requested_names:
                continue
            summary_path = args.output_dir / model_name / "summary.json"
            if not summary_path.is_file():
                continue
            preserved = load_existing_summary(args.output_dir, model_name)
            model_summaries.append(model_summary_row(preserved))
            all_rows.extend(task_rows_from_summary(preserved))
    failures = {
        name: error
        for name, error in previous.get("failures", {}).items()
        if name not in requested_names
    }
    for checkpoint in checkpoints:
        model_name = checkpoint.parent.parent.name
        if args.skip_existing and (args.output_dir / model_name / "summary.json").is_file():
            print(f"[{model_name}] skipping existing result", flush=True)
            summary = load_existing_summary(args.output_dir, model_name)
            model_summaries.append(model_summary_row(summary))
            all_rows.extend(task_rows_from_summary(summary))
            continue
        try:
            summary, rows = evaluate_model(checkpoint, tasks, args, args.output_dir)
            model_summaries.append(model_summary_row(summary))
            all_rows.extend(rows)
        except Exception as error:
            failures[model_name] = repr(error)
            print(f"[{model_name}] FAILED: {error}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise

    write_aggregate_outputs(args.output_dir, model_summaries, all_rows, failures)
    print(f"\nCompleted {len(model_summaries)}/{len(checkpoints)} models. Results: {args.output_dir}")
    if failures:
        print(f"Failed models: {', '.join(failures)}", file=sys.stderr)


if __name__ == "__main__":
    main()

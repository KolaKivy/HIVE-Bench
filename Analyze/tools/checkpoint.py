"""Checkpoint loading for fine-tuned visual encoders."""

import torch


POLICY_VISION_PREFIX = "vision_encoder."


def _strip_vision_prefixes(key: str) -> str:
    """Drop wrapper prefixes (policy prefix, DDP/compile wrappers) from a checkpoint key."""
    for prefix in (POLICY_VISION_PREFIX, "module.", "_orig_mod."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _map_ckpt_key_to_target(key: str, target_keys) -> str | None:
    """Map a checkpoint key onto a target parameter via unique dotted-suffix match."""
    if key in target_keys:
        return key
    parts = key.split(".")
    for size in range(len(parts) - 1, 0, -1):
        tail = ".".join(parts[-size:])
        candidates = [
            candidate
            for candidate in target_keys
            if candidate == tail or candidate.endswith("." + tail)
        ]
        if len(candidates) == 1:
            return candidates[0]
    return None


def load_vision_checkpoint(model, ckpt_dir: str, device) -> None:
    """Restore ``vision_encoder.*`` weights from a policy checkpoint into ``model.model``."""
    payload = torch.load(ckpt_dir, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a state dict in {ckpt_dir}, got {type(payload).__name__}"
        )

    vision_items = {
        key: value
        for key, value in payload.items()
        if key.startswith(POLICY_VISION_PREFIX)
    }
    if not vision_items:
        # Tolerate encoder-only state dicts.
        vision_items = {k: v for k, v in payload.items() if torch.is_tensor(v)}
    if not vision_items:
        raise RuntimeError(f"No tensors found in checkpoint: {ckpt_dir}")

    target_module = model.model if hasattr(model, "model") else model
    target = target_module.state_dict()

    restored: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []
    shape_mismatch: list[tuple[str, tuple, tuple]] = []
    for key, value in vision_items.items():
        mapped = _map_ckpt_key_to_target(_strip_vision_prefixes(key), target)
        if mapped is None:
            unmatched.append(key)
        elif tuple(value.shape) != tuple(target[mapped].shape):
            shape_mismatch.append((key, tuple(value.shape), tuple(target[mapped].shape)))
        else:
            restored[mapped] = value

    missing, unexpected = target_module.load_state_dict(restored, strict=False)

    print(f"Loaded checkpoint from: {ckpt_dir}")
    print(
        f"  vision_encoder tensors: {len(vision_items)}, restored: {len(restored)}, "
        f"unmatched: {len(unmatched)}, shape mismatch: {len(shape_mismatch)}"
    )
    print(
        f"  target params not covered by the checkpoint: {len(missing)} "
        "(frozen/untrained modules such as prediction heads are expected here)"
    )
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]} (total {len(unexpected)})")
    if unmatched:
        print(f"  Unmatched checkpoint keys (first 5): {unmatched[:5]}")
    if shape_mismatch:
        print(f"  Shape mismatches (first 5): {shape_mismatch[:5]}")

    if not restored:
        raise RuntimeError(
            f"No checkpoint tensors could be mapped onto {type(target_module).__name__}; "
            f"check that the encoder architecture in {ckpt_dir} matches the configured model."
        )

    # Guard against a silent no-op: verify a sample of restored tensors on device.
    applied = target_module.state_dict()
    not_applied = [
        key
        for key in list(restored)[:5]
        if not torch.equal(applied[key].detach().cpu(), restored[key].detach().cpu())
    ]
    if not_applied:
        raise RuntimeError(
            f"Checkpoint restore verification failed for: {not_applied[:3]}"
        )
    print(f"  verified {len(list(restored)[:5])} restored tensors match the checkpoint")

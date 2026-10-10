# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""VGGT-Omega (facebook/VGGT-Omega) encoder wrapper for the analysis framework.

RoboTwin policy checkpoints whose ``vision_model`` is ``vggt_omega`` store the
aggregator of VGGT-Omega, not the original VGGT-1B.  For example
``all_task_robotwin_vggt_og_ft/final_model/pytorch_model.pt`` contains
``vision_encoder.aggregator.inter_frame_blocks.*`` plus ``attn.qkv.bias_mask``
and ``patch_embed.storage_tokens``/``patch_embed.rope_embed.periods``.  The
original VGGT aggregator (``facebook/VGGT-1B-Commercial``) instead uses
``global_blocks`` and has none of those tensors, so those weights can never be
restored into ``vision_models.vgg_transformer.VGGTModel``.

This module mirrors the policy-side ``VGGTOmegaBackbone``
(``Policy/hivebench/model/modules/vision_model/hf_encoder.py``) so that analysis
features match the features seen during fine-tuning:

    frames -> resize 224 -> aggregator (S=1) -> final patch tokens [B, 196, 2048]

The calling convention matches ``vision_models.vgg_transformer.VGGTModel``:
``forward(list[np.ndarray | PIL.Image | Tensor]) -> Tensor[B, C, H, W]``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DEFAULT_REPO_PATH = "third_party/VGGT_Omega"
DEFAULT_WEIGHT_FILE = "vggt_omega_1b_512.pt"


def _resolve_weight_path(
    model_name: str,
    weight_file: str,
    weight_path: str | None = None,
) -> Path:
    """Locate the VGGT-Omega weights: explicit path, then HF cache, then download."""
    if weight_path:
        path = Path(weight_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"weight_path does not exist: {path}")
        return path

    hub_cache = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub"
    snapshots = sorted(
        hub_cache.glob(
            f"models--{model_name.replace('/', '--')}/snapshots/*/{weight_file}"
        )
    )
    if snapshots:
        return snapshots[-1]

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=model_name, filename=weight_file))


class VGGTOmegaModel(torch.nn.Module):
    """VGGT-Omega aggregator backbone exposing patch tokens as (B, C, H, W).

    Args:
        model_name: HF repo holding the released VGGT-Omega weights.
        weight_file: Weight file name inside that repo.
        weight_path: Optional explicit path to the weight file (overrides the HF repo).
        repo_path: Path to the ``vggt-omega`` checkout providing the ``vggt_omega`` package.
        image_size: Square input resolution (224 -> 14x14 = 196 patch tokens).
        token_mode: ``patch`` for patch tokens, ``register`` for register tokens.
    """

    def __init__(
        self,
        model_name: str = "facebook/VGGT-Omega",
        weight_file: str = DEFAULT_WEIGHT_FILE,
        weight_path: str | None = None,
        repo_path: str | None = None,
        image_size: int = 224,
        token_mode: str = "patch",
        device: str | torch.device = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.token_mode = token_mode
        self.image_size = int(image_size)
        self.num_channels = 0

        repo_path = repo_path or os.environ.get("VGGT_OMEGA_REPO", DEFAULT_REPO_PATH)
        if not Path(repo_path).is_dir():
            raise FileNotFoundError(
                f"VGGT-Omega repo not found at {repo_path}. Clone "
                "facebookresearch/vggt-omega there, or pass repo_path / set VGGT_OMEGA_REPO."
            )
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from vggt_omega.models.vggt_omega import VGGTOmega

        ckpt_path = _resolve_weight_path(model_name, weight_file, weight_path)
        print(f"Loading VGGT-Omega weights: {ckpt_path}")
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        except (TypeError, RuntimeError):
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        full_model = VGGTOmega()
        incompatible = full_model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            print(
                "Warning: VGGT-Omega checkpoint keys absent from the model: "
                f"{len(incompatible.unexpected_keys)}"
            )
        if incompatible.missing_keys:
            # Prediction heads are unused by the analysis probes.
            print(
                "Note: VGGT-Omega parameters not present in the checkpoint: "
                f"{len(incompatible.missing_keys)}"
            )

        self.model = full_model.to(self.device).eval()
        # VGGT-Omega normalises internally, so [0, 1] pixel values are expected here.
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        with torch.no_grad():
            probe = self(torch.zeros(1, 3, self.image_size, self.image_size, device=self.device))
        print(
            f"VGGT-Omega encoder ready: patch tokens {tuple(probe.shape)} (B, C, H, W), "
            f"num_channels={self.num_channels}"
        )

    def _to_pixel_values(self, images) -> torch.Tensor:
        tensors = []
        for image in images:
            if isinstance(image, torch.Tensor):
                tensors.append(image)
                continue
            if isinstance(image, np.ndarray):
                image = Image.fromarray(np.asarray(image, dtype=np.uint8))
            tensors.append(self.transform(image))
        return torch.stack(tensors).to(self.device)

    def forward(self, images, requires_grad: bool = False) -> torch.Tensor:
        pixel_values = self._to_pixel_values(images)
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            # Single-frame aggregation, exactly like the policy-side VGGTOmegaBackbone.
            tokens_list, patch_start_idx = self.model.aggregator(pixel_values.unsqueeze(1))

        tokens = tokens_list[-1][:, 0]  # [B, S=1, N_total, D] -> [B, N_total, D]
        if self.token_mode == "register":
            tokens = tokens[:, 1:patch_start_idx]
        else:
            tokens = tokens[:, patch_start_idx:]

        batch, num_tokens, channels = tokens.shape
        side = int(round(num_tokens**0.5))
        if side * side != num_tokens:
            raise ValueError(
                f"Expected a square patch grid, got {num_tokens} tokens for a "
                f"{self.image_size}px input."
            )
        self.num_channels = int(channels)
        # fp32 for the numpy-based analysis pipeline (bf16 has no numpy dtype).
        return tokens.float().transpose(1, 2).reshape(batch, channels, side, side)


def print_feature_size(
    model_name: str = "facebook/VGGT-Omega",
    weight_file: str = DEFAULT_WEIGHT_FILE,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = VGGTOmegaModel(model_name=model_name, weight_file=weight_file, device=device)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    visual_tokens = extractor.forward([image])

    print("--- VGGT-Omega Feature Info ---")
    print(f"Model Name:    {model_name}")
    print(f"Visual Tokens: {visual_tokens.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()

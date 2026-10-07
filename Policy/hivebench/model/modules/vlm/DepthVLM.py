# Copyright 2026 HIVE-Bench community. All rights reserved.
# Licensed under the MIT License.

"""DepthVLM adapter for the VLMVisionGR00T feature interface.

DepthVLM uses a Qwen3-VL-4B backbone plus a DPT depth head. Its official model
implementation targets Transformers 5.2, while HIVE-Bench currently runs 4.57.
This adapter retains the local Qwen3 implementation and loads the official DPT
head weights separately, avoiding a global dependency upgrade.
"""

from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from .QWen3 import _QWen3_VL_Interface
from .depthvlm.dpt_depth_head import DPTDepthHead


class _DepthVLM_Interface(_QWen3_VL_Interface):
    """Load the published DepthVLM checkpoint for frozen VLM encoding.

    ``build_qwenvl_inputs`` is inherited from Qwen3-VL. The action framework
    consumes normal Qwen hidden states, while ``depth_head`` retains the loaded
    official depth parameters for explicit depth-token fusion in later work.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        torch.nn.Module.__init__(self)
        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm")
        if not model_id:
            raise ValueError("framework.qwenvl.base_vlm is required for DepthVLM")

        model_config = AutoConfig.from_pretrained(model_id)
        text_config = model_config.text_config
        if getattr(text_config, "rope_scaling", None) is None:
            text_config.rope_scaling = getattr(text_config, "rope_parameters", None)
        if text_config.rope_scaling is None:
            raise ValueError("DepthVLM config is missing Qwen3 RoPE parameters")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            config=model_config,
            attn_implementation=qwenvl_config.get("attn_implementation", "flash_attention_2"),
            dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, extra_special_tokens={})
        self.processor.tokenizer.padding_side = "left"
        self.config = config
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        self.depth_head = DPTDepthHead(
            dim_in=self.model.config.text_config.hidden_size,
            features=256,
            out_channels=[256, 512, 1024, 1024],
        )
        self._load_depth_head(Path(model_id))

    def _load_depth_head(self, model_path: Path) -> None:
        checkpoint = model_path / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "DepthVLM requires a local checkpoint containing model.safetensors; "
                f"not found at {checkpoint}"
            )

        prefix = "depth_head."
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            keys = [key for key in weights.keys() if key.startswith(prefix)]
            if not keys:
                raise RuntimeError(f"No {prefix} weights found in {checkpoint}")
            state_dict = {key[len(prefix):]: weights.get_tensor(key) for key in keys}

        incompatible = self.depth_head.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "DepthVLM depth-head checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def forward(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

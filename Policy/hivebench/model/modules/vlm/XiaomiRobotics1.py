"""Encoder-only adapter for Xiaomi Robotics-1 (XR-1).

XR-1's published policy includes a state/action-conditioned DiT. HIVE uses
only its Qwen3-VL visual-language encoder and keeps HIVE's own action head, so
this adapter intentionally does not import XR-1's state/action interface.
"""

import importlib.util
from pathlib import Path
from typing import List, Optional

import torch
from accelerate import init_empty_weights
import transformers
from transformers import AutoConfig, AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration

from .vision_encoder import VisualTokenEncoder


IMAGE_TOKEN_ID = 151655


class _XiaomiRobotics1_Interface(VisualTokenEncoder):
    image_token_id = IMAGE_TOKEN_ID

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = config.framework.qwenvl.get("base_vlm")
        if not model_id:
            raise ValueError("framework.qwenvl.base_vlm is required for Xiaomi Robotics-1.")

        checkpoint = Path(model_id)
        if checkpoint.is_dir() and (checkpoint / "model_states.pt").is_file():
            self.model = self._load_raw_policy_checkpoint(checkpoint, qwenvl_config)
            processor_id = self._backbone_id(checkpoint, qwenvl_config)
        else:
            self.model = self._load_hf_encoder(model_id, qwenvl_config)
            processor_id = model_id

        self.processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        self.config = config
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self._warned_final_layer_only = False

    def _backbone_id(self, checkpoint: Path, qwenvl_config):
        configured = qwenvl_config.get("backbone_vlm")
        if configured:
            return configured
        sibling = checkpoint.parent / "Qwen3-VL-4B-Instruct"
        return str(sibling) if sibling.is_dir() else "Qwen/Qwen3-VL-4B-Instruct"

    def _load_raw_policy_checkpoint(self, checkpoint: Path, qwenvl_config):
        backbone_id = self._backbone_id(checkpoint, qwenvl_config)
        backbone_path = Path(backbone_id)
        if backbone_path.is_dir() and not (backbone_path / "config.json").is_file():
            raise ValueError(f"XR-1 backbone snapshot is missing config.json: {backbone_path}")
        model_config = AutoConfig.from_pretrained(backbone_id)
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration._from_config(
                model_config,
                attn_implementation=qwenvl_config.get("attn_implementation", "eager"),
                dtype=torch.bfloat16,
            )
        raw = torch.load(checkpoint / "model_states.pt", map_location="cpu", mmap=True, weights_only=False)["module"]
        vlm_state = {
            key[len("model.vlm."):]: value
            for key, value in raw.items()
            if key.startswith("model.vlm.")
            and key not in {"model.vlm.model.action_embed.weight", "model.vlm.model.score_embed.weight"}
        }
        incompatible = model.load_state_dict(vlm_state, strict=False, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "XR-1 VLM checkpoint does not match the local Qwen3-VL backbone: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        return model

    def _load_hf_encoder(self, model_id, qwenvl_config):
        if transformers.__version__ != "4.57.1":
            raise RuntimeError("Xiaomi Robotics-1 requires transformers==4.57.1; found " f"{transformers.__version__}.")
        if importlib.util.find_spec("liger_kernel") is None:
            raise RuntimeError("Xiaomi Robotics-1 HF remote code requires liger-kernel==0.6.5.")
        policy = AutoModel.from_pretrained(
            model_id, trust_remote_code=True,
            attn_implementation=qwenvl_config.get("attn_implementation", "flash_attention_2"),
            dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
        if not hasattr(policy, "vlm"):
            raise RuntimeError("The Xiaomi Robotics-1 checkpoint did not expose the expected `vlm` module.")
        model = policy.vlm
        del policy
        return model

    def forward(self, **kwargs):
        # XR-1's custom Qwen implementation returns its final sequence hidden
        # state directly in ``hidden_states`` and supports skip_logits.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs, use_cache=False)

    def _select_hidden_state(self, outputs, layer_idx):
        if isinstance(outputs.hidden_states, (tuple, list)):
            return super()._select_hidden_state(outputs, layer_idx)
        if layer_idx not in (-1, None) and not self._warned_final_layer_only:
            print(
                "[XiaomiRobotics1] The official XR-1 VLM exposes only final hidden states; "
                f"using the final layer instead of requested layer {layer_idx}."
            )
            self._warned_final_layer_only = True
        return super()._select_hidden_state(outputs, -1)

    def build_qwenvl_inputs(self, images: List[List], instructions: List[str], **kwargs):
        if len(images) != len(instructions):
            raise ValueError("Images and instructions must have the same batch size.")

        messages = []
        for sample_images, instruction in zip(images, instructions):
            prompt = instruction
            if (
                hasattr(self.config, "datasets")
                and hasattr(self.config.datasets, "vla_data")
                and "CoT_prompt" in self.config.datasets.vla_data
            ):
                prompt = self.config.datasets.vla_data.get("CoT_prompt", "").replace("{instruction}", instruction)
            content = [{"type": "image", "image": image} for image in sample_images]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

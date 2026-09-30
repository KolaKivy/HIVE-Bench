# OpenVLA.py — OpenVLA (Prismatic LLaMA-2-7B + DINOv2+SigLIP) interface for HIVE-Bench






# hidden_size = 4096 (LLaMA-2-7B)
# N_PATCH = 256 (DINOv2+SigLIP fused @ 224px)

import torch
import torch.nn as nn

from .vision_encoder import VisualTokenEncoder
from typing import Optional
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn.utils.rnn import pad_sequence

from accelerate.logging import get_logger
logger = get_logger(__name__)

N_PATCH = 256   


class _OpenVLA_Interface(VisualTokenEncoder):
    vision_token_count = N_PATCH
    """
    OpenVLA Prismatic wrapper。
    Interface compatible with _QWen3_VL_Interface:
        - build_qwenvl_inputs(images, instructions) → dict
        - forward(**kwargs)                         → CausalLMOutputWithPast
        - model.config.hidden_size                  → int (4096)

    Visual token extraction uses last_hidden[:, 1:1+N_PATCH, :].
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get(
            "base_vlm",
            "./playground/Pretrained_models/HIVE-Bench/openvla-7b"
        )

        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="eager",   
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self.model = model
        self.processor = processor
        self.config = config

        
        self.model.config.hidden_size = self.model.language_model.config.hidden_size  # 4096

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        
        if "pixel_values" in kwargs and kwargs["pixel_values"] is not None:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)
        return outputs

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        """
        Args:
            images:       List[List[PIL.Image]]  — [B, num_views]
            instructions: List[str]              — [B]
        Returns:
            dict on model device (input_ids, pixel_values, attention_mask)
        """
        assert len(images) == len(instructions)

        all_inputs = []
        for imgs, instruction in zip(images, instructions):
            
            img = imgs[0] if isinstance(imgs, (list, tuple)) else imgs

            if (hasattr(self.config, "datasets") and
                hasattr(self.config.datasets, "vla_data") and
                "CoT_prompt" in self.config.datasets.vla_data):
                prompt = self.config.datasets.vla_data.get(
                    "CoT_prompt", ""
                ).replace("{instruction}", instruction)
            else:
                prompt = instruction

            inp = self.processor(prompt, img, return_tensors="pt")
            all_inputs.append(inp)

        
        pad_id = self.processor.tokenizer.pad_token_id or 0
        input_ids = pad_sequence(
            [x["input_ids"].squeeze(0) for x in all_inputs],
            batch_first=True, padding_value=pad_id,
        )
        pixel_values = torch.cat([x["pixel_values"] for x in all_inputs], dim=0)
        attention_mask = (input_ids != pad_id).long()

        return {
            "input_ids":      input_ids.to(self.model.device),
            "pixel_values":   pixel_values.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
        }
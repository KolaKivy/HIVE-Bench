# Gemma3.py — Gemma3 VLM interface for HIVE-Bench
# Implemented following QWen3.py conventions.
# Gemma3 image token: <image_soft_token> = 262144

import torch
import torch.nn as nn

from .vision_encoder import VisualTokenEncoder
from typing import Optional
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 262144        # <image_soft_token>
IMAGE_START_TOKEN_INDEX = 255999  # <start_of_image>
IMAGE_END_TOKEN_INDEX = 256000    # <end_of_image>


class _Gemma3_VL_Interface(VisualTokenEncoder):
    image_token_id = IMAGE_TOKEN_INDEX
    """
    Lightweight wrapper around Gemma3ForConditionalGeneration.
    Unified interface matching _QWen3_VL_Interface:
        - build_qwenvl_inputs(images, instructions) → BatchFeature
        - forward(**kwargs)                          → CausalLMOutputWithPast
        - model.config.hidden_size                  → int
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get(
            "base_vlm",
            "./playground/Pretrained_models/Gemma3-4B-Instruct"
        )

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        
        self.model.config.hidden_size = self.model.config.text_config.hidden_size  # 2560

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)
        return outputs

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        """
        Build model inputs from images + instructions.
        Follows Gemma3 instruct format (same as apply_chat_template).

        Args:
            images:       List[List[PIL.Image]]  — [B, num_views]
            instructions: List[str]              — [B]

        Returns:
            BatchFeature on model device
        """
        assert len(images) == len(instructions)

        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if hasattr(self.config, "datasets") and hasattr(
                self.config.datasets, "vla_data"
            ) and "CoT_prompt" in self.config.datasets.vla_data:
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return batch_inputs.to(self.model.device)
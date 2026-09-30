# SmolVLM.py — SmolVLM VLM interface for HIVE-Bench
# image token: <image> = 49190

import torch
import torch.nn as nn

from .vision_encoder import VisualTokenEncoder
from typing import Optional
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.modeling_outputs import CausalLMOutputWithPast

from accelerate.logging import get_logger
logger = get_logger(__name__)

IMAGE_TOKEN_INDEX = 49190   # <image>


class _SmolVLM_Interface(VisualTokenEncoder):
    image_token_id = IMAGE_TOKEN_INDEX
    """
    Lightweight wrapper around SmolVLM.
    Interface compatible with _QWen3_VL_Interface:
        - build_qwenvl_inputs(images, instructions) → BatchFeature
        - forward(**kwargs)                         → CausalLMOutputWithPast
        - model.config.hidden_size                  → int (960)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get(
            "base_vlm",
            "./playground/Pretrained_models/SmolVLM-500M-Instruct"
        )

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        
        self.model.config.hidden_size = self.model.config.text_config.hidden_size  # 960

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)
        return outputs

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        """
        Args:
            images:       List[List[PIL.Image]]  — [B, num_views]
            instructions: List[str]              — [B]
        Returns:
            BatchFeature on model device
        """
        assert len(images) == len(instructions)

        texts = []
        all_images = []
        for imgs, instruction in zip(images, instructions):
            
            content = [{"type": "image"} for _ in imgs]

            if (hasattr(self.config, "datasets") and
                hasattr(self.config.datasets, "vla_data") and
                "CoT_prompt" in self.config.datasets.vla_data):
                prompt = self.config.datasets.vla_data.get(
                    "CoT_prompt", ""
                ).replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            texts.append(
                self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            )
            all_images.append(imgs)

        batch_inputs = self.processor(
            images=all_images,
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.model.device)
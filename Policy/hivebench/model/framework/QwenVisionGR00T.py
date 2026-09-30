# QwenVisionGR00T.py — VLM vision token + CLIP text token → DiT action head








#   img + text → QwenVL.forward() → last_hidden_state

#   text → CLIP text encoder → text_tokens [B, 77, 512]
#   vision_tokens → vision_proj [B, N_vis, cross_dim]
#   text_tokens   → text_proj   [B, 77,    cross_dim]
#   concat → encoder_hidden [B, N_vis+77, cross_dim] → DiT cross-attn → action

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from hivebench.model.framework.base_framework import baseframework
from hivebench.model.modules.vlm import get_vlm_model
from hivebench.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    get_action_model,
)
from hivebench.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("QwenVisionGR00T")
class QwenVisionGR00T(baseframework):
    'QwenVisionGR00T implementation.'

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        fusion_cfg = getattr(config.framework, "vision_text_fusion", None)
        cross_dim  = int(getattr(fusion_cfg, "cross_attention_dim", 1024)) if fusion_cfg else 1024
        freeze_vlm = bool(getattr(fusion_cfg, "freeze_vlm", True))         if fusion_cfg else True
        freeze_clip= bool(getattr(fusion_cfg, "freeze_clip", True))         if fusion_cfg else True

        # Build the VLM visual encoder.
        self.vlm = get_vlm_model(config=config)
        vlm_hidden = self.vlm.model.config.hidden_size
        vlm_name   = config.framework.qwenvl.base_vlm
        self.vlm_layer_idx = int(getattr(fusion_cfg, "vlm_layer_idx", -1)) if fusion_cfg else -1
        print(f"[QwenVisionGR00T] vlm_layer_idx={self.vlm_layer_idx}")

        print(f"[QwenVisionGR00T] VLM={vlm_name}")
        print(f"[QwenVisionGR00T] vlm_hidden={vlm_hidden}")

        if freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False

        # Build the CLIP text encoder.
        from transformers import CLIPTextModel, CLIPTokenizer
        clip_id = "openai/clip-vit-base-patch16"
        self.clip_text_model = CLIPTextModel.from_pretrained(clip_id)
        self.clip_tokenizer  = CLIPTokenizer.from_pretrained(clip_id)
        clip_dim = self.clip_text_model.config.hidden_size   # 512
        print(f"[QwenVisionGR00T] CLIP text hidden={clip_dim}")

        if freeze_clip:
            for param in self.clip_text_model.parameters():
                param.requires_grad = False

        
        self.vision_proj = nn.Linear(vlm_hidden, cross_dim)
        self.text_proj   = nn.Linear(clip_dim,   cross_dim)
        print(f"[QwenVisionGR00T] cross_attention_dim={cross_dim}")

        
        config.framework.action_model.hidden_size = cross_dim
        if hasattr(config.framework.action_model, "diffusion_model_cfg"):
            config.framework.action_model.diffusion_model_cfg.cross_attention_dim = cross_dim

        # Build the DiT action head.
        self.action_model: FlowmatchingActionHead = get_action_model(config=config)
        self.future_action_window_size = (
            config.framework.action_model.future_action_window_size
        )

    def _encode(self, batch_images: List[List], instructions: List[str]) -> torch.Tensor:
        """
        VLM vision tokens + CLIP text tokens → encoder_hidden [B, N_vis+77, cross_dim]
        """
        device = next(self.parameters()).device

        vision_tokens = self.vlm.encode_visual_tokens(
            images=batch_images,
            instructions=instructions,
            layer_idx=self.vlm_layer_idx,
        )
        vision_tokens = self.vision_proj(vision_tokens.float())   # [B, N_vis, cross_dim]

        # Encode instructions with CLIP.
        clip_inputs = self.clip_tokenizer(
            instructions, padding="max_length",
            max_length=77, truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            clip_out = self.clip_text_model(**clip_inputs, return_dict=True)
        text_tokens = self.text_proj(clip_out.last_hidden_state.float())  # [B, 77, cross_dim]

        return torch.cat([vision_tokens, text_tokens], dim=1)  # [B, N_vis+77, cross_dim]

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict:
        batch_images = [ex["image"]  for ex in examples]
        instructions = [ex["lang"]   for ex in examples]
        actions      = [ex["action"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        encoder_hidden = self._encode(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=encoder_hidden.device, dtype=encoder_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]
            repeated = getattr(self.config.framework.action_model, "repeated_diffusion_steps", 4)
            actions_rep = actions_target.repeat(repeated, 1, 1)
            hidden_rep  = encoder_hidden.repeat(repeated, 1, 1)

            state_rep = None
            if state is not None:
                state_t = torch.tensor(
                    np.array(state), device=encoder_hidden.device, dtype=encoder_hidden.dtype
                )
                state_rep = state_t.repeat(repeated, 1, 1)

            action_loss = self.action_model(hidden_rep, actions_rep, state_rep)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> Dict:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [ex["image"]  for ex in examples]
        instructions = [ex["lang"]   for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        encoder_hidden = self._encode(batch_images, instructions)
        state_tensor = (
            torch.from_numpy(np.array(state)).to(
                encoder_hidden.device, dtype=encoder_hidden.dtype
            ) if state is not None else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(encoder_hidden, state_tensor)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
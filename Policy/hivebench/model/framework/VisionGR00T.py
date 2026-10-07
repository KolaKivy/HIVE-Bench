# VisionGR00T.py — Pure-vision GR00T framework with pluggable vision encoders




#     dinov2 / dinov2_{small,base,large,giant} → DINOv2 (384/768/1024/1536-dim)

#     clip        → CLIP ViT-B/16 (768-dim, 196 tokens)
#     siglip      → SigLIP ViT-B/16 (768-dim, 196 tokens)
#     siglip2     → SigLIP2 ViT-B/16 (768-dim, 196 tokens)
#     mae         → MAE ViT-B (768-dim, 196 tokens)
#     vit         → ViT-B/16 (768-dim, 196 tokens)
#     radio       → RADIOv2.5-B (768-dim, 196 tokens)
#     cradio / cradio_{h,so400m} → C-RADIOv4 (1280/1152-dim)

#     theia       → Theia-base (768-dim, 196 tokens)

#     mocov3      → MoCoV3 ViT-B (768-dim, 196 tokens)
#     eva02       → EVA-02 ViT-B (768-dim, 256 tokens)

#     vggt        → VGGT-1B (2048-dim, 1369 tokens)
#     vjepa2 / vjepa2_{base,large,giant} → V-JEPA2.1 (768/1024/1408-dim, 384px)
#     vjepa2_1_{base,large,giant} (also vjepa2.1_{base,large,giant}) → same
#     levjepa → LeVJEPA-VideoMix-Large (1024-dim, copies one image to 16 frames; returns final 196 tokens)
#     mcr         → MCR ResNet-50 (768-dim after proj, 49 tokens)
#     da3         → DA3-BASE (768-dim, 256 tokens)


#   prefix-based:

#     vc1 / vc1_base / vc1_large → VC-1 (768/1024-dim, 196 tokens)




#   NOT SUPPORTED:







import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List

from hivebench.model.framework.base_framework import baseframework
from hivebench.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    get_action_model,
)
from hivebench.model.tools import FRAMEWORK_REGISTRY


def _build_vision_encoder(vision_model_name: str) -> nn.Module:
    ' build vision encoder function.'
    from hivebench.model.modules.vison_model import hf_encoder as hfe

    name = vision_model_name.lower()

    
    SHORTHAND = {
        # DINOv2 / DINOv3
        "dinov2":    lambda: hfe.HFViTBackbone("facebook/dinov2-base"),
        "dinov2_small": lambda: hfe.HFViTBackbone("facebook/dinov2-small"),
        "dinov2_base": lambda: hfe.HFViTBackbone("facebook/dinov2-base"),
        "dinov2_large": lambda: hfe.HFViTBackbone("facebook/dinov2-large"),
        "dinov2_giant": lambda: hfe.HFViTBackbone("facebook/dinov2-giant"),
        "dinov3":    lambda: hfe.DINOv3Backbone("facebook/dinov3-vitb16-pretrain-lvd1689m"),
        "dinov3_small": lambda: hfe.DINOv3Backbone("facebook/dinov3-vits16-pretrain-lvd1689m"),
        "dinov3_base": lambda: hfe.DINOv3Backbone("facebook/dinov3-vitb16-pretrain-lvd1689m"),
        "dinov3_large": lambda: hfe.DINOv3Backbone("facebook/dinov3-vitl16-pretrain-lvd1689m"),
        "dinov3_huge": lambda: hfe.DINOv3Backbone("facebook/dinov3-vith16plus-pretrain-lvd1689m"),
        "dinov3_h16plus": lambda: hfe.DINOv3Backbone("facebook/dinov3-vith16plus-pretrain-lvd1689m"),
        "dinov3_vith16plus": lambda: hfe.DINOv3Backbone("facebook/dinov3-vith16plus-pretrain-lvd1689m"),
        # CLIP family
        "clip":      lambda: hfe.CLIPBackbone("openai/clip-vit-base-patch16"),
        "siglip":    lambda: hfe.SigLIPBackbone("google/siglip-base-patch16-224"),
        "siglip2":   lambda: hfe.SigLIP2Backbone("google/siglip2-base-patch16-224"),
        # Standard ViT / MAE
        "mae":       lambda: hfe.MAEBackbone("facebook/vit-mae-base"),
        "vit":       lambda: hfe.HFViTBackbone("google/vit-base-patch16-224"),
        "sam3": lambda: hfe.SAM3Backbone("facebook/sam3"),
        # RADIO family (all via torch.hub to avoid HF weight key bugs)
        "radio":     lambda: hfe.RADIOTorchHubBackbone("radio_v2.5-b"),
        "cradio":    lambda: hfe.RADIOTorchHubBackbone("c-radio_v4-h"),
        "cradio_h": lambda: hfe.RADIOTorchHubBackbone("c-radio_v4-h"),
        "cradio_so400m": lambda: hfe.RADIOTorchHubBackbone("c-radio_v4-so400m"),
        "cradiov4_h": lambda: hfe.RADIOTorchHubBackbone("c-radio_v4-h"),
        "cradiov4_so400m": lambda: hfe.RADIOTorchHubBackbone("c-radio_v4-so400m"),
        "cradio_v4_h": lambda: hfe.RADIOTorchHubBackbone("c-radio_v4-h"),
        "cradio_v4_so400m": lambda: hfe.RADIOTorchHubBackbone("c-radio_v4-so400m"),
        "amradio":   lambda: hfe.RADIOTorchHubBackbone("radio_v2.5-b"),
        # Multi-teacher distillation
        "theia":     lambda: hfe.TheiaBackbone("theaiinstitute/theia-base-patch16-224-cdiv"),
        # General vision
        "internvit": lambda: hfe.InternViTBackbone("OpenGVLab/InternViT-300M-448px-V2_5"),
        "mocov3":    lambda: hfe.MoCoV3Backbone("nyu-visionx/moco-v3-vit-b"),
        "eva":  lambda: hfe.EVA01Backbone(),
        "eva02":     lambda: hfe.EVA02Backbone("eva02_base_patch14_224"),
        # Robot-specific
        "voltron":   lambda: hfe.VoltronBackbone("v-cond-base"),
        "mcr":       lambda: hfe.MCRBackbone("GqJiang/robots-pretrain-robots"),
        "lingbot_small": lambda: hfe.LingBotVisionBackbone("robbyant/lingbot-vision-vit-small"),
        "lingbot_base": lambda: hfe.LingBotVisionBackbone("robbyant/lingbot-vision-vit-base"),
        "lingbot_large": lambda: hfe.LingBotVisionBackbone("robbyant/lingbot-vision-vit-large"),
        "lingbot_giant": lambda: hfe.LingBotVisionBackbone("robbyant/lingbot-vision-vit-giant"),
        # 3D / geometry
        "vggt":      lambda: hfe.VGGTBackbone("facebook/VGGT-1B"),
        "vggt_omega": lambda: hfe.VGGTOmegaBackbone(),
        "vggt_omega_register": lambda: hfe.VGGTOmegaBackbone(token_mode="register"),
        "da3":       lambda: hfe.DA3Backbone("depth-anything/DA3-BASE"),
        # Video backbone (single-frame mode)
        "vjepa2":    lambda: hfe.VJEPA2Backbone("base"),
        "vjepa2_base": lambda: hfe.VJEPA2Backbone("base"),
        "vjepa2_large": lambda: hfe.VJEPA2Backbone("large"),
        "vjepa2_giant": lambda: hfe.VJEPA2Backbone("giant"),
        "vjepa2_1_base": lambda: hfe.VJEPA2Backbone("base"),
        "vjepa2_1_large": lambda: hfe.VJEPA2Backbone("large"),
        "vjepa2_1_giant": lambda: hfe.VJEPA2Backbone("giant"),
        # Accept both underscore and official V-JEPA2.1 spellings.
        "vjepa2.1_base": lambda: hfe.VJEPA2Backbone("base"),
        "vjepa2.1_large": lambda: hfe.VJEPA2Backbone("large"),
        "vjepa2.1_giant": lambda: hfe.VJEPA2Backbone("giant"),
        "levjepa": lambda: hfe.LeVJEPABackbone(),
        "levjepa_videomix_large": lambda: hfe.LeVJEPABackbone(),
    }

    if name in SHORTHAND:
        return SHORTHAND[name]()

    if "lingbot-vision" in name or "lingbot_vision" in name:
        return hfe.LingBotVisionBackbone(vision_model_name)

    # Accept the official V-JEPA2.1 torch.hub function names directly.
    vjepa_hub_names = {
        "vjepa2_1_vit_base_384": "base",
        "vjepa2_1_vit_large_384": "large",
        "vjepa2_1_vit_giant_384": "giant",
    }
    if name in vjepa_hub_names:
        return hfe.VJEPA2Backbone(vjepa_hub_names[name])

    # Resolve prefix-based encoder variants.
    if vision_model_name.startswith("distill_theia_"):
        ckpt_path = vision_model_name[len("distill_theia_"):]
        return hfe.DistillTheiaBackbone(ckpt_path)

    if name.startswith("spa_"):
        return hfe.SPABackbone(name.split("_", 1)[1])

    if name in {"vc1", "vc1_base"}:
        return hfe.VC1Backbone("base")
    if name == "vc1_large":
        return hfe.VC1Backbone("large")

    # Resolve torch.hub DINOv2 identifiers.
    if "/" not in vision_model_name:
        from hivebench.model.modules.vison_model.dino import get_dino_model
        return get_dino_model(vision_model_name)

    if name == "galilai-group/levjepa-videomix-large":
        return hfe.LeVJEPABackbone(vision_model_name)

    # DINOv3 official ids need the register-aware wrapper even when supplied
    # directly instead of through a shorthand.
    if name == "facebook/dinov3-vith16plus-pretrain-lvd1689m":
        return hfe.DINOv3Backbone(vision_model_name)
    if name.startswith("facebook/dinov3-"):
        return hfe.DINOv3Backbone(vision_model_name)

    
    return hfe.HFViTBackbone(vision_model_name)


@FRAMEWORK_REGISTRY.register("VisionGR00T")
class VisionGR00T(baseframework):
    'VisionGR00T implementation.'

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        vision_model_name = getattr(config.framework, "vision_model", "dinov2")
        self.vision_model_name = vision_model_name

        
        self.vision_encoder = _build_vision_encoder(vision_model_name)
        self.vision_dim = self.vision_encoder.num_channels
        print(f"[VisionGR00T] vision_encoder={vision_model_name}, vision_dim={self.vision_dim}")

        
        for param in self.vision_encoder.parameters():
            param.requires_grad = True

        
        config.framework.action_model.hidden_size = self.vision_dim
        if hasattr(config.framework.action_model, "diffusion_model_cfg"):
            config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.vision_dim

        
        self.action_model: FlowmatchingActionHead = get_action_model(config=config)
        self.future_action_window_size = config.framework.action_model.future_action_window_size

    def _encode_images(self, batch_images: List[List]) -> torch.Tensor:
        ' encode images function.'
        
        if len(batch_images) > 0 and not isinstance(batch_images[0], (list, tuple)):
            batch_images = [[img] for img in batch_images]
        B = len(batch_images)
        num_view = len(batch_images[0])
        pixel_values = self.vision_encoder.prepare_dino_input(batch_images)
        patch_tokens = self.vision_encoder(pixel_values)   # [B*views, N, D]
        _, N, D = patch_tokens.shape
        return patch_tokens.reshape(B, num_view, N, D).reshape(B, num_view * N, D)

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict:
        batch_images = [ex["image"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        last_hidden = self._encode_images(batch_images)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            repeated = getattr(self.config.framework.action_model, "repeated_diffusion_steps", 4)
            actions_rep = actions_target.repeat(repeated, 1, 1)
            hidden_rep = last_hidden.repeat(repeated, 1, 1)

            state_rep = None
            if state is not None:
                state_t = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_rep = state_t.repeat(repeated, 1, 1)

            action_loss = self.action_model(hidden_rep, actions_rep, state_rep)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> Dict:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [ex["image"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        last_hidden = self._encode_images(batch_images)

        state_tensor = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state_tensor)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

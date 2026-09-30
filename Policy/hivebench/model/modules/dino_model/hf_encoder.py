# hf_encoder.py — Vision encoder wrappers for DinoGR00T
# All backbones expose the same interface:
#   .num_channels                  int
#   .forward(pixel_values)         Tensor[B, N, D]   patch tokens, CLS stripped
#   .prepare_dino_input(img_list)  Tensor             preprocess List[List[PIL/ndarray]]

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .lingbot import LingBotVisionBackbone


# Resolve vendored research encoders relative to this file so users can run
# HIVE-Bench from any working directory without exporting PYTHONPATH manually.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_THIRD_PARTY_ROOT = _REPO_ROOT / "third_party"
_THIRD_PARTY_PATHS = {
    "spa": _THIRD_PARTY_ROOT / "SPA",
    "vggt_omega": _THIRD_PARTY_ROOT / "VGGT_Omega",
    "vjepa2": _THIRD_PARTY_ROOT / "VJEPA2",
}
_bundled_python_paths = [
    str(path) for path in _THIRD_PARTY_PATHS.values() if path.is_dir()
]
for _third_party_path in reversed(_bundled_python_paths):
    if _third_party_path not in sys.path:
        sys.path.insert(0, _third_party_path)
_existing_python_paths = [
    path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path
]
os.environ["PYTHONPATH"] = os.pathsep.join(
    dict.fromkeys(_bundled_python_paths + _existing_python_paths)
)


# Image preprocessing helpers.
def _to_pil(img):
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8))
    return img


def _imagenet_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _hf_prepare(img_list, processor, device):
    """Generic HuggingFace processor prepare."""
    flat = [_to_pil(img) for views in img_list for img in views]
    inputs = processor(images=flat, return_tensors="pt")
    return inputs["pixel_values"].to(device)


def _transform_prepare(img_list, transform, device):
    """torchvision transform prepare."""
    flat = [_to_pil(img) for views in img_list for img in views]
    return torch.stack([transform(img) for img in flat]).to(device)


# GROUP 1: Standard HuggingFace AutoModel (same interface, different weights)
# Covers: DINOv2-HF, DINOv3, ViT, MoCoV3(HF variant), and any generic HF ViT
class HFViTBackbone(nn.Module):
    """
    Generic HuggingFace AutoModel wrapper.
    Works for: facebook/dinov2-base, google/vit-base-patch16-224,
               facebook/dinov3-vitb16-pretrain-lvd1689m, etc.
    Output: last_hidden_state[:, 1:, :]  →  [B, N, D]  (CLS stripped)
    """
    def __init__(self, model_name: str):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.num_channels = self.model.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values, return_dict=True)
        h = out.last_hidden_state          # [B, N+1, D] (with CLS)
        # DINOv3 may have register tokens; strip everything before patch tokens
        return h[:, 1:, :]                # strip CLS → [B, N, D]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _hf_prepare(img_list, self.processor, device)

class DINOv3Backbone(nn.Module):
    'DINOv3Backbone implementation.'
    def __init__(self, model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.num_channels = self.model.config.hidden_size   # 768
        self.num_register_tokens = getattr(self.model.config, 'num_register_tokens', 0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values, return_dict=True)
        h = out.last_hidden_state   # [B, 1+num_reg+N, D]
        # strip CLS + register tokens → pure patch tokens
        return h[:, 1 + self.num_register_tokens:, :]   # [B, N, 768]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _hf_prepare(img_list, self.processor, device)
# GROUP 2: CLIP-family (CLIP, SigLIP, SigLIP-V2)
class CLIPBackbone(nn.Module):
    """
    CLIP vision encoder only (openai/clip-vit-base-patch16).
    Output: last_hidden_state[:, 1:, :]  →  [B, 196, 768]
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        super().__init__()
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.num_channels = self.model.config.hidden_size   # 768

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values, return_dict=True)
        return out.last_hidden_state[:, 1:, :]   # strip CLS

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _hf_prepare(img_list, self.processor, device)


class SigLIPBackbone(nn.Module):
    """
    SigLIP vision encoder (google/siglip-base-patch16-224).
    SigLIP has NO CLS token → all tokens are patch tokens.
    Output: last_hidden_state  →  [B, 196, 768]
    """
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        super().__init__()
        from transformers import SiglipVisionModel, SiglipImageProcessor
        self.model = SiglipVisionModel.from_pretrained(model_name)
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        self.num_channels = self.model.config.hidden_size   # 768

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values, return_dict=True)
        return out.last_hidden_state   # no CLS, all patch tokens

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _hf_prepare(img_list, self.processor, device)


class SigLIP2Backbone(nn.Module):
    """
    SigLIP-V2 vision encoder (google/siglip2-base-patch16-224).
    Output: last_hidden_state  →  [B, 196, 768]
    """
    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        full_model = AutoModel.from_pretrained(model_name)
        self.model = full_model.vision_model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.num_channels = self.model.config.hidden_size   # 768

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values, return_dict=True)
        return out.last_hidden_state   # no CLS

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _hf_prepare(img_list, self.processor, device)


# GROUP 3: MAE (masked autoencoder, must disable masking)
class MAEBackbone(nn.Module):
    """
    MAE ViT-B (facebook/vit-mae-base).
    CRITICAL: mask_ratio must be 0.0 to get all 196 tokens.
    Default mask_ratio=0.75 only returns 25% of tokens.
    Output: last_hidden_state[:, 1:, :]  →  [B, 196, 768]
    """
    def __init__(self, model_name: str = "facebook/vit-mae-base"):
        super().__init__()
        from transformers import ViTMAEModel, AutoImageProcessor
        self.model = ViTMAEModel.from_pretrained(model_name)
        self.model.config.mask_ratio = 0.0   # disable masking at inference
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.num_channels = self.model.config.hidden_size   # 768

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values, noise=None, return_dict=True)
        return out.last_hidden_state[:, 1:, :]   # strip CLS → [B, 196, 768]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _hf_prepare(img_list, self.processor, device)


# GROUP 4: NVIDIA RADIO family (RADIOv2.5-B, C-RADIOv4-B, AM-RADIO)
# All return (summary, spatial_features); we use spatial_features.
class RADIOBackbone(nn.Module):
    def __init__(self, model_name: str = "nvidia/RADIO-B"):
        super().__init__()
        
        if model_name == "nvidia/RADIO":
            self.model = torch.hub.load(
                "NVlabs/RADIO", "radio_model",
                version="radio_v2.5-b",
                progress=True, skip_validation=True
            )
            self.processor = None
            self._use_torchhub = True
            self.num_channels = 768
        else:
            from transformers import AutoModel, AutoImageProcessor
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._use_torchhub = False
            self.num_channels = 768

    def forward(self, pixel_values):
        if self._use_torchhub:
            summary, spatial = self.model(pixel_values)
            return spatial
        out = self.model(pixel_values)
        if isinstance(out, (tuple, list)):
            return out[1]
        return out.last_hidden_state[:, 1:, :]

    def prepare_dino_input(self, img_list):
        device = next(self.parameters()).device
        if self._use_torchhub:
            return _transform_prepare(img_list, _imagenet_transform(224), device)
        return _hf_prepare(img_list, self.processor, device)

class RADIOTorchHubBackbone(nn.Module):
    'RADIOTorchHubBackbone implementation.'
    def __init__(self, version: str = "radio_v2.5-b"):
        super().__init__()
        self.model = torch.hub.load(
            "NVlabs/RADIO", "radio_model",
            version=version,
            progress=True,
            skip_validation=True,
        )
        _RADIO_DIMS = {
            "radio_v2.5-b":  768,
            "radio_v2.5-l":  1024,
            "c-radio_v3-b":  768,
            "c-radio_v4-h":  1280,
            "c-radio_v4-so400m": 1152,
        }
        self.num_channels = _RADIO_DIMS.get(version, 768)
        self.transform = _imagenet_transform(224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        summary, spatial = self.model(pixel_values)
        return spatial   # [B, 256, 768]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)
# GROUP 5: Theia (multi-teacher distillation, robot-specific)
class TheiaBackbone(nn.Module):
    'TheiaBackbone implementation.'
    def __init__(self, model_name: str = "theaiinstitute/theia-base-patch16-224-cdiv"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.num_channels = 768

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, H, W, 3] uint8
        feat = self.model.forward_feature(pixel_values)
        if feat.shape[1] == 197:
            feat = feat[:, 1:, :]   # strip CLS
        return feat   # [B, 196, 768]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        'Prepare dino input function.'
        flat = [_to_pil(img).resize((224, 224)) for views in img_list for img in views]
        device = next(self.parameters()).device
        
        arrays = [np.array(img, dtype=np.uint8) for img in flat]
        tensors = torch.from_numpy(np.stack(arrays))   # [B, H, W, 3]
        return tensors.to(device)

# GROUP 6: InternViT-300M (OpenGVLab, 448px input)
# Resizes input to 448×448 → 1024 tokens. Larger than others.
class InternViTBackbone(nn.Module):
    'InternViTBackbone implementation.'
    def __init__(self, model_name: str = "OpenGVLab/InternViT-300M-448px-V2_5"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True
        )
        self.num_channels = 1024
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values.to(torch.bfloat16))
        h = out.last_hidden_state        # [B, 257, 1024]
        return h[:, 1:, :].float()       # strip CLS → [B, 256, 1024]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)

# GROUP 7: MoCoV3 (contrastive self-supervised, timm-based)
class MoCoV3Backbone(nn.Module):
    'MoCoV3Backbone implementation.'
    def __init__(self, model_name: str = "nyu-visionx/moco-v3-vit-b"):
        super().__init__()
        import timm
        from huggingface_hub import hf_hub_download

        
        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=False,
            num_classes=0, global_pool=""
        )
        
        ckpt_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.num_channels = 768
        self.transform = _imagenet_transform(224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(pixel_values)  # [B, 197, 768]
        if tokens.shape[1] == 197:
            tokens = tokens[:, 1:, :]  # strip CLS → [B, 196, 768]
        return tokens

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            tokens = self.model.forward_features(pixel_values)
        else:
            out = self.model(pixel_values=pixel_values, return_dict=True)
            tokens = out.last_hidden_state
        if tokens.shape[1] == 197:
            tokens = tokens[:, 1:, :]   # strip CLS
        return tokens   # [B, 196, 768]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)


# GROUP 8: EVA / EVA-02 (BAAI, custom weight loading)
# EVA-02-B: 86M params, patch14→16, 224px → 196 tokens, hidden=768
class EVA01Backbone(nn.Module):
    'EVA01Backbone implementation.'
    def __init__(self, model_id: str = "QuanSun/EVA-CLIP",
                 weight_file: str = "EVA01_g_psz14.pt"):
        super().__init__()
        import timm
        from huggingface_hub import hf_hub_download

        
        ckpt_path = hf_hub_download(repo_id=model_id, filename=weight_file)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("module", ckpt))

        
        self.model = timm.create_model(
            "eva_giant_patch14_224",
            pretrained=False,
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
            pretrained_strict=False,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.num_channels = self.model.embed_dim   # 1408
        self.transform = _imagenet_transform(224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(pixel_values)  # [B, 257, 1408]
        if tokens.shape[1] > 256:
            tokens = tokens[:, 1:, :]   # strip CLS → [B, 256, 1408]
        return tokens

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)
class EVA02Backbone(nn.Module):
    """
    EVA-02 ViT-B (QuanSun/EVA-CLIP, EVA02_B_psz14to16).
    Custom architecture with SwiGLU + 2D RoPE.
    Loads via the EVA-02 pip package.
    Requires: pip install git+https://github.com/baaivision/EVA.git
    Output: [B, 196, 768]
    """
    def __init__(self, variant: str = "eva02_base_patch14_224"):
        super().__init__()
        try:
            import timm
            # EVA-02 is available in timm >= 0.9
            self.model = timm.create_model(
                variant, pretrained=True, num_classes=0, global_pool="",pretrained_strict=False,
            )
            self.num_channels = self.model.embed_dim   # 768
        except Exception as e:
            raise RuntimeError(
                f"Failed to load EVA-02 via timm ({e}). "
                "Install timm>=0.9: pip install timm --upgrade"
            )
        self.transform = _imagenet_transform(224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(pixel_values)   # [B, N+1, D]
        if tokens.shape[1] > 196:
            tokens = tokens[:, 1:, :]   # strip CLS
        return tokens   # [B, 196, 768]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)


# GROUP 9: SPA (3D-aware spatial, ICLR 2025)
class SPABackbone(nn.Module):
    """
    SPA: Spatial Pre-training for 3D-aware robot manipulation (ICLR 2025).
    The bundled SPA checkout is discovered automatically from third_party/SPA.
    variant: "base" (768) | "large" (1024)
    Output: [B, 196, D]
    """
    def __init__(self, variant: str = "base"):
        super().__init__()
        if variant == "base":
            from spa.models import spa_vit_base_patch16
            self.model = spa_vit_base_patch16(pretrained=True)
            self.num_channels = 768
        elif variant == "large":
            from spa.models import spa_vit_large_patch16
            self.model = spa_vit_large_patch16(pretrained=True)
            self.num_channels = 1024
        else:
            raise ValueError(f"Unknown SPA variant: {variant}")
        self.transform = _imagenet_transform(224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self.model(pixel_values, feature_map=True, cat_cls=False)
        B, D, H, W = feat.shape
        return feat.flatten(2).permute(0, 2, 1)   # [B, H*W, D]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)


# GROUP 10: VC-1 (facebook, MAE on Ego4D)
class VC1Backbone(nn.Module):
    """
    VC-1: Visual Cortex pre-trained on Ego4D (facebook/vc1-base).
    variant: "base" (768) | "large" (1024)
    Output: [B, 196, D]
    """
    def __init__(self, variant: str = "base"):
        super().__init__()
        from huggingface_hub import hf_hub_download
        import timm
        model_id = f"facebook/vc1-{variant}"
        ckpt_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        arch = "vit_base_patch16_224" if variant == "base" else "vit_large_patch16_224"
        self.model = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        self.num_channels = 768 if variant == "base" else 1024
        self.transform = _imagenet_transform(224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(pixel_values)
        return tokens[:, 1:, :]   # strip CLS

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)


# GROUP 11: Voltron (language-driven robot representation)
class VoltronBackbone(nn.Module):
    """
    Voltron: Language-Driven Representation Learning for Robotics.
    Requires: pip install voltron-robotics
    variant: "v-cond" (384-dim ViT-S) | "v-cond-base" (768-dim ViT-B)
    Output: [B, 196, D]  (visual mode, no language)
    """
    def __init__(self, variant: str = "v-cond-base"):
        super().__init__()
        try:
            import voltron
            self.model, self.preprocess = voltron.load(variant, device="cpu")
            # ViT-B variants: 768-dim; ViT-S: 384-dim
            self.num_channels = 768 if "base" in variant else 384
        except ImportError:
            raise RuntimeError(
                "Voltron not installed. Run: pip install voltron-robotics"
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Voltron visual mode returns patch embeddings
        lang = None
        embeddings = self.model(pixel_values, lang, mode="visual")
        return embeddings   # [B, N, D]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        flat = [_to_pil(img) for views in img_list for img in views]
        device = next(self.parameters()).device
        
        to_tensor = transforms.ToTensor()
        tensors = torch.stack([self.preprocess(to_tensor(img)) for img in flat])
        return tensors.to(device)


# GROUP 12: timm generic fallback (ir413/mvp and other timm-format models)
class TimmViTBackbone(nn.Module):
    """
    Generic timm ViT loaded via hf_hub: prefix.
    For models stored in timm format on HuggingFace (e.g. ir413/mvp).
    Output: [B, N, D]  (CLS stripped)
    """
    def __init__(self, model_name: str):
        super().__init__()
        import timm
        self.model = timm.create_model(
            f"hf_hub:{model_name}", pretrained=True,
            num_classes=0, global_pool=""
        )
        self.num_channels = self.model.embed_dim
        self.transform = _imagenet_transform(224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(pixel_values)
        return tokens[:, 1:, :]   # strip CLS

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)
# VGGT (Visual Geometry Grounded Transformer, Meta 2025)
class VGGTBackbone(nn.Module):
    'VGGTBackbone implementation.'
    def __init__(self, model_id: str = "facebook/VGGT-1B"):
        super().__init__()
        from vggt.models.vggt import VGGT
        full_model = VGGT.from_pretrained(model_id)
        self.aggregator = full_model.aggregator
        self.num_channels = 2048   
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),   
        ])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 3, 518, 518]
        B = pixel_values.shape[0]
        images = pixel_values.unsqueeze(1)   
        tokens_list, patch_start_idx = self.aggregator(images)
        # tokens_list[-1]: [B, S, N_total, D] = [B, 1, 1374, 2048]
        tokens = tokens_list[-1][:, 0, :, :]   
        patch_tokens = tokens[:, patch_start_idx:, :]   # [B, 1369, 2048]
        return patch_tokens

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        flat = [_to_pil(img) for views in img_list for img in views]
        device = next(self.parameters()).device
        return torch.stack([self.transform(img) for img in flat]).to(device)

class VGGTOmegaBackbone(nn.Module):
    'VGGTOmegaBackbone implementation.'
    def __init__(self,
                 model_id: str = "facebook/VGGT-Omega",
                 weight_file: str = "vggt_omega_1b_512.pt",
                 repo_path: str | Path | None = None,
                 token_mode: str = "patch"
                 ):
        super().__init__()
        repo_path = Path(repo_path) if repo_path else _THIRD_PARTY_PATHS["vggt_omega"]
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from vggt_omega.models.vggt_omega import VGGTOmega
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id=model_id, filename=weight_file)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.token_mode = token_mode
        full_model = VGGTOmega()
        full_model.load_state_dict(ckpt, strict=False)
        self.aggregator = full_model.aggregator
        self.num_channels = 2048
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]
        images = pixel_values.unsqueeze(1)   # [B, 1, 3, H, W]
        tokens_list, patch_start_idx = self.aggregator(images)
        tokens = tokens_list[-1][:, 0, :, :]   # [B, N_total, 2048]
        if self.token_mode == "register":
            return tokens[:, 1:patch_start_idx, :]   
        else:
            return tokens[:, patch_start_idx:, :]    

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        flat = [_to_pil(img) for views in img_list for img in views]
        device = next(self.parameters()).device
        return torch.stack([self.transform(img) for img in flat]).to(device)
# LeVJEPA (video-pretrained ViT-L/16, image-mode via official 16-frame repeat)
class LeVJEPABackbone(nn.Module):
    """LeVJEPA VideoMix-Large adapted to the common image-encoder interface.

    The released encoder is trained for 16-frame clips with block-causal
    attention.  Following the model card's image-mode recipe, each still image
    is copied along time and passed in one batched video forward.  We preserve
    only the final temporal slot, yielding the same 14x14=196 patch-token
    budget per view used by the image encoders in this framework.
    """
    def __init__(self, model_name: str = "galilai-group/LeVJEPA-VideoMix-Large"):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.num_channels = self.model.config.embed_dim
        self.num_frames = self.model.config.num_frames
        self.patch_size = self.model.config.patch_size
        self.image_size = self.model.config.img_size
        self.transform = _imagenet_transform(self.image_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Official still-image recipe: repeat [B, C, H, W] into a 16-frame
        # video [B, C, T, H, W], keeping the released block-causal attention.
        video = pixel_values.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
        hidden = self.model(pixel_values=video, return_dict=True).last_hidden_state
        patch_tokens = hidden[:, 1:, :]  # discard CLS: [B, T*H_patches*W_patches, D]
        h_patches = pixel_values.shape[-2] // self.patch_size
        w_patches = pixel_values.shape[-1] // self.patch_size
        tokens_per_frame = h_patches * w_patches
        expected_tokens = self.num_frames * tokens_per_frame
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "Unexpected LeVJEPA token count: "
                f"got {patch_tokens.shape[1]}, expected {expected_tokens}."
            )
        return patch_tokens[:, -tokens_per_frame:, :]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)

# V-JEPA2 (facebook/vjepa2-vitl-fpc64-256, ViT-L backbone only)
class VJEPA2Backbone(nn.Module):
    """V-JEPA2.1 image encoder using the official local hub entry points.

    Variants: base (768), large (1024), giant (1408).  The official 2.1
    checkpoints are trained at 384px and expose image-mode features when the
    input has shape [B, C, 1, H, W].
    """
    _HUB_NAMES = {
        "base": "vjepa2_1_vit_base_384",
        "large": "vjepa2_1_vit_large_384",
        "giant": "vjepa2_1_vit_giant_384",
    }
    _DIMS = {"base": 768, "large": 1024, "giant": 1408}

    def __init__(self, variant: str = "base"):
        super().__init__()
        variant = variant.lower().replace("vjepa2_1_vit_", "")
        if variant not in self._HUB_NAMES:
            raise ValueError(f"Unknown V-JEPA2.1 variant {variant!r}; choose base, large, or giant")
        local_repo = str(_THIRD_PARTY_PATHS["vjepa2"])
        result = torch.hub.load(local_repo, self._HUB_NAMES[variant], source="local", trust_repo=True)
        self.model = result[0] if isinstance(result, tuple) else result
        self.num_channels = self._DIMS[variant]
        self.register_buffer("_dummy", torch.zeros(1))
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        
        x = pixel_values.unsqueeze(2)   # [B, C, H, W] → [B, C, 1, H, W]
        tokens = self.model(x)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        
        return tokens   # [B, N, 768]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = self._dummy.device
        return _transform_prepare(img_list, self.transform, device)

# MCR (Manipulation-Centric Robotic Representation, ICLR 2025)

class MCRBackbone(nn.Module):
    'MCRBackbone implementation.'
    def __init__(self, model_id: str = "GqJiang/robots-pretrain-robots",
                 out_dim: int = 768):
        super().__init__()
        import torchvision.models as tv_models
        from huggingface_hub import hf_hub_download

        
        resnet = tv_models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  

        
        ckpt_path = hf_hub_download(repo_id=model_id, filename="mcr_resnet50.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("fc") and not k.startswith("module.fc")}
        self.backbone.load_state_dict(state_dict, strict=False)

        # projection: 2048 → out_dim
        self.proj = nn.Linear(2048, out_dim)
        self.num_channels = out_dim   # 768
        self.transform = _imagenet_transform(224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(pixel_values)    # [B, 2048, 7, 7]
        B, C, H, W = feat.shape
        feat = feat.flatten(2).permute(0, 2, 1)   # [B, 49, 2048]
        return self.proj(feat)                     # [B, 49, 768]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)
class DA3Backbone(nn.Module):
    'DA3Backbone implementation.'
    def __init__(self, model_id: str = "depth-anything/DA3-BASE"):
        super().__init__()
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        ckpt_path = hf_hub_download(model_id, "model.safetensors")
        state_dict = load_file(ckpt_path)

        
        backbone_sd = {
            k.replace("model.backbone.pretrained.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.backbone.pretrained.")
        }

        
        self.model = timm.create_model(
            "vit_base_patch14_dinov2",
            pretrained=False,
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
        )
        self.model.load_state_dict(backbone_sd, strict=False)
        self.num_channels = 768
        self.transform = _imagenet_transform(224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(pixel_values)  # [B, 257, 768]
        return tokens[:, 1:, :]   # strip CLS → [B, 256, 768]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.parameters()).device
        return _transform_prepare(img_list, self.transform, device)



# Output: [B, 196, 768]
class DistillTheiaBackbone(nn.Module):
    'DistillTheiaBackbone implementation.'
    def __init__(self, ckpt_path: str):
        super().__init__()
        from transformers import ViTModel, ViTConfig

        
        full_ckpt = torch.load(ckpt_path, map_location="cpu")
        backbone_sd = {
            k[len("backbone.model."):]: v
            for k, v in full_ckpt.items()
            if k.startswith("backbone.model.")
        }

        
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            image_size=224,
            patch_size=16,
            num_channels=3,
            qkv_bias=True,
        )
        self.model = ViTModel(config)
        missing, unexpected = self.model.load_state_dict(backbone_sd, strict=False)
        if missing:
            print(f"[DistillTheia] missing keys: {len(missing)}, first: {missing[0]}")
        self.model.eval()
        self.num_channels = 768
        self._transform = _imagenet_transform(224)
        print(f"[DistillTheia] Loaded from {ckpt_path}, num_channels=768, tokens=196")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(
            next(self.model.parameters()).device,
            dtype=next(self.model.parameters()).dtype,
        )
        out = self.model(pixel_values=pixel_values, return_dict=True)
        return out.last_hidden_state[:, 1:, :]   # strip CLS → [B, 196, 768]

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return _transform_prepare(img_list, self._transform, device)
class SAM3Backbone(nn.Module):
    'SAM3Backbone implementation.'
    def __init__(self, model_name: str = "facebook/sam3"):
        super().__init__()
        from transformers import Sam3Model
        full_model = Sam3Model.from_pretrained(model_name)
        self.model = full_model.vision_encoder.backbone
        self.model.to(torch.bfloat16)
        self.num_channels = 1024
        self.transform = transforms.Compose([
            transforms.Resize((1008, 1008)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.to(torch.bfloat16)
        out = self.model(x)
        return out.last_hidden_state.float()

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return _transform_prepare(img_list, self.transform, device)
# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.



#   ViT-L/16, embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4, Mlp FFN,



import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, C, H, W) -> (B, N, D)。"""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # B C H W
        return x.flatten(2).transpose(1, 2)  # B HW C


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RopePositionEmbedding(nn.Module):
    'RopePositionEmbedding implementation.'

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        self.base = base
        self.D_head = embed_dim // num_heads
        self.register_buffer("periods", torch.empty(self.D_head // 4), persistent=True)
        self._init_weights()

    def _init_weights(self):
        periods = self.base ** (2 * torch.arange(self.D_head // 4, dtype=torch.float32) / (self.D_head // 2))
        self.periods.data = periods

    def forward(self, *, H: int, W: int):
        'Forward function.'
        device = self.periods.device
        coords_h = torch.arange(0.5, H, dtype=torch.float32, device=device) / H
        coords_w = torch.arange(0.5, W, dtype=torch.float32, device=device) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        return torch.sin(angles), torch.cos(angles)


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    'LinearKMaskedBias implementation.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, proj_bias: bool = True, mask_k_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, rope):
        'Apply rope function.'
        sin, cos = rope
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        q_prefix, q_patch = q[:, :, :prefix, :], q[:, :, prefix:, :]
        k_prefix, k_patch = k[:, :, :prefix, :], k[:, :, prefix:, :]
        q = torch.cat((q_prefix, rope_apply(q_patch, sin, cos)), dim=-2)
        k = torch.cat((k_prefix, rope_apply(k_patch, sin, cos)), dim=-2)
        return q, k

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        qkv = self.qkv(x)
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, heads, N, head_dim]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return self.proj(x.reshape(B, N, C))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0, layerscale_init: float | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SelfAttention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim, init_values=layerscale_init) if layerscale_init else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * ffn_ratio))
        self.ls2 = LayerScale(dim, init_values=layerscale_init) if layerscale_init else nn.Identity()

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class LingBotVisionTransformer(nn.Module):
    'LingBotVisionTransformer implementation.'

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        layerscale_init: float | None = 1e-5,
        n_storage_tokens: int = 4,
        pos_embed_rope_base: float = 100.0,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.n_storage_tokens = n_storage_tokens

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))
        self.rope_embed = RopePositionEmbedding(embed_dim=embed_dim, num_heads=num_heads, base=pos_embed_rope_base)
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, num_heads, ffn_ratio, layerscale_init) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if isinstance(m, LinearKMaskedBias) and m.bias is not None:
                    o = m.out_features
                    m.bias_mask.fill_(1.0)
                    m.bias_mask[o // 3 : 2 * o // 3].fill_(0.0)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, LayerScale):
                m.reset_parameters()
            elif isinstance(m, PatchEmbed):
                k = 1 / (m.proj.in_channels * (m.patch_size[0] ** 2))
                nn.init.uniform_(m.proj.weight, -math.sqrt(k), math.sqrt(k))
                if m.proj.bias is not None:
                    nn.init.uniform_(m.proj.bias, -math.sqrt(k), math.sqrt(k))

    def prepare_tokens_with_masks(self, x: torch.Tensor):
        'Prepare tokens with masks function.'
        x = self.patch_embed(x)  # B HW C
        B, N, D = x.shape
        cls_token = self.cls_token + 0 * self.mask_token
        x = torch.cat(
            [cls_token.expand(B, -1, -1), self.storage_tokens.expand(B, -1, -1), x], dim=1
        )
        H = W = int(round(N**0.5))
        return x, (H, W)

    def forward_features(self, x: torch.Tensor) -> dict:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        rope = self.rope_embed(H=H, W=W)
        for blk in self.blocks:
            x = blk(x, rope)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_storage_tokens": x_norm[:, 1 : 1 + self.n_storage_tokens],
            "x_norm_patchtokens": x_norm[:, 1 + self.n_storage_tokens :],
        }

    def forward(self, x: torch.Tensor, is_training: bool = False):
        ret = self.forward_features(x)
        if is_training:
            return ret
        return self.head(ret["x_norm_clstoken"])


class LingBot(torch.nn.Module):
    'LingBot implementation.'

    def __init__(
        self,
        model_name: str = "playground/Pretrained_models/lingbot-vision-vit-large",
        device: str | torch.device = "cuda",
        img_size: int = 224,
        **kwargs,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        
        self.model = LingBotVisionTransformer(img_size=img_size)
        ckpt = torch.load(
            f"{model_name}/model.pt", map_location="cpu", weights_only=True
        )
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = self.model.load_state_dict(state, strict=True)
        assert not missing and not unexpected, (
            f"LingBot checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
        self.model = self.model.to(self.device).eval()

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        'Forward function.'
        image_tensors = torch.stack(
            [self.transform(Image.fromarray(img)) for img in images], dim=0
        ).to(self.device)

        if requires_grad:
            out = self.model(image_tensors, is_training=True)
        else:
            with torch.no_grad():
                out = self.model(image_tensors, is_training=True)

        tokens = out["x_norm_patchtokens"]  # (B, H*W, C)
        B, N, C = tokens.size()
        side = int(round(N**0.5))
        return tokens.transpose(1, 2).reshape(B, C, side, side)


def print_feature_size(
    model_name: str = "playground/Pretrained_models/lingbot-vision-vit-large",
) -> None:
    'Print feature size function.'
    import requests

    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000000139.jpg",
        "http://images.cocodataset.org/val2017/000000000285.jpg",
    ]
    images = []
    for url in urls:
        try:
            images.append(np.array(Image.open(requests.get(url, stream=True, timeout=15).raw)))
        except Exception as e:
            print(f"Skip {url}: {e}")
    assert images, "No images downloaded"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lingbot_extractor = LingBot(model_name=model_name, device=device)
    visual_tokens = lingbot_extractor.forward(images)

    print(f"--- LingBot-Vision Feature Info ---")
    print(f"Model Name:      {model_name}")
    print(f"Num Images:      {len(images)}")
    print(f"Visual Tokens:   {visual_tokens.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()

"""
idm_model.py — Inverse Dynamics Model for visual encoder evaluation.

Pipeline:
    frame_t, frame_t+1  (PIL / ndarray)
        → vision_encoder (frozen)           [B, N, D] tokens each
        → ActionExpertHead (Transformer)
            · project tokens to model_dim
            · prepend learnable [DIFF] query token
            · concat token sequences: [DIFF, tokens_t, tokens_t+1]  [B, 1+2N, D]
            · 4-layer Transformer encoder (self-attn)
            · read out [DIFF] token → Linear → action_dim
        → MSE / L1 / per-dim L1 vs GT action

Design rationale (aligned with LARY "Action Expert"):
    - Keep ALL patch tokens (not pooled) so the Transformer can attend to
      spatial differences between frames — crucial for detecting small motions.
    - [DIFF] query token is the bottleneck: it must summarise the transition
      between frame_t and frame_t+1, directly probing encoder quality.
    - Transformer depth=4, heads=4 keeps it lightweight (~3M params) so the
      head doesn't overfit and the score reflects the encoder, not the head.

Usage:
    encoder = hydra.utils.instantiate(cfg.model)
    model   = IDMModel(encoder, action_dim=44, head_type="transformer")
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from typing import List, Optional


class VLMTokenBackbone(nn.Module):
    """Adapter exposing VLM visual tokens through the IDM encoder interface.

    layer_idx follows the policy VLM interface: -1 is the final hidden state
    and a non-negative value selects that transformer layer.
    """
    def __init__(self, name, device="cpu", layer_idx=-1):
        super().__init__()
        from Analyze.analyse.tsne_analysis import build_vlm
        self.vlm = build_vlm(name, device)
        self.name = name
        self.layer_idx = int(layer_idx)
        self.num_channels = int(getattr(getattr(self.vlm, "model", None), "config", None).hidden_size) if getattr(getattr(self.vlm, "model", None), "config", None) is not None and hasattr(getattr(self.vlm, "model", None).config, "hidden_size") else 4096
    def forward(self, frames):
        out=[]
        with torch.no_grad():
            for im in frames:
                if isinstance(im, np.ndarray): im=Image.fromarray(im.astype(np.uint8))
                z=self.vlm.encode_visual_tokens([[im]], [""], layer_idx=self.layer_idx)
                if z.ndim==3: z=z[0]
                out.append(z)
        # VLM backbones commonly emit bf16; IDM/FDM heads are fp32.
        return torch.stack(out).float()

# Encoder factory.
def build_encoder(vision_model_name: str, vlm_layer_idx: int = -1) -> nn.Module:
    """Build through HIVE-Bench's shared DinoGR00T encoder factory."""
    policy_root = Path(__file__).resolve().parents[1] / "Policy"
    if str(policy_root) not in sys.path:
        sys.path.insert(0, str(policy_root))
    raw_name = str(vision_model_name).strip().lower()
    # qwen3_layer16 (or *_l16) selects the same intermediate layer as
    # framework.vision_text_fusion.vlm_layer_idx=16 in QwenVisionGR00T.
    layer_idx = int(vlm_layer_idx)
    for suffix in ("_layer16", "_l16"):
        if raw_name.endswith(suffix):
            raw_name = raw_name[:-len(suffix)]
            if layer_idx < 0:
                layer_idx = 16
            break
    if raw_name in {"depthvlm", "qwen3", "xiaomi"}:
        return VLMTokenBackbone(raw_name, layer_idx=layer_idx)
    from hivebench.model.framework.DinoGR00T import _build_vision_encoder
    return _build_vision_encoder(raw_name)


def load_finetuned_vision_encoder(encoder: nn.Module, checkpoint_path: str) -> str:
    """Strictly restore only ``vision_encoder.*`` from a full policy checkpoint."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tuned policy checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a state-dict checkpoint, got {type(payload).__name__}: {path}")
    prefix = "vision_encoder."
    vision_state = {key[len(prefix):]: value for key, value in payload.items() if key.startswith(prefix)}
    if not vision_state:
        raise KeyError(f"No {prefix}* tensors found in policy checkpoint: {path}")
    # Strict loading is intentional: the supplied base encoder name must be the
    # same architecture used during policy fine-tuning. Action/text weights are
    # not read by the representation probes.
    encoder.load_state_dict(vision_state, strict=True)
    print(f"[FineTunedEncoder] loaded {len(vision_state)} vision tensors from {path}")
    return str(path)


# Positional encoding.
class SinusoidalPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# Transformer-based action expert head.
class ActionExpertHead(nn.Module):
    """
    Lightweight Transformer that attends over patch tokens from two frames
    and regresses the action.

    Sequence layout:
        [DIFF_query | tokens_t (N) | tokens_t+1 (N)]   length = 1 + 2N

    The [DIFF] token is forced to summarise the inter-frame transition.
    """

    def __init__(
        self,
        enc_dim: int,
        action_dim: int,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim  = model_dim
        self.action_dim = action_dim

        # Project encoder tokens → model_dim
        self.proj = nn.Linear(enc_dim, model_dim)

        # Learnable [DIFF] query token
        self.diff_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        # Segment embeddings: distinguish frame_t tokens vs frame_t+1 tokens
        self.seg_emb = nn.Embedding(3, model_dim)   # 0=diff, 1=frame_t, 2=frame_t+1

        # Positional encoding (within each frame's token sequence)
        self.pos_enc = SinusoidalPE(model_dim)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Output head: [DIFF] token → action
        self.output_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, action_dim),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ActionExpertHead] model_dim={model_dim}, layers={num_layers}, "
              f"heads={num_heads}, params={n_params:,}")

    def forward(
        self,
        tokens_t:  torch.Tensor,   # [B, N, enc_dim]
        tokens_t1: torch.Tensor,   # [B, N, enc_dim]
    ) -> torch.Tensor:             # [B, action_dim]
        B, N, _ = tokens_t.shape

        # Project to model_dim
        t  = self.proj(tokens_t)    # [B, N, D]
        t1 = self.proj(tokens_t1)   # [B, N, D]

        # Add positional encoding within each frame
        t  = self.pos_enc(t)
        t1 = self.pos_enc(t1)

        # Add segment embeddings
        seg_t   = self.seg_emb(torch.ones(B, N, dtype=torch.long, device=t.device))    # 1
        seg_t1  = self.seg_emb(torch.full((B, N), 2, dtype=torch.long, device=t.device))  # 2
        seg_d   = self.seg_emb(torch.zeros(B, 1, dtype=torch.long, device=t.device))   # 0

        t  = t  + seg_t
        t1 = t1 + seg_t1

        # Expand [DIFF] token and add segment embedding
        diff = self.diff_token.expand(B, -1, -1) + seg_d   # [B, 1, D]

        # Concatenate: [DIFF | frame_t tokens | frame_t+1 tokens]
        seq = torch.cat([diff, t, t1], dim=1)   # [B, 1+2N, D]

        # Transformer
        out = self.transformer(seq)             # [B, 1+2N, D]

        # Read [DIFF] token (position 0)
        diff_out = out[:, 0, :]                 # [B, D]

        return self.output_proj(diff_out)       # [B, action_dim]


# Legacy MLP head retained for ablation.
class MLPHead(nn.Module):
    """Simple 3-layer MLP baseline. Kept for ablation vs ActionExpertHead."""
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        print(f"[MLPHead] hidden={hidden_dim}, params={sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Full inverse dynamics model.
class IDMModel(nn.Module):
    """
    Inverse Dynamics Model: (frame_t, frame_t+1) → predicted_action

    Args:
        encoder:    vision encoder (frozen) instantiated from Hydra config
        action_dim: GT action dimension (44 for RoboCasa GR1)
        enc_dim:    Encoder channel dimension. Inferred from common encoder
                    wrappers when omitted.
        head_type:  'transformer' (recommended) or 'mlp' (ablation)
        model_dim:  Transformer hidden dim (only for head_type='transformer')
        num_layers: Transformer depth
        num_heads:  Attention heads
        hidden_dim: MLP hidden dim (only for head_type='mlp')
        pool:       token aggregation for MLP head — 'mean' or 'cls'
    """

    def __init__(
        self,
        encoder:    nn.Module,
        action_dim: int  = 44,
        enc_dim:    Optional[int] = None,
        head_type:  str  = "transformer",
        model_dim:  int  = 256,
        num_layers: int  = 4,
        num_heads:  int  = 4,
        hidden_dim: int  = 512,
        pool:       str  = "mean",
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.encoder   = encoder
        self.head_type = head_type
        self.pool      = pool
        enc_dim = enc_dim or self._infer_encoder_dim(encoder)

        if head_type == "transformer":
            self.head = ActionExpertHead(
                enc_dim=enc_dim, action_dim=action_dim,
                model_dim=model_dim, num_layers=num_layers,
                num_heads=num_heads, ffn_dim=model_dim * 2,
                dropout=dropout,
            )
        elif head_type == "mlp":
            self.head = MLPHead(
                input_dim=enc_dim * 2,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"head_type must be 'transformer' or 'mlp', got '{head_type}'")

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        total_trainable = sum(p.numel() for p in self.head.parameters())
        print(f"[IDMModel] encoder={encoder.__class__.__name__}, enc_dim={enc_dim}, "
              f"action_dim={action_dim}, head={head_type}, "
              f"trainable_params={total_trainable:,}")

    @staticmethod
    def _infer_encoder_dim(encoder: nn.Module) -> int:
        if hasattr(encoder, "num_channels"):
            return int(encoder.num_channels)

        model = getattr(encoder, "model", None)
        config = getattr(model, "config", None)
        for attr in ("hidden_size", "projection_dim", "embed_dim", "width"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)

        vision_config = getattr(config, "vision_config", None)
        for attr in ("hidden_size", "projection_dim", "embed_dim", "width"):
            value = getattr(vision_config, attr, None)
            if value is not None:
                return int(value)

        raise ValueError(
            "Cannot infer encoder channel dimension. Set encoder_dim in the Hydra config."
        )

    def _get_tokens(self, frames: List) -> torch.Tensor:
        """
        frames: List[PIL/ndarray] length B
        returns: [B, N, D]  — all patch tokens (NOT pooled)
        """
        if hasattr(self.encoder, "prepare_dino_input"):
            batch = [[f] for f in frames]
            pixel_values = self.encoder.prepare_dino_input(batch)
            with torch.no_grad():
                tokens = self.encoder(pixel_values)   # [B, N, D]
            return tokens

        with torch.no_grad():
            tokens = self.encoder(frames)

        if tokens.ndim == 4:
            b, c, h, w = tokens.shape
            return tokens.reshape(b, c, h * w).transpose(1, 2)
        if tokens.ndim != 3:
            raise ValueError(f"Expected encoder output to be [B,N,D] or [B,C,H,W], got {tuple(tokens.shape)}")
        return tokens

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool tokens for MLP head."""
        if self.pool == "mean":
            return tokens.mean(dim=1)
        elif self.pool == "cls":
            return tokens[:, 0, :]
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

    def forward(
        self,
        frames_t:   List,
        frames_t1:  List,
        actions_gt: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Returns dict with:
            pred_action  [B, action_dim]
            loss_mse     scalar  (if actions_gt provided)
            loss_l1      scalar  (if actions_gt provided)
            loss_per_dim [action_dim]  per-dimension L1 (if actions_gt provided)
        """
        tokens_t  = self._get_tokens(frames_t)    # [B, N, D]
        tokens_t1 = self._get_tokens(frames_t1)   # [B, N, D]

        if self.head_type == "transformer":
            pred_action = self.head(tokens_t, tokens_t1)          # [B, action_dim]
        else:
            feat = torch.cat([self._pool(tokens_t),
                              self._pool(tokens_t1)], dim=-1)      # [B, 2D]
            pred_action = self.head(feat)                           # [B, action_dim]

        out = {"pred_action": pred_action}

        if actions_gt is not None:
            gt = actions_gt.to(pred_action.device, dtype=pred_action.dtype)
            out["loss_mse"]     = nn.functional.mse_loss(pred_action, gt)
            out["loss_l1"]      = nn.functional.l1_loss(pred_action, gt)
            # Per-dimension L1: [action_dim]
            out["loss_per_dim"] = (pred_action - gt).abs().mean(dim=0)

        return out

    def predict(self, frames_t: List, frames_t1: List) -> torch.Tensor:
        return self.forward(frames_t, frames_t1)["pred_action"]


class FDMModel(nn.Module):
    """Frozen-encoder forward dynamics: (z_t, delta_actions_t:t+15) -> z_t+16."""
    def __init__(self, encoder, action_dim=29, enc_dim=None, model_dim=256, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.enc_dim = enc_dim or IDMModel._infer_encoder_dim(encoder)
        self.visual_proj = nn.Linear(self.enc_dim, model_dim)
        self.action_proj = nn.Linear(action_dim, model_dim)
        self.segment = nn.Embedding(2, model_dim)  # 0=visual, 1=action
        self.pos = SinusoidalPE(model_dim, max_len=8192)  # supports variable-N encoders, incl. SAM/VGGT
        layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*2,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, self.enc_dim))
        for p in self.encoder.parameters(): p.requires_grad = False
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[FDMModel] enc_dim={self.enc_dim}, model_dim={model_dim}, layers={num_layers}, heads={num_heads}, trainable={n:,}")

    def _tokens(self, frames):
        if hasattr(self.encoder, "prepare_dino_input"):
            pixels = self.encoder.prepare_dino_input([[x] for x in frames])
            with torch.no_grad(): return self.encoder(pixels)
        with torch.no_grad(): out = self.encoder(frames)
        if out.ndim == 4:
            b,c,h,w = out.shape; return out.reshape(b,c,h*w).transpose(1,2)
        if out.ndim != 3: raise ValueError(f"Expected [B,N,D] tokens, got {tuple(out.shape)}")
        return out

    def forward(self, frames_t, frames_th, action_tokens):
        source, target = self._tokens(frames_t), self._tokens(frames_th)
        if source.shape[1:] != target.shape[1:]:
            raise ValueError(f"Source/target token shapes differ: {tuple(source.shape)} vs {tuple(target.shape)}")
        b, n, _ = source.shape
        visual = self.pos(self.visual_proj(source)) + self.segment(torch.zeros(b, n, dtype=torch.long, device=source.device))
        actions = self.pos(self.action_proj(action_tokens.to(source.device, source.dtype))) + self.segment(torch.ones(b, action_tokens.shape[1], dtype=torch.long, device=source.device))
        pred = self.output_proj(self.transformer(torch.cat([visual, actions], dim=1))[:, :n])
        pred, target = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
        return {"pred_tokens": pred, "target_tokens": target, "loss_mse": F.mse_loss(pred, target), "cosine_similarity": (pred * target).sum(-1).mean()}


class StateModel(nn.Module):
    """Single-frame visual proprioception probe with an encoder-width query token."""
    def __init__(self, encoder, state_dim=29, enc_dim=None, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder=encoder; self.enc_dim=enc_dim or IDMModel._infer_encoder_dim(encoder)
        if self.enc_dim % num_heads: raise ValueError(f"encoder dim {self.enc_dim} is not divisible by {num_heads} heads")
        self.input_norm=nn.LayerNorm(self.enc_dim)
        self.query=nn.Parameter(torch.randn(1,1,self.enc_dim)*0.02)
        self.query_segment=nn.Parameter(torch.zeros(1,1,self.enc_dim))
        layer=nn.TransformerEncoderLayer(d_model=self.enc_dim,nhead=num_heads,dim_feedforward=self.enc_dim*2,dropout=dropout,batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=num_layers)
        self.output_proj=nn.Sequential(nn.LayerNorm(self.enc_dim),nn.Linear(self.enc_dim,self.enc_dim),nn.GELU(),nn.Linear(self.enc_dim,state_dim))
        for p in self.encoder.parameters(): p.requires_grad=False
        print(f"[StateModel] enc_dim={self.enc_dim}, layers={num_layers}, heads={num_heads}, trainable={sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _tokens(self, frames):
        if hasattr(self.encoder,"prepare_dino_input"):
            pixels=self.encoder.prepare_dino_input([[x] for x in frames])
            with torch.no_grad(): return self.encoder(pixels)
        with torch.no_grad(): out=self.encoder(frames)
        if out.ndim==4:
            b,c,h,w=out.shape; return out.reshape(b,c,h*w).transpose(1,2)
        if out.ndim!=3: raise ValueError(f"Expected [B,N,D] tokens, got {tuple(out.shape)}")
        return out

    def forward(self, frames, states=None):
        tokens=self.input_norm(self._tokens(frames)); q=self.query.expand(tokens.shape[0],-1,-1)+self.query_segment
        pred=self.output_proj(self.transformer(torch.cat([tokens,q],dim=1))[:,-1])
        out={"pred_state":pred}
        if states is not None:
            target=states.to(pred.device,pred.dtype); out["loss_mse"]=F.mse_loss(pred,target); out["loss_l1"]=F.l1_loss(pred,target)
        return out


class ObjectStateModel(nn.Module):
    """Fixed-semantic object query for camera-frame 3D task-object localization."""
    def __init__(self, encoder, num_queries=2, enc_dim=None, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder=encoder; self.enc_dim=enc_dim or IDMModel._infer_encoder_dim(encoder)
        if self.enc_dim % num_heads: raise ValueError(f"encoder dim {self.enc_dim} is not divisible by {num_heads} heads")
        self.input_norm=nn.LayerNorm(self.enc_dim); self.queries=nn.Parameter(torch.randn(1,num_queries,self.enc_dim)*0.02)
        layer=nn.TransformerEncoderLayer(d_model=self.enc_dim,nhead=num_heads,dim_feedforward=self.enc_dim*2,dropout=dropout,batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=num_layers)
        self.output_proj=nn.Sequential(nn.LayerNorm(self.enc_dim),nn.Linear(self.enc_dim,self.enc_dim),nn.GELU(),nn.Linear(self.enc_dim,3))
        for p in self.encoder.parameters(): p.requires_grad=False
        print(f"[ObjectStateModel] enc_dim={self.enc_dim}, queries={num_queries}, layers={num_layers}, heads={num_heads}, trainable={sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _tokens(self, frames):
        if hasattr(self.encoder,'prepare_dino_input'):
            pixels=self.encoder.prepare_dino_input([[x] for x in frames])
            with torch.no_grad(): return self.encoder(pixels)
        with torch.no_grad(): out=self.encoder(frames)
        if out.ndim==4:
            b,c,h,w=out.shape; return out.reshape(b,c,h*w).transpose(1,2)
        if out.ndim!=3: raise ValueError(f"Expected [B,N,D] tokens, got {tuple(out.shape)}")
        return out

    def forward(self, frames, positions=None, mask=None):
        tokens=self.input_norm(self._tokens(frames)); q=self.queries.expand(tokens.shape[0],-1,-1)
        pred=self.output_proj(self.transformer(torch.cat([tokens,q],dim=1))[:,-q.shape[1]:])
        out={'pred_position':pred}
        if positions is not None:
            target=positions.to(pred.device,pred.dtype); m=mask.to(pred.device,pred.dtype).unsqueeze(-1)
            out['loss_mse']=((pred-target).square()*m).sum()/m.sum().clamp_min(1)/pred.shape[-1]
            out['loss_l1']=((pred-target).abs()*m).sum()/m.sum().clamp_min(1)/pred.shape[-1]
        return out

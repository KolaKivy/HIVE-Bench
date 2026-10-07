"""Load pretrained LingBot-Vision backbones from local files or Hugging Face.

Entry points:

- :func:`load_pretrained_backbone` — one call from a variant name / repo id /
  local directory to a ready-to-use frozen eval model.
- :func:`load_config` + :func:`load_backbone_state` + :func:`load_backbone` —
  the same pipeline broken into steps for callers that manage paths themselves.
- :func:`extract_patch_tokens` — run the backbone on a normalized image batch
  and return the per-patch feature tokens.
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from .build import build_backbone_from_cfg


DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

PRETRAINED_VARIANTS = {
    "small": {
        "repo_id": "robbyant/lingbot-vision-vit-small",
        "config_file": "configs/lbot_vision_vits.yaml",
        "checkpoint_file": "model.pt",
        "local_checkpoint_file": "lbotv_vit_small.pt",
    },
    "base": {
        "repo_id": "robbyant/lingbot-vision-vit-base",
        "config_file": "configs/lbot_vision_vitb.yaml",
        "checkpoint_file": "model.pt",
        "local_checkpoint_file": "lbotv_vit_base.pt",
    },
    "large": {
        "repo_id": "robbyant/lingbot-vision-vit-large",
        "config_file": "configs/lbot_vision_vitl.yaml",
        "checkpoint_file": "model.pt",
        "local_checkpoint_file": "lbotv_vit_large.pt",
    },
    "giant": {
        "repo_id": "robbyant/lingbot-vision-vit-giant",
        "config_file": "configs/lbot_vision_vitg.yaml",
        "checkpoint_file": "model.pt",
        "local_checkpoint_file": "lbotv_vit_giant.pt",
    },
}

_VARIANT_ALIASES = {
    "s": "small",
    "vit_s": "small",
    "vits": "small",
    "small": "small",
    "b": "base",
    "vit_b": "base",
    "vitb": "base",
    "base": "base",
    "l": "large",
    "vit_l": "large",
    "vitl": "large",
    "large": "large",
    "g": "giant",
    "vit_g": "giant",
    "vitg": "giant",
    "giant": "giant",
}


def _infer_variant_from_repo_id(repo_id_or_path):
    if repo_id_or_path is None:
        return None
    name = str(repo_id_or_path).rstrip("/").split("/")[-1].lower().replace("_", "-")
    for variant in PRETRAINED_VARIANTS:
        if name.endswith(f"vit-{variant}") or name.endswith(f"vit{variant}") or name.endswith(variant):
            return variant
    return None


def _normalize_variant(variant="auto", repo_id_or_path=None):
    if variant is None or str(variant).lower() == "auto":
        return _infer_variant_from_repo_id(repo_id_or_path) or "large"
    key = str(variant).lower().replace("-", "_")
    if key not in _VARIANT_ALIASES:
        valid = ", ".join(sorted(PRETRAINED_VARIANTS))
        raise ValueError(f"unknown variant {variant!r}; expected one of: {valid}, auto")
    return _VARIANT_ALIASES[key]


def _resolve_dtype(dtype, device):
    if dtype == "auto":
        return torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    if isinstance(dtype, str):
        if dtype not in DTYPE_MAP:
            valid = ", ".join(sorted(DTYPE_MAP))
            raise ValueError(f"unknown dtype {dtype!r}; expected one of: {valid}, auto")
        return DTYPE_MAP[dtype]
    return dtype


def _packaged_config_path(config_file):
    """Resolve a config reference to an existing file, or return None.

    Tries the path as given first (absolute or relative to the working
    directory), then falls back to the copy shipped inside the package at
    ``lingbot_vision/configs/<basename>``.
    """
    path = Path(config_file).expanduser()
    if path.is_file():
        return path
    packaged = Path(__file__).parent / "configs" / Path(config_file).name
    if packaged.is_file():
        return packaged
    return None


def _hf_hub_download(repo_id, filename, *, cache_dir=None, revision=None, local_files_only=False):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Loading directly from Hugging Face requires `huggingface_hub`. "
            "Install it with `python -m pip install huggingface_hub`."
        ) from exc
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )


def resolve_pretrained_files(
    repo_id_or_path=None,
    variant="auto",
    *,
    config_file=None,
    checkpoint_file=None,
    cache_dir=None,
    revision=None,
    local_files_only=False,
):
    """Resolve (config_path, checkpoint_path) for a pretrained variant.

    ``repo_id_or_path`` may be a local directory or a Hugging Face repo id.
    With a local directory, the checkpoint must be present there (the legacy
    filename is also checked) and only a missing config falls back to the
    packaged copy. Otherwise the checkpoint is downloaded from Hugging Face,
    and the config comes from the packaged copy or, failing that, the same
    repo.
    """
    variant = _normalize_variant(variant, repo_id_or_path)
    spec = PRETRAINED_VARIANTS[variant]
    repo_id_or_path = repo_id_or_path or spec["repo_id"]
    config_file = config_file or spec["config_file"]
    checkpoint_file = checkpoint_file or spec["checkpoint_file"]
    local_checkpoint_file = spec.get("local_checkpoint_file")

    root = Path(repo_id_or_path).expanduser()
    if root.exists():
        if not root.is_dir():
            raise ValueError("repo_id_or_path must be a directory when using local files")
        config_path = root / config_file
        if not config_path.is_file():
            config_path = _packaged_config_path(config_file)
        checkpoint_path = root / checkpoint_file
        if not checkpoint_path.is_file() and local_checkpoint_file and checkpoint_file == spec["checkpoint_file"]:
            legacy_checkpoint_path = root / local_checkpoint_file
            if legacy_checkpoint_path.is_file():
                checkpoint_path = legacy_checkpoint_path
    else:
        config_path = _packaged_config_path(config_file)
        if config_path is None:
            config_path = Path(
                _hf_hub_download(
                    repo_id_or_path,
                    config_file,
                    cache_dir=cache_dir,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            )
        checkpoint_path = Path(
            _hf_hub_download(
                repo_id_or_path,
                checkpoint_file,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
            )
        )

    if not config_path or not config_path.is_file():
        raise FileNotFoundError(str(config_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    return config_path, checkpoint_path


def load_pretrained_backbone(
    repo_id_or_path=None,
    variant="auto",
    *,
    device=None,
    dtype="auto",
    cache_dir=None,
    revision=None,
    local_files_only=False,
    config_file=None,
    checkpoint_file=None,
    verbose=True,
):
    """Resolve, build and load a pretrained backbone; returns (model, embed_dim)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(dtype, device)
    config_path, checkpoint_path = resolve_pretrained_files(
        repo_id_or_path,
        variant,
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    cfg = load_config(config_path)
    ckpt = load_backbone_state(checkpoint_path)
    return load_backbone(cfg, ckpt, device=device, dtype=dtype, verbose=verbose)


def load_config(config_file, default_config=None):
    """Load an experiment config. Optionally merge it over a caller-provided default.

    ``config_file`` is resolved as given first, then against the configs
    shipped inside the package (so ``configs/lbot_vision_vitl.yaml`` works from
    any working directory).
    """
    config_path = _packaged_config_path(config_file)
    if config_path is None:
        raise FileNotFoundError(str(config_file))
    cfg = OmegaConf.load(config_path)
    if default_config is not None:
        cfg = OmegaConf.merge(OmegaConf.load(default_config), cfg)
    return cfg


def load_backbone_state(backbone_path, map_location="cpu"):
    """Load a backbone checkpoint (`.pt`) from a local path."""
    backbone_path = Path(backbone_path)
    if not backbone_path.is_file():
        raise FileNotFoundError(str(backbone_path))
    return torch.load(str(backbone_path), map_location=map_location, weights_only=True)


def _unwrap_state(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("teacher", "model_state", "state_dict", "model", "backbone"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _clean_compiled(key):
    return key.replace("_orig_mod.", "")


def extract_submodule(ckpt, prefix):
    sd = _unwrap_state(ckpt)
    sd = {_clean_compiled(k): v for k, v in sd.items()}
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_backbone(cfg, ckpt, device="cuda", dtype=torch.float32, verbose=True):
    """Build the backbone from config and load a checkpoint into it.

    Accepts either a raw backbone state dict, a ``{'backbone': state_dict}``
    wrapper (or the other common wrapper keys), or a full checkpoint whose
    keys carry a ``backbone.`` prefix. Returns a frozen eval-mode model and
    its embedding dimension.
    """
    model, embed_dim = build_backbone_from_cfg(cfg)
    backbone_sd = extract_submodule(ckpt, "backbone.")
    if not backbone_sd:
        backbone_sd = _unwrap_state(ckpt)
    if not backbone_sd:
        raise ValueError("could not locate backbone weights in file")

    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    n_params = len(model.state_dict())
    if verbose:
        print(
            f"[lingbot_vision] arch={cfg.student.arch} "
            f"impl={cfg.student.get('backbone_impl', 'lbot_vision')} "
            f"embed_dim={embed_dim} loaded={len(backbone_sd)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if unexpected:
            print(f"[lingbot_vision] unexpected first 5: {list(unexpected)[:5]}")
    if len(missing) > 0.5 * n_params:
        raise RuntimeError(
            f"checkpoint does not match the config: {len(missing)} of {n_params} "
            f"backbone parameters are missing (e.g. {list(missing)[:3]}); "
            "check that the config and weights belong to the same model variant"
        )

    model = model.to(device).to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, embed_dim


@torch.no_grad()
def extract_patch_tokens(backbone, img_norm, device, dtype):
    """Return (patch_tokens, (h, w)) for a normalized image batch [B, 3, H, W]."""
    x = img_norm.to(device).to(dtype)
    ps = backbone.patch_size
    _, _, H, W = x.shape
    h, w = H // ps, W // ps
    use_ac = device.startswith("cuda") and dtype != torch.float32
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_ac):
        out = backbone(x, is_training=True)
    return out["x_norm_patchtokens"], (h, w)

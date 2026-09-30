"""Backbone construction from an experiment config."""

from collections.abc import Iterable

from . import vit


def _cfg_get(obj, key, default=None):
    return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)


def _resolve_img_size(img_size):
    if isinstance(img_size, (str, bytes)):
        return int(img_size)
    if isinstance(img_size, Iterable):
        return max(int(x) for x in img_size)
    return int(img_size)


def build_backbone_from_cfg(cfg):
    """Build a LingBot-Vision ViT from a config and return ``(model, embed_dim)``.

    The config layout matches the shipped YAMLs: ``cfg.student`` selects the
    architecture (``student.arch``, e.g. ``vit_large``) and the encoder options
    under ``student.lbot_vision``; ``cfg.crops.global_crops_size`` sets the
    nominal image size. The model is returned freshly initialized — load
    pretrained weights with :func:`lingbot_vision.load_backbone`.
    """
    student = cfg.student
    impl = _cfg_get(student, "backbone_impl", "lbot_vision")
    if impl != "lbot_vision":
        raise ValueError(f"expected student.backbone_impl=lbot_vision, got {impl!r}")

    arch = student.arch
    builder = vit.__dict__.get(arch)
    if builder is None:
        raise ValueError(f"unknown architecture {arch!r} (expected a vit_* builder)")

    hcfg = _cfg_get(student, "lbot_vision", {})
    kwargs = dict(
        img_size=_resolve_img_size(cfg.crops.global_crops_size),
        patch_size=student.patch_size,
        layerscale_init=student.layerscale,
        qkv_bias=student.qkv_bias,
        proj_bias=student.proj_bias,
        ffn_bias=student.ffn_bias,
        n_storage_tokens=student.num_register_tokens,
        norm_layer=_cfg_get(hcfg, "norm_layer", "layernorm"),
        ffn_layer=_cfg_get(hcfg, "ffn_layer", "mlp"),
        mask_k_bias=_cfg_get(hcfg, "mask_k_bias", False),
        untie_cls_and_patch_norms=_cfg_get(hcfg, "untie_cls_and_patch_norms", False),
        untie_global_and_local_cls_norm=_cfg_get(hcfg, "untie_global_and_local_cls_norm", False),
        pos_embed_rope_base=_cfg_get(hcfg, "pos_embed_rope_base", 100.0),
        pos_embed_rope_min_period=_cfg_get(hcfg, "pos_embed_rope_min_period", None),
        pos_embed_rope_max_period=_cfg_get(hcfg, "pos_embed_rope_max_period", None),
        pos_embed_rope_normalize_coords=_cfg_get(hcfg, "pos_embed_rope_normalize_coords", "separate"),
        pos_embed_rope_shift_coords=_cfg_get(hcfg, "pos_embed_rope_shift_coords", None),
        pos_embed_rope_jitter_coords=_cfg_get(hcfg, "pos_embed_rope_jitter_coords", None),
        pos_embed_rope_rescale_coords=_cfg_get(hcfg, "pos_embed_rope_rescale_coords", None),
        pos_embed_rope_dtype=_cfg_get(hcfg, "pos_embed_rope_dtype", "bf16"),
        drop_path_rate=0.0,
    )
    model = builder(**kwargs)
    model.init_weights()
    return model, model.embed_dim

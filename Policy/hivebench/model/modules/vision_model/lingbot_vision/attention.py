"""SDPA attention for the LingBot-Vision backbone.

The backend is selected once per process, either via :func:`set_backend` or the
``LINGBOT_VISION_ATTN_BACKEND`` environment variable (read on first attention
call). Supported values:

  sdpa          torch SDPA with automatic kernel dispatch (default)
  sdpa-cudnn    torch SDPA forced to the cuDNN kernel
  sdpa-flash    torch SDPA forced to the flash kernel
  sdpa-mem_eff  torch SDPA forced to the memory-efficient kernel

Forcing a specific kernel uses ``torch.nn.attention.sdpa_kernel`` and therefore
requires torch >= 2.3 (``sdpa-cudnn`` needs torch >= 2.4, which added
``SDPBackend.CUDNN_ATTENTION``); the default ``sdpa`` backend works on older
versions.
"""
import os

import torch.nn.functional as F
from torch import Tensor

_DENSE_BACKENDS = ("sdpa", "sdpa-cudnn", "sdpa-flash", "sdpa-mem_eff")

_backend = None


def set_backend(name: str) -> None:
    """Select the process-wide attention backend (before the first attention call)."""
    global _backend
    if name not in _DENSE_BACKENDS:
        raise ValueError(f"unknown attention backend {name!r}; expected one of {_DENSE_BACKENDS}")
    if _backend is not None and _backend != name:
        raise RuntimeError(
            f"attention backend already initialized to {_backend!r}; cannot switch to {name!r} "
            "(the backend is fixed for the lifetime of the process)"
        )
    _backend = name


def get_backend() -> str:
    if _backend is None:
        set_backend(os.environ.get("LINGBOT_VISION_ATTN_BACKEND", "sdpa"))
    return _backend


def _sdpa_ctx(backend: str):
    # Imported lazily so the default auto-dispatch path never touches this API.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    forced = {
        "sdpa-cudnn": SDPBackend.CUDNN_ATTENTION,
        "sdpa-flash": SDPBackend.FLASH_ATTENTION,
        "sdpa-mem_eff": SDPBackend.EFFICIENT_ATTENTION,
    }[backend]
    return sdpa_kernel([forced])


def attention(q: Tensor, k: Tensor, v: Tensor, layout: str = "bnhd") -> Tensor:
    """Dense softmax attention through the selected backend.

    layout "bnhd": q/k/v are [B, N, heads, head_dim]. layout "bhnd":
    [B, heads, N, head_dim] (SDPA-native). Returns the same layout as the
    input.
    """
    backend = get_backend()
    if layout == "bnhd":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if backend == "sdpa":
        out = F.scaled_dot_product_attention(q, k, v)
    else:
        with _sdpa_ctx(backend):
            out = F.scaled_dot_product_attention(q, k, v)
    if layout == "bnhd":
        out = out.transpose(1, 2)
    return out

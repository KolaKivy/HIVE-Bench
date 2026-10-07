"""Image loading and ImageNet normalization."""

import os

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".JPG", ".PNG", ".BMP")


def _snap(size, patch_size):
    return max(patch_size, (size // patch_size) * patch_size)


def load_image(path, size=512, patch_size=16, mode="square"):
    """Load an image and return (normalized tensor, uint8 RGB array, (H, W)).

    ``size`` is snapped down to a multiple of ``patch_size``. Modes:

    - ``"square"``: plain resize to size x size (does not preserve aspect ratio).
    - ``"shortest"``: resize the shortest side to ``size``, then center-crop a
      size x size square.
    """
    size = _snap(size, patch_size)
    pil = Image.open(path).convert("RGB")
    if mode == "square":
        crop = pil.resize((size, size), resample=Image.BILINEAR)
    elif mode == "shortest":
        w0, h0 = pil.size
        if w0 < h0:
            new_w, new_h = size, int(round(size * h0 / w0))
        else:
            new_h, new_w = size, int(round(size * w0 / h0))
        resized = pil.resize((new_w, new_h), resample=Image.BILINEAR)
        left, top = (new_w - size) // 2, (new_h - size) // 2
        crop = resized.crop((left, top, left + size, top + size))
    else:
        raise ValueError(f"unknown resize mode: {mode!r}")

    img_rgb = np.asarray(crop, dtype=np.uint8)
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
    img_norm = (img_t - IMAGENET_MEAN) / IMAGENET_STD
    return img_norm, img_rgb, (size, size)


def iter_image_paths(input_path):
    if os.path.isfile(input_path):
        yield input_path
        return
    for root, _, files in os.walk(input_path):
        for f in sorted(files):
            if f.endswith(IMG_EXTS):
                yield os.path.join(root, f)

"""LingBot-Vision backbone adapter for VisionGR00T."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .lingbot_vision import load_pretrained_backbone


class LingBotVisionBackbone(nn.Module):
    """Expose LingBot-Vision normalized patch features as visual tokens."""

    def __init__(
        self,
        model_id: str = "robbyant/lingbot-vision-vit-large",
        image_size: int = 224,
        variant: str = "auto",
    ) -> None:
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError(
                f"LingBot-Vision image_size must be divisible by 16, got {image_size}."
            )
        self.image_size = int(image_size)
        self.model, self.num_channels = load_pretrained_backbone(
            repo_id_or_path=model_id,
            variant=variant,
            device="cpu",
            dtype="fp32",
        )

    @staticmethod
    def _to_tensor(image: Any) -> torch.Tensor:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(array).permute(2, 0, 1)

        if isinstance(image, np.ndarray):
            value = torch.from_numpy(image)
        elif torch.is_tensor(image):
            value = image.detach()
        else:
            raise TypeError(
                f"LingBot-Vision does not support image type {type(image)!r}."
            )

        if value.ndim == 4:
            value = value[0]
        if value.ndim != 3:
            raise ValueError(
                f"Expected an image with 3 dimensions, got {tuple(value.shape)}."
            )
        if value.shape[0] not in (1, 3) and value.shape[-1] in (1, 3):
            value = value.permute(2, 0, 1)

        value = value.float()
        if value.min() >= -0.1 and value.max() > 1.5:
            value = value / 255.0
        if value.min() < -0.1:
            return value[:3]
        if value.shape[0] == 1:
            value = value.repeat(3, 1, 1)
        return value[:3]

    def _preprocess(self, image: Any) -> torch.Tensor:
        tensor = F.interpolate(
            self._to_tensor(image).unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def prepare_dino_input(self, img_list) -> torch.Tensor:
        """Preprocess a batch of samples containing one or more camera views."""
        if not img_list or any(not views for views in img_list):
            raise ValueError(
                "LingBot-Vision requires at least one image for every sample."
            )
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        flat = torch.cat(
            [self._preprocess(image) for views in img_list for image in views],
            dim=0,
        )
        return flat.to(device=device, dtype=dtype)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return normalized LingBot-Vision patch tokens with shape [B, N, D]."""
        output = self.model(pixel_values, is_training=True)
        return output["x_norm_patchtokens"]

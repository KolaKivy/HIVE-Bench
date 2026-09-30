# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import io

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from PIL import Image
from torchvision import transforms


def resize_images_to_fixed_size(images, target_size=(504, 504)):
    resized_images = []
    resize_transform = transforms.Resize(
        target_size, interpolation=transforms.InterpolationMode.BICUBIC
    )

    for img in images:
        if isinstance(img, str):
            pil_img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        resized_images.append(np.array(resize_transform(pil_img)))

    return resized_images


class DA3(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "depth-anything/da3nested-giant-large",
        device: str | torch.device = "cuda",
        target_size: tuple[int, int] = (504, 504),
        process_res: int = 252,
        export_feat_layers: list[int] | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.target_size = target_size
        self.process_res = process_res
        self.export_feat_layers = export_feat_layers or [39]
        self.model = DepthAnything3.from_pretrained(model_name).to(device=self.device)

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        images = resize_images_to_fixed_size(images, target_size=self.target_size)
        kwargs = {
            "process_res": self.process_res,
            "export_feat_layers": self.export_feat_layers,
        }

        if requires_grad:
            prediction = self.model.inference(images, **kwargs)
        else:
            with torch.no_grad():
                prediction = self.model.inference(images, **kwargs)

        layer = self.export_feat_layers[-1]
        visual_tokens = getattr(prediction.aux, f"feat_layer_{layer}")
        if not isinstance(visual_tokens, torch.Tensor):
            visual_tokens = torch.from_numpy(visual_tokens).float()
        return visual_tokens.permute(0, 3, 1, 2).to(self.device)


def print_feature_size(
    model_name: str = "depth-anything/da3nested-giant-large",
    target_size: tuple[int, int] = (252, 252),
) -> None:
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    images = [np.array(image)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DA3(model_name=model_name, device=device, target_size=target_size)
    visual_tokens = extractor.forward(images)

    print(f"--- Depth Anything 3 Feature Info ---")
    print(f"Model Name:      {model_name}")
    print(f"Input Size:      {target_size}")
    print(f"Visual Tokens:   {visual_tokens.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()

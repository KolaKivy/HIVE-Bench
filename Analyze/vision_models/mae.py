# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from transformers import AutoImageProcessor, ViTMAEModel


class MAE(torch.nn.Module):
    'MAE implementation.'

    def __init__(
        self,
        model_name: str = "facebook/vit-mae-base",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTMAEModel.from_pretrained(model_name)
        self.model.config.mask_ratio = 0.0
        self.model = self.model.to(self.device)

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        if requires_grad:
            outputs = self.model(**inputs, noise=None)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, noise=None)

        visual_tokens = outputs.last_hidden_state[:, 1:]
        batch_size, num_patches, num_channels = visual_tokens.size()
        side_length = int(np.sqrt(num_patches))
        return visual_tokens.transpose(1, 2).reshape(batch_size, num_channels, side_length, side_length)


def print_feature_size(model_name: str = "facebook/vit-mae-base") -> None:
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = MAE(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)
    print(f"Model: {model_name}\nVisual Tokens: {visual_tokens.size()}")


if __name__ == "__main__":
    print_feature_size()

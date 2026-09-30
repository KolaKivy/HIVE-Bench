# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from transformers import AutoProcessor, CLIPVisionModel


class Clip(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        if requires_grad:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)

        visual_tokens = outputs.last_hidden_state[:, 1:]
        batch_size, num_patches, num_channels = visual_tokens.size()
        side_length = int(np.sqrt(num_patches))
        return visual_tokens.transpose(1, 2).reshape(batch_size, num_channels, side_length, side_length)


def print_feature_size(model_name: str = "openai/clip-vit-large-patch14") -> None:
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = Clip(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)
    print(model_name, visual_tokens.size())


if __name__ == "__main__":
    print_feature_size()

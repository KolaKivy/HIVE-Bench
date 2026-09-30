# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms


class PE(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "PE-Spatial-L14-448",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model = pe.VisionTransformer.from_config(model_name, pretrained=True).to(self.device)
        self.processor = transforms.get_image_transform(self.model.image_size)

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        image_tensors = torch.stack(
            [self.processor(Image.fromarray(image)) for image in images], dim=0
        ).to(self.device)

        if requires_grad:
            last_hidden_state = self.model(image_tensors)
        else:
            with torch.no_grad():
                last_hidden_state = self.model(image_tensors)

        patch_features = last_hidden_state[:, 1:, :]
        batch_size, num_patches, num_channels = patch_features.size()
        side_length = int(np.sqrt(num_patches))
        return patch_features.transpose(1, 2).reshape(batch_size, num_channels, side_length, side_length)


def print_feature_size(model_name: str = "PE-Spatial-L14-448") -> None:
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = PE(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)

    print(f"--- PE Feature Info ---")
    print(f"Model Name:      {model_name}")
    print(f"Visual Tokens:   {visual_tokens.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()

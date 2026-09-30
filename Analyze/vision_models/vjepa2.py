# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class VJEPA2(torch.nn.Module):
    'VJEPA2 implementation.'

    def __init__(
        self,
        model_name: str = "vjepa2_1_vit_base_384",
        repo_path: str = "third_party/VJEPA2",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        result = torch.hub.load(repo_path, model_name, source="local", trust_repo=True)
        
        self.model = result[0] if isinstance(result, tuple) else result
        self.model = self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        image_tensors = torch.stack(
            [self.transform(Image.fromarray(img)) for img in images], dim=0
        ).to(self.device)
        
        image_tensors = image_tensors.unsqueeze(2)

        if requires_grad:
            visual_tokens = self.model(image_tensors)
        else:
            with torch.no_grad():
                visual_tokens = self.model(image_tensors)

        if isinstance(visual_tokens, (tuple, list)):
            visual_tokens = visual_tokens[0]
        
        batch_size, num_patches, num_channels = visual_tokens.size()
        side_length = int(np.sqrt(num_patches))
        return visual_tokens.transpose(1, 2).reshape(batch_size, num_channels, side_length, side_length)


def print_feature_size(model_name: str = "vjepa2_1_vit_base_384") -> None:
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = VJEPA2(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)
    print(f"Model: {model_name}\nVisual Tokens: {visual_tokens.size()}")


if __name__ == "__main__":
    print_feature_size()

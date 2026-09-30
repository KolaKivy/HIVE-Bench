# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class Voltron_Extractor(torch.nn.Module):
    'Voltron_Extractor implementation.'

    def __init__(
        self,
        model_name: str = "v-cond-base",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        try:
            import voltron
        except ImportError:
            raise RuntimeError("Voltron not installed. Run: pip install voltron-robotics")
        self.device = torch.device(device)
        self.model, self.preprocess = voltron.load(model_name, device="cpu")
        self.model = self.model.to(self.device)
        self.to_tensor = transforms.ToTensor()

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        
        image_tensors = torch.stack(
            [self.preprocess(self.to_tensor(Image.fromarray(img))) for img in images], dim=0
        ).to(self.device)

        if requires_grad:
            visual_tokens = self.model(image_tensors, None, mode="visual")
        else:
            with torch.no_grad():
                visual_tokens = self.model(image_tensors, None, mode="visual")

        batch_size, num_patches, num_channels = visual_tokens.size()
        side_length = int(np.sqrt(num_patches))
        return visual_tokens.transpose(1, 2).reshape(batch_size, num_channels, side_length, side_length)


def print_feature_size(model_name: str = "v-cond-base") -> None:
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = Voltron_Extractor(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)
    print(f"Model: {model_name}\nVisual Tokens: {visual_tokens.size()}")


if __name__ == "__main__":
    print_feature_size()

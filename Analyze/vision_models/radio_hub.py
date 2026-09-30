# Copyright 2025 NVIDIA CORPORATION

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class RADIOHub(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "radio_v2.5-b",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        
        
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=model_name,
            progress=True,
            skip_validation=True,
        ).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        image_tensors = torch.stack(
            [self.transform(Image.fromarray(img)) for img in images], dim=0
        ).to(self.device)
        if requires_grad:
            _, spatial_features = self.model(image_tensors)
        else:
            with torch.no_grad():
                _, spatial_features = self.model(image_tensors)

        # spatial_features: (B, N, C) -> (B, C, H, W)
        batch_size, num_patches, num_channels = spatial_features.size()
        side_length = int(np.sqrt(num_patches))
        return spatial_features.transpose(1, 2).reshape(
            batch_size, num_channels, side_length, side_length
        )


def print_feature_size() -> None:
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision_encoder = RADIOHub(model_name="radio_v2.5-b", device=device)
    vision_encoder.to(device)
    with torch.no_grad():
        batch_image = [np.array(image)] * 2
        features = vision_encoder(batch_image)
    print(f"Batch of features shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")


if __name__ == "__main__":
    print_feature_size()

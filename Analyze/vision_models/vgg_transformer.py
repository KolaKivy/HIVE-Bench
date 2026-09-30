# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from transformers import AutoImageProcessor
from vggt.models.vggt import VGGT  # type: ignore


class VGGTModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/VGGT-1B-Commercial",
        device: str | torch.device = "cuda",
        processor_fallback: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        **kwargs,
    ):
        super().__init__()
        self.device = torch.device(device)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        except OSError:
            print(
                f"Warning: Could not find processor for {model_name}, "
                f"using {processor_fallback} processor as fallback."
            )
            self.processor = AutoImageProcessor.from_pretrained(processor_fallback)
        self.model = VGGT.from_pretrained(model_name).to(self.device)

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        processed_image = inputs["pixel_values"]

        if requires_grad:
            outputs = self.model.encoder(processed_image)
        else:
            with torch.no_grad():
                outputs = self.model.encoder(processed_image)

        last_hidden_state = outputs["aggregated_tokens_list"][-1]
        batch_size, sequence_length, num_tokens, channels = last_hidden_state.shape
        last_hidden_state = last_hidden_state.reshape(batch_size * sequence_length, num_tokens, channels)

        patch_start_idx = self.model.aggregator.patch_start_idx
        patch_features = last_hidden_state[:, patch_start_idx:, :]
        batch_size, num_patches, num_channels = patch_features.size()
        side_length = int(np.sqrt(num_patches))
        return patch_features.transpose(1, 2).reshape(batch_size, num_channels, side_length, side_length)


def print_feature_size(model_name: str = "facebook/VGGT-1B-Commercial") -> None:
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = [np.array(raw_img)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = VGGTModel(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)

    print(f"--- VGGT Feature Info ---")
    print(f"Model Name:    {model_name}")
    print(f"Visual Tokens: {visual_tokens.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()

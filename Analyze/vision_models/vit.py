# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from transformers import AutoImageProcessor, ViTModel


class ViT(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "google/vit-huge-patch14-224-in21k",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name, local_files_only=False).to(self.device)

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


def print_feature_size(model_name: str = "google/vit-huge-patch14-224-in21k") -> None:
    from datasets import load_dataset

    dataset = load_dataset("huggingface/cats-image")
    image = [np.array(dataset["test"]["image"][0])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ViT(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)
    print(visual_tokens.size())


if __name__ == "__main__":
    print_feature_size()

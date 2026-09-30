# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor


class SAM3(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.processor = Sam3Processor.from_pretrained(model_name)
        self.model = Sam3Model.from_pretrained(model_name).to(self.device)

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        if requires_grad:
            outputs = self.model.get_vision_features(pixel_values=inputs.pixel_values)
        else:
            with torch.no_grad():
                outputs = self.model.get_vision_features(pixel_values=inputs.pixel_values)

        raw_tokens = outputs.last_hidden_state
        batch_size, num_tokens, num_channels = raw_tokens.shape
        side_length = int(np.sqrt(num_tokens))
        return raw_tokens.transpose(1, 2).reshape(batch_size, num_channels, side_length, side_length)


def get_sam3_logits_with_random_points(model, processor, images, num_points=1):
    device = model.device
    batch_size = len(images)
    inputs = processor(images=images, return_tensors="pt").to(device)
    input_points = torch.rand((batch_size, 1, num_points, 2), device=device)
    input_labels = torch.ones((batch_size, 1, num_points), dtype=torch.long, device=device)

    with torch.no_grad():
        vision_outputs = model.vision_encoder(pixel_values=inputs.pixel_values)
        image_embeddings = vision_outputs.last_hidden_state
        geometry_embeddings = model.geometry_encoder(
            point_coords=input_points,
            point_labels=input_labels,
        )
        mask_outputs = model.mask_decoder(
            image_embeddings=image_embeddings,
            geometry_embeddings=geometry_embeddings,
            multimask_output=True,
        )

    return mask_outputs.pred_masks, input_points


def print_feature_size(model_name: str = "facebook/sam3") -> None:
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw).convert("RGB"))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SAM3(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)

    print(f"--- SAM 3 Feature Info ---")
    print(f"Model Name:    {model_name}")
    print(f"Visual Tokens: {visual_tokens.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()

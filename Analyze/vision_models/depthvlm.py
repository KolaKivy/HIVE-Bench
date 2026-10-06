# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.





import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class DepthVLM(torch.nn.Module):
    """Extract image-position hidden states with the full VLM backbone."""

    def __init__(
        self,
        model_name: str = "playground/Pretrained_models/DepthVLM-4B",
        device: str | torch.device = "cuda",
        layer_idx: int = -1,
        **kwargs,
    ):
        """Extract image-position hidden states with the full VLM backbone."""
        super().__init__()
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name, extra_special_tokens={})



        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model = model.model
        self.visual = self.model.visual
        self.model.eval()
        self.model.to(self.device)
        self.image_token_id = int(self.model.config.image_token_id)
        self.vision_start_token_id = int(self.model.config.vision_start_token_id)
        self.vision_end_token_id = int(self.model.config.vision_end_token_id)
        self.layer_idx = layer_idx

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        """Extract image-position hidden states with the full VLM backbone."""
        image_inputs = self.processor.image_processor(images=images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.device).type(self.visual.dtype)
        grid_thw = image_inputs["image_grid_thw"].to(self.device)


        merge = self.visual.spatial_merge_size
        tokens = []
        for num in (grid_thw.prod(-1) // (merge * merge)).tolist():
            tokens.append(self.vision_start_token_id)
            tokens.extend([self.image_token_id] * int(num))
            tokens.append(self.vision_end_token_id)
        input_ids = torch.tensor([tokens], device=self.device)

        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )


        hidden_states = outputs.hidden_states[self.layer_idx]

        visual_tokens = hidden_states[input_ids == self.image_token_id]


        per_image = []
        offset = 0
        for t, h, w in grid_thw.tolist():
            num = (t * h * w) // (merge * merge)
            feats = visual_tokens[offset : offset + num].transpose(0, 1)
            offset += num
            per_image.append(feats.reshape(1, feats.size(0), h // merge, w // merge))
        max_h = max(feats.size(2) for feats in per_image)
        max_w = max(feats.size(3) for feats in per_image)
        if any(feats.size(2) != max_h or feats.size(3) != max_w for feats in per_image):
            per_image = [
                torch.nn.functional.interpolate(feats, size=(max_h, max_w), mode="bilinear", align_corners=False)
                for feats in per_image
            ]
        return torch.cat(per_image, dim=0).float()


def print_feature_size(
    model_name: str = "playground/Pretrained_models/DepthVLM-4B",
) -> None:
    """Extract image-position hidden states with the full VLM backbone."""
    import requests
    from PIL import Image

    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000000139.jpg",
        "http://images.cocodataset.org/val2017/000000000285.jpg",
    ]
    images = []
    for url in urls:
        try:
            images.append(np.array(Image.open(requests.get(url, stream=True, timeout=15).raw)))
        except Exception as e:
            print(f"Skip {url}: {e}")
    assert images, "No images downloaded"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depthvlm_extractor = DepthVLM(model_name=model_name, device=device)
    visual_tokens = depthvlm_extractor.forward(images)

    print(f"--- DepthVLM Feature Info ---")
    print(f"Model Name:      {model_name}")
    print(f"Num Images:      {len(images)}")
    print(f"Visual Tokens:   {visual_tokens.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()

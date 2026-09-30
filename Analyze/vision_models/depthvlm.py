# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.




import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class DepthVLM(torch.nn.Module):
    'DepthVLM implementation.'

    def __init__(
        self,
        model_name: str = "playground/Pretrained_models/DepthVLM-4B",
        device: str | torch.device = "cuda",
        layer_idx: int = -1,
        **kwargs,
    ):
        '  init   function.'
        super().__init__()
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name, extra_special_tokens={})

        
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.visual = model.model.visual
        del model.model.language_model
        del model.lm_head
        self.visual.eval()
        self.visual.to(self.device)
        self.layer_idx = layer_idx

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        'Forward function.'
        inputs = self.processor(
            text=["<image>"] * len(images),
            images=images,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device).type(self.visual.dtype)
        grid_thw = inputs["image_grid_thw"].to(self.device)

        if requires_grad:
            vision_output = self.visual(pixel_values, grid_thw=grid_thw, return_dict=True, output_hidden_states=True)
        else:
            with torch.no_grad():
                vision_output = self.visual(pixel_values, grid_thw=grid_thw, return_dict=True, output_hidden_states=True)

        hidden_states = vision_output.hidden_states[self.layer_idx]

        spatial_merge = self.visual.spatial_merge_size
        visual_tokens = []
        offset = 0
        for thw in grid_thw:
            _, h, w = thw.tolist()
            num_patches = (h * w) // (spatial_merge * spatial_merge)
            feats = hidden_states[offset : offset + num_patches]
            offset += num_patches
            side_h, side_w = h // spatial_merge, w // spatial_merge
            feats = feats.transpose(0, 1).reshape(1, feats.size(1), side_h, side_w)
            visual_tokens.append(feats)

        max_h = max(t.size(2) for t in visual_tokens)
        max_w = max(t.size(3) for t in visual_tokens)
        if any(t.size(2) != max_h or t.size(3) != max_w for t in visual_tokens):
            visual_tokens = [
                torch.nn.functional.interpolate(
                    t, size=(max_h, max_w), mode="bilinear", align_corners=False
                )
                for t in visual_tokens
            ]
        return torch.cat(visual_tokens, dim=0).float()


def print_feature_size(
    model_name: str = "playground/Pretrained_models/DepthVLM-4B",
) -> None:
    'Print feature size function.'
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

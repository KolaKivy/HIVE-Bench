# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.





import importlib.util
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration


class XiaomiRobotics(torch.nn.Module):
    """Extract image-position hidden states with the full VLM backbone."""

    def __init__(
        self,
        model_name: str = "playground/Pretrained_models/Xiaomi-Robotics",
        device: str | torch.device = "cuda",
        backbone_vlm: str | None = None,
        layer_idx: int = -1,
        **kwargs,
    ):
        """Extract image-position hidden states with the full VLM backbone."""
        super().__init__()
        self.device = torch.device(device)

        checkpoint = Path(model_name)
        state_path = checkpoint / "model_states.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"XR-1 checkpoint not found: {state_path}")


        if backbone_vlm is None:
            sibling = checkpoint.parent / "Qwen3-VL-4B-Instruct"
            backbone_vlm = str(sibling) if sibling.is_dir() else "Qwen/Qwen3-VL-4B-Instruct"
        if Path(backbone_vlm).is_dir() and not (Path(backbone_vlm) / "config.json").is_file():
            raise ValueError(f"XR-1 backbone snapshot is missing config.json: {backbone_vlm}")

        model_config = AutoConfig.from_pretrained(backbone_vlm)
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration._from_config(
                model_config, attn_implementation="eager", dtype=torch.bfloat16
            )

        raw = torch.load(
            checkpoint / "model_states.pt", map_location="cpu", mmap=True, weights_only=False
        )["module"]
        vlm_state = {
            key[len("model.vlm."):]: value
            for key, value in raw.items()
            if key.startswith("model.vlm.")
            and key not in {"model.vlm.model.action_embed.weight", "model.vlm.model.score_embed.weight"}
        }
        incompatible = model.load_state_dict(vlm_state, strict=False, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "XR-1 VLM checkpoint does not match the local Qwen3-VL backbone: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

        self.model = model.model
        self.visual = self.model.visual
        self.model.eval()
        self.model.to(self.device)
        self.image_token_id = int(self.model.config.image_token_id)
        self.vision_start_token_id = int(self.model.config.vision_start_token_id)
        self.vision_end_token_id = int(self.model.config.vision_end_token_id)

        self.processor = AutoProcessor.from_pretrained(backbone_vlm, trust_remote_code=True)
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

        # Select image-position hidden states from the requested VLM layer.
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
    model_name: str = "playground/Pretrained_models/Xiaomi-Robotics",
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

    xr1_extractor = XiaomiRobotics(model_name=model_name, device=device)
    visual_tokens = xr1_extractor.forward(images)

    print(f"--- Xiaomi Robotics-1 Feature Info ---")
    print(f"Model Name:      {model_name}")
    print(f"Num Images:      {len(images)}")
    print(f"Visual Tokens:   {visual_tokens.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()

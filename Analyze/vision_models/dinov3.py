# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


class DinoV3(torch.nn.Module):
    def __init__(
        self,
        # The Robotwin VisionCLIPGR00T checkpoints use DINOv3 ViT-B/16
        # (hidden size 768), not the ViT-S/16 variant (hidden size 384).
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str | torch.device = "cuda",
        **kwargs,
    ):
        '  init   function.'
        super().__init__()
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        
    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        'Forward function.'
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        if requires_grad:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        num_registers = self.model.config.num_register_tokens
        
        
        patch_features_flat = last_hidden_state[:, 1 + num_registers :, :]
        
        
        batch_size, num_patches, num_channels = patch_features_flat.size()
        side_length = int(np.sqrt(num_patches)) 
        
        
        visual_tokens = patch_features_flat.transpose(1, 2)
        visual_tokens = visual_tokens.reshape(
            batch_size, num_channels, side_length, side_length
        )

        return visual_tokens


def print_feature_size(model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m") -> None:
    'Print feature size function.'
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    dinov3_extractor = DinoV3(model_name=model_name, device=device)

    # Load model
    print("Loading model...")
    ckpt_dir = "playground/Checkpoints/all_task_robotwin_dinov3_ft/final_model/pytorch_model.pt"
    if ckpt_dir is not None:
        state_dict = torch.load(ckpt_dir, map_location=device)
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("vision_encoder.model."):
                new_key = key[len("vision_encoder.model."):]  
                new_state_dict[new_key] = value
        
        missing, unexpected = dinov3_extractor.model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint from: {ckpt_dir}")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    dinov3_extractor.model.eval()

    visual_tokens = dinov3_extractor.forward(image)

    print(f"--- DINOv3 Feature Info ---")
    print(f"Model Name:      {model_name}")
    print(f"Registers Count: {dinov3_extractor.model.config.num_register_tokens}")
    print(f"Visual Tokens:   {visual_tokens.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()

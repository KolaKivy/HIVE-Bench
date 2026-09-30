# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from spa.models import spa_vit_base_patch16, spa_vit_large_patch16 # type: ignore


class SPA(torch.nn.Module):
    def __init__(
        self, 
        model_name: str = "HaoyiZhu/SPA_Large", 
        device: str | torch.device = "cuda",
        **kwargs,
    ):
        super().__init__()
        """Initialize SPA and configure its image preprocessing pipeline."""
        self.device = torch.device(device)
        
        
        if model_name == "HaoyiZhu/SPA_Large":
            self.model = spa_vit_large_patch16(pretrained=True)
        elif model_name == "HaoyiZhu/SPA_base":
            self.model = spa_vit_base_patch16(pretrained=True)
        else:
            raise ValueError(f"Unknown SPA model: {model_name}")
        
        self.model.freeze()
        self.model = self.model.to(self.device)

    def _process_images(self, images: list[np.ndarray]) -> torch.Tensor:
        ' process images function.'
        processed_images = []
        for image in images:
            
            image_tensor = torch.from_numpy(image).float() / 255.0
            
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = torch.nn.functional.interpolate(
                image_tensor, size=(224, 224), mode='bilinear', align_corners=False
            )
            image_tensor = image_tensor.squeeze(0)
            processed_images.append(image_tensor)
        
        
        return torch.stack(processed_images).to(self.device)

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        'Forward function.'
        image_tensor = self._process_images(images)
        
        if requires_grad:
            visual_tokens = self.model(image_tensor, feature_map=True, cat_cls=False)
        else:
            with torch.no_grad():
                visual_tokens = self.model(image_tensor, feature_map=True, cat_cls=False)

        return visual_tokens


def print_feature_size(model_name: str = "HaoyiZhu/SPA_Large") -> None:
    'Print feature size function.'
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    spa_extractor = SPA(model_name=model_name, device=device)
    visual_tokens = spa_extractor.forward(image)

    print(f"--- SPA Feature Info ---")
    print(f"Model Name:      {model_name}")
    print(f"Visual Tokens:   {visual_tokens.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()
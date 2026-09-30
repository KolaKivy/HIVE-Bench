# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class VC1(torch.nn.Module):
    'VC1 implementation.'

    def __init__(
        self,
        model_name: str = "facebook/vc1-base",
        device: str | torch.device = "cuda",
        **kwargs,
    ):
        super().__init__()
        import timm
        from huggingface_hub import hf_hub_download

        self.device = torch.device(device)
        variant = model_name.rsplit("-", 1)[-1]  # "base" | "large"
        ckpt_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        arch = "vit_base_patch16_224" if variant == "base" else "vit_large_patch16_224"
        self.model = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        self.model = self.model.to(self.device)
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
            tokens = self.model.forward_features(image_tensors)
        else:
            with torch.no_grad():
                tokens = self.model.forward_features(image_tensors)

        visual_tokens = tokens[:, 1:, :]
        batch_size, num_patches, num_channels = visual_tokens.size()
        side_length = int(np.sqrt(num_patches))
        return visual_tokens.transpose(1, 2).reshape(batch_size, num_channels, side_length, side_length)


def print_feature_size(model_name: str = "facebook/vc1-base") -> None:
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = VC1(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)
    print(f"Model: {model_name}\nVisual Tokens: {visual_tokens.size()}")


if __name__ == "__main__":
    print_feature_size()

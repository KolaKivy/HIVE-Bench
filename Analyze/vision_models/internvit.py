# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


class InternViT(torch.nn.Module):
    'InternViT implementation.'

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternViT-300M-448px-V2_5",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        
        
        
        
        from transformers.modeling_utils import PreTrainedModel

        original_get_init_context = getattr(PreTrainedModel, "get_init_context", None)
        original_mark_tied = getattr(PreTrainedModel, "mark_tied_weights_as_initialized", None)
        if original_get_init_context is not None:
            PreTrainedModel.get_init_context = classmethod(
                lambda cls, dtype, is_quantized, _is_ds_init_called: []
            )
        if original_mark_tied is not None:
            def _safe_mark_tied(self):
                if not hasattr(self, "all_tied_weights_keys"):
                    self.all_tied_weights_keys = {}
                return original_mark_tied(self)

            PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied
        try:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        finally:
            if original_get_init_context is not None:
                PreTrainedModel.get_init_context = original_get_init_context
            if original_mark_tied is not None:
                PreTrainedModel.mark_tied_weights_as_initialized = original_mark_tied
        self.model = self.model.to(torch.bfloat16).to(self.device)
        # self.model = AutoModel.from_pretrained(
        #     model_name, dtype=torch.bfloat16, trust_remote_code=True
        # ).to(self.device)
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
            outputs = self.model(pixel_values=image_tensors.to(torch.bfloat16))
        else:
            with torch.no_grad():
                outputs = self.model(pixel_values=image_tensors.to(torch.bfloat16))

        visual_tokens = outputs.last_hidden_state[:, 1:, :].float()
        batch_size, num_patches, num_channels = visual_tokens.size()
        side_length = int(np.sqrt(num_patches))
        return visual_tokens.transpose(1, 2).reshape(batch_size, num_channels, side_length, side_length)


def print_feature_size(model_name: str = "OpenGVLab/InternViT-300M-448px-V2_5") -> None:
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = InternViT(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)
    print(f"Model: {model_name}\nVisual Tokens: {visual_tokens.size()}")


if __name__ == "__main__":
    print_feature_size()

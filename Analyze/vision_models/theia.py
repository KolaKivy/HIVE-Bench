# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


class Theia(torch.nn.Module):
    'Theia implementation.'

    def __init__(
        self,
        model_name: str = "theaiinstitute/theia-base-patch16-224-cdiv",
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
        self.model = self.model.to(self.device)
        # self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def forward(self, images: list[np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        arrays = [
            np.array(Image.fromarray(img).resize((224, 224)), dtype=np.uint8)
            for img in images
        ]
        pixel_values = torch.from_numpy(np.stack(arrays)).to(self.device)

        if requires_grad:
            visual_tokens = self.model.forward_feature(pixel_values)
        else:
            with torch.no_grad():
                visual_tokens = self.model.forward_feature(pixel_values)

        if visual_tokens.shape[1] == 197:
            visual_tokens = visual_tokens[:, 1:, :]
        batch_size, num_patches, num_channels = visual_tokens.size()
        side_length = int(np.sqrt(num_patches))
        return visual_tokens.transpose(1, 2).reshape(batch_size, num_channels, side_length, side_length)


def print_feature_size(model_name: str = "theaiinstitute/theia-base-patch16-224-cdiv") -> None:
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = [np.array(Image.open(requests.get(url, stream=True).raw))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = Theia(model_name=model_name, device=device)
    visual_tokens = extractor.forward(image)
    print(f"Model: {model_name}\nVisual Tokens: {visual_tokens.size()}")


if __name__ == "__main__":
    print_feature_size()

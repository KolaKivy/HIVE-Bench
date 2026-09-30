import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

hf_repo = "nvidia/C-RADIOv4-H"
TARGET_SIZE = (512, 512)
FEATURES_SIZE = (TARGET_SIZE[0] // 16, TARGET_SIZE[1] // 16)


def resize_and_center_crop(img: Image.Image | np.ndarray, target_size: tuple[int, int]) -> Image.Image:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    target_w, target_h = target_size
    scale = max(target_w / img.width, target_h / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - target_w) / 2
    top = (new_h - target_h) / 2
    right = left + target_w
    bottom = top + target_h
    return img_resized.crop((left, top, right, bottom))


class CRADIO(torch.nn.Module):
    def __init__(
        self,
        model_name: str = hf_repo,
        device: str | torch.device = "cuda",
        target_size: tuple[int, int] = TARGET_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.target_size = target_size
        self.features_size = (target_size[0] // 16, target_size[1] // 16)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def forward(self, images: list[Image.Image | np.ndarray], requires_grad: bool = False) -> torch.Tensor:
        images = [resize_and_center_crop(img, self.target_size) for img in images]
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            size=self.target_size,
        ).pixel_values.to(self.device)

        if requires_grad:
            _, features = self.model(inputs)
        else:
            with torch.no_grad():
                _, features = self.model(inputs)

        batch_size, _, channels = features.shape
        features = features.reshape(batch_size, self.features_size[0], self.features_size[1], channels)
        return features.permute(0, 3, 1, 2)


def print_feature_size(model_name: str = hf_repo) -> None:
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    extractor = CRADIO(model_name=model_name)
    features = extractor.forward([image])

    print(f"--- C-RADIO Feature Info ---")
    print(f"Model Name:      {model_name}")
    print(f"Visual Tokens:   {features.size()} (BCHW)")


if __name__ == "__main__":
    print_feature_size()
